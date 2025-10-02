#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ToM_trust.py
------------------------------------
Self-contained 5-agent script with ToM + Trust for Melting Pot
gift_refinements. Implements “learn (homogeneous team) → gen (mixed team)”.

추가 기능
--live : 관측 RGB를 OpenCV 창으로 실시간 렌더링 (파일 저장 없음, q 로 종료)
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
from collections import deque
from typing import Dict, List, Tuple, Deque, Any
import os
import json
import math
import inspect

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cv2  # for --live render

# ====== 0) ENVIRONMENT ======
try:
    from meltingpot import substrate
except Exception:
    substrate = None
    print(
        "[WARN] dm-meltingpot not found; install to run the environment.\n"
        "       pip install dm-meltingpot==2.4.0"
    )


# ====== 1) CONFIG ======
@dataclass
class Config:
    # training
    episodes: int = 200
    max_steps: int = 200
    gamma: float = 0.99
    lr: float = 3e-4
    seed: int = 42
    save_dir: str = "ckpt"
    headless: bool = False
    phase: str = "learn"  # learn | gen
    team: str = "A"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ToM/Trust
    hist_len: int = 32
    trust_eta0: float = 0.15
    trust_kappa: float = 0.8

    # network sizes
    env_hid: int = 128
    opp_hid: int = 64
    self_hid: int = 32
    comb_hid: int = 128

    # logging
    log_interval: int = 10


# ====== 2) PERSONAS ======
PERSONAS = {
    "A": {"name": "A", "E": 0.8, "A": 0.8},
    "B": {"name": "B", "E": 0.8, "A": 0.2},
    "C": {"name": "C", "E": 0.2, "A": 0.8},
    "D": {"name": "D", "E": 0.2, "A": 0.2},
    "E": {"name": "E", "E": 0.5, "A": 0.5},
}


def make_traits_for_phase(phase: str, team: str) -> List[Dict[str, float]]:
    """learn: 동일 성격 5명 / gen: A,B,C,D,E 섞음"""
    if phase == "learn":
        if team not in PERSONAS:
            raise ValueError(f"team must be one of {list(PERSONAS)}")
        p = PERSONAS[team]
        traits = [dict(p) for _ in range(5)]
        for i, t in enumerate(traits):
            t["name"] = f"{t['name']}{i+1}"
        return traits
    elif phase == "gen":
        return [dict(PERSONAS[k]) for k in ["A", "B", "C", "D", "E"]]
    else:
        raise ValueError("phase must be 'learn' or 'gen'")


# ====== 3) TRUST ======
class AdaptiveTrust:
    """T[i,j] ∈ [0,1]. Update speed depends on estimated Â_j."""
    def __init__(self, n_agents: int, cfg: Config):
        self.n = n_agents
        self.T = np.full((n_agents, n_agents), 0.5, dtype=np.float32)
        np.fill_diagonal(self.T, 0.0)
        self.cfg = cfg

    def get(self, i: int, j: int) -> float:
        return float(self.T[i, j])

    def update(self, i: int, j: int, event: str, A_hat_j: float):
        """event ∈ {'coop','defect','none'}; A_hat_j ∈ [0,1]."""
        eta0 = self.cfg.trust_eta0
        k = self.cfg.trust_kappa
        up = eta0 * (1.0 + k * A_hat_j)
        down = eta0 * (1.0 + k * (1.0 - A_hat_j))
        t = self.T[i, j]
        if event == "coop":
            self.T[i, j] = t + up * (1.0 - t)
        elif event == "defect":
            self.T[i, j] = t + down * (0.0 - t)
        else:
            self.T[i, j] = 0.99 * t + 0.01 * 0.5


# ====== 4) ToM: Opponent Estimator + Pair Memory ======
class OppEstimator(nn.Module):
    """Tiny GRU that reads pair interaction features and outputs ẑ_j=(Ê, Â)."""
    def __init__(self, in_dim: int = 6, hid: int = 32):
        super().__init__()
        self.gru = nn.GRU(input_size=in_dim, hidden_size=hid, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hid, 32), nn.ReLU(),
            nn.Linear(32, 2), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(x)
        z = self.head(h[:, -1])
        return z


class PairMemory:
    """Fixed-length deques of per-pair interaction features."""
    def __init__(self, n_agents: int, L: int):
        self.L = L
        self.mem: Dict[Tuple[int, int], Deque[List[float]]] = {
            (i, j): deque(maxlen=L)
            for i in range(n_agents) for j in range(n_agents) if i != j
        }

    def push(self, i: int, j: int, feat: List[float]):
        self.mem[(i, j)].append(feat)

    def batch(self, pairs: List[Tuple[int, int]]) -> np.ndarray:
        out = []
        for i, j in pairs:
            buf = self.mem[(i, j)]
            if len(buf) == 0:
                out.append(np.zeros((self.L, 6), dtype=np.float32))
            else:
                pad = self.L - len(buf)
                arr = np.array(buf, dtype=np.float32)
                if pad > 0:
                    arr = np.concatenate(
                        [np.zeros((pad, arr.shape[1]), np.float32), arr], axis=0
                    )
                out.append(arr)
        return np.stack(out, axis=0)  # [B,L,6]


# ====== 5) POLICY: three-head ======
class ThreeHeadPolicy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, cfg: Config):
        super().__init__()
        self.env_head = nn.Sequential(nn.Linear(obs_dim, cfg.env_hid), nn.ReLU())
        # opponent summary: [mean T[i,*], mean Â_j, product, 1]
        self.opp_head = nn.Sequential(nn.Linear(4, cfg.opp_hid), nn.ReLU())
        self.self_head = nn.Sequential(nn.Linear(2, cfg.self_hid), nn.ReLU())
        self.comb = nn.Sequential(
            nn.Linear(cfg.env_hid + cfg.opp_hid + cfg.self_hid, cfg.comb_hid),
            nn.ReLU(),
        )
        self.pi = nn.Linear(cfg.comb_hid, action_dim)
        self.v = nn.Linear(cfg.comb_hid, 1)

    def forward(self, obs: torch.Tensor, opp_feat: torch.Tensor, self_z: torch.Tensor):
        h = torch.cat(
            [self.env_head(obs), self.opp_head(opp_feat), self.self_head(self_z)], dim=-1
        )
        h = self.comb(h)
        logits = self.pi(h)
        v = self.v(h)
        return logits, v


# ====== 6) UTILS ======
def flatten_obs(obs):
    """dict/list/tuple/ndarray/scalar 모두 평탄화."""
    def _flat(x, out):
        if isinstance(x, np.ndarray):
            if x.dtype != object:
                out.append(x.reshape(-1).astype(np.float32))
        elif isinstance(x, dict):
            for v in x.values():
                _flat(v, out)
        elif isinstance(x, (list, tuple)):
            for v in x:
                _flat(v, out)
        elif np.isscalar(x):
            out.append(np.array([x], dtype=np.float32))
    buf = []
    _flat(obs, buf)
    return np.concatenate(buf, axis=0) if buf else np.zeros((1,), dtype=np.float32)


def detect_event_from_extras(extras: Dict[str, Any], i: int, j: int) -> str:
    evs = extras.get("gift_events", []) if isinstance(extras, dict) else []
    if isinstance(evs, list):
        for e in evs:
            if isinstance(e, dict):
                if e.get("giver") == j and e.get("receiver") == i:
                    return "coop"
                if e.get("giver") == i and e.get("receiver") == j and e.get("stolen", False):
                    return "defect"
    return "none"


def trait_shaping_reward(E: float, A: float, features: Dict[str, float]) -> float:
    rE = 0.05 * features.get("unique_partners", 0.0) + 0.02 * features.get("attempts", 0.0)
    rA = 0.50 * features.get("repaid", 0.0) + 0.20 * features.get("forgave_new", 0.0)
    return float(E * rE + A * rA)


# ---- LIVE 렌더용: 관측에서 RGB 프레임 추출 ----
def extract_rgb_frame(ts):
    obs = getattr(ts, "observation", None)
    if obs is None:
        return None
    CAND_KEYS = ["WORLD.RGB", "RGB", "global_rgb", "WORLD.RGB.player_0"]

    def _pick(d):
        for k in CAND_KEYS + list(d.keys()):
            if k in d:
                arr = d[k]
                if isinstance(arr, np.ndarray):
                    if arr.ndim == 2:
                        arr = np.repeat(arr[..., None], 3, axis=2)
                    if arr.dtype != np.uint8:
                        arr = np.clip(arr, 0, 255).astype(np.uint8)
                    return arr
        return None

    if isinstance(obs, (list, tuple)):
        for it in obs:
            if isinstance(it, dict):
                f = _pick(it)
                if f is not None:
                    return f
        return None
    if isinstance(obs, dict):
        if 0 in obs and isinstance(obs[0], dict):
            f = _pick(obs[0])
            if f is not None:
                return f
        for v in obs.values():
            if isinstance(v, dict):
                f = _pick(v)
                if f is not None:
                    return f
        return _pick(obs)
    return None


# ====== 7) ENV BUILD (display 인자 없는 빌드 서명에 맞춤) ======
def build_env(num_players: int, headless: bool = True):
    assert substrate is not None, "dm-meltingpot is required."
    return substrate.build("gift_refinements", roles=["default"] * num_players)


# ====== 8) AGENT ======
class Agent:
    def __init__(self, obs_dim: int, action_dim: int, cfg: Config, trait: Dict[str, float]):
        self.device = cfg.device
        self.trait = trait
        self.net = ThreeHeadPolicy(obs_dim, action_dim, cfg).to(self.device)
        self.opt = optim.Adam(self.net.parameters(), lr=cfg.lr)
        self.memory = []  # (obs_v, logp, value, reward, opp_v)

    def policy_step(self, obs_v: torch.Tensor, opp_v: torch.Tensor):
        self_z = torch.tensor([[self.trait["E"], self.trait["A"]]],
                              device=self.net.pi.weight.device, dtype=torch.float32)
        logits, v = self.net(obs_v, opp_v, self_z)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        return int(a.item()), dist.log_prob(a), v

    def update(self, gamma: float = 0.99):
        if not self.memory:
            return
        R = 0.0
        returns = []
        for *_, r, _ in reversed(self.memory):
            R = r + gamma * R
            returns.append(R)
        returns = list(reversed(returns))
        returns_t = torch.tensor(returns, dtype=torch.float32,
                                 device=self.net.pi.weight.device).unsqueeze(1)

        logps = torch.cat([lp for (_, lp, _, _, _) in self.memory], dim=0)
        values = torch.cat([v for (_, _, v, _, _) in self.memory], dim=0)
        adv = returns_t - values.detach()
        pg = -(logps * adv).mean()
        vloss = 0.5 * (values - returns_t).pow(2).mean()
        loss = pg + vloss
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.opt.step()
        self.memory.clear()


# ====== 9) MAIN ======
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=Config.episodes)
    parser.add_argument("--max-steps", type=int, default=Config.max_steps)
    parser.add_argument("--gamma", type=float, default=Config.gamma)
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--save-dir", type=str, default=Config.save_dir)
    parser.add_argument("--phase", type=str, default=Config.phase, choices=["learn", "gen"])
    parser.add_argument("--team", type=str, default=Config.team)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--live", action="store_true", help="Show live view via OpenCV")
    args = parser.parse_args()

    cfg = Config(episodes=args.episodes, max_steps=args.max_steps, gamma=args.gamma,
                 lr=args.lr, save_dir=args.save_dir, headless=args.headless,
                 phase=args.phase, team=args.team)

    os.makedirs(cfg.save_dir, exist_ok=True)

    # Personas & env
    traits = make_traits_for_phase(cfg.phase, cfg.team)
    n = 5

    env = build_env(num_players=n, headless=cfg.headless)
    action_specs = env.action_spec()
    action_dim = int(np.max([spec.num_values for spec in action_specs]))

    # Seed
    np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)

    # ToM & Trust
    pairmem = PairMemory(n_agents=n, L=cfg.hist_len)
    estimator = OppEstimator(in_dim=6, hid=32).to(cfg.device)
    est_opt = optim.Adam(estimator.parameters(), lr=cfg.lr)
    trust = AdaptiveTrust(n_agents=n, cfg=cfg)

    # obs dim
    ts = env.reset()
    obs0 = ts.observation[0] if isinstance(ts.observation, (list, tuple)) else ts.observation
    obs_dim = flatten_obs(obs0).shape[0]

    # Agents
    agents = [Agent(obs_dim, action_dim, cfg, traits[i]) for i in range(n)]

    # --------- TRAIN/ROLL-OUT ---------
    for ep in range(cfg.episodes):
        ts = env.reset()
        ep_ret = np.zeros(n, dtype=np.float32)

        for step in range(cfg.max_steps):
            actions = []
            logps = []
            values = []
            obs_list = []
            opp_list = []

            # Precompute ẑ for all (i,j)
            pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
            batch_np = pairmem.batch(pairs)
            batch_t = torch.tensor(batch_np, dtype=torch.float32, device=cfg.device)
            with torch.no_grad():
                zhat = estimator(batch_t)
            # Agreeableness matrix
            zhat_A = np.zeros((n, n), dtype=np.float32)
            for k, (ii, jj) in enumerate(pairs):
                zhat_A[ii, jj] = float(zhat[k, 1].clamp(0, 1).item())

            # Action selection
            for i in range(n):
                obs_i = ts.observation[i] if isinstance(ts.observation, (list, tuple)) else ts.observation
                obs_v = torch.tensor(flatten_obs(obs_i), dtype=torch.float32, device=cfg.device).unsqueeze(0)
                Ti_mean = float(np.mean([trust.get(i, j) for j in range(n) if j != i]))
                Aj_mean = float(np.mean([zhat_A[i, j] for j in range(n) if j != i]))
                opp_v = torch.tensor([[Ti_mean, Aj_mean, Ti_mean * Aj_mean, 1.0]],
                                     dtype=torch.float32, device=cfg.device)
                a, lp, v = agents[i].policy_step(obs_v, opp_v)
                actions.append(a)
                logps.append(lp)
                values.append(v)
                obs_list.append(obs_v)
                opp_list.append(opp_v)

            # Step environment
            ts = env.step(actions)

            # ---- LIVE VIEW ----
            if args.live:
                frame = extract_rgb_frame(ts)
                if frame is not None:
                    h, w = frame.shape[:2]
                    scale = 720.0 / max(h, w)
                    img = cv2.resize(frame, (int(w * scale), int(h * scale)))[:, :, ::-1]  # RGB->BGR
                    cv2.imshow("ToM_trust LIVE", img)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        env.close()
                        cv2.destroyAllWindows()
                        return

            # Rewards & memory
            rewards = np.array(ts.reward, dtype=np.float32)
            ep_ret += rewards

            extras = getattr(ts, "extras", {}) if hasattr(ts, "extras") else {}
            # Trust & pair memory
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    ev = detect_event_from_extras(extras, i, j)
                    Ahat = float(zhat_A[i, j])
                    trust.update(i, j, ev, Ahat)
                    feat = [
                        1.0 if ev == "coop" else 0.0,
                        0.0,
                        1.0 if ev == "coop" else 0.0,
                        0.0,
                        float(rewards[i]) if i < len(rewards) else 0.0,
                        float(rewards[j]) if j < len(rewards) else 0.0,
                    ]
                    pairmem.push(i, j, feat)

            for i in range(n):
                shaping = trait_shaping_reward(traits[i]["E"], traits[i]["A"], {
                    "unique_partners": 0.0,
                    "attempts": 1.0,
                    "repaid": 0.0,
                    "forgave_new": 0.0,
                })
                r = rewards[i] + shaping
                agents[i].memory.append((obs_list[i], logps[i], values[i], r, opp_list[i]))

            if ts.last():
                break

        # Policy updates
        for i in range(n):
            agents[i].update(cfg.gamma)

        # Estimator training (weak supervision): A_target := current trust
        pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
        batch_np = pairmem.batch(pairs)
        batch_t = torch.tensor(batch_np, dtype=torch.float32, device=cfg.device)
        zhat_pred = estimator(batch_t)
        A_target = torch.tensor([[trust.get(i, j)] for (i, j) in pairs],
                                dtype=torch.float32, device=cfg.device)
        E_target = torch.zeros_like(A_target) + 0.5
        target = torch.cat([E_target, A_target], dim=1)
        est_loss = ((zhat_pred - target).pow(2)).mean()
        est_opt.zero_grad()
        est_loss.backward()
        est_opt.step()

        # Log
        if (ep + 1) % cfg.log_interval == 0:
            print(f"[ep {ep+1}] ret={ep_ret.mean():.2f} est_loss={float(est_loss):.4f}")

        # Save checkpoint
        ckpt = {
            "cfg": cfg.__dict__,
            "traits": traits,
            "trust": trust.T.tolist(),
            "estimator": estimator.state_dict(),
            "agents": [ag.net.state_dict() for ag in agents],
        }
        with open(os.path.join(cfg.save_dir, "last.meta.json"), "w") as f:
            json.dump({"ep": ep + 1, "ret_mean": float(ep_ret.mean())}, f)
        torch.save(ckpt, os.path.join(cfg.save_dir, "last.pt"))

    # --------- CLEANUP ---------
    env.close()
    if args.live:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
