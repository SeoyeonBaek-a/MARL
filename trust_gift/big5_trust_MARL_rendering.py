from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
from collections import deque, defaultdict
from typing import Any, Dict, List, Tuple

import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 시각화 / 비디오
import cv2
import imageio.v2 as imageio

# 기존 TrustLogger 사용 (같은 폴더에 big5_trust_logger.py 존재 가정)
from big5_trust_logger import TrustLogger  # noqa: F401

# =========================
# 0) 하이퍼파라미터
# =========================
@dataclass
class Config:
    episodes: int = 200
    max_steps: int = 200
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2
    policy_lr: float = 3e-4
    value_lr: float = 1e-3
    ent_coef: float = 0.05  # ↑ 초반 탐색 강화
    vf_coef: float = 0.5
    train_iters: int = 4
    minibatch_size: int = 256
    rollout_steps: int = 1024  # ↓ 업데이트 더 자주
    headless: bool = False  # 기본값을 False로 변경 (렌더링 ON)
    save_dir: str = "checkpoints"
    load_path: str | None = None
    log_interval: int = 10
    probe_steps_per_action: int = 40
    # 엔트로피 스케줄링
    ent_coef_warmup_ep: int = 20
    ent_coef_cool: float = 0.01
    # 렌더/녹화
    record: bool = False
    record_path: str = "episode_preview.mp4"
    record_fps: int = 10

# 성격 파라미터 (E/A만 차이)
TRAITS = [
    {"name": "A", "E": 0.8, "A": 0.8},
    {"name": "B", "E": 0.8, "A": 0.2},
    {"name": "C", "E": 0.2, "A": 0.8},
    {"name": "D", "E": 0.2, "A": 0.2},
    {"name": "E", "E": 0.8, "A": 0.8},
    {"name": "F", "E": 0.8, "A": 0.2},
    {"name": "G", "E": 0.2, "A": 0.8},
    {"name": "H", "E": 0.2, "A": 0.2},
]
ALPHA_E, ALPHA_A = 1.0, 1.0
HIST_LEN = 20

# =========================
# 1) 환경 유틸
# =========================
def build_env(num_players=8, headless=True):
    # dm-meltingpot==2.4.0 기준: build()에 render_mode/display 인자 없음
    from meltingpot import substrate
    from meltingpot.configs.substrates import gift_refinements as cfg
    config = cfg.get_config()
    roles = getattr(config, "default_player_roles", None)
    if roles is None or len(roles) != num_players:
        roles = ["default"] * num_players
    env = substrate.build("gift_refinements", roles=roles)
    print(f"[build_env] gift_refinements, roles={roles}, headless={headless}")
    return env

def _num_values_from_spec(s) -> int:
    nv = getattr(s, "num_values", None)
    if nv is not None:
        return int(nv)
    mn = int(np.asarray(getattr(s, "minimum")))
    mx = int(np.asarray(getattr(s, "maximum")))
    return int(mx - mn + 1)

def inspect_action_slots(env) -> Tuple[List[str], List[int]]:
    spec0 = env.action_spec()[0]
    try:
        it = list(spec0)
    except Exception:
        return ["slot0"], [_num_values_from_spec(spec0)]
    return [f"slot{i}" for i in range(len(it))], [_num_values_from_spec(v) for v in it]

# =========================
# 1-1) 렌더링 유틸
# =========================
def _find_world_or_agent_frame(obs_i):
    """관측 dict에서 WORLD.RGB(월드뷰)를 최우선으로, 없으면 RGB(에이전트뷰) 반환."""
    if not isinstance(obs_i, dict):
        return None

    # 1) WORLD.RGB 우선
    for k, v in obs_i.items():
        key = str(k).lower()
        if "world" in key and "rgb" in key:
            try:
                arr = np.asarray(v)
                if arr.ndim == 3 and arr.shape[-1] in (3, 4):
                    if arr.shape[-1] == 4:
                        arr = arr[..., :3]
                    return np.asarray(arr, dtype=np.uint8)
            except Exception:
                pass

    # 2) 일반 RGB(에이전트 1인칭)로 폴백
    for k, v in obs_i.items():
        key = str(k).lower()
        if "rgb" in key or "color" in key or "frame" in key:
            try:
                arr = np.asarray(v)
                if arr.ndim == 3 and arr.shape[-1] in (3, 4):
                    if arr.shape[-1] == 4:
                        arr = arr[..., :3]
                    return np.asarray(arr, dtype=np.uint8)
            except Exception:
                pass
    return None


def _show_frame_if_any(window: str, obs_list):
    """여러 에이전트 관측 중에서 월드뷰 있으면 그걸, 없으면 첫 RGB를 띄움."""
    if obs_list is None:
        return None

    frame = None
    for i in range(len(obs_list)):
        f = _find_world_or_agent_frame(obs_list[i])
        if f is not None:
            frame = f
            break

    if frame is not None:
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        try:
            cv2.imshow(window, bgr)
            cv2.waitKey(1)
        except Exception:
            pass
    return frame
# =========================
# 2) (삭제) probe 함수 호출 제거
#  - 함수 정의는 남겨두고 호출만 제거하여 최소 변경
# =========================
def probe_action_semantics(env, slot_size: int, n_agents: int, steps_per_a: int = 40):
    print("[PROBE] start probing slot0 actions...")
    findings = {}
    for a in range(slot_size):
        ts = env.reset()
        totals = np.zeros(n_agents, dtype=np.float32)
        for _ in range(steps_per_a):
            acts = [a] + [0] * (n_agents - 1)
            ts = env.step(acts)
            rew = np.asarray(getattr(ts, "reward", np.zeros(n_agents)))
            totals += rew
        obs0 = None
        try:
            obs0 = ts.observation[0]
            obs_keys = list(obs0.keys()) if isinstance(obs0, dict) else type(obs0)
        except Exception:
            obs_keys = []
        info = getattr(ts, "extras", {})
        info_keys = list(info.keys()) if isinstance(info, dict) else type(info)
        print(f"[PROBE] a={a} | cumulative_reward={totals.tolist()} | obs0_keys={obs_keys} | extras keys={info_keys}")
        findings[a] = {"cum_reward": totals.tolist(), "obs_keys": obs_keys, "extras_keys": info_keys}
    print("[PROBE] done\n")
    return findings

# =========================
# 3) Trait 보상 관련 버퍼/함수
# =========================
partner_history: List[deque] | None = None
debt_ij: List[defaultdict] | None = None

def init_trait_buffers(n: int):
    global partner_history, debt_ij
    partner_history = [deque(maxlen=HIST_LEN) for _ in range(n)]
    debt_ij = [defaultdict(int) for _ in range(n)]

def r_trait_components(i: int, partner: int | None, will_give: bool):
    assert partner_history is not None and debt_ij is not None
    unique_partners = len(set(partner_history[i]))
    r_E = (1.0 if will_give else 0.0) + 0.05 * unique_partners
    repay_bonus = 0.0
    if will_give and (partner is not None):
        owed = debt_ij[i].get(partner, 0)
        repay_bonus = 0.15 if owed > 0 else 0.0
    forgive_bonus = 0.05 if (partner is not None and will_give and partner not in partner_history[i]) else 0.0
    return r_E, repay_bonus + forgive_bonus

# =========================
# 4) 네트워크
# =========================
class PolicyNet(nn.Module):
    def __init__(self, input_dim: int, hidden: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return torch.distributions.Categorical(logits=self.net(x))

class CentralCritic(nn.Module):
    def __init__(self, input_dim: int, hidden: int):
        super().__init__()
        self.v = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.v(x).squeeze(-1)

# =========================
# 5) RolloutBuffer, GAE
# =========================
class RolloutBuffer:
    def __init__(self):
        self.clear()
    def add(self, **kw):
        self.storage.append(kw)
    def clear(self):
        self.storage: List[Dict[str, Any]] = []
    def __len__(self):
        return len(self.storage)
    def as_batches(self, minibatch_size: int, device: str):
        idx = np.arange(len(self.storage)); np.random.shuffle(idx)
        for s in range(0, len(idx), minibatch_size):
            mb = [self.storage[k] for k in idx[s:s + minibatch_size]]
            out = {}
            for k in mb[0]:
                vals = [m[k] for m in mb]
                if isinstance(vals[0], torch.Tensor):
                    out[k] = torch.stack(vals).to(device)
                else:
                    out[k] = torch.as_tensor(vals, device=device)
            yield out

def compute_gae(rews, dones, values, gamma, lam):
    adv = np.zeros_like(rews, dtype=np.float32)
    lastgaelam = 0.0
    for t in reversed(range(len(rews))):
        nonterminal = 1.0 - dones[t]
        delta = rews[t] + gamma * values[t + 1] * nonterminal - values[t]
        lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        adv[t] = lastgaelam
    returns = adv + values[:-1]
    return adv, returns

# =========================
# 6) PPO 업데이트
# =========================

def ppo_update(cfg: Config, device: str, policies: List[PolicyNet], critic: CentralCritic, opt_p: List[optim.Optimizer], opt_v: optim.Optimizer, buffer: RolloutBuffer):
    for _ in range(cfg.train_iters):
        for mb in buffer.as_batches(cfg.minibatch_size, device):
            # Policy loss (sum over agents' masked data)
            loss_pi, ent_all = 0.0, 0.0
            for i, pi in enumerate(policies):
                mask = (mb['mask_agent'] == i)
                if mask.sum() == 0:
                    continue
                dist = pi(mb['obs'][mask])
                logp = dist.log_prob(mb['act'][mask])
                ratio = torch.exp(logp - mb['old_logp'][mask])
                adv = mb['adv'][mask]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                l1 = ratio * adv
                l2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv
                loss_pi = loss_pi + (-(torch.min(l1, l2)).mean())
                ent_all = ent_all + dist.entropy().mean()
            loss_pi = loss_pi - cfg.ent_coef * ent_all
            for opt in opt_p:
                opt.zero_grad(set_to_none=True)
            loss_pi.backward()
            for opt in opt_p:
                opt.step()

            # Value loss (central critic)
            v = critic(mb['obs_global'])
            loss_v = cfg.vf_coef * ((v - mb['ret']) ** 2).mean()
            opt_v.zero_grad(set_to_none=True)
            loss_v.backward()
            opt_v.step()

# =========================
# 7) 상태 만들기 & 이벤트 파서(휴리스틱)
# =========================

def _infer_inventory(obs_i: Dict[str, Any]) -> Dict[str, float]:
    inv_sum = {}
    if not isinstance(obs_i, dict):
        return inv_sum
    for k, v in obs_i.items():
        kl = str(k).lower()
        if any(t in kl for t in ["inventory", "token", "refin", "resource", "items", "store"]):
            try:
                arr = np.asarray(v, dtype=np.float32).ravel()
                inv_sum[k] = float(arr.sum())
            except Exception:
                pass
    return inv_sum


def parse_gift_events(prev_obs: List[Dict[str, Any]] | None, curr_obs: List[Dict[str, Any]]) -> List[Tuple[int, int, float]]:
    events: List[Tuple[int, int, float]] = []
    if prev_obs is None:
        return events
    n = len(curr_obs)
    prev_sums = [sum(_infer_inventory(prev_obs[i]).values()) for i in range(n)]
    curr_sums = [sum(_infer_inventory(curr_obs[i]).values()) for i in range(n)]
    deltas = [curr_sums[i] - prev_sums[i] for i in range(n)]
    senders = [(i, d) for i, d in enumerate(deltas) if d < -1e-5]
    receivers = [(i, d) for i, d in enumerate(deltas) if d > 1e-5]
    if not senders or not receivers:
        return events
    senders.sort(key=lambda x: x[1])    # 더 음수 먼저
    receivers.sort(key=lambda x: -x[1]) # 더 양수 먼저
    for (si, sd), (ri, rd) in zip(senders, receivers):
        amt = min(-sd, rd)
        if amt > 0:
            events.append((si, ri, float(amt)))
    return events


def make_state(i: int, trust: TrustLogger, n: int, partner_hist: List[deque], debt: List[defaultdict]):
    row = trust.C[i].copy()
    row[i] = 0.0
    hist_counts = np.zeros(n, dtype=np.float32)
    for p in partner_hist[i]:
        hist_counts[p] += 1
    if hist_counts.sum() > 0:
        hist_counts /= hist_counts.sum()
    debt_vec = np.zeros(n, dtype=np.float32)
    for j, c in debt[i].items():
        debt_vec[j] = float(c)
    if debt_vec.sum() > 0:
        debt_vec /= (debt_vec.sum() + 1e-6)
    E, A = TRAITS[i]["E"], TRAITS[i]["A"]
    return np.concatenate([row, hist_counts, debt_vec, [E, A]], axis=0)

# =========================
# 8) 체크포인트
# =========================

def save_ckpt(path: Path, policies: List[PolicyNet], critic: CentralCritic, opt_p: List[optim.Optimizer], opt_v: optim.Optimizer, meta: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "policies": [p.state_dict() for p in policies],
        "critic": critic.state_dict(),
        "opt_p": [o.state_dict() for o in opt_p],
        "opt_v": opt_v.state_dict(),
        "meta": meta,
    }, path)
    print(f"[CKPT] saved -> {path}")


def load_ckpt(path: Path, policies: List[PolicyNet], critic: CentralCritic, opt_p: List[optim.Optimizer], opt_v: optim.Optimizer) -> Dict[str, Any] | None:
    if not path.exists():
        print(f"[CKPT] no file -> {path}")
        return None
    data = torch.load(path, map_location="cpu")
    for p, sd in zip(policies, data["policies"]):
        p.load_state_dict(sd)
    critic.load_state_dict(data["critic"])
    for o, sd in zip(opt_p, data["opt_p"]):
        o.load_state_dict(sd)
    opt_v.load_state_dict(data["opt_v"])
    print(f"[CKPT] loaded <- {path}")
    return data.get("meta", {})

# =========================
# 9) 메인 학습 루프 (+렌더/녹화)
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--max-steps', type=int, default=200)
    # 기본값 False로 바꿈 (렌더링 ON). 사용자가 --headless true 주면 비활성화 가능
    parser.add_argument('--headless', type=lambda x: str(x).lower() != "false", default=False)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--save-dir', type=str, default='checkpoints')
    parser.add_argument('--record', action='store_true')
    parser.add_argument('--record-path', type=str, default='episode_preview.mp4')
    parser.add_argument('--record-fps', type=int, default=10)
    args = parser.parse_args()

    cfg = Config(
        episodes=args.episodes, max_steps=args.max_steps,
        headless=args.headless, save_dir=args.save_dir, load_path=args.load,
        record=args.record, record_path=args.record_path, record_fps=args.record_fps
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    num_players = 8
    env = build_env(num_players, headless=cfg.headless)
    slot_names, slot_sizes = inspect_action_slots(env)
    print("[INFO] slot info:", slot_names, slot_sizes)

    # (중요) 행동 의미 디버깅 호출 제거
    # probe_action_semantics(env, slot_sizes[0], num_players, steps_per_a=cfg.probe_steps_per_action)

    n = len(env.action_spec())
    names = [t['name'] for t in TRAITS[:n]]

    trust = TrustLogger(
        n, names,
        deltas=[0.2] * n, omegas=[1.0] * n, init_trust=0.3,
        reciprocity_window=5, debt_horizon=3,
        extraversion=[t['E'] for t in TRAITS[:n]],
        agreeableness=[t['A'] for t in TRAITS[:n]],
        log_interval=1,
    )
    init_trait_buffers(n)

    state_dim = 3 * n + 2
    policies = [PolicyNet(state_dim, 128, slot_sizes[0]).to(device) for _ in range(n)]
    critic = CentralCritic(n * state_dim, 256).to(device)
    opt_p = [optim.Adam(p.parameters(), lr=cfg.policy_lr) for p in policies]
    opt_v = optim.Adam(critic.parameters(), lr=cfg.value_lr)

    # 체크포인트 로드(옵션)
    meta = {}
    if cfg.load_path:
        meta = load_ckpt(Path(cfg.load_path), policies, critic, opt_p, opt_v) or {}

    buffer = RolloutBuffer()
    total_steps = 0

    window_name = "MeltingPot"
    video_writer = None

    for ep in range(cfg.episodes):
        ts = env.reset()
        trust.new_episode(ep)

        # 엔트로피 스케줄러: 20ep 이후 낮춤
        if ep == cfg.ent_coef_warmup_ep:
            cfg.ent_coef = cfg.ent_coef_cool
            print(f"[SCHED] ent_coef -> {cfg.ent_coef} at ep {ep}")

        # 초기 관측 리스트
        try:
            prev_obs_list = [ts.observation[i] for i in range(n)]
        except Exception:
            prev_obs_list = None

        # 렌더/녹화 초기 프레임
        first_frame = None
        if not cfg.headless:
            first_frame = _show_frame_if_any(window_name, prev_obs_list)
        if cfg.record and first_frame is not None and video_writer is None:
            try:
                h, w = first_frame.shape[:2]
                video_writer = imageio.get_writer(cfg.record_path, fps=cfg.record_fps)
                video_writer.append_data(first_frame)
                print(f"[REC] start -> {cfg.record_path} ({w}x{h}@{cfg.record_fps}fps)")
            except Exception as e:
                print(f"[REC] failed to start writer: {e}")
                video_writer = None

        states = [make_state(i, trust, n, partner_history, debt_ij) for i in range(n)]
        obs_global = np.concatenate(states, axis=0)
        ep_reward = np.zeros(n, dtype=np.float32)

        quit_flag = False

        for t in range(cfg.max_steps):
            # 키보드 종료 (창 포커스 시 'q')
            if not cfg.headless:
                try:
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        quit_flag = True
                        break
                except Exception:
                    pass

            actions, logps = [], []
            for i in range(n):
                dist = policies[i](torch.as_tensor(states[i], dtype=torch.float32, device=device))
                a = dist.sample()
                actions.append(int(a.item()))
                logps.append(float(dist.log_prob(a).item()))

            ts = env.step(actions)
            info = getattr(ts, 'extras', {})
            if t % 20 == 0:
                try:
                    keys = list(info.keys()) if isinstance(info, dict) else type(info)
                except Exception:
                    keys = []
                print("[DBG] extras keys:", keys)

            # 현재 관측 리스트
            try:
                curr_obs_list = [ts.observation[i] for i in range(n)]
            except Exception:
                curr_obs_list = None

            # 렌더 & 녹화
            frame = None
            if not cfg.headless:
                frame = _show_frame_if_any(window_name, curr_obs_list)
            if cfg.record and frame is not None and video_writer is not None:
                try:
                    video_writer.append_data(frame)
                except Exception:
                    pass

            # 이벤트 파싱
            events: List[Tuple[int, int, float]] = []
            if isinstance(info, dict) and any(k in info for k in ["gift_events", "events", "gifts"]):
                ev_key = "gift_events" if "gift_events" in info else ("events" if "events" in info else "gifts")
                try:
                    raw = info[ev_key]
                    if isinstance(raw, (list, tuple)):
                        for e in raw:
                            if isinstance(e, (list, tuple)) and len(e) >= 3:
                                events.append((int(e[0]), int(e[1]), float(e[2])))
                except Exception:
                    pass
            else:
                events = parse_gift_events(prev_obs_list, curr_obs_list) if curr_obs_list is not None else []

            if t % 50 == 0:
                print("[DBG] parsed gift events:", events)

            # did_give, partner_history, debt_ij 업데이트
            did_give = [False] * n
            for g, r, amt in events:
                did_give[g] = True
                partner_history[g].append(r)
                debt_ij[r][g] += 1  # r는 g에게 빚 1 증가

            if t % 100 == 0:
                try:
                    upA = len(set(partner_history[0]))
                except Exception:
                    upA = 0
                print("[KPI] give_count=", sum(did_give), "unique_partners_A=", upA)

            base_r = np.asarray(getattr(ts, 'reward', np.zeros(n)), dtype=np.float32)
            shaped = base_r.copy()

            # trait shaping
            for i in range(n):
                partner = None
                if did_give[i]:
                    partner = partner_history[i][-1] if len(partner_history[i]) > 0 else None
                rE, rA = r_trait_components(i, partner, did_give[i])
                E, A = TRAITS[i]['E'], TRAITS[i]['A']
                shaped[i] += ALPHA_E * E * rE + ALPHA_A * A * rA

            ep_reward += shaped

            # GAE/버퍼 적재
            next_states = [make_state(i, trust, n, partner_history, debt_ij) for i in range(n)]
            next_obs_global = np.concatenate(next_states, axis=0)

            with torch.no_grad():
                v_now = critic(torch.as_tensor(obs_global, dtype=torch.float32, device=device)).cpu().numpy()
                v_next = critic(torch.as_tensor(next_obs_global, dtype=torch.float32, device=device)).cpu().numpy()

            for i in range(n):
                buffer.add(
                    obs=torch.tensor(states[i], dtype=torch.float32),
                    act=torch.tensor(actions[i], dtype=torch.int64),
                    old_logp=torch.tensor(logps[i], dtype=torch.float32),
                    rew=torch.tensor(shaped[i], dtype=torch.float32),
                    done=torch.tensor(0.0, dtype=torch.float32),
                    mask_agent=torch.tensor(i),
                    obs_global=torch.tensor(obs_global, dtype=torch.float32),
                    v=torch.tensor(v_now, dtype=torch.float32),
                    v_next=torch.tensor(v_next, dtype=torch.float32),
                )

            # 다음 상태로
            states = next_states
            obs_global = next_obs_global
            prev_obs_list = curr_obs_list

            # rollout 찼으면 업데이트
            if len(buffer) >= cfg.rollout_steps:
                rews = np.array([m['rew'].item() for m in buffer.storage], dtype=np.float32)
                dones = np.array([m['done'].item() for m in buffer.storage], dtype=np.float32)
                vals = np.array([m['v'].mean().item() for m in buffer.storage], dtype=np.float32)
                vals_next = np.array([m['v_next'].mean().item() for m in buffer.storage], dtype=np.float32)
                values = np.concatenate([vals, vals_next[-1:]], axis=0)
                adv, ret = compute_gae(rews, dones, values, cfg.gamma, cfg.lam)
                for k, a in enumerate(adv):
                    buffer.storage[k]['adv'] = torch.tensor(a, dtype=torch.float32)
                for k, r in enumerate(ret):
                    buffer.storage[k]['ret'] = torch.tensor(r, dtype=torch.float32)
                ppo_update(cfg, device, policies, critic, opt_p, opt_v, buffer)
                buffer.clear()

        # 에피소드 종료 로그
        print(f"[ep {ep}] sum_r={ep_reward.sum():.2f} per_agent={ep_reward.tolist()}")

        if quit_flag:
            print("[USER] quit requested (q).")
            break

        # 주기적으로 체크포인트
        if (ep + 1) % cfg.log_interval == 0:
            meta = {"episode": ep, "time": time.time()}
            save_ckpt(Path(cfg.save_dir) / f"marl_ep{ep:04d}.pt", policies, critic, opt_p, opt_v, meta)

    # 남은 버퍼 업데이트
    if len(buffer) > 0:
        rews = np.array([m['rew'].item() for m in buffer.storage], dtype=np.float32)
        dones = np.array([m['done'].item() for m in buffer.storage], dtype=np.float32)
        vals = np.array([m['v'].mean().item() for m in buffer.storage], dtype=np.float32)
        vals_next = np.array([m['v_next'].mean().item() for m in buffer.storage], dtype=np.float32)
        values = np.concatenate([vals, vals_next[-1:]], axis=0)
        adv, ret = compute_gae(rews, dones, values, cfg.gamma, cfg.lam)
        for k, a in enumerate(adv):
            buffer.storage[k]['adv'] = torch.tensor(a, dtype=torch.float32)
        for k, r in enumerate(ret):
            buffer.storage[k]['ret'] = torch.tensor(r, dtype=torch.float32)
        ppo_update(cfg, device, policies, critic, opt_p, opt_v, buffer)
        buffer.clear()

    # 마지막 저장
    meta = {"episode": cfg.episodes - 1, "time": time.time()}
    save_ckpt(Path(cfg.save_dir) / "marl_final.pt", policies, critic, opt_p, opt_v, meta)

    # 창/비디오 정리
    if not cfg.headless:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
    if cfg.record and video_writer is not None:
        try:
            video_writer.close()
            print(f"[REC] saved -> {cfg.record_path}")
        except Exception:
            pass

if __name__ == "__main__":
    main()
