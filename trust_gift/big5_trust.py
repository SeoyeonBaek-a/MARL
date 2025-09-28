# run_gift_trust_rendering_EA.py
# Gift Refinements (Melting Pot)
# - Headless(창 없음, 최대 속도)
# - 신뢰(C) 파라미터는 전원 동일(개인차 X)
# - 행동정책에만 E/A 성격 보상(shaping)을 반영

import numpy as np
import cv2
from pathlib import Path
from collections import deque
from big5_trust_logger import TrustLogger
# =========================
# 0) 파라미터 & 에이전트 성격
# =========================
HEADLESS = True
NUM_EPISODES = 50
MAX_STEPS = 100
np.random.seed(42)

REFINE_ID = 7
GIFT_ID   = 8

# 신뢰 파라미터(전원 동일)
NAMES  = ["A","B","C","D","E","F","G","H"]
DELTAS = [0.20]*8
OMEGAS = [1.00]*8

# 성격 (O 제거 → E/A만 개인차)
TRAITS = [
    {"name":"A","E":0.8,"A":0.8},
    {"name":"B","E":0.8,"A":0.2},
    {"name":"C","E":0.2,"A":0.8},
    {"name":"D","E":0.2,"A":0.2},
    {"name":"E","E":0.8,"A":0.8},
    {"name":"F","E":0.8,"A":0.2},
    {"name":"G","E":0.2,"A":0.8},
    {"name":"H","E":0.2,"A":0.2},
]

ALPHA_E, ALPHA_A = 1.0, 1.0
HIST_LEN = 20

# 기존 CSV 정리
for p in ["gift_trust_mu.csv", "gift_events.csv", "gift_trust_C.csv"]:
    Path(p).unlink(missing_ok=True)

# =========================
# 1) 환경 빌드
# =========================
def build_env(num_players = 8):
    from meltingpot import substrate
    from meltingpot.configs.substrates import gift_refinements as cfg
    config = cfg.get_config()
    roles = getattr(config, "default_player_roles", None)
    if roles is None or len(roles) != num_players:
        roles = ["default"] * num_players
    env = substrate.build("gift_refinements", roles=roles)
    print(f"[build_env] using substrate: gift_refinements, roles={roles}")
    return env

def per_player_num_values(env):
    specs = env.action_spec()
    out = []
    for s in specs:
        nv = getattr(s, "num_values", None)
        if nv is None:
            mn = int(np.asarray(getattr(s, "minimum")))
            mx = int(np.asarray(getattr(s, "maximum")))
            nv = mx - mn + 1
        out.append(int(nv))
    return out

# =========================
# 2) OpenCV (headless)
# =========================
def live_open(title="Gift Refinements (live)"):
    if HEADLESS:
        return None
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 720, 720)
    return title

def live_show(title, frame):
    if HEADLESS or title is None:
        return
    if frame is None:
        cv2.waitKey(1); return
    bgr = frame[:, :, ::-1]
    cv2.imshow(title, bgr)
    cv2.waitKey(1)

# =========================
# 3) RGB 탐색
# =========================
def _as_uint8(arr):
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    return arr

def _find_rgb_anywhere(obj):
    if isinstance(obj, np.ndarray):
        if obj.ndim == 3 and obj.shape[-1] == 3:
            return _as_uint8(obj)
        return None
    if isinstance(obj, dict):
        for k in ("WORLD.RGB","world.rgb","RGB","rgb","world_rgb"):
            if k in obj and isinstance(obj[k], np.ndarray):
                return _as_uint8(obj[k])
        for v in obj.values():
            hit = _find_rgb_anywhere(v)
            if hit is not None: return hit
        return None
    if isinstance(obj, (list,tuple)):
        for v in obj:
            hit = _find_rgb_anywhere(v)
            if hit is not None: return hit
        return None
    return None

def extract_frame(obs, info=None):
    if isinstance(info, dict):
        hit = _find_rgb_anywhere(info)
        if hit is not None: return hit
    if isinstance(obs,(list,tuple)) and len(obs)>0:
        hit = _find_rgb_anywhere(obs[0])
        if hit is not None: return hit
        for o in obs:
            hit = _find_rgb_anywhere(o)
            if hit is not None: return hit
        return None
    return _find_rgb_anywhere(obs)

# =========================
# 4) 매칭
# =========================
def nearest_matching(givers, receivers, positions):
    pairs = []
    receivers_left = receivers.copy()
    for g in givers:
        if not receivers_left: break
        dists = [np.linalg.norm(positions[g]-positions[r]) for r in receivers_left]
        r = receivers_left[int(np.argmin(dists))]
        pairs.append((g, r))
        receivers_left.remove(r)
    return pairs

def greedy_zip_matching(givers, receivers):
    m = min(len(givers), len(receivers))
    return [(givers[i], receivers[i]) for i in range(m)]

# =========================
# 5) 성격 보상 기반 행동정책 (O 제거)
# =========================
partner_history = None
# 🔥 전역 부채 → 페어별 부채
debt_ij = None   # list[defaultdict(int)], i: 주체, debt_ij[i][j] = i가 j에게 진 부채 수

def init_trait_buffers(n):
    global partner_history, debt_ij
    from collections import defaultdict
    partner_history = [deque(maxlen=HIST_LEN) for _ in range(n)]
    debt_ij = [defaultdict(int) for _ in range(n)]

def r_trait_components(i, partner, will_give):
    """성격 보상 r_E, r_A 계산 (최소 패치)"""
    # ----- Extraversion: 상호작용 빈도/개시 -----
    unique_partners = len(set(partner_history[i]))
    r_E = (1.0 if will_give else 0.0) + 0.05*unique_partners

    # ----- Agreeableness: 호혜/용서 -----
    # 🔥 repay_bonus 약화 & 페어별로만 적용 (0.5 → 0.15)
    if will_give and (partner is not None):
        owed = debt_ij[i].get(partner, 0)
        repay_bonus = 0.15 if owed > 0 else 0.0
    else:
        repay_bonus = 0.0
    # 🔥 용서 보너스 완화 (0.2 → 0.05)
    forgive_bonus = 0.05 if (partner is not None and will_give and partner not in partner_history[i]) else 0.0

    r_A = repay_bonus + forgive_bonus
    return r_E, r_A

def mellowmax_probs(scores, omega=1.0, eps=1e-8):
    tau = 1.0 / max(omega, 1e-6)
    z = (scores / tau) - np.max(scores)
    e = np.exp(z)
    return e / (e.sum() + eps)

def pick_partner_by_trust(i, trust):
    row = trust.C[i].copy()
    row[i] = 0.0
    s = row.sum()
    if s <= 1e-9:
        return None
    probs = row / s
    return int(np.random.choice(len(row), p=probs))

def choose_action(i, trust, action_space=("noop","gift","refine+gift"), omega=1.0):
    partner = pick_partner_by_trust(i, trust)

    E, A = TRAITS[i]["E"], TRAITS[i]["A"]
    cand = []
    for act in action_space:
        will_give = (act in ("gift","refine+gift"))
        r_env = 1.0 if will_give else 0.0
        rE, rA = r_trait_components(i, partner, will_give)
        r_trait = ALPHA_E*E*rE + ALPHA_A*A*rA
        score = r_env + r_trait
        cand.append((act, score, partner, will_give))

    scores = np.array([c[1] for c in cand], dtype=np.float32)
    probs  = mellowmax_probs(scores, omega=omega)
    idx = int(np.random.choice(len(cand), p=probs))
    return cand[idx]

# =========================
# 6) 메인 실행
# =========================
def main():
    global REFINE_ID, GIFT_ID

    num_players = 8
    env = build_env(num_players)
    n = getattr(env, "num_players", num_players)
    names = NAMES[:n]
    print(f"[init] players={n}, names={names}")

    trust = TrustLogger(
        n, names, DELTAS[:n], OMEGAS[:n],
        # 🔥 초기 신뢰 낮춤: 0.5 → 0.3
        init_trust=0.3, reciprocity_window=5, debt_horizon=3,
        log_interval=1,
        # 🔥 E/A 전달(로거는 E/A 가중치만 반영, 나머지는 원본 로직 유지)
        extraversion=[t["E"] for t in TRAITS],
        agreeableness=[t["A"] for t in TRAITS]
    )

    init_trait_buffers(n)

    win = live_open()
    per_nv = per_player_num_values(env)
    print(f"[action_spec] per-player num_values={per_nv}")
    print(f"[use] REFINE_ID={REFINE_ID}, GIFT_ID={GIFT_ID}")

    cum_gifts = 0
    try:
        for ep in range(NUM_EPISODES):
            ts = env.reset()
            trust.new_episode(ep)

            obs = getattr(ts, "observation", ts)
            info = getattr(ts, "extras", {}) if hasattr(ts, "extras") else {}
            frame = extract_frame(obs, info); live_show(win, frame)

            gifts_ep = 0
            c_refine = c_consume = c_pairs = 0

            for t in range(MAX_STEPS):
                actions = [0]*n
                chosen_partner = [None]*n
                did_give = [False]*n

                for i in range(n):
                    act, _score, partner, will_give = choose_action(
                        i, trust, action_space=("noop","gift","refine+gift"), omega=1.0
                    )
                    chosen_partner[i] = partner
                    did_give[i] = bool(will_give)

                    num_values = per_nv[0] if len(per_nv)==1 else per_nv[i]
                    if act == "refine+gift" and REFINE_ID < num_values:
                        actions[i] = REFINE_ID
                    elif act == "gift" and GIFT_ID < num_values:
                        actions[i] = GIFT_ID
                    else:
                        actions[i] = 0

                ts = env.step(actions)
                obs = getattr(ts, "observation", ts)
                info = getattr(ts, "extras", {}) if hasattr(ts, "extras") else {}

                givers = [i for i,a in enumerate(actions) if a == REFINE_ID]
                receivers = [i for i,a in enumerate(actions) if a == GIFT_ID]
                c_refine += len(givers)
                c_consume += len(receivers)

                events = []
                if givers and receivers:
                    pos = None
                    if isinstance(info, dict) and "positions" in info:
                        try:
                            pos = np.array(info["positions"]).reshape(n, -1)
                        except Exception:
                            pos = None
                    pairs = nearest_matching(givers, receivers, pos) if pos is not None else greedy_zip_matching(givers, receivers)
                    c_pairs += len(pairs)
                    for (g, r) in pairs:
                        gifts_ep += 1
                        events.append((g, r, 1.0))

                # ===== 최소 패치: 수신/상환을 '페어별'로 관리 =====
                did_receive_from = [set() for _ in range(n)]
                for (g, r, amt) in events:
                    if amt > 0:
                        did_receive_from[r].add(g)
                        # r는 g에게 부채 1건 추가
                        debt_ij[r][g] += 1

                for i in range(n):
                    p = chosen_partner[i]
                    if p is not None:
                        partner_history[i].append(p)
                    # i가 이번 스텝에 p에게 '줬고', p에 대해 부채가 있으면 1건 상환
                    if did_give[i] and (p is not None) and (debt_ij[i].get(p, 0) > 0):
                        debt_ij[i][p] -= 1
                        # (선택) 로거에 상환 신호를 별도로 보내고 싶으면:
                        # trust.mark_repay(i, p)

                trust.update(events)
                # ===============================================

                frame = extract_frame(obs, info); live_show(win, frame)
                if hasattr(ts, "last") and ts.last():
                    break

            cum_gifts += gifts_ep
            print(f"[episode {ep}] gifts={gifts_ep}, cum={cum_gifts}")

    finally:
        trust.dumps()
        if win is not None:
            cv2.destroyWindow(win)

if __name__ == "__main__":
    main()
