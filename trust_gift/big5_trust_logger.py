import numpy as np
from collections import defaultdict, deque

def mellowmax(x, omega, axis=-1, eps=1e-8):
    x = np.asarray(x, dtype=np.float32)
    tau = 1.0 / max(omega, 1e-6)
    z = x / tau
    z -= np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / (np.sum(e, axis=axis, keepdims=True) + eps)

class TrustLogger:
    def __init__(self, n, names, deltas, omegas,
                 init_trust=0.5, reciprocity_window=5, debt_horizon=3,
                 pos_weight=1.0, neg_weight=1.0,
                 log_interval=1,
                 # 🔥 추가된 인자
                 extraversion=None, agreeableness=None):
        assert len(deltas) == n and len(omegas) == n
        self.n, self.names = n, names
        self.delta = np.array(deltas, dtype=np.float32)   # δ_i
        self.omega = np.array(omegas, dtype=np.float32)   # ω_i
        self.init_trust = float(init_trust)
        self.reciprocity_hist = deque(maxlen=reciprocity_window)
        self.debt_horizon = int(debt_horizon)
        self.pos_w, self.neg_w = float(pos_weight), float(neg_weight)
        self.log_interval = int(log_interval)

        # 🔥 E/A 성격 벡터 추가
        if extraversion is None:
            extraversion = np.full(n, 0.5, dtype=np.float32)
        if agreeableness is None:
            agreeableness = np.full(n, 0.5, dtype=np.float32)
        self.E = np.asarray(extraversion, dtype=np.float32)
        self.A = np.asarray(agreeableness, dtype=np.float32)

        self.reset()

    def reset(self):
        self.t = 0
        self.ep = 0
        self.C = np.full((self.n, self.n), self.init_trust, dtype=np.float32)
        np.fill_diagonal(self.C, 0.0)
        self.last_gifts_from_to = defaultdict(lambda: deque(maxlen=self.debt_horizon))
        self.M_rows, self.E_rows, self.C_rows = [], [], []

    def new_episode(self, ep_idx:int):
        self.ep = int(ep_idx)
        self.t = 0
        self.last_gifts_from_to.clear()

    def _update_C(self, signals):
        for i in range(self.n):
            for j in range(self.n):
                if i == j or signals[j] == 0:
                    continue
                if signals[j] > 0:
                    target = 1.0
                    w = self.pos_w * (1.0 + 0.8*(self.E[i]-0.5) + 0.2*(self.A[i]-0.5))
                else:
                    target = 0.0
                    w = self.neg_w * (1.0 - 0.8*(self.A[i]-0.5))
                w = max(w, 0.05)
                self.C[i, j] = (1 - self.delta[i]) * self.C[i, j] + self.delta[i] * (w * target + (1 - w) * self.C[i, j])

    def _log_mu(self):
        for i in range(self.n):
            mu = mellowmax(self.C[i], self.omega[i]).astype(float)
            mu[i] = 0.0
            s = float(np.sum(mu))
            mu = (mu / s).tolist() if s > 0 else mu.tolist()
            self.M_rows.append((self.ep, self.t, self.names[i], mu))

    def _log_C(self):
        if self.log_interval > 1 and (self.t % self.log_interval) != 0:
            return
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    continue
                self.C_rows.append((self.ep, self.t, self.names[i], self.names[j], float(self.C[i, j])))

    def update(self, gift_events):
        for (g, r, amt) in gift_events:
            self.E_rows.append((self.ep, self.t, self.names[g], self.names[r], float(amt)))

        recip_den = 0; recip_cnt = 0
        gave_pairs = {(g, r) for (g, r, _a) in gift_events}
        for (r, g), q in list(self.last_gifts_from_to.items()):
            if len(q):
                recip_den += 1
                if (g, r) in gave_pairs:
                    recip_cnt += 1
        if recip_den:
            self.reciprocity_hist.append(recip_cnt / recip_den)

        overdue_pairs = []
        for (giver, receiver), q in list(self.last_gifts_from_to.items()):
            q.append(self.t)
            if len(q) == q.maxlen and (giver, receiver) not in gave_pairs:
                overdue_pairs.append((giver, receiver))

        for (g, r, _a) in gift_events:
            self.last_gifts_from_to[(r, g)].clear()
            self.last_gifts_from_to[(r, g)].append(self.t)

        signals = np.zeros(self.n, dtype=np.int8)
        for (g, _r, _a) in gift_events: signals[g] = 1
        for (_victim, offender) in overdue_pairs: signals[offender] = -1

        self._update_C(signals)
        self._log_mu()
        self._log_C()
        self.t += 1

    def dumps(self, prefix="gift"):
        import csv
        with open(f"{prefix}_trust_mu.csv","w",newline="",encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["ep","t","agent","mu_vec_json"]); w.writerows(self.M_rows)
        with open(f"{prefix}_events.csv","w",newline="",encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["ep","t","giver","receiver","amount"]); w.writerows(self.E_rows)
        with open(f"{prefix}_trust_C.csv","w",newline="",encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["ep","t","src","tgt","C_ij"]); w.writerows(self.C_rows)
