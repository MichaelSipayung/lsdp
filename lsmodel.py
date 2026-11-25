"""lsmodel.py — vectorised evaluator of the Kania et al. (CIE 2022) lot sizing model,
identical to the formulas used in lsdp.cpp (same instance construction).

Decision encoding (shared by NSGA-III and the exact DP):
  a[0..Tp-1]  integer, order in period tau = a * r   (moq = r for both SKUs)
  ss          integer, safety stock in units
  ksot        integer, SOT = 0.05 * ksot weeks
Objectives (minimisation form): [POC, HC, -CSL, -ITO]
Constraints (<= 0): SS/mu + SOT - MS ;  SS + SOT*D(t) - I(t) for t=L+1..T ;  1 - Ibar(t)
"""
import csv
import math
import numpy as np

PRESETS = {
    1: dict(p=134.0, c=200.0, L=5, moq=70, r=70, mu=163.0, sigma=41.0, MS=1.0),
    2: dict(p=483.85, c=200.0, L=4, moq=3, r=3, mu=12.2, sigma=3.6, MS=1.4),
}
DSOT = 0.05


class Instance:
    def __init__(self, sku, demand_csv):
        P = PRESETS[sku]
        self.sku = sku
        self.__dict__.update(P)
        self.h = self.p * 0.1 / 52.0
        D = [int(row["D"]) for row in csv.DictReader(open(demand_csv))]
        self.T = len(D)
        self.D = np.array([0] + D, dtype=float)            # D[1..T]
        self.cumD = np.concatenate([[0.0], np.cumsum(D)])   # cumD[0..T]
        self.I0 = self.cumD[self.L] + round(self.mu)       # same as lsdp.cpp (lround)
        self.Tp = self.T - self.L
        self.ss_max = int(math.floor(self.mu * self.MS + 1e-9))
        self.ksot_max = int(round(self.MS / DSOT))
        # generous upper bound on one order (in lattice units): total requirement + one lot
        beta_max = self.MS * self.mu + self.D.max() * self.MS + self.cumD[self.T] - self.I0
        self.a_max = int(math.ceil((max(beta_max, 0) + self.moq) / self.r)) + 1

    # X: (n, Tp+2) integer matrix [a_1..a_Tp, ss, ksot]
    def evaluate(self, X):
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        T, L, Tp = self.T, self.L, self.Tp
        Q = X[:, :Tp] * self.r                                  # orders tau = 1..Tp
        SS = X[:, Tp]
        SOT = X[:, Tp + 1] * DSOT
        Xc = np.concatenate([np.zeros((n, 1)), np.cumsum(Q, axis=1)], axis=1)  # X(0..Tp)
        # cumulative arrivals by end of period t: X(t-L) for t>L else 0
        arr = np.zeros((n, T + 1))
        arr[:, L + 1:] = Xc[:, 1:Tp + 1]
        I = self.I0 + arr - self.cumD[None, :]                  # I(0..T)
        Ibar = 0.5 * (I[:, :-1] + I[:, 1:])                     # periods 1..T
        n_orders = (Q > 0).sum(axis=1)
        POC = self.p * Xc[:, Tp] + self.c * n_orders
        HC = self.h * Ibar.sum(axis=1)
        z = (SS + self.mu * SOT) / self.sigma
        CSL = 0.5 * np.array([math.erfc(-v / math.sqrt(2.0)) for v in z])
        with np.errstate(divide="ignore", invalid="ignore"):
            ITO = ((self.D[None, 1:] + self.sigma) / np.maximum(Ibar, 1e-9)).sum(axis=1)
        F = np.column_stack([POC, HC, -CSL, -ITO])
        g_ms = SS / self.mu + SOT - self.MS - 1e-9
        req = SS[:, None] + SOT[:, None] * self.D[None, L + 1:]
        g_inv = req - I[:, L + 1:] - 1e-7                        # <= 0 (float tolerance)
        g_ibar = 1.0 - Ibar[:, L:] - 1e-9                        # <= 0, periods L+1..T as in lsdp.cpp
        G = np.column_stack([g_ms, g_inv, g_ibar])
        return F, G, dict(POC=POC, HC=HC, CSL=CSL, ITO=ITO, orders=n_orders)


def repair(inst, X):
    """Problem-specific repair (strong baseline): makes every individual feasible and
    structurally sound, using the paper-independent properties proven for the DP:
      * (SS, SOT) clipped into the domain SS/mu + SOT <= MS,
      * Lemma D: no order when the requirement is already met (unless Ibar < 1),
      * an order that is needed is raised to the smallest lattice quantity that meets
        the requirement (larger proposed orders are kept: diversity),
      * no overshoot beyond the terminal level kappa = ceil_lattice(max beta).
    X: integer matrix [a_1..a_Tp, ss, ksot]; returns a repaired copy."""
    X = np.array(X, dtype=float, copy=True)
    Tp, L, r = inst.Tp, inst.L, inst.r
    for i in range(X.shape[0]):
        ksot = int(np.clip(round(X[i, Tp + 1]), 0, inst.ksot_max))
        SOT = ksot * DSOT
        ss = int(np.clip(round(X[i, Tp]), 0, math.floor(inst.mu * (inst.MS - SOT) + 1e-9)))
        X[i, Tp], X[i, Tp + 1] = ss, ksot
        beta = [ss + SOT * inst.D[t + L] + inst.cumD[t + L] - inst.I0 for t in range(1, Tp + 1)]
        bmax = max(beta)
        kappa = 0 if bmax <= 0 else int(math.ceil((bmax - 1e-9) / r)) * r
        x = 0
        for t in range(Tp):
            base = inst.I0 - 0.5 * (inst.cumD[t + L] + inst.cumD[t + L + 1])
            q = int(max(0, round(X[i, t]))) * r
            needed = (x + 1e-9 < beta[t]) or (base + x < 1.0)
            if not needed:
                q = 0
            else:
                need = max(beta[t] - x, 0.0)
                qmin = max(r, int(math.ceil((need - 1e-9) / r)) * r)
                while base + x + 0.5 * qmin < 1.0:   # Ibar >= 1
                    qmin += r
                q = max(q, qmin)
                q = min(q, max(qmin, kappa - x))     # no overshoot beyond kappa
            X[i, t] = q // r
            x += q
    return X
