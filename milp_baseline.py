"""milp_baseline.py — exact generic baseline: epsilon-constraint MILP (HiGHS via
scipy.optimize.milp) for the bi-objective (POC, HC) inner problem of the
Kania et al. (CIE 2022) model at one fixed outer point (SS, SOT).
ITO (a sum of reciprocals of inventory) is not MILP-representable without large
piecewise encodings, so the exact MILP comparison is made on the sub-model
without ITO; lsdp is run on the identical problem with `--noito 1`.

For every bound k_max = 0..T' on the number of orders:
  stage 1: min HC            s.t. sum(y) <= k_max, feasibility
  stage 2: min POC           s.t. HC <= HC* (lexicographic refinement)
then the non-dominated (POC, HC) points are kept. Because the number of orders is an
integer in 0..T', this enumerates the complete non-dominated set.

usage: python milp_baseline.py --sku 1 --demand hasil/sku1_s42.csv.demand.csv \
         --SS 50 --SOT 0.3 --out milp_sku1.csv
"""
import argparse
import csv
import math
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from lsmodel import Instance


def build(inst, SS, SOT):
    Tp, L, r = inst.Tp, inst.L, inst.r
    beta = np.array([SS + SOT * inst.D[tau + L] + inst.cumD[tau + L] - inst.I0
                     for tau in range(1, Tp + 1)])
    base = np.array([inst.I0 - 0.5 * (inst.cumD[tau + L - 1] + inst.cumD[tau + L])
                     for tau in range(1, Tp + 1)])
    C0 = sum(inst.I0 - 0.5 * (inst.cumD[t - 1] + inst.cumD[t]) for t in range(1, inst.T + 1))
    U = int(math.ceil((max(beta.max(), 0) + inst.moq) / r)) + 1
    n = 2 * Tp  # variables: a_1..a_Tp (integer lots), y_1..y_Tp (binary order flags)
    rows, lo, hi = [], [], []
    for t in range(Tp):                       # a_t <= U*y_t ; a_t >= y_t  (Q = r*a, moq = r)
        e = np.zeros(n); e[t] = 1; e[Tp + t] = -U; rows.append(e); lo.append(-np.inf); hi.append(0)
        e = np.zeros(n); e[t] = 1; e[Tp + t] = -1; rows.append(e); lo.append(0); hi.append(np.inf)
    cum = np.tril(np.ones((Tp, Tp)))          # X(tau) = r * sum_{s<=tau} a_s
    for t in range(Tp):                       # X(tau) >= beta_tau
        e = np.zeros(n); e[:Tp] = r * cum[t]; rows.append(e); lo.append(beta[t]); hi.append(np.inf)
    for t in range(Tp):                       # Ibar >= 1 : base + (X(tau-1)+X(tau))/2 >= 1
        e = np.zeros(n)
        e[:Tp] = 0.5 * r * cum[t] + (0.5 * r * cum[t - 1] if t > 0 else 0)
        rows.append(e); lo.append(1 - base[t]); hi.append(np.inf)
    A = np.array(rows)
    hc = np.zeros(n)
    hc[:Tp] = inst.h * np.array([r * (Tp - s) - 0.5 * r for s in range(Tp)])  # s = 0..Tp-1
    poc = np.zeros(n); poc[:Tp] = inst.p * r; poc[Tp:] = inst.c
    korders = np.zeros(n); korders[Tp:] = 1
    integ = np.ones(n)
    bnds = Bounds(np.zeros(n), np.concatenate([np.full(Tp, U), np.ones(Tp)]))
    return dict(A=A, lo=np.array(lo), hi=np.array(hi), hc=hc, hc0=inst.h * C0, poc=poc,
                k=korders, integ=integ, bnds=bnds, Tp=Tp)


def solve(m, obj, extra, tlim):
    A = np.vstack([m["A"]] + [e[0][None, :] for e in extra])
    lo = np.concatenate([m["lo"], [e[1] for e in extra]])
    hi = np.concatenate([m["hi"], [e[2] for e in extra]])
    res = milp(c=obj, integrality=m["integ"], bounds=m["bnds"],
               constraints=LinearConstraint(A, lo, hi),
               options={"time_limit": tlim, "mip_rel_gap": 1e-9, "presolve": True})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sku", type=int, required=True)
    ap.add_argument("--demand", required=True)
    ap.add_argument("--SS", type=float, required=True)
    ap.add_argument("--SOT", type=float, required=True)
    ap.add_argument("--tlim", type=float, default=600.0, help="time limit per MILP (s)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    inst = Instance(a.sku, a.demand)
    m = build(inst, a.SS, a.SOT)
    t0 = time.time()
    pts, nsolve, nopt = [], 0, 0
    for kmax in range(0, m["Tp"] + 1):
        r1 = solve(m, m["hc"], [(m["k"], -np.inf, kmax)], a.tlim); nsolve += 1
        if r1.status != 0:
            continue  # infeasible (too few orders) or time limit
        hcstar = float(m["hc"] @ r1.x)
        r2 = solve(m, m["poc"], [(m["k"], -np.inf, kmax), (m["hc"], -np.inf, hcstar + 1e-6)], a.tlim)
        nsolve += 1
        x = np.round(r2.x if r2.status == 0 else r1.x)
        nopt += int(r1.status == 0 and r2.status == 0)
        Tp = m["Tp"]
        Xt = inst.r * x[:Tp].sum()
        k = int(x[Tp:].sum())
        pts.append((inst.p * Xt + inst.c * k, m["hc0"] + float(m["hc"] @ x), k, Xt,
                    [int(v) * inst.r for v in x[:Tp]]))
    wall = time.time() - t0
    nd = []
    for i, p in enumerate(pts):
        dom = any((q[0] <= p[0] + 1e-6 and q[1] <= p[1] + 1e-6 and (q[0] < p[0] - 1e-6 or q[1] < p[1] - 1e-6))
                  or (abs(q[0] - p[0]) <= 1e-6 and abs(q[1] - p[1]) <= 1e-6 and j < i)
                  for j, q in enumerate(pts) if j != i)
        if not dom:
            nd.append(p)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["POC", "HC", "orders", "XT"] + [f"Q{t}" for t in range(1, inst.T + 1)])
        for p in sorted(nd):
            w.writerow([f"{p[0]:.4f}", f"{p[1]:.6f}", p[2], int(p[3])] + p[4] + [0] * inst.L)
    print(f"MILP eps-constraint SKU{a.sku} SS={a.SS} SOT={a.SOT}: {nsolve} MILP solves "
          f"({nopt} optimal pairs), non-dominated points {len(nd)}, wall {wall:.2f}s")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
