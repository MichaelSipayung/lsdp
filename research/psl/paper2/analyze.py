#!/usr/bin/env python3
"""analyze.py - regret distribution analysis of the evaluation detail files (Paper 2).

usage (from research/psl):  python3 paper2/analyze.py
Reads eval_{syn,e6}_<model>.txt (lines: name l1 l2 l3 l4 reg_dec reg_best reg_greedy) and prints,
per data set and method: share of pairs with regret <= 1e-6 (optimal for the scalarisation),
<= 0.01, <= 0.05, > 0.5; the same split by preference type (vertex = one weight equal to 1,
edge = two non-zero weights, interior = three or four); and, for E6, by data source.
Writes paper2/tables/dist.tex and paper2/tables/preftype.tex.
"""
import os, statistics
from collections import defaultdict

MODELS = [("model_v3", "Order events"), ("model_q", "Quantities (5000 it.)"), ("model_q.it3000", "Quantities (equal time)")]
os.makedirs("paper2/tables", exist_ok=True)

def load(fn):
    R = []
    for line in open(fn):
        p = line.split()
        if len(p) == 8:
            lam = [float(x) for x in p[1:5]]
            R.append((p[0], lam, float(p[5]), float(p[6]), float(p[7])))
    return R

def ptype(lam):
    nz = sum(1 for x in lam if x > 1e-9)
    return "vertex" if nz == 1 else ("edge" if nz == 2 else "interior")

def shares(v):
    n = len(v)
    return [100 * sum(x <= 1e-6 for x in v) / n, 100 * sum(x <= 0.01 for x in v) / n,
            100 * sum(x <= 0.05 for x in v) / n, 100 * sum(x > 0.5 for x in v) / n, statistics.median(v)]

dist = [r"\begin{tabular}{llrrrrr}", r"\toprule",
        r"Data & Method & reg $\le10^{-6}$ (\%) & $\le0.01$ (\%) & $\le0.05$ (\%) & $>0.5$ (\%) & Median\\", r"\midrule"]
pref = [r"\begin{tabular}{llrrr}", r"\toprule",
        r"Data & Method & Vertex & Edge & Interior\\", r"\midrule"]
for data, title in (("syn", "Synthetic"), ("e6", "Real (E6)")):
    base_done = False
    for model, label in MODELS:
        fn = f"eval_{data}_{model}.txt"
        if not os.path.exists(fn):
            print("missing", fn); continue
        R = load(fn)
        rows = [(label + ", greedy decoding", [r[2] for r in R]), (label + ", best of 17", [r[3] for r in R])]
        if not base_done:
            rows.append(("Greedy heuristic", [r[4] for r in R])); base_done = True
        print(f"\n== {title}: {fn} ({len(R)} pairs)")
        print(f"{'method':42s} {'=0':>6s} {'<=.01':>6s} {'<=.05':>6s} {'>0.5':>6s} {'median':>8s}")
        for name, v in rows:
            s = shares(v)
            print(f"{name:42s} {s[0]:6.1f} {s[1]:6.1f} {s[2]:6.1f} {s[3]:6.2f} {s[4]:8.4f}")
            dist.append(f"{title} & {name} & {s[0]:.1f} & {s[1]:.1f} & {s[2]:.1f} & {s[3]:.2f} & {s[4]:.4f}\\\\")
        # by preference type (median regret, greedy decoding)
        by = defaultdict(list)
        for r in R: by[ptype(r[1])].append(r[2])
        med = {k: statistics.median(v) for k, v in by.items()}
        print("median by preference type (greedy decoding):", {k: round(v, 4) for k, v in med.items()},
              {k: len(v) for k, v in by.items()})
        pref.append(f"{title} & {label} & {med.get('vertex', float('nan')):.4f} & {med.get('edge', float('nan')):.4f} & {med.get('interior', float('nan')):.4f}\\\\")
        if data == "e6":
            src = defaultdict(list)
            for r in R: src[r[0].split("/")[0]].append(r[2])
            print("median by source (greedy decoding):", {k: (round(statistics.median(v), 4), len(v) // 84) for k, v in src.items()})
    dist.append(r"\midrule")
dist[-1] = r"\bottomrule"; dist.append(r"\end{tabular}")
pref += [r"\bottomrule", r"\end{tabular}"]
open("paper2/tables/dist.tex", "w").write("\n".join(dist) + "\n")
open("paper2/tables/preftype.tex", "w").write("\n".join(pref) + "\n")
print("\nwritten paper2/tables/dist.tex, paper2/tables/preftype.tex")
