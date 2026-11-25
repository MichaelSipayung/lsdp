"""summarize.py — collect all lsdp logs of the suite into tables (mean ± sd).
usage: python3 summarize.py suite   (writes suite/summary_*.csv and prints tables)"""
import csv
import glob
import os
import re
import statistics as st
import sys

root = sys.argv[1] if len(sys.argv) > 1 else "suite"
PAT = {
    "T": r"instance: SKU\d+ T=(\d+)",
    "cands": r"-> (\d+) (?:combinations|candidates)",
    "keys": r", (\d+) distinct requirement vectors",
    "labels": r"labels generated \(total\): (\d+)",
    "candpts": r"candidate points: (\d+)",
    "front": r"Pareto front: (\d+)",
    "time": r"TOTAL ([\d.]+)s",
}


def parse(path):
    s = open(path).read()
    d = {}
    for k, p in PAT.items():
        m = re.search(p, s)
        d[k] = float(m.group(1)) if m else float("nan")
    csvf = path[:-4] + ".csv"
    if os.path.exists(csvf):
        R = list(csv.DictReader(open(csvf)))
        if R and "attained" in R[0]:
            d["attained_pct"] = 100.0 * sum(int(r["attained"]) for r in R) / len(R)
    return d


def ms(v):
    v = [x for x in v if x == x]
    if not v:
        return "-"
    return f"{st.mean(v):.4g} ± {st.stdev(v):.3g}" if len(v) > 1 else f"{v[0]:.4g}"


def table(title, groups, cols):
    print(f"\n### {title}")
    print("| config | n | " + " | ".join(cols) + " |")
    print("|---|---|" + "---|" * len(cols))
    rows = []
    for g, L in sorted(groups.items()):
        print(f"| {g} | {len(L)} | " + " | ".join(ms([d.get(c, float('nan')) for d in L]) for c in cols) + " |")
        rows.append([g, len(L)] + [st.mean([d[c] for d in L if c in d and d[c] == d[c]] or [float('nan')]) for c in cols])
    return rows


def group(pattern, keyfun):
    G = {}
    for f in glob.glob(os.path.join(root, pattern)):
        k = keyfun(os.path.basename(f))
        if k:
            G.setdefault(k, []).append(parse(f))
    return G


# E2
G2 = group("E2/*.log", lambda b: re.sub(r"_s\d+", "", b[:-4]))
table("E2 main study (T=48, 10 demand seeds)", G2, ["time", "cands", "keys", "labels", "front", "attained_pct"])
# E3 single DP scaling
G3 = group("E3/one_*.log", lambda b: re.sub(r"_s\d+", "", b[4:-4]))
table("E3 one inner DP vs horizon T", G3, ["time", "labels", "front"])
G3f = group("E3/full_*.log", lambda b: b[5:-4])
if G3f:
    table("E3 full continuous front at T=96", G3f, ["time", "keys", "labels", "front", "attained_pct"])
# E4
G4 = group("E4/*.log", lambda b: re.sub(r"_s\d+", "", b[:-4]))
table("E4 sensitivity (SKU1, T=48, continuous, 5 seeds)", G4, ["time", "keys", "labels", "front", "attained_pct"])
# E1
v = os.path.join(root, "E1", "verify.log")
if os.path.exists(v):
    print("\n### E1 correctness\n" + open(v).read().strip().splitlines()[-1])
vc = os.path.join(root, "E1", "verify_cont.log")
if os.path.exists(vc):
    L = open(vc).read().splitlines()
    print(f"verify-cont: {sum('OK' in x for x in L)} / {len(L)} OK")
