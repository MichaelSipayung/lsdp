#!/usr/bin/env python3
"""compare_runs.py - check that two result folders contain the same Pareto fronts.

usage: python3 bench/compare_runs.py OLD_DIR NEW_DIR
Compares every front file (*.csv, not *.demand.csv, not *_ref.csv, not *.demand) present in
both folders, recursively. Two fronts are equal if they have the same number of points and
every point of one has a partner in the other within the output precision:
|dPOC| <= 2e-4, |dHC| <= 2e-6, |dCSL| <= 1e-7, |dITO| <= 2e-8 (two
half-units of the printed digits; the summation order of the two programmes differs). (Points with numerically equal
objective vectors may be represented by different but equivalent (SS, SOT) vertices, whose
service levels differ only through the rounding of the vertex coordinates.)
"""
import csv
import glob
import os
import sys
from collections import defaultdict

TOL = (2e-4, 2e-6, 1e-7, 2e-8)  # two half-units of the printed digits (POC, HC, ITO)


def load(f):
    return [(float(r["POC"]), float(r["HC"]), float(r["CSL"]), float(r["ITO"])) for r in csv.DictReader(open(f))]


def bucket(p):
    return (int(p[0] // 0.01), int(p[1] // 0.001))


def equal(A, B):
    if len(A) != len(B):
        return False, f"sizes {len(A)} vs {len(B)}"
    idx = defaultdict(list)
    for j, q in enumerate(B):
        idx[bucket(q)].append(j)
    used = [False] * len(B)
    unmatched = 0
    for p in A:
        bx, by = bucket(p)
        found = False
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in idx.get((bx + dx, by + dy), []):
                    if not used[j] and all(abs(p[i] - B[j][i]) <= TOL[i] for i in range(4)):
                        used[j] = found = True
                        break
                if found:
                    break
            if found:
                break
        unmatched += not found
    return unmatched == 0, f"{unmatched} unmatched points"


old, new = sys.argv[1], sys.argv[2]
same = diff = 0
for f in sorted(glob.glob(os.path.join(new, "**", "*.csv"), recursive=True)):
    if f.endswith(".demand.csv") or f.endswith("_ref.csv"):
        continue
    g = os.path.join(old, os.path.relpath(f, new))
    if not os.path.exists(g):
        continue
    try:
        ok, why = equal(load(g), load(f))
    except (KeyError, ValueError):
        continue
    if ok:
        same += 1
    else:
        diff += 1
        print(f"DIFFERENT: {os.path.relpath(f, new)}  ({why})")
print(f"{new} vs {old}: {same} identical fronts, {diff} different")
