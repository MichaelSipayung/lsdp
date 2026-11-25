"""compare.py — quality of an approximate front (e.g. NSGA-III) against the exact
Pareto front produced by lsdp.cpp, on the same instance.

Objectives in minimisation form: POC, HC, -CSL, -ITO, normalised with the ideal and
nadir of the exact front. Reported indicators:
  * share of approximate points that are Pareto optimal (lie on the exact front)
  * share of approximate points strictly dominated by the exact front
  * share of exact points dominated by the approximate set  (must be 0: sanity check)
  * IGD+ (Ishibuchi et al. 2015) w.r.t. the exact front
  * additive epsilon indicator eps+(A, P*) (Zitzler et al. 2003)
usage: python compare.py --exact hasil/sku1_s42.csv --approx hasil/nsga3_sku1_s42_r1.csv
"""
import argparse
import csv

import numpy as np


def load(path):
    R = list(csv.DictReader(open(path)))
    return np.array([[float(r["POC"]), float(r["HC"]), -float(r["CSL"]), -float(r["ITO"])]
                     for r in R])


# absolute tolerances per objective (POC, HC, -CSL, -ITO), above the print precision of
# the CSV files (POC 4, HC 6, CSL/ITO 8 decimals): differences below these are rounding.
TOL = np.array([1e-4, 1e-6, 1e-8, 1e-8])


def dominated_by(A, B, chunk=512, strict=True):
    """mask over rows of A: True if some row of B dominates it (B <= A, B != A)."""
    out = np.zeros(len(A), dtype=bool)
    for s in range(0, len(A), chunk):
        a = A[s:s + chunk][:, None, :]
        le = np.all(B[None, :, :] <= a + TOL, axis=2)
        lt = np.any(B[None, :, :] < a - TOL, axis=2)
        out[s:s + chunk] = np.any(le & lt, axis=1) if strict else np.any(le, axis=1)
    return out


def on_front(A, P, chunk=512):
    out = np.zeros(len(A), dtype=bool)
    for s in range(0, len(A), chunk):
        a = A[s:s + chunk][:, None, :]
        eq = np.all(np.abs(P[None, :, :] - a) <= TOL, axis=2)
        out[s:s + chunk] = np.any(eq, axis=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exact", required=True)
    ap.add_argument("--approx", required=True)
    a = ap.parse_args()
    P = load(a.exact)
    A = load(a.approx)
    ideal, nadir = P.min(axis=0), P.max(axis=0)
    span = np.where(nadir - ideal > 0, nadir - ideal, 1.0)
    Pn, An = (P - ideal) / span, (A - ideal) / span

    exact_hits = on_front(A, P)
    dom_A = dominated_by(A, P)
    dom_P = dominated_by(P, A)
    # IGD+ : mean over exact points z of min_a || max(a - z, 0) ||
    igd = np.empty(len(Pn))
    for s in range(0, len(Pn), 256):
        z = Pn[s:s + 256][:, None, :]
        d = np.sqrt((np.maximum(An[None, :, :] - z, 0.0) ** 2).sum(axis=2))
        igd[s:s + 256] = d.min(axis=1)
    # eps+ : smallest eps such that every exact point is weakly dominated by some a - eps
    eps = np.empty(len(Pn))
    for s in range(0, len(Pn), 256):
        z = Pn[s:s + 256][:, None, :]
        eps[s:s + 256] = (An[None, :, :] - z).max(axis=2).min(axis=1)
    print(f"exact front: {len(P)} points | approximate set: {len(A)} points")
    print(f"approx points lying on exact front : {exact_hits.sum():6d} ({100*exact_hits.mean():.2f}%)")
    print(f"approx points dominated by exact    : {dom_A.sum():6d} ({100*dom_A.mean():.2f}%)")
    print(f"exact points dominated by approx    : {dom_P.sum():6d}  (must be 0)")
    print(f"exact points recovered              : {100*exact_hits.sum()/len(P):.3f}% of the exact front")
    print(f"IGD+ (normalised)                   : {igd.mean():.5f}")
    print(f"eps+ (normalised)                   : {eps.max():.5f}")


if __name__ == "__main__":
    main()
