#!/usr/bin/env python3
"""make_commands.py - turn benchmark manifests into lsdp runs (resumable).

usage:  python3 bench/make_commands.py bench/instances/*/manifest.csv | bash
Each instance writes suite/E6/<source>/<series>_w<k>.{csv,log}; an instance whose log already
contains TOTAL is skipped, so an interrupted batch can simply be started again.
Every run is limited to --tlim seconds (default 1800); a run that hits the limit writes
TIMEOUT to its log, is counted as unsolved and is not retried.
With --check N, the first N instances (in a fixed random order) are also solved without the
pruning rules with the independent period-by-period programme over all classes
(--cover 0 --classdom 0 --lemmaD 0) into *_ref.csv, to verify exactness on real demand.
Reference runs have their own limit --ref-tlim (default 1800 s); a reference run that hits
it is reported as not completed and does not count as a check.
"""
import argparse
import csv
import os
import random

ap = argparse.ArgumentParser()
ap.add_argument("manifests", nargs="+")
ap.add_argument("--suite", default="suite")
ap.add_argument("--bin", default=os.environ.get("BIN", "./build/lsdp"))
ap.add_argument("--threads", default=os.environ.get("THREADS", "6"))
ap.add_argument("--check", type=int, default=30)
ap.add_argument("--tlim", type=int, default=int(os.environ.get("TLIM", "7200")))
ap.add_argument("--ref-tlim", type=int, default=int(os.environ.get("REF_TLIM", "1800")))
a = ap.parse_args()

rows = []
for m in a.manifests:
    rows += list(csv.DictReader(open(m)))
print("set -u")
for r in rows:
    d = os.path.join(a.suite, "E6", r["source"])
    base = os.path.join(d, f"{r['series']}_w{r['window']}")
    args = (f"--sku 1 --T {r['T']} --demand {r['demand_file']} --mu {r['mu']} --sigma {r['sigma']} "
            f"--p {r['p']} --r {r['r']} --moq {r['moq']} --L {r['L']} --c {r['c']} --MS {r['MS']} "
            f"--hfrac {r['hfrac']} --outer continuous --threads {a.threads}")
    print(f"mkdir -p {d}; grep -qE 'TOTAL|TIMEOUT' {base}.log 2>/dev/null || "
          f"{{ timeout {a.tlim} {a.bin} front {args} --out {base}.csv > {base}.log 2>&1; rc=$?; "
          f"[ $rc -eq 124 ] && echo 'TIMEOUT {a.tlim}s' >> {base}.log; "
          f"[ $rc -ne 0 ] && [ $rc -ne 124 ] && echo 'FAILED {base}'; true; }}")
random.seed(2026)
for r in random.sample(rows, min(a.check, len(rows))):
    base = os.path.join(a.suite, "E6", r["source"], f"{r['series']}_w{r['window']}")
    args = (f"--sku 1 --T {r['T']} --demand {r['demand_file']} --mu {r['mu']} --sigma {r['sigma']} "
            f"--p {r['p']} --r {r['r']} --moq {r['moq']} --L {r['L']} --c {r['c']} --MS {r['MS']} "
            f"--hfrac {r['hfrac']} --outer continuous --threads {a.threads} --cover 0 --classdom 0 --lemmaD 0")
    print(f"grep -qE 'TOTAL|TIMEOUT' {base}_ref.log 2>/dev/null || "
          f"{{ timeout {a.ref_tlim} {a.bin} front {args} --out {base}_ref.csv > {base}_ref.log 2>&1; rc=$?; "
          f"[ $rc -eq 124 ] && echo 'TIMEOUT {a.ref_tlim}s' >> {base}_ref.log; true; }}")
print("echo 'E6 runs finished'")
