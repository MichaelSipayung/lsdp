#!/usr/bin/env bash
# run_suite.sh — full experimental protocol for the exact multiobjective DP
# (Kania et al. CIE 2022 lot sizing model). Run from ~/riset/lsdp.
#   bash run_suite.sh E1   # correctness (minutes)
#   bash run_suite.sh E2   # main study T=48, 2 SKUs x 10 seeds x {grid, continuous}
#   bash run_suite.sh E3   # scaling in T (one inner DP per point) + full front at T=96
#   bash run_suite.sh E4   # sensitivity: demand CV, lot size r, ordering cost c
#   bash run_suite.sh E3x  # optional, heavy (hours): full SKU2 front at T=96
#   bash run_suite.sh E5a  # MILP baseline vs DP (needs scipy; up to ~3 h)
#   bash run_suite.sh E5b  # NSGA-III with/without repair (needs pymoo; ~2 h)
#   bash run_suite.sh all  # E1-E4
set -euo pipefail
cd "$(dirname "$0")"
THREADS=${THREADS:-12}
BIN=${BIN:-./build/release/lsdp}
OUT=suite
mkdir -p $OUT/E1 $OUT/E2 $OUT/E3 $OUT/E4

env_info() {
  { echo "date: $(date -Is)"; uname -a; g++ --version | head -1; nproc; free -h | head -2;
    lscpu | grep -E "Model name|^CPU\(s\)|Thread|MHz" || true; } > $OUT/environment.txt
}

E1() {
  $BIN verify --seeds 50 | tee $OUT/E1/verify.log | tail -1
  for sku in 1 2; do for s in $(seq 1 10); do
    $BIN verify-cont --sku $sku --seed $s --samples 1000000
  done; done | tee $OUT/E1/verify_cont.log | grep -c OK
}

E2() {
  for sku in 1 2; do for s in $(seq 1 10); do for outer in grid continuous; do
    f=$OUT/E2/sku${sku}_s${s}_${outer}
    $BIN front --sku $sku --T 48 --seed $s --outer $outer --threads $THREADS --out $f.csv > $f.log
    echo "E2 sku$sku seed$s $outer: $(grep TOTAL $f.log)"
  done; done; done
}

E3() {
  for sku in 1 2; do for T in 12 24 36 48 60 72 84 96; do for s in 1 2 3; do
    f=$OUT/E3/one_sku${sku}_T${T}_s${s}
    $BIN front --sku $sku --T $T --seed $s --dSS 1000 --dSOT 5 --threads 1 --out $f.csv > $f.log
    echo "E3 single-DP sku$sku T=$T seed$s: $(grep TOTAL $f.log)"
  done; done; done
  # full continuous front at T=96 for SKU1 (about 0.4 GB per thread -> 6 threads)
  f=$OUT/E3/full_sku1_T96_s1
  $BIN front --sku 1 --T 96 --seed 1 --outer continuous --threads 6 --out $f.csv > $f.log
  echo "E3 full sku1 T=96: $(grep TOTAL $f.log)"
}

# optional, heavy: SKU2 at T=96 needs about 1 GB per thread and several hours
E3x() {
  f=$OUT/E3/full_sku2_T96_s1
  $BIN front --sku 2 --T 96 --seed 1 --outer continuous --threads 6 --out $f.csv > $f.log
  echo "E3x full sku2 T=96: $(grep TOTAL $f.log)"
}

E4() {
  for s in $(seq 1 5); do
    for cv in 0.1 0.25 0.5; do
      f=$OUT/E4/cv${cv}_s${s}
      $BIN front --sku 1 --T 48 --seed $s --outer continuous --cv $cv --threads $THREADS --out $f.csv > $f.log
    done
    for r in 35 70 140; do
      f=$OUT/E4/r${r}_s${s}
      $BIN front --sku 1 --T 48 --seed $s --outer continuous --r $r --threads $THREADS --out $f.csv > $f.log
    done
    for c in 50 200 800; do
      f=$OUT/E4/c${c}_s${s}
      $BIN front --sku 1 --T 48 --seed $s --outer continuous --c $c --threads $THREADS --out $f.csv > $f.log
    done
    echo "E4 seed $s done"
  done
}

# E5a: exact generic baseline (epsilon-constraint MILP, HiGHS) vs DP on the bi-objective
#      (POC, HC) inner problem, one outer point (SS=8, SOT=0.3), growing horizon T.
#      Requires: pip install scipy  (inside the venv). Runs in parallel, 3 h cap per run.
E5a() {
  mkdir -p $OUT/E5
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  for sku in 1 2; do for T in 16 24 32 40 48; do
    f=$OUT/E5/dpnoito_sku${sku}_T${T}
    $BIN front --sku $sku --T $T --seed 1 --SS 8 --SOT 0.3 --noito 1 --threads 1 --out $f.csv > $f.log
    ( timeout 3h python3 milp_baseline.py --sku $sku --demand $f.csv.demand.csv --SS 8 --SOT 0.3 \
        --tlim 1800 --out $OUT/E5/milp_sku${sku}_T${T}.csv > $OUT/E5/milp_sku${sku}_T${T}.log 2>&1 \
      || echo "TIMEOUT (3h)" >> $OUT/E5/milp_sku${sku}_T${T}.log ) &
  done; done
  wait
  for sku in 1 2; do for T in 16 24 32 40 48; do
    echo "SKU$sku T=$T | DP: $(grep TOTAL $OUT/E5/dpnoito_sku${sku}_T${T}.log | sed 's/.*TOTAL //') | MILP: $(grep -h 'wall\|TIMEOUT' $OUT/E5/milp_sku${sku}_T${T}.log | tail -1)"
    python3 -c "
import csv,sys
try:
    A=sorted((round(float(r['POC']),3),round(float(r['HC']),3)) for r in csv.DictReader(open('$OUT/E5/dpnoito_sku${sku}_T${T}.csv')))
    B=sorted((round(float(r['POC']),3),round(float(r['HC']),3)) for r in csv.DictReader(open('$OUT/E5/milp_sku${sku}_T${T}.csv')))
    print('   fronts identical:',A==B,len(A),len(B))
except FileNotFoundError: print('   MILP did not finish')"
  done; done | tee $OUT/E5/E5a_summary.txt
}

# E5b: NSGA-III without / with problem-specific repair, 1 h each, on the E2 instances
#      (seeds 1-3); compared with the exact continuous and grid fronts. 6 runs in parallel.
E5b() {
  mkdir -p $OUT/E5
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  for rep in 0 1; do
    for sku in 1 2; do for s in 1 2 3; do
      python3 nsga3_baseline.py --sku $sku --demand $OUT/E2/sku${sku}_s${s}_grid.csv.demand.csv \
        --seed $s --time 01:00:00 --repair $rep --out $OUT/E5/nsga_rep${rep}_sku${sku}_s${s}.csv \
        > $OUT/E5/nsga_rep${rep}_sku${sku}_s${s}.log 2>&1 &
    done; done
    wait
  done
  for rep in 0 1; do for sku in 1 2; do for s in 1 2 3; do
    for ref in continuous grid; do
      echo "== repair=$rep SKU$sku seed$s vs exact $ref"
      python3 compare.py --exact $OUT/E2/sku${sku}_s${s}_${ref}.csv --approx $OUT/E5/nsga_rep${rep}_sku${sku}_s${s}.csv
    done
  done; done; done | tee $OUT/E5/E5b_compare.txt
}

env_info
case "${1:-all}" in
  E1) E1 ;; E2) E2 ;; E3) E3 ;; E3x) E3x ;; E4) E4 ;; E5a) E5a ;; E5b) E5b ;;
  all) E1; E2; E3; E4 ;;
  *) echo "usage: bash run_suite.sh [E1|E2|E3|E3x|E4|E5a|E5b|all]"; exit 2 ;;
esac
python3 summarize.py $OUT > $OUT/summary.txt && echo "summary written to $OUT/summary.txt"
