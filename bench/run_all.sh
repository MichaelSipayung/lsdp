#!/usr/bin/env bash
# run_all.sh - regenerate every result of the paper with the current lsdp, plus independent
# reference runs of the period-by-period programme without any pruning, and the comparisons.
# usage (repository root, virtualenv active):
#     BIN=./build/lsdp THREADS=6 systemd-inhibit --what=sleep:idle bash bench/run_all.sh
# Each step is skipped when its output already exists, so the script can be restarted.
set -uo pipefail
export BIN=${BIN:-./build/lsdp} THREADS=${THREADS:-6}
REF="--cover 0 --classdom 0 --lemmaD 0"
log() { echo "[$(date +%H:%M:%S)] $*"; }

log "E1"; [ -f suite/E1/verify.log ] || bash run_suite.sh E1
log "E2"; [ -f suite/E2/sku2_s10_continuous.log ] || bash run_suite.sh E2
log "E4"; [ -f suite/E4/c800_s5.log ] || bash run_suite.sh E4
log "E3"; [ -f suite/E3/full_sku1_T96_s1.log ] || bash run_suite.sh E3
log "E5a (DP part)"
for sku in 1 2; do for T in 16 24 32 40 48; do f=suite/E5/dpnoito_sku${sku}_T${T}
  $BIN front --sku $sku --T $T --seed 1 --SS 8 --SOT 0.3 --noito 1 --threads 1 --out $f.csv > $f.log; done; done
log "E6"; TLIM=7200 bash bench/run_e6.sh

log "reference runs (period-by-period programme, all classes, no pruning)"
mkdir -p suite/ref/E2 suite/ref/E4
for sku in 1 2; do for s in $(seq 1 10); do for o in grid continuous; do
  f=suite/ref/E2/sku${sku}_s${s}_${o}
  grep -q TOTAL $f.log 2>/dev/null || $BIN front --sku $sku --T 48 --seed $s --outer $o --threads $THREADS $REF --out $f.csv > $f.log
done; done; done
for s in $(seq 1 5); do
  for a in "cv 0.1" "cv 0.25" "cv 0.5" "r 35" "r 70" "r 140" "c 50" "c 200" "c 800"; do
    set -- $a; f=suite/ref/E4/$1$2_s$s
    grep -q TOTAL $f.log 2>/dev/null || $BIN front --sku 1 --T 48 --seed $s --outer continuous --$1 $2 --threads $THREADS $REF --out $f.csv > $f.log
  done
done

log "E5b comparison against the regenerated exact fronts"
for rep in 0 1; do for sku in 1 2; do for s in 1 2 3; do for ref in continuous grid; do
  echo "== repair=$rep SKU$sku seed$s vs exact $ref"
  python3 compare.py --exact suite/E2/sku${sku}_s${s}_${ref}.csv --approx suite/E5/nsga_rep${rep}_sku${sku}_s${s}.csv
done; done; done; done > suite/E5/E5b_compare.txt
grep -c "(must be 0)" suite/E5/E5b_compare.txt > /dev/null && \
  echo "E5b: exact points dominated by NSGA-III (sum, must be 0): $(grep 'must be 0' suite/E5/E5b_compare.txt | awk '{s+=$7} END{print s}')"

log "exactness: current programme vs independent reference"
python3 bench/compare_runs.py suite/ref/E2 suite/E2
python3 bench/compare_runs.py suite/ref/E4 suite/E4
python3 - <<'EOF'
import glob, os, shutil, subprocess
shutil.rmtree('/tmp/e6cmp', ignore_errors=True)
for ref in glob.glob("suite/E6/*/*_ref.csv"):
    base = ref[:-8] + ".csv"
    d = os.path.dirname(ref)
    os.makedirs("/tmp/e6cmp/a", exist_ok=True); os.makedirs("/tmp/e6cmp/b", exist_ok=True)
    name = os.path.basename(base)
    os.system(f"cp {ref} /tmp/e6cmp/a/{name}; cp {base} /tmp/e6cmp/b/{name}")
out = subprocess.run(["python3", "bench/compare_runs.py", "/tmp/e6cmp/a", "/tmp/e6cmp/b"],
                     capture_output=True, text=True).stdout
print(out.strip().replace("/tmp/e6cmp/b vs /tmp/e6cmp/a", "suite/E6 reference checks"))
EOF
log "done"
