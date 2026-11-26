#!/usr/bin/env bash
# run_e6.sh - E6: exact fronts on public demand data (M5 and UCI Online Retail II).
# usage (repository root, virtualenv active, after bench/prepare.py):
#     BIN=./build/lsdp THREADS=6 TLIM=7200 systemd-inhibit --what=sleep:idle bash bench/run_e6.sh
# Resumable: finished instances (log contains TOTAL) are skipped, so the script can be
# stopped (Ctrl-C, sleep, power loss) and started again without losing results.
set -euo pipefail
BIN=${BIN:-./build/lsdp}
THREADS=${THREADS:-6}
OUT=${OUT:-suite}
CHECK=${CHECK:-30}
TLIM=${TLIM:-7200}
ls bench/instances/*/manifest.csv > /dev/null 2>&1 || { echo "run bench/prepare.py first"; exit 1; }
mkdir -p "$OUT/E6"
{ echo "date: $(date -Is)"; "$BIN" 2>&1 | head -1 || true; } > "$OUT/E6/environment.txt"
# progress line every 10 instances, so that a long run is visibly alive
python3 bench/make_commands.py bench/instances/*/manifest.csv --suite "$OUT" --bin "$BIN" \
  --threads "$THREADS" --check "$CHECK" --tlim "$TLIM" | bash 2>&1 | tee -a "$OUT/E6/run.log" | grep -E "FAILED|finished" || true
echo "solved: $(grep -l TOTAL $OUT/E6/*/*.log | grep -vc _ref) instances, \
time limit reached: $(grep -l TIMEOUT $OUT/E6/*/*.log | grep -vc _ref), \
reference checks: $(grep -l TOTAL $OUT/E6/*/*_ref.log 2>/dev/null | wc -l) solved / \
$(grep -l TIMEOUT $OUT/E6/*/*_ref.log 2>/dev/null | wc -l) at limit"
