#!/usr/bin/env bash
# Main data campaign on v1.44, one board per process, boards as loaded (no redeploy).
#   bash campaign_v144/run_6_3.sh A|B
# Order: calibrate + 20-shot in-loop smoke (stops the board's run unless 20/20 on the v1.44 banner), in-loop on the full
# split, NN_live run 1, NN_live run 2 (halves swapped), NN_axi TLAST on the 20k set, Fig. 7 capture and sweep.
# Recalibrated before each phase. Rerunning resumes: finished chunks are skipped, the in-loop run uses its checkpoint.
# Watchdog: a log silent for 15 min has its python process tree killed, then a retry.
set -u
cd "$(dirname "$0")/.."
B=$1
case $B in A) H=10.1.9.26; R1=0; R2=50000;; B) H=10.1.4.105; R1=50000; R2=0;; *) echo "board A or B"; exit 2;; esac
PY="${PYTHON:-python}"; DG=analysis_scripts/diagnostics; D=campaign_v144/final   # set PYTHON to your environment's interpreter, e.g. PYTHON=.venv/bin/python
BANNER="Nov  7 19:57:07"; SILENCE=900
mkdir -p $D/logs $D/calib $D/live $D/inloop $D/axi $D/fig7 $D/drift
RLOG=$D/runner_$B.log
export QICK_NS_HOST=$H QICK_LIVE_N=576
say() { echo "$(date '+%F %T') $*" | tee -a "$RLOG"; }

# watch <log> <cmd...>: run with the silence watchdog, append to the log, return the command's rc
watch() {
  local log=$1 pid winpid age; shift
  "$@" >> "$log" 2>&1 < /dev/null &
  pid=$!; winpid=$(cat /proc/$pid/winpid 2>/dev/null || echo "")
  touch "$log"
  while kill -0 $pid 2>/dev/null; do
    sleep 20
    age=$(( $(date +%s) - $(stat -c %Y "$log") ))
    if [ $age -gt $SILENCE ]; then
      say "WATCHDOG: $log silent ${age}s, killing pid $pid (win $winpid)"
      [ -n "$winpid" ] && taskkill //PID "$winpid" //T //F >> "$RLOG" 2>&1
      kill -9 $pid 2>/dev/null
      break
    fi
  done
  wait $pid
}

# calibrate <phase>: one board at a time (both write the root k/c); the used pair and its provenance are stored
calibrate() {
  local ph=$1 log
  log=$D/logs/${B}_calib_$ph.log
  [ -f $D/calib/${B}_${ph}.json ] && { say "calibration $ph already done"; return 0; }
  until mkdir $D/calib/.lock 2>/dev/null; do sleep 10; done
  watch "$log" $PY tools/run_calibration.py; local rc=$?
  rmdir $D/calib/.lock
  [ $rc -eq 0 ] || { say "calibration $ph FAILED rc=$rc"; return 1; }
  cp calib/k_$H.npy $D/calib/${B}_${ph}_k.npy; cp calib/c_$H.npy $D/calib/${B}_${ph}_c.npy
  $PY - "$B" "$H" "$ph" > $D/calib/${B}_${ph}.json <<'EOF'
import datetime, hashlib, json, subprocess, sys
print(json.dumps(dict(board=sys.argv[1], host=sys.argv[2], phase=sys.argv[3], time=datetime.datetime.now().isoformat(),
    bit_sha256=hashlib.sha256(open("C:/qick_projects/top/OG_build_v1.44.bit", "rb").read()).hexdigest(),
    git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()), indent=1))
EOF
  say "calibrated for $ph: $(grep -E 'k ?=|\|k\|' "$log" | tail -1 | cut -c1-100)"
}

# ---- phase 0
calibrate inloop || exit 3
SM=$D/inloop/${B}_SMOKE
rm -f $SM.npz $SM.log
watch $SM.log $PY -u audit_fixes/R18/inloop_campaign.py --start 20000 --shots 20 --out $SM.npz
if ! grep -q "$BANNER" $SM.log || ! grep -q "(correct): 20/20" $SM.log; then
  say "SMOKE FAILED, stopping this board: $(tail -2 $SM.log | cut -c1-200)"; exit 3
fi
say "smoke OK: $(grep '^pulse == gate' $SM.log | cut -c1-150)"

# ---- in-loop, whole split
OUT=$D/inloop/${B}_inloop_0_100000.npz; LOG=$D/inloop/${B}_inloop_0_100000.log
for try in 1 2 3 4 5 6; do
  grep -q "^pulse == gate == NN_live decision" "$LOG" 2>/dev/null && break
  res=""; [ -f "$OUT" ] && res="--resume"
  say "in-loop attempt $try $res"
  watch "$LOG" $PY -u audit_fixes/R18/inloop_campaign.py --start 0 --shots 100000 --out "$OUT" $res
  say "in-loop attempt $try rc=$?"
done
grep -q "^pulse == gate == NN_live decision" "$LOG" 2>/dev/null && say "in-loop DONE: $(grep '^pulse == gate' "$LOG" | tail -1 | cut -c1-170)" || say "in-loop NOT FINISHED"

# ---- NN_live, gate idle, 1000-shot chunks. run1 = this board's half, run2 = the other half
live() {
  local run=$1 first=$2 s e out log try
  calibrate live$run || return 1
  for ((s = first; s < first + 50000; s += 1000)); do
    e=$((s + 1000)); out=$D/live/${B}_run${run}_${s}_${e}.npz; log=$D/logs/${B}_live_run${run}.log
    [ -f "$out" ] && continue
    for try in 1 2 3; do
      watch "$log" $PY -u $DG/box_live_capture.py --x data/test100k_real_X.npy --y data/test100k_real_y.npy \
        --scale-ref data/test20k_real_X.npy --clip-dac --start $s --shots 1000 --out "$out"
      [ $? -eq 0 ] && [ -f "$out" ] && break
      say "live run$run chunk $s try $try failed"; sleep 30
    done
    [ -f "$out" ] || say "live run$run chunk $s MISSING"
    [ $(( (s - first) % 5000 )) -eq 0 ] && say "live run$run at row $s"
  done
  say "live run$run done"
}
live 1 $R1
live 2 $R2

# ---- NN_axi with TLAST, no host alignment, 20k set on both boards
calibrate axi
for ((s = 0; s < 20000; s += 2000)); do
  out=$D/axi/${B}_axi_$s.npz; [ -f "$out" ] && continue
  for try in 1 2 3; do
    watch $D/axi/${B}_axi_$s.log $PY -u $DG/score_armed_fidelity.py --start $s --shots 2000 --chunk 2000 --norm dataset \
      --no-align --live --live-n 576 --live-capture --save "$out"
    [ -f "$out" ] && break
    say "axi $s try $try failed"; sleep 30
  done
  [ -f "$out" ] || say "axi $s MISSING"
done
say "axi done"

# ---- Fig. 7, then the drift check
watch $D/logs/${B}_fig7_capture.log $PY -u $DG/box_probe_fig7.py --board $B; say "fig7 capture rc=$?"
watch $D/logs/${B}_fig7_sweep.log $PY -u $DG/box_sweep_fig7.py --board $B; say "fig7 sweep rc=$?"
say "RUNNER $B FINISHED"
