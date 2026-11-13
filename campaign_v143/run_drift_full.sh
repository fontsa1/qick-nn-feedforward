#!/usr/bin/env bash
# 3-hour drift check: paired score of 6000 shots at start and end, on the start calibration. drift_probe.py reconnects
# after a dropped connection and scores TLAST windows with --no-align. Output in final/drift_full. Boards as loaded, no redeploy.
#   bash campaign_v143/run_drift_full.sh A|B
set -u
cd "$(dirname "$0")/.."
B=$1
case $B in A) H=10.1.9.26;; B) H=10.1.4.105;; *) echo "board A or B"; exit 2;; esac
PY="${PYTHON:-python}"; DG=analysis_scripts/diagnostics; D=campaign_v143/final   # set PYTHON to your environment's interpreter, e.g. PYTHON=.venv/bin/python
SILENCE=900
OUT=$D/drift_full
mkdir -p $OUT $D/logs
RLOG=$OUT/runner_$B.log
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
    bit_sha256=hashlib.sha256(open("C:/qick_projects/top/OG_build_v1.43.bit", "rb").read()).hexdigest(),
    git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()), indent=1))
EOF
  say "calibrated for $ph: $(grep -E 'k ?=|\|k\|' "$log" | tail -1 | cut -c1-100)"
}

calibrate drift_full || { say "RUNNER $B STOPPED: calibration failed"; exit 3; }
watch $OUT/${B}_drift.log $PY -u $DG/drift_probe.py --minutes 180 --score 6000 --out-dir $OUT; say "drift rc=$?"
say "RUNNER $B FINISHED"
