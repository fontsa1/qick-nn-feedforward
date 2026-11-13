#!/usr/bin/env bash
# Full v1.44 data campaign (bitstream built from QICK's project plus the delta Tcl). Every output goes under
# campaign_v144/final. Unattended.
#   bash campaign_v144/run_all.sh     (start with launch_v144.ps1). Rerunning resumes finished chunks.
# Both boards in parallel: load, calibrate, small tests, gate sweep, main campaign, 3 h drift, tProc path, cal repeats.
# Then one at a time: calibration swap, 5 reloads per board for the latency groups, phase across loads.
# NOT here: the latency study's 2 cold starts per board. They need a power cycle by hand; see README.md.
set -u
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"; DG=analysis_scripts/diagnostics; D=campaign_v144/final; C=campaign_v144; T=$D/tests   # set PYTHON to your environment's interpreter, e.g. PYTHON=.venv/bin/python
TOP=C:/qick_projects/top; BANNER="Nov  7 19:57:07"; ADV=0.1135
export QICK_LIVE_N=576 QICK_BANNER="$BANNER" QICK_FIG7_BANNER="$BANNER" QICK_FIG7_DIR=$D/fig7 QICK_TAG=v144
mkdir -p $D/calib $T $D/fig7 $D/fig8b $D/leak_energy $D/cal_repeat $D/latency/reloads $D/latency/cold
rmdir $D/calib/.lock $D/.load_lock 2>/dev/null
say() { echo "$(date '+%F %T') $*" | tee -a $D/run_all.log; }

load() { # load <board> <host>: deploy, restart the server if the deploy could not, check the banner
  local r=$T/$1_load
  until mkdir $D/.load_lock 2>/dev/null; do sleep 10; done
  timeout 900 $PY board_tools/deploy_and_verify.py --deploy $TOP/OG_build_v1.44.bit $TOP/OG_build_v1.44.hwh qick67 --expect-banner-contains "$BANNER" > $r.deploy.log 2>&1 < /dev/null
  if ! grep -q 'verification PASSED' $r.deploy.log; then
    timeout 900 $PY board_tools/start_server.py $2 --restart > $r.restart.log 2>&1 < /dev/null
    timeout 300 $PY board_tools/deploy_and_verify.py --expect-banner-contains "$BANNER" > $r.verify.log 2>&1 < /dev/null
    grep -q 'verification PASSED' $r.verify.log || { rmdir $D/.load_lock; return 1; }
  fi
  rmdir $D/.load_lock
}

board() { # board <A|B> <host>
  local B=$1 H=$2 t
  export QICK_NS_HOST=$H
  step() { local n=$1 to=$2; shift 2; timeout $to "$@" > $T/${B}_$n.log 2>&1 < /dev/null; say "$B $n rc=$?"; }
  load $B $H || { say "$B BANNER FAILED, board stopped"; return 3; }
  say "$B v1.44 banner confirmed"
  until mkdir $D/calib/.lock 2>/dev/null; do sleep 10; done
  timeout 1200 $PY tools/run_calibration.py > $T/${B}_calib.log 2>&1 < /dev/null; say "$B calibration rc=$?"
  rmdir $D/calib/.lock
  step functional 1800 $PY $DG/qick_functional_test.py --out $T/${B}_functional.json
  step binding_fault 2400 $PY $DG/binding_fault_check.py --cycles 25 --control 40 --out $T/${B}_binding_fault.json
  step fail_safe 1800 $PY $DG/fail_safe_check.py --nn-advance $ADV --out $T/${B}_fail_safe.json
  step tproc_decision 2400 $PY $DG/tproc_decision_check.py --shots 400 --out $T/${B}_tproc_decision.json
  for t in 2000 8; do
    timeout 3000 $PY $DG/leak_energy_check.py --shots 40 --nz-thr $t --out $D/leak_energy/${B}_v144_leak_thr$t.json > $D/leak_energy/${B}_v144_leak_thr$t.log 2>&1 < /dev/null
    say "$B leak thr$t rc=$?"
  done
  timeout 3000 $PY $DG/gate_probe.py --split --nn-advance $ADV --pre 0.15 --slacks=-0.072,-0.070,-0.068,-0.066,-0.064,-0.062,-0.060,-0.040,0.0 --out $D/fig8b/${B}_gate_probe_box.npz > $D/fig8b/${B}_gate_probe.log 2>&1 < /dev/null
  say "$B gate sweep rc=$?"
  bash $C/run_6_3.sh $B
  bash $C/run_drift_full.sh $B
  bash $C/run_board2.sh $B
  timeout 3000 $PY -u $DG/cal_repeat.py --out-dir $D/cal_repeat > $D/cal_repeat/run_$H.txt 2>&1 < /dev/null; say "$B cal_repeat rc=$?"
  say "$B BOARD DONE"
}

say "START v1.44 reacquisition, git $(git rev-parse --short HEAD)"
board A 10.1.9.26 &
board B 10.1.4.105 &
wait

if [ "$(grep -c 'BOARD DONE' $D/run_all.log)" -eq 2 ]; then
  bash $C/run_board2.sh swap
  bash $C/run_loads.sh $D/latency/reloads r1 r2 r3 r4 r5 > $D/latency/reloads.log 2>&1 < /dev/null; say "reloads rc=$?"
  bash $C/run_board2_phrst.sh > $D/board2/phrst_runner.log 2>&1 < /dev/null; say "phrst rc=$?"
  say "ALL DONE (cold starts still to do by hand)"
else
  say "STOPPED: a board did not finish; shared steps skipped"
fi
touch $D/RUN_ALL_FINISHED
