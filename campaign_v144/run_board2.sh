#!/usr/bin/env bash
# Second board session on v1.44: measurements outside the main campaign.
#   bash campaign_v144/run_board2.sh A|B|swap|phrst
# A, B: calibrate, tProc path (boxcar and FIR), boxcar vs raw buffer with a banner in its log.
# swap: board A and board B on board A's calibration, and on their own (run after A and B).
# phrst: phase across four bitstream loads on board A, then recalibrate (run last).
set -u
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"; DG=analysis_scripts/diagnostics; D=campaign_v144/final/board2; V43="Nov  7 19:57:07"   # set PYTHON to your environment's interpreter, e.g. PYTHON=.venv/bin/python
mkdir -p $D
export QICK_LIVE_N=576
say() { echo "$(date '+%F %T') $*" | tee -a $D/runner.log; }
try() { # try <log> <cmd...>: up to 4 tries, for Wi-Fi drops
  local log=$1 t; shift
  for t in 1 2 3 4; do "$@" >> "$log" 2>&1 < /dev/null && return 0; say "try $t failed: $log"; sleep 60; done
  say "GAVE UP: $log"; return 1
}
calibrate() { # board host
  until mkdir campaign_v144/final/calib/.lock 2>/dev/null; do sleep 10; done
  try $D/$1_calib.log $PY tools/run_calibration.py; rmdir campaign_v144/final/calib/.lock
  say "$1 calibrated: $(grep -E '\|k\|' $D/$1_calib.log | tail -1 | cut -c1-100)"
}
board() {
  local B=$1 H=$2
  export QICK_NS_HOST=$H
  $PY board_tools/deploy_and_verify.py --expect-banner-contains "$V43" > $D/${B}_verify.log 2>&1 < /dev/null || { say "$B banner check FAILED"; exit 3; }
  calibrate $B
  try $D/${B}_tproc_box.log $PY -u $DG/tproc_path_v143.py --path box --variants nobranch,qick --margins 0.025,0.03,0.035,0.04,0.045,0.05,0.055,0.06 --shots 100 --n-force 25 --out $D/${B}_tproc_box_v144.json
  say "$B tproc box: $(grep 'earliest' $D/${B}_tproc_box.log | tr '\n' ';')"
  try $D/${B}_tproc_fir.log $PY -u $DG/tproc_path_v143.py --path fir --variants nobranch --margins 0.025,0.03,0.035,0.04,0.045,0.05 --shots 40 --n-force 20 --out $D/${B}_tproc_fir_v144.json
  say "$B tproc fir: $(grep 'earliest' $D/${B}_tproc_fir.log | tr '\n' ';')"
  $PY board_tools/deploy_and_verify.py --expect-banner-contains "$V43" > $D/${B}_boxcar_mr_v144.log 2>&1 < /dev/null
  try $D/${B}_boxcar_mr_v144.log $PY -u $DG/boxcar_mr_check.py --out $D/${B}_boxcar_mr_v144.json
  say "$B boxcar_mr: $(grep -iE 'verdict|PASS|FAIL' $D/${B}_boxcar_mr_v144.log | tail -1 | cut -c1-120)"
  say "$B done"
}
score() { # host tag: 2,000 rows that are not domain-map rows, as the NN_axi campaign runs
  QICK_NS_HOST=$1 try $D/swap_$2.log $PY -u $DG/score_armed_fidelity.py --start 2000 --shots 2000 --chunk 2000 --norm dataset --no-align --live --live-n 576 --live-capture --save $D/swap_$2_v144.npz
  say "swap $2: $(grep 'FIDELITY (live)' $D/swap_$2.log | tail -1 | cut -c1-90)"
}
case $1 in
  A) board A 10.1.9.26;;
  B) board B 10.1.4.105;;
  swap)
    A=10.1.9.26; B=10.1.4.105
    score $A A_own; score $B B_own
    cp calib/k_$B.npy $D/k_B_saved.npy; cp calib/c_$B.npy $D/c_B_saved.npy
    trap 'cp '$D'/k_B_saved.npy calib/k_'$B'.npy; cp '$D'/c_B_saved.npy calib/c_'$B'.npy' EXIT
    cp calib/k_$A.npy calib/k_$B.npy; cp calib/c_$A.npy calib/c_$B.npy
    score $B B_on_A_calibration
    say "swap done; board B calibration restored at exit";;
  phrst)
    export QICK_NS_HOST=10.1.9.26
    try $D/A_phrst.log $PY -u $DG/phrst_probe.py --loads 4 --out $D/A_phrst_probe_v144.json
    say "phrst: $(tail -1 $D/A_phrst.log | cut -c1-120)"
    $PY board_tools/deploy_and_verify.py --expect-banner-contains "$V43" > $D/A_verify_after_phrst.log 2>&1 < /dev/null
    calibrate A;;
esac
