#!/usr/bin/env bash
# Load-to-load latency study, v1.43 (2026-11-05). Small test: 100 + 40 shots per load, plus sensitivity on each
# board's first load. usage: run_loads.sh <outdir> <label> [<label> ...]; boards alternate so load is not tied to time.
cd "${REPO_ROOT:-.}"
PY="${PYTHON:-python}"; D=analysis_scripts/diagnostics; TOP=C:/qick_projects/top; V43="Nov  4 23:39:37"   # set PYTHON to your environment's interpreter, e.g. PYTHON=.venv/bin/python
O=$1; shift; mkdir -p $O
export QICK_LIVE_N=576
one() { # one <board> <host> <label> [--sensitivity]
  n=$1; export QICK_NS_HOST=$2; r=$O/${1}_$3; s=$4
  echo "START $n $3 $(date +%T)"
  timeout 900 $PY board_tools/deploy_and_verify.py --deploy $TOP/OG_build_v1.43.bit $TOP/OG_build_v1.43.hwh qick66 --expect-banner-contains "$V43" > $r.deploy.log 2>&1 < /dev/null
  if ! grep -q 'verification PASSED' $r.deploy.log; then
    timeout 900 $PY board_tools/start_server.py $2 --restart > $r.restart.log 2>&1 < /dev/null
    timeout 300 $PY board_tools/deploy_and_verify.py --expect-banner-contains "$V43" > $r.verify.log 2>&1 < /dev/null
    grep -q 'verification PASSED' $r.verify.log || { echo "BANNER_FAILED $n $3"; return 1; }
  fi
  timeout 900 $PY tools/run_calibration.py > $r.calib.log 2>&1 < /dev/null; echo "  calib rc=$?"
  timeout 600 $PY $D/box_probe.py --out $r.box_probe.json > $r.box_probe.log 2>&1 < /dev/null; echo "  box rc=$?"
  timeout 1500 $PY $D/latency_protocol.py --label $3 --box-json $r.box_probe.json --expect-banner "$V43" --bit $TOP/OG_build_v1.43.bit --sens-shots 40 $s --out $r.npz > $r.log 2>&1 < /dev/null
  echo "  protocol rc=$?  $(grep -E '^ +op:' $r.log)"
  grep -E '^  FAIL' $r.log
  echo "DONE $n $3 $(date +%T)"
}
first=--sensitivity
for l in "$@"; do
  one A 10.1.9.26 $l $first
  one B 10.1.4.105 $l $first
  first=
done
$PY $D/latency_protocol_summary.py $O --md $O/summary.md
echo LOADS_DONE
