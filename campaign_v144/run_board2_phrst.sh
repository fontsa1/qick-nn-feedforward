#!/usr/bin/env bash
# Four fresh v1.43 loads of board A, phrst_probe on each (one load per run; its own reload method needs a server notebook
# that deploy_and_verify does not use). Each run starts from a fresh load, so the probe's none mode comes first, before any reset.
set -u
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"; DG=analysis_scripts/diagnostics; D=campaign_v144/final/board2; TOP=C:/qick_projects/top; V43="Nov  7 19:57:07"   # set PYTHON to your environment's interpreter, e.g. PYTHON=.venv/bin/python
export QICK_NS_HOST=10.1.9.26 QICK_LIVE_N=576
rm -f $D/A_phrst_probe_v144.json
for L in 0 1 2 3; do
  r=$D/A_phrst_load$L
  timeout 900 $PY board_tools/deploy_and_verify.py --deploy $TOP/OG_build_v1.44.bit $TOP/OG_build_v1.44.hwh qick67 --expect-banner-contains "$V43" > $r.deploy.log 2>&1 < /dev/null
  if ! grep -q 'verification PASSED' $r.deploy.log; then
    timeout 900 $PY board_tools/start_server.py $QICK_NS_HOST --restart > $r.restart.log 2>&1 < /dev/null
    timeout 300 $PY board_tools/deploy_and_verify.py --expect-banner-contains "$V43" > $r.verify.log 2>&1 < /dev/null
    grep -q 'verification PASSED' $r.verify.log || { echo "BANNER FAILED load $L"; exit 3; }
  fi
  $PY -u $DG/phrst_probe.py --loads 1 --out $r.json > $r.log 2>&1 < /dev/null; echo "load $L rc=$?  $(grep -E '^ +none|^ +both' $r.log | cut -c1-70 | tr '\n' ';')"
done
$PY - <<'P'
import json
D = "campaign_v144/final/board2/"
A = [json.load(open(f"{D}A_phrst_load{i}.json")) for i in range(4)]
out = dict(A[0]); out["loads"] = [x["loads"][0] for x in A]
json.dump(out, open(D + "A_phrst_probe_v144.json", "w"), indent=1)
P
$PY board_tools/deploy_and_verify.py --expect-banner-contains "$V43" > $D/A_verify_after_phrst.log 2>&1 < /dev/null
timeout 900 $PY board_tools/deploy_and_verify.py --deploy $TOP/OG_build_v1.44.bit $TOP/OG_build_v1.44.hwh qick67 --expect-banner-contains "$V43" > $D/A_final_deploy.log 2>&1 < /dev/null
$PY tools/run_calibration.py > $D/A_calib_final.log 2>&1 < /dev/null; echo "final calibration rc=$? $(grep -E '\|k\|' $D/A_calib_final.log | tail -1 | cut -c1-90)"
