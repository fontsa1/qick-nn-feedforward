#!/usr/bin/env bash
# Generates the v1.44 runners from the v1.43 ones: new banner, bitstream, name on the board, output folder and file tags.
# run_board2.sh calibrations take run_6_3.sh's lock, since both boards run at once. run_6_3.sh's short drift check is
# removed; run_drift_full.sh does the drift run.
cd "$(dirname "$0")/.."
gen() { # gen <source> <dest name>
  sed -e 's#campaign_v143#campaign_v144#g' -e 's#Nov  4 23:39:37#Nov  7 19:57:07#g' -e 's#OG_build_v1\.43#OG_build_v1.44#g' \
      -e 's#qick66#qick67#g' -e 's#_v143\.#_v144.#g' -e 's#tproc_path_v144\.py#tproc_path_v143.py#g' \
      -e 's#mkdir \$D/\.lock#mkdir campaign_v144/final/calib/.lock#' -e 's#rmdir \$D/\.lock#rmdir campaign_v144/final/calib/.lock#' \
      -e '/^calibrate drift$/d' -e '/drift_probe.py --minutes 180 --score 500/d' "$1" > "campaign_v144/$2"
}
gen campaign_v143/run_6_3.sh run_6_3.sh
gen campaign_v143/run_drift_full.sh run_drift_full.sh
gen campaign_v143/run_board2.sh run_board2.sh
gen campaign_v143/run_board2_phrst.sh run_board2_phrst.sh
gen campaign_v140L1/latency_protocol/run_loads.sh run_loads.sh
# v1.44's boxcar path is 113.3 ns (v1.43: 115.1): centre the latency closure check on 1.4 ns
sed -i 's#latency_protocol.py --label#latency_protocol.py --expect-path-adv 1.4 --label#' campaign_v144/run_loads.sh
