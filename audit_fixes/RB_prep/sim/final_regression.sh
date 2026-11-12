#!/bin/sh
# Final regression on the deliverable sources (rtl/, orig/). Run from sim/: ./final_regression.sh > logs/final_regression.txt
f() { echo "=== $1"; shift; TAG=final ./run.sh "$@" | grep -i "RESULT\|tb_box_diff:\|mismatches\|latency\|OLD\|NEW\|phase\|records\|windows (buffer)\|== \|^A\|^B\|^C\|^D\|^E\|magic\|ERROR" ; }
md5sum ../rtl/*.v ../orig/*.v ../xdc/*.xdc
f box_diff   tb_box_diff tb_box_diff.sv ref/boxcar_tap_ref.v ../rtl/boxcar_tap.v
f boxcar     tb_boxcar tb_boxcar.sv ../rtl/boxcar_tap.v
f box_fault  tb_box_fault tb_box_fault.sv ref/boxcar_tap_ref.v ../rtl/boxcar_tap.v
f box_dump   tb_box_dump tb_box_dump.sv ../rtl/boxcar_tap.v
"${PYTHON:-python}" box_check.py | sed -n '1p;/total/p'
f gate_v141  tb_gate tb_gate_v141.sv ../rtl/dac_gate.v ../orig/live_res_cdc.v
f gate_fix   tb_gate_fix tb_gate_fix.sv ref/dac_gate_ref.v ../rtl/dac_gate.v
f gate_diff  tb_gate_diff tb_gate_diff.sv ref/dac_gate_ref.v ../rtl/dac_gate.v ../orig/live_res_cdc.v
f lwg        tb_lwg tb_lwg.sv ../orig/live_window_gate.v buffer.sv
f seg        tb_seg tb_seg.sv ../orig/live_window_gate.v ../orig/seg_stamp.v ../orig/trig_sync.v ../orig/rf_tap.v
echo FINAL_DONE
