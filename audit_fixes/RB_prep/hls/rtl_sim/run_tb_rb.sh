#!/bin/bash
# tb_live (copied from neural_net4_live576s/rtl_sim) against an HLS syn/verilog dir. From hls/rtl_sim:
#   ./run_tb_rb.sh <syn/verilog dir> <tag>
set -e
HLSV=$1; TAG=$2
V=/c/embedded-software/Vivado/2023.1/bin
QBUF=../../sim/buffer.sv
rm -f *.dat; cp -f $HLSV/*.dat .
cmd.exe //c "$(cygpath -w $V/xvlog.bat)" --sv tb_live.sv ../../orig/live_window_gate.v $QBUF $HLSV/*.v > xvlog_$TAG.txt 2>&1 || { echo XVLOG FAILED; exit 1; }
cmd.exe //c "$(cygpath -w $V/xelab.bat)" -debug off -timescale 1ps/1ps tb_live -s tb_live_sim > xelab_$TAG.txt 2>&1 || { echo XELAB FAILED; exit 1; }
cmd.exe //c "$(cygpath -w $V/xsim.bat)" tb_live_sim -R > xsim_$TAG.txt 2>&1
sed -n '/=== tb_live ===/,/RESULT/p' xsim_$TAG.txt
