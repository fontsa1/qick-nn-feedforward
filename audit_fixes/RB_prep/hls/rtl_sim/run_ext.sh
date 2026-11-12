#!/bin/bash
# ./run_ext.sh <syn/verilog dir> <tag> [EXPECT_WRAP]
HLSV=$1; TAG=$2; DEF=${3:+-d $3}
V=/c/embedded-software/Vivado/2023.1/bin
rm -f *.dat; cp -f $HLSV/*.dat .
cmd.exe //c "$(cygpath -w $V/xvlog.bat)" --sv $DEF tb_live_ext.sv $HLSV/*.v > xvlog_ext_$TAG.txt 2>&1 || { echo XVLOG FAILED; tail xvlog_ext_$TAG.txt; exit 1; }
cmd.exe //c "$(cygpath -w $V/xelab.bat)" -debug off -timescale 1ps/1ps tb_live_ext -s tb_live_ext_sim > xelab_ext_$TAG.txt 2>&1 || { echo XELAB FAILED; exit 1; }
cmd.exe //c "$(cygpath -w $V/xsim.bat)" tb_live_ext_sim -R > xsim_ext_$TAG.txt 2>&1
grep "tb_live_ext\|^logit" xsim_ext_$TAG.txt
