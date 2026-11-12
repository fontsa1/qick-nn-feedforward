#!/bin/sh
# usage: run.sh <top> [-d DEFINE=..]... <files...>   (run from audit_fixes/RB_prep/sim)
# xvlog -> xelab -> xsim, logs in logs/<top>[_<tag>]_{xvlog,xelab,xsim}.txt ; TAG env var optional
V=/c/embedded-software/Vivado/2023.1/bin
top=$1; shift
tag=${TAG:+_$TAG}
mkdir -p logs
GEN=${GEN:-}
cmd.exe //c "$(cygpath -w $V/xvlog.bat)" --sv "$@" > logs/${top}${tag}_xvlog.txt 2>&1 || { echo XVLOG FAIL; tail -20 logs/${top}${tag}_xvlog.txt; exit 1; }
cmd.exe //c "$(cygpath -w $V/xelab.bat)" -debug off -timescale 1ps/1ps $GEN $top -s ${top}${tag}_sim > logs/${top}${tag}_xelab.txt 2>&1 || { echo XELAB FAIL; tail -20 logs/${top}${tag}_xelab.txt; exit 1; }
cmd.exe //c "$(cygpath -w $V/xsim.bat)" ${top}${tag}_sim -R > logs/${top}${tag}_xsim.txt 2>&1
grep -v "^\*\*\|Copyright\|^ECHO\|^source\|^#\|INFO\|^Time resolution\|^start_gui\|^run -all\|^exit\|^\$finish\|^quit" logs/${top}${tag}_xsim.txt
