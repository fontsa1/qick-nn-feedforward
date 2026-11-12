#!/bin/bash
# v1.44 offline analyses, in dependency order. Log: campaign_v144/final/analysis/run_analyses.log. No board access.
# Not here: loop_groups_data_v144.py (needs the latency closure question settled and the cold starts).
cd "${REPO_ROOT:-.}"
export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=4 TF_NUM_INTRAOP_THREADS=4
PY="${PYTHON:-python}"; A=campaign_v144/final/analysis; LOG=$PWD/$A/run_analyses.log   # set PYTHON to your environment's interpreter, e.g. PYTHON=.venv/bin/python
say () { echo "$(date '+%F %T') $*" >> $LOG; }
step () { n=$1; shift; "$@" > $A/$n.log 2>&1 < /dev/null; say "$n rc=$?"; }
say START
step analyze_axi_emu $PY $A/analyze_axi_emu_v144.py
step linear_v144_A $PY $A/linear_v144.py A
step linear_v144_B $PY $A/linear_v144.py B
step box_vs_fir_v144 $PY $A/box_vs_fir_v144.py
step box_vs_fir_delaymap $PY $A/box_vs_fir_delaymap_v144.py
step scaling_v144 $PY $A/scaling_v144.py
( cd campaign_v144/final/fig7 && ../../../$PY make_fig7_v144.py > ../analysis/fig7_v144.log 2>&1 < /dev/null; say "fig7 rc=$?" )
( cd campaign_v144/final/fig8b && ../../../$PY make_fig8b_v144.py > ../analysis/fig8b_v144.log 2>&1 < /dev/null; say "fig8b rc=$?" )
bash $A/r9/run_r9_v144.sh > $A/r9/run.log 2>&1 < /dev/null &
R=$!
ARCH_EVAL_X=$A/replay20k_A.npy ARCH_PRED=$A/arch/preds ARCH_OUT=$A/arch/arch_sweep2_src_v144.json $PY campaign_v140L1/arch_sweep2.py --src --arms seg_hgb,seg_mlp,bin_mlp,cnn1d,bin_gru > $A/arch/run.log 2>&1 < /dev/null
say "arch rc=$?"
wait $R; say "r9 train rc=$?"
R9_RESULTS=$A/r9/results R9_STATS=$A/r9/r9_stats_v144.json $PY audit_fixes/R9/r9_stats.py > $A/r9/stats.txt 2>&1 < /dev/null; say "r9 stats rc=$?"
step fig4_lags $PY $A/fig4_lags_v144.py
step channel_v144 $PY $A/channel_v144.py
say ALL_DONE
