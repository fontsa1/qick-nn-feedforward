#!/bin/bash
# R9 full pipeline: generator validation + refits, then training arms. Sequential, 4 threads.
# Launch (detached):  nohup bash audit_fixes/R9/run_all.sh > audit_fixes/R9/logs/run_all.log 2>&1 &
# Each step skips itself if its output already exists (safe to relaunch after an interruption).
set -u
cd "${REPO_ROOT:-.}"/audit_fixes/R9
PY="${PYTHON:-python}"   # set PYTHON to your environment's interpreter, e.g. PYTHON=.venv/bin/python
S5="20260901 20260902 20260903 20260904 20260905"
S10="$S5 20260906 20260907 20260908 20260909 20260910"
S6_10="20260906 20260907 20260908 20260909 20260910"

echo "=== start $(date)"
[ -f data/validate_regen_x_synth_v2.json ] || $PY r9_fit_generators.py --validate
i=0
for p in fit10k_p0 fit10k_p1 fit10k_p2 fit30k_p0 fit30k_p1 fit30k_p2; do
  s=$((20261020 + i)); i=$((i + 1))
  [ -f data/synth_${p}_n10k_s${s}.json ] || $PY r9_fit_generators.py --pool $p --n 10000 --seed $s
done
[ -f data/synth_own10k_n50k_s20261030.json ] || $PY r9_fit_generators.py --pool own10k --n 50000 --seed 20261030
echo "=== generators done $(date)"

run () {  # run OUT seeds... -- arms...
  out=$1; shift
  if [ -f results/$out.done ]; then echo "skip $out"; return; fi
  $PY r9_train_eval.py --out results/$out "$@" && touch results/$out.done
}
# 10k arms, draw 0 with 10 seeds (reproduces + extends task33 real_10k / synth_10k)
run t10k_d0 --seeds $S10 --arms real10k_d0=real:0:10000 v2_10k_d0=v2:0:10000
# extra data draws, 5 seeds each
run t10k_draws --seeds $S5 --arms real10k_d1=real:110000:120000 real10k_d2=real:120000:130000 \
    v2_10k_d1=v2:10000:20000 v2_10k_d2=v2:20000:30000
# small fit pools, 3 pools x 5 seeds
run t10k_fit10k --seeds $S5 --arms fit10k_p0=file:synth_fit10k_p0_n10k_s20261020 \
    fit10k_p1=file:synth_fit10k_p1_n10k_s20261021 fit10k_p2=file:synth_fit10k_p2_n10k_s20261022
run t10k_fit30k --seeds $S5 --arms fit30k_p0=file:synth_fit30k_p0_n10k_s20261023 \
    fit30k_p1=file:synth_fit30k_p1_n10k_s20261024 fit30k_p2=file:synth_fit30k_p2_n10k_s20261025
# generator fit on the real_10k rows themselves (equal information)
run t10k_own --seeds $S10 --arms own10k_synth10k=file:synth_own10k_n50k_s20261030:0:10000
# 50k arms
run t50k --seeds $S10 --arms real50k=real:0:50000 v2_50k=v2:0:50000 \
    own10k_synth50k=file:synth_own10k_n50k_s20261030 \
    real10k_plus_own50k=real:0:10000+file:synth_own10k_n50k_s20261030
echo "=== all done $(date)"
