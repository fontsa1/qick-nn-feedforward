"""R9: refit the v2 synthetic generator on smaller real pools (training split only).

Uses qubit_synth_lib exactly as tools/generate_synth_holdout_v2.py does (same fit,
same noise_mode="measured", same warm start), only the fit-pool rows, the number
of shots and the generator seed change. Output is saved as float32 (the training
pipeline casts to float32 before the domain map, so this is lossless for training).

Pools (rows of 0528_X_train_0_770.npy; all inside the 900k TRAINING split, so no
test shot can enter; 5000-shot class blocks -> every pool below is exactly balanced):
    fit10k_p0  10000:20000     fit30k_p0  10000:40000
    fit10k_p1  20000:30000     fit30k_p1  40000:70000
    fit10k_p2  30000:40000     fit30k_p2  70000:100000
    own10k     0:10000   (= the real_10k arm's own training rows)
The fit10k/fit30k pools are subsets of the original v2 fit pool 10000:110000, so
they are disjoint from the real_10k training rows (0:10000).

Modes:
    --validate        refit on 10000:110000 and regenerate the first 300 rows of
                      x_synth_v2 with seed 20260902 (checks this wrapper = original path)
    --pool NAME       fit + generate one pool (see POOLS), --n shots, --seed
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "4")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "tools"))
from qubit_synth_lib import (fit_cavity_and_noise_model, generate_dataset,  # noqa: E402
                             matched_filter_ceiling)

# Copied verbatim from generate_holdout_measured_noise.FULL_POOL_WARM_START (only the
# least-squares warm start; in noise_mode="measured" the cavity fit affects nothing but
# the Phi bookkeeping array). Copied, not imported, so this script does not depend on
# the module that writes x_50k_heldout_measnoise, which is not used.
FULL_POOL_WARM_START = [
    1.01720706e+07, -4.54101244e+04, 3.64588334e+06, -3.07607069e-02,
    2.98501165e-02, 2.70625462e-02, -4.07594194e-02, 9.14256911e-02,
    3.18694928e-02, 6.96860703e-06, 3.61911904e+03,
]

POOLS = {
    "fit10k_p0": (10000, 20000), "fit10k_p1": (20000, 30000), "fit10k_p2": (30000, 40000),
    "fit30k_p0": (10000, 40000), "fit30k_p1": (40000, 70000), "fit30k_p2": (70000, 100000),
    "own10k": (0, 10000),
    "fit100k": (10000, 110000),
}
OUT = Path(__file__).resolve().parent / "data"


def load_pool(lo, hi):
    X = np.load(REPO / "0528_X_train_0_770.npy", mmap_mode="r")
    y = np.load(REPO / "0528_y_train_0_770.npy", mmap_mode="r")
    yp = np.asarray(y[lo:hi]).astype(int)
    Xp = np.asarray(X[lo:hi]).astype(np.float64)
    X0, X1 = Xp[yp == 0], Xp[yp == 1]
    del Xp
    return X0, X1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--pool")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261020)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    if a.validate:
        X0, X1 = load_pool(10000, 110000)
        model = fit_cavity_and_noise_model(X0, X1, initial_guess=FULL_POOL_WARM_START,
                                           verbose=False)
        del X0, X1
        Xg, yg, _ = generate_dataset(model, 300, add_noise=True, seed=20260902,
                                     noise_mode="measured", verbose=False)
        S = np.load(REPO / "x_synth_v2.npy", mmap_mode="r")
        Sy = np.load(REPO / "y_synth_v2.npy")
        res = dict(labels_equal=bool(np.array_equal(yg, Sy[:300])),
                   max_abs_diff=float(np.max(np.abs(Xg - S[:300]))))
        print(res)
        (OUT / "validate_regen_x_synth_v2.json").write_text(json.dumps(res, indent=2))
        return

    lo, hi = POOLS[a.pool]
    t0 = time.time()
    X0, X1 = load_pool(lo, hi)
    n0, n1 = len(X0), len(X1)
    # real-pool matched-filter ceiling (same acceptance test as the original script)
    nr = min(6000, n0, n1)
    dr, fr = matched_filter_ceiling(np.concatenate([X0[:nr], X1[:nr]]),
                                    np.r_[np.zeros(nr, int), np.ones(nr, int)])
    model = fit_cavity_and_noise_model(X0, X1, initial_guess=FULL_POOL_WARM_START,
                                       verbose=False)
    del X0, X1
    t_fit = time.time() - t0
    Xg, yg, _ = generate_dataset(model, a.n, add_noise=True, seed=a.seed,
                                 noise_mode="measured", verbose=False)
    d, f = matched_filter_ceiling(Xg, yg)
    tag = f"synth_{a.pool}_n{a.n // 1000}k_s{a.seed}"
    np.save(OUT / f"{tag}_X.npy", Xg.astype(np.float32))
    np.save(OUT / f"{tag}_y.npy", yg.astype(np.int64))
    info = dict(pool=a.pool, rows=[lo, hi], n_class0=n0, n_class1=n1,
                cov_rows_per_class=[min(n0, 40000), min(n1, 40000)],
                n_generated=a.n, seed=a.seed, balance=np.bincount(yg).tolist(),
                real_pool_mf_d=dr, real_pool_mf_F=fr, synth_mf_d=d, synth_mf_F=f,
                fit_s=t_fit, total_s=time.time() - t0)
    (OUT / f"{tag}.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info))


if __name__ == "__main__":
    main()
