"""Task 3.3: real-only vs synthetic-only vs combined training (Fig. 12).

All arms use the deployed recipe exactly (export_signed_domain_model.py): the
ternary 1540 -> 4 -> 1 network, the label-free domain map (k2, c2) applied to
the raw training data, the signed hw_raw_transform, Adam 1e-3, batch 32, up to
50 epochs, early stopping on val_loss (patience 5) over a 20% validation split
of the arm's OWN training data. All arms are scored on the same independent
test set: `data/loopback20k_real.npy`, 20,000 real shots played through the DAC and
analog loopback and back through the ADC -- the deployment domain, fed to the
model with no map, exactly as eval_labelfree_model_on_acquired.py does for the
deployed model's 95.34%.

Arms:
    real_10k      0528_X_train_0_770[:10000]          (the deployed model's data)
    synth_10k     x_synth_v2[:10000]                  (size-matched to real)
    combined_20k  real_10k + synth_10k
    synth_50k     all of x_synth_v2                   (does MORE synthetic help?)
    real_50k      0528_X_train_0_770[:50000]          (size control for synth_50k;
                  `--arms real_50k --out ...` -- not in the default set)

INDEPENDENCE. x_synth_v2 comes from the v2 generator fitted on
0528_X_train_0_770[10000:110000] -- disjoint from the real training rows [:10000]
and from every test file (tools/verify_no_leakage.py). The x_50k* files are
fitted on a pool that contains test shots and are not used here.

CHECK: real_10k seed 20260902 must reproduce the deployed model's 95.34%.

Reported per arm: fidelity per seed, mean +/- sd over seeds, and the
val_loss-selected seed (the deployment protocol) with a Wilson CI.

"""
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Input, Activation
from qkeras.qlayers import QDense
from qkeras.quantizers import ternary

BASE = Path(".")
SCALING_FACTOR = 1

def hw_raw_transform(X):
    """Signed int16 sample * scaling_factor, as NN_axi's LOAD_L."""
    return (np.clip(np.round(X), -32768, 32767).astype(np.int16)
            .astype(np.float32) * SCALING_FACTOR)

def apply_affine(X, k, c):
    z = X[:, 0::2] + 1j * X[:, 1::2]
    w = k * z + c
    out = np.empty_like(X)
    out[:, 0::2] = w.real
    out[:, 1::2] = w.imag
    return out

def build_model(input_dim, seed):
    tf.keras.utils.set_random_seed(seed)
    L2 = tf.keras.regularizers.l2(1e-3)
    m = tf.keras.Sequential([
        Input(shape=(input_dim,), name="input_1"),
        QDense(4, kernel_quantizer=ternary(), bias_quantizer=ternary(), name="q_dense_1"),
        BatchNormalization(name="batchnorm1", gamma_regularizer=L2, beta_regularizer=L2),
        Activation("relu", name="relu1"),
        QDense(1, kernel_quantizer=ternary(), bias_quantizer=ternary(),
               kernel_regularizer=L2, bias_regularizer=L2, name="q_output"),
        Activation("sigmoid", name="sigmoid"),
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss="binary_crossentropy", metrics=["accuracy"])
    return m

def fidelity(pred, y):
    g, e = y == 0, y == 1
    p10 = float(np.mean(pred[g] != y[g]))      # prepared |0>, read |1>
    p01 = float(np.mean(pred[e] != y[e]))      # prepared |1>, read |0>
    return 1.0 - 0.5 * (p01 + p10), p01, p10

def wilson(k, n, z=1.96):
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    hw = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - hw), 100 * (c + hw)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[20260901, 20260902, 20260903, 20260904, 20260905])
    ap.add_argument("--arms", nargs="+",
                    default=["real_10k", "synth_10k", "combined_20k", "synth_50k"])
    ap.add_argument("--out", default="results/task33_three_way_comparison.json")
    a = ap.parse_args()

    k2 = complex(np.load(BASE / "calibration/k2_loopback.npy"))
    c2 = complex(np.load(BASE / "calibration/c2_loopback.npy"))
    print(f"domain map (label-free): k2 = {k2:.6f}  c2 = {c2:.4f}")
    assert abs(k2 - (0.167813 + 0.006101j)) < 1e-5, \
        "calibration/k2_loopback.npy is not the label-free global map the deployed model uses"

    real_X = np.load(BASE / "0528_X_train_0_770.npy", mmap_mode="r")
    real_y = np.load(BASE / "0528_y_train_0_770.npy", mmap_mode="r")
    syn_X = np.load(BASE / "x_synth_v2.npy", mmap_mode="r")
    syn_y = np.load(BASE / "y_synth_v2.npy", mmap_mode="r")

    # float32 BEFORE the map, exactly as build_domain_calibrated_training_set.py
    # does. A float64 map flips 96 of 15.4M int16 samples across a rounding
    # boundary, and that alone moves seed 20260902 from 95.34% to 94.97%:
    # training is deterministic but sensitive to its input. float32 reproduces
    # the deployed model bit for bit.
    def take(X, y, n):
        return np.asarray(X[:n]).astype(np.float32), np.asarray(y[:n]).astype(np.float32)

    def arm_data(name):
        if name == "real_10k":
            return take(real_X, real_y, 10000)
        if name.startswith("real_") and name.endswith("k"):   # size sweep
            return take(real_X, real_y, 1000 * int(name[5:-1]))
        if name == "synth_10k":
            return take(syn_X, syn_y, 10000)
        if name == "synth_50k":
            return take(syn_X, syn_y, len(syn_y))
        if name == "combined_20k":
            Xr, yr = take(real_X, real_y, 10000)
            Xs, ys = take(syn_X, syn_y, 10000)
            return np.concatenate([Xr, Xs]), np.concatenate([yr, ys])
        raise SystemExit(f"unknown arm {name}")

    Xev = hw_raw_transform(np.load(BASE / "data/loopback20k_real.npy").astype(np.float64))
    yev = np.load(BASE / "data/loopback20k_real_y.npy").astype(int)
    print(f"eval: loopback20k_real {Xev.shape}  balance {np.bincount(yev)}")

    results = []
    print(f"\n{'arm':14s} {'seed':>9s} {'val_loss':>9s} {'F_acq %':>8s} "
          f"{'P(0|1)%':>8s} {'P(1|0)%':>8s} {'epochs':>6s} {'s':>5s}")
    print("-" * 74)
    for arm in a.arms:
        Xraw, ytr = arm_data(arm)
        Xtr = hw_raw_transform(apply_affine(Xraw, k2, c2))
        del Xraw
        for seed in a.seeds:
            t0 = time.time()
            perm = np.random.default_rng(seed).permutation(len(ytr))
            m = build_model(Xtr.shape[1], seed)
            es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                                  restore_best_weights=True)
            h = m.fit(Xtr[perm], ytr[perm], validation_split=0.2, epochs=50,
                      batch_size=32, verbose=0, callbacks=[es])
            vl = float(min(h.history["val_loss"]))
            p = m.predict(Xev, verbose=0).reshape(-1)
            pred = (p > 0.5).astype(int)
            F, p01, p10 = fidelity(pred, yev)
            results.append(dict(arm=arm, seed=seed, n_train=int(len(ytr)),
                                val_loss=vl, F=F, p01=p01, p10=p10,
                                correct=int((pred == yev).sum()),
                                epochs=len(h.history["val_loss"]),
                                degenerate=bool(p.min() == p.max())))
            print(f"{arm:14s} {seed:9d} {vl:9.5f} {100*F:8.2f} {100*p01:8.2f} "
                  f"{100*p10:8.2f} {len(h.history['val_loss']):6d} "
                  f"{time.time()-t0:5.0f}", flush=True)
            tf.keras.backend.clear_session()
        del Xtr

    print("\n" + "=" * 74)
    print(f"{'arm':14s} {'n_train':>8s} {'F mean +/- sd (seeds)':>24s} "
          f"{'val-selected F [95% CI]':>30s}")
    summary = {}
    for arm in a.arms:
        rs = [r for r in results if r["arm"] == arm]
        fs = np.array([100 * r["F"] for r in rs])
        sel = min(rs, key=lambda r: r["val_loss"])
        lo, hi = wilson(sel["correct"], len(yev))
        summary[arm] = dict(n_train=rs[0]["n_train"], mean=float(fs.mean()),
                            sd=float(fs.std(ddof=1)) if len(fs) > 1 else 0.0,
                            selected_seed=sel["seed"], selected_F=100 * sel["F"],
                            selected_acc_ci=[lo, hi])
        print(f"{arm:14s} {rs[0]['n_train']:8d} {fs.mean():14.2f} +/- "
              f"{summary[arm]['sd']:.2f}   {100*sel['F']:10.2f} "
              f"(seed {sel['seed']}) acc CI [{lo:.2f}, {hi:.2f}]")

    out = BASE / a.out
    out.write_text(json.dumps(dict(k2=[k2.real, k2.imag], c2=[c2.real, c2.imag],
                                   eval="loopback20k_real", results=results,
                                   summary=summary), indent=2))
    print(f"\nwrote {out}")

if __name__ == "__main__":
    main()
