"""Ablation of the (k2, c2) domain calibration: does it improve fidelity, and do its labels matter?

(k2, c2) is a global complex affine map applied to all of the training data. Three arms:
  raw           no transform
  cal_perclass  the per-class mean fit (uses data/domain_calib_y.npy, test-file labels)
  cal_global    a label-free global complex affine fit (mean over all shots, lstsq, as tools/run_calibration.py)

Evaluation domain: `data/loopback20k_real.npy`, 20,000 real shots played through the DAC, the analog loopback and the
ADC, with the labels of test20k_real (matched-filter ceiling 95.42% vs 95.40% on disk; mean |z| 248 vs 1063, ratio
0.234, consistent with |k2| = 0.200).

BatchNorm's statistics are frozen after training, so it cannot absorb the ~4.3x scale difference between raw
training data (mean |z| ~ 1063) and acquired data (~248); `raw` is expected to do worst.
"""
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Input, Activation
from qkeras.qlayers import QDense
from qkeras.quantizers import ternary
from pathlib import Path

BASE = Path(".")
N_CPLX = 770
SCALING_FACTOR = 1

WINDOWS = {"full_770": (0, 770), "w100_500": (100, 400)}

def cols_for(start, length):
    s = np.arange(start, start + length)
    return np.stack([2 * s, 2 * s + 1], axis=1).ravel()

def hw_raw_transform(X):
    """Signed int16 sample * scaling_factor, as NN_axi's LOAD_L."""
    return np.clip(np.round(X), -32768, 32767).astype(np.int16).astype(np.float32) * SCALING_FACTOR

def apply_affine(X, k, c):
    z = X[:, 0::2] + 1j * X[:, 1::2]
    w = k * z + c
    out = np.empty_like(X)
    out[:, 0::2] = w.real
    out[:, 1::2] = w.imag
    return out

def fit_global_affine():
    """Label-free complex affine fit: mean trajectory over ALL shots, lstsq.

    Mirrors tools/run_calibration.py. Four real degrees of freedom (a complex gain
    and a complex offset). Consumes no labels, so it carries no F1 leakage.
    """
    Xr = np.load(BASE / "data/domain_calib_raw_X.npy").astype(np.float64)
    Xa = np.load(BASE / "data/domain_calib_acquired.npy").astype(np.float64)
    zr = (Xr[:, 0::2] + 1j * Xr[:, 1::2]).mean(axis=0)
    za = (Xa[:, 0::2] + 1j * Xa[:, 1::2]).mean(axis=0)
    A = np.stack([zr, np.ones_like(zr)], axis=1)
    k, c = np.linalg.lstsq(A, za, rcond=None)[0]
    return complex(k), complex(c)

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
    p10 = float(np.mean(pred[g] != y[g])) if g.sum() else np.nan   # prepared 0, read 1
    p01 = float(np.mean(pred[e] != y[e])) if e.sum() else np.nan   # prepared 1, read 0
    return 1.0 - 0.5 * (p01 + p10), p01, p10

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=10000)
    ap.add_argument("--n-eval", type=int, default=20000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[20260901, 20260902, 20260903])
    a = ap.parse_args()

    k2 = complex(np.load(BASE / "calibration/k2_loopback.npy"))
    c2 = complex(np.load(BASE / "calibration/c2_loopback.npy"))
    kg, cg = fit_global_affine()
    print(f"deployed  (per-class, USES TEST LABELS):  k2 = {k2:.6f}  c2 = {c2:.4f}")
    print(f"label-free (global mean, no labels)    :  kg = {kg:.6f}  cg = {cg:.4f}")
    print(f"  |k2| = {abs(k2):.5f}   |kg| = {abs(kg):.5f}   "
          f"relative gain difference = {abs(kg)/abs(k2)-1:+.3%}")

    # ---- training pool (raw) and labels
    Xtr_raw = np.load(BASE / "0528_X_train_0_770.npy", mmap_mode="r")[:a.n_train].astype(np.float64)
    ytr = np.load(BASE / "0528_y_train_0_770.npy", mmap_mode="r")[:a.n_train].astype(np.float32)
    print(f"\ntrain pool {Xtr_raw.shape}  balance {np.bincount(ytr.astype(int))}  "
          f"mean|z| {np.abs(Xtr_raw[:, 0::2] + 1j*Xtr_raw[:, 1::2]).mean():.1f}")

    # ---- evaluation: REAL ACQUIRED loopback shots
    Xev = np.load(BASE / "data/loopback20k_real.npy", mmap_mode="r")[:a.n_eval].astype(np.float64)
    yev = np.load(BASE / "data/loopback20k_real_y.npy")[:a.n_eval].astype(int)
    print(f"eval (ACQUIRED) {Xev.shape}  balance {np.bincount(yev)}  "
          f"mean|z| {np.abs(Xev[:, 0::2] + 1j*Xev[:, 1::2]).mean():.1f}")

    ARMS = {
        "raw":          lambda X: X,
        "cal_perclass": lambda X: apply_affine(X, k2, c2),
        "cal_global":   lambda X: apply_affine(X, kg, cg),
    }

    results = []
    print(f"\n{'arm':14s} {'window':10s} {'seed':>9s} {'F_acq %':>8s} "
          f"{'P(0|1)%':>8s} {'P(1|0)%':>8s}")
    print("-" * 64)
    for arm, fn in ARMS.items():
        Xtr_arm = hw_raw_transform(fn(Xtr_raw))
        for wname, (ws, wl) in WINDOWS.items():
            cols = cols_for(ws, wl)
            Xtr_w = Xtr_arm[:, cols]
            Xev_w = hw_raw_transform(Xev)[:, cols]
            for seed in a.seeds:
                rng = np.random.default_rng(seed)
                perm = rng.permutation(len(ytr))
                model = build_model(Xtr_w.shape[1], seed)
                es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                                      restore_best_weights=True)
                model.fit(Xtr_w[perm], ytr[perm], validation_split=0.2, epochs=50,
                          batch_size=32, verbose=0, callbacks=[es])
                p = model.predict(Xev_w, verbose=0).reshape(-1)
                pred = (p > 0.5).astype(int)
                F, p01, p10 = fidelity(pred, yev)
                results.append(dict(arm=arm, window=wname, seed=seed, F=F,
                                    p01=p01, p10=p10,
                                    degenerate=bool(p.min() == p.max())))
                print(f"{arm:14s} {wname:10s} {seed:9d} {100*F:8.2f} "
                      f"{100*p01:8.2f} {100*p10:8.2f}")
                tf.keras.backend.clear_session()

    # ---- summary
    print("\n" + "=" * 64)
    print(f"{'arm':14s} {'window':10s} {'F_acq mean+/-sd %':>22s}")
    print("-" * 64)
    summary = {}
    for arm in ARMS:
        for wname in WINDOWS:
            fs = [100 * r["F"] for r in results if r["arm"] == arm and r["window"] == wname]
            summary[f"{arm}|{wname}"] = (float(np.mean(fs)), float(np.std(fs)))
            print(f"{arm:14s} {wname:10s} {np.mean(fs):16.2f} +/- {np.std(fs):.2f}")

    print("\nVERDICT")
    for wname in WINDOWS:
        r = summary[f"raw|{wname}"][0]
        pc = summary[f"cal_perclass|{wname}"][0]
        gl = summary[f"cal_global|{wname}"][0]
        print(f"  [{wname}] calibration is worth {pc - r:+.2f} points over raw; "
              f"label-free fit differs from the test-label fit by {gl - pc:+.2f} points")

    out = BASE / "results/phase0_calibration_ablation.json"
    out.write_text(json.dumps(dict(k2=[k2.real, k2.imag], kg=[kg.real, kg.imag],
                                   c2=[c2.real, c2.imag], cg=[cg.real, cg.imag],
                                   results=results, summary=summary), indent=2))
    print(f"\nwrote {out}")

if __name__ == "__main__":
    main()
