"""R9: synthetic-vs-real training comparison with per-shot predictions saved.

Training/eval recipe copied verbatim from analysis_scripts/analysis/task33_three_way_comparison.py
(the section 6.3 protocol): ternary 1540->4->1 QKeras net, label-free domain map (k2, c2)
applied in float32 to the raw training data, signed hw_raw_transform, Adam 1e-3,
batch 32, <= 50 epochs, early stopping on val_loss (patience 5, restore best) over a
20% validation split of the arm's OWN training data; permutation and init from the
seed. Scored on loopback20k_real (20,000 replayed real shots, no map).

Differences from task33: (1) per-shot sigmoid outputs are saved for every run so arms
can be compared PAIRED on the same test shots; (2) more arms / data draws (below);
(3) TF/BLAS limited to 4 threads (audit compute budget).

Arm spec  NAME=SOURCE[+SOURCE...]  where SOURCE is
    real:LO:HI              rows LO:HI of 0528_X_train_0_770.npy
    v2:LO:HI                rows LO:HI of x_synth_v2.npy (generator fit on 100k real)
    file:TAG                audit_fixes/R9/data/TAG_X.npy / TAG_y.npy (refit generators)
"""
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_NUM_INTRAOP_THREADS"] = "4"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "4"

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(1)
from tensorflow.keras.layers import BatchNormalization, Input, Activation  # noqa: E402
from qkeras.qlayers import QDense  # noqa: E402
from qkeras.quantizers import ternary  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SCALING_FACTOR = 1

def hw_raw_transform(X):
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
    return 1.0 - 0.5 * (np.mean(pred[e] != y[e]) + np.mean(pred[g] != y[g]))

def load_source(src):
    kind, *rest = src.split(":")
    if kind == "real":
        lo, hi = int(rest[0]), int(rest[1])
        X = np.load(REPO / "0528_X_train_0_770.npy", mmap_mode="r")
        y = np.load(REPO / "0528_y_train_0_770.npy", mmap_mode="r")
    elif kind == "v2":
        lo, hi = int(rest[0]), int(rest[1])
        X = np.load(REPO / "x_synth_v2.npy", mmap_mode="r")
        y = np.load(REPO / "y_synth_v2.npy", mmap_mode="r")
    elif kind == "file":
        X = np.load(HERE / "data" / f"{rest[0]}_X.npy", mmap_mode="r")
        y = np.load(HERE / "data" / f"{rest[0]}_y.npy", mmap_mode="r")
        lo = int(rest[1]) if len(rest) > 1 else 0
        hi = int(rest[2]) if len(rest) > 2 else len(y)
    else:
        raise SystemExit(f"bad source {src}")
    # float32 BEFORE the map, exactly as task33 / build_domain_calibrated_training_set
    return np.asarray(X[lo:hi]).astype(np.float32), np.asarray(y[lo:hi]).astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    k2 = complex(np.load(REPO / "calibration/k2_loopback.npy"))
    c2 = complex(np.load(REPO / "calibration/c2_loopback.npy"))
    assert abs(k2 - (0.167813 + 0.006101j)) < 1e-5
    Xev = hw_raw_transform(np.load(__import__("os").environ.get("R9_EVAL_X", str(REPO / "data/loopback20k_real.npy"))).astype(np.float64))  # v1.43: replay20k_A
    yev = np.load(REPO / "data/test20k_real_y.npy").astype(int)
    keep = np.ones(len(yev), bool)
    keep[1000:1600] = False                       # domain-map calibration rows

    out = HERE / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    results, probs = [], {}
    for spec in a.arms:
        name, srcs = spec.split("=", 1)
        parts = [load_source(s) for s in srcs.split("+")]
        Xraw = np.concatenate([p[0] for p in parts]); ytr = np.concatenate([p[1] for p in parts])
        del parts
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
            p = m.predict(Xev, verbose=0).reshape(-1).astype(np.float32)
            pred = (p > 0.5).astype(int)
            r = dict(arm=name, spec=srcs, seed=seed, n_train=int(len(ytr)), val_loss=vl,
                     F20k=float(fidelity(pred, yev)),
                     F19400=float(fidelity(pred[keep], yev[keep])),
                     epochs=len(h.history["val_loss"]), s=time.time() - t0)
            results.append(r)
            probs[f"{name}|{seed}"] = p
            print(f"{name:22s} {seed:9d} vl {vl:.5f} F20k {100*r['F20k']:.3f} "
                  f"F19400 {100*r['F19400']:.3f} ep {r['epochs']:2d} {r['s']:4.0f}s", flush=True)
            tf.keras.backend.clear_session()
            # checkpoint after every run (safe to resume / inspect)
            Path(str(out) + ".json").write_text(json.dumps(results, indent=1))
            np.savez_compressed(str(out) + "_probs.npz", **probs)
        del Xtr
    print("done", out)

if __name__ == "__main__":
    main()
