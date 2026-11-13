"""Fidelity vs classification window: how much latency can
a shorter window buy?

NN_live decides 15 cycles after the window's last sample, so the decision time
after the readout trigger is (window end E) + 27 ns. Shortening the window is the
only large latency lever left in the classifier. This measures what it costs.

Data, as linear_baselines.py: 200k training shots through the label-free domain
map (k2/c2) and hw_raw_transform; scored on the 20,000 acquired loopback shots
(loopback20k_real). On this data the 200k ternary network equals full-covariance
LDA (95.61% vs 95.61%), so the sweep uses:
  MF   matched filter (w = mu1 - mu0)
  LDA  Fisher discriminant, pooled covariance + ridge; the ridge and both
       thresholds are chosen on a validation slice of the TRAINING pool
and then retrains the deployed ternary 1540-4-1 recipe (task33 build_model)
on the chosen windows (--net), to confirm the network tracks LDA.

Window = samples [s, s+L) of the 770-sample window (columns 2s .. 2(s+L)).
Windows that start BEFORE the current one cannot be evaluated: the data has no
samples before sample 0.

Writes campaign_v140L1/window_sweep.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis_scripts" / "analysis"))
import task33_three_way_comparison as T  # noqa: E402

FS_MHZ = 552.96
N_TRAIN, N_VAL = 200_000, 20_000      # last N_VAL of the training pool = validation
RIDGES = (1e-4, 1e-3, 1e-2, 1e-1)

def fid(pred, y):
    return 100 * (1 - 0.5 * (np.mean(pred[y == 0] != 0) + np.mean(pred[y == 1] != 1)))

def best_thr(s, y):
    o = np.sort(s)
    cand = o[::max(1, len(o) // 4000)]
    f = [fid((s > t).astype(int), y) for t in cand]
    return cand[int(np.argmax(f))]

def cols(s, L):
    return slice(2 * s, 2 * (s + L))

def linear(Xtr, ytr, Xva, yva, Xev, yev, s, L):
    c = cols(s, L)
    A, B, E = Xtr[:, c].astype(np.float64), Xva[:, c].astype(np.float64), Xev[:, c]
    m0, m1 = A[ytr == 0].mean(0), A[ytr == 1].mean(0)
    w = m1 - m0
    f_mf = fid((E @ w > best_thr(A @ w, ytr)).astype(int), yev)
    Ac = A - np.where(ytr[:, None] == 1, m1, m0)
    S = (Ac.T @ Ac) / len(Ac)
    best = None
    for lam in RIDGES:
        wl = np.linalg.solve(S + lam * np.trace(S) / len(S) * np.eye(len(S)), w)
        t = best_thr(A @ wl, ytr)
        fv = fid((B @ wl > t).astype(int), yva)
        if best is None or fv > best[0]:
            best = (fv, lam, wl, t)
    _, lam, wl, t = best
    return f_mf, fid((E @ wl > t).astype(int), yev), lam

def train_net(Xtr, ytr, Xev, yev, s, L, seed=20260904):
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf
    c = cols(s, L)
    A = Xtr[:, c].astype(np.float32)
    perm = np.random.default_rng(seed).permutation(len(ytr))
    m = T.build_model(A.shape[1], seed)
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                          restore_best_weights=True)
    m.fit(A[perm], ytr[perm].astype(np.float32), validation_split=0.1, epochs=50,
          batch_size=32, verbose=0, callbacks=[es])
    p = m.predict(Xev[:, c].astype(np.float32), verbose=0).reshape(-1)
    return fid((p > 0.5).astype(int), yev)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="", help="comma-separated window lengths (s=0) "
                    "to retrain the ternary network on")
    ap.add_argument("--skip-linear", action="store_true")
    a = ap.parse_args()
    out_path = ROOT / "campaign_v140L1" / "window_sweep.json"
    res = json.loads(out_path.read_text()) if out_path.exists() else {}

    t0 = time.time()
    k2 = complex(np.load(ROOT / "calibration/k2_loopback.npy"))
    c2 = complex(np.load(ROOT / "calibration/c2_loopback.npy"))
    X = np.load(ROOT / "0528_X_train_0_770.npy", mmap_mode="r")[:N_TRAIN]
    X = T.hw_raw_transform(T.apply_affine(np.asarray(X).astype(np.float32), k2, c2))
    y = np.load(ROOT / "0528_y_train_0_770.npy")[:N_TRAIN].astype(int)
    Xtr, ytr, Xva, yva = X[:-N_VAL], y[:-N_VAL], X[-N_VAL:], y[-N_VAL:]
    Xev = T.hw_raw_transform(np.load(ROOT / "data/loopback20k_real.npy").astype(np.float64)).astype(np.float64)
    yev = np.load(ROOT / "data/loopback20k_real_y.npy").astype(int)
    print(f"loaded {len(ytr)} train / {len(yva)} val / {len(yev)} eval ({time.time()-t0:.0f}s)", flush=True)

    if not a.skip_linear:
        # where the information is: per-sample class-mean separation |mu1 - mu0|
        d = Xtr[ytr == 1].mean(0) - Xtr[ytr == 0].mean(0)
        res["sep_per_sample"] = np.hypot(d[0::2], d[1::2]).round(3).tolist()

        print("\n[A] window [0, E): decision at E samples after the trigger (+27 ns)")
        print(f"{'E':>5} {'E (ns)':>8} {'MF':>7} {'LDA':>7} ridge")
        res["prefix"] = []
        for E in (770, 704, 640, 576, 512, 448, 384, 352, 320, 288, 256, 224,
                  192, 160, 128, 96, 64, 32):
            f_mf, f_lda, lam = linear(Xtr, ytr, Xva, yva, Xev, yev, 0, E)
            ns = 1e3 * E / FS_MHZ
            res["prefix"].append(dict(E=E, ns=round(ns, 1), mf=round(f_mf, 3),
                                      lda=round(f_lda, 3), ridge=lam))
            print(f"{E:5d} {ns:8.1f} {f_mf:6.2f}% {f_lda:6.2f}% {lam:g}", flush=True)

        print("\n[B] window [s, s+L): LDA, start x length")
        res["grid"] = []
        starts = (0, 32, 64, 128, 192, 256, 384)
        lengths = (64, 128, 256, 384, 512)
        print("  s\\L " + " ".join(f"{L:>7d}" for L in lengths))
        for s in starts:
            row = []
            for L in lengths:
                if s + L > 770:
                    row.append("      -")
                    continue
                _, f_lda, lam = linear(Xtr, ytr, Xva, yva, Xev, yev, s, L)
                res["grid"].append(dict(s=s, L=L, lda=round(f_lda, 3), ridge=lam))
                row.append(f"{f_lda:6.2f}%")
            print(f"{s:5d} " + " ".join(row), flush=True)
        out_path.write_text(json.dumps(res, indent=1))

    if a.net:
        res.setdefault("net", [])
        print("\n[C] ternary 1540-4-1 recipe retrained on window [0, E), 200k shots, seed 20260904")
        for E in (int(v) for v in a.net.split(",")):
            t1 = time.time()
            f = train_net(X, y, Xev, yev, 0, E)
            res["net"] = [r for r in res["net"] if r["E"] != E] + [dict(E=E, net=round(f, 3))]
            out_path.write_text(json.dumps(res, indent=1))
            print(f"  E={E:4d} ({1e3*E/FS_MHZ:7.1f} ns): network {f:.2f}%  ({time.time()-t1:.0f}s)", flush=True)

    out_path.write_text(json.dumps(res, indent=1))
    print(f"\nwrote {out_path} ({time.time()-t0:.0f}s total)")

if __name__ == "__main__":
    main()
