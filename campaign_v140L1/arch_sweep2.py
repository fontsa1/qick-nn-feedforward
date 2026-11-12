"""Unconventional classifier shapes: is there structure the deployed linear-like
ternary 1540-4-1 misses? Follow-up to arch_sweep.py (same data, same scoring,
paired against its selected t4 predictions).

The one physical nonlinearity a single projection cannot use is TIME: an
excited shot that decays (T1) partway through the window traces a trajectory
that starts on the |1> mean and ends on the |0> mean. Every arm below sees the
time axis explicitly.

Features (all from the domain-mapped hw_raw_transform input):
  seg16   the full-covariance LDA projection (ridge 1e-3, train set) split
          into 16 consecutive time segments: s_k = w_k . x_k. Sum = LDA score.
          Hardware: 16 accumulators in the NN_live stream instead of 4.
  bin22   the I/Q trace boxcar-averaged into 22 bins of 35 samples (44 values)

Arms:
  seg_mlp   seg16 -> 16 -> 8 -> 1 (float)
  seg_qda   seg16 -> quadratic discriminant
  seg_hgb   seg16 + bin22 -> gradient-boosted trees (sklearn HistGB)
  bin_mlp   bin22 -> 32 -> 16 -> 1 (float)
  bin_gru   bin22 as a 22-step x 2 sequence -> GRU(16) -> 1
  cnn1d     770 x 2 trace -> Conv1D(16,9,s4) -> Conv1D(16,9,s4) -> 16 -> 1
  loopcv    5-fold CV on loop20k ITSELF: LDA vs seg_mlp vs seg_hgb trained in
            the loopback domain (16k train / 4k test per fold, same folds).
            Answers whether loopback data has any non-linear structure at all,
            independent of the domain map.

Writes campaign_v140L1/arch_sweep2.json (merged).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis_scripts" / "analysis"))
sys.path.insert(0, str(ROOT / "campaign_v140L1"))
import task33_three_way_comparison as T  # noqa: E402
import window_sweep as W                 # noqa: E402
import arch_sweep as A                   # noqa: E402

import tensorflow as tf                   # noqa: E402
from tensorflow.keras import layers as KL  # noqa: E402

PRED = Path(os.environ.get("ARCH_PRED", str(ROOT / "campaign_v140L1" / "arch_sweep_preds")))
PRED.mkdir(parents=True, exist_ok=True)
SEEDS = (20260904, 20260905)
NSEG, NBIN = 16, 22

def lda_w(X, y, lam=1e-3):
    X = X.astype(np.float64)
    m0, m1 = X[y == 0].mean(0), X[y == 1].mean(0)
    Xc = X - np.where(y[:, None] == 1, m1, m0)
    S = (Xc.T @ Xc) / len(Xc)
    return np.linalg.solve(S + lam * np.trace(S) / len(S) * np.eye(len(S)), m1 - m0)

def seg_feats(X, w):
    n = len(X)
    p = (X.astype(np.float64) * w).reshape(n, 770, 2).sum(2)
    edges = np.linspace(0, 770, NSEG + 1).astype(int)
    return np.stack([p[:, a:b].sum(1) for a, b in zip(edges[:-1], edges[1:])], 1)

def bin_feats(X):
    n = len(X)
    return X[:, :2 * 35 * NBIN].reshape(n, NBIN, 35, 2).mean(2).reshape(n, -1)

class Std:
    def __init__(self, X):
        self.m, self.s = X.mean(0), X.std(0) + 1e-9

    def __call__(self, X):
        return ((X - self.m) / self.s).astype(np.float32)

def keras_model(arm, d, seed):
    tf.keras.utils.set_random_seed(seed)
    if arm == "seg_mlp":
        L = [KL.Input((d,)), KL.Dense(16, activation="relu"), KL.Dense(8, activation="relu")]
    elif arm == "bin_mlp":
        L = [KL.Input((d,)), KL.Dense(32, activation="relu"), KL.Dense(16, activation="relu")]
    elif arm == "bin_gru":
        L = [KL.Input((d,)), KL.Reshape((NBIN, 2)), KL.GRU(16)]
    elif arm == "cnn1d":
        L = [KL.Input((d,)), KL.Reshape((770, 2)),
             KL.Conv1D(16, 9, strides=4, activation="relu"),
             KL.Conv1D(16, 9, strides=4, activation="relu"),
             KL.Flatten(), KL.Dense(16, activation="relu")]
    m = tf.keras.Sequential(L + [KL.Dense(1, activation="sigmoid")])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy")
    return m

def fit_keras(arm, Ftr, ytr, Fva, yva, batch=256):
    best = None
    for seed in SEEDS:
        m = keras_model(arm, Ftr.shape[1], seed)
        es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                              restore_best_weights=True)
        h = m.fit(Ftr, ytr.astype(np.float32), validation_data=(Fva, yva.astype(np.float32)),
                  epochs=60, batch_size=batch, verbose=0, callbacks=[es])
        vl = float(min(h.history["val_loss"]))
        print(f"    seed {seed}: val_loss {vl:.5f} ({len(h.history['val_loss'])} ep)", flush=True)
        if best is None or vl < best[0]:
            best = (vl, seed, m)
    return best

def fit_hgb(Ftr, ytr):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05,
                                          early_stopping=True, validation_fraction=0.1,
                                          n_iter_no_change=20, random_state=0).fit(Ftr, ytr)

def fit_qda(Ftr, ytr):
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    return QuadraticDiscriminantAnalysis(reg_param=1e-3).fit(Ftr, ytr)

def loopcv(Xe, ye, res):
    """5-fold CV inside the loopback domain."""
    rng = np.random.default_rng(0)
    folds = np.array_split(rng.permutation(len(ye)), 5)
    pred = {k: np.zeros(len(ye), bool) for k in ("lda", "seg_mlp", "seg_hgb", "seg_qda")}
    for i, te in enumerate(folds):
        tr = np.concatenate([f for j, f in enumerate(folds) if j != i])
        # 10% of the fold's train set picks thresholds / early-stops
        nv = len(tr) // 10
        trn, va = tr[:-nv], tr[-nv:]
        w = lda_w(Xe[trn], ye[trn])
        thr = W.best_thr(Xe[trn].astype(np.float64) @ w, ye[trn])
        pred["lda"][te] = Xe[te].astype(np.float64) @ w > thr
        S = seg_feats(Xe, w)
        st = Std(S[trn])
        _, _, m = fit_keras("seg_mlp", st(S[trn]), ye[trn], st(S[va]), ye[va])
        pred["seg_mlp"][te] = m.predict(st(S[te]), verbose=0).reshape(-1) > 0.5
        tf.keras.backend.clear_session()
        pred["seg_hgb"][te] = fit_hgb(S[trn], ye[trn]).predict(S[te]) == 1
        pred["seg_qda"][te] = fit_qda(st(S[trn]), ye[trn]).predict(st(S[te])) == 1
        print(f"  fold {i}: " + "  ".join(f"{k} {W.fid(v[te].astype(int), ye[te]):.2f}"
                                         for k, v in pred.items()), flush=True)
    r = {k: round(W.fid(v.astype(int), ye), 3) for k, v in pred.items()}
    for k in ("seg_mlp", "seg_hgb", "seg_qda"):
        b, c, p = A.mcnemar(pred[k] == ye, pred["lda"] == ye)
        r[f"{k}_vs_lda"] = dict(b=b, c=c, p=p)
    res["loopcv"] = r
    print("[loopcv] " + json.dumps(r), flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="loopcv,seg_mlp,seg_qda,seg_hgb,bin_mlp,bin_gru,cnn1d")
    ap.add_argument("--src", action="store_true",
                    help="add src20k (test20k_real_X, the shots loop20k was played from, through "
                         "the same map, NOT loopback-acquired) and pair every arm against LDA; "
                         "separates loopback distortion from dataset shift")
    a = ap.parse_args()
    out_path = Path(os.environ["ARCH_OUT"]) if "ARCH_OUT" in os.environ else ROOT / "campaign_v140L1" / ("arch_sweep2_src.json" if a.src else "arch_sweep2.json")
    res = json.loads(out_path.read_text()) if out_path.exists() else {}
    t0 = time.time()
    k2 = complex(np.load(ROOT / "calibration/k2_loopback.npy"))
    c2 = complex(np.load(ROOT / "calibration/c2_loopback.npy"))
    Xr = np.load(ROOT / "0528_X_train_0_770.npy", mmap_mode="r")
    yr = np.load(ROOT / "0528_y_train_0_770.npy", mmap_mode="r")
    X = T.hw_raw_transform(T.apply_affine(np.asarray(Xr[:A.N_TRAIN]).astype(np.float32), k2, c2))
    y = np.asarray(yr[:A.N_TRAIN]).astype(int)
    Xtr, ytr, Xva, yva = X[:-A.N_VAL], y[:-A.N_VAL], X[-A.N_VAL:], y[-A.N_VAL:]
    evals = {
        "loop20k": (T.hw_raw_transform(np.load(os.environ.get("ARCH_EVAL_X", str(ROOT / "data/loopback20k_real.npy"))).astype(np.float32)),
                    np.load(ROOT / "data/loopback20k_real_y.npy").astype(int)),
        "pool100k": (T.hw_raw_transform(T.apply_affine(
            np.asarray(Xr[A.POOL[0]:A.POOL[1]]).astype(np.float32), k2, c2)),
            np.asarray(yr[A.POOL[0]:A.POOL[1]]).astype(int)),
    }
    if a.src:
        evals["src20k"] = (T.hw_raw_transform(T.apply_affine(
            np.load(ROOT / "data/test20k_real_X.npy").astype(np.float32), k2, c2)),
            np.load(ROOT / "data/test20k_real_y.npy").astype(int))
        wl = lda_w(Xtr, ytr)
        thr = W.best_thr(Xtr.astype(np.float64) @ wl, ytr)
        t4 = {ev: Xe.astype(np.float64) @ wl > thr for ev, (Xe, _) in evals.items()}
        res["lda_ref"] = {ev: round(W.fid(t4[ev].astype(int), ye), 3) for ev, (_, ye) in evals.items()}
        print("reference (pairing baseline) = LDA:", res["lda_ref"], flush=True)
    else:
        t4 = {ev: np.load(PRED / f"t4_20260904_{ev}.npy") > 0.5 for ev in evals}
    print(f"loaded ({time.time() - t0:.0f}s)", flush=True)

    arms = a.arms.split(",")
    if "loopcv" in arms:
        t1 = time.time()
        loopcv(*evals["loop20k"], res)
        out_path.write_text(json.dumps(res, indent=1))
        print(f"  ({time.time() - t1:.0f}s)", flush=True)

    w = lda_w(Xtr, ytr)
    feats = {}
    if any(x.startswith("seg") for x in arms):
        feats["seg"] = [seg_feats(Z, w) for Z in (Xtr, Xva) + tuple(e[0] for e in evals.values())]
    if any(x.startswith("bin") or x == "seg_hgb" for x in arms):
        feats["bin"] = [bin_feats(Z) for Z in (Xtr, Xva) + tuple(e[0] for e in evals.values())]
    if "seg_hgb" in arms:
        feats["segbin"] = [np.hstack([s, b]) for s, b in zip(feats["seg"], feats["bin"])]
    if "cnn1d" in arms:
        sc = float(Xtr[:20000].std())
        feats["raw"] = [(Z / sc).astype(np.float32) for Z in (Xtr, Xva) + tuple(e[0] for e in evals.values())]

    for arm in arms:
        if arm == "loopcv":
            continue
        t1 = time.time()
        print(f"[{arm}]", flush=True)
        key = {"seg_mlp": "seg", "seg_qda": "seg", "seg_hgb": "segbin",
               "bin_mlp": "bin", "bin_gru": "bin", "cnn1d": "raw"}[arm]
        Ftr, Fva, *Fev = feats[key]
        st = Std(Ftr) if key != "raw" else (lambda Z: Z)
        r = {}
        if arm == "seg_qda":
            m = fit_qda(st(Ftr), ytr)
            prob = lambda F: m.predict_proba(st(F))[:, 1]          # noqa: E731
        elif arm == "seg_hgb":
            m = fit_hgb(Ftr, ytr)
            prob = lambda F: m.predict_proba(F)[:, 1]              # noqa: E731
        else:
            vl, seed, m = fit_keras(arm, st(Ftr), ytr, st(Fva), yva)
            r.update(val_loss=round(vl, 5), seed=seed, params=int(m.count_params()))
            prob = lambda F: m.predict(st(F), verbose=0, batch_size=4096).reshape(-1)  # noqa: E731
        r["val"] = round(W.fid((prob(Fva) > 0.5).astype(int), yva), 3)
        for (ev, (_, ye)), F in zip(evals.items(), Fev):
            p = prob(F) > 0.5
            np.save(PRED / f"x2{'s' if a.src else ''}_{arm}_{ev}.npy", p)
            b, c, pv = A.mcnemar(p == ye, t4[ev] == ye)
            r[ev] = round(W.fid(p.astype(int), ye), 3)
            r[f"{ev}_vs_t4"] = dict(delta=round(r[ev] - W.fid(t4[ev].astype(int), ye), 3), b=b, c=c, p=pv)
        res[arm] = r
        out_path.write_text(json.dumps(res, indent=1))
        if a.src:
            print(f"[{arm}] " + "  ".join(f"{ev} {r[ev]} ({r[ev + '_vs_t4']['delta']:+.3f} vs LDA, "
                  f"p={r[ev + '_vs_t4']['p']:.2g})" for ev in evals), flush=True)
            tf.keras.backend.clear_session()
            continue
        print(f"[{arm}] val {r['val']}  loop20k {r['loop20k']} ({r['loop20k_vs_t4']['delta']:+.3f}, "
              f"p={r['loop20k_vs_t4']['p']:.2g})  pool100k {r['pool100k']} "
              f"({r['pool100k_vs_t4']['delta']:+.3f}, p={r['pool100k_vs_t4']['p']:.2g})  "
              f"({time.time() - t1:.0f}s)", flush=True)
        tf.keras.backend.clear_session()
    print(f"\nwrote {out_path} ({time.time() - t0:.0f}s total)")

if __name__ == "__main__":
    main()
