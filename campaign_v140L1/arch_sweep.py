"""Larger classifier architectures: does a 1540-64-32-1 network buy fidelity over
the deployed ternary 1540-4-1?

Data, as window_sweep.py: the first 200k shots of 0528_X_train_0_770 through the
label-free domain map (k2/c2) and hw_raw_transform; the last 20k of those are
the validation slice (early stopping and seed choice only). Scored on
  loop20k   the 20,000 acquired loopback shots (loopback20k_real), the
            deployment-domain number
  pool100k  0528_X_train_0_770[800000:900000] through the same map: real
            qubit shots no model here or in the deployed build trained on
            (not loopback-acquired, so a domain-mapped proxy, but 5x the shots)

Arms (BatchNormalization + ReLU after every hidden layer, sigmoid output):
  t4        ternary 1540-4-1, the deployed recipe (task33 build_model)
  t64_32    ternary 1540-64-32-1, every layer ternary
  t64_q8    ternary 1540-64, then 8-bit (quantized_bits(8,0)) 64-32-1
  f64_32    float 1540-64-32-1 (upper bound for this shape)
  lda       full-covariance Fisher discriminant (window_sweep.linear), the
            linear reference the 200k ternary 4-1 was found to equal

Each network arm trains --seeds seeds (Adam 1e-3, batch 32, early stopping on
val_loss, patience 5); the arm's reported model is the lowest-val_loss seed, as
export_signed_domain_model.py selects. Per-seed numbers are kept too. Paired
comparisons against t4 on the same shots (McNemar's exact test on the
discordant shots) are printed for the selected models.

Writes campaign_v140L1/arch_sweep.json (merged, so arms can be run separately).
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

import tensorflow as tf                   # noqa: E402
from tensorflow.keras.layers import Activation, BatchNormalization, Dense, Input  # noqa: E402
from qkeras.qlayers import QDense         # noqa: E402
from qkeras.quantizers import quantized_bits, ternary  # noqa: E402

N_TRAIN, N_VAL = 200_000, 20_000
POOL = (800_000, 900_000)
SEED0 = 20260904

def dense(kind, units, name, reg=None):
    if kind == "t":
        return QDense(units, kernel_quantizer=ternary(), bias_quantizer=ternary(),
                      kernel_regularizer=reg, bias_regularizer=reg, name=name)
    if kind == "q8":
        q = quantized_bits(8, 0, alpha=1)
        return QDense(units, kernel_quantizer=q, bias_quantizer=q,
                      kernel_regularizer=reg, bias_regularizer=reg, name=name)
    return Dense(units, kernel_regularizer=reg, bias_regularizer=reg, name=name)

# arm -> (hidden widths, quantizer per dense layer incl. output)
ARMS = {
    "t4": ((4,), ("t", "t")),
    "t64_32": ((64, 32), ("t", "t", "t")),
    "t64_q8": ((64, 32), ("t", "q8", "q8")),
    "f64_32": ((64, 32), ("f", "f", "f")),
}

def build(arm, input_dim, seed):
    if arm == "t4":
        return T.build_model(input_dim, seed)
    tf.keras.utils.set_random_seed(seed)
    L2 = tf.keras.regularizers.l2(1e-3)
    widths, kinds = ARMS[arm]
    layers = [Input(shape=(input_dim,), name="input_1")]
    for i, (w, k) in enumerate(zip(widths, kinds)):
        layers += [dense(k, w, f"q_dense_{i + 1}", None if i == 0 else L2),
                   BatchNormalization(name=f"batchnorm{i + 1}", gamma_regularizer=L2,
                                      beta_regularizer=L2),
                   Activation("relu", name=f"relu{i + 1}")]
    layers += [dense(kinds[-1], 1, "q_output", L2), Activation("sigmoid", name="sigmoid")]
    m = tf.keras.Sequential(layers)
    m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss="binary_crossentropy", metrics=["accuracy"])
    return m

def mcnemar(ca, cb):
    """Exact two-sided McNemar p for paired correctness vectors."""
    from math import comb
    b = int(np.sum(ca & ~cb)); c = int(np.sum(~ca & cb)); n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return b, c, min(1.0, p)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="t4,t64_32,t64_q8,f64_32,lda")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    out_path = ROOT / "campaign_v140L1" / "arch_sweep.json"
    res = json.loads(out_path.read_text()) if out_path.exists() else {}
    pred_dir = ROOT / "campaign_v140L1" / "arch_sweep_preds"
    pred_dir.mkdir(exist_ok=True)

    t0 = time.time()
    k2 = complex(np.load(ROOT / "calibration/k2_loopback.npy"))
    c2 = complex(np.load(ROOT / "calibration/c2_loopback.npy"))
    Xr = np.load(ROOT / "0528_X_train_0_770.npy", mmap_mode="r")
    yr = np.load(ROOT / "0528_y_train_0_770.npy", mmap_mode="r")
    X = T.hw_raw_transform(T.apply_affine(np.asarray(Xr[:N_TRAIN]).astype(np.float32), k2, c2))
    y = np.asarray(yr[:N_TRAIN]).astype(int)
    Xtr, ytr, Xva, yva = X[:-N_VAL], y[:-N_VAL], X[-N_VAL:], y[-N_VAL:]
    evals = {
        "loop20k": (T.hw_raw_transform(np.load(ROOT / "data/loopback20k_real.npy").astype(np.float32)),
                    np.load(ROOT / "data/loopback20k_real_y.npy").astype(int)),
        "pool100k": (T.hw_raw_transform(T.apply_affine(
            np.asarray(Xr[POOL[0]:POOL[1]]).astype(np.float32), k2, c2)),
            np.asarray(yr[POOL[0]:POOL[1]]).astype(int)),
    }
    print(f"loaded {len(ytr)} train / {len(yva)} val / "
          + " / ".join(f"{len(v[1])} {k}" for k, v in evals.items())
          + f" ({time.time() - t0:.0f}s)", flush=True)

    for arm in a.arms.split(","):
        t1 = time.time()
        if arm == "lda":
            r = {}
            for ev, (Xe, ye) in evals.items():
                _, f, lam = W.linear(Xtr, ytr, Xva, yva, Xe.astype(np.float64), ye, 0, 770)
                r[ev] = round(f, 3)
            r["ridge"] = lam
            res["lda"] = r
            print(f"[lda] " + "  ".join(f"{k} {v}" for k, v in r.items())
                  + f"  ({time.time() - t1:.0f}s)", flush=True)
            out_path.write_text(json.dumps(res, indent=1))
            continue
        runs = []
        for s in range(a.seeds):
            seed = SEED0 + s
            m = build(arm, Xtr.shape[1], seed)
            es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                                  restore_best_weights=True)
            perm = np.random.default_rng(seed).permutation(len(ytr))
            h = m.fit(Xtr[perm], ytr[perm].astype(np.float32), validation_data=(Xva, yva.astype(np.float32)),
                      epochs=50, batch_size=a.batch, verbose=0, callbacks=[es])
            vl = float(min(h.history["val_loss"]))
            row = dict(seed=seed, val_loss=round(vl, 5), epochs=len(h.history["val_loss"]),
                       params=int(m.count_params()))
            pv = m.predict(Xva, verbose=0, batch_size=4096).reshape(-1)
            row["val"] = round(W.fid((pv > 0.5).astype(int), yva), 3)
            for ev, (Xe, ye) in evals.items():
                p = m.predict(Xe, verbose=0, batch_size=4096).reshape(-1)
                row[ev] = round(W.fid((p > 0.5).astype(int), ye), 3)
                np.save(pred_dir / f"{arm}_{seed}_{ev}.npy", p.astype(np.float32))
            runs.append(row)
            print(f"[{arm}] seed {seed}: val_loss {vl:.5f} ({row['epochs']} ep)  val {row['val']}%  "
                  f"loop20k {row['loop20k']}%  pool100k {row['pool100k']}%  "
                  f"({time.time() - t1:.0f}s)", flush=True)
            tf.keras.backend.clear_session()
        best = min(runs, key=lambda r: r["val_loss"])
        res[arm] = dict(runs=runs, selected=best["seed"],
                        mean={ev: round(float(np.mean([r[ev] for r in runs])), 3)
                              for ev in ("val", "loop20k", "pool100k")},
                        sd={ev: round(float(np.std([r[ev] for r in runs])), 3)
                            for ev in ("val", "loop20k", "pool100k")})
        out_path.write_text(json.dumps(res, indent=1))

    # paired comparison vs t4 (selected seeds)
    if "t4" in res:
        print("\npaired vs t4 (selected seeds; b = arm right & t4 wrong, c = reverse)")
        res["paired"] = {}
        for arm in ARMS:
            if arm == "t4" or arm not in res:
                continue
            for ev, (_, ye) in evals.items():
                pa = np.load(pred_dir / f"{arm}_{res[arm]['selected']}_{ev}.npy") > 0.5
                pb = np.load(pred_dir / f"t4_{res['t4']['selected']}_{ev}.npy") > 0.5
                b, c, p = mcnemar(pa == ye, pb == ye)
                d = W.fid(pa.astype(int), ye) - W.fid(pb.astype(int), ye)
                res["paired"][f"{arm}_{ev}"] = dict(delta=round(d, 3), b=b, c=c, p=p)
                print(f"  {arm:>8} {ev:>8}: {d:+.3f} pts  b={b} c={c}  p={p:.3g}")
        out_path.write_text(json.dumps(res, indent=1))
    print(f"\nwrote {out_path} ({time.time() - t0:.0f}s total)")

if __name__ == "__main__":
    main()
