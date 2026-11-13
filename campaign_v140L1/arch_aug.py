"""Nonlinear arms retrained on training shots passed through a fitted loopback channel model, scored on loopback
shots. Un-augmented reference: arch_sweep2.py --src (nonlinear arms +0.20..+0.34 over LDA on the domain-mapped source
shots, -0.0..-0.9 on the same shots after DAC -> ADC).

Measured loopback channel (loopback20k_real vs the mapped source shot):
the window starts 3-5 samples later (jittering per shot), and a large
coloured noise is added (within-class noise 92 -> 163, per-sample correlation
with the source fluctuation 0.30, 35-sample-smoothed 0.85).

Channel model, fitted ONLY on loopback shots [0, N_FIT) and their sources:
  loop = h * src(t - d) + m(t) + n(t)
  d ~ the per-shot best-lag histogram; h a linear filter fitted per FFT bin
  on the lag-aligned shots (a scalar gain fails: the channel passes slow
  structure at ~1.4x and attenuates fast fluctuations); m the mean residual
  trace; n Gaussian with the residual's average power spectrum.
Training shots (0528 [0:200k] through k2/c2, as arch_sweep) are passed through
this channel once (static augmentation, seed 0), then every arm retrains.
Scored on loopback shots [N_FIT, 20000) only; LDA trained on clean and on
augmented data are both reported, and arms are paired against the clean LDA
(the deployed 4-1 equals it) and against the augmented LDA.

Writes campaign_v140L1/arch_aug.json.
"""
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis_scripts" / "analysis"))
sys.path.insert(0, str(ROOT / "campaign_v140L1"))
import task33_three_way_comparison as T  # noqa: E402
import window_sweep as W                 # noqa: E402
import arch_sweep as A                   # noqa: E402
import arch_sweep2 as X2                 # noqa: E402

import tensorflow as tf                   # noqa: E402

N_FIT = 4000
LAGS = range(-8, 9)

def cplx(X):
    return X[:, 0::2] + 1j * X[:, 1::2]

def real(z):
    out = np.empty((len(z), 2 * z.shape[1]), np.float32)
    out[:, 0::2], out[:, 1::2] = z.real, z.imag
    return out

def shift(z, d):
    """z(t - d) with edge hold (no wrap)."""
    if d == 0:
        return z
    out = np.empty_like(z)
    out[..., d:] = z[..., :-d]
    out[..., :d] = z[..., :1]
    return out

def fit_channel(zs, zl):
    ds, dl = zs - zs.mean(0), zl - zl.mean(0)
    C = np.array([[np.abs(np.vdot(np.roll(ds[i], l)[10:-10], dl[i][10:-10])) for l in LAGS]
                  for i in range(len(zs))])
    lag = np.array(LAGS)[C.argmax(1)]
    al = np.stack([shift(zs[i:i + 1], int(lag[i]))[0] for i in range(len(zs))])
    # linear time-invariant channel: L(f) = H(f) S(f) + N(f), fitted per bin
    S, L = np.fft.fft(al, axis=1), np.fft.fft(zl, axis=1)
    H = (np.conj(S) * L).sum(0) / (np.abs(S) ** 2).sum(0)
    r = zl - np.fft.ifft(H * S, axis=1)
    psd = (np.abs(np.fft.fft(r - r.mean(0), axis=1)) ** 2).mean(0)
    hist = Counter(lag.tolist())
    return dict(H=H, psd=psd, lag_vals=np.array(sorted(hist)),
                lag_p=np.array([hist[k] for k in sorted(hist)]) / len(lag),
                resid_std=float(np.abs(r - r.mean(0)).std()), mean_offset=r.mean(0))

def apply_channel(X, ch, rng, chunk=20000):
    out = np.empty_like(X)
    n_t = X.shape[1] // 2
    for s in range(0, len(X), chunk):
        z = cplx(X[s:s + chunk].astype(np.float64))
        d = rng.choice(ch["lag_vals"], size=len(z), p=ch["lag_p"])
        for v in np.unique(d):
            z[d == v] = shift(z[d == v], int(v))
        wn = (rng.standard_normal((len(z), n_t)) + 1j * rng.standard_normal((len(z), n_t))) / np.sqrt(2)
        noise = np.fft.ifft(np.fft.fft(wn, axis=1) * np.sqrt(ch["psd"] / n_t), axis=1)
        z = np.fft.ifft(ch["H"] * np.fft.fft(z, axis=1), axis=1) + ch["mean_offset"] + noise
        out[s:s + chunk] = T.hw_raw_transform(real(z))
    return out

def main():
    t0 = time.time()
    out_path = ROOT / "campaign_v140L1" / "arch_aug.json"
    k2 = complex(np.load(ROOT / "calibration/k2_loopback.npy"))
    c2 = complex(np.load(ROOT / "calibration/c2_loopback.npy"))
    src = T.apply_affine(np.load(ROOT / "data/test20k_real_X.npy")[:N_FIT].astype(np.float64), k2, c2)
    Lraw = np.load(ROOT / "data/loopback20k_real.npy").astype(np.float64)
    ch = fit_channel(cplx(src), cplx(Lraw[:N_FIT]))
    # check the generated noise power against the measured residual
    f = np.abs(np.fft.fftfreq(len(ch["H"]), 1 / 552.96))
    print("channel |H|: " + "  ".join(f"{lo}-{hi} MHz {np.abs(ch['H'][(f >= lo) & (f < hi)]).mean():.2f}"
                                      for lo, hi in ((0, 5), (5, 20), (20, 60), (60, 150), (150, 277))), flush=True)
    print(f"channel: lags {dict(zip(ch['lag_vals'].tolist(), ch['lag_p'].round(3).tolist()))}"
          f" resid std {ch['resid_std']:.1f}", flush=True)
    rng = np.random.default_rng(0)
    test = apply_channel(src.astype(np.float32), ch, rng)
    zl = cplx(Lraw[:N_FIT])
    y4 = np.load(ROOT / "data/loopback20k_real_y.npy")[:N_FIT]
    zt = cplx(test.astype(np.float64))
    for nm, z in (("loop", zl), ("channel(src)", zt)):
        m0, m1 = z[y4 == 0].mean(0), z[y4 == 1].mean(0)
        wc = np.concatenate([z[y4 == 0] - m0, z[y4 == 1] - m1])
        print(f"sanity {nm}: |mu1-mu0| {np.abs(m1 - m0).mean():.1f}  within-class {np.abs(wc).std():.1f}", flush=True)
    print(f"sanity (fit shots): loop std {np.abs(zl - zl.mean(0)).std():.1f} vs "
          f"channel(src) std {np.abs(cplx(test.astype(np.float64)) - cplx(test.astype(np.float64)).mean(0)).std():.1f}",
          flush=True)

    Xr = np.load(ROOT / "0528_X_train_0_770.npy", mmap_mode="r")
    yr = np.load(ROOT / "0528_y_train_0_770.npy", mmap_mode="r")
    Xc = T.apply_affine(np.asarray(Xr[:A.N_TRAIN]).astype(np.float32), k2, c2)
    y = np.asarray(yr[:A.N_TRAIN]).astype(int)
    Xa = apply_channel(Xc, ch, rng)
    Xc = T.hw_raw_transform(Xc)
    Xe = T.hw_raw_transform(Lraw[N_FIT:].astype(np.float32))
    ye = np.load(ROOT / "data/loopback20k_real_y.npy")[N_FIT:].astype(int)
    print(f"data ready ({time.time() - t0:.0f}s)", flush=True)

    res, preds = {}, {}
    for tag, X in (("clean", Xc), ("aug", Xa)):
        Xtr, ytr, Xva, yva = X[:-A.N_VAL], y[:-A.N_VAL], X[-A.N_VAL:], y[-A.N_VAL:]
        w = X2.lda_w(Xtr, ytr)
        thr = W.best_thr(Xtr.astype(np.float64) @ w, ytr)
        preds[f"lda_{tag}"] = Xe.astype(np.float64) @ w > thr
        res[f"lda_{tag}"] = round(W.fid(preds[f"lda_{tag}"].astype(int), ye), 3)
        print(f"[lda_{tag}] loop16k {res[f'lda_{tag}']}", flush=True)
        if tag == "clean":
            continue
        feats = {"seg": [X2.seg_feats(Z, w) for Z in (Xtr, Xva, Xe)],
                 "bin": [X2.bin_feats(Z) for Z in (Xtr, Xva, Xe)]}
        sc = float(Xtr[:20000].std())
        feats["raw"] = [(Z / sc).astype(np.float32) for Z in (Xtr, Xva, Xe)]
        for arm, key in (("seg_mlp", "seg"), ("bin_mlp", "bin"), ("bin_gru", "bin"), ("cnn1d", "raw")):
            t1 = time.time()
            Ftr, Fva, Fe = feats[key]
            st = X2.Std(Ftr) if key != "raw" else (lambda Z: Z)
            _, _, m = X2.fit_keras(arm, st(Ftr), ytr, st(Fva), yva)
            p = m.predict(st(Fe), verbose=0, batch_size=4096).reshape(-1) > 0.5
            preds[f"{arm}_aug"] = p
            r = dict(loop16k=round(W.fid(p.astype(int), ye), 3))
            for ref in ("lda_clean", "lda_aug"):
                b, c, pv = A.mcnemar(p == ye, preds[ref] == ye)
                r[f"vs_{ref}"] = dict(delta=round(r["loop16k"] - res[ref], 3), b=b, c=c, p=pv)
            res[f"{arm}_aug"] = r
            out_path.write_text(json.dumps(res, indent=1))
            print(f"[{arm}_aug] loop16k {r['loop16k']}  vs clean LDA {r['vs_lda_clean']['delta']:+.3f} "
                  f"(p={r['vs_lda_clean']['p']:.2g})  vs aug LDA {r['vs_lda_aug']['delta']:+.3f} "
                  f"(p={r['vs_lda_aug']['p']:.2g})  ({time.time() - t1:.0f}s)", flush=True)
            tf.keras.backend.clear_session()
    # the deployed recipe (ternary 1540-4-1, raw input), clean (arch_sweep's
    # selected seed) and retrained on the augmented set (2 seeds, lowest val_loss)
    preds["t4_clean"] = np.load(A.ROOT / "campaign_v140L1" / "arch_sweep_preds"
                                / "t4_20260904_loop20k.npy")[N_FIT:] > 0.5
    Xtr, ytr, Xva, yva = Xa[:-A.N_VAL], y[:-A.N_VAL], Xa[-A.N_VAL:], y[-A.N_VAL:]
    best = None
    for seed in (20260904, 20260905):
        t1 = time.time()
        m = A.build("t4", Xtr.shape[1], seed)
        es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        perm = np.random.default_rng(seed).permutation(len(ytr))
        h = m.fit(Xtr[perm], ytr[perm].astype(np.float32), validation_data=(Xva, yva.astype(np.float32)),
                  epochs=50, batch_size=32, verbose=0, callbacks=[es])
        vl = float(min(h.history["val_loss"]))
        p = m.predict(Xe, verbose=0, batch_size=4096).reshape(-1) > 0.5
        print(f"  t4_aug seed {seed}: val_loss {vl:.5f}  loop16k {W.fid(p.astype(int), ye):.3f}  "
              f"({time.time() - t1:.0f}s)", flush=True)
        if best is None or vl < best[0]:
            best = (vl, p)
        tf.keras.backend.clear_session()
    preds["t4_aug"] = best[1]
    pd = A.ROOT / "campaign_v140L1" / "arch_sweep_preds"
    for k, v in preds.items():
        np.save(pd / f"aug_{k}_loop16k.npy", v)
    for k in ("t4_clean", "t4_aug"):
        res[k] = round(W.fid(preds[k].astype(int), ye), 3)
    print(f"[t4] clean {res['t4_clean']}  aug {res['t4_aug']}  (loop16k)", flush=True)
    res["paired_vs_t4"] = {}
    for k in preds:
        if k.startswith("t4"):
            continue
        for ref in ("t4_clean", "t4_aug"):
            b, c, pv = A.mcnemar(preds[k] == ye, preds[ref] == ye)
            d = W.fid(preds[k].astype(int), ye) - res[ref]
            res["paired_vs_t4"][f"{k}_vs_{ref}"] = dict(delta=round(d, 3), b=b, c=c, p=pv)
            print(f"  {k:>12} vs {ref}: {d:+.3f} pts  b={b} c={c}  p={pv:.2g}", flush=True)
    out_path.write_text(json.dumps(res, indent=1))
    print(f"wrote {out_path} ({time.time() - t0:.0f}s)")

if __name__ == "__main__":
    main()
