"""v1.44 copy of audit_fixes/R3_R2/r3_linear.py: same training and recipes (classifiers fixed, trained once), scored on the
v1.44 NN_axi replay of this board instead of loopback20k_real.

Original notes follow.
R3: the linear-baseline numbers that are scored on loopback20k_real (the replayed 20k
test subset), recomputed with per-shot predictions so they can be reported on the
19,400-row headline set (rows 1000:1600 excluded) and on the full 20k.

Two recipes, reproduced as they are in the repo (no protocol change here; R6 fixes
the ridge selection):
  (a) analysis_scripts/analysis/linear_baselines.py  (PAPER §8.3 table, §3.5, §14.1):
      train rows [0:N) of 0528_X_train_0_770 through (k2, c2), N = 10k/50k/200k;
      MF w = mu1 - mu0; LDA pooled covariance + ridge lam*tr(S)/d*I, lam in
      {1e-3, 1e-2, 1e-1}; threshold chosen on the training scores. The quoted LDA
      number is the best ridge ON THE TEST SET (audit A10/B1) -- reported per ridge here.
  (b) campaign_v140L1/window_tradeoff.py LDA arm (PAPER §8.4, §7.4 "95.55% replayed"):
      train rows [0:180k), ridge in {1e-4..1e-1} chosen on validation rows [180k:200k),
      window [0, E) for E on the 32-sample grid; paired against E = 640.

Memory-light: the training pool is held once as float32 (1.2 GB); class moments are
accumulated in float64 in 10k-row chunks; the window-E covariance is the leading
2E x 2E block of the full one (window = first E complex samples).

Writes r3_linear.json and r3_linear_preds.npz (per-shot decisions) in this folder.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BOARD = sys.argv[1]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "audit_fixes" / "R3_R2"))
sys.path.insert(0, str(ROOT / "analysis_scripts" / "analysis"))
sys.path.insert(0, str(OUT))
import task33_three_way_comparison as T  # noqa: E402
from r3_stats import fstats, paired, calmask  # noqa: E402

GRID = (770, 736, 704, 672, 640, 608, 576, 544, 512, 480, 448, 416, 384)
RIDGES_TRADE = (1e-4, 1e-3, 1e-2, 1e-1)
RIDGES_LB = (1e-3, 1e-2, 1e-1)
CH = 10000

def fid(pred, y):
    return 100 * (1 - 0.5 * (np.mean(pred[y == 0] != 0) + np.mean(pred[y == 1] != 1)))

def best_thr(s, y):
    o = np.sort(s); cand = o[::max(1, len(o) // 4000)]
    f = [fid((s > t).astype(int), y) for t in cand]
    return cand[int(np.argmax(f))]

t0 = time.time()
k2 = complex(np.load(ROOT / "calibration/k2_loopback.npy")); c2 = complex(np.load(ROOT / "calibration/c2_loopback.npy"))
Xm = np.load(ROOT / "0528_X_train_0_770.npy", mmap_mode="r")
y = np.load(ROOT / "0528_y_train_0_770.npy")[:200000].astype(int)
X = np.empty((200000, Xm.shape[1]), np.float32)
for i in range(0, 200000, CH):
    X[i:i + CH] = T.hw_raw_transform(T.apply_affine(np.asarray(Xm[i:i + CH]).astype(np.float32), k2, c2))
Xev = T.hw_raw_transform(np.load(OUT / f"replay20k_{BOARD}.npy").astype(np.float64)).astype(np.float64)
yev = np.load(ROOT / "data/test20k_real_y.npy").astype(int)
print(f"loaded ({time.time()-t0:.0f}s)", flush=True)
y20 = np.load(ROOT / "data/test20k_real_y.npy").astype(int)
assert np.array_equal(y20, yev), "loopback20k_real rows are not test20k rows 0..19999 in order"
rows = np.arange(20000); hm = calmask(rows)

# cumulative per-class moments at the block boundaries we need
bounds = sorted({10000, 50000, 180000, 200000})
D = X.shape[1]
cum = {}
s1 = {0: np.zeros(D), 1: np.zeros(D)}; s2 = {0: np.zeros((D, D)), 1: np.zeros((D, D))}; nn = {0: 0, 1: 0}
for i in range(0, 200000, CH):
    A = X[i:i + CH].astype(np.float64); yy = y[i:i + CH]
    for c in (0, 1):
        Ac = A[yy == c]; s1[c] += Ac.sum(0); s2[c] += Ac.T @ Ac; nn[c] += len(Ac)
    if i + CH in bounds:
        cum[i + CH] = ({c: s1[c].copy() for c in (0, 1)}, {c: s2[c].copy() for c in (0, 1)}, dict(nn))
        print(f"  moments to {i+CH} ({time.time()-t0:.0f}s)", flush=True)
del s2

def class_stats(N, d):
    S1, S2, n = cum[N]
    m = {c: S1[c][:d] / n[c] for c in (0, 1)}
    S = sum(S2[c][:d, :d] - n[c] * np.outer(m[c], m[c]) for c in (0, 1)) / (n[0] + n[1])
    return m[0], m[1], S

def scores(lo, hi, w):
    d = len(w)
    return np.concatenate([X[i:min(i + CH, hi), :d].astype(np.float64) @ w for i in range(lo, hi, CH)])

res = {"linear_baselines": {}, "window_tradeoff": {}}
preds = {}
# (a) linear_baselines.py
for N in (10000, 50000, 200000):
    m0, m1, S = class_stats(N, D)
    w = m1 - m0
    p_mf = (Xev @ w > best_thr(scores(0, N, w), y[:N])).astype(int)
    r = {"MF": dict(h19400=fstats(p_mf[hm], yev[hm]), full=fstats(p_mf, yev))}
    preds[f"lb_mf_{N}"] = p_mf
    for lam in RIDGES_LB:
        wl = np.linalg.solve(S + lam * np.trace(S) / D * np.eye(D), w)
        p = (Xev @ wl > best_thr(scores(0, N, wl), y[:N])).astype(int)
        r[f"LDA_ridge{lam:g}"] = dict(h19400=fstats(p[hm], yev[hm]), full=fstats(p, yev))
        preds[f"lb_lda_{N}_{lam:g}"] = p
    res["linear_baselines"][str(N)] = r
    print(f"N={N}: " + "  ".join(f"{k} {v['full']['fid']:.3f}/{v['h19400']['fid']:.3f}" for k, v in r.items()),
          f"({time.time()-t0:.0f}s)", flush=True)

# (b) window_tradeoff.py LDA arm (validation-selected ridge)
NTR = 180000
P = {}
for E in GRID:
    d = 2 * E
    m0, m1, S = class_stats(NTR, d)
    # validation slice moments are not needed: scores on X[180k:200k]
    best = None
    for lam in RIDGES_TRADE:
        wl = np.linalg.solve(S + lam * np.trace(S) / d * np.eye(d), m1 - m0)
        t = best_thr(scores(0, NTR, wl), y[:NTR])
        fv = fid((scores(NTR, 200000, wl) > t).astype(int), y[NTR:200000])
        if best is None or fv > best[0]:
            best = (fv, lam, wl, t)
    fv, lam, wl, t = best
    P[E] = (Xev[:, :d] @ wl > t).astype(int)
    preds[f"wt_lda_{E}"] = P[E]
    res["window_tradeoff"][str(E)] = dict(ridge=lam, val_fid=fv, h19400=fstats(P[E][hm], yev[hm]),
                                          full=fstats(P[E], yev))
    print(f"E={E}: ridge {lam:g} val {fv:.3f}  20k {fid(P[E], yev):.3f}  19400 {fid(P[E][hm], yev[hm]):.3f}"
          f" ({time.time()-t0:.0f}s)", flush=True)
for E in GRID:
    res["window_tradeoff"][str(E)]["vs640"] = dict(h19400=paired(P[E][hm], P[640][hm], yev[hm]),
                                                   full=paired(P[E], P[640], yev))

np.savez(OUT / f"linear_preds_{BOARD}.npz", y=yev, **preds)
(OUT / f"linear_v144_{BOARD}.json").write_text(json.dumps(res, indent=1, default=float))
print(f"done ({time.time()-t0:.0f}s)")
