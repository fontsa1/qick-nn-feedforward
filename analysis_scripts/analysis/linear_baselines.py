"""Fair linear baselines on the deployment eval set (loopback20k_real).

Trained on the SAME data as the networks (real training pool through the
label-free domain map, float32 path), scored on the 20,000 acquired shots,
threshold chosen on the training data (never on the eval set). Reports:
  MF   matched filter: w = mu1 - mu0 (complex), real part of <z, w>
  LDA  Fisher discriminant with full 1540x1540 pooled covariance (+ ridge)
and, for reference, an MF fitted on half of the eval set and scored on the other half.
"""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis_scripts" / "analysis"))
import task33_three_way_comparison as T

k2 = complex(np.load(ROOT / "calibration/k2_loopback.npy")); c2 = complex(np.load(ROOT / "calibration/c2_loopback.npy"))
Xev = T.hw_raw_transform(np.load(ROOT / "data/loopback20k_real.npy").astype(np.float64)).astype(np.float64)
yev = np.load(ROOT / "data/loopback20k_real_y.npy").astype(int)

def fid(pred, y):
    return 100 * (1 - 0.5 * (np.mean(pred[y == 0] != 0) + np.mean(pred[y == 1] != 1)))

def best_thr(s, y):
    o = np.sort(s); cand = o[::max(1, len(o) // 4000)]
    f = [fid((s > t).astype(int), y) for t in cand]
    return cand[int(np.argmax(f))]

for N in (10000, 50000, 200000):
    X = np.load(ROOT / "0528_X_train_0_770.npy", mmap_mode="r")[:N]
    X = T.hw_raw_transform(T.apply_affine(np.asarray(X).astype(np.float32), k2, c2)).astype(np.float64)
    y = np.load(ROOT / "0528_y_train_0_770.npy")[:N].astype(int)
    m0, m1 = X[y == 0].mean(0), X[y == 1].mean(0)
    w = m1 - m0                      # real-valued over interleaved I,Q == Re<z, mu1-mu0>
    s_tr = X @ w; t = best_thr(s_tr, y)
    f_mf = fid((Xev @ w > t).astype(int), yev)
    Xc = np.concatenate([X[y == 0] - m0, X[y == 1] - m1])
    S = (Xc.T @ Xc) / len(Xc)
    out = []
    for lam in (1e-3, 1e-2, 1e-1):
        reg = S + lam * np.trace(S) / S.shape[0] * np.eye(S.shape[0])
        wl = np.linalg.solve(reg, w)
        tl = best_thr(X @ wl, y)
        out.append((lam, fid((Xev @ wl > tl).astype(int), yev)))
    print(f"N={N:6d}: MF {f_mf:.2f}%   LDA " + "  ".join(f"(ridge {l:g}) {f:.2f}%" for l, f in out), flush=True)

# MF fitted on half of the eval set
n = len(yev) // 2
m0, m1 = Xev[:n][yev[:n] == 0].mean(0), Xev[:n][yev[:n] == 1].mean(0); w = m1 - m0
s = Xev @ w; t = 0.5 * (m0 @ w + m1 @ w)
print(f"half-split (MF fit on eval[:10000], scored on eval[10000:]): {fid((s[n:] > t).astype(int), yev[n:]):.2f}%")
