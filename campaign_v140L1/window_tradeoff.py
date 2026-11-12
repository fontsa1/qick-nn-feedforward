"""Where does shortening the window start costing too much fidelity per ns of
latency saved? Fine-grid follow-up to window_sweep.py (same data, same LDA).

Window [0, E) for E on a 32-sample grid from 770 down to 384. Per E: LDA
predictions on the 20k acquired loopback shots (window_sweep.linear's
recipe: pooled covariance + ridge chosen on the training pool's validation
slice). Then, on the SAME shots (paired):
  vs640     fidelity change against E = 640 (the free cut), McNemar p
  slope     fidelity lost per 100 ns saved between adjacent grid points, with
            a 95% bootstrap CI (2000 shot resamples, paired)
Time axes: our loopback (552.96 MS/s, 1.808 ns/sample) and the source qubit's
(3.25 ns/sample: 770 samples = the 2.5 us readout of the reference dataset).
--net E1,E2 retrains the ternary 1540-4-1 recipe at those E (window_sweep.train_net).

Writes campaign_v140L1/window_tradeoff.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis_scripts" / "analysis"))
sys.path.insert(0, str(ROOT / "campaign_v140L1"))
import task33_three_way_comparison as T  # noqa: E402
import window_sweep as W                 # noqa: E402
import arch_sweep as A                   # noqa: E402

GRID = (770, 736, 704, 672, 640, 608, 576, 544, 512, 480, 448, 416, 384)
NS_LOOP, NS_QUBIT = 1e3 / W.FS_MHZ, 3.25

def lda_pred(Xtr, ytr, Xva, yva, Xev, E):
    c = W.cols(0, E)
    A_, B = Xtr[:, c].astype(np.float64), Xva[:, c].astype(np.float64)
    m0, m1 = A_[ytr == 0].mean(0), A_[ytr == 1].mean(0)
    Ac = A_ - np.where(ytr[:, None] == 1, m1, m0)
    S = (Ac.T @ Ac) / len(Ac)
    best = None
    for lam in W.RIDGES:
        wl = np.linalg.solve(S + lam * np.trace(S) / len(S) * np.eye(len(S)), m1 - m0)
        t = W.best_thr(A_ @ wl, ytr)
        fv = W.fid((B @ wl > t).astype(int), yva)
        if best is None or fv > best[0]:
            best = (fv, wl, t)
    _, wl, t = best
    return Xev[:, c] @ wl > t

def fid_idx(p, y, idx):
    pi, yi = p[idx], y[idx]
    return 100 * (1 - 0.5 * (np.mean(pi[yi == 0]) + np.mean(~pi[yi == 1])))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="")
    a = ap.parse_args()
    out = ROOT / "campaign_v140L1" / "window_tradeoff.json"
    res = json.loads(out.read_text()) if out.exists() else {}
    t0 = time.time()
    k2 = complex(np.load(ROOT / "calibration/k2_loopback.npy"))
    c2 = complex(np.load(ROOT / "calibration/c2_loopback.npy"))
    X = np.load(ROOT / "0528_X_train_0_770.npy", mmap_mode="r")[:W.N_TRAIN]
    X = T.hw_raw_transform(T.apply_affine(np.asarray(X).astype(np.float32), k2, c2))
    y = np.load(ROOT / "0528_y_train_0_770.npy")[:W.N_TRAIN].astype(int)
    Xtr, ytr, Xva, yva = X[:-W.N_VAL], y[:-W.N_VAL], X[-W.N_VAL:], y[-W.N_VAL:]
    Xev = T.hw_raw_transform(np.load(ROOT / "data/loopback20k_real.npy").astype(np.float64)).astype(np.float64)
    yev = np.load(ROOT / "data/loopback20k_real_y.npy").astype(int)
    print(f"loaded ({time.time() - t0:.0f}s)", flush=True)

    if "lda" not in res:
        P = {}
        for E in GRID:
            P[E] = lda_pred(Xtr, ytr, Xva, yva, Xev, E)
            print(f"  E={E}: {W.fid(P[E].astype(int), yev):.3f}%", flush=True)
        rng = np.random.default_rng(0)
        boots = [rng.integers(0, len(yev), len(yev)) for _ in range(2000)]
        rows = []
        for i, E in enumerate(GRID):
            f = W.fid(P[E].astype(int), yev)
            b, c, p = A.mcnemar(P[E] == yev, P[640] == yev)
            row = dict(E=E, ns_loop=round(E * NS_LOOP, 1), ns_qubit=round(E * NS_QUBIT, 1),
                       lda=round(f, 3), vs640=round(f - W.fid(P[640].astype(int), yev), 3), p640=p)
            if i > 0:
                Ep = GRID[i - 1]
                d = [fid_idx(P[Ep], yev, ix) - fid_idx(P[E], yev, ix) for ix in boots]
                dns = (Ep - E) * NS_LOOP / 100
                lo, hi = np.percentile(d, [2.5, 97.5])
                row.update(loss_from_prev=round(W.fid(P[Ep].astype(int), yev) - f, 3),
                           slope_per100ns_loop=round((W.fid(P[Ep].astype(int), yev) - f) / dns, 3),
                           slope_ci=[round(lo / dns, 3), round(hi / dns, 3)])
            rows.append(row)
        res["lda"] = rows
        out.write_text(json.dumps(res, indent=1))
    print(f"\n{'E':>4} {'loop ns':>8} {'qubit ns':>9} {'LDA %':>7} {'vs 640':>7} {'p':>8} "
          f"{'loss/100ns (loop)':>18} {'95% CI':>16}")
    for r in res["lda"]:
        s = (f"{r['slope_per100ns_loop']:+18.3f} [{r['slope_ci'][0]:+.2f},{r['slope_ci'][1]:+.2f}]"
             if "slope_ci" in r else "")
        print(f"{r['E']:4d} {r['ns_loop']:8.1f} {r['ns_qubit']:9.1f} {r['lda']:7.3f} {r['vs640']:+7.3f} "
              f"{r['p640']:8.2g} {s}")

    if a.net:
        res.setdefault("net", {})
        for E in (int(v) for v in a.net.split(",")):
            t1 = time.time()
            f = W.train_net(X, y, Xev, yev, 0, E)
            res["net"][str(E)] = round(f, 3)
            out.write_text(json.dumps(res, indent=1))
            print(f"  net E={E}: {f:.3f}%  ({time.time() - t1:.0f}s)", flush=True)
    print(f"wrote {out} ({time.time() - t0:.0f}s)")

if __name__ == "__main__":
    main()
