"""R10 step 4b: statistics. Balanced fidelity (19,400-row headline = rows 1000:1600 removed; full 20k
footnote), 95% CIs, exact McNemar and paired bootstrap for the differences.

Default: the R10 set.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent

def fid(ok, y):
    return 100 * 0.5 * (ok[y == 0].mean() + ok[y == 1].mean())

def ci_fid(ok, y, z=1.96):
    """normal approximation on the two per-class binomials (n large)."""
    a0, a1 = ok[y == 0].mean(), ok[y == 1].mean()
    n0, n1 = (y == 0).sum(), (y == 1).sum()
    se = 0.5 * np.sqrt(a0 * (1 - a0) / n0 + a1 * (1 - a1) / n1)
    f = 0.5 * (a0 + a1)
    return 100 * (f - z * se), 100 * (f + z * se)

def paired(okA, okB, y, nboot=10000, seed=20261020):
    b = int((okA & ~okB).sum()); c = int((okB & ~okA).sum())
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    rng = np.random.default_rng(seed)
    i0, i1 = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    d = np.empty(nboot)
    for t in range(nboot):  # stratified paired bootstrap
        j0 = rng.choice(i0, len(i0)); j1 = rng.choice(i1, len(i1))
        d[t] = 50 * ((okA[j0].mean() - okB[j0].mean()) + (okA[j1].mean() - okB[j1].mean()))
    return dict(delta=fid(okA, y) - fid(okB, y), ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                A_only=b, B_only=c, p_exact=float(p))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--pairs", nargs="*", default=None)
    ap.add_argument("--out", default="r10_stats.json")
    a = ap.parse_args()
    files = a.files or ["fpe_new_fir_delaymap_fir.npz:new FIR model (delay map)",
                        "fpe_current_fir_model_fir.npz:current FIR model",
                        "fpe_boxcar_model_box.npz:deployed boxcar path"]
    pairs = a.pairs or ["0,1", "0,2", "2,1"]
    D = []
    for f in files:
        path, lab = f.split(":", 1)
        z = np.load(HERE / path)
        D.append((lab, z["rows"], z["y"].astype(int), (z["p_fixed"] > 0.5).astype(int), (z["p_float"] > 0.5).astype(int)))
    rows, y = D[0][1], D[0][2]
    for d in D:
        assert np.array_equal(d[1], rows) and np.array_equal(d[2], y)
    m19 = (rows < 1000) | (rows >= 1600)
    out = dict(models={}, pairs={})
    for sel_name, sel in [("19400", m19), ("20000", np.ones_like(m19))]:
        ys = y[sel]
        print(f"=== n = {sel.sum()} ({'rows 1000:1600 excluded' if sel_name == '19400' else 'full 20k'}) ===")
        for lab, _, _, pf, pfl in D:
            ok = pf[sel] == ys
            lo, hi = ci_fid(ok, ys)
            r = dict(fid=fid(ok, ys), ci=[lo, hi], p10=100 * np.mean(pf[sel][ys == 0] != 0),
                     p01=100 * np.mean(pf[sel][ys == 1] != 1), float_fid=fid(pfl[sel] == ys, ys), n=int(sel.sum()))
            out["models"].setdefault(lab, {})[sel_name] = r
            print(f"  {lab:34s} {r['fid']:.2f}% [{lo:.2f}, {hi:.2f}]  P(1|0) {r['p10']:.2f}%  P(0|1) {r['p01']:.2f}%"
                  f"  (float {r['float_fid']:.2f}%)")
        for pr in pairs:
            i, j = (int(v) for v in pr.split(","))
            okA, okB = D[i][3][sel] == ys, D[j][3][sel] == ys
            r = paired(okA, okB, ys)
            out["pairs"].setdefault(f"{D[i][0]} vs {D[j][0]}", {})[sel_name] = r
            print(f"  {D[i][0]} - {D[j][0]}: {r['delta']:+.2f} pts, paired bootstrap 95% [{r['ci'][0]:+.2f}, "
                  f"{r['ci'][1]:+.2f}], discordant {r['A_only']} vs {r['B_only']}, exact McNemar p = {r['p_exact']:.3f}")
    json.dump(out, open(HERE / a.out, "w"), indent=1)
    print(f"saved {a.out}")

if __name__ == "__main__":
    main()
