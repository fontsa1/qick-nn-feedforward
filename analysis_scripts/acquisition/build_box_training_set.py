"""Domain map + training set for the boxcar readout path (task 4.1h, v1.40L7 -> L8).

The deployed model (v1.40L3, 576 samples) trains on `k2 * z + c2`: the reference
rows mapped by a label-free affine fit to what readout 0's FIR stream delivers in
loopback (build_domain_calibrated_training_set.py). The boxcar path (boxcar_tap.v)
replaces the FIR with box8 (an average over one decimated sample) and a moving sum
of M samples, so its domain map is

    box[n] ~= kb * MA_M(z)[n - s] + cb

MA_M is the causal M-sample moving average at the 552.96 MHz sample rate (the
row's samples are at that rate: 770 samples = the 1.393 us readout). The shift s
(fractional, 0.25-sample grid) absorbs box8's half-sample delay and whatever
trigger alignment remains after run_loopback_box.py's --box-advance-us. Before
the waveform the DAC is silent, i.e. the reference value c (the calibration
offset), so the filter is primed with c.

The fit is LABEL-FREE (mean trajectory over all calibration shots, lstsq for kb,
cb; grid search for s), on the same rows as the deployed map: test20k_real_X
[1000:1600]. It writes the mapped training rows (default the v1.40L3 recipe's
0:10000,110000:300000 of 0528_X_train_0_770) to --out, in row order, so train
with --rows 0:<n>.

"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

def ma(z, M, prime):
    """Causal M-sample moving average along axis 1, primed with `prime` before n = 0."""
    pad = np.concatenate([np.full(z.shape[:-1] + (M - 1,), prime, dtype=z.dtype), z], axis=-1)
    cs = np.cumsum(pad, axis=-1)
    cs = np.concatenate([np.zeros(z.shape[:-1] + (1,), dtype=z.dtype), cs], axis=-1)
    return (cs[..., M:] - cs[..., :-M]) / M

def fshift(z, s):
    """z[n - s] by linear interpolation (edge-held), along the last axis."""
    n = np.arange(z.shape[-1], dtype=float)
    src = np.clip(n - s, 0, z.shape[-1] - 1)
    i0 = np.floor(src).astype(int); i1 = np.minimum(i0 + 1, z.shape[-1] - 1); f = src - i0
    return z[..., i0] * (1 - f) + z[..., i1] * f

def cz(X):
    X = np.asarray(X, dtype=np.float64)
    return X[..., 0::2] + 1j * X[..., 1::2]

def fit(zr_mean, za_mean, M, prime, lo=8, hi=760, shifts=np.arange(-12, 12.01, 0.25)):
    best = None
    f = ma(zr_mean[None], M, prime)[0]
    for s in shifts:
        g = fshift(f, s)[lo:hi]
        A = np.vstack([g, np.ones_like(g)]).T
        sol, *_ = np.linalg.lstsq(A, za_mean[lo:hi], rcond=None)
        r = za_mean[lo:hi] - A @ sol
        e = float(np.sqrt(np.mean(np.abs(r) ** 2)))
        if best is None or e < best[0]:
            best = (e, float(s), complex(sol[0]), complex(sol[1]))
    return best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True, nargs="+", help="run_loopback_box.py npz file(s) of test20k_real_X")
    ap.add_argument("--m", type=int, required=True, help="the moving-sum length the captures were taken with")
    ap.add_argument("--calib-rows", default="1000:1600")
    ap.add_argument("--rows", default="0:10000,110000:300000")
    ap.add_argument("--out", default="0528_X_train_box_calibrated.npy")
    ap.add_argument("--fit-only", action="store_true")
    ap.add_argument("--host", default="10.1.9.26", help="board whose calibration c primes the filter (checkpoint input only)")
    a = ap.parse_args()

    caps = []
    for f in a.capture:
        c = dict(np.load(ROOT / f))
        if "rows" not in c:                      # a run_loopback_box.py checkpoint (.partial.npz)
            n = int(c["done"]); c["rows"] = int(c["start"]) + np.arange(n)
            c["box"], c["fir"] = c["box"][:n], c["fir"][:n]
        caps.append(c)
    rows = np.concatenate([c["rows"] for c in caps])
    box = np.concatenate([c["box"] for c in caps]); fir = np.concatenate([c["fir"] for c in caps])
    lo, hi = (int(v) for v in a.calib_rows.split(":"))
    sel = (rows >= lo) & (rows < hi)
    print(f"calibration shots: {sel.sum()} (dataset rows [{lo}:{hi}])")
    Xref = np.load(ROOT / "data/test20k_real_X.npy", mmap_mode="r")
    zr = cz(Xref[rows[sel]]); za = cz(box[sel]); zf = cz(fir[sel])
    # silence before the waveform: the reference value whose predistorted DAC sample is 0
    mf = ROOT / a.capture[0].replace(".partial.npz", ".npz").replace(".npz", ".json")
    if mf.exists():
        c_dac = complex(*json.load(open(mf))["c"])
    else:                                        # checkpoint: the board's calibration offset
        import sys; sys.path.insert(0, str(ROOT / "board_tools")); import calib
        c_dac = calib.load_kc(host=a.host, strict=False)[1]
    print(f"silence level (calibration c): {c_dac}")
    zr_m, za_m, zf_m = zr.mean(0), za.mean(0), zf.mean(0)

    res = {}
    for M in sorted({1, a.m}):
        e, s, kb, cb = fit(zr_m, za_m, M, c_dac)
        res[M] = dict(rms=e, shift=s, k=[kb.real, kb.imag], c=[cb.real, cb.imag])
        print(f"boxcar map, MA_{M:<2d}: s = {s:+.2f} samples  |k| = {abs(kb):.4f} arg {np.degrees(np.angle(kb)):+.1f} deg"
              f"  rms resid {e:.2f} (level {np.abs(za_m).mean():.1f})")
    # the FIR path's own map (identity filter) for reference
    e, s, kf, cf = fit(zr_m, zf_m, 1, c_dac, shifts=np.arange(-3, 3.01, 0.25))
    print(f"FIR map (reference):   s = {s:+.2f}  |k| = {abs(kf):.4f}  rms resid {e:.2f} (level {np.abs(zf_m).mean():.1f})")
    M = a.m
    kb = complex(*res[M]["k"]); cb = complex(*res[M]["c"]); s = res[M]["shift"]

    # per-shot residual: how much of each acquired shot the map explains
    pred = kb * fshift(ma(zr, M, c_dac), s) + cb
    r = za - pred
    print(f"per-shot: resid rms {np.sqrt(np.mean(np.abs(r[:, 8:760]) ** 2)):.2f}, "
          f"signal rms {np.sqrt(np.mean(np.abs(za[:, 8:760] - za[:, 8:760].mean()) ** 2)):.2f}")

    meta = dict(m=M, shift=s, k=[kb.real, kb.imag], c=[cb.real, cb.imag], prime=[c_dac.real, c_dac.imag],
                calib_rows=a.calib_rows, captures=a.capture, fits={str(k): v for k, v in res.items()})
    (ROOT / "calibration/box_domain_map.json").write_text(json.dumps(meta, indent=1))
    print("saved calibration/box_domain_map.json")
    if a.fit_only:
        return

    ranges = [tuple(int(v) for v in r.split(":")) for r in a.rows.split(",")]
    Xtr = np.load(ROOT / "0528_X_train_0_770.npy", mmap_mode="r")
    n = sum(h - l for l, h in ranges)
    out = np.lib.format.open_memmap(ROOT / a.out, mode="w+", dtype=np.float32, shape=(n, Xtr.shape[1]))
    k = 0
    for l, h in ranges:
        for b in range(l, h, 20000):
            e_ = min(h, b + 20000)
            z = kb * fshift(ma(cz(Xtr[b:e_]), M, c_dac), s) + cb
            out[k:k + (e_ - b), 0::2] = z.real; out[k:k + (e_ - b), 1::2] = z.imag
            k += e_ - b
    out.flush()
    src_rows = np.concatenate([np.arange(l, h) for l, h in ranges])
    np.save(ROOT / (Path(a.out).stem + "_rows.npy"), src_rows)
    print(f"saved {a.out} ({n} rows = {ranges}); train with --xcal {a.out} --rows 0:{n}")

if __name__ == "__main__":
    main()
