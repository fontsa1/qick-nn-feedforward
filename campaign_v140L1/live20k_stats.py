"""Statistics of the 20k-shot NN_live capture (boxcar path): Wilson CI and McNemar, as for NN_axi's 20k.

Inputs: box_live_capture.py chunks (rows of test20k_real_X) and the emulator file
check_live_bitexact.py --box --emu-out writes for the merged capture.

Reports: window == boxcar buffer; bit-exact (sums, sigmoid, rule) counts; chip
fidelity (balanced, as everywhere in this project) with a Wilson 95% CI on
accuracy; chip vs emulator McNemar; and NN_live (576-sample boxcar window) vs
NN_axi v1.40 (770-sample FIR window, campaign_v140/aligned) on identical rows.
Writes <capture>.stats.json.
"""
import glob
import json
import sys
from math import erf, sqrt
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent

def wilson(k, n, z=1.96):
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    hw = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - hw), 100 * (c + hw)

def fid(pred, y):
    return 100 * (1 - 0.5 * (np.mean(pred[y == 0] != 0) + np.mean(pred[y == 1] != 1)))

def mcnemar(a_ok, b_ok):
    b = int((a_ok & ~b_ok).sum()); c = int((b_ok & ~a_ok).sum())
    z = (b - c) / sqrt(b + c) if b + c else 0.0
    return b, c, z, 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))

def merge(out, parts):
    zs = [np.load(HERE / p if not Path(p).is_absolute() else p) for p in parts]
    keys = ["live_words", "live_rec", "live_hdr", "y", "rows", "box", "fir", "window_eq_buffer"]
    d = {k: np.concatenate([z[k] for z in zs]) for k in keys}
    o = np.argsort(d["rows"])
    d = {k: v[o] for k, v in d.items()}
    assert len(np.unique(d["rows"])) == len(d["rows"]), "overlapping chunks"
    d["live_n"] = zs[0]["live_n"]
    np.savez(HERE / out, **d)
    print(f"merged {len(parts)} chunks, {len(d['rows'])} shots (rows {d['rows'][0]}..{d['rows'][-1]}) -> {out}")

def axi_v140():
    idx, p, y = [], [], []
    for f in glob.glob(str(ROOT / "campaign_v140" / "aligned" / "board*_*.npz")):
        z = np.load(f, allow_pickle=True)
        idx.append(z["idx"]); p.append(z["probs"]); y.append(z["y"])
    idx = np.concatenate(idx); o = np.argsort(idx)
    return idx[o], np.concatenate(p)[o], np.concatenate(y)[o]

def stats(cap, emu):
    d, e = np.load(cap), np.load(emu)
    rows, y = d["rows"], d["y"].astype(int)
    n = len(rows)
    assert np.array_equal(e["y"].astype(int), y) and len(e["p_emu"]) == n
    dec = (d["live_rec"][:, 7] & 1).astype(int)
    pe = e["p_emu"]
    ok_chip, ok_emu = dec == y, (pe > 0.5).astype(int) == y
    exact = e["ok_acc"] & e["ok_sigmoid"] & e["ok_rule"]
    lo, hi = wilson(int(ok_chip.sum()), n)
    b, c, z, pv = mcnemar(ok_emu, ok_chip)
    hdr = d["live_hdr"]
    r = dict(n=int(n), rows=[int(rows[0]), int(rows[-1])],
             window_eq_buffer=int(d["window_eq_buffer"].sum()),
             bit_exact=dict(all=int(exact.sum()), sums=int(e["ok_acc"].sum()),
                            sigmoid=int(e["ok_sigmoid"].sum()), rule=int(e["ok_rule"].sum())),
             frame_errors=[int(hdr[0, 1]), int(hdr[-1, 1])],
             chip_fid=fid(dec, y), chip_acc=100 * ok_chip.mean(), acc_ci=[lo, hi],
             p_ge=100 * np.mean(dec[y == 0] != 0), p_eg=100 * np.mean(dec[y == 1] != 1),
             emu_fid=fid((pe > 0.5).astype(int), y), emu_only=b, chip_only=c, p_vs_emu=pv)
    print(f"{n} shots, rows {rows[0]}..{rows[-1]}")
    print(f"window == boxcar buffer: {r['window_eq_buffer']}/{n}; frame errors {r['frame_errors'][0]} -> {r['frame_errors'][1]}")
    print(f"bit-exact: all {r['bit_exact']['all']}/{n} (sums {r['bit_exact']['sums']}, sigmoid "
          f"{r['bit_exact']['sigmoid']}, rule {r['bit_exact']['rule']})")
    print(f"NN_live on chip: fidelity {r['chip_fid']:.2f}%  (accuracy {r['chip_acc']:.2f}%, Wilson 95% "
          f"[{lo:.2f}, {hi:.2f}]; P(e|g) {r['p_ge']:.2f}%, P(g|e) {r['p_eg']:.2f}%)")
    print(f"emulator on the same windows: {r['emu_fid']:.2f}%; McNemar chip vs emulator {b} vs {c}, p={pv:.3g}")
    ia, pa, ya = axi_v140()
    common, i_l, i_a = np.intersect1d(rows, ia, return_indices=True)
    if len(common):
        assert np.array_equal(ya[i_a], y[i_l])
        ok_axi = (pa[i_a] > 0.5).astype(int) == ya[i_a]
        b2, c2, z2, pv2 = mcnemar(ok_chip[i_l], ok_axi)
        r["vs_nn_axi_v140"] = dict(n=int(len(common)), live_fid=fid(dec[i_l], y[i_l]),
                                   axi_fid=fid((pa[i_a] > 0.5).astype(int), ya[i_a]),
                                   live_only=b2, axi_only=c2, z=z2, p=pv2)
        print(f"vs NN_axi v1.40 on {len(common)} identical rows: NN_live {r['vs_nn_axi_v140']['live_fid']:.2f}% vs "
              f"NN_axi {r['vs_nn_axi_v140']['axi_fid']:.2f}%; live-only-right {b2}, axi-only-right {c2}, "
              f"McNemar z={z2:.2f} p={pv2:.3g}")
    out = Path(str(cap).replace(".npz", ".stats.json"))
    out.write_text(json.dumps(r, indent=2, default=float))
    print("saved", out)

if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "--merge":
        merge(a[1], a[2:])
    else:
        stats(a[0], a[1])
