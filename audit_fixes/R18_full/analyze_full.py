"""In-loop fidelity on the full test split, both boards: R18's 19,400 rows (rows 0-19,999 minus the domain-map
rows) plus the 80,000 further rows (rows 20,000-99,999, both passes).

Per board: in-loop F on the 99,400 and on the fresh 80k; loop failures (stale, late, unsafe, pulse != decision,
binding rejects); late-rate 95% upper bound; paired against the same board's gate-idle NN_live on identical
rows (A: campaign_v140L1/A_v141_100k_merged_0_50000 + A_v141_100k_swap_merged_50000_100000; B: the swap run's
0-50k + B_v141_100k_merged_50000_100000); loop latency over passed shots (median, order-statistic 95% CI);
blanked-shot plateau / pass plateau (R19 input). Boards A vs B paired on identical rows.
Each run's in-loop answer uses that run's own always-pass plateau (calibrated at its start).

Writes audit_fixes/R18_full/inloop_full_stats.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "audit_fixes" / "R3_R2"))
from r3_stats import fstats, paired  # noqa: E402

RUNS = {
    "A": ["audit_fixes/R18/A_v141_inloop_20k.npz", "audit_fixes/R18_full/A_v141_inloop_20000_60000.npz",
          "audit_fixes/R18_full/A_v141_inloop_60000_100000.npz"],
    "B": ["audit_fixes/R18/B_v141_inloop_20k.npz", "audit_fixes/R18_full/B_v141_inloop_60000_100000.npz",
          "audit_fixes/R18_full/B_v141_inloop_20000_60000.npz"],
}
IDLE = {
    "A": ["campaign_v140L1/A_v141_100k_merged_0_50000.npz", "campaign_v140L1/A_v141_100k_swap_merged_50000_100000.npz"],
    "B": ["campaign_v140L1/B_v141_100k_swap_merged_0_50000.npz", "campaign_v140L1/B_v141_100k_merged_50000_100000.npz"],
}
KEYS = ("rows", "y", "gate", "plat", "dec", "late", "L_buf", "n_clip", "fresh")

def load_runs(files):
    parts, meta = [], []
    for f in files:
        z = np.load(ROOT / f, allow_pickle=True)
        n = int(z["n_done"])
        d = {k: z[k][:n] for k in KEYS}
        d["came"] = d["plat"] > 0.5 * float(z["plat_pass"])
        d["plat_frac"] = d["plat"] / float(z["plat_pass"])
        d["rej"] = int(d["gate"][-1, 19] - d["gate"][0, 19]) if n > 1 else 0
        meta.append(dict(file=f, n=n, rows=[int(d["rows"][0]), int(d["rows"][-1])], plat_pass=float(z["plat_pass"]),
                         delay=int(z["delay"]), slack_us=float(z["slack"]), k=[float(np.real(z["k"])), float(np.imag(z["k"]))]))
        parts.append(d)
    out = {k: np.concatenate([p[k] for p in parts]) for k in KEYS + ("came", "plat_frac")}
    out["rej"] = sum(p["rej"] for p in parts)
    o = np.argsort(out["rows"])
    for k in KEYS + ("came", "plat_frac"):
        out[k] = out[k][o]
    assert len(np.unique(out["rows"])) == len(out["rows"])
    return out, meta

def idle_dec(files, rows):
    r, d = [], []
    for f in files:
        z = np.load(ROOT / f, allow_pickle=True)
        r.append(z["rows"]); d.append((z["live_rec"][:, 7] & 1).astype(int))
    r = np.concatenate(r); d = np.concatenate(d); o = np.argsort(r); r, d = r[o], d[o]
    j = np.searchsorted(r, rows)
    assert np.array_equal(r[j], rows)
    return d[j]

def median_ci(x):
    x = np.sort(np.asarray(x, float)); n = len(x)
    lo, hi = int(np.floor(n / 2 - 1.96 * np.sqrt(n) / 2)), int(np.ceil(n / 2 + 1.96 * np.sqrt(n) / 2))
    return [float(np.median(x)), float(x[max(lo, 0)]), float(x[min(hi, n - 1)])]

res, keep = {}, {}
for b in "AB":
    d, meta = load_runs(RUNS[b])
    rows, y = d["rows"], d["y"].astype(int)
    assert len(rows) == 99400 and not np.isin(rows, np.arange(1000, 1600)).any()
    flags = d["gate"][:, 10]
    passed = (flags & 1) > 0; stale = (flags & 2) > 0; gbit = ((flags >> 2) & 1).astype(int)
    came, dec = d["came"], d["dec"].astype(int)
    correct = (came == passed) & ~stale & (gbit == dec)
    unsafe = came & ~(~stale & (gbit == dec) & (dec == 1))
    pred = came.astype(int)
    idle = idle_dec(IDLE[b], rows)
    fresh = rows >= 20000
    nlate = int(d["late"].sum())
    pc = passed & came
    bl = ~passed & ~stale
    res[b] = dict(
        runs=meta, n=int(len(rows)),
        inloop=fstats(pred, y), inloop_fresh80k=fstats(pred[fresh], y[fresh]),
        decision=fstats(dec, y), gate_idle=fstats(idle, y),
        inloop_vs_gate_idle=paired(pred, idle, y),
        inloop_vs_gate_idle_fresh80k=paired(pred[fresh], idle[fresh], y[fresh]),
        loop=dict(correct=int(correct.sum()), not_correct=int((~correct).sum()), stale=int(stale.sum()), late=nlate,
                  unsafe=int(unsafe.sum()), pulse_ne_decision=int((pred != dec).sum()), binding_rejects=int(d["rej"]),
                  late_rate_upper95=float(1 - 0.05 ** (1 / len(rows))) if nlate == 0 else None,
                  dac_clipped_shots=int((d["n_clip"] > 0).sum())),
        L_buf_passed=dict(n=int(pc.sum()), median_ci=median_ci(d["L_buf"][pc]),
                          range=[float(d["L_buf"][pc].min()), float(d["L_buf"][pc].max())]),
        blanked_plateau_frac=dict(n=int(bl.sum()), median=float(np.median(d["plat_frac"][bl])),
                                  p99=float(np.percentile(d["plat_frac"][bl], 99)), max=float(d["plat_frac"][bl].max())),
    )
    keep[b] = (rows, y, pred)
    F, P = res[b]["inloop"], res[b]["inloop_vs_gate_idle"]
    print(f"== board {b}: n = {len(rows)}  in-loop F {F['fid']:.2f}% [{F['fid_ci'][0]:.2f}, {F['fid_ci'][1]:.2f}]  "
          f"P(1|0) {F['p10']:.2f} P(0|1) {F['p01']:.2f} | fresh 80k {res[b]['inloop_fresh80k']['fid']:.2f}% | "
          f"gate idle {res[b]['gate_idle']['fid']:.2f}%")
    print(f"   vs gate idle: {P['A_only']} vs {P['B_only']}, p = {P['p_exact']:.3f}, dF {P['dF']:+.3f} "
          f"[{P['dF_ci'][0]:+.3f}, {P['dF_ci'][1]:+.3f}]")
    print(f"   loop: {res[b]['loop']}")
    print(f"   L_buf passed: {res[b]['L_buf_passed']}")
    print(f"   blanked plateau: {res[b]['blanked_plateau_frac']}")

ra, ya, pa = keep["A"]; rb, yb, pb = keep["B"]
assert np.array_equal(ra, rb) and np.array_equal(ya, yb)
res["A_vs_B"] = paired(pa, pb, ya); res["A_vs_B"]["n"] = int(len(ra))
pooled_y = np.concatenate([ya, yb]); pooled_p = np.concatenate([pa, pb])
res["pooled_both_boards"] = fstats(pooled_p, pooled_y)
P = res["A_vs_B"]
print(f"== A vs B, in loop, {len(ra)} identical traces: {P['A_only']} vs {P['B_only']}, p = {P['p_exact']:.3f}, "
      f"dF {P['dF']:+.3f} [{P['dF_ci'][0]:+.3f}, {P['dF_ci'][1]:+.3f}]")
(HERE / "inloop_full_stats.json").write_text(json.dumps(res, indent=1, default=float))
print("wrote", HERE / "inloop_full_stats.json")
