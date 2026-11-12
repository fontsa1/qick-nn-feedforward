"""Combine latency_protocol.py runs: per-load values, between-load spread, and the number the paper may quote.

The uncertainty on a board's L_50 is taken across loads, not across shots: the load is the unit of replication, since
shots within a load share its clock and converter alignment.

"""
import argparse
import glob
import json
import os

import numpy as np
from scipy import stats

def between(means, sems):
    """Load mean, sd of the load means, between-load sd with shot noise removed, 95% t interval over loads."""
    m, s = np.asarray(means), np.asarray(sems)
    n = len(m)
    out = dict(n_loads=n, mean=float(m.mean()), min=float(m.min()), max=float(m.max()), range=float(m.max() - m.min()))
    if n > 1:
        sd = float(m.std(ddof=1))
        hw = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n))
        out.update(sd_loads=sd, sd_between=float(np.sqrt(max(sd**2 - np.mean(s**2), 0.0))), ci95=[out["mean"] - hw, out["mean"] + hw])
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--md", default=None)
    a = ap.parse_args()
    runs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(a.dir, "*.json"))) if "settings" in json.load(open(p))]
    L = []
    w = L.append
    for host in sorted({r["settings"]["host"] for r in runs}):
        rs = [r for r in runs if r["settings"]["host"] == host]
        w(f"\n## {host}: {len(rs)} loads; banner {sorted({r['settings']['banner'] for r in rs})}")
        w(f"advance ticks {sorted({r['settings']['realised_ticks'] for r in rs})}; op slack {sorted({r['settings']['op_slack_us'] for r in rs})}\n")
        w("| load | checks | pulses | L_50 (ns) | counter | L_buf | L_50−counter | L_buf−L_50 | path | S_nn | S_cdc | S_gate | S_conv | S_fe | phase° |")
        w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        good = []
        for r in rs:
            p = r["points"]["op"]
            if not p["n_pulse"]:
                w(f"| {r['settings']['label']} | no pulses |")
                continue
            sg = p.get("segments", {})
            f = lambda k: f"{sg[k]:.2f}" if sg.get(k) is not None else "-"
            w(f"| {r['settings']['label']} | {'pass' if r['passed'] else 'FAIL: ' + ', '.join(c['name'] for c in r['checks'] if not c['ok'])} | {p['n_pulse']}/{p['n']} | "
              f"{p['L_50']['mean']:.2f} ± {p['L_50']['sem']:.2f} | {p['counter']['mean']:.2f} | {p['L_buf']['mean']:.2f} | {p['l50_minus_counter']['mean']:+.2f} | "
              f"{p['lbuf_minus_l50']['mean']:+.2f} | {r['settings']['path_ns']:.2f} | {f('S_nn')} | {f('S_cdc')} | {f('S_gate')} | {f('S_conv')} | {f('S_fe')} | {p['phase_deg']:+.0f} |")
            if r["passed"]:
                good.append(p)
        w("")
        if good:
            for name, key in (("L_50", "L_50"), ("counter", "counter"), ("L_buf", "L_buf")):
                b = between([p[key]["mean"] for p in good], [p[key]["sem"] for p in good])
                ci = f"; 95% over loads [{b['ci95'][0]:.2f}, {b['ci95'][1]:.2f}]; sd of loads {b['sd_loads']:.2f}; between-load sd {b['sd_between']:.2f}" if "ci95" in b else ""
                w(f"- **{name}**: {b['mean']:.2f} ns over {b['n_loads']} passing loads, range {b['min']:.2f} to {b['max']:.2f} ({b['range']:.2f}){ci}")
            w(f"- per-shot sd of L_50 within a load: {np.mean([p['L_50']['sd'] for p in good]):.2f} ns")
            w(f"- shots without a pulse at the operating slack: {sum(p['n'] - p['n_pulse'] for p in good)} of {sum(p['n'] for p in good)}")
        sens = [r for r in rs if r.get("sensitivity")]
        for r in sens:
            b0 = r["sensitivity"]["base"]
            w(f"\nSensitivity, {r['settings']['label']} (slack 0, change in L_50 from the base point {b0['L_50']['mean']:.2f}; base gain 0.6):\n")
            w("| point | pulses | thr | plateau IQ | L_50 change (ns) | sem |")
            w("|---|---|---|---|---|---|")
            for nm, s in r["sensitivity"].items():
                if s["n_pulse"] and nm != "base":
                    w(f"| {nm} | {s['n_pulse']}/{s['n']} | {s['thr']} | {s['plat_iq']:.0f} | {s['L_50']['mean'] - b0['L_50']['mean']:+.2f} | {s['L_50']['sem']:.2f} |")
                elif nm != "base":
                    w(f"| {nm} | 0/{s['n']} | {s['thr']} | - | - | - |")
    text = "\n".join(L)
    print(text.encode("ascii", "replace").decode())
    if a.md:
        open(a.md, "w", encoding="utf-8").write(text + "\n")

if __name__ == "__main__":
    main()
