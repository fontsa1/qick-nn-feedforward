"""Loop latency measurement for one bitstream load (2026-11-05): settings record, closure checks, per-load result.

The measurement is lbuf_direct.py's: one long capture on NN_live's own stream (buffer 1, same trigger as NN_live), so
the window's last sample (index 575, checked on every shot) and the returning conditional pulse are in one record.
Definition of the number: L_50 = last window sample -> 50% of the returning pulse's |IQ| plateau, interpolated. Both
are at the readout output, so it is the time between a sample and the response to it at one point, cable included.

Per run:
  1. record every setting and the firmware banner; refuse to measure if the realised advance is not the expected one
  2. operating slack and slack 0: L_50, counter, L_buf, segment split, carrier phase, per shot
  3. --sensitivity: L_50 against the conditional pulse gain and the carrier phase, and with an all-zero stimulus (no
     replayed tail under the pulse's edge). Each point sets the seg_stamp threshold to half its own |I|+|Q| plateau.
  4. closure checks. Any failure: the JSON says so, the exit code is 3, and the run must not be reported.
     The checks are written to catch settings errors: 0.1078 instead of 0.1135 fails the path check by 4 ns.

The gate passes every fresh decision here (PASS1 = PASS0 = 1, stale blanks), so a shot with no pulse is a late decision.

"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, "analysis_scripts/diagnostics")
import lbuf_direct as D                         # noqa: E402  (brings up soc)

G, C, L, h, S, soc, soccfg = D.G, D.C, D.L, D.h, D.S, D.soc, D.soccfg
T, N = D.T, D.N
PLAT_AVG = 60                                   # samples; the conditional pulse is 0.2 us = 110 samples

# closure tolerances, ns. Expected values and observed ranges: CURRENT_BEST_RESULTS.md, 2026-11-05.
EXPECT_NS = dict(
    path_minus_adv=(3.2, 1.0),                  # 115.1 (box_probe) - 111.9; the stale 0.1078 gives 7.3. Centre per build: --expect-path-adv
    lbuf_minus_l50=(3.2, 0.6),                  # must follow the path term measured on this load (centre replaced below)
    l50_minus_counter=(-0.9, 0.6),              # half a sample, from a threshold at 50% on a sampled edge
)
THR_FRACTION = (0.5, 0.03)                      # seg threshold / |I|+|Q| plateau
RES_MINUS_LAST = 9                              # NN_live result - last sample, cycles, fixed by the design

def sha(path):
    try:
        return hashlib.sha256(open(path, "rb").read()).hexdigest()
    except OSError:
        return None

def realised_ticks(adv):
    """Ticks between readout 0's trigger and NN_live's, as the tProc will schedule them."""
    off = dict(G.CFG0)["adc_trig_offset"]
    return int(round(off / L.TICK) - round((off - adv) / L.TICK))

def mag_sum(box):
    b = box.astype(np.float64)
    return np.abs(b[:, 0]) + np.abs(b[:, 1])

def plateau_sum(box):
    """|I|+|Q| plateau of the returning pulse: the largest PLAT_AVG-sample mean after the window."""
    return np.convolve(mag_sum(box)[N:], np.ones(PLAT_AVG) / PLAT_AVG, mode="valid").max()

def set_fair_thr(wfs, cf, n_win):
    ps = [plateau_sum(D.shot(w, cf, n_win)["box"]) for w in wfs]
    thr = int(0.5 * np.median(ps))
    soc.seg_write(1, thr)
    D.shot(wfs[0], cf, n_win)                   # seg_stamp reads thr between shots
    return thr

def measure(wfs, cf, thr, n_win, want_segs, delay):
    """One point. Rows are NaN for a shot with no pulse."""
    rows, segs, boxes, seg_raw = [], [], [], []
    delta = None
    for w in wfs:
        r = D.shot(w, cf, n_win)
        g = np.array(soc.gate_stats(), dtype=np.int64)
        b = r["box"].astype(np.float64)
        boxes.append(r["box"])
        seg_raw.append(r["seg"])
        m1, m2 = np.abs(b[:, 0]) + np.abs(b[:, 1]), np.hypot(b[:, 0], b[:, 1])
        sg = r["seg"][3:]
        i1 = D.edge_rule(m1, thr)
        if i1 is None:
            rows.append([np.nan] * 11)
            segs.append([np.nan] * 6)
            continue
        hi = b[i1 + 25:i1 + 85]
        pmag, psum = np.median(m2[i1 + 25:i1 + 85]), np.median(m1[i1 + 25:i1 + 85])
        i2 = D.edge_rule(m2, 0.5 * pmag)
        l50 = (D.cross(m2, i2, 0.5 * pmag) - (N - 1)) * T
        rows.append([sg[6] - sg[3], i1 - (N - 1), sg[4] - sg[3], l50, r["L_buf"], pmag, psum,
                     np.degrees(np.arctan2(np.median(hi[:, 1]), np.median(hi[:, 0]))), thr / psum, float(r["same"]),
                     (D.cross(m1, i1, thr) - (N - 1)) * T])
        sp = G.split_segments(g, r["seg"], delta, delay) if want_segs else None
        segs.append([sp[k] for k in ("S_nn", "S_cdc", "S_gate", "S_conv", "S_fe", "sum")] if sp else [np.nan] * 6)
    return np.array(rows, dtype=np.float64), np.array(segs, dtype=np.float64), np.array(boxes, dtype=np.int16), np.array(seg_raw)

COLS = ["counter_cyc", "sw_edge", "res_minus_last", "L_50", "L_buf", "plat_iq", "plat_sum", "phase_deg", "thr_frac", "same", "L_thr"]
SEGS = ["S_nn", "S_cdc", "S_gate", "S_conv", "S_fe", "sum"]

def summarise(row, seg=None):
    f = np.isfinite(row[:, 0])
    x = row[f]
    out = dict(n=int(len(row)), n_pulse=int(f.sum()))
    if not f.any():
        return out
    n = len(x)
    cnt, l50, lb = x[:, 0] * T, x[:, 3], x[:, 4]
    def ms(v):
        return dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)) if n > 1 else 0.0,
                    sem=float(np.std(v, ddof=1) / np.sqrt(n)) if n > 1 else 0.0)
    out.update(L_50=ms(l50), L_buf=ms(lb), counter=ms(cnt), lbuf_minus_l50=ms(lb - l50), l50_minus_counter=ms(l50 - cnt),
               gap=ms(lb - cnt), plat_iq=float(np.mean(x[:, 5])), plat_sum=float(np.mean(x[:, 6])),
               phase_deg=float(np.degrees(np.angle(np.mean(np.exp(1j * np.radians(x[:, 7])))))), thr_frac=float(np.mean(x[:, 8])),
               counter_cycles=sorted(set(x[:, 0].astype(int).tolist())), counter_minus_sw=sorted(set((x[:, 0] - x[:, 1]).astype(int).tolist())),
               res_minus_last=sorted(set(x[:, 2].astype(int).tolist())), same_all=bool(np.all(row[:, 9][f] == 1)))
    if seg is not None:
        out["segments"] = {k: float(np.nanmean(seg[:, i])) if np.isfinite(seg[:, i]).any() else None for i, k in enumerate(SEGS)}
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="load name, e.g. reload3 or cold1")
    ap.add_argument("--box-json", required=True, help="box_probe.py output for this load (the measured path difference)")
    ap.add_argument("--out", required=True, help="npz; the summary goes to the same name with .json")
    ap.add_argument("--shots", type=int, default=100, help="shots at the operating slack")
    ap.add_argument("--shots0", type=int, default=40, help="shots at slack 0")
    ap.add_argument("--sens-shots", type=int, default=20, help="shots per sensitivity point")
    ap.add_argument("--slack", type=float, default=-0.064)
    ap.add_argument("--nn-advance", type=float, default=0.1135)
    ap.add_argument("--expect-ticks", type=int, default=55, help="realised advance, tProc ticks; 0 skips the refusal")
    ap.add_argument("--expect-path-adv", type=float, default=EXPECT_NS["path_minus_adv"][0],
                    help="centre of the path_minus_adv check, ns: 3.2 on v1.43 (path 115.1), 1.4 on v1.44 (path 113.3). The boxcar path "
                         "moves by one sample between place-and-route runs; the check guards against a stale advance (about 7 ns)")
    ap.add_argument("--expect-banner", default=None)
    ap.add_argument("--bit", default=None, help="local .bit that was deployed, for the record")
    ap.add_argument("--sensitivity", action="store_true")
    ap.add_argument("--gains", default="0.15,0.3,0.9")
    ap.add_argument("--phases", default="90,180,270")
    ap.add_argument("--rf-thr", type=int, default=2048)
    a = ap.parse_args()

    host = os.environ.get("QICK_NS_HOST")
    banner = [l.strip() for l in str(soccfg).splitlines() if "built" in l]
    banner = banner[0] if banner else "?"
    ticks = realised_ticks(a.nn_advance)
    box = json.load(open(a.box_json))
    path_ns = float(box["A"]["saved_ns"])
    adv_ns = ticks * L.TICK * 1e3
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    settings = dict(label=a.label, host=host, banner=banner, time=time.strftime("%Y-%m-%d %H:%M:%S"), git_head=git,
                    script_sha256=sha(__file__), bit_sha256=sha(a.bit) if a.bit else None, k_sha256=sha(f"calib/k_{host}.npy"),
                    c_sha256=sha(f"calib/c_{host}.npy"), nn_advance_us=a.nn_advance, expect_path_adv_ns=a.expect_path_adv, realised_ticks=ticks, realised_adv_ns=adv_ns,
                    tick_ns=L.TICK * 1e3, sample_ns=T, op_slack_us=a.slack, path_ns=path_ns, box_shift=7, box_m=11,
                    cp_gain=0.6, cp_len_us=0.2, cp_pre_us=0.15, fb_margin_us=0.035, rf_thr=a.rf_thr,
                    live_n=os.environ.get("QICK_LIVE_N"), box_json=a.box_json)
    print("SETTINGS", json.dumps(settings, indent=1), flush=True)
    if a.expect_banner and a.expect_banner not in banner:
        raise SystemExit(f"REFUSED: banner {banner!r} does not contain {a.expect_banner!r}")
    if a.expect_ticks and ticks != a.expect_ticks:
        raise SystemExit(f"REFUSED: advance {a.nn_advance} us is {ticks} ticks, expected {a.expect_ticks}")

    print("boxcar path:", soc.box_config(sel=1, shift=7, m=11))
    k, c = C.calib.load_kc(strict=False)
    X_all = np.load("data/test20k_real_X.npy", mmap_mode="r")
    n_need = 8 + a.shots + a.shots0 + a.sens_shots
    X = np.asarray(X_all[:n_need])
    W = h.prepare_chunk(X, k, c) * (S.raw_peak(X, k, c) / S.raw_peak(X_all, k, c))
    n_win = max(4, int(0.5 * 0.2 * D.FS))
    base = dict(cp_gain0=0.0, cp_gain1=0.6, cp_len=0.2, cp_pre=0.15, trig_width=0.5, fb_margin=0.035, fb_thr=16, win_extra=0.05,
                nn_adv=a.nn_advance, split=True, long_len=1.75, cp_phase=0.0)

    soc.gate_write(0, G.cfg_word(en=1, pass1=1, pass0=1, pass_stale=1))
    mp = []
    for i in range(8):
        D.shot(W[i], dict(base, cp_slack=0.0), n_win)
        mp.append(int(soc.gate_stats()[7]))
    if any(v == 0xFFFFFFFF for v in mp):
        raise SystemExit(f"gate saw no pulse after the marker: {mp}")
    delay = min(mp) - G.DELAY_LEAD
    soc.seg_write(4, a.rf_thr)
    soc.gate_write(0, G.cfg_word(en=1, pass1=1, pass0=1, pass_stale=0, delay=delay))
    print(f"DELAY {delay}", flush=True)

    res = dict(settings=settings, delay=delay, points={}, sensitivity={})
    save = dict(settings=json.dumps(settings), columns=np.array(COLS), seg_columns=np.array(SEGS))
    first = 8

    def point(name, wfs, cf, want_segs=False):
        thr = set_fair_thr(wfs[:4], cf, n_win)
        row, seg, boxes, sr = measure(wfs, cf, thr, n_win, want_segs, delay)
        s = summarise(row, seg if want_segs else None)
        s.update(thr=thr, slack=cf["cp_slack"], gain=cf["cp_gain1"], cp_phase=cf["cp_phase"])
        save[f"row_{name}"], save[f"seg_{name}"], save[f"raw_{name}"] = row, seg, sr
        save[f"box_{name}"] = boxes
        txt = (f"L_50 {s['L_50']['mean']:7.2f} (sd {s['L_50']['sd']:.2f}, sem {s['L_50']['sem']:.2f})  counter {s['counter']['mean']:7.2f}  L_buf {s['L_buf']['mean']:7.2f}  "
               f"L_50-counter {s['l50_minus_counter']['mean']:+.2f}  L_buf-L_50 {s['lbuf_minus_l50']['mean']:+.2f}  phase {s['phase_deg']:+.0f}") if s["n_pulse"] else "no pulses"
        print(f"{name:>14}: thr {thr:5d}  {s['n_pulse']}/{s['n']} with a pulse  {txt}", flush=True)
        return s

    res["points"]["op"] = point("op", list(W[first:first + a.shots]), dict(base, cp_slack=a.slack), True)
    res["points"]["slack0"] = point("slack0", list(W[first + a.shots:first + a.shots + a.shots0]), dict(base, cp_slack=0.0), True)

    if a.sensitivity:
        ws = list(W[first + a.shots + a.shots0:first + a.shots + a.shots0 + a.sens_shots])
        cf0 = dict(base, cp_slack=0.0)
        res["sensitivity"]["base"] = point("sens_base", ws, cf0)
        for gn in [float(x) for x in a.gains.split(",")]:
            res["sensitivity"][f"gain_{gn}"] = point(f"gain_{gn}", ws, dict(cf0, cp_gain1=gn))
        for ph in [float(x) for x in a.phases.split(",")]:
            res["sensitivity"][f"phase_{ph:.0f}"] = point(f"phase_{ph:.0f}", ws, dict(cf0, cp_phase=ph))
        z = [np.zeros_like(W[0])] * a.sens_shots
        res["sensitivity"]["zero_stim"] = point("zero_stim", z, cf0)
    soc.gate_write(0, 0)

    # ---- closure
    checks = []

    def check(name, ok, detail):
        checks.append(dict(name=name, ok=bool(ok), detail=detail))
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)

    print("\nclosure checks:")
    c0, t0 = a.expect_path_adv, EXPECT_NS["path_minus_adv"][1]
    p_a = path_ns - adv_ns
    check("path_minus_adv", abs(p_a - c0) <= t0, f"path {path_ns:.2f} - advance {adv_ns:.2f} = {p_a:.2f} ns; expected {c0} +/- {t0}")
    for nm in ("op", "slack0"):
        s = res["points"][nm]
        if not s["n_pulse"]:
            check(f"{nm}_pulses", False, "no pulses")
            continue
        check(f"{nm}_all_pulses", s["n_pulse"] == s["n"], f"{s['n_pulse']}/{s['n']} (a shot with no pulse = late decision)")
        check(f"{nm}_window_is_buffer", s["same_all"], "NN_live's window == buffer 1 on every shot")
        check(f"{nm}_counter_is_samples", s["counter_minus_sw"] == [0], f"counter - edge rule on samples: {s['counter_minus_sw']}")
        check(f"{nm}_nn_cycles", s["res_minus_last"] == [RES_MINUS_LAST], f"result - last: {s['res_minus_last']} cycles")
        check(f"{nm}_thr_fraction", abs(s["thr_frac"] - THR_FRACTION[0]) <= THR_FRACTION[1], f"{s['thr_frac']:.3f}")
        v = s["l50_minus_counter"]["mean"]; c1, t1 = EXPECT_NS["l50_minus_counter"]
        tn = t1 + 2.5 * s["l50_minus_counter"]["sem"]      # fixed tolerance + shot noise of this point
        check(f"{nm}_l50_minus_counter", abs(v - c1) <= tn, f"{v:+.2f} ns; expected {c1} +/- {tn:.2f}")
        v = s["lbuf_minus_l50"]["mean"]; t2 = EXPECT_NS["lbuf_minus_l50"][1]
        tn = t2 + 2.5 * s["lbuf_minus_l50"]["sem"]
        check(f"{nm}_lbuf_minus_l50", abs(v - p_a) <= tn, f"{v:+.2f} ns; path - advance on this load {p_a:.2f} +/- {tn:.2f}")
        sg = s.get("segments", {})
        if sg.get("sum") is not None:
            check(f"{nm}_segments_sum", abs(sg["sum"] - s["counter"]["mean"]) <= 2.5,
                  f"segment sum {sg['sum']:.2f}, counter {s['counter']['mean']:.2f} ns")
    res["checks"] = checks
    res["passed"] = all(c_["ok"] for c_ in checks)
    out_json = a.out.rsplit(".", 1)[0] + ".json"
    np.savez_compressed(a.out, **save)
    json.dump(res, open(out_json, "w"), indent=1)
    print(f"\n{'ALL CHECKS PASSED' if res['passed'] else 'CHECKS FAILED: do not report this run'}; saved {a.out}, {out_json}; gate left at word 0")
    sys.exit(0 if res["passed"] else 3)

if __name__ == "__main__":
    main()
