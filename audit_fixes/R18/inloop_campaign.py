"""In-loop fidelity campaign -- NN_live's decision acting on the DAC through dac_gate, every shot scored.

The program is gate_probe.py's (GateProg): the conditional pulse is scheduled unconditionally on gen 0 with the
gate's window marker; dac_gate passes it if NN_live decided 1 for this shot and blanks it otherwise (PASS1 = 1,
PASS0 = 0, stale -> blank). The returning pulse is captured on the loopback ADC. Per shot the in-loop answer is
"a pulse came back", scored against the trace's label -- the classification as the loop delivers it, with every
loop failure (stale, late, wrong shot) counted as an error. Also per shot: the gate record (passed / stale /
decision bit / margin / late counter / binding rejects), NN_live's own record, seg_stamp timestamps (loop split),
L_buf, and the capture itself.

Traces are replayed exactly as box_live_capture.py does for the classification campaigns (same k/c predistortion,
same dataset-wide DAC scale from the 20k set, --clip-dac behaviour), so a row's in-loop decision can be paired with
its gate-idle decision from the 100k campaign. The gate is calibrated on the domain-map rows 1000:1008 (never
scored). Defaults: QICK_LIVE_N=576, --pre 0.15, --nn-advance 0.1135, margin 0.035 us, slack -0.064 us.

Checkpoints every --save-every shots to --out (rerun with --resume to continue after an interruption).

"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

if os.environ.get("QICK_LIVE_N") != "576":
    raise SystemExit("set QICK_LIVE_N=576; without it the window lands 351 ns late")
ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "analysis_scripts" / "diagnostics"))
sys.path.insert(0, str(ROOT / "board_tools"))
import board_control as BC  # noqa: E402
import gate_probe as G  # noqa: E402  (brings up soc via latency_loop)

L, C, h, soc, soccfg = G.L, G.C, G.h, G.soc, G.soccfg
CFG0, FS_RO = G.CFG0, G.FS_RO
CAL_ROWS = np.arange(1000, 1008)          # domain-map rows: excluded from every scored set

def waveforms(Xall, Xref, k, c, maxv):
    """box_live_capture.py's replay: (x - c) / k, interpolated to 11000 points, one dataset-wide scale."""
    z = np.asarray(Xref[:, 0::2]) + 1j * np.asarray(Xref[:, 1::2])
    zs = (z - c) / k
    sf = 32766 * 0.99 / (max(np.abs(zs.real).max(), np.abs(zs.imag).max()) * 1.0001)
    del z, zs
    xo = np.linspace(0, 1, 770); xn = np.linspace(0, 1, 11000)

    def wf(r):
        zr = (np.asarray(Xall[r, 0::2]) + 1j * np.asarray(Xall[r, 1::2]) - c) / k
        w = np.empty(22000); w[0::2] = np.interp(xn, xo, zr.real) * sf; w[1::2] = np.interp(xn, xo, zr.imag) * sf
        w = h.prepare_waveform(w)
        n_clip = int(np.sum(np.abs(w) > maxv))
        return np.clip(w, -maxv, maxv), n_clip
    return wf

def shot(w, cf, n_win):
    """gate_probe.one() on an already prepared waveform; also returns the capture and the full NN_live record."""
    w = C.trim_wf(w)
    cfg = dict(CFG0, waveform_I=w[0::2], waveform_Q=w[1::2], **cf)
    iq = np.asarray(G.GateProg(soccfg, cfg).acquire_decimated(soc, progress=False))
    cap = iq.reshape(-1, iq.shape[-2], 2)[-1]
    m = np.asarray(soc.tproc.read_mem("dmem", length=2)).reshape(-1).astype(np.int64)
    rec = C.live_record(C.shot_window(soc.live_read_block(0, 3)))
    g = np.array(soc.gate_stats(), dtype=np.int64)
    seg = np.array(soc.seg_latest(), dtype=np.int64)
    on, plat = L.onset_interp(cap, n_win)
    full = dict(CFG0, **cf)
    t_dec = full["adc_trig_offset"] - full["nn_adv"] + C.live_len(full) + full["fb_margin"]
    t2 = round((t_dec + full["adc_trig_offset"] - full["cp_pre"]) / L.TICK) * L.TICK
    t_dec0 = full["adc_trig_offset"] + C.live_len(full) + full["fb_margin"]
    lb = L.l_buf(cf, np.array([(t2 - t_dec0) + on / FS_RO]))[0]
    return dict(gate=g, seg=seg, plat=plat, cap=cap.astype(np.float32), rec=np.asarray(rec, np.int64),
                fresh=(int(np.uint32(m[1])) & C.SHOT_MASK) == rec[0], dec=int(rec[7] & 1), L_buf=lb)

def fid(pred, y):
    g, e = y == 0, y == 1
    p10, p01 = np.mean(pred[g] != 0), np.mean(pred[e] != 1)
    F = 1 - 0.5 * (p10 + p01)
    se = 0.5 * np.sqrt(p10 * (1 - p10) / g.sum() + p01 * (1 - p01) / e.sum())
    return 100 * F, 100 * (F - 1.96 * se), 100 * (F + 1.96 * se), 100 * p10, 100 * p01

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=20000)
    ap.add_argument("--shots", type=int, default=2000)
    ap.add_argument("--slack", type=float, default=-0.064)
    ap.add_argument("--margin", type=float, default=0.035)
    ap.add_argument("--pre", type=float, default=0.15)
    ap.add_argument("--nn-advance", type=float, default=0.1135)
    ap.add_argument("--rf-thr", type=int, default=2048, help="rf_tap_0 threshold")
    ap.add_argument("--x", default="data/test100k_real_X.npy")
    ap.add_argument("--y", default="data/test100k_real_y.npy")
    ap.add_argument("--scale-ref", default="data/test20k_real_X.npy")
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = ROOT / a.out

    banner = BC.firmware_banner(soc)["built"]
    print("host", os.environ.get("QICK_NS_HOST"), "| banner", banner)
    print("boxcar path:", soc.box_config(sel=1, shift=7, m=11))
    k, c = C.calib.load_kc(strict=False)
    print(f"calibration |k| {abs(k):.5f} arg {np.degrees(np.angle(k)):.2f} deg")
    maxv = soccfg["gens"][0]["maxv"]
    Xall = np.load(ROOT / a.x, mmap_mode="r"); yall = np.load(ROOT / a.y).astype(int)
    wf = waveforms(Xall, np.load(ROOT / a.scale_ref, mmap_mode="r"), k, c, maxv)
    n_win = max(4, int(0.5 * 0.2 * FS_RO))
    base = dict(cp_gain0=0.0, cp_gain1=0.6, cp_len=0.2, cp_pre=a.pre, trig_width=0.5, fb_margin=a.margin,
                fb_thr=16, win_extra=0.05, nn_adv=a.nn_advance, split=True)

    # gate calibration on the domain-map rows: marker -> pulse at the gate input, always-pass plateau
    soc.gate_write(0, G.cfg_word(en=1, pass1=1, pass0=1, pass_stale=1))
    mp, plats = [], []
    for r in CAL_ROWS:
        res = shot(wf(r)[0], dict(base, cp_slack=0.0), n_win)
        mp.append(int(res["gate"][7])); plats.append(res["plat"])
    gmag = int(soc.gate_stats()[11])
    if gmag != G.GAT4:
        raise SystemExit(f"gate magic {gmag:08x}, expected GAT4 (v1.41)")
    delay = min(mp) - G.DELAY_LEAD
    plat_pass = float(np.median(plats))
    g10 = int(soc.gate_stats()[10])
    print(f"gate GAT4: EARLY_BLANK {g10 >> 8 & 1} SHOT_BIND {g10 >> 9 & 1} FAIL_SAFE {g10 >> 10 & 1} "
          f"CFG_SAFE {g10 >> 11 & 1}; marker -> pulse {sorted(set(mp))} cycles, DELAY {delay}; plateau {plat_pass:.0f}")
    if delay != 29:
        print(f"WARNING: DELAY {delay}, expected 29")
    # half the |I|+|Q| plateau (the recorded runs used half the |IQ| plateau); seg_stamp only observes
    seg_thr = G.seg_threshold([wf(r)[0] for r in CAL_ROWS[:4]], dict(base, cp_slack=0.0))
    print(f"seg thr {seg_thr} (half the |I|+|Q| plateau)")
    soc.seg_write(1, seg_thr); soc.seg_write(4, a.rf_thr)

    rows = np.arange(a.start, a.start + a.shots)
    rows = rows[~np.isin(rows, np.arange(1000, 1600))]       # domain-map rows are never scored: skip them
    a.shots = len(rows)
    print(f"rows {rows[0]}-{rows[-1]}: {len(rows)} shots (domain-map rows 1000:1600 skipped)")
    keys = ("gate", "seg", "plat", "cap", "rec", "fresh", "dec", "L_buf", "late", "n_clip", "t")
    acc = {k_: [] for k_ in keys}
    done = 0
    if a.resume and out.exists():
        z = np.load(out, allow_pickle=True)
        assert int(z["start"]) == a.start and float(z["slack"]) == a.slack
        done = int(z["n_done"])
        for k_ in keys:
            acc[k_] = list(z[k_][:done])
        print(f"resuming after {done} shots")

    def save():
        np.savez(out, start=a.start, rows=rows[:len(acc["gate"])], y=yall[rows[:len(acc["gate"])]],
                 n_done=len(acc["gate"]), slack=a.slack, margin=a.margin, pre=a.pre, nn_adv=a.nn_advance,
                 delay=delay, plat_pass=plat_pass, marker_to_pulse=np.array(mp), rf_thr=a.rf_thr, seg_thr=seg_thr, banner=banner,
                 host=os.environ.get("QICK_NS_HOST"), k=k, c=c, lbuf_ref=L.LBUF_REF, fs_ro=FS_RO,
                 **{k_: np.array(v) for k_, v in acc.items()})

    soc.gate_write(0, G.cfg_word(en=1, pass1=1, pass0=0, pass_stale=0, delay=delay))
    late_prev = int(soc.gate_stats()[12]); rej0 = int(soc.gate_stats()[19])
    t0 = time.time()
    try:
        for i in range(done, a.shots):
            w, n_clip = wf(rows[i])
            res = shot(w, dict(base, cp_slack=a.slack), n_win)
            late_now = int(res["gate"][12])
            res["late"] = late_now - late_prev; late_prev = late_now
            res["n_clip"] = n_clip; res["t"] = time.time()
            for k_ in keys:
                acc[k_].append(res[k_])
            n = i + 1
            if n % a.save_every == 0 or n == a.shots:
                save()
                g = np.array(acc["gate"]); came = np.array(acc["plat"]) > 0.5 * plat_pass
                y = yall[rows[:n]]
                F = fid(came.astype(int), y)
                rate = (n - done) / (time.time() - t0)
                print(f"{n:5d}/{a.shots}  {rate:.2f} shots/s  in-loop F {F[0]:.2f}%  stale {int((g[:, 10] & 2 > 0).sum())}"
                      f"  late {int(np.sum(acc['late']))}  rejects {int(g[-1, 19]) - rej0}", flush=True)
    finally:
        soc.gate_write(0, G.cfg_word(en=0, delay=delay))
        if acc["gate"]:
            save()
        print("gate left disabled; saved", out)

    # summary
    g = np.array(acc["gate"]); y = yall[rows]
    flags = g[:, 10]
    passed = (flags & 1) > 0; stale = (flags & 2) > 0; gbit = (flags >> 2) & 1
    came = np.array(acc["plat"]) > 0.5 * plat_pass
    dec = np.array(acc["dec"]); fresh = np.array(acc["fresh"])
    correct = (came == passed) & ~stale & (gbit == dec)
    unsafe = came & ~(~stale & (gbit == dec) & (dec == 1))
    Fin = fid(came.astype(int), y); Fdec = fid(dec, y)
    print(f"\nrows {rows[0]}-{rows[-1]} (n = {len(rows)}), slack {a.slack} us")
    print(f"in-loop fidelity (pulse came back vs label): {Fin[0]:.2f}% [{Fin[1]:.2f}, {Fin[2]:.2f}]  "
          f"P(1|0) {Fin[3]:.2f}  P(0|1) {Fin[4]:.2f}")
    print(f"NN_live decision fidelity, same shots:        {Fdec[0]:.2f}% [{Fdec[1]:.2f}, {Fdec[2]:.2f}]")
    print(f"pulse == gate == NN_live decision (correct): {int(correct.sum())}/{len(rows)}; stale {int(stale.sum())}; "
          f"late {int(np.sum(acc['late']))}; unsafe {int(unsafe.sum())}; fresh {int(fresh.sum())}; "
          f"binding rejects {int(g[-1, 19]) - rej0}; DAC-clipped shots {int((np.array(acc['n_clip']) > 0).sum())}")

if __name__ == "__main__":
    main()
