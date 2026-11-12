"""L_buf vs segment sum (2026-11-05): the whole loop in ONE capture on NN_live's own stream.

Readout 1 (buffer 1 = the boxcar stream) shares NN_live's classify trigger (tProc trig_11), and NN_live's window is
buffer 1's first 576 samples, sample for sample (checked here on every shot). So one long capture holds both the
window's last sample (index 575, by that check, not by a programmed advance) and the returning conditional pulse, in
one clock and one trigger. Per shot:
  counter  seg_stamp e2 - last, cycles (what the segment sum telescopes to)
  sw_edge  the seg_stamp edge2 rule replayed on the captured samples (|I|+|Q| >= thr, after 16 samples below thr);
           sw_edge - 575 must equal the counter up to a fixed pipeline offset
  L_thr    the same crossing, interpolated, in ns after sample 575
  L_50     50% of the |IQ| plateau, interpolated, in ns after sample 575: the capture definition, on NN_live's chain
  L_buf    the FIR capture measure as gate_probe computes it, same shot
The gate passes every fresh decision, so every shot carries a pulse. Small test: 40 shots per slack.
--phases sweeps the conditional pulse's phase: the edge detector is on |I|+|Q|, so with a fixed threshold the fraction
of the rising edge at which it fires depends on the phase; L_50 must not. --fair-thr sets the threshold per phase to
half the |I|+|Q| plateau, which must remove that dependence.
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, "analysis_scripts/diagnostics")
import gate_probe as G                          # noqa: E402  (brings up soc)

C, A, L, h, S, soc, soccfg = G.C, G.A, G.L, G.h, G.S, G.soc, G.soccfg
FS, T = G.FS_RO, G.T_A
N = 576
LOW_N = 16

class P(G.GateProg):
    def _initialize(s, cf):
        G.GateProg._initialize(s, cf)
        s.declare_readout(ch=1, length=cf["long_len"])
        s.add_readoutconfig(ch=1, name="r1", freq=cf["freq"], gen_ch=cf["gen_ch"])
        s.add_pulse(ch=cf["gen_ch"], name="cph", style="const", length=cf["cp_len"], freq=cf["freq"],
                    phase=cf["cp_phase"], gain=cf["cp_gain1"], ro_ch=cf["ro_ch"])

    def _body(s, cf):
        s.pulse(ch=cf["gen_ch"], name="p")
        s.send_readoutconfig(ch=cf["ro_ch"], name="r", t=0)
        s.send_readoutconfig(ch=1, name="r1", t=0)
        s.trigger(ros=[cf["ro_ch"]], t=cf["adc_trig_offset"], width=cf["trig_width"])
        s.trigger(ros=[1], pins=[0, A.NN_TRIGGER_PIN], t=cf["adc_trig_offset"] - cf["nn_adv"], width=10)
        t_dec = cf["adc_trig_offset"] - cf["nn_adv"] + C.live_len(cf) + cf["fb_margin"]
        t_cp = t_dec + cf["cp_slack"]
        s.trigger(ros=[cf["ro_ch"]], t=t_dec + cf["adc_trig_offset"] - cf["cp_pre"], width=cf["trig_width"])
        s.pulse(ch=cf["gen_ch"], name="cph", t=t_cp)
        s.trigger(pins=[G.GATE_PIN], t=t_cp, width=cf["cp_len"] + cf["win_extra"])
        s.wait(t=t_cp + cf["cp_len"] + 0.1)
        s.asm_inst({"CMD": "DPORT_RD", "DST": str(C.LIVE_PORT)})
        s.write_dmem(0, "s_port_l")
        s.write_dmem(1, "s_port_h")

def shot(D_i, cf, n_win):
    wf = C.trim_wf(h.prepare_waveform(D_i))
    iq = P(soccfg, dict(G.CFG0, waveform_I=wf[0::2], waveform_Q=wf[1::2], **cf)).acquire_decimated(soc, progress=False)
    a0 = np.asarray(iq[0])
    fir = a0.reshape(-1, a0.shape[-2], 2)[-1]
    box = np.asarray(iq[1]).reshape(-1, 2)
    words = np.asarray(soc.live_read_block(16, N), dtype=np.int64)
    li = ((words & 0xFFFF) ^ 0x8000) - 0x8000
    lq = (((words >> 16) & 0xFFFF) ^ 0x8000) - 0x8000
    same = np.array_equal(li, box[:N, 0]) and np.array_equal(lq, box[:N, 1])
    seg = np.array(soc.seg_latest(), dtype=np.int64)
    on, plat = L.onset_interp(fir, n_win)
    full = dict(G.CFG0, **cf)
    t_dec = full["adc_trig_offset"] - full["nn_adv"] + C.live_len(full) + full["fb_margin"]
    t2 = round((t_dec + full["adc_trig_offset"] - full["cp_pre"]) / L.TICK) * L.TICK
    t_dec0 = full["adc_trig_offset"] + C.live_len(full) + full["fb_margin"]
    return dict(box=box, same=same, seg=seg, fir_plat=plat, L_buf=L.l_buf(cf, np.array([(t2 - t_dec0) + on / FS]))[0])

def edge_rule(x, thr):
    """The seg_stamp edge2 rule on x[N:]: armed after LOW_N samples below thr, then the first sample >= thr."""
    low = 0
    for i in range(N, len(x)):
        if x[i] >= thr:
            if low >= LOW_N:
                return i
            low = 0
        else:
            low += 1
    return None

def cross(x, i, level):
    return i - 1 + (level - x[i - 1]) / (x[i] - x[i - 1])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=40)
    ap.add_argument("--slacks", default="-0.064,0.0")
    ap.add_argument("--nn-advance", type=float, default=0.1135)
    ap.add_argument("--phases", default="0")
    ap.add_argument("--fair-thr", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    print("boxcar path:", soc.box_config(sel=1, shift=7, m=11))
    k, c = C.calib.load_kc(strict=False)
    X_all = np.load("data/test20k_real_X.npy", mmap_mode="r")
    X = np.asarray(X_all[:a.shots + 8])
    D = h.prepare_chunk(X, k, c) * (S.raw_peak(X, k, c) / S.raw_peak(X_all, k, c))
    n_win = max(4, int(0.5 * 0.2 * FS))
    slacks = [float(x) for x in a.slacks.split(",")]
    base = dict(cp_gain0=0.0, cp_gain1=0.6, cp_len=0.2, cp_pre=0.15, trig_width=0.5, fb_margin=0.035, fb_thr=16,
                win_extra=0.05, nn_adv=a.nn_advance, split=True, long_len=1.75, cp_phase=0.0)
    soc.gate_write(0, G.cfg_word(en=1, pass1=1, pass0=1, pass_stale=1))
    cal = [shot(D[i], dict(base, cp_slack=slacks[-1]), n_win) for i in range(8)]
    m2p = int(soc.gate_stats()[7])
    plat = float(np.median([r["fir_plat"] for r in cal]))
    delay = m2p - G.DELAY_LEAD
    thr = int(0.5 * plat)
    soc.seg_write(1, thr)
    soc.seg_write(4, 2048)
    soc.gate_write(0, G.cfg_word(en=1, pass1=1, pass0=1, pass_stale=0, delay=delay))
    print(f"DELAY {delay}; FIR plateau {plat:.0f}; seg thr {thr}; advance {a.nn_advance} us")
    phases = [float(x) for x in a.phases.split(",")]
    points = [(sl, ph) for sl in slacks for ph in phases]
    save = dict(slacks=np.array([p[0] for p in points]), phases=np.array([p[1] for p in points]), nn_adv=a.nn_advance,
                fir_plat=plat, delay=delay, fair_thr=a.fair_thr)
    for si, (sl, cph) in enumerate(points):
        cfp = dict(base, cp_slack=sl, cp_phase=cph)
        if a.fair_thr:
            ps = []
            for i in range(4):
                b = shot(D[i], cfp, n_win)["box"].astype(np.float64)
                m = np.abs(b[:, 0]) + np.abs(b[:, 1])
                j = edge_rule(m, thr)
                ps.append(np.median(m[j + 25:j + 85]))
            thr = int(0.5 * np.median(ps))
            soc.seg_write(1, thr)
            shot(D[0], cfp, n_win)          # seg_stamp reads thr between shots
        save[f"thr_{si}"] = thr
        rs = [shot(D[i + 8], cfp, n_win) for i in range(a.shots)]
        row = []
        for r in rs:
            b = r["box"].astype(np.float64)
            m1, m2 = np.abs(b[:, 0]) + np.abs(b[:, 1]), np.hypot(b[:, 0], b[:, 1])
            sg = r["seg"][3:]
            cnt = int(sg[6] - sg[3])
            i1 = edge_rule(m1, thr)
            if i1 is None:
                row.append([np.nan] * 9 + [float(r["same"])])
                continue
            pmag = np.median(m2[i1 + 25:i1 + 85])
            psum = np.median(m1[i1 + 25:i1 + 85])
            lev = 0.5 * pmag
            i2 = edge_rule(m2, lev)
            on2 = cross(m2, i2, lev)
            phase = np.degrees(np.arctan2(np.median(b[i1 + 25:i1 + 85, 1]), np.median(b[i1 + 25:i1 + 85, 0])))
            row.append([cnt, i1 - (N - 1), (cross(m1, i1, thr) - (N - 1)) * T, (on2 - (N - 1)) * T, r["L_buf"], pmag, psum,
                        phase, thr / psum, float(r["same"])])
        row = np.array(row, dtype=np.float64)
        save[f"row_{si}"] = row
        save[f"box_{si}"] = np.array([r["box"] for r in rs], dtype=np.int16)
        f = np.isfinite(row[:, 0])
        cnt, sw, Lt, L5, Lb, pm, ps, ph, frac, same = row[f].T
        print(f"\nslack {sl:+.3f}: phase {cph:+.0f} thr {thr}: {int(f.sum())}/{len(row)} shots with a pulse; window == buffer {int(row[:, 9].sum())}/{len(row)}")
        print(f"  counter e2-last cycles {sorted(set(cnt.astype(int).tolist()))}; sw_edge-575 {sorted(set(sw.astype(int).tolist()))}; "
              f"counter - sw {sorted(set((cnt - sw).astype(int).tolist()))}")
        print(f"  counter {np.mean(cnt) * T:6.2f}  L_thr {np.mean(Lt):6.2f}  L_50 {np.mean(L5):6.2f}  L_buf {np.mean(Lb):6.2f} ns")
        print(f"  L_50 - counter {np.mean(L5 - cnt * T):+5.2f}; L_50 - L_thr {np.mean(L5 - Lt):+5.2f}; L_buf - L_50 {np.mean(Lb - L5):+5.2f} ns "
              f"(sd {np.std(Lb - L5):.2f}); plateau |IQ| {np.mean(pm):.0f}, |I|+|Q| {np.mean(ps):.0f}, phase {np.mean(ph):+.0f} deg, thr/plateau(sum) {np.mean(frac):.3f}")
    soc.gate_write(0, 0)
    np.savez(a.out, **save)
    print("saved", a.out, "(gate left at word 0)")

if __name__ == "__main__":
    main()
