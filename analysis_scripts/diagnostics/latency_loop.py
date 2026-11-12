"""Measured feed-forward loop latency on v1.40L2 (task 4.1b).

Builds on conditional_pulse_demo.py (4.1a). Everything is measured in the
readout buffer's own time frame, with two captures on the loopback ADC (ro 0)
per shot:
  read 0  the classified window (trigger 1 at adc_trig_offset, 770 samples;
          NN_live's gate captures the same window, +-1 sample)
  read 1  the conditional pulse coming back through DAC -> cable -> ADC
Both triggers travel the same tProc -> buffer path and both signals pass the
same ADC, DDC and FIR, so

  L_buf = (t_trig2 - t_trig1) + onset1 / fs - (N - 1) / fs

is the time from the window's last sample (sample index N-1, at t_trig1 + (N-1)/fs)
to the conditional pulse's 50% point (an interpolated sample index on the same grid),
both at the buffer. Convention (decided 2026-10-20, audit R2/E1): latency is measured
to the last sample INSTANT. Until then the code used the window END, t_trig1 + N/fs,
which is 1/fs = 1.808 ns later and gives an L_buf 1.808 ns smaller; that convention is
still available with --lbuf-ref window_end (or env QICK_LBUF_REF=window_end, which
also reaches gate_probe / seg_probe / tproc_savings through l_buf). A sample at the ADC pin reaches the buffer d_in later and a
pulse leaving the DAC pin reaches it d_cable + d_in later, so

  L_buf = (pulse at DAC pin) - (last sample at ADC pin) + d_cable

i.e. the pin-to-pin loop plus the loopback cable delay (a few ns). d_in cancels.

Sweeps:
  --slacks   pulse time relative to the DPORT_RD wait. Negative = scheduled in
             the past, so the tProc dispatches it as soon as WPORT_WR executes
             (ASAP). Arrival is flat below some slack (the ASAP floor) and
             tracks the slack above it.
  --margins  DPORT_RD time after the programmed window end, at ASAP slack.
             The earliest margin with every shot fresh (and branch == NN_live
             decision) gives the measured loop.
Also reports the loopback onset (pulse timestamp -> 50% at the buffer, relative
to a trigger at the same timestamp) from the scheduled, slack > floor points.

"""
import argparse
import sys

import numpy as np

sys.path.insert(0, "analysis_scripts/diagnostics")
import conditional_pulse_demo as C              # noqa: E402  (brings up soc)

A, h, S, soc, soccfg = C.A, C.h, C.S, C.soc, C.soccfg
CFG0, FS_RO = C.CFG0, C.FS_RO                   # FS_RO in MHz (samples / us)
N_WIN = int(__import__("os").environ.get("QICK_LIVE_N", 770))   # live window, samples (L3: 576)
TICK = 1.0 / soccfg["tprocs"][0]["f_time"]       # tProc timestamp unit, us
# L_buf reference: "last_sample" = t1 + (N-1)/fs (default since 2026-10-20, audit R2);
# "window_end" = t1 + N/fs (the convention of every L_buf recorded before that date).
LBUF_REFS = ("last_sample", "window_end")
LBUF_REF = __import__("os").environ.get("QICK_LBUF_REF", "last_sample")
if LBUF_REF not in LBUF_REFS:
    raise SystemExit(f"QICK_LBUF_REF={LBUF_REF!r}: expected one of {LBUF_REFS}")

class LatProg(C.CondProg):
    """CondProg with the second capture pinned to the DPORT_RD time, not the pulse.
    cf["variant"]: "qick" (cond_jump as QICK emits it), "jlit" (literal jump
    target, no REG_WR s15), "nobranch" (conditional REG_WR + register-addressed
    WPORT_WR, no jump)."""

    def _initialize(s, cf):
        C.CondProg._initialize(s, cf)
        s.add_reg(name="cp_addr")

    def _body(s, cf):
        s.pulse(ch=cf["gen_ch"], name="p")
        s.send_readoutconfig(ch=cf["ro_ch"], name="r", t=0)
        s.trigger(ros=[cf["ro_ch"]], t=cf["adc_trig_offset"], width=cf["trig_width"])
        s.trigger(pins=[0, A.NN_TRIGGER_PIN], t=cf["adc_trig_offset"], width=10)  # port p4
        t_dec = cf["adc_trig_offset"] + C.live_len(cf) + cf["fb_margin"]
        t_cp = t_dec + cf["cp_slack"]
        # Pulse first in each branch; the bookkeeping writes come after it.
        # With cf["stamps"], dmem[3] / dmem[4] = s_usr_time (tProc time - ref,
        # in f_time ticks) right after the wait and right after the WPORT_WR;
        # the first stamp costs one instruction before DPORT_RD.
        # The capture trigger is queued BEFORE the wait: it is timestamped in the
        # future, so it fires on time. Issued after the branch, it can be
        # dispatched late by a branch-dependent amount and stop being a reference.
        s.trigger(ros=[cf["ro_ch"]], t=t_dec + cf["adc_trig_offset"] - cf["cp_pre"],
                  width=cf["trig_width"])
        variant = cf.get("variant", "qick")
        if variant == "nobranch":
            s._body_nobranch(cf, t_dec, t_cp)
            return
        s.wait(t=t_dec)
        if cf.get("stamps"):
            s.write_dmem(3, "s_usr_time")
        s.asm_inst({"CMD": "DPORT_RD", "DST": str(C.LIVE_PORT)})
        if variant == "jlit":
            # QICK's cond_jump loads the target into s15 first (REG_WR s15 label)
            # because pmem > 2^11 words; this program is far smaller, so the
            # label resolves to an 11-bit literal and that instruction goes.
            s.asm_inst({"CMD": "TEST", "OP": f"s8 - #{cf['fb_thr']}", "UF": "1"})
            s.asm_inst({"CMD": "JUMP", "IF": "NS", "LABEL": "fb_one"})
        else:
            s.cond_jump("fb_one", "s_port_l", "NS", "-", cf["fb_thr"])
        s.pulse(ch=cf["gen_ch"], name="c0", t=t_cp)
        if cf.get("stamps"):
            s.write_dmem(4, "s_usr_time")
        s.write_dmem(2, 0)
        s.jump("fb_done")
        s.label("fb_one")
        s.pulse(ch=cf["gen_ch"], name="c1", t=t_cp)
        if cf.get("stamps"):
            s.write_dmem(4, "s_usr_time")
        s.write_dmem(2, 1)
        s.label("fb_done")
        s.write_dmem(0, "s_port_l")
        s.write_dmem(1, "s_port_h")

    def _body_nobranch(s, cf, t_dec, t_cp):
        """No jump: the pulse's waveform address comes from a register that a
        conditional REG_WR sets from the TEST flags. After DPORT_RD:
        TEST, REG_WR -if(NS), WPORT_WR wmem [reg] = 3 instructions."""
        reg = s._get_reg("cp_addr")
        i0, i1 = s.wave2idx["c0_w0"], s.wave2idx["c1_w0"]
        tick = round(t_cp / TICK)
        s.asm_inst({"CMD": "REG_WR", "DST": reg, "SRC": "imm", "LIT": f"#{i0}"})
        s.wait(t=t_dec)
        if cf.get("stamps"):
            s.write_dmem(3, "s_usr_time")
        s.asm_inst({"CMD": "DPORT_RD", "DST": str(C.LIVE_PORT)})
        s.asm_inst({"CMD": "TEST", "OP": f"s8 - #{cf['fb_thr']}", "UF": "1"})
        s.asm_inst({"CMD": "REG_WR", "DST": reg, "SRC": "imm", "LIT": f"#{i1}", "IF": "NS"})
        s.asm_inst({"CMD": "WPORT_WR", "DST": str(soccfg["gens"][cf["gen_ch"]]["tproc_ch"]),
                    "SRC": "wmem", "ADDR": reg, "TIME": f"@{tick}"})
        if cf.get("stamps"):
            s.write_dmem(4, "s_usr_time")
        # branch record for the host: 1 if the c1 waveform was chosen
        s.asm_inst({"CMD": "REG_WR", "DST": reg, "SRC": "op", "OP": f"{reg} - #{i0}"})
        s.write_dmem(2, "cp_addr")
        s.write_dmem(0, "s_port_l")
        s.write_dmem(1, "s_port_h")

def onset_interp(cap, n_win):
    """50% crossing of |IQ| (relative to the plateau), linearly interpolated, in samples."""
    mag = np.hypot(cap[:, 0], cap[:, 1])
    p = np.convolve(mag, np.ones(n_win) / n_win, mode="valid").max()
    i = int(np.argmax(mag >= 0.5 * p))
    if i == 0:
        return np.nan, p
    f = (0.5 * p - mag[i - 1]) / (mag[i] - mag[i - 1])
    return i - 1 + f, p

def run_point(D, cf, n_win):
    """One (margin, slack) point: per-shot arrival, freshness, branch, tProc stamps."""
    n = len(D)
    out = dict(arr=np.zeros(n), fresh=np.zeros(n, bool), br=np.zeros(n, int),
               dec=np.zeros(n, int), st_wait=np.full(n, np.nan), st_wport=np.full(n, np.nan))
    full = dict(CFG0, **cf)
    t_dec = full["adc_trig_offset"] + C.live_len(full) + full["fb_margin"]
    t_dec_tick = round(t_dec / TICK)
    t2 = round((t_dec + full["adc_trig_offset"] - full["cp_pre"]) / TICK) * TICK
    for i in range(n):
        wf = C.trim_wf(h.prepare_waveform(D[i]))
        cfg = dict(CFG0, waveform_I=wf[0::2], waveform_Q=wf[1::2], **cf)
        iq = np.asarray(LatProg(soccfg, cfg).acquire_decimated(soc, progress=False))
        iq = iq.reshape(-1, iq.shape[-2], 2)
        m = np.asarray(soc.tproc.read_mem("dmem", length=5)).reshape(-1).astype(np.int64)
        hdr = soc.live_read_block(0, 3)
        rec = C.live_record(C.shot_window(hdr))
        on, _ = onset_interp(iq[-1], n_win)
        # pulse 50% point at the buffer, in tProc time relative to t_dec (us)
        out["arr"][i] = (t2 - t_dec) + on / FS_RO
        out["fresh"][i] = (int(np.uint32(m[1])) & C.SHOT_MASK) == rec[0]
        out["br"][i], out["dec"][i] = int(m[2]), rec[7] & 1
        if cf.get("stamps"):
            out["st_wait"][i] = (m[3] - t_dec_tick) * TICK
            out["st_wport"][i] = (m[4] - t_dec_tick) * TICK
    return out

def l_buf(cf, arr_us, ref=None):
    """Window's last sample -> conditional pulse 50% point, both at the buffer, ns.
    Trigger 1 (t = adc_trig_offset) and trigger 2 share the tProc -> buffer path, so
    L_buf = (t_dec + arr) - (t_trig1 + (N_WIN - 1) / fs)   [ref "last_sample", default]
    L_buf = (t_dec + arr) - (t_trig1 + N_WIN / fs)         [ref "window_end", pre-2026-10-20]"""
    ref = ref or LBUF_REF
    n_ref = {"last_sample": N_WIN - 1, "window_end": N_WIN}[ref]
    full = dict(CFG0, **cf)
    t1 = round(full["adc_trig_offset"] / TICK) * TICK
    t_dec = full["adc_trig_offset"] + C.live_len(full) + full["fb_margin"]
    return 1e3 * (t_dec + arr_us - t1 - n_ref / FS_RO)

def ns(x):
    return f"{1e3 * x:6.1f}"

def bmed(x, br, q):
    sel = br == q
    return np.nanmedian(x[sel]) if sel.any() else np.nan

def main():
    global LBUF_REF
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=25, help="shots per sweep point")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--dataset", default="test20k_real_X")
    ap.add_argument("--slacks", default="-0.2,0,0.1,0.15,0.2,0.25,0.3,0.4,0.5")
    ap.add_argument("--slack-margin", type=float, default=0.1,
                    help="DPORT_RD margin for the slack sweep")
    ap.add_argument("--margins", default="-0.04,-0.03,-0.02,-0.01,0,0.01,0.02,0.03,0.04,0.06")
    ap.add_argument("--asap-slack", type=float, default=-0.2,
                    help="slack for the margin sweep (in the past = ASAP)")
    ap.add_argument("--pre", type=float, default=0.05,
                    help="us the second capture opens before (DPORT_RD time + adc_trig_offset)")
    ap.add_argument("--thr", type=int, default=16)
    ap.add_argument("--g0", type=float, default=0.2)
    ap.add_argument("--g1", type=float, default=0.6)
    ap.add_argument("--cp-len", type=float, default=0.2)
    ap.add_argument("--trig-width", type=float, default=0.5)
    ap.add_argument("--out", default="campaign_v140L1/A_L2_latency_loop.npz")
    ap.add_argument("--lbuf-ref", choices=LBUF_REFS, default=LBUF_REF,
                    help="L_buf reference: last window sample instant (default) or window end "
                         "(the convention of runs before 2026-10-20; 1/fs = 1.808 ns smaller L_buf)")
    a = ap.parse_args()
    LBUF_REF = a.lbuf_ref
    print(f"L_buf reference: {LBUF_REF}")

    if not soc.live_present():
        raise SystemExit("no live core on this bitstream / NNQickSoc unpatched")
    banner = [l for l in str(soccfg).splitlines() if "built" in l]
    print("BANNER:", banner[0].strip() if banner else "?")
    print(f"tProc time unit {1e3 * TICK:.4f} ns, core {soccfg['tprocs'][0]['f_core']:.1f} MHz; "
          f"readout {FS_RO:.2f} MSa/s ({1e3 / FS_RO:.4f} ns/sample)")

    xf, yf = {nm: (x_, y_) for nm, x_, y_ in A.DATASETS}[a.dataset]
    k, c = C.calib.load_kc(strict=False)
    X_all = np.load(xf, mmap_mode="r")
    X = np.asarray(X_all[a.start:a.start + a.shots])
    D = h.prepare_chunk(X, k, c) * (S.raw_peak(X, k, c) / S.raw_peak(X_all, k, c))
    n_win = max(4, int(0.5 * a.cp_len * FS_RO))
    base = dict(cp_gain0=a.g0, cp_gain1=a.g1, cp_len=a.cp_len, cp_pre=a.pre,
                trig_width=a.trig_width, fb_thr=a.thr)
    save = {}

    # 1) slack sweep: arrival vs pulse timestamp. Points that track the slack give
    #    the loopback leg (timestamp -> 50% at the buffer, trigger-referenced);
    #    the flat floor is ASAP dispatch.
    print(f"\n-- slack sweep (margin {a.slack_margin:+.3f} us) --")
    print(f"{'slack us':>9} {'arr-t_dec ns':>13} {'arr-sched ns':>13} {'br0 arr':>8} {'br1 arr':>8}")
    legs = []
    for sl in [float(v) for v in a.slacks.split(",")]:
        cf = dict(base, fb_margin=a.slack_margin, cp_slack=sl)
        r = run_point(D, cf, n_win)
        md = np.nanmedian(r["arr"])
        b0, b1 = bmed(r["arr"], r["br"], 0), bmed(r["arr"], r["br"], 1)
        print(f"{sl:+9.3f} {ns(md):>13} {ns(md - sl):>13} {ns(b0):>8} {ns(b1):>8}", flush=True)
        save[f"slack_{sl:+.3f}_arr"] = r["arr"]
        save[f"slack_{sl:+.3f}_br"] = r["br"]
        legs.append((sl, md, b0, b1))
    legs = np.array(legs)
    floor = np.nanmin(legs[:, 1])
    sched = legs[(legs[:, 1] - floor) > 0.05]
    leg = np.median(sched[:, 1] - sched[:, 0]) if len(sched) else np.nan
    print(f"loopback leg (scheduled; timestamp -> 50% at buffer, trigger-referenced): "
          f"{ns(leg)} ns from {len(sched)} points")

    # 2) instruction timeline at ASAP (the stamp before DPORT_RD costs one instruction)
    cf = dict(base, fb_margin=a.slack_margin, cp_slack=a.asap_slack, stamps=1)
    r = run_point(D, cf, n_win)
    print("\n-- tProc stamps at ASAP (s_usr_time after the wait / after WPORT_WR, vs t_dec) --")
    for q in (0, 1):
        sel = r["br"] == q
        if sel.any():
            print(f"  branch {q} (n={sel.sum()}): wait release {ns(np.median(r['st_wait'][sel]))} ns"
                  f" [{ns(np.min(r['st_wait'][sel]))},{ns(np.max(r['st_wait'][sel]))}],"
                  f" WPORT_WR done {ns(np.median(r['st_wport'][sel]))} ns,"
                  f" effective pulse time {ns(np.median(r['arr'][sel]) - leg)} ns")
    save.update(stamp_wait=r["st_wait"], stamp_wport=r["st_wport"], stamp_br=r["br"],
                stamp_arr=r["arr"])

    # 3) margin sweep at ASAP, no stamps: the measured loop
    print(f"\n-- margin sweep (ASAP, slack {a.asap_slack:+.3f}) --")
    print(f"{'margin us':>9} {'fresh':>8} {'br==dec':>8} {'L_buf med':>10} {'min':>7} {'max':>7}"
          f" {'br0 med':>8} {'br1 med':>8}")
    rows = []
    for mg in [float(v) for v in a.margins.split(",")]:
        cf = dict(base, fb_margin=mg, cp_slack=a.asap_slack)
        r = run_point(D, cf, n_win)
        L = l_buf(cf, r["arr"])
        nf = int(r["fresh"].sum())
        agree = int((r["fresh"] & (r["br"] == r["dec"])).sum())
        b0, b1 = bmed(L, r["br"], 0), bmed(L, r["br"], 1)
        print(f"{mg:+9.3f} {nf:4d}/{len(L):<3d} {agree:4d}/{nf:<3d} {np.nanmedian(L):10.1f}"
              f" {np.nanmin(L):7.1f} {np.nanmax(L):7.1f} {b0:8.1f} {b1:8.1f}", flush=True)
        rows.append((mg, np.nanmedian(L), nf / len(L), agree / max(nf, 1), b0, b1,
                     np.nanmin(L), np.nanmax(L)))
        save[f"margin_{mg:+.3f}_L"] = L
        save[f"margin_{mg:+.3f}_fresh"] = r["fresh"]
        save[f"margin_{mg:+.3f}_br"] = r["br"]
    good = [row for row in rows if row[2] == 1.0 and row[3] == 1.0]
    if good:
        g = min(good, key=lambda row: row[0])
        print(f"\nEARLIEST ALL-FRESH MARGIN {g[0]:+.3f} us: L_buf median {g[1]:.1f} ns "
              f"[{g[6]:.1f}, {g[7]:.1f}] (branch 0 {g[4]:.1f}, branch 1 {g[5]:.1f})\n"
              f"  = last window sample at the ADC pin -> conditional pulse at the DAC pin "
              f"+ loopback cable delay")
    np.savez(a.out, slack=legs, margin=np.array(rows), leg_us=leg, args=str(vars(a)),
             fs_ro=FS_RO, tick=TICK, lbuf_ref=LBUF_REF, **save)
    print("saved", a.out)

if __name__ == "__main__":
    main()
