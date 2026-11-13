"""Feed-forward through the DAC-stream gate (dac_gate).

The conditional pulse (c1, gain cp_gain1) is scheduled unconditionally at t_cp = t_dec + cp_slack on gen 0, with the
gate's window marker (tProc trig_10, output pin GATE_PIN) at the same time. dac_gate_0 passes it if NN_live decided 1
for the shot and blanks it otherwise (PASS1 = 1, PASS0 = 0, stale -> blank), so the decision acts on the DAC without a
tProc read and branch. The tProc still reads input 3 afterwards, for freshness bookkeeping only. A second capture
(queued first, as in latency_loop) records the pulse coming back.

Steps:
  1. gate passing everything (statistics only): marker -> pulse at the gate input, in clk_dac2 cycles (gate word 7);
     DELAY = that minus DELAY_LEAD, so the window opens just before the pulse.
  2. sweep cp_slack: per point, shots with the gate enabled; per shot the gate's record (stale, decision bit, passed,
     margin = window start - decision arrival) and whether a pulse came back (capture plateau > half the always-pass
     plateau). Correct = (pulse came back) == (gate passed) == (NN_live decision) on fresh shots. Also per point:
     late = decisions that arrived after their window opened, and unsafe = shots where the pulse came back although
     the shot's own fresh decision was not 1 (must be 0).
  3. at each point, L_buf (latency_loop's loop measure: window's last sample -> pulse 50% at the buffer) over the
     passed shots. Reference: the last sample instant t1 + (N-1)/fs (default), or the window end t1 + N/fs with
     --lbuf-ref window_end (L_buf 1.808 ns smaller). Saved in the npz as lbuf_ref.
The earliest slack with no stale shot gives the gate's loop latency.

--split: per shot also the segment timestamps (seg_stamp, rf_tap_0), and per slack point the split of the loop
(passed shots):
  S_nn    last sample at the readout output -> NN_live result (clk_adc0_x2)
  S_cdc   result -> decision at dac_gate (crosses into clk_dac2)
  S_gate  decision -> conditional pulse leaves the gate (window wait + 1 register)
  S_conv  pulse leaves the gate -> back at the ADC tile output (rf_tap_0): DAC tile + cable + ADC tile
  S_fe    ADC tile output -> readout output (QICK's DDC + boxcar path, fabric)
S_nn + S_cdc + S_gate + S_conv + S_fe = L_buf. The two clock domains are tied together through the classify trigger,
which both see through synchronizers; S_cdc and S_conv each carry about +-1 ns from that, opposite in sign, so their
sum does not.
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, "analysis_scripts/diagnostics")
import latency_loop as L                        # noqa: E402  (brings up soc)

C, A, h, S, soc, soccfg = L.C, L.A, L.h, L.S, L.soc, L.soccfg
CFG0, FS_RO = L.CFG0, L.FS_RO
F_DAC = soccfg["gens"][0]["f_fabric"]           # clk_dac2, MHz
DELAY_LEAD = 4                                  # window opens this many cycles before the pulse
T_A = 1e3 / FS_RO                               # clk_adc0_x2 period, ns (readout sample)
T_D = 1e3 / F_DAC                               # clk_dac2 period, ns
GAT2, GAT3, SEG2, SEG3 = 0x47415432, 0x47415433, 0x53454732, 0x53454733
GAT4 = 0x47415434                               # GAT3 layout + word 19 binding rejects, word 20 expected shot field
RF_STAGES = {SEG2: 3, SEG3: 4}                  # rf_tap_0 hit, cycles behind the tile output

PINS = soccfg["tprocs"][0]["output_pins"]
GATE_PIN = [i for i, p in enumerate(PINS) if p[0] == "trig" and p[1] == 10]
if not GATE_PIN:
    raise SystemExit("no output pin for tProc trig_10 (patch_nameserver_gate.py applied and server restarted?)")
GATE_PIN = GATE_PIN[0]

def cfg_word(en, pass1=1, pass0=0, pass_stale=0, delay=0, nz_thr=2000):
    return (int(en) | int(pass1) << 1 | int(pass0) << 2 | int(pass_stale) << 3
            | (int(delay) & 0xFF) << 8 | (int(nz_thr) & 0xFFFF) << 16)

class GateProg(L.LatProg):
    def _body(s, cf):
        s.pulse(ch=cf["gen_ch"], name="p")
        s.send_readoutconfig(ch=cf["ro_ch"], name="r", t=0)
        s.trigger(ros=[cf["ro_ch"]], t=cf["adc_trig_offset"], width=cf["trig_width"])
        # NN_live's window: on the boxcar path (--nn-advance) the same ADC samples reach NN_live nn_adv earlier, so
        # its trigger and t_dec move earlier
        s.trigger(pins=[0, A.NN_TRIGGER_PIN], t=cf["adc_trig_offset"] - cf["nn_adv"], width=10)
        t_dec = cf["adc_trig_offset"] - cf["nn_adv"] + C.live_len(cf) + cf["fb_margin"]
        t_cp = t_dec + cf["cp_slack"]
        s.trigger(ros=[cf["ro_ch"]], t=t_dec + cf["adc_trig_offset"] - cf["cp_pre"],
                  width=cf["trig_width"])
        # the conditional pulse and the gate's window, both scheduled now
        s.pulse(ch=cf["gen_ch"], name="c1", t=t_cp)
        s.trigger(pins=[GATE_PIN], t=t_cp, width=cf["cp_len"] + cf["win_extra"])
        # bookkeeping only, after the pulse
        s.wait(t=t_cp + cf["cp_len"] + 0.1)
        s.asm_inst({"CMD": "DPORT_RD", "DST": str(C.LIVE_PORT)})
        s.write_dmem(0, "s_port_l")
        s.write_dmem(1, "s_port_h")

class StreamProg(GateProg):
    """GateProg + readout 1 on NN_live's classify trigger: one long capture of the boxcar stream, pulse included."""

    def _initialize(s, cf):
        GateProg._initialize(s, cf)
        s.declare_readout(ch=1, length=cf["stream_len"])
        s.add_readoutconfig(ch=1, name="r1", freq=cf["freq"], gen_ch=cf["gen_ch"])

    def _body(s, cf):
        s.pulse(ch=cf["gen_ch"], name="p")
        s.send_readoutconfig(ch=cf["ro_ch"], name="r", t=0)
        s.send_readoutconfig(ch=1, name="r1", t=0)
        s.trigger(ros=[cf["ro_ch"]], t=cf["adc_trig_offset"], width=cf["trig_width"])
        s.trigger(ros=[1], pins=[0, A.NN_TRIGGER_PIN], t=cf["adc_trig_offset"] - cf["nn_adv"], width=10)
        t_dec = cf["adc_trig_offset"] - cf["nn_adv"] + C.live_len(cf) + cf["fb_margin"]
        t_cp = t_dec + cf["cp_slack"]
        s.trigger(ros=[cf["ro_ch"]], t=t_dec + cf["adc_trig_offset"] - cf["cp_pre"], width=cf["trig_width"])
        s.pulse(ch=cf["gen_ch"], name="c1", t=t_cp)
        s.trigger(pins=[GATE_PIN], t=t_cp, width=cf["cp_len"] + cf["win_extra"])

def seg_threshold(waveforms, cf):
    """seg_stamp edge2 threshold: half the conditional pulse's |I|+|Q| plateau on NN_live's own input stream.
    edge2 tests |I|+|Q|, so a threshold from the |IQ| plateau would fire 34-46% of the way up the edge depending on
    the carrier phase; half the |I|+|Q| plateau fires at 50% at any phase. Boxcar path (nn_adv != 0): measured on
    readout 1. FIR path: on readout 0's second capture. The gate must be passing the pulse. waveforms: prepared
    waveforms, one calibration shot each."""
    n_avg = max(4, int(0.5 * cf["cp_len"] * FS_RO))
    plats = []
    for w in waveforms:
        w = C.trim_wf(w)
        full = dict(CFG0, waveform_I=w[0::2], waveform_Q=w[1::2], **cf)
        n_skip = int(round(C.live_len(full) * FS_RO))
        full["stream_len"] = C.live_len(full) + cf["fb_margin"] + max(cf["cp_slack"], 0.0) + cf["cp_len"] + 0.35
        iq = StreamProg(soccfg, full).acquire_decimated(soc, progress=False)
        if cf["nn_adv"]:
            x = np.asarray(iq[1]).reshape(-1, 2)[n_skip:]
        else:
            a0 = np.asarray(iq[0])
            x = a0.reshape(-1, a0.shape[-2], 2)[-1]
        m = np.abs(x[:, 0].astype(np.float64)) + np.abs(x[:, 1].astype(np.float64))
        plats.append(np.convolve(m, np.ones(n_avg) / n_avg, mode="valid").max())
    return int(0.5 * np.median(plats))

def one(D_i, cf, n_win):
    wf = C.trim_wf(h.prepare_waveform(D_i))
    cfg = dict(CFG0, waveform_I=wf[0::2], waveform_Q=wf[1::2], **cf)
    iq = np.asarray(GateProg(soccfg, cfg).acquire_decimated(soc, progress=False))
    cap = iq.reshape(-1, iq.shape[-2], 2)[-1]
    m = np.asarray(soc.tproc.read_mem("dmem", length=2)).reshape(-1).astype(np.int64)
    rec = C.live_record(C.shot_window(soc.live_read_block(0, 3)))
    g = np.array(soc.gate_stats(), dtype=np.int64)
    seg = np.array(soc.seg_latest(), dtype=np.int64) if cf.get("split") else np.zeros(19, np.int64)
    on, plat = L.onset_interp(cap, n_win)
    full = dict(CFG0, **cf)
    t_dec = full["adc_trig_offset"] - full["nn_adv"] + C.live_len(full) + full["fb_margin"]
    t2 = round((t_dec + full["adc_trig_offset"] - full["cp_pre"]) / L.TICK) * L.TICK
    # L_buf stays physical: the window's last sample and the pulse, both in readout 0's
    # (FIR) time base, where the NN window starts at adc_trig_offset by construction of
    # nn_adv; l_buf references its own (unadvanced) t_dec
    t_dec0 = full["adc_trig_offset"] + C.live_len(full) + full["fb_margin"]
    return dict(gate=g, seg=seg, plat=plat, fresh=(int(np.uint32(m[1])) & C.SHOT_MASK) == rec[0],
                dec=int(rec[7] & 1), L_buf=L.l_buf(cf, np.array([(t2 - t_dec0) + on / FS_RO]))[0])

def s32(v):
    return (int(v) + 2**31) % 2**32 - 2**31

def split_segments(g, sg, delta=None, delay=None):
    """Segment times (ns) for one shot from its gate record g (GAT3/GAT4, words 0..18) and seg_stamp record sg
    ([records, thr, magic] + 16 words).
    seg fields are cycles after seg's trigger stamp (clk_adc0_x2: classify trigger -> trig_sync (2 sync + 1 reg) ->
    live_gate (2 sync) -> stamp, so the trigger is at -(4 + U(0, 1)) cycles, mean -4.5); gate words 15-18 are
    absolute clk_dac2 counts, the classify trigger at -2 cycles from word 18 (2 sync flops on the tProc's clock).
    Times below are ns after the trigger.
    GAT2 records (no words 15-18): the marker and the classify trigger are tProc outputs at programmed times on
    clk_dac2, delta cycles apart; marker rise = cls stamp + delta + 1, window open = marker rise + DELAY + 2,
    decision = open - margin (word 6), pulse at the gate input = marker rise + word 7."""
    r = sg[3:]
    last, res, e2, e3 = r[3], r[4], r[6], r[9]
    if r[7] & 0x56 != 0x56:                            # last, res, edge2, edge3 seen
        return None
    t_a = lambda c: (c + 4.5) * T_A
    rs = RF_STAGES[int(sg[2])]
    t_d = lambda c: (c + 2.0) * T_D
    if g[11] in (GAT3, GAT4):
        cls = int(g[18])
        dec = s32(int(g[15]) - cls)
        gate_out = s32(int(g[16]) + int(g[7]) + 1 - cls)   # marker + (marker -> pulse at input) + output reg
    else:
        mk = delta + 1
        dec = mk + delay + 2 - s32(int(g[6]))
        gate_out = mk + int(g[7]) + 1
    last_ro, res_t = t_a(last - 1), t_a(res)           # last word left the gate one register after the readout
    S = dict(S_nn=res_t - last_ro, S_cdc=t_d(dec) - res_t, S_gate=t_d(gate_out) - t_d(dec),
             S_conv=t_a(e3 - rs) - t_d(gate_out), S_fe=(e2 - 1 - (e3 - rs)) * T_A)
    S["sum"] = sum(S.values())
    return S

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=40, help="natural shots per slack point")
    ap.add_argument("--slacks", default="-0.30,-0.25,-0.20,-0.15,-0.10,-0.05,0.0")
    ap.add_argument("--margin", type=float, default=0.035, help="t_dec after the window end (us)")
    ap.add_argument("--n-cal", type=int, default=8)
    ap.add_argument("--pre", type=float, default=0.05, help="capture 2 opens this long before t_dec (us)")
    ap.add_argument("--out", default="campaign_v140L1/A_L6b_gate_probe.npz")
    ap.add_argument("--nn-advance", type=float, default=0.0,
                    help="boxcar path: NN_live's input runs this much (us) ahead of "
                         "readout 0's FIR; sets box_config sel = 1. Use 0.1135 (55 ticks, 111.9 ns); "
                         "box_probe.py measures the path at ~115 ns")
    ap.add_argument("--box-shift", type=int, default=7)
    ap.add_argument("--box-m", type=int, default=11)
    ap.add_argument("--split", action="store_true",
                    help="v1.40L9: record seg_stamp SEG2 + gate GAT3 timestamps and split the loop")
    ap.add_argument("--rf-thr", type=int, default=0, help="rf_tap_0 threshold (0: calibrate)")
    ap.add_argument("--lbuf-ref", choices=L.LBUF_REFS, default=L.LBUF_REF,
                    help="L_buf reference: last window sample instant (default) or window end "
                         "(1/fs = 1.808 ns smaller)")
    a = ap.parse_args()
    L.LBUF_REF = a.lbuf_ref
    print(f"L_buf reference: {L.LBUF_REF}")
    if a.nn_advance:
        print("boxcar path:", soc.box_config(sel=1, shift=a.box_shift, m=a.box_m))
    elif hasattr(soc, "box_present") and soc.box_present():
        print("FIR path:", soc.box_config(sel=0))
    if not soc.gate_present():
        raise SystemExit("no dac_gate BRAM: needs v1.40L6 and patch_nameserver_gate.py")
    print(f"gate pin {GATE_PIN} = {PINS[GATE_PIN]}, NN pin {A.NN_TRIGGER_PIN} = {PINS[A.NN_TRIGGER_PIN]}; "
          f"clk_dac2 {F_DAC:.2f} MHz")

    k, c = C.calib.load_kc(strict=False)
    X_all = np.load("data/test20k_real_X.npy", mmap_mode="r")
    X = np.asarray(X_all[:a.shots + a.n_cal])
    D = h.prepare_chunk(X, k, c) * (S.raw_peak(X, k, c) / S.raw_peak(X_all, k, c))
    n_win = max(4, int(0.5 * 0.2 * FS_RO))
    base = dict(cp_gain0=0.0, cp_gain1=0.6, cp_len=0.2, cp_pre=a.pre, trig_width=0.5,
                fb_margin=a.margin, fb_thr=16, win_extra=0.05, nn_adv=a.nn_advance, split=a.split)
    slacks = [float(x) for x in a.slacks.split(",")]

    # 1. marker -> pulse at the gate input, gate passing everything: enabled with every pass bit set rather than
    # disabled, because with FAIL_SAFE=1 windows blank while ENABLE = 0
    soc.gate_write(0, cfg_word(en=1, pass1=1, pass0=1, pass_stale=1))
    mp, plats = [], []
    for i in range(a.n_cal):
        r = one(D[i], dict(base, cp_slack=slacks[-1]), n_win)
        mp.append(r["gate"][7]); plats.append(r["plat"])
    mp = np.array(mp)
    if np.any(mp == 0xFFFFFFFF):
        raise SystemExit(f"gate saw no pulse after the marker in some calibration shots: {mp.tolist()}")
    delay = int(np.min(mp)) - DELAY_LEAD
    if not 0 <= delay <= 255:
        raise SystemExit(f"marker -> pulse {mp.tolist()} cycles: DELAY {delay} out of range")
    plat_pass = float(np.median(plats))
    # the record (magic included) is written at each window's close, so read it now
    gmag = soc.gate_stats()[11]
    gat2 = gmag in (GAT2, GAT3, GAT4)
    gat4 = gmag == GAT4
    print(f"gate magic {gmag:08x} ({'v1.41' if gat4 else 'v1.40L9' if gmag == GAT3 else 'v1.40L6b' if gat2 else 'v1.40L6'})")
    if gat4:
        g10 = int(soc.gate_stats()[10])
        print(f"gate build parameters: EARLY_BLANK {g10 >> 8 & 1}  SHOT_BIND {g10 >> 9 & 1}  "
              f"FAIL_SAFE {g10 >> 10 & 1}  CFG_SAFE {g10 >> 11 & 1}")
    rf_thr = 0
    seg_thr = 0
    if a.split:
        if not gat2 or soc.seg_latest()[2] not in RF_STAGES:
            raise SystemExit("--split needs v1.40L9+ (seg_stamp SEG2/SEG3, gate GAT2/GAT3/GAT4) and patch_nameserver_L9.py")
        if gmag not in (GAT3, GAT4):
            print("split: gate is GAT2 (no words 15-18); DAC-side times from the programmed trigger times")
        # seg edges on |I|+|Q| of NN_live's input stream: half that plateau, so the 50% point at any carrier phase
        seg_thr = seg_threshold([h.prepare_waveform(D[i]) for i in range(4)], dict(base, cp_slack=slacks[-1]))
        soc.seg_write(1, seg_thr)
        # rf_tap threshold: the largest power of 2 at which every calibration shot still
        # shows the conditional pulse at the tile output (edge3), divided by 4
        rf_thr = a.rf_thr
        if not rf_thr:
            best = 0
            for thr in [2 ** k for k in range(6, 15)]:
                soc.seg_write(4, thr)
                ok = all(((one(D[i], dict(base, cp_slack=slacks[-1]), n_win)["seg"][10] >> 6) & 1)
                         for i in range(4))
                if not ok:
                    break
                best = thr
            if best == 0:
                raise SystemExit("rf_tap saw no conditional pulse at the tile output at threshold 64")
            rf_thr = best // 4
        soc.seg_write(4, rf_thr)
        print(f"split: seg thr {seg_thr} (half the |I|+|Q| plateau), rf_tap thr {rf_thr} (|x|, raw ADC)")
    print(f"marker -> pulse at the gate input: {sorted(set(mp.tolist()))} cycles "
          f"({np.median(mp) * 1e3 / F_DAC:.1f} ns); DELAY = {delay}; always-pass plateau {plat_pass:.0f}")

    # 2./3. slack sweep, gate enabled
    soc.gate_write(0, cfg_word(en=1, pass1=1, pass0=0, pass_stale=0, delay=delay))
    rows = []
    print(f"\n{'slack us':>8} {'stale':>7} {'fresh':>7} {'correct':>9} {'passed':>7} {'late':>5} {'unsafe':>6} "
          f"{'margin ns (med, min)':>21} {'L_buf ns (passed: med [min, max])':>34}")
    for sl in slacks:
        late0 = int(soc.gate_stats()[12]) if gat2 else 0
        rs = [one(D[a.n_cal + i], dict(base, cp_slack=sl), n_win) for i in range(a.shots)]
        g = np.array([r["gate"] for r in rs])
        late = np.diff(np.r_[late0, g[:, 12]]) if gat2 else np.zeros(len(rs), dtype=np.int64)
        flags = g[:, 10]
        passed = (flags & 1).astype(bool); stale = (flags & 2).astype(bool); gbit = ((flags >> 2) & 1).astype(int)
        came = np.array([r["plat"] > 0.5 * plat_pass for r in rs])
        fresh = np.array([r["fresh"] for r in rs]); dec = np.array([r["dec"] for r in rs])
        Lb = np.array([r["L_buf"] for r in rs])
        correct = (came == passed) & (~stale) & (gbit == dec)
        unsafe = came & ~((~stale) & (gbit == dec) & (dec == 1))
        marg = np.array([s32(v) for v in g[:, 6]], dtype=float)
        marg[stale] = np.nan
        mtxt = (f"{np.nanmedian(marg) * 1e3 / F_DAC:7.1f}, {np.nanmin(marg) * 1e3 / F_DAC:6.1f}"
                if np.isfinite(marg).any() else "      -")
        ltxt = (f"{np.median(Lb[passed & came]):6.1f} [{Lb[passed & came].min():6.1f}, {Lb[passed & came].max():6.1f}]"
                if (passed & came).any() else "      -")
        print(f"{sl:8.3f} {int(stale.sum()):3d}/{len(rs):<3d} {int(fresh.sum()):3d}/{len(rs):<3d} "
              f"{int(correct.sum()):4d}/{len(rs):<4d} {int(passed.sum()):3d}/{len(rs):<3d} {int(late.sum()):5d} {int(unsafe.sum()):6d} {mtxt:>21} {ltxt:>34}",
              flush=True)
        segs = np.array([r["seg"] for r in rs])
        rows.append(dict(slack=sl, gate=g, seg=segs, came=came, fresh=fresh, dec=dec, L_buf=Lb, late=late, unsafe=unsafe))
        if a.split and (passed & came).any():
            full = dict(CFG0, **base, cp_slack=sl)
            t_cls = full["adc_trig_offset"] - full["nn_adv"]
            t_dec = t_cls + C.live_len(full) + full["fb_margin"]
            delta = round((t_dec + sl) / L.TICK) - round(t_cls / L.TICK)
            sp = [split_segments(g[i], segs[i], delta, delay) for i in np.flatnonzero(passed & came)]
            sp = [x for x in sp if x is not None]
            if sp:
                med = {k: float(np.median([x[k] for x in sp])) for k in sp[0]}
                print("         split (ns, median of %d passed shots): " % len(sp)
                      + "  ".join(f"{k} {v:6.1f}" for k, v in med.items())
                      + f"  | L_buf {np.median(Lb[passed & came]):6.1f}", flush=True)
    soc.gate_write(0, cfg_word(en=0, delay=delay))
    if gat4:
        gs = soc.gate_stats()
        if len(gs) > 20:
            print(f"\nshot binding (v1.41): decisions rejected (word 19) {int(gs[19])}; expected shot field now {int(gs[20])}")
        else:
            print("\nshot binding: gate_stats() returned 19 words -- apply patch_nameserver_v141.py to read words 19-20")
    print("\nstale: no fresh decision when the window opened (pulse blanked).  correct: pulse came back"
          " exactly when the gate passed, the gate's decision bit equals NN_live's record, not stale."
          "  margin: window start - decision arrival at the gate.  late: decision after its window opened"
          " (discarded, v1.40L6b).  unsafe: pulse played without this shot's own decision = 1.  gate left disabled.")
    np.savez(a.out, slacks=np.array(slacks), delay=delay, marker_to_pulse=mp, plat_pass=plat_pass, rf_thr=rf_thr, seg_thr=seg_thr,
             lbuf_ref=L.LBUF_REF, fs_ro=FS_RO, nn_adv=a.nn_advance, margin=a.margin,
             **{f"{k}_{i}": v for i, r in enumerate(rows) for k, v in r.items() if k != "slack"})
    print("saved", a.out)

if __name__ == "__main__":
    main()
