"""Conditional-pulse demo through the tProc: the tProc plays a DAC pulse chosen by NN_live's decision, and the pulse
is captured in loopback.

Per shot, on gen 0 / ro 0 (the loopback pair):
  1. the dataset waveform + readout trigger (NN_live classifies this window and feeds tProc input 3)
  2. wait until (trigger + readout_length + margin); DPORT_RD 3; TEST s_port_l - thr ; JUMP NS
  3. branch 0 plays const pulse "c0" (gain --g0), branch 1 plays "c1" (--g1), both scheduled at the same tProc time
     t_cp = t_dec + --slack
  4. a second readout trigger on ro 0 captures the conditional pulse
The host then checks, per shot:
  played              which pulse the ADC saw (plateau amplitude vs the midpoint of the two levels, measured first
                      on forced-branch shots)
  played == branch    the DAC did what the tProc decided
  branch == decision  the tProc decided what NN_live decided (fresh shots)
The record of the classified window is read by its shot number from the 512-record ring (NN_live.h).

The readout trigger is narrowed to --trig-width (A.Prog holds it 10 us, and a second trigger on the same FIFO port
would then fire ~10 us late). The pulse is scheduled with slack, not as soon as possible, so this demonstrates
actuation, not latency (the captures are saved for latency analysis).
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, "analysis_scripts/diagnostics")
sys.path.insert(0, "board_tools")
sys.path.insert(0, ".")
import calib                                    # noqa: E402
import acquire_classifier_input_captures as A   # noqa: E402  (brings up soc)
import score_armed_fidelity as S                # noqa: E402  (raw_peak, wilson)

soc, soccfg = A.soc, A.soccfg
h = A.h
CFG0 = A.CFG0
LIVE_PORT = 3                                   # tProc input fed by axis_cc_live
SHOT_MASK = (1 << 29) - 1
REC_BASE, REC_N, REC_W = 1024, 512, 8           # NN_live.h result ring
FORCE_1, FORCE_0 = -40000, 40000                # thresholds outside the Q.10 logit range
FS_RO = soccfg["readouts"][0]["fs"] / 8         # decimated readout rate, MHz

def live_len(cf):
    """Programmed window end for the live decision, us after the readout trigger.
    NN_live frames QICK_LIVE_N samples (env, 576 in the deployed build) at the decimated rate; unset, the readout
    length (770 samples)."""
    import os
    n = os.environ.get("QICK_LIVE_N")
    return cf["readout_length"] if not n else int(n) / (soccfg["readouts"][0]["fs"] / 8)

# With a shorter live window (QICK_LIVE_N), the first readout must end with it:
# a 770-sample readout would still be running when the second capture (the
# conditional pulse) is triggered, and QICK queues that capture behind it
# ("Readout time 1 appears to conflict with previous readout"), which breaks
# the loop-latency timing. The readout then equals the live window, as on L2.
import os as _os                                # noqa: E402
if _os.environ.get("QICK_LIVE_N"):
    CFG0 = dict(CFG0, readout_length=live_len(CFG0))

def trim_wf(wf):
    """With QICK_LIVE_N, play only the part of the input trace inside the live
    window (rounded up to 16 generator samples): the rest is never classified,
    and on a qubit the readout pulse would stop with the window. Without the
    trim, the trace is still playing when the conditional pulse arrives (~350 ns
    earlier than on L2) and buries the small branch-0 pulse in capture 2."""
    n = _os.environ.get("QICK_LIVE_N")
    if not n:
        return wf
    m = len(wf) // 2
    keep = min(m, int(np.ceil(m * int(n) / 770 / 16)) * 16)
    return wf[:2 * keep]

class CondProg(A.Prog):
    """A.Prog's pulse + readout, branch on tProc input 3, play c0 or c1, capture it."""

    def _initialize(s, cf):
        A.Prog._initialize(s, cf)
        for nm, g in (("c0", cf["cp_gain0"]), ("c1", cf["cp_gain1"])):
            s.add_pulse(ch=cf["gen_ch"], name=nm, style="const", length=cf["cp_len"],
                        freq=cf["freq"], phase=0.0, gain=g, ro_ch=cf["ro_ch"])

    def _body(s, cf):
        # A.Prog._body, except the readout trigger (port p0, also live_gate_0) is
        # cf["trig_width"] wide instead of 10 us: the tProc's trigger port is a
        # FIFO, so a second p0 trigger queued behind a 10 us clear fires ~10 us
        # late. The NN_axi trigger (its own port) keeps A.Prog's width.
        s.pulse(ch=cf["gen_ch"], name="p")
        s.send_readoutconfig(ch=cf["ro_ch"], name="r", t=0)
        s.trigger(ros=[cf["ro_ch"]], t=cf["adc_trig_offset"], width=cf["trig_width"])
        s.trigger(pins=[0, A.NN_TRIGGER_PIN], t=cf["adc_trig_offset"], width=10)  # port p4
        t_dec = cf["adc_trig_offset"] + live_len(cf) + cf["fb_margin"]
        t_cp = t_dec + cf["cp_slack"]
        s.wait(t=t_dec)
        s.asm_inst({"CMD": "DPORT_RD", "DST": str(LIVE_PORT)})
        s.cond_jump("fb_one", "s_port_l", "NS", "-", cf["fb_thr"])
        s.write_dmem(2, 0)
        s.pulse(ch=cf["gen_ch"], name="c0", t=t_cp)
        s.jump("fb_done")
        s.label("fb_one")
        s.write_dmem(2, 1)
        s.pulse(ch=cf["gen_ch"], name="c1", t=t_cp)
        s.label("fb_done")
        s.write_dmem(0, "s_port_l")
        s.write_dmem(1, "s_port_h")
        s.trigger(ros=[cf["ro_ch"]], t=t_cp + cf["adc_trig_offset"] - cf["cp_pre"],
                  width=cf["trig_width"])

_win_last_n = None
_win_back = 0          # records after this shot's classified window (0, or 1 on builds that framed every trigger)

def shot_window(hdr):
    """NN_live record number of this shot's classified window, from the header read after the shot.
    With the window gate on the classify trigger only (deployed build) it is hdr[0]; on builds whose gate framed every
    readout trigger, a feed-forward shot left two records and it is hdr[0] - 1. The layout is taken from the record
    count's step between calls (1 or 2 per shot); until one is seen, the last known layout is assumed."""
    global _win_last_n, _win_back
    n = int(hdr[0])
    if _win_last_n is not None and n - _win_last_n in (1, 2):
        _win_back = n - _win_last_n - 1
    _win_last_n = n
    return n - _win_back

def live_record(n):
    """NN_live's record for shot n (1-based), from the ring."""
    return np.asarray(soc.live_read_block(REC_BASE + ((n - 1) % REC_N) * REC_W, REC_W),
                      dtype=np.int64)

def plateau(cap, n_win):
    """Peak sliding-mean |IQ| over n_win samples, and the 50%-crossing index."""
    mag = np.hypot(cap[:, 0], cap[:, 1])
    k = np.ones(n_win) / n_win
    sm = np.convolve(mag, k, mode="valid")
    p = float(sm.max())
    above = np.nonzero(mag >= 0.5 * p)[0]
    return p, (int(above[0]) if len(above) else -1)

def shot(wf, cfg_extra):
    wf = trim_wf(wf)
    cfg = dict(CFG0, waveform_I=wf[0::2], waveform_Q=wf[1::2], **cfg_extra)
    iq = np.asarray(CondProg(soccfg, cfg).acquire_decimated(soc, progress=False))
    iq = iq.reshape(-1, iq.shape[-2], 2)        # (2 reads, length, IQ)
    m = np.asarray(soc.tproc.read_mem("dmem", length=3)).reshape(-1)
    hdr = soc.live_read_block(0, 3)
    n1 = shot_window(hdr)                       # this shot's classified window
    return iq[-1], np.int32(m[0]), int(np.uint32(m[1])), int(m[2]), live_record(n1), hdr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=200)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--dataset", default="test20k_real_X")
    ap.add_argument("--margin", type=float, default=0.1,
                    help="us after the nominal window end before DPORT_RD (fresh from 0.02)")
    ap.add_argument("--slack", type=float, default=0.2,
                    help="us from DPORT_RD to the scheduled conditional pulse")
    ap.add_argument("--thr", type=int, default=16,
                    help="branch on s_port_l >= thr (Q.10; 16 = NN_live's own rule)")
    ap.add_argument("--g0", type=float, default=0.2, help="gain of the branch-0 pulse")
    ap.add_argument("--g1", type=float, default=0.6, help="gain of the branch-1 pulse")
    ap.add_argument("--cp-len", type=float, default=0.2, help="conditional pulse length, us")
    ap.add_argument("--pre", type=float, default=0.1,
                    help="us the second capture opens before the pulse's expected arrival")
    ap.add_argument("--trig-width", type=float, default=0.5,
                    help="us; readout-trigger (p0) pulse width, must clear before the 2nd trigger")
    ap.add_argument("--n-force", type=int, default=5, help="forced shots per branch")
    ap.add_argument("--out", default="campaign_v140L1/A_L2_condpulse.npz")
    a = ap.parse_args()

    if not soc.live_present():
        raise SystemExit("no live core on this bitstream / NNQickSoc unpatched")
    banner = [l for l in str(soccfg).splitlines() if "built" in l]
    print("BANNER:", banner[0].strip() if banner else "?")
    print("tProc input ports:", soccfg["tprocs"][0].get("in_port_qty", "?"))

    xf, yf = {nm: (x_, y_) for nm, x_, y_ in A.DATASETS}[a.dataset]
    k, c = calib.load_kc(strict=False)
    X_all = np.load(xf, mmap_mode="r")
    X = np.asarray(X_all[a.start:a.start + a.shots])
    D = h.prepare_chunk(X, k, c) * (S.raw_peak(X, k, c) / S.raw_peak(X_all, k, c))
    Y = np.load(yf)[a.start:a.start + a.shots].astype(int)
    n_win = max(4, int(0.5 * a.cp_len * FS_RO))
    base = dict(fb_margin=a.margin, cp_slack=a.slack, cp_gain0=a.g0, cp_gain1=a.g1,
                cp_len=a.cp_len, cp_pre=a.pre, trig_width=a.trig_width)
    t_trig2 = (CFG0["adc_trig_offset"] + live_len(CFG0) + a.margin + a.slack
               + CFG0["adc_trig_offset"] - a.pre)
    if a.trig_width >= t_trig2 - CFG0["adc_trig_offset"]:
        raise SystemExit("--trig-width must end before the second readout trigger")

    # forced branches: the two pulse levels, independent of the classifier
    lvl = {}
    for want, thr in ((0, FORCE_0), (1, FORCE_1)):
        ps = []
        for i in range(a.n_force):
            cap, _, _, br, _, _ = shot(h.prepare_waveform(D[i % len(D)]), dict(base, fb_thr=thr))
            if br != want:
                raise SystemExit(f"forced branch {want} took branch {br} (thr {thr})")
            ps.append(plateau(cap, n_win)[0])
        lvl[want] = np.array(ps)
        print(f"forced branch {want}: plateau {lvl[want].mean():.1f} +- {lvl[want].std():.1f}")
    mid = 0.5 * (lvl[0].mean() + lvl[1].mean())
    sep = (lvl[1].mean() - lvl[0].mean()) / max(lvl[0].std() + lvl[1].std(), 1e-9)
    print(f"threshold {mid:.1f}; level separation {sep:.1f} (sum of stds)")
    if lvl[1].mean() <= lvl[0].mean() or sep < 3:
        raise SystemExit("pulse levels not separable; check --g0/--g1 and the capture window")

    n = len(D)
    caps = np.zeros((n, int(round(CFG0["readout_length"] * FS_RO)) + 8, 2))
    br = np.zeros(n, int); dec = np.zeros(n, int); fresh = np.zeros(n, bool)
    pk = np.zeros(n); onset = np.zeros(n, int); ferr = np.zeros(n, np.int64)
    for i in range(n):
        cap, lo, hi, br[i], rec, hdr = shot(h.prepare_waveform(D[i]), dict(base, fb_thr=a.thr))
        L = min(len(cap), caps.shape[1])
        caps[i, :L] = cap[:L]
        pk[i], onset[i] = plateau(cap, n_win)
        fresh[i] = (hi & SHOT_MASK) == rec[0]
        dec[i] = rec[7] & 1
        ferr[i] = hdr[1]
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{n}", flush=True)
    played = (pk > mid).astype(int)

    nf = int(fresh.sum())
    w = lambda kk, nn: S.wilson(kk, nn) if nn else (float("nan"), float("nan"))
    ok_dac = int((played == br).sum())
    ok_dec = int((fresh & (br == dec)).sum())
    ok_all = int((fresh & (played == dec)).sum())
    print(f"\nshots {n}   fresh {nf}/{n}   frame_err {int(ferr.max())}")
    print(f"played == branch (DAC did what the tProc decided): {ok_dac}/{n}  CI {w(ok_dac, n)}")
    print(f"branch == NN_live decision (fresh shots):         {ok_dec}/{nf}  CI {w(ok_dec, nf)}")
    print(f"played == NN_live decision (end to end, fresh):    {ok_all}/{nf}  CI {w(ok_all, nf)}")
    print(f"played == prepared state (readout accuracy):       {100 * np.mean(played == Y):.2f}%")
    print(f"branch 1 rate {br.mean():.3f}; pulse onset in capture: median "
          f"{np.median(onset)} samples ({1e3 * np.median(onset) / FS_RO:.1f} ns after the "
          f"second trigger; window opens {1e3 * a.pre:.0f} ns before nominal arrival)")
    np.savez(a.out, caps=caps, branch=br, decision=dec, fresh=fresh, plateau=pk,
             onset=onset, played=played, Y=Y, mid=mid, lvl0=lvl[0], lvl1=lvl[1],
             args=str(vars(a)), fs_ro=FS_RO, cfg=str(CFG0))
    print("saved", a.out)

if __name__ == "__main__":
    main()
