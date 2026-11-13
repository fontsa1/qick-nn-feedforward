"""Leaked energy over blanked shots with a shaped (Gaussian) conditional pulse.

"0 unsafe" is judged at the 50% plateau, so it cannot see a low-amplitude edge of a pulse that the gate
blanks a few cycles late. This measures the energy that comes back when the gate blanks every window, as a
fraction of the full pulse's energy:
  E_off   pulse gain 0 (gate setting irrelevant): the floor (replayed trace tail + noise)
  E_pass  gate enabled, every pass bit set: the whole Gaussian
  E_blank(lead)  gate enabled, PASS1 = PASS0 = 0 (blank every window), DELAY = marker->pulse - lead
leak(lead) = (E_blank - E_off) / (E_pass - E_off), same shots in every mode, energy = sum |IQ|^2 over the
pulse-centred capture. Leads under test: 2, 3, 4 (DELAY_LEAD 4 is the deployed setting). Negative leads open
the window AFTER the pulse reaches the gate: the positive control, which must show a leak growing with
|lead|, so that a zero at 2-4 means "none" and not "cannot see it".
"""
import argparse
import json
import sys

import numpy as np

sys.path.insert(0, "analysis_scripts/diagnostics")
import gate_probe as G                          # noqa: E402  (brings up soc)

C, A, h, S, soc, soccfg, L = G.C, G.A, G.h, G.S, G.soc, G.soccfg, G.L
CFG0 = G.CFG0

class GaussGateProg(G.GateProg):
    """GateProg whose conditional pulse c1 is a Gaussian (sigma = cp_len / 6) instead of a constant."""

    def _initialize(s, cf):
        A.Prog._initialize(s, cf)
        s.add_reg(name="cp_addr")
        s.add_pulse(ch=cf["gen_ch"], name="c0", style="const", length=cf["cp_len"],
                    freq=cf["freq"], phase=0.0, gain=cf["cp_gain0"], ro_ch=cf["ro_ch"])
        s.add_gauss(ch=cf["gen_ch"], name="cp_env", sigma=cf["cp_len"] / 6, length=cf["cp_len"])
        s.add_pulse(ch=cf["gen_ch"], name="c1", style="arb", envelope="cp_env",
                    freq=cf["freq"], phase=0.0, gain=cf["cp_gain1"], ro_ch=cf["ro_ch"])

def shot(D_i, cf):
    wf = C.trim_wf(h.prepare_waveform(D_i))
    cfg = dict(CFG0, waveform_I=wf[0::2], waveform_Q=wf[1::2], **cf)
    iq = np.asarray(GaussGateProg(soccfg, cfg).acquire_decimated(soc, progress=False))
    cap = iq.reshape(-1, iq.shape[-2], 2)[-1].astype(float)
    g = np.array(soc.gate_stats(), dtype=np.int64)
    return cap, g

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=40, help="shots per mode / lead")
    ap.add_argument("--leads", default="-36,-24,-12,-6,0,2,3,4")
    ap.add_argument("--nn-advance", type=float, default=0.1135)
    ap.add_argument("--slack", type=float, default=0.0)
    ap.add_argument("--nz-thr", type=int, default=8,
                    help="gate arrival threshold (cfg word [31:16]) used to measure marker -> pulse. 8 finds a "
                         "Gaussian's first nonzero samples; 2000 (every earlier gate script) finds it ~2 sigma "
                         "before its peak, ~19 cycles late, and lets the leading edge through (0.28%% at lead 4, "
                         "v1.41 dry run 2026-11-03)")
    ap.add_argument("--out", default="campaign_v140L1/A_v142_leak_energy.json")
    a = ap.parse_args()
    print("boxcar path:", soc.box_config(sel=1, shift=7, m=11))
    leads = [int(x) for x in a.leads.split(",")]
    k, c = C.calib.load_kc(strict=False)
    X_all = np.load("data/test20k_real_X.npy", mmap_mode="r")
    X = np.asarray(X_all[:a.shots])
    D = h.prepare_chunk(X, k, c) * (S.raw_peak(X, k, c) / S.raw_peak(X_all, k, c))
    base = dict(cp_gain0=0.0, cp_gain1=0.6, cp_len=0.2, cp_pre=0.15, trig_width=0.5, fb_margin=0.035,
                fb_thr=16, win_extra=0.05, nn_adv=a.nn_advance, split=False, cp_slack=a.slack)

    # marker -> pulse at the gate input (cycles), with the Gaussian pulse
    soc.gate_write(0, G.cfg_word(en=1, pass1=1, pass0=1, pass_stale=1, nz_thr=a.nz_thr))
    mp = np.array([shot(D[i], base)[1][7] for i in range(8)])
    if np.any(mp == 0xFFFFFFFF):
        raise SystemExit(f"gate saw no pulse after the marker: {mp.tolist()}")
    m0 = int(mp.min())
    print(f"marker -> pulse at the gate input: {sorted(set(mp.tolist()))} cycles")
    leads = [ld for ld in leads if 0 <= m0 - ld <= 255]

    def energies(cf, word0):
        soc.gate_write(0, word0)
        caps = [shot(D[i], cf)[0] for i in range(a.shots)]
        return np.array([np.sum(cp[:, 0] ** 2 + cp[:, 1] ** 2) for cp in caps]), caps

    E_off, _ = energies(dict(base, cp_gain1=0.0), G.cfg_word(en=1, pass1=1, pass0=1, pass_stale=1, delay=m0 - 4, nz_thr=a.nz_thr))
    E_pass, caps_pass = energies(base, G.cfg_word(en=1, pass1=1, pass0=1, pass_stale=1, delay=m0 - 4, nz_thr=a.nz_thr))
    full = np.median(E_pass - E_off)
    print(f"E_pass - E_off: median {full:.4g} (per-shot spread {np.std(E_pass - E_off) / full:.3f} of it); "
          f"E_off median {np.median(E_off):.4g}")
    res = dict(nz_thr=a.nz_thr, marker_to_pulse=mp.tolist(), shots=a.shots, sigma_us=base["cp_len"] / 6, cp_len_us=base["cp_len"],
               E_off=E_off.tolist(), E_pass=E_pass.tolist(), leads={})
    print(f"\n{'lead':>5} {'DELAY':>5}  leak fraction: median  mean [95% CI of the mean]   max shot")
    for ld in leads:
        Eb, _ = energies(base, G.cfg_word(en=1, pass1=0, pass0=0, pass_stale=0, delay=m0 - ld, nz_thr=a.nz_thr))
        f = (Eb - E_off) / full
        mean, se = float(np.mean(f)), float(np.std(f, ddof=1) / np.sqrt(len(f)))
        res["leads"][str(ld)] = dict(delay=m0 - ld, E_blank=Eb.tolist(), frac_median=float(np.median(f)),
                                     frac_mean=mean, frac_ci=[mean - 1.96 * se, mean + 1.96 * se],
                                     frac_max=float(np.max(f)))
        print(f"{ld:5d} {m0 - ld:5d}  {np.median(f):+.5f}  {mean:+.5f} [{mean - 1.96 * se:+.5f}, {mean + 1.96 * se:+.5f}]"
              f"  {np.max(f):+.5f}", flush=True)
    soc.gate_write(0, G.cfg_word(en=0, delay=m0 - 4, nz_thr=a.nz_thr))
    tested = [res["leads"][str(ld)] for ld in (2, 3, 4) if str(ld) in res["leads"]]
    ctrl = [res["leads"][str(ld)] for ld in leads if ld < 0]
    # the control resolves a leak if some negative lead's CI lies entirely above zero and above every
    # tested lead's upper bound
    res["control_sees_leak"] = bool(ctrl and tested) and max(x["frac_ci"][0] for x in ctrl) > max(
        0.0, max(x["frac_ci"][1] for x in tested))
    res["max_upper_ci_lead_2_4"] = max(x["frac_ci"][1] for x in tested) if tested else None
    print(f"\nleads 2-4: largest upper 95% bound on the leaked fraction {res['max_upper_ci_lead_2_4']:+.5f}; "
          f"positive control (negative leads) resolves a leak: {res['control_sees_leak']}  (gate left disabled)")
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print("saved", a.out)

if __name__ == "__main__":
    main()
