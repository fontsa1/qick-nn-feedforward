"""The tProc-programmed feed-forward path on v1.43: the program waits for NN_live's result, reads it, and plays the conditional
pulse itself, with no gate. The gate's counterpart is L_50 (Fig. 8). The paper's earlier 454.5 ns dates from the L5 build, on the FIR path.

Same measurement as latency_loop.l_buf and gate_probe.one: window's last sample to the pulse's 50% point, both on readout 0's
(FIR) time base. Boxcar path (the deployed one, --path box): NN_live's trigger and t_dec move earlier by nn_adv, as in
gate_probe.GateProg. FIR path (--path fir): the original program, for comparison with L5. On the FIR path the boxcar network
scores a stream it was not trained on, so only the timing means anything there.

For each variant and margin: natural decisions, freshness, branch == decision, played pulse == branch, and forced branches.
Reports the earliest margin where every shot is fresh and correct.

"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, "analysis_scripts/diagnostics")
import latency_loop as L                        # noqa: E402  (brings up soc)

C, h, soc, soccfg, CFG0, A = L.C, L.h, L.soc, L.soccfg, L.CFG0, L.A
BANNER = os.environ.get("QICK_BANNER", "Nov  4 23:39:37")

class AdvProg(L.LatProg):
    """LatProg with NN_live's trigger and the decision wait moved earlier by nn_adv."""

    def _body(s, cf):
        adv = cf["nn_adv"]
        s.pulse(ch=cf["gen_ch"], name="p")
        s.send_readoutconfig(ch=cf["ro_ch"], name="r", t=0)
        s.trigger(ros=[cf["ro_ch"]], t=cf["adc_trig_offset"], width=cf["trig_width"])
        s.trigger(pins=[0, A.NN_TRIGGER_PIN], t=cf["adc_trig_offset"] - adv, width=10)
        t_dec = cf["adc_trig_offset"] - adv + C.live_len(cf) + cf["fb_margin"]
        t_cp = t_dec + cf["cp_slack"]
        s.trigger(ros=[cf["ro_ch"]], t=t_dec + cf["adc_trig_offset"] - cf["cp_pre"], width=cf["trig_width"])
        v = cf["variant"]
        if v == "nobranch":
            s._body_nobranch(cf, t_dec, t_cp)
            return
        s.wait(t=t_dec)
        s.asm_inst({"CMD": "DPORT_RD", "DST": str(C.LIVE_PORT)})
        s.cond_jump("fb_one", "s_port_l", "NS", "-", cf["fb_thr"])
        s.pulse(ch=cf["gen_ch"], name="c0", t=t_cp)
        s.write_dmem(2, 0)
        s.jump("fb_done")
        s.label("fb_one")
        s.pulse(ch=cf["gen_ch"], name="c1", t=t_cp)
        s.write_dmem(2, 1)
        s.label("fb_done")
        s.write_dmem(0, "s_port_l")
        s.write_dmem(1, "s_port_h")

def shot(wf, cf, n_win):
    wf = C.trim_wf(wf)
    cfg = dict(CFG0, waveform_I=wf[0::2], waveform_Q=wf[1::2], **cf)
    iq = np.asarray(AdvProg(soccfg, cfg).acquire_decimated(soc, progress=False))
    iq = iq.reshape(-1, iq.shape[-2], 2)
    m = np.asarray(soc.tproc.read_mem("dmem", length=3)).reshape(-1).astype(np.int64)
    rec = C.live_record(C.shot_window(soc.live_read_block(0, 3)))
    on, pk = L.onset_interp(iq[-1], n_win)
    full = dict(CFG0, **cf)
    t_dec = full["adc_trig_offset"] - full["nn_adv"] + C.live_len(full) + full["fb_margin"]
    t2 = round((t_dec + full["adc_trig_offset"] - full["cp_pre"]) / L.TICK) * L.TICK
    t_dec0 = full["adc_trig_offset"] + C.live_len(full) + full["fb_margin"]     # l_buf's own, unadvanced reference
    arr = (t2 - t_dec0) + on / L.FS_RO
    return arr, pk, int(m[2]), int(rec[7] & 1), (int(np.uint32(m[1])) & C.SHOT_MASK) == rec[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", choices=("box", "fir"), default="box")
    ap.add_argument("--variants", default="nobranch,qick")
    ap.add_argument("--margins", default="0.02,0.03,0.04,0.05,0.06,0.07,0.08")
    ap.add_argument("--shots", type=int, default=40)
    ap.add_argument("--n-force", type=int, default=20)
    ap.add_argument("--advance", type=float, default=0.1135)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    banner = str(soc.get_cfg().get("fw_timestamp", "")) if hasattr(soc, "get_cfg") else ""
    print("BANNER:", banner, flush=True)
    assert BANNER in banner, f"not v1.43: {banner!r}"
    assert os.environ.get("QICK_LIVE_N") == "576", "set QICK_LIVE_N=576"
    adv = a.advance if a.path == "box" else 0.0
    print("readout path:", soc.box_config(sel=1, shift=7, m=11) if a.path == "box" else soc.box_config(sel=0), flush=True)
    k, c = C.calib.load_kc(strict=False)
    X_all = np.load("data/test20k_real_X.npy", mmap_mode="r")
    n = max(a.shots, a.n_force)
    X = np.asarray(X_all[:n])
    D = h.prepare_chunk(X, k, c) * (C.S.raw_peak(X, k, c) / C.S.raw_peak(X_all, k, c))
    WF = [h.prepare_waveform(d) for d in D]
    n_win = int(0.5 * 0.2 * L.FS_RO)
    base = dict(cp_gain0=0.2, cp_gain1=0.6, cp_len=0.2, cp_pre=0.05, trig_width=0.5, cp_slack=-0.2, nn_adv=adv)
    out = dict(banner=banner, path=a.path, advance=adv, host=os.environ.get("QICK_NS_HOST"), points=[])
    for v in a.variants.split(","):
        for mg in (float(x) for x in a.margins.split(",")):
            cf = dict(base, variant=v, fb_margin=mg)
            Lf, lv = {}, {}
            ok_force = True
            for want, thr in ((0, C.FORCE_0), (1, C.FORCE_1)):
                res = [shot(WF[i], dict(cf, fb_thr=thr), n_win) for i in range(a.n_force)]
                ok_force &= {r[2] for r in res} == {want}
                Lf[want] = L.l_buf(cf, np.array([r[0] for r in res])); lv[want] = float(np.mean([r[1] for r in res]))
            mid = 0.5 * (lv[0] + lv[1])
            res = [shot(WF[i], dict(cf, fb_thr=16), n_win) for i in range(a.shots)]
            Lb = L.l_buf(cf, np.array([r[0] for r in res]))
            br = np.array([r[2] for r in res]); dec = np.array([r[3] for r in res]); fresh = np.array([r[4] for r in res])
            played = np.array([r[1] > mid for r in res]).astype(int)
            p = dict(variant=v, margin=mg, n=a.shots, forced_ok=bool(ok_force), L_forced0=float(np.nanmedian(Lf[0])), L_forced1=float(np.nanmedian(Lf[1])),
                     L_median=float(np.nanmedian(Lb)), L_min=float(np.nanmin(Lb)), L_max=float(np.nanmax(Lb)), L_sd=float(np.nanstd(Lb)),
                     fresh=int(fresh.sum()), br_eq_dec=int((fresh & (br == dec)).sum()), played_eq_br=int((played == br).sum()))
            out["points"].append(p)
            print(f"{v:>9} margin {mg:+.3f}: L {p['L_median']:.1f} [{p['L_min']:.1f}, {p['L_max']:.1f}]  forced {p['L_forced0']:.1f} / {p['L_forced1']:.1f}  "
                  f"fresh {p['fresh']}/{a.shots}  br==dec {p['br_eq_dec']}  played==br {p['played_eq_br']}", flush=True)
            json.dump(out, open(a.out, "w"), indent=1)
    for v in a.variants.split(","):
        good = [p for p in out["points"] if p["variant"] == v and p["fresh"] == p["n"] == p["br_eq_dec"] == p["played_eq_br"]]
        print(f"{v}: earliest clean margin:", f"{good[0]['margin']:+.3f}, L {good[0]['L_median']:.1f} ns" if good else "none", flush=True)
    soc.box_config(sel=1, shift=7, m=11)
    print("saved", a.out, "; boxcar path restored")

if __name__ == "__main__":
    main()
