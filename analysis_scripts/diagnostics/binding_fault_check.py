"""Forced shot-binding fault on the dac_gate with SHOT_BIND = 1 (GAT4).

The fault: a second, narrow classify trigger (tProc trig_11) 0.4 us into NN_live's window. live_window_gate
ignores a trigger while it is filling a window (MEMW), so NN_live still produces ONE result, numbered for the
first trigger; dac_gate counts BOTH rising edges and expects the result of the second. The result's shot
field is therefore one behind: the gate must reject it (word 19 +1) and blank the pulse, and the binding must
be back in step at the next shot (dac_gate.v: exp_f = latest field + triggers since latest).

Gate policy for the test: ENABLE, PASS1 = PASS0 = 1, PASS_STALE = 0 -- any fresh, bound decision passes, so
a blanked pulse can only mean the decision was rejected or missing. Pattern per cycle: normal, FAULT,
normal (first after the fault), normal. Pass criteria:
  * fault shots: pulse never comes back, word 19 rises by exactly 1, flags show stale
  * normal shots (incl. the first after a fault): pulse comes back, word 19 unchanged, decision bit == NN_live
  * control run without faults: every pulse comes back, word 19 unchanged
"""
import argparse
import json
import sys

import numpy as np

sys.path.insert(0, "analysis_scripts/diagnostics")
import gate_probe as G                          # noqa: E402  (brings up soc)

C, A, h, S, soc, soccfg, L = G.C, G.A, G.h, G.S, G.soc, G.soccfg, G.L
CFG0 = G.CFG0

class FaultProg(G.GateProg):
    """GateProg with narrow classify triggers; cf["fault"] adds a second one cf["fault_dt"] into the window."""

    def _body(s, cf):
        s.pulse(ch=cf["gen_ch"], name="p")
        s.send_readoutconfig(ch=cf["ro_ch"], name="r", t=0)
        s.trigger(ros=[cf["ro_ch"]], t=cf["adc_trig_offset"], width=cf["trig_width"])
        t_cls = cf["adc_trig_offset"] - cf["nn_adv"]
        s.trigger(pins=[0, A.NN_TRIGGER_PIN], t=t_cls, width=cf["cls_width"])
        if cf.get("fault"):
            s.trigger(pins=[0, A.NN_TRIGGER_PIN], t=t_cls + cf["fault_dt"], width=cf["cls_width"])
        t_dec = t_cls + C.live_len(cf) + cf["fb_margin"]
        t_cp = t_dec + cf["cp_slack"]
        s.trigger(ros=[cf["ro_ch"]], t=t_dec + cf["adc_trig_offset"] - cf["cp_pre"], width=cf["trig_width"])
        s.pulse(ch=cf["gen_ch"], name="c1", t=t_cp)
        s.trigger(pins=[G.GATE_PIN], t=t_cp, width=cf["cp_len"] + cf["win_extra"])
        s.wait(t=t_cp + cf["cp_len"] + 0.1)
        s.asm_inst({"CMD": "DPORT_RD", "DST": str(C.LIVE_PORT)})
        s.write_dmem(0, "s_port_l")
        s.write_dmem(1, "s_port_h")

def shot(D_i, cf, n_win):
    wf = C.trim_wf(h.prepare_waveform(D_i))
    cfg = dict(CFG0, waveform_I=wf[0::2], waveform_Q=wf[1::2], **cf)
    iq = np.asarray(FaultProg(soccfg, cfg).acquire_decimated(soc, progress=False))
    cap = iq.reshape(-1, iq.shape[-2], 2)[-1]
    rec = C.live_record(C.shot_window(soc.live_read_block(0, 3)))
    g = np.array(soc.gate_stats(), dtype=np.int64)
    _, plat = L.onset_interp(cap, n_win)
    return dict(g=g, plat=float(plat), nn_shot=int(rec[0]), nn_dec=int(rec[7] & 1))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=25, help="normal/fault/normal/normal cycles")
    ap.add_argument("--control", type=int, default=40, help="shots in the no-fault control")
    ap.add_argument("--n-cal", type=int, default=8)
    ap.add_argument("--nn-advance", type=float, default=0.1135)
    ap.add_argument("--slack", type=float, default=0.0, help="cp_slack (us); 0 = comfortably fresh")
    ap.add_argument("--fault-dt", type=float, default=0.4, help="second trigger, us after the first")
    ap.add_argument("--out", default="campaign_v140L1/A_v142_binding_fault.json")
    a = ap.parse_args()
    print("boxcar path:", soc.box_config(sel=1, shift=7, m=11))
    if C.live_len(dict(CFG0)) <= a.fault_dt + 0.1:
        raise SystemExit("fault trigger would land after the window; lower --fault-dt")
    k, c = C.calib.load_kc(strict=False)
    X_all = np.load("data/test20k_real_X.npy", mmap_mode="r")
    n_need = a.n_cal + a.control + 4 * a.cycles
    X = np.asarray(X_all[:n_need])
    D = h.prepare_chunk(X, k, c) * (S.raw_peak(X, k, c) / S.raw_peak(X_all, k, c))
    n_win = max(4, int(0.5 * 0.2 * G.FS_RO))
    base = dict(cp_gain0=0.0, cp_gain1=0.6, cp_len=0.2, cp_pre=0.15, trig_width=0.5, fb_margin=0.035,
                fb_thr=16, win_extra=0.05, nn_adv=a.nn_advance, split=False, cp_slack=a.slack,
                cls_width=0.1, fault_dt=a.fault_dt, fault=False)

    soc.gate_write(0, G.cfg_word(en=1, pass1=1, pass0=1, pass_stale=1))
    cal = [shot(D[i], base, n_win) for i in range(a.n_cal)]
    mp = np.array([r["g"][7] for r in cal])
    delay = int(mp.min()) - G.DELAY_LEAD
    plat_pass = float(np.median([r["plat"] for r in cal]))
    magic = int(cal[-1]["g"][11])
    params = int(cal[-1]["g"][10]) >> 8 & 0xF
    if magic != G.GAT4 or not params >> 1 & 1:
        raise SystemExit(f"needs a GAT4 gate with SHOT_BIND = 1 (magic {magic:08x}, params {params:04b})")
    print(f"gate GAT4, params [EB SB FS CS] {[params >> b & 1 for b in range(4)]}; DELAY {delay}; "
          f"always-pass plateau {plat_pass:.0f}")
    soc.gate_write(0, G.cfg_word(en=1, pass1=1, pass0=1, pass_stale=0, delay=delay))

    def run(kinds, rows):
        out = []
        prev19 = int(soc.gate_stats()[19])
        for kind, i in zip(kinds, rows):
            r = shot(D[i], dict(base, fault=(kind == "fault")), n_win)
            w19 = int(r["g"][19])
            flags = int(r["g"][10])
            out.append(dict(kind=kind, row=int(i), came=bool(r["plat"] > 0.5 * plat_pass),
                            rej_delta=w19 - prev19, stale=bool(flags & 2), passed=bool(flags & 1),
                            gate_dec=int(flags >> 2 & 1), nn_dec=r["nn_dec"], nn_shot=r["nn_shot"],
                            exp_shot=int(r["g"][20])))
            prev19 = w19
        return out

    rows = np.arange(a.n_cal, a.n_cal + a.control)
    control = run(["control"] * a.control, rows)
    kinds = ["normal", "fault", "after", "normal"] * a.cycles
    rows = np.arange(a.n_cal + a.control, a.n_cal + a.control + len(kinds))
    test = run(kinds, rows)
    soc.gate_write(0, G.cfg_word(en=0, delay=delay))

    def summ(lst, kind):
        s = [x for x in lst if x["kind"] == kind]
        return dict(n=len(s), came=sum(x["came"] for x in s), rejected=sum(x["rej_delta"] for x in s),
                    stale=sum(x["stale"] for x in s),
                    dec_match=sum(x["gate_dec"] == x["nn_dec"] for x in s if not x["stale"]))
    res = dict(magic=f"{magic:08x}", params=params, delay=delay, plat_pass=plat_pass, fault_dt_us=a.fault_dt,
               summary={k: summ(control if k == "control" else test, k)
                        for k in ("control", "normal", "fault", "after")},
               shots=control + test)
    sm = res["summary"]
    for k, v in sm.items():
        print(f"{k:8s} n {v['n']:3d}: pulse came back {v['came']:3d}, rejected (word 19) {v['rejected']:3d}, "
              f"stale {v['stale']:3d}, gate bit == NN_live bit on fresh {v['dec_match']}")
    ok = (sm["control"]["came"] == sm["control"]["n"] and sm["control"]["rejected"] == 0
          and sm["fault"]["came"] == 0 and sm["fault"]["rejected"] == sm["fault"]["n"]
          and sm["fault"]["stale"] == sm["fault"]["n"]
          and sm["normal"]["came"] == sm["normal"]["n"] and sm["normal"]["rejected"] == 0
          and sm["after"]["came"] == sm["after"]["n"] and sm["after"]["rejected"] == 0)
    res["verdict"] = "PASS" if ok else "FAIL"
    print("binding fault check:", res["verdict"], "(gate left disabled)")
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print("saved", a.out)

if __name__ == "__main__":
    main()
