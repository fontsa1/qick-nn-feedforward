"""Logs the decision bit as the tProc receives it (input port 3) and compares it with NN_live's record, shot by
shot.

Uses gate_probe's program unchanged: after the conditional pulse the tProc runs DPORT_RD on port 3 and writes
the 64-bit result word to dmem 0 (bits [31:0], logit) and dmem 1 (bits [63:32]: [28:0] shot field, [31]
decision bit). Per shot: tProc shot field == NN_live record shot, tProc decision bit == NN_live decision bit,
tProc logit == NN_live logit. Gate enabled with the normal policy, so the gate's own bit is compared too.
"""
import argparse
import json
import sys

import numpy as np

sys.path.insert(0, "analysis_scripts/diagnostics")
import gate_probe as G                          # noqa: E402  (brings up soc)

C, h, S, soc, soccfg, L = G.C, G.h, G.S, G.soc, G.soccfg, G.L
CFG0 = G.CFG0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=400)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--nn-advance", type=float, default=0.1135)
    ap.add_argument("--slack", type=float, default=0.0)
    ap.add_argument("--out", default="campaign_v140L1/A_v142_tproc_decision.json")
    a = ap.parse_args()
    print("boxcar path:", soc.box_config(sel=1, shift=7, m=11))
    k, c = C.calib.load_kc(strict=False)
    X_all = np.load("data/test20k_real_X.npy", mmap_mode="r")
    X = np.asarray(X_all[a.start:a.start + a.shots])
    D = h.prepare_chunk(X, k, c) * (S.raw_peak(X, k, c) / S.raw_peak(X_all, k, c))
    cf = dict(cp_gain0=0.0, cp_gain1=0.6, cp_len=0.2, cp_pre=0.15, trig_width=0.5, fb_margin=0.035,
              fb_thr=16, win_extra=0.05, nn_adv=a.nn_advance, split=False, cp_slack=a.slack)
    soc.gate_write(0, G.cfg_word(en=1, pass1=1, pass0=0, pass_stale=0, delay=0))

    rows = []
    for i in range(a.shots):
        wf = C.trim_wf(h.prepare_waveform(D[i]))
        cfg = dict(CFG0, waveform_I=wf[0::2], waveform_Q=wf[1::2], **cf)
        G.GateProg(soccfg, cfg).acquire_decimated(soc, progress=False)
        m = np.asarray(soc.tproc.read_mem("dmem", length=2)).reshape(-1).astype(np.int64)
        rec = C.live_record(C.shot_window(soc.live_read_block(0, 3)))
        g = np.array(soc.gate_stats(), dtype=np.int64)
        hi = int(np.uint32(m[1]))
        rows.append(dict(row=a.start + i, tp_shot=hi & C.SHOT_MASK, tp_bit=hi >> 31 & 1,
                         tp_logit=int(np.int32(np.uint32(m[0]))),               # Q.10, sign-extended
                         nn_shot=int(rec[0]), nn_bit=int(rec[7] & 1),
                         nn_logit=int(np.int16(np.uint16(rec[5] & 0xFFFF))),    # record word 5: sigmoid input
                         gate_bit=int(g[10] >> 2 & 1), gate_stale=bool(g[10] & 2)))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{a.shots}", flush=True)
    soc.gate_write(0, G.cfg_word(en=0))
    n = len(rows)
    shot_ok = sum(r["tp_shot"] == r["nn_shot"] for r in rows)
    bit_ok = sum(r["tp_bit"] == r["nn_bit"] for r in rows)
    logit_ok = sum(r["tp_logit"] == r["nn_logit"] for r in rows)
    gate_ok = sum(r["gate_bit"] == r["nn_bit"] for r in rows if not r["gate_stale"])
    n_fresh = sum(not r["gate_stale"] for r in rows)
    ones = sum(r["nn_bit"] for r in rows)
    print(f"tProc shot field == NN_live shot: {shot_ok}/{n}")
    print(f"tProc decision bit == NN_live decision bit: {bit_ok}/{n}  (NN_live decided 1 on {ones}/{n})")
    print(f"tProc logit == NN_live record logit: {logit_ok}/{n}")
    print(f"gate decision bit == NN_live decision bit (fresh shots): {gate_ok}/{n_fresh}")
    res = dict(shots=n, shot_match=shot_ok, bit_match=bit_ok, logit_match=logit_ok, gate_bit_match=gate_ok,
               gate_fresh=n_fresh, nn_ones=ones,
               verdict="PASS" if shot_ok == n and bit_ok == n and logit_ok == n else "FAIL", rows=rows)
    print("tProc decision check:", res["verdict"], "(gate left disabled)")
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print("saved", a.out)

if __name__ == "__main__":
    main()
