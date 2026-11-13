"""FAIL_SAFE check: with the gate's ENABLE = 0 a conditional pulse must not play.

Uses gate_probe.py's program (conditional pulse + gate window marker, scheduled unconditionally) and
counts, per case, the shots whose pulse came back (capture plateau > half the always-pass plateau) and
the gate's passed / blanked counters (words 2 / 3, deltas over the case):
  A. word 0 = 0, the value after bitstream load (DELAY 0, so the marker is widened to cover the pulse)
  B. ENABLE = 0 with PASS1 = PASS0 = PASS_STALE = 1 and DELAY calibrated: FAIL_SAFE = 1 must blank every
     window (a v1.40 gate, or FAIL_SAFE = 0, passes every one)
  C. control: ENABLE = 1 with every pass bit set: every pulse must come back
Leaves word 0 = 0.
"""
import argparse
import json
import sys

import numpy as np

sys.path.insert(0, "analysis_scripts/diagnostics")
import gate_probe as G                          # noqa: E402  (brings up soc)

C, h, S, soc = G.C, G.h, G.S, G.soc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=40, help="shots per case")
    ap.add_argument("--n-cal", type=int, default=8)
    ap.add_argument("--nn-advance", type=float, default=0.1135)
    ap.add_argument("--slack", type=float, default=0.0, help="cp_slack (us)")
    ap.add_argument("--out", default="campaign_v140L1/A_v141_fail_safe.json")
    a = ap.parse_args()
    print("boxcar path:", soc.box_config(sel=1, shift=7, m=11))
    k, c = C.calib.load_kc(strict=False)
    X_all = np.load("data/test20k_real_X.npy", mmap_mode="r")
    X = np.asarray(X_all[:a.shots + a.n_cal])
    D = h.prepare_chunk(X, k, c) * (S.raw_peak(X, k, c) / S.raw_peak(X_all, k, c))
    n_win = max(4, int(0.5 * 0.2 * G.FS_RO))
    base = dict(cp_gain0=0.0, cp_gain1=0.6, cp_len=0.2, cp_pre=0.15, trig_width=0.5, fb_margin=0.035,
                fb_thr=16, win_extra=0.05, nn_adv=a.nn_advance, split=False, cp_slack=a.slack)

    # DELAY and the always-pass plateau, as gate_probe step 1
    soc.gate_write(0, G.cfg_word(en=1, pass1=1, pass0=1, pass_stale=1))
    cal = [G.one(D[i], base, n_win) for i in range(a.n_cal)]
    mp = np.array([r["gate"][7] for r in cal])
    delay = int(mp.min()) - G.DELAY_LEAD
    plat_pass = float(np.median([r["plat"] for r in cal]))
    magic = int(cal[-1]["gate"][11])
    params = int(cal[-1]["gate"][10]) >> 8 & 0xF
    print(f"gate magic {magic:08x}, build parameters [EARLY_BLANK SHOT_BIND FAIL_SAFE CFG_SAFE] = "
          f"{[params >> b & 1 for b in range(4)]}; marker -> pulse {mp.tolist()} cycles, DELAY {delay}, "
          f"always-pass plateau {plat_pass:.0f}")

    cases = [("A_reset_word0", 0, dict(base, win_extra=0.12)),
             ("B_disabled_all_pass", G.cfg_word(en=0, pass1=1, pass0=1, pass_stale=1, delay=delay), base),
             ("C_enabled_all_pass", G.cfg_word(en=1, pass1=1, pass0=1, pass_stale=1, delay=delay), base)]
    res = dict(magic=magic, params=params, delay=delay, plat_pass=plat_pass, cases={})
    for name, word0, cf in cases:
        soc.gate_write(0, word0)
        g0 = np.array(soc.gate_stats(), dtype=np.int64)
        plats = [G.one(D[a.n_cal + i], cf, n_win)["plat"] for i in range(a.shots)]
        g1 = np.array(soc.gate_stats(), dtype=np.int64)
        came = int(np.sum(np.array(plats) > 0.5 * plat_pass))
        d = dict(word0=hex(word0), shots=a.shots, pulse_came_back=came,
                 windows=int(g1[1] - g0[1]), passed=int(g1[2] - g0[2]), blanked=int(g1[3] - g0[3]),
                 plat_max_frac=float(np.nanmax(plats) / plat_pass))
        res["cases"][name] = d
        print(f"{name:22s} word0 {word0:#010x}: pulse came back {came}/{a.shots}; gate windows {d['windows']}, "
              f"passed {d['passed']}, blanked {d['blanked']}; max plateau {d['plat_max_frac']:.3f} x always-pass")
    soc.gate_write(0, 0)
    ok = (res["cases"]["A_reset_word0"]["pulse_came_back"] == 0
          and res["cases"]["B_disabled_all_pass"]["pulse_came_back"] == 0
          and res["cases"]["C_enabled_all_pass"]["pulse_came_back"] == a.shots)
    res["verdict"] = "PASS" if ok else "FAIL"
    print("FAIL_SAFE check:", res["verdict"], "(gate left at word 0 = 0)")
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print("saved", a.out)

if __name__ == "__main__":
    main()
