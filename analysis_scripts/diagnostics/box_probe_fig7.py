"""Fig. 7 capture (v1.41): box_probe parts A, B and D with the raw waveforms saved.

Same programs as box_probe.py (const 50 MHz pulse in loopback; readout 0 = QICK's FIR stream in
avg buffer 0, readout 1 = boxcar_tap_0's stream in avg buffer 1, one trigger), but every captured
trace is stored, so Fig. 7 plots measured waveforms and the image level is estimated offline
without the Hann-window bias of box_probe.py (whose estimate reads ~0.6 dB low).
  A. 20 shots: rising edge of a 0.4 us pulse through both paths (raw complex traces).
  B. M = 1..16: long 7 us pulse, raw plateau traces through both paths (3 shots per M).
  D. seg_stamp edge time at NN_live's input, sel 0 (FIR) vs 1 (boxcar), 20 shots each.
The boxcar cfg found at start is restored at exit. Refuses to run on anything but v1.41.
Writes paper_figures/fig7/fig7_capture_<board>_v143.npz (+ .json summary).

"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "board_tools"))
import deploy_and_verify as D                                    # noqa: E402
from qick.asm_v2 import AveragerProgramV2                        # noqa: E402

F = 50.0
BANNER = __import__("os").environ.get("QICK_FIG7_BANNER", "built Wed Nov  4 23:39:37 2026")                        # v1.43, final
OUTDIR = Path(__import__("os").environ.get("QICK_FIG7_DIR", ROOT / "paper_figures" / "fig7"))

soc, soccfg = D.connect()
soccfg['tprocs'][0]['output_pins'].append(('trig', 11, 0, 'NN classify trigger (trig_11_o)'))
CLS_PIN = len(soccfg['tprocs'][0]['output_pins']) - 1

class P(AveragerProgramV2):
    def _initialize(s, cfg):
        s.declare_gen(ch=0, nqz=1)
        for ch in (0, 1):
            s.declare_readout(ch=ch, length=cfg['ro_len'])
            s.add_readoutconfig(ch=ch, name=f"ro{ch}", freq=F, gen_ch=0)
        s.add_pulse(ch=0, name="p", style="const", freq=F, ro_ch=0, length=cfg['plen'], phase=0, gain=cfg['gain'])
    def _body(s, cfg):
        s.send_readoutconfig(ch=0, name="ro0", t=0)
        s.send_readoutconfig(ch=1, name="ro1", t=0)
        s.pulse(ch=0, name="p", t=cfg['tp'])
        s.trigger(ros=[0, 1], pins=[CLS_PIN] if cfg.get('cls') else [], t=cfg['tt'], width=10)

def shot(**kw):
    cfg = dict(ro_len=1.2, plen=0.4, gain=0.5, tp=0.4, tt=0.2, cls=False)
    cfg.update(kw)
    iq = P(soccfg, reps=1, final_delay=5.0, cfg=cfg).acquire_decimated(soc, rounds=1, progress=False)
    return [np.asarray(x).reshape(-1, 2).astype(np.float64) for x in iq]

def cplx(a):
    return a[:, 0] + 1j * a[:, 1]

def plateau(z):
    a = np.abs(z)
    return float(np.median(a[np.argmax(a > 0.5 * a.max()):][:50]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", required=True, choices=["A", "B"])
    a = ap.parse_args()
    banner = str(soccfg).splitlines()[2] if len(str(soccfg).splitlines()) > 2 else str(soccfg)[:200]
    full = str(soccfg)
    if BANNER not in full:
        raise SystemExit(f"firmware is not v1.43 ({BANNER!r} not in the config printout): {banner}")
    if not soc.box_present():
        raise SystemExit("no boxcar_tap_0 in this bitstream / server not patched")
    print("BANNER OK:", BANNER, flush=True)
    cfg0 = soc.box_get()
    print("boxcar cfg at start:", cfg0, flush=True)
    fs = soccfg['readouts'][0]['f_output']
    out = dict(fs_MHz=fs, board=a.board, banner=BANNER, cfg_at_start=cfg0)
    try:
        # ---- A. edge, both paths, at the deployed M = 11 / shift 7 (FIR-matched amplitude)
        soc.box_config(sel=0, shift=7, m=11)
        A0, A1 = [], []
        for _ in range(20):
            r0, r1 = shot()
            assert np.abs(r1).max() < 32000, "boxcar output near full scale"
            A0.append(cplx(r0)); A1.append(cplx(r1))
        A0, A1 = np.array(A0), np.array(A1)
        print(f"A. {len(A0)} shots, {A0.shape[1]} / {A1.shape[1]} samples; plateau FIR {plateau(A0.mean(0)):.1f}, "
              f"box {plateau(A1.mean(0)):.1f}", flush=True)
        # FIR/boxcar amplitude ratio at shift 0 (measured at shift 10 to avoid clipping), as box_probe.py
        soc.box_config(sel=0, shift=10, m=11)
        g = []
        for _ in range(10):
            r0, r1 = shot()
            g.append(plateau(cplx(r1)) * 2 ** 10 / plateau(cplx(r0)))
        gain = float(np.median(g))
        out['gain_box_over_fir_shift0_m11'] = gain

        # ---- B. long plateau, M = 1..16
        B = {}
        for m in range(1, 17):
            sh = int(np.clip(np.round(np.log2(gain * m / 11)), 0, 15))
            soc.box_config(sel=0, shift=sh, m=m)
            f0, f1 = [], []
            for _ in range(3):
                r0, r1 = shot(ro_len=8.0, plen=7.0, tp=0.4, tt=0.2)
                assert np.abs(r1).max() < 32000, f"boxcar near full scale at M={m}"
                f0.append(cplx(r0)); f1.append(cplx(r1))
            B[m] = dict(shift=sh, fir=np.array(f0), box=np.array(f1))
            print(f"B. M={m:2d} shift={sh}: captured", flush=True)

        # ---- D. seg_stamp edge at NN_live's input, sel 0 vs 1
        soc.seg_write(1, 600)
        Dd = {}
        for sel in (0, 1):
            soc.box_config(sel=sel, shift=7, m=11)
            e = []
            for _ in range(20):
                shot(ro_len=1.2, cls=True)
                rec = soc.seg_latest()[3:]
                if rec[5] != 0xFFFFFFFF:
                    e.append(int(rec[5]))
            Dd[sel] = e
            print(f"D. sel={sel}: edge1 cycles median {np.median(e) if e else None} (n={len(e)})", flush=True)
    finally:
        soc.box_config(sel=cfg0['sel'], shift=cfg0['shift'], m=cfg0['m'])
        print("boxcar cfg restored:", soc.box_get(), flush=True)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    stem = OUTDIR / f"fig7_capture_{a.board}_{__import__("os").environ.get("QICK_TAG", "v143")}"
    arrays = dict(A_fir=A0, A_box=A1, D_sel0=np.array(Dd[0]), D_sel1=np.array(Dd[1]))
    for m, v in B.items():
        arrays[f"B{m}_fir"] = v['fir']; arrays[f"B{m}_box"] = v['box']; arrays[f"B{m}_shift"] = np.array(v['shift'])
    np.savez_compressed(str(stem) + ".npz", **arrays)
    out['D_cycles_median'] = {s: (float(np.median(Dd[s])) if Dd[s] else None) for s in (0, 1)}
    Path(str(stem) + ".json").write_text(json.dumps(out, indent=1, default=str))
    print("saved", stem, flush=True)

if __name__ == "__main__":
    main()
