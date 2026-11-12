"""Fig. 7b measured frequency response (v1.41): image level vs image frequency, both readout paths.

A real tone at the readout frequency f, mixed with the readout DDS at f, gives a DC component and a -2f image of
equal amplitude in the mixed stream (both come from the same real ADC signal, so the analog chain's response cancels
in their ratio). The ratio image/DC after each path's filter is therefore that filter's normalised magnitude
response at 2f, as in box_probe_fig7.py part B, which measured it at 2f = 100 MHz only. Sweeping f moves the image
across 0-200 MHz.
  For each f: QICK's FIR path (readout 0 -> avg buffer 0) and the boxcar path (readout 1 -> boxcar_tap -> avg
  buffer 1) captured together on the same shots, with the boxcar at M = 1 (box-8 only) and M = 11 (deployed);
  3 shots of a 7 us plateau each. Raw traces saved; the image level is fitted offline (make_fig7_boxcar.py).
The boxcar cfg found at start is restored at exit. Refuses to run on anything but v1.41.
Writes paper_figures/fig7/fig7_sweep_<board>_v143.npz (+ .json summary).

"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "board_tools"))
import deploy_and_verify as D                                    # noqa: E402
from qick.asm_v2 import AveragerProgramV2                        # noqa: E402

BANNER = __import__("os").environ.get("QICK_FIG7_BANNER", "built Wed Nov  4 23:39:37 2026")                        # v1.43, final
OUTDIR = Path(__import__("os").environ.get("QICK_FIG7_DIR", ROOT / "paper_figures" / "fig7"))
FS = 552.96
# readout frequency f; the image sits at 2f. Uniform 1 MHz grid in f (2 MHz in 2f) plus the M = 11 nulls
# (2f = k*FS/11) and their immediate neighbourhood, where the response changes fastest.
NULLS_2F = [k * FS / 11 for k in (1, 2, 3)]
F_LIST = sorted(set(np.round(np.r_[np.arange(2.5, 100.01, 1.0),
                                   [n / 2 + d for n in NULLS_2F for d in (-0.5, -0.2, 0.0, 0.2, 0.5)]], 3)))
MS = (1, 11)
NSHOT = 3

soc, soccfg = D.connect()

class P(AveragerProgramV2):
    def _initialize(s, cfg):
        s.declare_gen(ch=0, nqz=1)
        for ch in (0, 1):
            s.declare_readout(ch=ch, length=cfg['ro_len'])
            s.add_readoutconfig(ch=ch, name=f"ro{ch}", freq=cfg['f'], gen_ch=0)
        s.add_pulse(ch=0, name="p", style="const", freq=cfg['f'], ro_ch=0, length=cfg['plen'], phase=0,
                    gain=cfg['gain'])

    def _body(s, cfg):
        s.send_readoutconfig(ch=0, name="ro0", t=0)
        s.send_readoutconfig(ch=1, name="ro1", t=0)
        s.pulse(ch=0, name="p", t=cfg['tp'])
        s.trigger(ros=[0, 1], t=cfg['tt'])

def shot(f):
    cfg = dict(f=float(f), ro_len=8.0, plen=7.0, gain=0.5, tp=0.4, tt=0.2)
    iq = P(soccfg, reps=1, final_delay=5.0, cfg=cfg).acquire_decimated(soc, rounds=1, progress=False)
    return [np.asarray(x).reshape(-1, 2) for x in iq]

def quick_db(a, f):
    """Online check only: LS fit of DC + image at -2f (+ the ADC offset tone at -f) on the plateau."""
    z = (a[:, 0] + 1j * a[:, 1]).astype(np.complex128)[200:-300]
    t = np.arange(len(z)) / FS
    X = np.stack([np.ones_like(t), np.exp(-2j * np.pi * 2 * f * t), np.exp(-2j * np.pi * f * t)], 1)
    co = np.linalg.lstsq(X, z, rcond=None)[0]
    return 20 * np.log10(abs(co[1]) / abs(co[0]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", required=True, choices=["A", "B"])
    a = ap.parse_args()
    if BANNER not in str(soccfg):
        raise SystemExit(f"firmware is not v1.43 ({BANNER!r} not in the config printout)")
    if not soc.box_present():
        raise SystemExit("no boxcar_tap_0 in this bitstream / server not patched")
    print(f"[{a.board}] BANNER OK: {BANNER}; {len(F_LIST)} frequencies x M {MS} x {NSHOT} shots", flush=True)
    cfg0 = soc.box_get()
    print(f"[{a.board}] boxcar cfg at start: {cfg0}", flush=True)
    arrays, rows = {}, []
    t0 = time.time()
    try:
        for i, f in enumerate(F_LIST):
            for m in MS:
                sh = 7 if m == 11 else 4                          # ~ FIR-matched amplitude (box_probe_fig7 gain)
                for _ in range(4):                                # raise the shift if anything nears full scale
                    soc.box_config(sel=0, shift=sh, m=m)
                    f0, f1 = [], []
                    for _ in range(NSHOT):
                        r0, r1 = shot(f)
                        f0.append(r0); f1.append(r1)
                    f0, f1 = np.array(f0), np.array(f1)
                    if np.abs(f1).max() < 30000 and np.abs(f0).max() < 30000:
                        break
                    sh += 1
                else:
                    print(f"[{a.board}] WARN f={f} M={m}: still near full scale at shift {sh}", flush=True)
                arrays[f"f{i}_M{m}_fir"] = f0.astype(np.int16)
                arrays[f"f{i}_M{m}_box"] = f1.astype(np.int16)
                rows.append(dict(i=i, f_MHz=float(f), image_MHz=2 * float(f), M=m, shift=sh,
                                 fir_dB=round(float(np.mean([quick_db(x, f) for x in f0])), 2),
                                 box_dB=round(float(np.mean([quick_db(x, f) for x in f1])), 2)))
            r1, r11 = rows[-2], rows[-1]
            print(f"[{a.board}] {i + 1:3d}/{len(F_LIST)} 2f={2 * f:7.2f} MHz  FIR {r11['fir_dB']:7.2f}  "
                  f"M1 {r1['box_dB']:7.2f}  M11 {r11['box_dB']:7.2f} dB  ({time.time() - t0:.0f} s)", flush=True)
    finally:
        soc.box_config(sel=cfg0['sel'], shift=cfg0['shift'], m=cfg0['m'])
        print(f"[{a.board}] boxcar cfg restored: {soc.box_get()}", flush=True)
        OUTDIR.mkdir(parents=True, exist_ok=True)
        stem = OUTDIR / f"fig7_sweep_{a.board}_{__import__("os").environ.get("QICK_TAG", "v143")}"
        arrays["f_MHz"] = np.array(F_LIST[:len(rows) // len(MS)])
        np.savez_compressed(str(stem) + ".npz", **arrays)
        Path(str(stem) + ".json").write_text(json.dumps(dict(board=a.board, banner=BANNER, fs_MHz=FS,
                                                             cfg_at_start=cfg0, complete=len(rows) == len(F_LIST) * len(MS),
                                                             rows=rows), indent=1, default=str))
        print(f"[{a.board}] saved {stem} ({len(rows)} rows)", flush=True)

if __name__ == "__main__":
    main()
