"""Low-latency readout path (boxcar_tap_0): hardware probe.

Readout 0's FIR stream lands in avg buffer 0 (readout channel 0), boxcar_tap_0's
stream in avg buffer 1 (readout channel 1); both are triggered together, so one
shot gives the same ADC samples through both paths. Measures:
  A. latency saved: rising-edge time of a const pulse in ch0 (FIR) vs ch1 (boxcar),
     same trigger; and the amplitude ratio FIR/boxcar (sets the output shift);
  B. 2f-image suppression vs the moving-sum length M: a long 50 MHz pulse, image
     power at -100 MHz relative to the baseband level, ch1 vs ch0;
  C. NN_live sees exactly buffer 1 with sel = 1 (and buffer 0 with sel = 0):
     NN_live's captured window (live BRAM words 16..16+N) vs acquire_decimated()
     on the same shot, allowing a +/-2-sample offset (the 4.1i trigger skew);
  D. seg_stamp edge1 - trig with sel 0 vs 1: the saving at NN_live's input.
Leaves the cfg as given by --m/--shift with sel = 0 (FIR) at exit.

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

F = 50.0                       # the loopback campaign's carrier (tools/run_loopback.py)
N_LIVE = 576

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

def edge(z):
    """Sub-sample time of the 50% crossing of |z| on the rising edge (samples)."""
    a = np.abs(z)
    hi = np.median(a[np.argmax(a > 0.5 * a.max()):][:50])
    k = int(np.argmax(a > 0.5 * hi))
    return k - 1 + (0.5 * hi - a[k - 1]) / (a[k] - a[k - 1]), hi

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=11)
    ap.add_argument("--shift", type=int, default=None, help="default: chosen to match the FIR amplitude")
    ap.add_argument("--out", default="campaign_v140L1/A_L7_box_probe.json")
    a = ap.parse_args()
    if not soc.box_present():
        raise SystemExit("no boxcar_tap_0 in this bitstream / server not patched")
    fs = soccfg['readouts'][0]['f_output']
    dt = 1e3 / fs
    res = dict(fs_MHz=fs)

    # ---- A. amplitude at shift 10 (the 23-bit moving sum saturates a 16-bit output at
    # low shifts: shift 0 clips the plateau), then the shift
    # that matches the FIR, then the edges at that shift (unclipped, matched levels)
    soc.box_config(sel=0, shift=10, m=a.m)
    g = []
    for _ in range(10):
        r0, r1 = shot()
        g.append(edge(cplx(r1))[1] * 2 ** 10 / edge(cplx(r0))[1])
    gain = float(np.median(g))                              # boxcar(shift 0, unclipped) / FIR
    shift = a.shift if a.shift is not None else int(np.clip(np.round(np.log2(gain)), 0, 15))
    soc.box_config(sel=0, shift=shift, m=a.m)
    e0, e1 = [], []
    for _ in range(20):
        r0, r1 = shot()
        assert np.abs(r1).max() < 32000, "boxcar output near full scale: lower the gain"
        e0.append(edge(cplx(r0))[0]); e1.append(edge(cplx(r1))[0])
    d = np.array(e0) - np.array(e1)
    res['A'] = dict(m=a.m, edge_fir_samples=round(float(np.median(e0)), 3), edge_box_samples=round(float(np.median(e1)), 3),
                    saved_samples=round(float(np.median(d)), 3), saved_sd=round(float(np.std(d)), 3),
                    saved_ns=round(float(np.median(d)) * dt, 2), box_over_fir_gain_shift0=round(gain, 2),
                    shift_chosen=shift, gain_after_shift=round(float(gain / 2 ** shift), 4))
    print("A.", res['A'], flush=True)

    # ---- B. image suppression vs M (long pulse, plateau spectrum)
    res['B'] = {}
    for m in (1, 4, 8, 11, 16):
        sh = int(np.clip(np.round(np.log2(gain * m / a.m)), 0, 15))
        soc.box_config(sel=0, shift=sh, m=m)
        r0, r1 = shot(ro_len=8.0, plen=7.0, tp=0.4, tt=0.2)
        out = {}
        for nm, r in (("fir", r0), ("box", r1)):
            z = cplx(r)[200:-300]
            base = np.abs(z.mean())
            zz = (z - z.mean()) * np.hanning(len(z))
            sp = np.abs(np.fft.fft(zz)) / (np.sum(np.hanning(len(z))))
            fr = np.fft.fftfreq(len(z), dt * 1e-3)
            sel = np.abs(np.abs(fr) - 2 * F) < 3
            img = sp[sel].max() if sel.any() else 0.0
            nse = np.median(sp)
            out[nm] = dict(image_dBc=round(20 * np.log10(max(img, 1e-12) / base), 1),
                           floor_dBc=round(20 * np.log10(max(nse, 1e-12) / base), 1), level=round(float(base), 1))
        res['B'][m] = dict(shift=sh, **out)
        print(f"B. M={m:2d} shift={sh}: box image {out['box']['image_dBc']} dBc (floor {out['box']['floor_dBc']}), "
              f"fir image {out['fir']['image_dBc']} dBc", flush=True)

    # ---- C. NN_live input == buffer, sel 0 and 1
    soc.box_config(sel=0, shift=shift, m=a.m)
    res['C'] = {}
    for sel in (0, 1):
        soc.box_config(sel=sel)
        ex, offs = 0, []
        for i in range(30):
            r0, r1 = shot(ro_len=1.2, cls=True)
            words = np.asarray(soc.live_read_block(16, N_LIVE), dtype=np.uint32)
            li = (words & 0xFFFF).astype(np.uint16).view(np.int16).astype(np.int64)
            lq = (words >> 16).astype(np.uint16).view(np.int16).astype(np.int64)
            buf = (r1 if sel else r0).astype(np.int64)
            hit = None
            for o in (0, 1, -1, 2, -2):
                if 0 <= o and o + N_LIVE <= len(buf):
                    seg = buf[o:o + N_LIVE]
                    if np.array_equal(seg[:, 0], li) and np.array_equal(seg[:, 1], lq):
                        hit = o; break
                elif o < 0:
                    if np.array_equal(buf[:N_LIVE + o, 0], li[-o:]) and np.array_equal(buf[:N_LIVE + o, 1], lq[-o:]):
                        hit = o; break
            if hit is not None:
                ex += 1; offs.append(hit)
        res['C'][sel] = dict(shots=30, matched=ex, offsets={int(k): int(v) for k, v in zip(*np.unique(offs, return_counts=True))} if offs else {})
        print(f"C. sel={sel} ({'boxcar' if sel else 'FIR'}): NN_live window == buffer {1 if sel else 0} on {ex}/30 shots, offsets {res['C'][sel]['offsets']}", flush=True)

    # ---- D. seg_stamp edge1 - trig, sel 0 vs 1
    res['D'] = {}
    soc.seg_write(1, 600)
    for sel in (0, 1):
        soc.box_config(sel=sel)
        e = []
        for _ in range(20):
            shot(ro_len=1.2, cls=True)
            rec = soc.seg_latest()[3:]
            if rec[5] != 0xFFFFFFFF:
                e.append(rec[5])
        res['D'][sel] = dict(edge1_cycles_median=float(np.median(e)) if e else None, n=len(e))
    if res['D'][0]['edge1_cycles_median'] is not None and res['D'][1]['edge1_cycles_median'] is not None:
        res['D']['saved_cycles'] = res['D'][0]['edge1_cycles_median'] - res['D'][1]['edge1_cycles_median']
        res['D']['saved_ns'] = round(res['D']['saved_cycles'] * dt, 2)
    print("D.", res['D'], flush=True)

    soc.box_config(sel=0, shift=shift, m=a.m)
    res['final_cfg'] = soc.box_get()
    Path(ROOT / a.out).write_text(json.dumps(res, indent=1, default=str))
    print("saved", a.out, "| cfg left at", res['final_cfg'])

if __name__ == "__main__":
    main()
