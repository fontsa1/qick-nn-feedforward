"""Paired loopback acquisition through the readout FIR and the boxcar path.

Same shot recipe as tools/run_loopback.py (dataset row -> (z - c)/k predistortion ->
11000-sample DAC envelope at 50 MHz, readout 1.393 us = 770 samples), but every
shot is captured twice from the same ADC samples:
  ch0  avg buffer 0  readout 0's FIR stream
  ch1  avg buffer 1  boxcar_tap_0's stream (boxcar-8 + moving sum)
The boxcar path is faster, so its trigger is ADVANCED by --box-advance-us (the
latency difference measured by box_probe.py) to frame the same signal in its
window as ch0's. The DAC-fill scale is computed over the WHOLE dataset (as
score_armed_fidelity --norm dataset), so split runs are poolable.

Board-scoped k/c via calib.load_kc (QICK_NS_HOST). Checkpoints every 1000 shots
to <out>.partial.npz and resumes from it.

"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "board_tools"))
import calib                                                     # noqa: E402
import deploy_and_verify as D                                    # noqa: E402
from qick.asm_v2 import AveragerProgramV2                        # noqa: E402

def prepare_waveform(arr):
    arr = np.asarray(arr, dtype=float).reshape(-1)
    if arr.size % 32 != 0:
        t = int(np.ceil(arr.size / 32) * 32)
        arr = np.pad(arr, (0, t - arr.size))
    return arr

class BoxLoopback(AveragerProgramV2):
    def _initialize(self, cfg):
        self.declare_gen(ch=0, nqz=1)
        for ch in (0, 1):
            self.declare_readout(ch=ch, length=cfg["readout_length"])
            self.add_readoutconfig(ch=ch, name=f"ro{ch}", freq=cfg["freq"], gen_ch=0)
        self.add_envelope(ch=0, name="env", idata=cfg["waveform_I"], qdata=cfg["waveform_Q"])
        self.add_pulse(ch=0, name="p", style="arb", envelope="env", gain=1, freq=cfg["freq"],
                       phase=0.0, outsel="product", mode="oneshot", stdysel="zero", ro_ch=0)

    def _body(self, cfg):
        self.pulse(ch=0, name="p")
        self.send_readoutconfig(ch=0, name="ro0", t=0)
        self.send_readoutconfig(ch=1, name="ro1", t=0)
        self.trigger(ros=[0], pins=[0], t=cfg["adc_trig_offset"], width=10)
        self.trigger(ros=[1], t=cfg["adc_trig_offset"] - cfg["box_advance"], width=10)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--shots", type=int, default=0, help="0 = to the end of --dataset")
    ap.add_argument("--box-advance-us", type=float, required=True)
    ap.add_argument("--adc-trig-offset", type=float, default=0.31)
    a = ap.parse_args()

    outp = ROOT / a.out
    if outp.exists():
        raise SystemExit(f"{outp} exists -- refusing to overwrite")
    part = outp.with_suffix(".partial.npz")

    soc, soccfg = D.connect()
    host = calib.host_from_env()
    k, c = calib.load_kc(host=host, strict=False)
    print(f"[{host}] k={k} c={c}; boxcar cfg {soc.box_get()}")

    Xall = np.load(ROOT / a.dataset, mmap_mode="r")
    yall = np.load(ROOT / a.labels).astype(int)
    end = len(Xall) if not a.shots else min(len(Xall), a.start + a.shots)
    rows = np.arange(a.start, end)

    # dataset-wide DAC-fill scale (identical for every split of this dataset)
    zfull = np.asarray(Xall[:, 0::2]) + 1j * np.asarray(Xall[:, 1::2])
    zs = (zfull - c) / k
    peak = max(np.abs(zs.real).max(), np.abs(zs.imag).max()) * 1.0001
    sf = 32766 * 0.99 / peak
    del zfull, zs
    xo = np.linspace(0, 1, Xall.shape[1] // 2); xn = np.linspace(0, 1, 11000)

    cfg0 = dict(readout_length=1.393, adc_trig_offset=a.adc_trig_offset, box_advance=a.box_advance_us, freq=50.0)
    n = len(rows)
    fir = np.zeros((n, 1540), np.int16); box = np.zeros((n, 1540), np.int16)
    i0 = 0
    if part.exists():
        p = np.load(part)
        if int(p["start"]) == a.start and len(p["fir"]) == n:
            fir, box, i0 = p["fir"], p["box"], int(p["done"])
            print(f"resuming at {i0}/{n}")
    t0 = time.time()
    for i in range(i0, n):
        r = rows[i]
        z = (np.asarray(Xall[r, 0::2]) + 1j * np.asarray(Xall[r, 1::2]) - c) / k
        I = np.interp(xn, xo, z.real) * sf; Q = np.interp(xn, xo, z.imag) * sf
        w = np.empty(22000); w[0::2] = I; w[1::2] = Q
        w = prepare_waveform(w)
        cfg = dict(cfg0, waveform_I=w[0::2], waveform_Q=w[1::2])
        iq = BoxLoopback(soccfg, reps=1, final_delay=100, cfg=cfg).acquire_decimated(soc, rounds=1, progress=False)
        fir[i] = np.asarray(iq[0]).reshape(-1)[:1540]
        box[i] = np.asarray(iq[1]).reshape(-1)[:1540]
        if (i + 1) % 1000 == 0 or i == n - 1:
            np.savez(part, fir=fir, box=box, done=i + 1, start=a.start)
        if i % 500 == 0 or i == n - 1:
            el = time.time() - t0; rate = (i + 1 - i0) / max(el, 1e-9)
            print(f"  {i + 1:6d}/{n}  {rate:.2f} shots/s  eta {(n - i - 1) / max(rate, 1e-9) / 60:.1f} min", flush=True)
        if i % 200 == 0:
            gc.collect()
    np.savez(outp, fir=fir, box=box, y=yall[rows], rows=rows)
    json.dump(dict(dataset=a.dataset, labels=a.labels, host=host, start=a.start, shots=int(n),
                   k=[k.real, k.imag], c=[c.real, c.imag], dac_fill=float(sf), box_advance_us=a.box_advance_us,
                   adc_trig_offset=a.adc_trig_offset, box_cfg=soc.box_get(),
                   firmware=str(soccfg).split("\n")[2], elapsed_min=(time.time() - t0) / 60),
              open(outp.with_suffix(".json"), "w"), indent=1)
    part.unlink(missing_ok=True)
    print(f"saved {outp}")

if __name__ == "__main__":
    main()
