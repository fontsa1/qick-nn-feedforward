"""Repeatability of the k/c calibration on one bitstream load.

Fires tools/run_calibration.py's pass 1 (same rows, same program) --repeats times, with the fixed-pulse phase probe of
drift_probe.py before each repeat, and fits (k, c) for each. Answers: is the scatter of stored arg(k) between
calibrations capture noise, or something that changes between them?

Never writes calibration/k.npy, calibration/c.npy, calib/ or acquired_dataset*.npy; output goes to --out-dir only.
"""
import argparse
import gc
import json
import os
import sys
import time

import numpy as np
import Pyro4

Pyro4.config.SERIALIZER = "pickle"
Pyro4.config.PICKLE_PROTOCOL_VERSION = 4

from qick import QickConfig                     # noqa: E402
from qick.asm_v2 import AveragerProgramV2       # noqa: E402

sys.path.insert(0, "analysis_scripts/diagnostics")
import phrst_probe as P                         # noqa: E402

HOST = os.environ.get("QICK_NS_HOST", "10.1.9.26")

class LoopbackProgram(AveragerProgramV2):       # as in tools/run_calibration.py
    def __init__(self, soccfg, cfg):
        super().__init__(soccfg, reps=cfg["reps"], final_delay=cfg["final_delay"], cfg=cfg)

    def _initialize(self, cfg):
        self.declare_gen(ch=cfg["gen_ch"], nqz=cfg["nqz"])
        self.declare_readout(ch=cfg["ro_ch"], length=cfg["readout_length"])
        self.add_envelope(ch=cfg["gen_ch"], name="readout_env", idata=cfg["waveform_I"], qdata=cfg["waveform_Q"])
        self.add_pulse(ch=cfg["gen_ch"], name="readout_pulse", style="arb", envelope="readout_env", gain=cfg["gain"],
                       freq=cfg["freq"], phase=cfg["phase"], outsel=cfg["outsel"], mode=cfg["mode"],
                       stdysel=cfg["stdysel"], ro_ch=cfg["ro_ch"])
        self.add_readoutconfig(ch=cfg["ro_ch"], name="ro_cfg", freq=cfg["freq"], gen_ch=cfg["gen_ch"])

    def _body(self, cfg):
        self.pulse(ch=cfg["gen_ch"], name="readout_pulse")
        self.send_readoutconfig(ch=cfg["ro_ch"], name="ro_cfg", t=0)
        self.trigger(ros=[cfg["ro_ch"]], pins=[0, cfg["nn_pin"]], t=cfg["adc_trig_offset"], width=10)

BASE = dict(gen_ch=0, ro_ch=0, nqz=1, reps=1, soft_avgs=100, final_delay=100, readout_length=1.393, adc_trig_offset=0.31,
            length=16, freq=50, mode="oneshot", stdysel="zero", outsel="product", gain=1, phase=0.0)

def pad32(a):
    a = np.asarray(a, dtype=float).reshape(-1)
    return a if a.size % 32 == 0 else np.pad(a, (0, int(np.ceil(a.size / 32) * 32) - a.size))

def fit(zc, zref):
    (k, c), *_ = np.linalg.lstsq(np.vstack([zref, np.ones_like(zref)]).T, zc, rcond=None)
    return k, c

def probe(soc, cfg, shots=20):
    base = dict(freq=50.0, gain=0.5, p_len=0.6, ro_len=0.5, trig=0.31, final_delay=10.0, phrst_gen=None, phrst_ro=None)
    ph, mag = [], []
    for _ in range(shots):
        iq = np.asarray(P.PhProg(cfg, dict(base, reps=1)).acquire_decimated(soc, progress=False))
        p, m = P.plateau_phase(iq.reshape(-1, iq.shape[-2], 2)[0])
        ph.append(float(p)); mag.append(float(m))
    m, s = P.circ(np.array(ph))
    return float(m), float(s), float(np.mean(mag))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=8)
    ap.add_argument("--out-dir", default="campaign_v143/final/cal_repeat")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    ns = Pyro4.locateNS(host=HOST, port=8888)
    soc = Pyro4.Proxy(ns.lookup("myqick"))
    cfg = QickConfig(soc.get_cfg())
    cfg["tprocs"][0]["output_pins"].append(("trig", 4, 0, "NN_trigger (trig_4_o)"))
    nn_pin = len(cfg["tprocs"][0]["output_pins"]) - 1
    rows = np.load("X_2_padded_11k.npy")[:500] * 8
    ref = np.load("x_50k.npy", mmap_mode="r")[:500]
    mr = np.asarray(ref[:, 0::2] + 1j * ref[:, 1::2]).mean(0)
    out = []
    path = os.path.join(a.out_dir, f"cal_repeat_{HOST}.json")
    for rep in range(a.repeats):
        t0 = time.time()
        pm, ps, pa = probe(soc, cfg)
        shots = []
        for wf in rows:
            w = pad32(wf)
            c = dict(BASE, waveform_I=w[0::2], waveform_Q=w[1::2], nn_pin=nn_pin)
            iq = LoopbackProgram(cfg, c).acquire_decimated(soc)
            r = np.concatenate([np.asarray(x).reshape(-1) for x in iq]) if isinstance(iq, list) else np.asarray(iq).reshape(-1)
            shots.append(r)
            del iq
        gc.collect()
        s = np.vstack(shots)
        z = (s[:, 0::2] + 1j * s[:, 1::2]).astype(np.complex64)
        k, c_ = fit(z.mean(0), mr)
        rec = dict(rep=rep, t=t0, secs=time.time() - t0, probe_arg=pm, probe_sd=ps, probe_mag=pa, arg_k=float(np.degrees(np.angle(k))),
                   abs_k=float(abs(k)), c_re=float(c_.real), c_im=float(c_.imag))
        out.append(rec)
        np.savez_compressed(os.path.join(a.out_dir, f"cal_repeat_{HOST}_{rep}.npz"), z=z)
        json.dump(out, open(path, "w"), indent=1)
        print(f"[{HOST}] rep {rep}: arg(k) {rec['arg_k']:.3f}  |k| {rec['abs_k']:.5f}  c {c_.real:+.2f}{c_.imag:+.2f}j  "
              f"probe {pm:.3f} sd {ps:.2f}  {rec['secs']:.0f}s", flush=True)

if __name__ == "__main__":
    main()
