"""
Parameterised loopback acquisition.

Fires an arbitrary (N, 1540) I/Q dataset through DAC -> analog loopback -> ADC
and saves what comes back, so the on-chip classifier and the offline ones can
be scored on identical hardware-acquired traces.

PHASE CONTRACT: the per-shot rotation is baked into the waveform by the
generator (and is intrinsic to real measured data), so the DAC pulse phase is
held at 0 and Phi is never sent.

Shot ORDER is whatever the input file has. Use tools/make_test_subsets.py to build a
shuffled, class-balanced subset first -- firing a class-ordered file confounds
drift with the label.

"""
import argparse, json, gc, time
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
import Pyro4
from qick import QickConfig
from qick.asm_v2 import AveragerProgramV2

Pyro4.config.SERIALIZER = "pickle"
Pyro4.config.PICKLE_PROTOCOL_VERSION = 4
base = Path(".")

def prepare_waveform(waveform):
    arr = np.asarray(waveform, dtype=float).reshape(-1)
    if arr.size % 32 != 0:
        t = int(np.ceil(arr.size / 32) * 32)
        arr = np.pad(arr, (0, max(0, t - arr.size)), mode="constant")[:t]
    return arr

class LoopbackProgram(AveragerProgramV2):
    def __init__(self, soccfg, cfg):
        super().__init__(soccfg, reps=cfg["reps"], final_delay=cfg["final_delay"], cfg=cfg)

    def _initialize(self, cfg):
        self.declare_gen(ch=cfg["gen_ch"], nqz=cfg["nqz"])
        self.declare_readout(ch=cfg["ro_ch"], length=cfg["readout_length"])
        self.add_envelope(ch=cfg["gen_ch"], name="readout_env",
                          idata=cfg["waveform_I"], qdata=cfg["waveform_Q"])
        self.add_pulse(ch=cfg["gen_ch"], name="readout_pulse", style="arb",
                       envelope="readout_env", gain=cfg["gain"], freq=cfg["freq"],
                       phase=cfg["phase"], outsel=cfg["outsel"], mode=cfg["mode"],
                       stdysel=cfg["stdysel"], ro_ch=cfg["ro_ch"])
        self.add_readoutconfig(ch=cfg["ro_ch"], name="ro_cfg", freq=cfg["freq"],
                               gen_ch=cfg["gen_ch"])

    def _body(self, cfg):
        self.pulse(ch=cfg["gen_ch"], name="readout_pulse")
        self.send_readoutconfig(ch=cfg["ro_ch"], name="ro_cfg", t=0)
        self.trigger(ros=[cfg["ro_ch"]], pins=[0], t=cfg["adc_trig_offset"], width=10)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shots", type=int, default=0, help="0 = all rows in --dataset")
    ap.add_argument("--kfile", default="calibration/k.npy")
    ap.add_argument("--cfile", default="calibration/c.npy")
    a = ap.parse_args()

    outp = base / a.out
    if outp.exists():
        raise RuntimeError("%s already exists -- refusing to overwrite (no-resume guard)" % outp)

    ns = Pyro4.locateNS(host="10.1.9.26", port=8888)
    soc = Pyro4.Proxy(ns.lookup("myqick"))
    soccfg = QickConfig(soc.get_cfg())
    print("Connected.")

    X = np.load(base / a.dataset)
    y = np.load(base / a.labels).astype(int)
    n = a.shots if a.shots else len(X)
    X, y = X[:n], y[:n]
    k = complex(np.load(base / a.kfile)); c = complex(np.load(base / a.cfile))
    print("dataset %s %s   k=%s  c=%s" % (a.dataset, X.shape, k, c))
    print("balance:", dict(zip(*np.unique(y, return_counts=True))))

    z = X[:, 0::2] + 1j * X[:, 1::2]
    z_sent = (z - c) / k
    xo = np.linspace(0, 1, z_sent.shape[1]); xn = np.linspace(0, 1, 11000)
    Ii = interp1d(xo, z_sent.real, axis=1, kind="linear", fill_value="extrapolate")(xn)
    Qi = interp1d(xo, z_sent.imag, axis=1, kind="linear", fill_value="extrapolate")(xn)
    ds = np.empty((n, 22000)); ds[:, 0::2] = Ii; ds[:, 1::2] = Qi

    peak = np.max(np.abs(ds)); sf = 32766 * 0.99 / peak
    ds *= sf
    print("pre-scale peak %.1f -> DAC-fill factor %.6f (post %.1f)" % (peak, sf, np.max(np.abs(ds))))
    assert np.max(np.abs(ds)) < 32766

    cfg0 = {"gen_ch": 0, "ro_ch": 0, "nqz": 1, "reps": 1, "final_delay": 100,
            "readout_length": 1.393, "adc_trig_offset": 0.31, "length": 16,
            "freq": 50, "mode": "oneshot", "stdysel": "zero", "outsel": "product"}

    rows = []
    t0 = time.time()
    for i in range(n):
        w = prepare_waveform(ds[i])
        cfg = dict(cfg0)
        cfg.update({"waveform_I": np.asarray(w[0::2], dtype=float).reshape(-1),
                    "waveform_Q": np.asarray(w[1::2], dtype=float).reshape(-1),
                    "gain": 1, "phase": 0.0, "waveform": w})
        prog = LoopbackProgram(soccfg, cfg)
        iq = prog.acquire_decimated(soc, rounds=1)
        rec = (np.concatenate([np.asarray(x).reshape(-1) for x in iq])
               if isinstance(iq, list) else np.asarray(iq).reshape(-1))
        rows.append(rec)
        if i % 500 == 0 or i == n - 1:
            el = time.time() - t0
            print("  %6d/%d  %.2f shots/s  elapsed %.1f min  eta %.1f min"
                  % (i + 1, n, (i + 1) / max(el, 1e-9), el / 60,
                     (n - i - 1) / max((i + 1) / max(el, 1e-9), 1e-9) / 60), flush=True)
        del prog, iq, cfg, w, rec
        if i % 200 == 0:
            gc.collect()

    A = np.vstack(rows)
    np.save(outp, A)
    np.save(base / a.out.replace(".npy", "_y.npy"), y)
    json.dump(dict(dataset=a.dataset, labels=a.labels, shots=int(n),
                   kfile=a.kfile, cfile=a.cfile, k=[k.real, k.imag], c=[c.real, c.imag],
                   dac_fill_factor=float(sf), phase="0.0 (baked into waveform)",
                   elapsed_min=(time.time() - t0) / 60, out_shape=list(A.shape)),
              open(base / a.out.replace(".npy", "_manifest.json"), "w"), indent=1)
    print("Saved %s %s in %.1f min" % (outp, A.shape, (time.time() - t0) / 60))

if __name__ == "__main__":
    main()
