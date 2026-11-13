"""
Two-pass k/c calibration of the replay channel. Run after loading any bitstream, before any loopback acquisition:
it writes calibration/k.npy and calibration/c.npy, which every acquisition script predistorts with.

The pair is stored per board as calib/k_<host>.npy / calib/c_<host>.npy and copied to calibration/k.npy /
calibration/c.npy, with calib/ACTIVE recording the owner. QICK_NS_HOST picks the board. The two boards differ by
about 119 degrees in arg(k), so one board's coefficients on the other score ~31% instead of ~94%; switch boards
without recalibrating with `board_tools/calib.py activate <host>`.

Pass 1 plays the uncalibrated reference envelope (X_2_padded_11k) and fits (k, c); pass 2 predistorts x_50k with
them and records the result (acquired_dataset.npy). Uses trigger pin 0 + acquire_decimated() only (plus the NN
trigger pin).
"""
import gc
import time
from pathlib import Path
import numpy as np
from scipy.interpolate import interp1d
import Pyro4

Pyro4.config.SERIALIZER = "pickle"
Pyro4.config.PICKLE_PROTOCOL_VERSION = 4

from qick import *
from qick.asm_v2 import AveragerProgramV2
from qick import QickConfig

import sys as _sys
_sys.path.insert(0, "board_tools")
import calib                                  # board-scoped (k, c) store

ns_host = __import__("os").environ.get("QICK_NS_HOST", "10.1.9.26")
ns_port = 8888
proxy_name = "myqick"
print(f"calibrating against {ns_host} ({calib.KNOWN.get(ns_host, 'unknown board')})")

# Retry the lookup: right after a deploy the server may not have registered `myqick` yet (NamingError).
def _connect(max_attempts=40, retry_delay_s=3.0):
    last = None
    for attempt in range(1, max_attempts + 1):
        try:
            _ns = Pyro4.locateNS(host=ns_host, port=ns_port)
            _soc = Pyro4.Proxy(_ns.lookup(proxy_name))
            _cfg = QickConfig(_soc.get_cfg())
            if attempt > 1:
                print("connected to board on attempt %d" % attempt)
            return _ns, _soc, _cfg
        except Exception as e:                      # NamingError, comms, etc.
            last = e
            if attempt % 5 == 1:
                print("waiting for board (attempt %d/%d): %s: %s"
                      % (attempt, max_attempts, type(e).__name__, e))
            time.sleep(retry_delay_s)
    raise RuntimeError("board never came up after %d attempts; last error: %r"
                       % (max_attempts, last))

ns, soc, soccfg = _connect()
print(soccfg)

# NN_axi drains its stream input only when its own trigger fires; if it never drains, its FIFO backs up and stalls
# the shared broadcast, DMA branch included. So the NN trigger pin is pulsed with every readout here too. It does
# not affect the acquire_decimated() data path.
soccfg['tprocs'][0]['output_pins'].append(('trig', 4, 0, 'NN_trigger (trig_4_o)'))
NN_TRIGGER_PIN = len(soccfg['tprocs'][0]['output_pins']) - 1
print("NN_TRIGGER_PIN index:", NN_TRIGGER_PIN)

gen_ch = 0
ro_ch = 0

base = Path(".")

# --- clear this board's stale k/c first (only this board's files) ---
_kp, _cp = calib.paths(ns_host)
for _p in (_kp, _cp, base / "calibration/k.npy", base / "calibration/c.npy"):
    if _p.exists():
        _p.unlink()
        print(f"deleted stale {_p.name}")

class LoopbackProgram(AveragerProgramV2):
    def __init__(self, soccfg, cfg):
        super().__init__(
            soccfg,
            reps=cfg["reps"],
            final_delay=cfg["final_delay"],
            cfg=cfg,
        )

    def _initialize(self, cfg):
        self.declare_gen(ch=cfg["gen_ch"], nqz=cfg["nqz"])
        self.declare_readout(ch=cfg["ro_ch"], length=cfg["readout_length"])
        self.add_envelope(
            ch=cfg["gen_ch"],
            name="readout_env",
            idata=cfg["waveform_I"],
            qdata=cfg["waveform_Q"],
        )
        self.add_pulse(
            ch=cfg["gen_ch"],
            name="readout_pulse",
            style="arb",
            envelope="readout_env",
            gain=cfg["gain"],
            freq=cfg["freq"],
            phase=cfg["phase"],
            outsel=cfg["outsel"],
            mode=cfg["mode"],
            stdysel=cfg["stdysel"],
            ro_ch=cfg["ro_ch"],
        )
        self.add_readoutconfig(
            ch=cfg["ro_ch"], name="ro_cfg", freq=cfg["freq"], gen_ch=cfg["gen_ch"]
        )

    def _body(self, cfg):
        self.pulse(ch=cfg["gen_ch"], name="readout_pulse")
        self.send_readoutconfig(ch=cfg["ro_ch"], name="ro_cfg", t=0)
        self.trigger(ros=[cfg["ro_ch"]], pins=[0, NN_TRIGGER_PIN], t=cfg["adc_trig_offset"], width=10)

base_config = {
    "gen_ch": gen_ch,
    "ro_ch": ro_ch,
    "nqz": 1,
    "reps": 1,
    "soft_avgs": 100,
    "final_delay": 100,
    "readout_length": 1.393,
    "adc_trig_offset": 0.31,
    "length": 16,
    "freq": 50,
    "mode": "oneshot",
    "stdysel": "zero",
    "outsel": "product",
}

def prepare_waveform(waveform):
    arr = np.asarray(waveform, dtype=float).reshape(-1)
    if arr.size == 0:
        return np.zeros(16, dtype=float)
    if arr.size % 32 != 0:
        target_len = int(np.ceil(arr.size / 32) * 32)
        if arr.size < target_len:
            arr = np.pad(arr, (0, target_len - arr.size), mode="constant")
        else:
            arr = arr[:target_len]
    return arr

def fire_pass(dataset, phi, shot_idx, out_path, max_rows=500):
    """Fires `max_rows` shots of `dataset` at DAC phase 0 (the per-shot
    rotation is already in the waveform -- see run_loopback_measnoise5000.py)
    for every shot (matches load_waveform.ipynb's own, deliberate quirk --
    see loopback_nn_fidelity_500_smoketest.ipynb cell 11's note on this)."""

    def calculate_waveform_vals(shot):
        I = shot[0::2]
        Q = shot[1::2]
        I = np.asarray(I, dtype=float).reshape(-1)
        Q = np.asarray(Q, dtype=float).reshape(-1)
        return {
            "waveform_I": I,
            "waveform_Q": Q,
            "gain": 1,
            "freq": 50,
            "phase": 0.0,
            "length": 16,
        }

    acquired_rows = []
    for idx in range(max_rows):
        waveform = prepare_waveform(dataset[idx])
        shot_info = calculate_waveform_vals(waveform)
        config = base_config.copy()
        config.update(shot_info)
        config["waveform"] = waveform
        config["shot_idx"] = idx

        prog = LoopbackProgram(soccfg, config)
        iq = prog.acquire_decimated(soc)

        if isinstance(iq, list):
            record = np.concatenate([np.asarray(a).reshape(-1) for a in iq])
        else:
            record = np.asarray(iq).reshape(-1)

        acquired_rows.append(record)

        if idx % 50 == 0 or idx == max_rows - 1:
            print(f"  [{out_path.name}] {idx + 1}/{max_rows} shots fired")

        del prog, iq, config, shot_info, waveform, record
        gc.collect()

    acquired_data = np.vstack(acquired_rows)
    np.save(out_path, acquired_data)
    print(f"saved {acquired_data.shape[0]} rows to {out_path}")
    return acquired_data

# =========================================================================
# PASS 1: fire the raw/uncalibrated reference envelope
# =========================================================================
print("\n=== PASS 1: uncalibrated reference (X_2_padded_11k.npy[:500]*8) ===")
dataset_pass1 = np.load(base / "X_2_padded_11k.npy")[:500] * 8
phi = np.load(base / "Phi.npy")[::500]
shot_idx = 0  # matches load_waveform.ipynb -- fixed for every shot in a pass

acquired_uncalibrated = fire_pass(
    dataset_pass1, phi, shot_idx, base / "acquired_dataset_uncalibrated.npy"
)

# =========================================================================
# Compute k, c via complex least-squares against real x_50k reference
# =========================================================================
print("\n=== Computing k, c ===")
z_acq_raw = acquired_uncalibrated
z_real_raw = np.load(base / "x_50k.npy")[:500]

I_acq = z_acq_raw[:, 0::2]
Q_acq = z_acq_raw[:, 1::2]
z_acq = I_acq + 1j * Q_acq

I_real = z_real_raw[:, 0::2]
Q_real = z_real_raw[:, 1::2]
z_real = I_real + 1j * Q_real

mean_z_orig = np.mean(z_real, axis=0)
mean_z_acq = np.mean(z_acq, axis=0)

A_matrix = np.vstack([mean_z_orig, np.ones_like(mean_z_orig)]).T
sol, *_ = np.linalg.lstsq(A_matrix, mean_z_acq, rcond=None)
k, c = sol
print("k:", k, "c:", c)

z_sent_new = (z_real - c) / k

idata_dataset = np.asarray(z_sent_new.real, dtype=float)
qdata_dataset = np.asarray(z_sent_new.imag, dtype=float)

orig_len = idata_dataset.shape[1]
target_len = 11000
x_old = np.linspace(0, 1, orig_len)
x_new = np.linspace(0, 1, target_len)

f_I = interp1d(x_old, idata_dataset, axis=1, kind="linear", fill_value="extrapolate")
f_Q = interp1d(x_old, qdata_dataset, axis=1, kind="linear", fill_value="extrapolate")
idata_dataset = f_I(x_new)
qdata_dataset = f_Q(x_new)

dataset_pass2 = np.empty((idata_dataset.shape[0], 2 * idata_dataset.shape[1]), dtype=float)
dataset_pass2[:, 0::2] = idata_dataset
dataset_pass2[:, 1::2] = qdata_dataset

ENVELOPE_LIMIT = 32766
TARGET_FRACTION = 0.99
target_ceiling = ENVELOPE_LIMIT * TARGET_FRACTION
current_peak = np.max(np.abs(dataset_pass2))
scale_factor = target_ceiling / current_peak
dataset_pass2 = dataset_pass2 * scale_factor
new_peak = np.max(np.abs(dataset_pass2))
pct_of_limit = 100 * new_peak / ENVELOPE_LIMIT

print(f"pre-scale peak: {current_peak:.2f}")
print(f"scale factor applied: {scale_factor:.6f}")
print(f"post-scale peak: {new_peak:.2f}")
print(f"percent of envelope limit: {pct_of_limit:.2f}%")
assert new_peak < ENVELOPE_LIMIT, "still exceeds limit"

calib.save_kc(k, c, ns_host)     # calib/k_<host>.npy + root calibration/k.npy/c.npy

# =========================================================================
# PASS 2: fire the predistorted x_50k envelope -> acquired_dataset.npy
# =========================================================================
print("\n=== PASS 2: calibrated (predistorted x_50k) ===")
acquired_final = fire_pass(
    dataset_pass2, phi, shot_idx, base / "acquired_dataset.npy"
)

print("\n=== CALIBRATION COMPLETE ===")
print("k:", k, "c:", c)
print("acquired_dataset.npy shape:", acquired_final.shape)
