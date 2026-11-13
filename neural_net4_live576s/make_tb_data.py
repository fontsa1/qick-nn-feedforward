"""Test vectors for the NN_live csim.

Writes tb_data/live_words.bin and tb_data/live_meta.bin:
  * the 20,000 loopback20k_real shots, packed the way the readout emits them (word = Q << 16 | I, both int16), with
    labels and the bit-accurate emulator's output (fpe_v140_loopback20k.npz, p_fixed) for each;
  * the 1,000 classifier-input windows captured on chip (campaign_v139/captures, words as NN_axi received them),
    labels only.
Every window is cut to its first N_SAMP words, the samples live_window_gate (N = 576) passes, and the emulator
outputs come from the 576-sample model (fixed_point_emulator.py --window 576).

Formats (little-endian):
  live_words.bin : int32 N, then N*N_SAMP uint32
  live_meta.bin  : int32 N, then N int32 labels, then N float32 emulator outputs (NaN where none exists)
"""
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
N_SAMP = 576

def pack(X):
    """(n, 1540) interleaved I/Q -> (n, 770) uint32, as hw_raw_transform sees it."""
    q = np.clip(np.round(X), -32768, 32767).astype(np.int16)
    i_ = q[:, 0::2].astype(np.uint16).astype(np.uint32)
    q_ = q[:, 1::2].astype(np.uint16).astype(np.uint32)
    return (q_ << 16) | i_

# boxcar path: the 20,000 test20k_real_X shots captured through the boxcar path (run_loopback_box.py, avg buffer 1
# = NN_live's input with sel = 1) and the boxcar model's bit-accurate emulator outputs on them; or the 5,400-shot
# subset campaign_v140L1/box_csim_set.npz (first 6,000 shots, calibration rows [1000:1600] excluded).
cap = np.load(ROOT / "campaign_v140L1" / "box_csim_set.npz")
X = cap["box"][:20000].astype(np.float64)
y = cap["y"][:20000].astype(np.int32)
fpe = np.load(ROOT / "campaign_v140L1" / "fpe_w576_box_csim_set.npz")
assert (fpe["y"] == y).all()
words = [pack(np.asarray(X, dtype=np.float64))[:, :N_SAMP]]
labels = [y]
emu = [fpe["p_fixed"].astype(np.float32)]

for f in sorted((ROOT / "campaign_v139" / "captures").glob("*.npz")):
    d = np.load(f)
    words.append(d["words"].astype(np.uint32)[:, :N_SAMP])
    labels.append(d["y"].astype(np.int32))
    emu.append(np.full(len(d["y"]), np.nan, dtype=np.float32))

words = np.concatenate(words)
labels = np.concatenate(labels)
emu = np.concatenate(emu)
assert words.shape[1] == N_SAMP
n = np.array([len(labels)], dtype=np.int32)

out = HERE / "tb_data"
out.mkdir(exist_ok=True)
with open(out / "live_words.bin", "wb") as fh:
    fh.write(n.tobytes())
    fh.write(np.ascontiguousarray(words, dtype="<u4").tobytes())
with open(out / "live_meta.bin", "wb") as fh:
    fh.write(n.tobytes())
    fh.write(labels.astype("<i4").tobytes())
    fh.write(emu.astype("<f4").tobytes())
print(f"wrote {n[0]} shots ({int(np.isfinite(emu).sum())} with emulator outputs)")
