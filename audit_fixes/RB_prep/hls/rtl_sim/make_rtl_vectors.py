"""Vectors for tb_live.sv (576-sample window, NN_live 1.3/1.4): K real windows
from tb_data/live_words.bin with a v1.40 emulator value, and their expected
outputs.

windows.hex  : K*576 lines, one 32-bit word each (Q<<16 | I)
expected.hex : K lines: sigmoid output raw bits (18), then the four
               first-layer sums (32 each), hex, space-separated
Expected values come from the bit-accurate emulator (tb_data/live_meta.bin) and
an integer dot product with w2, independently of NN_live's C++.
"""
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
LIVE = HERE.parent
K, N = 24, 576
raw = np.fromfile(LIVE / "tb_data" / "live_words.bin", dtype="<u4")
n = int(raw[0].view(np.int32))
allw = raw[1:].reshape(n, N)
meta = np.fromfile(LIVE / "tb_data" / "live_meta.bin", dtype="<i4")
emu_all = meta[1 + n:1 + 2 * n].view(np.float32)
sel = np.flatnonzero(np.isfinite(emu_all))[:K]
words, emu = allw[sel], emu_all[sel]
w2 = np.array([float(t) for t in (LIVE / "firmware/weights/w2.txt").read_text().split(",")]).astype(np.int64).reshape(4, 2 * N)
b2 = np.array([float(t) for t in (LIVE / "firmware/weights/b2.txt").read_text().split(",")]).astype(np.int64)

i_ = (words & 0xFFFF).astype(np.uint16).view(np.int16).astype(np.int64)
q_ = (words >> 16).astype(np.uint16).view(np.int16).astype(np.int64)
x = np.empty((K, 2 * N), dtype=np.int64)
x[:, 0::2], x[:, 1::2] = i_, q_
sums = x @ w2.T + b2
sig = np.round(emu.astype(np.float64) * 1024).astype(np.int64)
assert np.all(sig / 1024 == emu)

with open(HERE / "windows.hex", "w") as f:
    for w in words.reshape(-1):
        f.write(f"{int(w):08x}\n")
with open(HERE / "expected.hex", "w") as f:
    for k in range(K):
        f.write(f"{int(sig[k]) & 0x3FFFF:05x} " + " ".join(f"{int(s) & 0xFFFFFFFF:08x}" for s in sums[k]) + "\n")
print(f"wrote {K} windows (shots {sel[:6].tolist()}...); expected sigmoid raw {sig.tolist()[:6]}...")
