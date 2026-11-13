"""Extreme-input vectors for tb_live_ext.sv: windows and the expected res_o of NN_live from an
exact integer model in numpy (independent of the C++ and of hls4ml):
  sums = w2 . x + b2;  as = s4raw*sums + b4raw (2^-32);  l4 = floor(as / 2^16);  l5 = relu(l4);
  st = sum_u w6_u l5_u + (b6 + b10) 2^16;  L = floor(st / 2^6)  (logit raw, 2^-10)
  sat = clamp(L, -32768, 32767);  wrap = L mod 2^16;  decision = L >= 16
s4raw / b4raw are the weights quantised to ap_fixed<40,8> (floor at 2^-32), as the firmware loads them.
Writes ext_windows.hex (576 words per window) and ext_expected.hex (sat16 wrap16 decision, hex).
"""
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
FW = HERE.parent / "live_sat3" / "firmware" / "weights"
N = 576
rd = lambda n: np.array([float(t) for t in (FW / n).read_text().replace("{", "").replace("}", "").split(",") if t.strip()])
w2 = rd("w2.txt").astype(np.int64).reshape(4, 2 * N)
b2 = rd("b2.txt").astype(np.int64)
s4 = [int(np.floor(float(v) * 2**32)) for v in (FW / "s4.txt").read_text().split(",")]
b4 = [int(np.floor(float(v) * 2**32)) for v in (FW / "b4.txt").read_text().split(",")]
w6 = rd("w6.txt").astype(np.int64)
b6 = int(rd("b6.txt")[0]); b10 = int(rd("b10.txt")[0])
assert s4 == [226161, 236037, 478404, 226455] and b4 == [2748088064, 2850147840, 3558887168, 2595880448], (s4, b4)
assert list(w6) == [1, 1, -1, 1] and b6 == 0 and b10 == -1

raw = np.fromfile(HERE.parent / "run" / "tb_data" / "live_words.bin", dtype="<u4")
nall = int(raw[0].view(np.int32)); allw = raw[1:].reshape(nall, N)
def unpack(w):
    return (w & 0xFFFF).astype(np.uint16).view(np.int16).astype(np.int64), (w >> 16).astype(np.uint16).view(np.int16).astype(np.int64)
def pack(i, q):
    return ((q.astype(np.int64) & 0xFFFF) << 16 | (i.astype(np.int64) & 0xFFFF)).astype(np.uint32)

wins = []
wins.append(pack(np.zeros(N, int), np.zeros(N, int)))
wins.append(pack(np.full(N, -32768), np.full(N, -32768)))
wins.append(pack(np.full(N, 32767), np.full(N, 32767)))
for u in range(4):
    for sg in (1, -1):
        wi, wq = w2[u, 0::2] * sg, w2[u, 1::2] * sg
        wins.append(pack(np.where(wi > 0, 32767, -32768), np.where(wq > 0, 32767, -32768)))
rng = np.random.default_rng(17)
for _ in range(16):
    wins.append(pack(rng.integers(-32768, 32768, N), rng.integers(-32768, 32768, N)))
for _ in range(16):
    wins.append(pack(np.where(rng.random(N) < .5, 32767, -32768), np.where(rng.random(N) < .5, 32767, -32768)))
for sc in (1, 4, 8, 16, 64):
    for k in range(16):
        i, q = unpack(allw[k])
        wins.append(pack(np.clip(i * sc, -32768, 32767), np.clip(q * sc, -32768, 32767)))
W = np.array(wins)
i, q = unpack(W)
x = np.empty((len(W), 2 * N), np.int64); x[:, 0::2], x[:, 1::2] = i, q
sums = x @ w2.T + b2
st = np.full(len(W), (b6 + b10) << 16, dtype=object)
for u in range(4):
    a = np.array([s4[u] * int(s) + b4[u] for s in sums[:, u]], dtype=object)
    l4 = np.array([v >> 16 for v in a], dtype=object)
    l5 = np.array([v if v > 0 else 0 for v in l4], dtype=object)
    st = st + int(w6[u]) * l5
L = np.array([int(v) >> 6 for v in st])
sat = np.clip(L, -32768, 32767); wrap = ((L + 32768) % 65536) - 32768; dec = (L >= 16).astype(int)
with open(HERE / "ext_windows.hex", "w") as f:
    for w in W.reshape(-1):
        f.write(f"{int(w):08x}\n")
with open(HERE / "ext_expected.hex", "w") as f:
    for k in range(len(W)):
        f.write(f"{int(sat[k]) & 0xFFFF:04x} {int(wrap[k]) & 0xFFFF:04x} {int(dec[k]):x}\n")
print(f"{len(W)} windows; out of range (|logit| >= 32): {int((sat != L).sum())}; wrap != sat: {int((wrap != sat).sum())}; "
      f"wrap decision wrong: {int(((wrap >= 16).astype(int) != dec).sum())}")
