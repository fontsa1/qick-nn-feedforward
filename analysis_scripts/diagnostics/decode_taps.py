"""Decode v1.28's DATAFLOW-tap capture slots into layer values.

The tee processes inside NN() write each intermediate layer as RAW
fixed-point BITS (see NN.h's v1.28 note for why raw bits rather than floats:
a (float) cast of ap_fixed<48,32> synthesises a real converter, and 14 of
them would be hundreds of LUTs for nothing). Those bits ride through the
float BRAM word untouched, exactly as v1.27's input capture does, so the host
must reinterpret rather than read a number.

BRAM record for one armed shot (789 words):
    [0   .. 769]  packed I/Q stream words   -- the exact NN input
    [770]         the sigmoid the chip produced
    [771 .. 788]  the 18 tap words decoded here

Slot map inside the tap region, matching NN.h's N_DBG_* constants:
    +0  .. +7   layer2  4 x ap_fixed<48,32>, low word then high 16 bits
    +8  .. +11  layer4  4 x ap_fixed<32,16>
    +12 .. +15  layer5  4 x ap_fixed<32,16>
    +16         layer6  1 x ap_fixed<32,16>
    +17         layer10 1 x ap_fixed<16,6>
"""
import struct
import numpy as np

N_WORDS = 770
SIG_SLOT = N_WORDS
DBG_BASE = N_WORDS + 1
N_DBG_L2, N_DBG_L4, N_DBG_L5, N_DBG_L6, N_DBG_L10 = 8, 4, 4, 1, 1
N_DBG_TOTAL = N_DBG_L2 + N_DBG_L4 + N_DBG_L5 + N_DBG_L6 + N_DBG_L10  # 18

# v1.29 in_local readback: 128 I/Q pairs at stride 6 PAIRS, i.e. word indices
# 12k and 12k+1 for k = 0..127 (covering in_local[0..1525]). Must match
# N_INLOC_PAIRS / INLOC_STRIDE / CAP_INLOC_BASE in NN_axi.h.
INLOC_BASE = DBG_BASE + N_DBG_TOTAL                                  # 789
INLOC_PRE_BASE = INLOC_BASE                                          # 789

# ---- v1.34 layout (PREPPED, not yet built) --------------------------------
# PRE is now CONTIGUOUS: word k is in_local[k], k = 0..N_PRE_WORDS-1, covering
# capture pairs 0..N_PRE_PAIRS-1 (65% of the window, every word). The v1.29
# POST gather is gone -- it agreed with PRE 256/256 on every build from v1.30
# on, so its 256 BRAM words went to PRE's coverage instead.
#   [0    .. 769 ] capture      [770] sigmoid      [771 .. 788] taps
#   [789  .. 1790] in_local PRE (1002 words)
#   [1791 .. 2046] dense-input tap (256 words)
# v1.35: PRE re-pointed at the TAIL (in_local[1002..1539]) -- the only region
# never observed -- plus a HEAD control on the same shots.
N_PRE_TAIL_WORDS = 538
PRE_TAIL_START = 1002
N_PRE_HEAD_WORDS = 128
PRE_TAIL_BASE = INLOC_PRE_BASE                                       # 789
PRE_HEAD_BASE = PRE_TAIL_BASE + N_PRE_TAIL_WORDS                     # 1327
INLOC_FRAC = 12          # input_t = ap_fixed<32,20>

# The dense-input tap still samples at DBG_IN_STRIDE across the FULL window,
# so it keeps coverage of the tail PRE cannot reach.
N_INLOC_PAIRS = 96
N_INLOC = 2 * N_INLOC_PAIRS                                          # 192
INLOC_STRIDE = 8                                                     # in PAIRS
DENSE_IN_BASE = PRE_HEAD_BASE + N_PRE_HEAD_WORDS                     # 1455
N_DENSE_IN = N_INLOC                                                 # 256
RAW_N = 32
N_RAW_RB = 4 * RAW_N                                                 # 128
RAW_LAGS = (1, 32, 256, 700)
RAW_ISTART = (1, 40, 300, 730)
RAW_BASE = DENSE_IN_BASE + N_DENSE_IN                                # 1647
# ---- v1.36 PRELOAD probe: in_local read BEFORE LOAD_L, same indices ----
N_PRELOAD_WORDS = 64
PRELOAD_START = 1540 - N_PRELOAD_WORDS                               # 1476
PRELOAD_BASE = RAW_BASE + N_RAW_RB                                   # 1775
RECORD_WORDS = PRELOAD_BASE + N_PRELOAD_WORDS                        # 1839
assert RECORD_WORDS <= 2048, RECORD_WORDS


def pre_tail_indices():
    """in_local[] index of each TAIL readback slot."""
    return PRE_TAIL_START + np.arange(N_PRE_TAIL_WORDS, dtype=int)


def pre_head_indices():
    return np.arange(N_PRE_HEAD_WORDS, dtype=int)


def decode_pre(words):
    """Decode raw in_local readback words to input_t values."""
    w = np.asarray(words, dtype=np.uint64).reshape(-1)
    return np.array([_signed(int(v) & 0xFFFFFFFF, 32) / 2.0 ** INLOC_FRAC
                     for v in w])


def inloc_word_indices():
    """The in_local[] word indices the readback samples, in slot order."""
    out = []
    for k in range(N_INLOC_PAIRS):
        m = INLOC_STRIDE * k            # pair index
        out += [2 * m, 2 * m + 1]
    return np.array(out, dtype=int)


def inloc_pair_indices():
    """The CAPTURE word (pair) index each sampled in_local pair came from."""
    return np.array([INLOC_STRIDE * k for k in range(N_INLOC_PAIRS)], dtype=int)


def decode_inloc(words):
    """Decode the 256 in_local readback words to input_t values."""
    w = np.asarray(words, dtype=np.uint64).reshape(-1)
    if len(w) != N_INLOC:
        raise ValueError(f"expected {N_INLOC} in_local words, got {len(w)}")
    return np.array([_signed(int(v) & 0xFFFFFFFF, 32) / 2.0 ** INLOC_FRAC
                     for v in w])


def float_to_word(f):
    """Recover the 32 bits the core wrote from the float the host read."""
    return struct.unpack("I", struct.pack("f", np.float32(f)))[0]


def _signed(raw, bits):
    return raw - (1 << bits) if raw >= (1 << (bits - 1)) else raw


def decode(words):
    """words: the 18 raw uint32 tap words. Returns a dict of float arrays.

    Fractional bits come straight from the HLS typedefs in defines.h:
    layer2_t = ap_fixed<48,32> (16 frac), layer4/5/6_t = ap_fixed<32,16>
    (16 frac), q_output_alpha_result_t = ap_fixed<16,6> (10 frac).
    """
    w = np.asarray(words, dtype=np.uint64).reshape(-1)
    if len(w) != N_DBG_TOTAL:
        raise ValueError(f"expected {N_DBG_TOTAL} tap words, got {len(w)}")

    l2 = np.empty(4)
    for i in range(4):
        raw = int(w[2 * i]) | ((int(w[2 * i + 1]) & 0xFFFF) << 32)
        l2[i] = _signed(raw, 48) / 2.0 ** 16

    o = N_DBG_L2
    l4 = np.array([_signed(int(w[o + i]) & 0xFFFFFFFF, 32) / 2.0 ** 16
                   for i in range(4)])
    o += N_DBG_L4
    l5 = np.array([_signed(int(w[o + i]) & 0xFFFFFFFF, 32) / 2.0 ** 16
                   for i in range(4)])
    o += N_DBG_L5
    l6 = np.array([_signed(int(w[o]) & 0xFFFFFFFF, 32) / 2.0 ** 16])
    o += N_DBG_L6
    l10 = np.array([_signed(int(w[o]) & 0xFFFF, 16) / 2.0 ** 10])
    return {"l2": l2, "l4": l4, "l5": l5, "l6": l6, "l10": l10}


def read_taps(soc):
    """Read slots 771..788 off a live board and decode them."""
    w = [soc.nn_read_bram((DBG_BASE + i) * 4) & 0xFFFFFFFF
         for i in range(N_DBG_TOTAL)]
    return decode(w), np.array(w, dtype=np.uint32)


def decode_from_floats(floats):
    """Decode when the caller already read the slots as floats."""
    return decode([float_to_word(f) for f in floats])
