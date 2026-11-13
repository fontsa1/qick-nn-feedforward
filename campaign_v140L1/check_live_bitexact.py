"""Bit-exactness of NN_live on hardware, from a --live-capture run.

For each shot, score_armed_fidelity.py --live --live-capture saved the window NN_live itself captured (live_words, raw
readout words) and its record (live_rec: shot, acc0..acc3, sigmoid input, sigmoid output, decision; layout in
NN_live.h). This checks, on the chip's own input:
  1. acc0..acc3 == w2 . x (+ b2), exactly (integer arithmetic);
  2. sigmoid output == the hls4ml fixed-point emulator of the model (same NN.cpp / ap_fixed types as synthesis) on
     the same window;
  3. decision bit == (logit >= 1/64), the rule the kernel implements.
A routed timing failure inside NN_live would show up as mismatches here.
"""
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis_scripts" / "analysis"))
sys.path.insert(0, str(ROOT / "analysis_scripts" / "acquisition"))
W = ROOT / "neural_net4_live" / "firmware" / "weights"
MODEL = "models/model_v140_200k_seed20260904.h5"
# 576-sample window: pass --l3, or --weights/--model.
L3_W = ROOT / "neural_net4_live576" / "firmware" / "weights"
L3_MODEL = "models/model_v140L3_w576_seed20260904.h5"
# boxcar readout path: pass --box.
BOX_W = ROOT / "neural_net4_live576box" / "firmware" / "weights"
BOX_MODEL = "models/model_signed_domain_seed20260901_w576_box.h5"

def load_txt(name):
    return np.array([float(t) for t in (W / name).read_text().split(",")])

def sext(v, bits):
    v = v.astype(np.int64) & ((1 << bits) - 1)
    return np.where(v >= 1 << (bits - 1), v - (1 << bits), v)

def main(path):
    d = np.load(path)
    words, rec = d["live_words"].astype(np.uint32), d["live_rec"].astype(np.int64)
    n = len(words)
    i_ = (words & 0xFFFF).astype(np.uint16).view(np.int16).astype(np.int64)
    q_ = (words >> 16).astype(np.uint16).view(np.int16).astype(np.int64)
    x = np.empty((n, 2 * words.shape[1]), dtype=np.int64)
    x[:, 0::2], x[:, 1::2] = i_, q_

    # 1. first-layer sums
    w = load_txt("w2.txt").astype(np.int64).reshape(4, x.shape[1])
    b2 = load_txt("b2.txt").astype(np.int64)
    acc = sext(rec[:, 1:5], 27)
    s = x @ w.T
    ok_nb = (acc == s).all(axis=1)
    ok_b = (acc == s + b2).all(axis=1)
    ok1 = ok_b if ok_b.sum() >= ok_nb.sum() else ok_nb
    print(f"1. acc0..3 == w2.x{' + b2' if ok_b.sum() >= ok_nb.sum() else ''}: "
          f"{ok1.sum()}/{n} shots exact")

    # 2. probability vs the fixed-point emulator
    from qkeras.utils import load_qmodel
    import fixed_point_emulator as F
    import task33_three_way_comparison as T
    model = load_qmodel(str(ROOT / MODEL), compile=False)
    outdir = Path(os.environ.get("TEMP", ".")) / "hls4ml_fixed_point_emulator" / Path(MODEL).stem
    hm = F.fixed_point_model(model, outdir)
    X = T.hw_raw_transform(x.astype(np.float64))
    p_emu = np.asarray(hm.predict(np.ascontiguousarray(X, dtype=np.float32))).reshape(-1)
    p_chip = (rec[:, 6] & 0x3FFFF) / 1024.0
    ok2 = p_chip == p_emu
    print(f"2. sigmoid output == fixed-point emulator: {ok2.sum()}/{n} shots identical"
          f" (max |diff| {np.abs(p_chip - p_emu).max():.6f})")

    # 3. decision rule
    logit = sext(rec[:, 5] & 0xFFFF, 16) / 1024.0
    ok3 = (rec[:, 7] & 1) == (logit >= 1 / 64)
    print(f"3. decision bit == (logit >= 1/64): {ok3.sum()}/{n}")
    y = d["y"].astype(int)
    print(f"   emulator decisions on the live windows vs chip: "
          f"{((p_emu > 0.5) == (rec[:, 7] & 1).astype(bool)).sum()}/{n} agree; "
          f"emulator accuracy {100 * np.mean((p_emu > 0.5) == y):.2f}%")
    if EMU_OUT:
        np.savez(EMU_OUT, p_emu=p_emu, p_chip=p_chip, dec_chip=(rec[:, 7] & 1).astype(np.int8), y=y,
                 ok_acc=ok1, ok_sigmoid=ok2, ok_rule=ok3)
        print("saved", EMU_OUT)
    bad = np.flatnonzero(~(ok1 & ok2 & ok3))
    if len(bad):
        print("first mismatching shots:", bad[:10].tolist())
    print("ALL BIT-EXACT" if not len(bad) else f"{len(bad)} SHOTS MISMATCH")

EMU_OUT = None

if __name__ == "__main__":
    args = sys.argv[1:]
    if "--emu-out" in args:
        # per-shot emulator and chip outputs, for live20k_stats.py
        i = args.index("--emu-out")
        EMU_OUT = args[i + 1]
        del args[i:i + 2]
    if "--l3" in args:
        args.remove("--l3")
        W, MODEL = L3_W, L3_MODEL
    if "--box" in args:
        args.remove("--box")
        W, MODEL = BOX_W, BOX_MODEL
    print(f"weights {W.relative_to(ROOT)}, model {MODEL}")
    main(args[0])
