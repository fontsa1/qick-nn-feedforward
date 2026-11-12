"""NN_axi on v1.44: the chip's probability against the bit-accurate emulator on the window the buffer recorded, every shot,
and the emulator's fidelity. Also writes the replayed 20,000 traces per board (git-ignored) for the linear baselines.

Writes campaign_v144/final/analysis/axi_emu_v144.json and replay20k_{A,B}.npy.
"""
import glob
import json
import os
import re
import sys
from pathlib import Path

os.environ.update(KERAS_BACKEND="tensorflow", TF_USE_LEGACY_KERAS="1", TF_CPP_MIN_LOG_LEVEL="3")
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
os.chdir(ROOT)
sys.path[:0] = ["analysis_scripts/analysis", "analysis_scripts/acquisition", "audit_fixes/R3_R2", "."]
from qkeras.utils import load_qmodel  # noqa: E402
import fixed_point_emulator as F  # noqa: E402
import task33_three_way_comparison as T  # noqa: E402
from r3_stats import fstats, paired, calmask  # noqa: E402

D = "campaign_v144/final/"
M = "models/model_v140_200k_seed20260904.h5"
hm = F.fixed_point_model(load_qmodel(M, compile=False), Path(os.environ["TEMP"]) / "hls4ml_fixed_point_emulator" / Path(M).stem)
res = {}
for b in "AB":
    fs = sorted(glob.glob(f"{D}axi/{b}_axi_*.npz"), key=lambda f: int(re.findall(r"_axi_(\d+)\.npz", f)[0]))
    X, P, Pe, I, Y = [], [], [], [], []
    for f in fs:
        z = np.load(f, allow_pickle=True)
        x = z["dec_iq"].astype(np.float64).reshape(len(z["idx"]), -1)
        Pe.append(np.asarray(hm.predict(np.ascontiguousarray(T.hw_raw_transform(x), dtype=np.float32))).reshape(-1))
        X.append(x.astype(np.int32)); P.append(z["probs"]); I.append(z["idx"]); Y.append(z["y"])
    I = np.concatenate(I); o = np.argsort(I); assert np.array_equal(I[o], np.arange(20000))
    X, P, Pe, Y = np.concatenate(X)[o], np.concatenate(P)[o], np.concatenate(Pe)[o], np.concatenate(Y)[o].astype(int)
    np.save(f"{D}analysis/replay20k_{b}.npy", X)
    m = calmask(np.arange(20000))
    dc, de = (P > .5).astype(int), (Pe > .5).astype(int)
    res[b] = dict(n=20000, identical=int((P == Pe).sum()), decisions_agree=int((dc == de).sum()), max_abs_diff=float(abs(P - Pe).max()),
                  emu_subset19400=fstats(de[m], Y[m]), chip_subset19400=fstats(dc[m], Y[m]), chip_vs_emu=paired(dc[m], de[m], Y[m]))
    print(b, {k: res[b][k] for k in ("n", "identical", "decisions_agree", "max_abs_diff")}, "emu", round(res[b]["emu_subset19400"]["fid"], 3), flush=True)
Path(f"{D}analysis/axi_emu_v144.json").write_text(json.dumps(res, indent=1))
