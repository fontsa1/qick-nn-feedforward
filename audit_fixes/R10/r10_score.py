"""Bit-accurate scoring (hls4ml C++, the deployed precision config) of a model on the rows of an acquired capture
(default loopback20k_box_A.npz), all 20,000 rows; r10_stats.py takes the 19,400-row set (rows 1000:1600 removed).

Uses fixed_point_model() from analysis_scripts/analysis/fixed_point_emulator.py (same make_hls_config as the export)
and hw_raw_transform from task33_three_way_comparison.py.
"""
import argparse
import os
import sys
import time
from pathlib import Path

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "analysis_scripts" / "analysis"))
sys.path.insert(0, str(ROOT / "analysis_scripts" / "acquisition"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--key", choices=["fir", "box"], required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--eval", default="loopback20k_box_A.npz")
    ap.add_argument("--window", type=int, default=576)
    a = ap.parse_args()
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import task33_three_way_comparison as T
    from fixed_point_emulator import fixed_point_model, stats
    from qkeras.utils import load_qmodel

    model = load_qmodel(str(ROOT / a.model), compile=False)
    ev = np.load(ROOT / a.eval)
    rows, y = ev["rows"], ev["y"].astype(int)
    Xev = T.hw_raw_transform(np.asarray(ev[a.key]).astype(np.float64))[:, :2 * a.window]
    t0 = time.time()
    p_float = model.predict(Xev, verbose=0).reshape(-1)
    outdir = HERE / "hls4ml_emu" / a.name
    hm = fixed_point_model(model, outdir)
    p_fixed = np.asarray(hm.predict(np.ascontiguousarray(Xev, dtype=np.float32))).reshape(-1)
    sf, sx = stats(p_float, y), stats(p_fixed, y)
    m = np.ones(len(rows), bool); m[(rows >= 1000) & (rows < 1600)] = False
    s19 = stats(p_fixed[m], y[m])
    print(f"{a.name}: {a.model} on {a.eval}[{a.key}] ({len(y)} rows, {time.time()-t0:.0f}s)")
    print(f"  float 20k {sf['fid']:.2f}%  fixed 20k {sx['fid']:.2f}%  fixed 19,400 {s19['fid']:.2f}% "
          f"(P(1|0) {s19['p10']:.2f}%, P(0|1) {s19['p01']:.2f}%)")
    np.savez(HERE / f"fpe_{a.name}.npz", p_float=p_float, p_fixed=p_fixed, y=y, rows=rows,
             model=a.model, eval=a.eval, key=a.key)
    print(f"  saved fpe_{a.name}.npz")

if __name__ == "__main__":
    main()
