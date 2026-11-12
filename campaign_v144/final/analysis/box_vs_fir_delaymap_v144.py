"""As box_vs_fir_v144.py, FIR side only, with the delay-corrected FIR models (audit_fixes/R10/models_fir_delaymap, 5 seeds), 19,400 rows."""
import glob, json, os, re, sys
from pathlib import Path
os.environ.update(KERAS_BACKEND="tensorflow", TF_USE_LEGACY_KERAS="1", TF_CPP_MIN_LOG_LEVEL="3", OMP_NUM_THREADS="4", TF_NUM_INTRAOP_THREADS="4")
import numpy as np
ROOT = Path(__file__).resolve().parents[3]; os.chdir(ROOT)
sys.path[:0] = ["analysis_scripts/analysis", "analysis_scripts/acquisition", "audit_fixes/R3_R2", "."]
from qkeras.utils import load_qmodel
import fixed_point_emulator as F, task33_three_way_comparison as T
from r3_stats import fstats, paired, calmask
fs = sorted(glob.glob("campaign_v144/final/live/A_run1_*.npz"), key=lambda f: int(re.findall(r"_(\d+)_\d+\.npz", f)[0]))[:20]
X, Y, R, D = [], [], [], []
for f in fs:
    z = np.load(f); X.append(z["fir"]); Y.append(z["y"]); R.append(z["rows"]); D.append(z["live_rec"][:, 7] & 1)
X, y, r, d = np.concatenate(X), np.concatenate(Y).astype(int), np.concatenate(R), np.concatenate(D).astype(int)
assert np.array_equal(r, np.arange(20000)); m = calmask(r)
out = {}
for s in range(20260901, 20260906):
    name = f"audit_fixes/R10/models_fir_delaymap/model_fir_delaymap_seed{s}.h5"
    hm = F.fixed_point_model(load_qmodel(name, compile=False), Path(os.environ["TEMP"]) / "hls4ml_fixed_point_emulator" / Path(name).stem)
    p = (np.asarray(hm.predict(np.ascontiguousarray(T.hw_raw_transform(X.astype(np.float64))[:, :1152], dtype=np.float32))).reshape(-1) > .5).astype(int)
    q = paired(d[m], p[m], y[m]); out[s] = dict(fir=fstats(p[m], y[m]), box_vs_fir=q)
    print(s, round(out[s]["fir"]["fid"], 3), q["A_only"], q["B_only"], round(q["p_exact"], 3), round(q["dF"], 3), flush=True)
Path("campaign_v144/final/analysis/box_vs_fir_delaymap.json").write_text(json.dumps(out, indent=1))
