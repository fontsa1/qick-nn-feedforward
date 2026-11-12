"""Boxcar vs FIR readout path on v1.44: each path's own 576-sample network, bit-accurate emulator, on the captures of the same
shots (box_live_capture saves both buffers per shot). Same method as audit_fixes/R10/r10_score.py. Board A.
Writes campaign_v144/final/analysis/box_vs_fir_v144.json."""
import glob, json, os, re, sys
from pathlib import Path
os.environ.update(KERAS_BACKEND="tensorflow", TF_USE_LEGACY_KERAS="1", TF_CPP_MIN_LOG_LEVEL="3", OMP_NUM_THREADS="2", TF_NUM_INTRAOP_THREADS="2")
import numpy as np
ROOT = Path(__file__).resolve().parents[3]; os.chdir(ROOT)
sys.path[:0] = ["analysis_scripts/analysis", "analysis_scripts/acquisition", "audit_fixes/R3_R2", "."]
from qkeras.utils import load_qmodel
import fixed_point_emulator as F, task33_three_way_comparison as T
from r3_stats import fstats, paired, calmask
def emu(name):
    return F.fixed_point_model(load_qmodel(name, compile=False), Path(os.environ["TEMP"]) / "hls4ml_fixed_point_emulator" / Path(name).stem)
mb, mf = emu("models/model_signed_domain_seed20260901_w576_box.h5"), emu("models/model_signed_domain_seed20260901_w576.h5")
R, Y, PB, PF, DC = [], [], [], [], []
for f in sorted(glob.glob("campaign_v144/final/live/A_run*_*.npz"), key=lambda f: int(re.findall(r"_(\d+)_\d+\.npz", f)[0])):
    z = np.load(f)
    pr = lambda m, k: np.asarray(m.predict(np.ascontiguousarray(T.hw_raw_transform(z[k].astype(np.float64))[:, :1152], dtype=np.float32))).reshape(-1)
    PB.append(pr(mb, "box")); PF.append(pr(mf, "fir")); R.append(z["rows"]); Y.append(z["y"]); DC.append(z["live_rec"][:, 7] & 1)
    print(f, flush=True)
R = np.concatenate(R); o = np.argsort(R); R = R[o]; y = np.concatenate(Y)[o].astype(int)
b, c, chip = (np.concatenate(PB)[o] > .5).astype(int), (np.concatenate(PF)[o] > .5).astype(int), np.concatenate(DC)[o].astype(int)
m = calmask(R); s = m & (R < 20000)
out = dict(box_emu_eq_chip=int((b == chip).sum()), n=int(len(R)), subset19400=dict(box=fstats(b[s], y[s]), fir=fstats(c[s], y[s]), paired=paired(b[s], c[s], y[s])),
           all99400=dict(box=fstats(b[m], y[m]), fir=fstats(c[m], y[m]), paired=paired(b[m], c[m], y[m])))
Path("campaign_v144/final/analysis/box_vs_fir_v144.json").write_text(json.dumps(out, indent=1))
for k in ("subset19400", "all99400"):
    print(k, round(out[k]["box"]["fid"], 3), round(out[k]["fir"]["fid"], 3), out[k]["paired"]["A_only"], out[k]["paired"]["B_only"], round(out[k]["paired"]["p_exact"], 3), round(out[k]["paired"]["dF"], 3))
