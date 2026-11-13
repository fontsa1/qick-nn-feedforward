"""Statistics of the v1.44 campaign, with the methods of
(campaign_v140L1/check_live_bitexact.py, live20k_stats.py, audit_fixes/R18_full/analyze_full.py, r3_stats).

Reads only campaign_v144/final/{live,axi,inloop}. Run 1 = each board's own half (A rows 0-49,999, B rows 50,000-99,999),
run 2 = the halves swapped, so every trace is scored on both boards. Headline set = run 1, minus rows 1000-1599.

Writes campaign_v144/final/analysis/v144_stats.json and, git-ignored, merged per-shot files in the same folder.
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
D = ROOT / "campaign_v144" / "final"
OUT = D / "analysis"
sys.path[:0] = [str(ROOT / "audit_fixes" / "R3_R2"), str(ROOT / "analysis_scripts" / "analysis"), str(ROOT / "analysis_scripts" / "acquisition"), str(ROOT)]
from r3_stats import fstats, paired, calmask  # noqa: E402

W = ROOT / "neural_net4_live576box" / "firmware" / "weights"
LIVE_MODEL = "models/model_signed_domain_seed20260901_w576_box.h5"
EMU = "--no-emu" not in sys.argv

def sext(v, bits):
    v = v.astype(np.int64) & ((1 << bits) - 1)
    return np.where(v >= 1 << (bits - 1), v - (1 << bits), v)

class Emu:
    def __init__(self, name):
        from qkeras.utils import load_qmodel
        import fixed_point_emulator as F
        import task33_three_way_comparison as T
        self.T = T
        m = load_qmodel(str(ROOT / name), compile=False)
        self.hm = F.fixed_point_model(m, Path(os.environ.get("TEMP", ".")) / "hls4ml_fixed_point_emulator" / Path(name).stem)

    def __call__(self, x):
        X = self.T.hw_raw_transform(x.astype(np.float64))
        return np.asarray(self.hm.predict(np.ascontiguousarray(X, dtype=np.float32))).reshape(-1)

w2 = np.array([float(t) for t in (W / "w2.txt").read_text().split(",")]).astype(np.int64).reshape(4, 2 * 576)
b2 = np.array([float(t) for t in (W / "b2.txt").read_text().split(",")]).astype(np.int64)

def bitexact(words, rec, emu, tot):
    """Adds this chunk's counts to tot: sums, sigmoid, decision rule, all three on the chip's own input."""
    words = words.astype(np.uint32); rec = rec.astype(np.int64)
    x = np.empty((len(words), 1152), dtype=np.int64)
    x[:, 0::2] = (words & 0xFFFF).astype(np.uint16).view(np.int16).astype(np.int64)
    x[:, 1::2] = (words >> 16).astype(np.uint16).view(np.int16).astype(np.int64)
    acc = sext(rec[:, 1:5], 27)
    s = x @ w2.T
    ok_a = (acc == s + b2).all(1) if ((acc == s + b2).all(1)).sum() >= ((acc == s).all(1)).sum() else (acc == s).all(1)
    logit = sext(rec[:, 5] & 0xFFFF, 16) / 1024.0
    ok_r = (rec[:, 7] & 1) == (logit >= 1 / 64)
    tot["n"] += len(words); tot["sums"] += int(ok_a.sum()); tot["rule"] += int(ok_r.sum())
    if emu is not None:
        ok_s = (rec[:, 6] & 0x3FFFF) / 1024.0 == emu(x)
        tot["sigmoid"] += int(ok_s.sum()); tot["all"] += int((ok_a & ok_r & ok_s).sum())

def load_live(board, run, emu):
    fs = sorted(glob.glob(str(D / "live" / f"{board}_run{run}_*.npz")), key=lambda f: int(re.findall(r"_(\d+)_\d+\.npz", f)[0]))
    assert len(fs) == 50, (board, run, len(fs))
    R, Y, C, CL, EQ = [], [], [], [], []
    tot = dict(n=0, sums=0, rule=0, sigmoid=0, all=0)
    for f in fs:
        z = np.load(f)
        assert int(z["live_n"]) == 576
        R.append(z["rows"]); Y.append(z["y"]); C.append(z["live_rec"]); CL.append(z["dac_clipped"]); EQ.append(z["window_eq_buffer"])
        bitexact(z["live_words"], z["live_rec"], emu, tot)
    d = dict(rows=np.concatenate(R), y=np.concatenate(Y).astype(int), rec=np.concatenate(C), clip=np.concatenate(CL), eq=np.concatenate(EQ))
    o = np.argsort(d["rows"]); d = {k: v[o] for k, v in d.items()}
    assert len(np.unique(d["rows"])) == 50000
    d["dec"] = (d["rec"][:, 7] & 1).astype(int)
    d["bitexact"] = tot
    print(board, "run", run, "rows", d["rows"][0], d["rows"][-1], "bit-exact", tot, "window==buffer", int(d["eq"].sum()), flush=True)
    return d

def join(a, b):
    out = {k: np.concatenate([a[k], b[k]]) for k in ("rows", "y", "rec", "clip", "eq", "dec")}
    o = np.argsort(out["rows"])
    return {k: v[o] for k, v in out.items()}

def F(d, m=None):
    return fstats(d["dec"], d["y"]) if m is None else fstats(d["dec"][m], d["y"][m])

emu = Emu(LIVE_MODEL) if EMU else None
L = {(b, r): load_live(b, r, emu) for b in "AB" for r in (1, 2)}
res = {"bit_exact": {f"{b}_run{r}": L[(b, r)]["bitexact"] for b, r in L}}
A = join(L[("A", 1)], L[("A", 2)])           # every trace on board A
B = join(L[("B", 2)], L[("B", 1)])           # every trace on board B
H = join(L[("A", 1)], L[("B", 1)])           # headline: run 1 assignment
assert np.array_equal(A["rows"], B["rows"]) and np.array_equal(A["rows"], np.arange(100000)) and np.array_equal(H["rows"], A["rows"])
assert np.array_equal(A["y"], B["y"]) and np.array_equal(A["y"], H["y"])
cal = calmask(H["rows"]); fresh = H["rows"] >= 20000; sub = H["rows"] < 20000
res["live"] = dict(
    headline=F(H, cal), full100k=F(H), fresh80k=F(H, fresh), headline_subset19400_boardA=F(A, cal & sub), headline_subset19400_boardB=F(B, cal & sub),
    run1_A=F(H, cal & (H["rows"] < 50000)), run1_B=F(H, cal & (H["rows"] >= 50000)),
    A_all=F(A, cal), B_all=F(B, cal), A_vs_B=paired(A["dec"][cal], B["dec"][cal], A["y"][cal]),
    A_vs_B_full100k=paired(A["dec"], B["dec"], A["y"]), agreement_pct=100 * float((A["dec"][cal] == B["dec"][cal]).mean()),
    window_eq_buffer={"A": int(A["eq"].sum()), "B": int(B["eq"].sum())},
    dac_clipped_shots={"A": int((A["clip"] > 0).sum()), "B": int((B["clip"] > 0).sum())},
    run1_vs_run2_note="run 1 and run 2 are separate calibrations of one load",
)
ok = cal & (H["clip"] == 0)
res["live"]["headline_without_clipped"] = F(H, ok)

# ---------------------------------------------------------------- in the loop
KEYS = ("rows", "y", "gate", "plat", "dec", "late", "L_buf", "n_clip", "fresh")

def median_ci(x):
    x = np.sort(np.asarray(x, float)); n = len(x)
    lo, hi = int(np.floor(n / 2 - 1.96 * np.sqrt(n) / 2)), int(np.ceil(n / 2 + 1.96 * np.sqrt(n) / 2))
    return [float(np.median(x)), float(x[max(lo, 0)]), float(x[min(hi, n - 1)])]

keep = {}
for b, idle in (("A", A), ("B", B)):
    z = np.load(D / "inloop" / f"{b}_inloop_0_100000.npz", allow_pickle=True)
    n = int(z["n_done"])
    assert n == 99400 and str(z["banner"]).endswith("Sat Nov  7 19:57:07 2026"), (n, z["banner"])
    d = {k: z[k][:n] for k in KEYS}
    rows, y = d["rows"], d["y"].astype(int)
    o = np.argsort(rows); assert np.array_equal(o, np.arange(n))
    pp = float(z["plat_pass"]); came = d["plat"] > 0.5 * pp; frac = d["plat"] / pp
    flags = d["gate"][:, 10]
    passed = (flags & 1) > 0; stale = (flags & 2) > 0; gbit = ((flags >> 2) & 1).astype(int)
    dec = d["dec"].astype(int)
    correct = (came == passed) & ~stale & (gbit == dec)
    unsafe = came & ~(~stale & (gbit == dec) & (dec == 1))
    pred = came.astype(int)
    id_ = idle["dec"][rows]
    assert np.array_equal(idle["y"][rows], y)
    nlate = int(d["late"].sum()); pc = passed & came; bl = ~passed & ~stale; f = rows >= 20000; s = rows < 20000
    res.setdefault("inloop", {})[b] = dict(
        n=n, slack_us=float(z["slack"]), delay=int(z["delay"]), plat_pass=pp, k=[float(np.real(z["k"])), float(np.imag(z["k"]))], banner=str(z["banner"]),
        inloop=fstats(pred, y), inloop_fresh80k=fstats(pred[f], y[f]), inloop_subset19400=fstats(pred[s], y[s]),
        decision=fstats(dec, y), gate_idle=fstats(id_, y), inloop_vs_gate_idle=paired(pred, id_, y),
        inloop_vs_gate_idle_subset19400=paired(pred[s], id_[s], y[s]),
        loop=dict(correct=int(correct.sum()), not_correct=int((~correct).sum()), stale=int(stale.sum()), late=nlate, unsafe=int(unsafe.sum()),
                  pulse_ne_decision=int((pred != dec).sum()), binding_rejects=int(d["gate"][-1, 19] - d["gate"][0, 19]),
                  late_rate_upper95=float(1 - 0.05 ** (1 / n)) if nlate == 0 else None, dac_clipped_shots=int((d["n_clip"] > 0).sum()),
                  correct_subset19400=int(correct[s].sum()), fresh=int(d["fresh"].sum())),
        L_buf_passed=dict(n=int(pc.sum()), median_ci=median_ci(d["L_buf"][pc]), range=[float(d["L_buf"][pc].min()), float(d["L_buf"][pc].max())]),
        blanked_plateau_frac=dict(n=int(bl.sum()), median=float(np.median(frac[bl])), p99=float(np.percentile(frac[bl], 99)), max=float(frac[bl].max())),
    )
    keep[b] = (rows, y, pred, dec)
    r = res["inloop"][b]
    print(f"in-loop {b}: F {r['inloop']['fid']:.3f}  subset {r['inloop_subset19400']['fid']:.3f}  vs idle {r['inloop_vs_gate_idle']['A_only']}/{r['inloop_vs_gate_idle']['B_only']} p {r['inloop_vs_gate_idle']['p_exact']:.3f}  {r['loop']}", flush=True)
ra, ya, pa, _ = keep["A"]; rb, yb, pb, _ = keep["B"]
assert np.array_equal(ra, rb)
s = ra < 20000
res["inloop"]["A_vs_B"] = paired(pa, pb, ya)
res["inloop"]["A_vs_B_subset19400"] = paired(pa[s], pb[s], ya[s])

# ---------------------------------------------------------------- NN_axi, TLAST, no host alignment
AX = {}
for b in "AB":
    fs = sorted(glob.glob(str(D / "axi" / f"{b}_axi_*.npz")), key=lambda f: int(re.findall(r"_axi_(\d+)\.npz", f)[0]))
    assert len(fs) == 10, (b, len(fs))
    I, P, Y, RC = [], [], [], []
    for f in fs:
        z = np.load(f, allow_pickle=True)
        assert not bool(z["align"]) and int(z["realigns"].shape[0]) == 0
        I.append(z["idx"]); P.append(z["probs"]); Y.append(z["y"]); RC.append(z["live_rec"])
    d = dict(rows=np.concatenate(I), p=np.concatenate(P), y=np.concatenate(Y).astype(int), rec=np.concatenate(RC))
    o = np.argsort(d["rows"]); d = {k: v[o] for k, v in d.items()}
    assert np.array_equal(d["rows"], np.arange(20000))
    d["dec"] = (d["p"] > 0.5).astype(int); d["ldec"] = (d["rec"][:, 7] & 1).astype(int)
    AX[b] = d
m = calmask(np.arange(20000))
res["axi"] = {b: dict(subset19400=fstats(AX[b]["dec"][m], AX[b]["y"][m]), live_same_session=fstats(AX[b]["ldec"][m], AX[b]["y"][m]),
                      axi_vs_live_same_session=paired(AX[b]["dec"][m], AX[b]["ldec"][m], AX[b]["y"][m]),
                      axi_vs_live_run=paired(AX[b]["dec"][m], (A if b == "A" else B)["dec"][:20000][m], AX[b]["y"][m])) for b in "AB"}
res["axi"]["A_vs_B"] = paired(AX["A"]["dec"][m], AX["B"]["dec"][m], AX["A"]["y"][m])
pool_d = np.concatenate([AX[b]["dec"][m] for b in "AB"]); pool_y = np.concatenate([AX[b]["y"][m] for b in "AB"])
res["axi"]["both_boards_pooled"] = fstats(pool_d, pool_y)
for b in "AB":
    print(f"axi {b}: {res['axi'][b]['subset19400']['fid']:.3f}; live same session {res['axi'][b]['live_same_session']['fid']:.3f}; vs live {res['axi'][b]['axi_vs_live_same_session']['A_only']}/{res['axi'][b]['axi_vs_live_same_session']['B_only']}", flush=True)

np.savez_compressed(OUT / "merged_live.npz", rows=A["rows"], y=A["y"], dec_A=A["dec"], dec_B=B["dec"], dec_headline=H["dec"], rec_A=A["rec"], rec_B=B["rec"], clip_A=A["clip"], clip_B=B["clip"])
(OUT / "v144_stats.json").write_text(json.dumps(res, indent=1, default=float))
print("headline", res["live"]["headline"]["fid"], res["live"]["headline"]["fid_ci"])
print("wrote", OUT / "v144_stats.json")
