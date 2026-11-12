"""Generates paper_notes/V143_RESULTS.md and v143_results.json: the only list of results the paper may call v1.43.

The rules that keep other results out:
  1. Nobody edits the .md. It is written here, and --verify fails if it or any source file has changed since.
  2. A value enters only through src(), which refuses a file unless its path names v1.43 and it passes a build check.
  3. Each row states how strong its build evidence is, so a weak source is visible rather than hidden:
       banner     the file or its own log holds the v1.43 build banner read from the board during the run
       bracketed  no banner in the file; the same uninterrupted runner logged the banner before and after it
       build-file a Vivado report named v1.43, identical to the one beside the v1.43 bitstream
       name+date  named v1.43 and dated after the build, with no banner; weakest, listed at the end of the file
  check_paper_v143.py then compares the draft's numbers with this file.

"""
import glob
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
os.chdir(ROOT)
sys.path.insert(0, "audit_fixes/R3_R2")
from r3_stats import fstats, paired, calmask  # noqa: E402

MD = Path("paper_notes/V143_RESULTS.md")
JS = Path("campaign_v143/final/analysis/v143_results.json")
BANNER = "Nov  4 23:39:37"
BIT = Path("C:/qick_projects/top/OG_build_v1.43.bit")
BIT_SHA = "fe52e168296bd1b05b6e1ec052a028713d5e44c8dc2a662e3f14beac24f87bcd"
BUILT = datetime(2026, 11, 4, 23, 39, 37).timestamp()
D = "campaign_v143/final/"

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

SRC, ROWS = {}, []

def src(path, kind, log=None):
    """Admits a source file, or stops. Returns the path."""
    p = str(path).replace("\\", "/")
    if p in SRC:
        return p
    assert Path(p).exists(), f"missing source {p}"
    assert re.search(r"v143|v1\.43", p), f"path does not name v1.43: {p}"
    assert not re.search(r"v1\.?4[0-2]|v1\.?3\d", Path(p).name), f"file name names another build: {p}"
    assert Path(p).stat().st_mtime > BUILT, f"older than the v1.43 build: {p}"
    if kind == "banner":
        t = Path(log or p).read_text(encoding="utf-8", errors="ignore")
        assert BANNER in t, f"no v1.43 banner in {log or p}"
    elif kind == "bracketed":
        board = log
        for q in (f"{D}inloop/{board}_inloop_0_100000.log", f"{D}axi/{board}_axi_0.log"):
            assert BANNER in Path(q).read_text(encoding="utf-8", errors="ignore"), q
        for ph in ("live1", "live2"):
            assert json.load(open(f"{D}calib/{board}_{ph}.json"))["bit_sha256"] == BIT_SHA, ph
    elif kind == "build-file":
        name = Path(p).name
        assert sha(p) == sha(BIT.parent / name), f"differs from the report beside the bitstream: {p}"
    else:
        assert kind == "name+date", kind
    SRC[p] = dict(kind=kind, sha256=sha(p)[:16])
    return p

def row(section, key, value, text, source, note=""):
    """text = every form the value may take in the paper, e.g. ['95.55']."""
    ROWS.append(dict(section=section, key=key, value=value, text=[str(t) for t in (text if isinstance(text, list) else [text])], source=source, kind=SRC[source]["kind"], note=note))

def fid(section, key, s, source, note=""):
    row(section, key, f"{s['fid']:.2f}% [{s['fid_ci'][0]:.2f}, {s['fid_ci'][1]:.2f}], P(1|0) {s['p10']:.2f}, P(0|1) {s['p01']:.2f}, n = {s['n']:,}",
        [f"{s['fid']:.2f}", f"{s['fid_ci'][0]:.2f}", f"{s['fid_ci'][1]:.2f}", f"{s['p10']:.2f}", f"{s['p01']:.2f}", f"{s['n']:,}"], source, note)

def pv(p):
    return f"{p:.2f}" if p >= 0.01 else f"{p:.1g}"

def pair(section, key, s, source, note=""):
    row(section, key, f"{s['A_only']} vs {s['B_only']}, exact p = {pv(s['p_exact'])}, difference {s['dF']:+.3f} [{s['dF_ci'][0]:+.3f}, {s['dF_ci'][1]:+.3f}]",
        [str(s["A_only"]), str(s["B_only"]), pv(s["p_exact"]), f"{s['dF']:+.3f}"[1:], f"{s['dF_ci'][0]:+.3f}"[1:], f"{s['dF_ci'][1]:+.3f}"[1:]], source, note)

# ----------------------------------------------------------------------- the analysis output
assert sha(BIT) == BIT_SHA
stats_file = D + "analysis/v143_stats.json"
for b in "AB":
    src(f"{D}inloop/{b}_inloop_0_100000.log", "banner")
    src(f"{D}axi/{b}_axi_0.log", "banner")
src(stats_file, "bracketed", "A"); src(stats_file, "bracketed", "B")
S = json.load(open(stats_file))
T = S["bit_exact"]
n = sum(v["n"] for v in T.values())
assert all(v["all"] == v["n"] == 50000 for v in T.values()), T
row("NN_live", "Bit-exact shots (window, sums, sigmoid, decision), both boards", f"{n:,}/{n:,}", ["200,000"], stats_file)
L = S["live"]
assert L["window_eq_buffer"] == {"A": 100000, "B": 100000}
fid("NN_live", "Headline, 99,400 rows, run 1", L["headline"], stats_file)
fid("NN_live", "Full 100,000 rows, run 1", L["full100k"], stats_file)
fid("NN_live", "Fresh 80,000 rows", L["fresh80k"], stats_file)
fid("NN_live", "19,400 subset, board A", L["headline_subset19400_boardA"], stats_file)
fid("NN_live", "19,400 subset, board B", L["headline_subset19400_boardB"], stats_file)
fid("NN_live", "All 99,400 traces on board A", L["A_all"], stats_file)
fid("NN_live", "All 99,400 traces on board B", L["B_all"], stats_file)
pair("NN_live", "Boards A vs B, identical 99,400 traces", L["A_vs_B"], stats_file)
row("NN_live", "DAC-clipped shots, A / B", f"{L['dac_clipped_shots']['A']} / {L['dac_clipped_shots']['B']}", [L["dac_clipped_shots"]["A"]], stats_file)

M = np.load(D + "analysis/merged_live.npz")
src(D + "analysis/merged_live.npz", "bracketed", "A")
cal = calmask(M["rows"]); da, db, yy = M["dec_A"][cal], M["dec_B"][cal], M["y"][cal]
wa, wb = da != yy, db != yy
both, either = 100 * float((wa & wb).mean()), 100 * float((wa | wb).mean())
trace = 100 * 2 * both / (2 * both + (either - both))
row("Replay channel", "Decisions agree between boards", f"{100 * float((da == db).mean()):.2f}%", [f"{100 * float((da == db).mean()):.2f}"], stats_file)
row("Replay channel", "Wrong on both boards / on at least one", f"{both:.2f}% / {either:.2f}%", [f"{both:.2f}", f"{either:.2f}"], stats_file)
row("Replay channel", "Share of error that belongs to the trace / to replay noise", f"{trace:.0f}% / {100 - trace:.0f}%", [f"{trace:.0f}", f"{100 - trace:.0f}"], stats_file)

for b in "AB":
    r = S["inloop"][b]; lp = r["loop"]
    assert r["banner"].endswith("Wed Nov  4 23:39:37 2026")
    fid("In the loop", f"Board {b}, 99,400 rows", r["inloop"], stats_file)
    fid("In the loop", f"Board {b}, 19,400 subset", r["inloop_subset19400"], stats_file)
    fid("In the loop", f"Board {b}, fresh 80,000 rows", r["inloop_fresh80k"], stats_file)
    pair("In the loop", f"Board {b}, vs its gate-idle run", r["inloop_vs_gate_idle"], stats_file)
    row("In the loop", f"Board {b}, pulse == gate == decision", f"{lp['correct']:,}/{r['n']:,}", [f"{lp['correct']:,}"], stats_file)
    row("In the loop", f"Board {b}, stale / late / unsafe / binding rejects", f"{lp['stale']} / {lp['late']} / {lp['unsafe']} / {lp['binding_rejects']}", ["0"], stats_file)
    row("In the loop", f"Board {b}, slack", f"{1e3 * r['slack_us']:.0f} ns", [f"{1e3 * r['slack_us']:.0f}"[1:]], stats_file)
    row("In the loop", f"Board {b}, DAC-clipped shots", f"{lp['dac_clipped_shots']}", [lp["dac_clipped_shots"]], stats_file)
row("In the loop", "In-loop shots, both boards", "198,800", ["198,800"], stats_file)
ub = S["inloop"]["A"]["loop"]["late_rate_upper95"]
row("In the loop", "Late-rate 95% upper bound, per board / pooled", f"{100 * ub:.4f}% / {100 * (1 - 0.05 ** (1 / 198800)):.4f}%", [f"{100 * ub:.3f}", f"{100 * (1 - 0.05 ** (1 / 198800)):.4f}"], stats_file)
pair("In the loop", "Boards A vs B", S["inloop"]["A_vs_B"], stats_file)

for b in "AB":
    x = S["axi"][b]
    fid("NN_axi", f"Board {b}, 19,400 rows, TLAST, no host alignment", x["subset19400"], stats_file)
    pair("NN_axi", f"Board {b}, vs NN_live in the same session", x["axi_vs_live_same_session"], stats_file)
    pair("NN_axi", f"Board {b}, vs NN_live run on the same rows", x["axi_vs_live_run"], stats_file)
pair("NN_axi", "Boards A vs B", S["axi"]["A_vs_B"], stats_file)
fid("NN_axi", "Both boards pooled, 38,800 shots", S["axi"]["both_boards_pooled"], stats_file)

# Timing argument in Sec. X: NN_live vs NN_axi on the same 20,000 rows of one board
for b in "AB":
    I, P, Y = [], [], []
    for f in glob.glob(f"{D}axi/{b}_axi_*.npz"):
        z = np.load(f, allow_pickle=True); I.append(z["idx"]); P.append(z["probs"]); Y.append(z["y"])
    I = np.concatenate(I); o = np.argsort(I); dx = (np.concatenate(P)[o] > .5).astype(int); yx = np.concatenate(Y)[o].astype(int)
    dl = M[f"dec_{b}"][:20000]
    fid("Timing checks", f"Board {b}, NN_live, same 20,000 rows", fstats(dl, yx), stats_file)
    fid("Timing checks", f"Board {b}, NN_axi, same 20,000 rows", fstats(dx, yx), stats_file)
    pair("Timing checks", f"Board {b}, NN_live vs NN_axi, 20,000 rows", paired(dl, dx, yx), stats_file)

# ----------------------------------------------------------------------- NN_axi vs its emulator, and the linear baselines on v1.43 replays
ae = src(D + "analysis/axi_emu.json", "banner", D + "axi/A_axi_0.log"); AE = json.load(open(ae))
assert all(AE[b]["identical"] == AE[b]["n"] == 20000 for b in "AB")
row("NN_axi", "Identical to its bit-accurate emulator on the recorded window, both boards", "40,000/40,000", ["40,000", "20,000"], ae)
for b in "AB":
    fid("NN_axi", f"Emulator, board {b}, 19,400 rows", AE[b]["emu_subset19400"], ae)
    q = src(D + f"analysis/linear_v143_{b}.json", "banner", D + f"axi/{b}_axi_0.log"); Q = json.load(open(q))
    fid("Linear baselines", f"Matched filter, board {b}", Q["linear_baselines"]["200000"]["MF"]["h19400"], q)
    W_ = Q["window_tradeoff"]
    fid("Linear baselines", f"LDA, 770 samples, board {b}", W_["770"]["h19400"], q)
    fid("Linear baselines", f"LDA, 576 samples, board {b}", W_["576"]["h19400"], q)
    pair("Linear baselines", f"LDA, 576 vs 640 samples, board {b}", W_["576"]["vs640"]["h19400"], q)
    pair("Linear baselines", f"LDA, 770 vs 640 samples, board {b}", W_["770"]["vs640"]["h19400"], q)
pair("In the loop", "Boards A vs B, 19,400 subset", S["inloop"]["A_vs_B_subset19400"], stats_file)
for b in "AB":
    pair("In the loop", f"Board {b}, vs gate-idle, 19,400 subset", S["inloop"][b]["inloop_vs_gate_idle_subset19400"], stats_file)

# ----------------------------------------------------------------------- drift
for h, b in (("10.1.9.26", "A"), ("10.1.4.105", "B")):
    p = src(f"{D}drift_full/drift_{h}.json", "banner", f"{D}drift_full/{b}_drift.log")
    J = json.load(open(p)); ph = np.array([s["phase"] for s in J["samples"]])
    row("Drift, 3 h", f"Board {b}, phase mean, SD, largest distance from mean", f"{ph.mean():.2f} deg, {ph.std(ddof=1):.3f} deg, {abs(ph - ph.mean()).max():.2f} deg", [f"{abs(ph - ph.mean()).max():.2f}", "0.25", f"{ph.std(ddof=1):.2f}"], p)
    s0, s1 = [], []
    for tag, out in (("start", s0), ("end", s1)):
        for f in J[f"score_{tag}"]:
            z = np.load(f.replace("\\", "/"), allow_pickle=True); out.append((z["idx"], z["probs"], z["y"]))
    i0, i1 = (np.concatenate([x[0] for x in s]) for s in (s0, s1))
    assert np.array_equal(i0, i1) and len(i0) == 6000
    a, c, y = (np.concatenate([x[1] for x in s0]) > .5).astype(int), (np.concatenate([x[1] for x in s1]) > .5).astype(int), np.concatenate([x[2] for x in s0]).astype(int)
    q = paired(c, a, y)
    row("Drift, 3 h", f"Board {b}, end minus start, 6,000 paired shots", f"{q['dF']:+.2f} points, {q['B_only']} vs {q['A_only']}, p = {pv(q['p_exact'])}", [f"{q['dF']:+.2f}"[1:], "6,000"], p)

# ----------------------------------------------------------------------- second board session and rescoring (2026-11-06)
B2 = D + "board2/"
for b in "AB":
    q = src(f"{B2}{b}_tproc_box_v143.json", "banner"); J = json.load(open(q))
    for v in ("nobranch", "qick"):
        g_ = [x for x in J["points"] if x["variant"] == v and x["fresh"] == x["n"] == x["br_eq_dec"] == x["played_eq_br"]][0]
        row("tProc path", f"Board {b}, boxcar path, {v}, earliest clean margin", f"{g_['L_median']:.1f} ns at +{1e3 * g_['margin']:.0f} ns, {g_['fresh']}/{g_['n']} fresh", [f"{g_['L_median']:.1f}", f"{g_['L_median']:.0f}"], q)
    q = src(f"{B2}{b}_tproc_fir_v143.json", "banner"); J = json.load(open(q))
    g_ = [x for x in J["points"] if abs(x["margin"] - 0.035) < 1e-6][0]
    row("tProc path", f"Board {b}, FIR path, same margin as the old L5 value", f"{g_['L_median']:.1f} ns", [f"{g_['L_median']:.1f}"], q)
    q = src(f"{B2}{b}_boxcar_mr_v143.json", "banner", f"{B2}{b}_boxcar_mr_v143.log"); J = json.load(open(q))
    row("Safety and tests", f"Boxcar vs raw buffer, board {b}, with banner", f"{J['shots_exact']}/{J['shots']} shots exact, {J['verdict']}", [], q)
# R.19 leaked energy over blanked shots, Gaussian conditional pulse (analysis_scripts/diagnostics/leak_energy_check.py)
LK = "campaign_v143/final/leak_energy/"
lk = {}
for b in "AB":
    for thr in (8, 2000):
        q = src(f"{LK}{b}_v143_leak_thr{thr}.json", "banner", f"{LK}{b}_v143_leak_thr{thr}.log"); lk[b, thr] = (q, json.load(open(q)))
        assert lk[b, thr][1]["control_sees_leak"], (b, thr)
row("Safety and tests", "Leaked energy, gate blanking every window, DELAY from the first nonzero sample (nz_thr 8): largest upper 95% bound at leads 2-4, A / B",
    f"{100 * lk['A', 8][1]['max_upper_ci_lead_2_4']:.3f}% / {100 * lk['B', 8][1]['max_upper_ci_lead_2_4']:.3f}%, {lk['A', 8][1]['shots']} shots per setting; positive control resolves a leak",
    ["0.02", str(lk["A", 8][1]["shots"])], lk["A", 8][0], "paper: below 0.02%; B's file is " + lk["B", 8][0])
row("Safety and tests", "Leaked energy with DELAY from the 2000-amplitude crossing (nz_thr 2000), lead 4 mean, A / B",
    f"{100 * lk['A', 2000][1]['leads']['4']['frac_mean']:.3f}% / {100 * lk['B', 2000][1]['leads']['4']['frac_mean']:.3f}%",
    ["0.25"], lk["A", 2000][0], "paper: about 0.25%; B's file is " + lk["B", 2000][0])
for b in "AB":
    for thr in (8, 2000):
        src(f"{LK}{b}_v143_leak_thr{thr}.json", "banner", f"{LK}{b}_v143_leak_thr{thr}.log")
La = [x for x in json.load(open(f"{B2}A_tproc_box_v143.json"))["points"] if x["variant"] == "nobranch" and x["fresh"] == x["n"]][0]["L_median"]
row("tProc path", "Gate saving over the tProc path: board A tProc minus L_50 (176.0), and its population worth at T1 = 32 us", f"{La - 176.0:.0f} ns; {(La - 176.0) / 960:.2f} to {(La - 176.0) / 320:.2f} points", [f"{La - 176.0:.0f}", f"{(La - 176.0) / 960:.2f}", f"{(La - 176.0) / 320:.2f}"], f"{B2}A_tproc_box_v143.json", "t/(3 T1) and t/T1")

def _f(path):
    z = np.load(path, allow_pickle=True); y = z["y"].astype(int); d = (z["live_rec"][:, 7] & 1).astype(int)
    return fstats(d, y), float(np.degrees(np.angle(complex(z["k"]))))

for t, lab in (("A_own", "Board A, own calibration"), ("B_own", "Board B, own calibration"), ("B_on_A_calibration", "Board B on board A's calibration")):
    q = src(f"{B2}swap_{t}_v143.npz", "banner", f"{B2}swap_{t}.log"); fs, ang = _f(q)
    fid("Calibration swap", f"{lab}, 2,000 shots, arg(k) {ang:.1f} deg", fs, q)
ka, kb = (_f(f"{B2}swap_{t}_v143.npz")[1] for t in ("A_own", "B_own"))
row("Calibration swap", "Phase difference of the boards in that session", f"{abs(ka - kb):.1f} deg", [f"{abs(ka - kb):.0f}"], f"{B2}swap_B_own_v143.npz")
q = src(f"{B2}A_phrst_probe_v143.json", "banner"); J = json.load(open(q))
assert len(J["loads"]) == 4 and all(BANNER in l["banner"] for l in J["loads"])
for m_ in ("none", "both"):
    row("Phase across loads", f"Board A, 4 loads, mode {m_}: mean per load", ", ".join(f"{float(np.degrees(np.angle(np.exp(1j * np.radians(l['modes'][m_]['shots'])).mean()))):.0f}" for l in J["loads"]) + " deg", [], q)

q = src(D + "analysis/box_vs_fir_v143.json", "banner", D + "axi/A_axi_0.log"); J = json.load(open(q))
assert J["box_emu_eq_chip"] == J["n"] == 100000
fid("Boxcar vs FIR", "Boxcar, 19,400, board A", J["subset19400"]["box"], q)
fid("Boxcar vs FIR", "FIR model, 19,400, board A", J["subset19400"]["fir"], q)
pair("Boxcar vs FIR", "Boxcar vs FIR, 19,400", J["subset19400"]["paired"], q)
pair("Boxcar vs FIR", "Boxcar vs FIR, 99,400", J["all99400"]["paired"], q)

q = src(D + "analysis/r9/r9_stats_v143.json", "banner", D + "axi/A_axi_0.log"); J = json.load(open(q))["19400"]
for c_ in J["comparisons"]:
    row("Synthetic data", f"{c_['A']} minus {c_['B']}", f"{c_['diff']:+.2f} [{c_['ci95'][0]:+.2f}, {c_['ci95'][1]:+.2f}]", [f"{c_['diff']:+.2f}"[1:], f"{c_['ci95'][0]:+.2f}"[1:], f"{c_['ci95'][1]:+.2f}"[1:]], q)
q = src(D + "analysis/arch/arch_sweep2_src_v143.json", "banner", D + "axi/A_axi_0.log"); J = json.load(open(q))
row("Nonlinear models", "LDA, 20,000 rows: raw / replayed", f"{J['lda_ref']['src20k']:.2f}% / {J['lda_ref']['loop20k']:.2f}%", [f"{J['lda_ref']['src20k']:.2f}", f"{J['lda_ref']['loop20k']:.2f}"], q)
for k_ in ("seg_hgb", "seg_mlp", "bin_mlp", "cnn1d", "bin_gru"):
    v_ = J[k_]; t_ = v_["loop20k_vs_t4"]
    row("Nonlinear models", f"{k_}: raw / replayed, vs LDA", f"{v_['src20k']:.2f}% / {v_['loop20k']:.2f}%, {t_['delta']:+.2f}, p = {pv(t_['p'])}", [f"{v_['src20k']:.2f}", f"{v_['loop20k']:.2f}", f"{abs(t_['delta']):.2f}", pv(t_["p"])], q)
q = src(D + "analysis/channel_v143.json", "banner", D + "axi/A_axi_0.log"); J = json.load(open(q))
row("Replay channel", "Class SNR per sample, source / replayed board A (rms mean separation over rms noise)", f"{J['snr_source']:.2f} / {J['snr_A']:.2f}", [f"{J['snr_source']:.2f}", f"{J['snr_A']:.2f}"], q)
row("Replay channel", "Capture delay against the aligned source, 4,000 shots", f"{min(J['lags'], key=int)}-{max(J['lags'], key=int)} samples", [], q)

# ----------------------------------------------------------------------- Fig. 7
fv = src("paper_figures/fig7/fig7_capture_A_v143.json", "banner")
src("paper_figures/fig7/fig7_sweep_A_v143.json", "banner"); src("paper_figures/fig7/fig7_sweep_B_v143.json", "banner"); src("paper_figures/fig7/fig7_capture_B_v143.json", "banner")
V = json.load(open("paper_figures/fig7/fig7_values.json"))
assert "v143" in Path("paper_figures/fig7/make_fig7_boxcar.py").read_text(encoding="utf-8") and "v141" not in Path("paper_figures/fig7/make_fig7_boxcar.py").read_text(encoding="utf-8")
row("Readout path", "Boxcar saves, board A", f"{V['saved_ns']:.1f} ns", [f"{V['saved_ns']:.1f}"], fv)
im = dict(zip(V["image_dBc"]["M"], V["image_dBc"]["box_measured"]))
_fir = V["saved_ns"] + 9 * 1e3 / 552.96 + 5 * 1e3 / 552.96
row("Readout path", "QICK FIR path, mixer output to decimated edge: saving + boxcar path (9 cycles + group delay (M-1)/2 = 5 samples)", f"{_fir:.1f} ns", [f"{_fir:.0f}"], fv, "check: FIR core C_LATENCY 67 cycles + 120-tap group delay 59.5/4423.68 us = 134.6 ns")
row("Readout path", "Image rejection at M = 11, board A", f"{im[11]:.1f} dB", [f"{abs(im[11]):.1f}"], fv)

# ----------------------------------------------------------------------- Table III
sys.path.insert(0, "paper_tables/table3"); import v141_reports as R3  # noqa: E402
R3.HIER, R3.FLAT, R3.TIMING = (Path("paper_tables/table3/data") / f"v1.43_{r}.rpt" for r in ("routed_util_hier", "routed_utilization", "routed_timing"))  # pinned: the readers default to v1.44 since 2026-11-09
TR = "paper_tables/table3/data/"
for r in ("routed_timing", "routed_util_hier", "routed_utilization"):
    src(f"{TR}v1.43_{r}.rpt", "build-file")
tv = json.load(open("paper_tables/table3/table3_values.json")); tt = f"{TR}v1.43_routed_timing.rpt"; tu = f"{TR}v1.43_routed_util_hier.rpt"
row("Implementation", "NN_live LUT / FF", f"{tv['nn_live']['lut']:,} / {tv['nn_live']['ff']:,}", [f"{tv['nn_live']['lut']:,}", f"{tv['nn_live']['ff']:,}"], tu)
row("Implementation", "NN_axi LUT / FF", f"{tv['nn_axi']['lut']:,} / {tv['nn_axi']['ff']:,}", [f"{tv['nn_axi']['lut']:,}", f"{tv['nn_axi']['ff']:,}", f"{100 * tv['nn_axi']['lut'] / 425280:.1f}"], tu)
row("Implementation", "WNS / failing setup endpoints", f"{tv['timing']['wns']} ns / {tv['timing']['setup_failing']}", [f"{abs(tv['timing']['wns']):.3f}", tv["timing"]["setup_failing"]], tt)
_rs = json.load(open("scaling_study/resource_scaling_v143.json"))
assert _rs["per_unit"]["classifier"]["lut"] == sum(R3.blocks()[n]["lut"] for n in ("NN_live_0", "boxcar_tap_0", "live_gate_0", "live_res_cdc_0", "live_res_cdc_1", "trig_sync_0"))
_n = [v["n_max"] for k, v in _rs["n_max"].items() if "front end none" not in k]
row("Implementation", "Classifier units that fit with a per-tone front end (scaling_study/resource_scaling.py, SCALING_OUT=resource_scaling_v143.json)", f"{min(_n)} to {max(_n)}", [min(_n), max(_n)], tu, "unit RTL unchanged since the OOC timing runs of 2026-10-27")
m = re.search(r"^clk_pl_0\s+(-?[\d.]+)", Path(tt).read_text(errors="ignore"), flags=re.M)
row("Implementation", "NN_axi clock (clk_pl_0) WNS", f"+{m[1]} ns", [m[1], f"{float(m[1]):.2f}"], tt)

# ----------------------------------------------------------------------- loop latency and safety, weaker evidence
G = json.load(open("campaign_v143/final/analysis/loop_groups_values_v143.json"))
assert G == json.load(open("paper_figures/fig8/loop_groups_values.json")), "copy of the Fig. 8(a) values is out of date"
for name in ("low", "high"):                 # every load in both groups: the banner read from the board in that load's own run
    for load in G[name]["loads"]:
        sub = "cold" if "_c" in load else "reloads"
        q = f"campaign_v140L1/latency_protocol/{sub}/{load}.json"
        assert BANNER in Path(q).read_text(errors="ignore"), f"no v1.43 banner in {q}"
g = src("campaign_v143/final/analysis/loop_groups_values_v143.json", "banner", f"campaign_v140L1/latency_protocol/reloads/{G['low']['loads'][0]}.json")
for name in ("low", "high"):
    x = G[name]
    row("Loop latency", f"L_50, {name} load group ({x['n_loads']} loads)", f"{x['L_50']:.1f} ns, range {x['L_50_min']:.1f}-{x['L_50_max']:.1f}", [f"{x['L_50']:.1f}", f"{x['L_50_min']:.1f}", f"{x['L_50_max']:.1f}"], g)
    row("Loop latency", f"Segments, {name} group: S_fe, S_NN, S_CDC, S_gate, S_conv", ", ".join(f"{v:.1f}" for v in x["segs"].values()) + " ns", [f"{v:.1f}" for v in x["segs"].values()], g)
_h = G["high"]["segs"]
row("Loop latency", "Our logic (S_NN + S_CDC + S_gate) / converters and cable (S_conv), rounded, 176 ns group", f"{_h['S_nn'] + _h['S_cdc'] + _h['S_gate']:.0f} / {_h['S_conv']:.0f} ns", [f"{_h['S_nn'] + _h['S_cdc'] + _h['S_gate']:.0f}", f"{_h['S_conv']:.0f}", "176", "174"], g)
_sc = (G["low"]["segs"]["S_conv"] + _h["S_conv"]) / 2
row("Loop latency", "Converters and cable, mean of the two load groups", f"{_sc:.1f} ns ({G['low']['segs']['S_conv']:.1f} and {_h['S_conv']:.1f})", [f"{_sc:.0f}"], g)
_b = src("campaign_v140L1/v143_fig8b/A_gate_probe.log", "banner")
_v = json.load(open("paper_figures/fig8/fig8b_values.json"))
_l = {b: [x for x in _v[b] if x["slack_ns"] == -66.0][0]["late"] for b in "AB"}
row("Loop latency", "Fig. 8(b), late at -66 ns, boards A / B, of 40", f"{_l['A']} / {_l['B']}", [_l["A"], _l["B"]], _b)
row("Implementation", "NN_live LUT, rounded for prose", "about 1,340", ["1,340"], tu)
E = list(__import__("csv").DictReader(open("campaign_v143/pre_6_3/v143_endpoints_summary.csv")))
assert sum(int(r["n"]) for r in E) == tv["timing"]["setup_failing"]
cnt = {}
for r in E:
    cnt[r["block"]] = cnt.get(r["block"], 0) + int(r["n"])
row("Implementation", "Failing endpoints: NN_live / its clock crossing / stock readouts 1 and 2", f"{cnt.get('NN_live_0', 0)} / {cnt.get('live_res_cdc_1', 0)} / {cnt['axis_dyn_readout_v1_1'] + cnt['axis_dyn_readout_v1_2']}", [cnt["axis_dyn_readout_v1_1"] + cnt["axis_dyn_readout_v1_2"]], tt)
for f, name in (("campaign_v140L1/A_v143_binding_fault.json", "Binding fault test, board A"), ("campaign_v143/pre_6_3/B_binding_fault.json", "Binding fault test, board B"),
                ("campaign_v140L1/A_v143_fail_safe.json", "Fail-safe test, board A"), ("campaign_v140L1/A_v143_tproc_decision.json", "tProc decision test, board A"),
                ("campaign_v143/pre_6_3/B_tproc_decision.json", "tProc decision test, board B")):
    lg = f[:-5] + ".log"
    has = Path(lg).exists() and BANNER in Path(lg).read_text(encoding="utf-8", errors="ignore")
    p = src(f, "banner", lg) if has else src(f, "name+date"); j = json.load(open(p))
    row("Safety and tests", name, str(j.get("verdict")), [], p)
for f, name in (("campaign_v140L1/A_v143_functional.json", "A"), ("campaign_v143/pre_6_3/B_functional.json", "B")):
    p = src(f, "banner", f[:-5] + ".log"); j = json.load(open(p))["summary"]
    row("Safety and tests", f"QICK functional suite, board {name}", f"{j['PASS']} pass, {j['WARN']} warning, {j['FAIL']} fail, {j['NOT WIRED']} not wired", [j["PASS"], j["WARN"], j["FAIL"]], p)

# ----------------------------------------------------------------------- output
order = list(dict.fromkeys(r["section"] for r in ROWS))
out = ["# v1.43 results ledger", "", "**Generated by `campaign_v143/final/analysis/make_v143_ledger.py`. Do not edit by hand.** Add a result by adding it to that script,",
       "which refuses any source that fails a build check. `--verify` fails if this file or a source has changed since it was generated.", "",
       f"Build: `{BIT.name}`, sha256 `{BIT_SHA[:16]}...`, banner `{BANNER} 2026`.", "",
       "Evidence: **banner** = the v1.43 banner read from the board during that run; **bracketed** = banner logged before and after by the same uninterrupted runner; "
       "**build-file** = the report beside the v1.43 bitstream; **name+date** = named v1.43 and dated after the build, no banner (weakest).", ""]
for sec in order:
    out += [f"## {sec}", "", "| Result | Value | Evidence | Source |", "|---|---|---|---|"]
    for r in ROWS:
        if r["section"] == sec:
            out.append(f"| {r['key']} | {r['value']} | {r['kind']} | `{r['source']}` |")
    out.append("")
weak = sorted(p for p, v in SRC.items() if v["kind"] == "name+date")
out += ["## Sources with weak build evidence", "", "These carry no banner. Confirm them or re-measure before submission.", ""] + [f"- `{p}`" for p in weak] + [""]
text = "\n".join(out)
payload = dict(rows=ROWS, sources=SRC, md_sha256=hashlib.sha256(text.encode()).hexdigest())
if "--verify" in sys.argv:
    old = json.load(open(JS))
    bad = [p for p, v in old["sources"].items() if sha(p)[:16] != v["sha256"]]
    if MD.read_text(encoding="utf-8") != text or old["md_sha256"] != payload["md_sha256"]:
        bad.append(str(MD))
    print("VERIFY", "FAILED: " + ", ".join(bad) if bad else f"OK: {len(ROWS)} results, {len(SRC)} sources")
    sys.exit(1 if bad else 0)
MD.write_text(text, encoding="utf-8")
JS.write_text(json.dumps(payload, indent=1, default=str))
print(f"wrote {MD}: {len(ROWS)} results, {len(SRC)} sources, {len(weak)} weak")
