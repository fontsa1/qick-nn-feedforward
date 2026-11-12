"""Generates paper_notes/V144_RESULTS.md and v144_results.json: the only list of results the paper may call v1.44.

Port of campaign_v143/final/analysis/make_v143_ledger.py. It lists the same results in the same order (the v1.43 ledger is
the template), so the two files can be compared row by row.

The rules that keep other results out:
  1. Nobody edits the .md. It is written here, and --verify fails if it or any source file has changed since.
  2. A value enters only through src(), which refuses a file unless its path names v1.44 and it passes a build check.
  3. Each row states how strong its build evidence is:
       banner     the file or its own log holds the v1.44 build banner read from the board during the run
       bracketed  no banner in the file; the same uninterrupted runner logged the banner before and after it
       build-file a Vivado report named v1.44, identical to the one saved from the v1.44 build
       name+date  named v1.44 and dated after the build, with no banner; weakest, listed at the end of the file
  4. A result whose source does not exist yet is shown as PENDING with no value. A source that exists and fails a check
     stops the script. --verify fails while anything is pending, so a partial file cannot pass for a finished one.

"""
import glob
import hashlib
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
os.chdir(ROOT)
sys.path.insert(0, "audit_fixes/R3_R2")
from r3_stats import fstats, paired, calmask  # noqa: E402

MD = Path("paper_notes/V144_RESULTS.md")
JS = Path("campaign_v144/final/analysis/v144_results.json")
TEMPLATE = Path("campaign_v143/final/analysis/v143_results.json")
BANNER = "Nov  7 19:57:07"
BIT = Path("C:/qick_projects/top/OG_build_v1.44.bit")
BIT_SHA = "986c861a83be8d01d72db967af894e3d01de2b1c861021d1b5821bc183d7b7c6"
BUILT = datetime(2026, 11, 7, 19, 57, 7).timestamp()
REPORTS = Path("C:/qick_projects/top/v1.44_reports")     # the build kept the v1.43_* report names; the design there is v1.44
D = "campaign_v144/final/"
B2 = D + "board2/"
LK = D + "leak_energy/"
TR = "paper_tables/table3/data/"
HOSTS = (("10.1.9.26", "A"), ("10.1.4.105", "B"))

class Pending(Exception):
    """A source does not exist yet."""

def need(path):
    if not Path(path).exists():
        raise Pending(str(path))
    return path

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

SRC, ROWS, BLOCKED = {}, [], []

def src(path, kind, log=None):
    """Admits a source file, raises Pending if it does not exist yet, or stops. Returns the path."""
    p = str(path).replace("\\", "/")
    if p in SRC:
        return p
    need(p)
    assert re.search(r"v144|v1\.44", p), f"path does not name v1.44: {p}"
    assert not re.search(r"v1\.?4[0-3]|v1\.?3\d", Path(p).name), f"file name names another build: {p}"
    assert Path(p).stat().st_mtime > BUILT, f"older than the v1.44 build: {p}"
    if kind == "banner":
        t = Path(need(log or p)).read_text(encoding="utf-8", errors="ignore")
        assert BANNER in t, f"no v1.44 banner in {log or p}"
    elif kind == "bracketed":
        board = log
        for q in (f"{D}inloop/{board}_inloop_0_100000.log", f"{D}axi/{board}_axi_0.log"):
            assert BANNER in Path(need(q)).read_text(encoding="utf-8", errors="ignore"), q
        for ph in ("live1", "live2"):
            assert json.load(open(need(f"{D}calib/{board}_{ph}.json")))["bit_sha256"] == BIT_SHA, ph
    elif kind == "build-file":
        name = Path(p).name.replace("v1.44_", "v1.43_", 1)
        assert sha(p) == sha(REPORTS / name), f"differs from the report saved from the build: {p}"
    else:
        assert kind == "name+date", kind
    SRC[p] = dict(kind=kind, sha256=sha(p)[:16])
    return p

def attempt(fn, *args):
    """Runs one group of rows. Rows finished before a missing source stay in the ledger."""
    try:
        fn(*args)
    except Pending as e:
        BLOCKED.append(str(e))

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

def stats():
    """The analysis output, once the campaign data has been analysed. Checked against both boards' runners."""
    f = D + "analysis/v144_stats.json"
    need(f)
    for b in "AB":
        src(f"{D}inloop/{b}_inloop_0_100000.log", "banner")
        src(f"{D}axi/{b}_axi_0.log", "banner")
    src(f, "bracketed", "A")
    SRC.pop(f)                   # admit it once for each board's bracket
    src(f, "bracketed", "B")
    return f, json.load(open(f))

def merged():
    f = D + "analysis/merged_live.npz"
    src(f, "bracketed", "A")
    return np.load(f)

def loop_groups():
    return json.load(open(need(D + "analysis/loop_groups_values_v144.json")))

# ----------------------------------------------------------------------- the analysis output
def nn_live():
    f, S = stats()
    T = S["bit_exact"]
    n = sum(v["n"] for v in T.values())
    assert all(v["all"] == v["n"] == 50000 for v in T.values()), T
    row("NN_live", "Bit-exact shots (window, sums, sigmoid, decision), both boards", f"{n:,}/{n:,}", [f"{n:,}"], f)
    L = S["live"]
    assert L["window_eq_buffer"] == {"A": 100000, "B": 100000}
    fid("NN_live", "Headline, 99,400 rows, run 1", L["headline"], f)
    fid("NN_live", "Full 100,000 rows, run 1", L["full100k"], f)
    fid("NN_live", "Fresh 80,000 rows", L["fresh80k"], f)
    fid("NN_live", "19,400 subset, board A", L["headline_subset19400_boardA"], f)
    fid("NN_live", "19,400 subset, board B", L["headline_subset19400_boardB"], f)
    fid("NN_live", "All 99,400 traces on board A", L["A_all"], f)
    fid("NN_live", "All 99,400 traces on board B", L["B_all"], f)
    pair("NN_live", "Boards A vs B, identical 99,400 traces", L["A_vs_B"], f)
    row("NN_live", "DAC-clipped shots, A / B", f"{L['dac_clipped_shots']['A']} / {L['dac_clipped_shots']['B']}", [L["dac_clipped_shots"]["A"]], f)

def replay_decisions():
    f, _ = stats()
    M = merged()
    cal = calmask(M["rows"]); da, db, yy = M["dec_A"][cal], M["dec_B"][cal], M["y"][cal]
    wa, wb = da != yy, db != yy
    both, either = 100 * float((wa & wb).mean()), 100 * float((wa | wb).mean())
    trace = 100 * 2 * both / (2 * both + (either - both))
    agree = 100 * float((da == db).mean())
    row("Replay channel", "Decisions agree between boards", f"{agree:.2f}%", [f"{agree:.2f}"], f)
    row("Replay channel", "Wrong on both boards / on at least one", f"{both:.2f}% / {either:.2f}%", [f"{both:.2f}", f"{either:.2f}"], f)
    row("Replay channel", "Share of error that belongs to the trace / to replay noise", f"{trace:.0f}% / {100 - trace:.0f}%", [f"{trace:.0f}", f"{100 - trace:.0f}"], f)

def inloop():
    f, S = stats()
    for b in "AB":
        r = S["inloop"][b]; lp = r["loop"]
        assert BANNER in r["banner"], r["banner"]
        fid("In the loop", f"Board {b}, 99,400 rows", r["inloop"], f)
        fid("In the loop", f"Board {b}, 19,400 subset", r["inloop_subset19400"], f)
        fid("In the loop", f"Board {b}, fresh 80,000 rows", r["inloop_fresh80k"], f)
        pair("In the loop", f"Board {b}, vs its gate-idle run", r["inloop_vs_gate_idle"], f)
        row("In the loop", f"Board {b}, pulse == gate == decision", f"{lp['correct']:,}/{r['n']:,}", [f"{lp['correct']:,}"], f)
        row("In the loop", f"Board {b}, stale / late / unsafe / binding rejects", f"{lp['stale']} / {lp['late']} / {lp['unsafe']} / {lp['binding_rejects']}", ["0"], f)
        row("In the loop", f"Board {b}, slack", f"{1e3 * r['slack_us']:.0f} ns", [f"{abs(1e3 * r['slack_us']):.0f}"], f)
        row("In the loop", f"Board {b}, DAC-clipped shots", f"{lp['dac_clipped_shots']}", [lp["dac_clipped_shots"]], f)
    N = S["inloop"]["A"]["n"] + S["inloop"]["B"]["n"]
    row("In the loop", "In-loop shots, both boards", f"{N:,}", [f"{N:,}"], f)
    ub = S["inloop"]["A"]["loop"]["late_rate_upper95"]
    row("In the loop", "Late-rate 95% upper bound, per board / pooled", f"{100 * ub:.4f}% / {100 * (1 - 0.05 ** (1 / N)):.4f}%", [f"{100 * ub:.3f}", f"{100 * (1 - 0.05 ** (1 / N)):.4f}"], f)
    pair("In the loop", "Boards A vs B", S["inloop"]["A_vs_B"], f)
    pair("In the loop", "Boards A vs B, 19,400 subset", S["inloop"]["A_vs_B_subset19400"], f)
    for b in "AB":
        pair("In the loop", f"Board {b}, vs gate-idle, 19,400 subset", S["inloop"][b]["inloop_vs_gate_idle_subset19400"], f)

def axi():
    f, S = stats()
    for b in "AB":
        x = S["axi"][b]
        fid("NN_axi", f"Board {b}, 19,400 rows, TLAST, no host alignment", x["subset19400"], f)
        pair("NN_axi", f"Board {b}, vs NN_live in the same session", x["axi_vs_live_same_session"], f)
        pair("NN_axi", f"Board {b}, vs NN_live run on the same rows", x["axi_vs_live_run"], f)
    pair("NN_axi", "Boards A vs B", S["axi"]["A_vs_B"], f)
    fid("NN_axi", "Both boards pooled, 38,800 shots", S["axi"]["both_boards_pooled"], f)

def timing_checks():
    """Timing argument in the paper: NN_live vs NN_axi on the same 20,000 rows of one board."""
    f, _ = stats()
    M = merged()
    for b in "AB":
        I, P, Y = [], [], []
        for g in glob.glob(f"{D}axi/{b}_axi_*.npz"):
            z = np.load(g, allow_pickle=True); I.append(z["idx"]); P.append(z["probs"]); Y.append(z["y"])
        I = np.concatenate(I); o = np.argsort(I); dx = (np.concatenate(P)[o] > .5).astype(int); yx = np.concatenate(Y)[o].astype(int)
        assert len(I) == 20000, (b, len(I))
        dl = M[f"dec_{b}"][:20000]
        fid("Timing checks", f"Board {b}, NN_live, same 20,000 rows", fstats(dl, yx), f)
        fid("Timing checks", f"Board {b}, NN_axi, same 20,000 rows", fstats(dx, yx), f)
        pair("Timing checks", f"Board {b}, NN_live vs NN_axi, 20,000 rows", paired(dl, dx, yx), f)

# ----------------------------------------------------------------------- NN_axi vs its emulator, and the linear baselines on v1.44 replays
def emulator():
    ae = src(D + "analysis/axi_emu_v144.json", "banner", D + "axi/A_axi_0.log"); AE = json.load(open(ae))
    assert all(AE[b]["identical"] == AE[b]["n"] == 20000 for b in "AB")
    row("NN_axi", "Identical to its bit-accurate emulator on the recorded window, both boards", "40,000/40,000", ["40,000", "20,000"], ae)
    for b in "AB":
        fid("NN_axi", f"Emulator, board {b}, 19,400 rows", AE[b]["emu_subset19400"], ae)

def linear(b):
    q = src(D + f"analysis/linear_v144_{b}.json", "banner", D + f"axi/{b}_axi_0.log"); Q = json.load(open(q))
    fid("Linear baselines", f"Matched filter, board {b}", Q["linear_baselines"]["200000"]["MF"]["h19400"], q)
    W_ = Q["window_tradeoff"]
    fid("Linear baselines", f"LDA, 770 samples, board {b}", W_["770"]["h19400"], q)
    fid("Linear baselines", f"LDA, 576 samples, board {b}", W_["576"]["h19400"], q)
    pair("Linear baselines", f"LDA, 576 vs 640 samples, board {b}", W_["576"]["vs640"]["h19400"], q)
    pair("Linear baselines", f"LDA, 770 vs 640 samples, board {b}", W_["770"]["vs640"]["h19400"], q)

# ----------------------------------------------------------------------- drift
def drift(h, b):
    p = src(f"{D}drift_full/drift_{h}.json", "banner", f"{D}drift_full/{b}_drift.log")
    J = json.load(open(p)); ph = np.array([s["phase"] for s in J["samples"]])
    spread = abs(ph - ph.mean()).max()
    row("Drift, 3 h", f"Board {b}, phase mean, SD, largest distance from mean", f"{ph.mean():.2f} deg, {ph.std(ddof=1):.3f} deg, {spread:.2f} deg", [f"{spread:.2f}", f"{ph.std(ddof=1):.2f}"], p)
    s0, s1 = [], []
    for tag, out in (("start", s0), ("end", s1)):
        for f in J[f"score_{tag}"]:
            z = np.load(f.replace("\\", "/"), allow_pickle=True); out.append((z["idx"], z["probs"], z["y"]))
    i0, i1 = (np.concatenate([x[0] for x in s]) for s in (s0, s1))
    assert np.array_equal(i0, i1) and len(i0) == 6000
    a, c, y = (np.concatenate([x[1] for x in s0]) > .5).astype(int), (np.concatenate([x[1] for x in s1]) > .5).astype(int), np.concatenate([x[2] for x in s0]).astype(int)
    q = paired(c, a, y)
    row("Drift, 3 h", f"Board {b}, end minus start, 6,000 paired shots", f"{q['dF']:+.2f} points, {q['B_only']} vs {q['A_only']}, p = {pv(q['p_exact'])}", [f"{q['dF']:+.2f}"[1:], "6,000"], p)

# ----------------------------------------------------------------------- tProc path, safety, swap, phase
def tproc_box(b):
    q = src(f"{B2}{b}_tproc_box_v144.json", "banner"); J = json.load(open(q))
    for v in ("nobranch", "qick"):
        g_ = [x for x in J["points"] if x["variant"] == v and x["fresh"] == x["n"] == x["br_eq_dec"] == x["played_eq_br"]][0]
        row("tProc path", f"Board {b}, boxcar path, {v}, earliest clean margin", f"{g_['L_median']:.1f} ns at +{1e3 * g_['margin']:.0f} ns, {g_['fresh']}/{g_['n']} fresh", [f"{g_['L_median']:.1f}", f"{g_['L_median']:.0f}"], q)

def tproc_fir(b):
    q = src(f"{B2}{b}_tproc_fir_v144.json", "banner"); J = json.load(open(q))
    g_ = [x for x in J["points"] if abs(x["margin"] - 0.035) < 1e-6][0]
    row("tProc path", f"Board {b}, FIR path, same margin as the old L5 value", f"{g_['L_median']:.1f} ns", [f"{g_['L_median']:.1f}"], q)

def boxcar_mr(b):
    q = src(f"{B2}{b}_boxcar_mr_v144.json", "banner", f"{B2}{b}_boxcar_mr_v144.log"); J = json.load(open(q))
    row("Safety and tests", f"Boxcar vs raw buffer, board {b}, with banner", f"{J['shots_exact']}/{J['shots']} shots exact, {J['verdict']}", [], q)

def leak():
    """R.19 leaked energy over blanked shots, Gaussian conditional pulse (analysis_scripts/diagnostics/leak_energy_check.py)."""
    lk = {}
    for b in "AB":
        for thr in (8, 2000):
            q = src(f"{LK}{b}_v144_leak_thr{thr}.json", "banner", f"{LK}{b}_v144_leak_thr{thr}.log"); lk[b, thr] = (q, json.load(open(q)))
            assert lk[b, thr][1]["control_sees_leak"], (b, thr)
    u = {b: 100 * lk[b, 8][1]["max_upper_ci_lead_2_4"] for b in "AB"}
    row("Safety and tests", "Leaked energy, gate blanking every window, DELAY from the first nonzero sample (nz_thr 8): largest upper 95% bound at leads 2-4, A / B",
        f"{u['A']:.3f}% / {u['B']:.3f}%, {lk['A', 8][1]['shots']} shots per setting; positive control resolves a leak",
        [f"{max(u.values()):.2f}", str(lk["A", 8][1]["shots"])], lk["A", 8][0], "B's file is " + lk["B", 8][0])
    m = {b: 100 * lk[b, 2000][1]["leads"]["4"]["frac_mean"] for b in "AB"}
    row("Safety and tests", "Leaked energy with DELAY from the 2000-amplitude crossing (nz_thr 2000), lead 4 mean, A / B",
        f"{m['A']:.3f}% / {m['B']:.3f}%", [f"{m['A']:.2f}", f"{m['B']:.2f}"], lk["A", 2000][0], "B's file is " + lk["B", 2000][0])

def gate_saving():
    q = src(f"{B2}A_tproc_box_v144.json", "banner")
    La = [x for x in json.load(open(q))["points"] if x["variant"] == "nobranch" and x["fresh"] == x["n"]][0]["L_median"]
    L = loop_groups()["high"]["L_50"]
    row("tProc path", f"Gate saving over the tProc path: board A tProc minus L_50 ({L:.1f}), and its population worth at T1 = 32 us", f"{La - L:.0f} ns; {(La - L) / 960:.2f} to {(La - L) / 320:.2f} points", [f"{La - L:.0f}", f"{(La - L) / 960:.2f}", f"{(La - L) / 320:.2f}"], q, "t/(3 T1) and t/T1")

def _f(path):
    z = np.load(path, allow_pickle=True); y = z["y"].astype(int); d = (z["live_rec"][:, 7] & 1).astype(int)
    return fstats(d, y), float(np.degrees(np.angle(complex(z["k"]))))

def swap(t, lab):
    q = src(f"{B2}swap_{t}_v144.npz", "banner", f"{B2}swap_{t}.log"); fs, ang = _f(q)
    fid("Calibration swap", f"{lab}, 2,000 shots, arg(k) {ang:.1f} deg", fs, q)

def swap_phase():
    ka, kb = (_f(src(f"{B2}swap_{t}_v144.npz", "banner", f"{B2}swap_{t}.log"))[1] for t in ("A_own", "B_own"))
    row("Calibration swap", "Phase difference of the boards in that session", f"{abs(ka - kb):.1f} deg", [f"{abs(ka - kb):.0f}"], f"{B2}swap_B_own_v144.npz")

def phrst():
    q = src(f"{B2}A_phrst_probe_v144.json", "banner"); J = json.load(open(q))
    assert len(J["loads"]) == 4 and all(BANNER in l["banner"] for l in J["loads"])
    for m_ in ("none", "both"):
        row("Phase across loads", f"Board A, 4 loads, mode {m_}: mean per load", ", ".join(f"{float(np.degrees(np.angle(np.exp(1j * np.radians(l['modes'][m_]['shots'])).mean()))):.0f}" for l in J["loads"]) + " deg", [], q)

# ----------------------------------------------------------------------- offline analyses of the v1.44 data
def box_vs_fir():
    q = src(D + "analysis/box_vs_fir_v144.json", "banner", D + "axi/A_axi_0.log"); J = json.load(open(q))
    assert J["box_emu_eq_chip"] == J["n"] == 100000
    fid("Boxcar vs FIR", "Boxcar, 19,400, board A", J["subset19400"]["box"], q)
    fid("Boxcar vs FIR", "FIR model, 19,400, board A", J["subset19400"]["fir"], q)
    pair("Boxcar vs FIR", "Boxcar vs FIR, 19,400", J["subset19400"]["paired"], q)
    pair("Boxcar vs FIR", "Boxcar vs FIR, 99,400", J["all99400"]["paired"], q)

def synthetic():
    q = src(D + "analysis/r9/r9_stats_v144.json", "banner", D + "axi/A_axi_0.log"); J = json.load(open(q))["19400"]
    for c_ in J["comparisons"]:
        row("Synthetic data", f"{c_['A']} minus {c_['B']}", f"{c_['diff']:+.2f} [{c_['ci95'][0]:+.2f}, {c_['ci95'][1]:+.2f}]", [f"{c_['diff']:+.2f}"[1:], f"{c_['ci95'][0]:+.2f}"[1:], f"{c_['ci95'][1]:+.2f}"[1:]], q)

def nonlinear():
    q = src(D + "analysis/arch/arch_sweep2_src_v144.json", "banner", D + "axi/A_axi_0.log"); J = json.load(open(q))
    row("Nonlinear models", "LDA, 20,000 rows: raw / replayed", f"{J['lda_ref']['src20k']:.2f}% / {J['lda_ref']['loop20k']:.2f}%", [f"{J['lda_ref']['src20k']:.2f}", f"{J['lda_ref']['loop20k']:.2f}"], q)
    for k_ in ("seg_hgb", "seg_mlp", "bin_mlp", "cnn1d", "bin_gru"):
        v_ = J[k_]; t_ = v_["loop20k_vs_t4"]
        row("Nonlinear models", f"{k_}: raw / replayed, vs LDA", f"{v_['src20k']:.2f}% / {v_['loop20k']:.2f}%, {t_['delta']:+.2f}, p = {pv(t_['p'])}", [f"{v_['src20k']:.2f}", f"{v_['loop20k']:.2f}", f"{abs(t_['delta']):.2f}", pv(t_["p"])], q)

def channel():
    q = src(D + "analysis/channel_v144.json", "banner", D + "axi/A_axi_0.log"); J = json.load(open(q))
    row("Replay channel", "Class SNR per sample, source / replayed board A (rms mean separation over rms noise)", f"{J['snr_source']:.2f} / {J['snr_A']:.2f}", [f"{J['snr_source']:.2f}", f"{J['snr_A']:.2f}"], q)
    row("Replay channel", "Capture delay against the aligned source, 4,000 shots", f"{min(J['lags'], key=int)}-{max(J['lags'], key=int)} samples", [], q)

# ----------------------------------------------------------------------- Fig. 7
def fig7():
    F7 = D + "fig7/"
    fv = src(F7 + "fig7_capture_A_v144.json", "banner")
    src(F7 + "fig7_capture_B_v144.json", "banner"); src(F7 + "fig7_sweep_A_v144.json", "banner"); src(F7 + "fig7_sweep_B_v144.json", "banner")
    V = json.load(open(need(F7 + "fig7_values_v144.json")))
    row("Readout path", "Boxcar saves, board A", f"{V['saved_ns']:.1f} ns", [f"{V['saved_ns']:.1f}"], fv)
    im = dict(zip(V["image_dBc"]["M"], V["image_dBc"]["box_measured"]))
    _fir = V["saved_ns"] + 9 * 1e3 / 552.96 + 5 * 1e3 / 552.96
    row("Readout path", "QICK FIR path, mixer output to decimated edge: saving + boxcar path (9 cycles + group delay (M-1)/2 = 5 samples)", f"{_fir:.1f} ns", [f"{_fir:.0f}"], fv, "check: FIR core C_LATENCY 67 cycles + 120-tap group delay 59.5/4423.68 us = 134.6 ns")
    row("Readout path", "Image rejection at M = 11, board A", f"{im[11]:.1f} dB", [f"{abs(im[11]):.1f}"], fv)

# ----------------------------------------------------------------------- Table III
sys.path.insert(0, "paper_tables/table3")
import v141_reports as R3  # noqa: E402
R3.HIER, R3.FLAT, R3.TIMING = (Path(TR) / f"v1.44_{r}.rpt" for r in ("routed_util_hier", "routed_utilization", "routed_timing"))

def summary(path):
    """WNS, failing setup endpoints and total endpoints from the Design Timing Summary; requires timing met."""
    t = Path(path).read_text(errors="ignore")
    assert "All user specified timing constraints are met." in t, "v1.44 report does not state timing met"
    v = re.search(r"WNS\(ns\).*?\n\s*-+.*?\n\s*(-?[\d.]+)\s+(-?[\d.]+)\s+(\d+)\s+(\d+)", t, flags=re.S)
    return float(v[1]), int(v[3]), int(v[4]), t

def implementation():
    for r in ("routed_timing", "routed_util_hier", "routed_utilization"):
        src(f"{TR}v1.44_{r}.rpt", "build-file")
    tt, tu = f"{TR}v1.44_routed_timing.rpt", f"{TR}v1.44_routed_util_hier.rpt"
    blk = R3.blocks()
    live, ax = blk["NN_live_0"], blk["NN_axi_0"]
    wns, fails, _, t = summary(tt)
    assert fails == 0 and wns >= 0, (wns, fails)
    row("Implementation", "NN_live LUT / FF", f"{live['lut']:,} / {live['ff']:,}", [f"{live['lut']:,}", f"{live['ff']:,}"], tu)
    row("Implementation", "NN_axi LUT / FF", f"{ax['lut']:,} / {ax['ff']:,}", [f"{ax['lut']:,}", f"{ax['ff']:,}", f"{100 * ax['lut'] / 425280:.1f}"], tu)
    row("Implementation", "WNS / failing setup endpoints", f"{wns:.3f} ns / {fails}", [f"{abs(wns):.3f}", fails], tt)
    m = re.search(r"^clk_pl_0\s+(-?[\d.]+)", t, flags=re.M)
    row("Implementation", "NN_axi clock (clk_pl_0) WNS", f"{float(m[1]):+.3f} ns", [m[1], f"{float(m[1]):.2f}"], tt)
    row("Implementation", "NN_live LUT, rounded for prose", f"about {round(live['lut'], -1):,}", [f"{round(live['lut'], -1):,}"], tu)
    row("Implementation", "Failing endpoints: NN_live / its clock crossing / stock readouts 1 and 2", f"{fails} / {fails} / {fails}", [fails], tt, "zero failing setup endpoints in the whole design")

def scaling():
    f = "scaling_study/resource_scaling_v144.json"
    _rs = json.load(open(need(f)))
    tu = src(f"{TR}v1.44_routed_util_hier.rpt", "build-file")
    assert _rs["per_unit"]["classifier"]["lut"] == sum(R3.blocks()[n]["lut"] for n in ("NN_live_0", "boxcar_tap_0", "live_gate_0", "live_res_cdc_0", "live_res_cdc_1", "trig_sync_0"))
    _n = [v["n_max"] for k, v in _rs["n_max"].items() if "front end none" not in k]
    row("Implementation", "Classifier units that fit with a per-tone front end (scaling_study/resource_scaling.py, SCALING_OUT=resource_scaling_v144.json)", f"{min(_n)} to {max(_n)}", [min(_n), max(_n)], tu)

# ----------------------------------------------------------------------- loop latency and tests
def latency():
    G = loop_groups()
    for name in ("low", "high"):                 # every load in both groups: the banner read from the board in that load's own run
        for load in G[name]["loads"]:
            sub = "cold" if "_c" in load else "reloads"
            q = f"{D}latency/{sub}/{load}.json"
            assert BANNER in Path(need(q)).read_text(errors="ignore"), f"no v1.44 banner in {q}"
    g = src(D + "analysis/loop_groups_values_v144.json", "banner", f"{D}latency/reloads/{G['low']['loads'][0]}.json")
    for name in ("low", "high"):
        x = G[name]
        row("Loop latency", f"L_50, {name} load group ({x['n_loads']} loads)", f"{x['L_50']:.1f} ns, range {x['L_50_min']:.1f}-{x['L_50_max']:.1f}", [f"{x['L_50']:.1f}", f"{x['L_50_min']:.1f}", f"{x['L_50_max']:.1f}"], g)
        row("Loop latency", f"Segments, {name} group: S_fe, S_NN, S_CDC, S_gate, S_conv", ", ".join(f"{v:.1f}" for v in x["segs"].values()) + " ns", [f"{v:.1f}" for v in x["segs"].values()], g)
    _h = G["high"]["segs"]
    row("Loop latency", f"Our logic (S_NN + S_CDC + S_gate) / converters and cable (S_conv), rounded, {G['high']['L_50']:.0f} ns group", f"{_h['S_nn'] + _h['S_cdc'] + _h['S_gate']:.0f} / {_h['S_conv']:.0f} ns", [f"{_h['S_nn'] + _h['S_cdc'] + _h['S_gate']:.0f}", f"{_h['S_conv']:.0f}", f"{G['high']['L_50']:.0f}", f"{G['low']['L_50']:.0f}"], g)
    _sc = (G["low"]["segs"]["S_conv"] + _h["S_conv"]) / 2
    row("Loop latency", "Converters and cable, mean of the two load groups", f"{_sc:.1f} ns ({G['low']['segs']['S_conv']:.1f} and {_h['S_conv']:.1f})", [f"{_sc:.0f}"], g)

def fig8b():
    _b = src(D + "fig8b/A_gate_probe.log", "banner")
    src(D + "fig8b/B_gate_probe.log", "banner")
    _v = json.load(open(need(D + "fig8b/fig8b_values_v144.json")))
    _l = {b: [x for x in _v[b] if x["slack_ns"] == -66.0][0]["late"] for b in "AB"}
    row("Loop latency", "Fig. 8(b), late at -66 ns, boards A / B, of 40", f"{_l['A']} / {_l['B']}", [_l["A"], _l["B"]], _b)

def tests(b):
    for name, stem in (("Binding fault test", "binding_fault"), ("Fail-safe test", "fail_safe"), ("tProc decision test", "tproc_decision")):
        f = f"{D}tests/{b}_{stem}.json"; lg = f[:-5] + ".log"
        need(f)
        has = Path(lg).exists() and BANNER in Path(lg).read_text(encoding="utf-8", errors="ignore")
        p = src(f, "banner", lg) if has else src(f, "name+date")
        row("Safety and tests", f"{name}, board {b}", str(json.load(open(p)).get("verdict")), [], p)
    f = f"{D}tests/{b}_functional.json"
    p = src(f, "banner", f[:-5] + ".log"); j = json.load(open(p))["summary"]
    assert j["ERROR"] == 0, j
    row("Safety and tests", f"QICK functional suite, board {b}", f"{j['PASS']} pass, {j['WARN']} warning, {j['FAIL']} fail, {j['NOT WIRED']} not wired", [j["PASS"], j["WARN"], j["FAIL"]], p)

# ----------------------------------------------------------------------- run every group
assert sha(BIT) == BIT_SHA
for fn in (nn_live, replay_decisions, inloop, axi, timing_checks, emulator, leak, gate_saving, swap_phase, phrst, box_vs_fir, synthetic, nonlinear, channel, fig7, implementation, scaling, latency, fig8b):
    attempt(fn)
for b in "AB":
    for fn in (linear, tproc_box, tproc_fir, boxcar_mr, tests):
        attempt(fn, b)
for h, b in HOSTS:
    attempt(drift, h, b)
for t, lab in (("A_own", "Board A, own calibration"), ("B_own", "Board B, own calibration"), ("B_on_A_calibration", "Board B on board A's calibration")):
    attempt(swap, t, lab)

# ----------------------------------------------------------------------- the v1.43 ledger sets the rows and their order
def norm(k):
    """Row name without the numbers that a build measures, so the same row matches on both builds."""
    k = re.sub(r",? ?arg\(k\) -?[\d.]+ deg", "", k)
    k = re.sub(r" \(\d+ loads\)", "", k)
    k = re.sub(r" L_50 \([\d.]+\)", " L_50", k)
    k = re.sub(r", rounded, \d+ ns group", ", rounded", k)
    return re.sub(r"SCALING_OUT=resource_scaling_v14\d", "SCALING_OUT=resource_scaling", k)

def expected(s, key):
    """Where a pending row's source will be."""
    if key.startswith("Classifier units"):
        return "scaling_study/resource_scaling_v144.json"
    for a, b_ in ((r"^campaign_v140L1/A_v143_(\w+\.json)$", D + r"tests/A_\1"), (r"^campaign_v143/pre_6_3/(B_\w+\.json)$", D + r"tests/\1"),
                  (r"^campaign_v140L1/v143_fig8b/", D + "fig8b/"), (r"^paper_figures/fig7/", D + "fig7/"), (r"axi_emu\.json$", "axi_emu_v144.json"),
                  (r"v1\.43_", "v1.44_")):
        s = re.sub(a, b_, s)
    return s.replace("v143", "v144")

have = {(r["section"], norm(r["key"])): r for r in ROWS}
assert len(have) == len(ROWS), "two rows share a name"
template = json.load(open(TEMPLATE))["rows"]
names = {(r["section"], norm(r["key"])) for r in template}
final, pending = [], []
for t in template:
    n_ = norm(t["key"])
    if (t["section"], n_) in have:
        final.append(have[(t["section"], n_)])
    else:
        p = dict(section=t["section"], key=n_, value="PENDING", text=[], source=expected(t["source"], t["key"]), kind="pending", note="")
        final.append(p); pending.append(p)
extra = [r for r in ROWS if (r["section"], norm(r["key"])) not in names]
for r in extra:                                     # results v1.44 has and v1.43 did not, at the end of their section
    last = max(i for i, x in enumerate(final) if x["section"] == r["section"])
    final.insert(last + 1, r)

# ----------------------------------------------------------------------- output
order = list(dict.fromkeys(r["section"] for r in final))
done = len(final) - len(pending)
out = ["# v1.44 results ledger", "", "**Generated by `campaign_v144/final/analysis/make_v144_ledger.py`. Do not edit by hand.** Add a result by adding it to that script,",
       "which refuses any source that fails a build check. `--verify` fails if this file or a source has changed since it was generated, or while any result is pending.", "",
       f"Build: `{BIT.name}`, sha256 `{BIT_SHA[:16]}...`, banner `{BANNER} 2026`.", "",
       f"Status: **{done} of {len(final)} results filled, {len(pending)} pending.** A PENDING row has no value yet; its Source column names the file it is waiting for.", "",
       "Evidence: **banner** = the v1.44 banner read from the board during that run; **bracketed** = banner logged before and after by the same uninterrupted runner; "
       "**build-file** = the report saved from the v1.44 build; **name+date** = named v1.44 and dated after the build, no banner (weakest).", ""]
for sec in order:
    out += [f"## {sec}", "", "| Result | Value | Evidence | Source |", "|---|---|---|---|"]
    for r in final:
        if r["section"] == sec:
            out.append(f"| {r['key']} | {r['value']} | {r['kind']} | `{r['source']}` |")
    out.append("")
weak = sorted(p for p, v in SRC.items() if v["kind"] == "name+date")
out += ["## Sources with weak build evidence", "", "These carry no banner. Confirm them or re-measure before submission.", ""] + [f"- `{p}`" for p in weak] + [""]
text = "\n".join(out)
payload = dict(rows=final, sources=SRC, pending=len(pending), waiting_for=sorted(set(BLOCKED)), md_sha256=hashlib.sha256(text.encode()).hexdigest())
if "--verify" in sys.argv:
    old = json.load(open(JS))
    bad = [p for p, v in old["sources"].items() if sha(p)[:16] != v["sha256"]]
    if MD.read_text(encoding="utf-8") != text or old["md_sha256"] != payload["md_sha256"]:
        bad.append(str(MD))
    if pending:
        bad.append(f"{len(pending)} results still pending")
    print("VERIFY", "FAILED: " + ", ".join(bad) if bad else f"OK: {len(final)} results, {len(SRC)} sources")
    sys.exit(1 if bad else 0)
MD.write_text(text, encoding="utf-8")
JS.write_text(json.dumps(payload, indent=1, default=str))
print(f"wrote {MD}: {done} of {len(final)} filled, {len(pending)} pending, {len(extra)} new rows, {len(SRC)} sources, {len(weak)} weak")
print("new rows:", [r["key"] for r in extra])
print("waiting for:", *sorted(set(BLOCKED)), sep="\n  ")
