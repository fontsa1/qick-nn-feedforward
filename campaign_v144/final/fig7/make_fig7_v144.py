"""Fig. 7: the low-latency readout path (boxcar_tap) vs QICK's readout FIR, measured on board A (v1.44).
(a) Rising edge of a 50 MHz const pulse through both paths on the same shots: mean |z| of 20 shots, each path
    normalised to its plateau. Buffer 1 (boxcar) is triggered through trig_sync, 3 cycles after buffer 0 (FIR), so
    its samples sit at (n + 3)/fs on the shared time axis; t = 0 is the FIR's 50% crossing.
(b) Magnitude response vs frequency offset from the readout tone (the usual way IEEE papers show a filter's
    rejection): theory curves for box-8 alone (M = 1), the deployed box-8 + 11-sample sum and QICK's FIR (from its
    120 coefficients, qick_readout_fir.coe). Markers: measured response from the frequency sweep
    (fig7_sweep_{A,B}_v144.npz, box_sweep_fig7.py): readout tone f, image/DC at 2f after each path's filter,
    board A plotted, board B compared in fig7_values_v144.json; ring = the 100 MHz value of the original capture.
Supplementary (fig7s_image_vs_M, for an appendix): image level vs M = 1..16, measured vs theory.
Data: fig7_capture_A_v144.npz (analysis_scripts/diagnostics/box_probe_fig7.py, 2026-11-06).
Arial 9 pt; script, data and outputs live in this folder."""
import json
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans"],
                     "font.size": 9, "svg.fonttype": "none", "pdf.fonttype": 42, "axes.linewidth": 0.8,
                     "xtick.major.width": 0.8, "ytick.major.width": 0.8, "xtick.major.size": 3,
                     "ytick.major.size": 3, "legend.frameon": False})
STOCK = "#6b6b6b"; OURS = "#1b4f8f"; LIGHT = "#b9d3f0"

FS_IN, FS = 4423.68, 552.96            # MS/s: ADC samples into the readout, decimated stream
DT = 1e3 / FS                           # ns per decimated sample
F_IMG = 100.0                           # MHz: 2f image of the 50 MHz readout
TRIG_OFFSET = 3                         # cycles: trig_sync delays buffer 1's trigger vs buffer 0's

d = np.load(HERE / "fig7_capture_A_v144.npz")


# ---------------------------------------------------------------- analysis
def edge(z):
    """50% crossing of |z| on the rising edge, in samples (as box_probe.py), and the plateau."""
    a = np.abs(z)
    hi = np.median(a[np.argmax(a > 0.5 * a.max()):][:50])
    k = int(np.argmax(a > 0.5 * hi))
    return k - 1 + (0.5 * hi - a[k - 1]) / (a[k] - a[k - 1]), hi


def image_dbc(z):
    """Image level re. baseband: least-squares fit of DC + exp(j 2 pi f t), f scanned near -100 MHz.
    No window, so no scalloping bias (the old Hann-peak estimate read ~0.6 dB low; audit D12)."""
    z = z[200:-300]
    t = np.arange(len(z)) / FS
    best = None
    for f in np.linspace(-F_IMG - 1, -F_IMG + 1, 401):
        X = np.stack([np.ones_like(t), np.exp(2j * np.pi * f * t)], 1)
        co = np.linalg.lstsq(X, z, rcond=None)[0]
        r = np.sum(np.abs(z - X @ co) ** 2)
        if best is None or r < best[0]:
            best = (r, co)
    return 20 * np.log10(abs(best[1][1]) / abs(best[1][0]))


def resp(h, f, rate):
    n = np.arange(len(h))
    return abs(np.sum(h * np.exp(-2j * np.pi * f / rate * n))) / abs(np.sum(h))


coe = np.array([int(x) for x in re.findall(r"-?\d+", (HERE.parents[2] / "paper_figures" / "fig7" / "qick_readout_fir.coe").read_text().split("=")[2])], float)
fir_theory = 20 * np.log10(resp(coe, F_IMG, FS_IN))
Ms = np.arange(1, 17)
box_theory = np.array([20 * np.log10(resp(np.ones(8), F_IMG, FS_IN) * resp(np.ones(m), F_IMG, FS)) for m in Ms])
box_meas = np.array([np.mean([image_dbc(z) for z in d[f"B{m}_box"]]) for m in Ms])
fir_meas = np.array([np.mean([image_dbc(z) for z in d[f"B{m}_fir"]]) for m in Ms])

e_fir = np.array([edge(z)[0] for z in d["A_fir"]])
e_box = np.array([edge(z)[0] for z in d["A_box"]])
raw = float(np.median(e_fir - e_box))
saved_ns = (raw - TRIG_OFFSET) * DT

summary = dict(edge_fir_samples=float(np.median(e_fir)), edge_box_samples=float(np.median(e_box)),
               raw_diff_samples=raw, raw_diff_sd=float(np.std(e_fir - e_box)), trig_offset_cycles=TRIG_OFFSET,
               saved_ns=round(saved_ns, 2),
               fabric_timestamp_saved_cycles=float(np.median(d["D_sel0"]) - np.median(d["D_sel1"])),
               image_dBc=dict(M=Ms.tolist(), box_measured=np.round(box_meas, 2).tolist(),
                              box_theory=np.round(box_theory, 2).tolist(),
                              fir_measured=round(float(np.mean(fir_meas)), 2), fir_theory=round(fir_theory, 2)))


# measured frequency response (box_sweep_fig7.py): readout tone f, image at 2f; image/DC after each path's filter
def sweep_db(a, f):
    """LS fit of DC + the image at -2f (+ the ADC offset, which lands at -f) on the plateau; image re. DC in dB."""
    z = (a[:, 0] + 1j * a[:, 1]).astype(np.complex128)[200:-300]
    t = np.arange(len(z)) / FS
    X = np.stack([np.ones_like(t), np.exp(-2j * np.pi * 2 * f * t), np.exp(-2j * np.pi * f * t)], 1)
    co = np.linalg.lstsq(X, z, rcond=None)[0]
    return 20 * np.log10(abs(co[1]) / abs(co[0]))


sw = {}
for b in "AB":
    s = np.load(HERE / f"fig7_sweep_{b}_v144.npz")
    fl = s["f_MHz"]
    sw[b] = {k: np.array([np.mean([sweep_db(x, f) for x in s[f"f{i}_{key}"]]) for i, f in enumerate(fl)])
             for k, key in (("fir", "M11_fir"), ("m1", "M1_box"), ("m11", "M11_box"))}
sw_f2 = 2 * fl
sw_th = dict(fir=np.array([20 * np.log10(resp(coe, x, FS_IN)) for x in sw_f2]),
             m1=np.array([20 * np.log10(resp(np.ones(8), x, FS_IN)) for x in sw_f2]))
sw_th["m11"] = sw_th["m1"] + np.array([20 * np.log10(max(resp(np.ones(11), x, FS), 1e-9)) for x in sw_f2])
summary["sweep"] = dict(image_MHz=np.round(sw_f2, 3).tolist(), n_freq=len(sw_f2), shots_per_point=3)
for k in ("fir", "m1", "m11"):
    ok = sw_th[k] > -30
    summary["sweep"][k] = dict(
        A_dB=np.round(sw["A"][k], 2).tolist(), B_dB=np.round(sw["B"][k], 2).tolist(),
        theory_dB=np.round(np.maximum(sw_th[k], -200), 2).tolist(),
        max_abs_meas_minus_theory_where_theory_above_m30dB=dict(
            A=round(float(np.abs(sw["A"][k] - sw_th[k])[ok].max()), 3),
            B=round(float(np.abs(sw["B"][k] - sw_th[k])[ok].max()), 3)),
        max_abs_A_minus_B_where_theory_above_m30dB=round(float(np.abs(sw["A"][k] - sw["B"][k])[ok].max()), 3))
summary["sweep"]["m11_null_floor_dB"] = dict(A=round(float(sw["A"]["m11"].min()), 1),
                                             B=round(float(sw["B"]["m11"].min()), 1))
(HERE / "fig7_values_v144.json").write_text(json.dumps(summary, indent=1))

# ---------------------------------------------------------------- figure
# single column (3.5 in): (a) above (b)
fig = plt.figure(figsize=(3.5, 4.6))
ax = fig.add_axes([0.165, 0.605, 0.80, 0.375])
bx = fig.add_axes([0.165, 0.095, 0.80, 0.375])

# (a) edges on a shared time axis, t = 0 at the FIR's 50% crossing
# bold: |complex mean of 20 shots| (the image's phase varies shot to shot, so it averages out);
# faint: one single shot, |z| (on the FIR path the unattenuated image beats against the signal)
mf = np.abs(d["A_fir"].mean(0)); mb = np.abs(d["A_box"].mean(0))
sf = np.abs(d["A_fir"][0]); sb = np.abs(d["A_box"][0])
pf = edge(d["A_fir"].mean(0))[1]; pb = edge(d["A_box"].mean(0))[1]
nf = np.arange(len(mf)); nb = np.arange(len(mb)) + TRIG_OFFSET
t0 = np.median(e_fir) * DT
tf = nf * DT - t0; tb = nb * DT - t0
sel_f = (tf > -175) & (tf < 70); sel_b = (tb > -175) & (tb < 70)
ax.plot(tf[sel_f], sf[sel_f] / pf, color=STOCK, lw=0.6, alpha=0.45, zorder=1)
ax.plot(tb[sel_b], sb[sel_b] / pb, color=OURS, lw=0.6, alpha=0.45, zorder=1)
ax.plot(tf[sel_f], mf[sel_f] / pf, color=STOCK, lw=1.0, marker="o", ms=2.2, mfc="white", mew=0.7, zorder=3)
ax.plot(tb[sel_b], mb[sel_b] / pb, color=OURS, lw=1.2, marker="o", ms=2.2, mew=0.7, zorder=3)
ax.text(-170, 2.08, "faint: one shot\nbold: mean of 20 shots", ha="left", va="top", color="#333333",
        linespacing=1.1)
tb50 = t0 - saved_ns
ax.annotate("", xy=(-saved_ns, 0.5), xytext=(0, 0.5),
            arrowprops=dict(arrowstyle="<->", color="black", lw=0.8, shrinkA=0, shrinkB=0))
ax.text(-saved_ns / 2, 0.535, f"{saved_ns:.1f} ns", ha="center", va="bottom")
ax.text(-170, 1.12, "boxcar decimation path\n(box-8 + 11-sample moving sum)", ha="left", va="bottom", color=OURS,
        linespacing=1.1)
ax.text(-7, 0.24, "QICK\nreadout FIR", ha="right", va="center", color=STOCK, linespacing=1.1)
ax.set_xlim(-175, 70); ax.set_ylim(-0.04, 2.15)
ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0])
ax.set_xlabel("time relative to the FIR edge (ns)")
ax.set_ylabel("magnitude (plateau = 1)")
ax.spines[["top", "right"]].set_visible(False)
ax.text(-0.20, 1.02, "(a)", transform=ax.transAxes, fontweight="bold", va="top")

# (b) magnitude response vs offset from the readout tone, measured image level at 100 MHz
f = np.linspace(0, 200, 2000)
def db(h, rate):
    return 20 * np.log10(np.maximum([resp(h, x, rate) for x in f], 1e-6))
h_b8 = db(np.ones(8), FS_IN)
h_box = h_b8 + db(np.ones(11), FS)
h_fir = db(coe, FS_IN)
bx.plot([F_IMG, F_IMG], [-80, 5], color="#999999", lw=0.6, ls=":", zorder=0)
bx.text(F_IMG + 3, -79, "2f image", ha="left", va="bottom", color="#555555")
l_fir, = bx.plot(f, h_fir, color=STOCK, lw=1.0, zorder=2)
l_b8, = bx.plot(f, h_b8, color=OURS, lw=0.8, ls="--", zorder=2)
l_box, = bx.plot(f, h_box, color=OURS, lw=1.2, zorder=3)
# measured (board A; board B agrees, see fig7_values_v144.json): every other grid point for the two flat curves,
# interleaved so their markers do not overlap; every point for M = 11 (incl. the points around its nulls)
grid = np.isclose((fl - 2.5) % 1.0, 0) | np.isclose((fl - 2.5) % 1.0, 1.0)
idx = np.flatnonzero(grid)
bx.plot(sw_f2[idx[0::4]], sw["A"]["fir"][idx[0::4]], ls="none", marker="s", ms=2.6, mfc="white", mec=STOCK,
        mew=0.7, zorder=5)
bx.plot(sw_f2[idx[2::4]], sw["A"]["m1"][idx[2::4]], ls="none", marker="o", ms=2.6, mfc="white", mec=OURS,
        mew=0.7, zorder=5)
bx.plot(sw_f2, sw["A"]["m11"], ls="none", marker="o", ms=2.6, mfc=OURS, mec=OURS, mew=0.0, zorder=5)
bx.plot([F_IMG], [box_meas[10]], ls="none", marker="o", ms=5.5, mfc="none", mec=OURS, mew=0.9, zorder=6)
bx.text(F_IMG - 6, box_meas[10] - 1, f"−{abs(box_meas[10]):.1f} dB", ha="right", va="center", color=OURS)
h_m, = bx.plot([], [], ls="none", marker="o", ms=3, mfc="#777777", mec="#777777")
bx.legend([l_fir, l_b8, l_box, h_m],
          ["QICK FIR", "box-8 only (M = 1)", "M = 11, deployed", "measured"],
          loc="upper left", bbox_to_anchor=(0.0, 1.03), ncol=2, handlelength=1.4, columnspacing=0.8,
          handletextpad=0.5, borderaxespad=0.2, fontsize=8)
bx.set_xlim(0, 200); bx.set_ylim(-80, 30)
bx.set_yticks([0, -20, -40, -60, -80])
bx.spines["left"].set_bounds(-80, 5)
bx.set_xticks([0, 50, 100, 150, 200])
bx.set_xlabel("frequency offset from the readout tone (MHz)")
bx.set_ylabel("magnitude response (dB)")
bx.spines[["top", "right"]].set_visible(False)
bx.text(-0.20, 1.02, "(b)", transform=bx.transAxes, fontweight="bold", va="top")

for ext in ("pdf", "svg", "png"):
    fig.savefig(HERE / f"fig7_boxcar_path.{ext}", dpi=600 if ext == "png" else None)

# supplementary: image level vs M (appendix)
fig = plt.figure(figsize=(3.5, 2.45))
bx = fig.add_axes([0.17, 0.19, 0.80, 0.76])
bx.axhline(fir_theory, color=STOCK, lw=0.8, ls="--", zorder=1)
bx.plot(Ms, fir_meas, ls="none", marker="s", ms=3.2, mfc="white", mec=STOCK, mew=0.8, zorder=3)
bx.text(16.4, fir_theory - 2.2, f"QICK readout FIR: −{abs(fir_theory):.2f} dB", ha="right", va="top", color=STOCK)
bx.plot(Ms, box_theory, color=OURS, lw=0.8, ls="--", zorder=2)
bx.plot(Ms, box_meas, ls="none", marker="o", ms=3.4, mfc=OURS, mec=OURS, zorder=3)
i11 = 10
bx.plot([11], [box_meas[i11]], ls="none", marker="o", ms=6.5, mfc="none", mec=OURS, mew=1.0, zorder=4)
bx.annotate(f"M = 11 (deployed): −{abs(box_meas[i11]):.1f} dB\nnull at 2f$_s$/11 = 100.5 MHz",
            xy=(11, box_meas[i11]), xytext=(10.2, -37.5), ha="right", va="center", color=OURS, linespacing=1.15,
            arrowprops=dict(arrowstyle="-", color=OURS, lw=0.6, shrinkA=2, shrinkB=4))
bx.annotate("box-8 alone (M = 1)", xy=(1, box_meas[0]), xytext=(1.7, 3.4), ha="left", va="center", color=OURS,
            arrowprops=dict(arrowstyle="-", color=OURS, lw=0.6, shrinkA=1, shrinkB=3))
h1, = bx.plot([], [], ls="none", marker="o", ms=3.4, color="black", mfc="black")
h2, = bx.plot([], [], color="black", lw=0.8, ls="--")
bx.legend([h1, h2], ["measured", "theory"], loc="lower right", bbox_to_anchor=(1.0, 0.16), handlelength=1.6, borderaxespad=0.2)
bx.set_xlim(0.3, 16.7); bx.set_ylim(-50, 6)
bx.set_xticks([1, 4, 8, 11, 16])
bx.set_xlabel("moving-sum length M (samples)")
bx.set_ylabel("2f image relative to baseband (dB)")
bx.spines[["top", "right"]].set_visible(False)
for ext in ("pdf", "svg", "png"):
    fig.savefig(HERE / f"fig7s_image_vs_M.{ext}", dpi=600 if ext == "png" else None)
print(json.dumps({k: summary[k] for k in ("saved_ns", "raw_diff_samples", "fabric_timestamp_saved_cycles")}))
print("image M=1 / 11:", summary["image_dBc"]["box_measured"][0], summary["image_dBc"]["box_measured"][10],
      "| FIR", summary["image_dBc"]["fir_measured"], "theory", summary["image_dBc"]["fir_theory"])
