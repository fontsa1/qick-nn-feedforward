"""Fig. 8(b): the slack sweep, v1.44, both boards. The tProc schedules the conditional pulse at a slack relative
to the decision time; at each slack 40 replayed traces (17 with decision 1) are run with the gate enabled.
Per slack: decision-1 shots whose pulse came back (of 17), shots the gate blanked as late (of 40), and unsafe pulses
(a pulse with no fresh decision 1 for its own shot; 0 everywhere). Too early, every decision is late and every pulse
is blanked; two 2.03 ns steps later (-68 -> -64 ns), all-correct. Boards A (filled) and B (open) differ only at -66 ns, on the edge of the window.
The sweep points are not evenly spaced in ns, so the x axis is categorical (sweep points, not to scale).
Data: {A,B}_gate_probe_box.npz in this folder (gate_probe.py --split)."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans"],
                     "font.size": 9, "svg.fonttype": "none", "pdf.fonttype": 42,
                     "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
                     "xtick.major.size": 3, "ytick.major.size": 3})
BLUE, ORANGE, GREY, RED = "#1b4f8f", "#c2571a", "#555555", "#b2182b"

rows = {}
for b in "AB":
    d = np.load(HERE / f"{b}_gate_probe_box.npz", allow_pickle=True)
    sl = d["slacks"]
    r = []
    for i in range(len(sl)):
        dec, came = d[f"dec_{i}"] == 1, d[f"came_{i}"].astype(bool)
        r.append(dict(slack_ns=round(1e3 * float(sl[i]), 1), n=len(dec), n_dec1=int(dec.sum()),
                      played_dec1=int((came & dec).sum()), late=int((d[f"late_{i}"] > 0).sum()),
                      unsafe=int(d[f"unsafe_{i}"].sum()), played_dec0=int((came & ~dec).sum())))
    rows[b] = r

fig, ax = plt.subplots(figsize=(3.5, 2.3))
fig.subplots_adjust(left=0.15, right=0.97, bottom=0.21, top=0.95)
x = np.arange(len(rows["A"]))
for b, mfc in (("A", None), ("B", "white")):
    r = rows[b]
    off = -0.08 if b == "A" else 0.08
    ax.plot(x + off, [100 * q["played_dec1"] / q["n_dec1"] for q in r], "o-", color=BLUE, ms=4, lw=1.0,
            mfc=mfc or BLUE, label=f"pulse played, decision 1 ({b})")
    ax.plot(x + off, [100 * q["late"] / q["n"] for q in r], "s--", color=ORANGE, ms=3.5, lw=1.0,
            mfc=mfc or ORANGE, label=f"blanked as late ({b})")
    hx, = ax.plot(x + off, [100 * q["unsafe"] / q["n"] for q in r], "x", color=RED, ms=4, lw=0)
ax.set_xticks(x); ax.set_xticklabels([f"{q['slack_ns']:g}".replace("-", "−") for q in rows["A"]])
ax.set_xlabel("pulse slack relative to the decision (ns, not to scale)")
ax.set_ylabel("shots (%)"); ax.set_ylim(-5, 108)
h, l = ax.get_legend_handles_labels()
leg = ax.legend([h[0], h[1], hx], ["played (of 17)", "late (of 40)", "unsafe (none)"],
                fontsize=8, frameon=False, handlelength=1.0, borderaxespad=0,
                loc="upper right", bbox_to_anchor=(0.99, 0.90))
ax.text(0.98, 0.20, "board A filled, B open", transform=ax.transAxes, ha="right", va="center",
        fontsize=8, color=GREY)
fig.text(0.005, 0.985, "(b)", fontweight="bold", va="top")
for ext in ("pdf", "svg", "png"):
    fig.savefig(HERE / f"fig8b_slack_sweep.{ext}", dpi=300)
(HERE / "fig8b_values_v144.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
print(json.dumps(rows["A"]))
