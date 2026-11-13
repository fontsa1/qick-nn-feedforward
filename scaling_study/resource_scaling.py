"""Resource scaling of the classifier unit.

Per-unit cost from a routed hierarchical utilization report (xczu48dr-ffvg1517-2-e), and
the largest number of units that fits in the resources the build leaves free, for three per-tone front ends:
  none      -- the unit only (all units share an existing readout stream; the lower bound)
  fir_free  -- + a DDS/mixer front end: axis_dyn_readout_v1_1 as routed (its FIR output m1 is unconnected, so the
               FIR is trimmed): what the boxcar path needs per tone
  stock     -- + a full stock readout (axis_dyn_readout_v1_0, FIR included), if every tone keeps QICK's FIR path
and two baselines: the design as built, and without NN_axi and the time tagger (a scaling build would drop them).
Writes resource_scaling.json (or $SCALING_OUT) and prints a table. Needs paper_tables/table3/v141_reports.py and the
routed reports it reads (not in this repository).
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "paper_tables" / "table3"))
import v141_reports as R                                              # noqa: E402

B = R.blocks()
T = R.totals()
CAP = dict(lut=425280, ff=850560, bram=1080.0, dsp=4272)               # xczu48dr (flat report "Available")
USED = dict(lut=T["lut"]["used"], ff=T["ff"]["used"], bram=T["bram"]["used"], dsp=T.get("dsp", {}).get("used", 542))


def cost(names):
    c = dict(lut=0, ff=0, bram=0.0, dsp=0)
    for n in names:
        b = B[n]
        c["lut"] += b["lut"]; c["ff"] += b["ff"]; c["dsp"] += b["dsp"]
        c["bram"] += b["ramb36"] + 0.5 * b["ramb18"]
    return c


def add(*cs):
    return {k: sum(c[k] for c in cs) for k in cs[0]}


core = cost(["NN_live_0", "boxcar_tap_0", "live_gate_0", "live_res_cdc_0", "live_res_cdc_1", "trig_sync_0"])
gate = cost(["dac_gate_0"])
record = cost([n for n in ("axi_bram_ctrl_live", "axi_bram_ctrl_live_bram", "live_rec_wreg_0") if n in B])  # live_rec_wreg_0 from v1.43
fe = dict(none=dict(lut=0, ff=0, bram=0.0, dsp=0),
          fir_free=cost(["axis_dyn_readout_v1_1"]),
          stock=cost(["axis_dyn_readout_v1_0"]))
dropped = cost([n for n in ("NN_axi_0", "qick_time_tagger_0", "axi_dma_tt", "axis_register_slice_5") if n in B])  # the time tagger is absent from v1.43

free = {"as built": {k: CAP[k] - USED[k] for k in CAP},
        "minus NN_axi (and time tagger if present)": {k: CAP[k] - USED[k] + dropped[k] for k in CAP}}

out = dict(per_unit=dict(classifier=core, dac_gate=gate, record_memory=record, front_end=fe),
           device_capacity=CAP, v141_used=USED, free=free, n_max={})
print(f"classifier (NN_live+boxcar+gate+trig_sync+2 CDC): {core}")
print(f"dac_gate: {gate}   record memory (verification): {record}")
for k, v in fe.items():
    print(f"front end {k}: {v}")
print()
for base, fr in free.items():
    for fname, f in fe.items():
        for rec in (True, False):
            u = add(core, gate, f, record) if rec else add(core, gate, f)
            lim = {k: (int(fr[k] // u[k]) if u[k] else None) for k in u}
            n = min(v for v in lim.values() if v is not None)
            bind = [k for k, v in lim.items() if v == n]
            key = f"{base} | front end {fname} | record {'yes' if rec else 'no'}"
            out["n_max"][key] = dict(unit=u, limits=lim, n_max=n, binding=bind)
            print(f"{key:80s} N_max {n:4d} (binding {','.join(bind)}; limits {lim})")
import os  # noqa: E402
(HERE / os.environ.get("SCALING_OUT", "resource_scaling.json")).write_text(json.dumps(out, indent=1))
