"""Loop latency by load group, from latency_protocol.py runs (v1.44): 10 reloads + 4 cold starts, both boards.
L_50 falls in two groups set at bitstream load.
Values per group: mean over loads of each load's mean. Run alone to write loop_groups_values_v144.json."""
import glob
import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SEGS = ["S_fe", "S_nn", "S_cdc", "S_gate", "S_conv"]
PATH_MINUS_ADV = (1.4, 1.0)      # ns, see closure()
SPLIT_NS = 175.0                # between the groups: 173.96-174.38 and 175.64-176.35


def closure(p, r):
    """path_minus_adv is centred on 1.4 ns for v1.44 (boxcar path 113.3 ns minus the 111.9 ns advance), +/- 1.0 ns;
    latency_protocol.py's default centre, 3.2 ns, is v1.43's. Every other closure check must pass as written; see
    campaign_v144/final/latency/README.md."""
    bad = [c for c in r["checks"] if not c["ok"]]
    if [c["name"] for c in bad] not in ([], ["path_minus_adv"]):
        raise SystemExit(f"{p} failed closure checks: {[c['name'] for c in bad]}")
    d = next(c["detail"] for c in r["checks"] if c["name"] == "path_minus_adv")
    v = float(re.search(r"= (-?[\d.]+) ns", d)[1])
    if abs(v - PATH_MINUS_ADV[0]) > PATH_MINUS_ADV[1]:
        raise SystemExit(f"{p}: path - advance {v} outside the v1.44 window {PATH_MINUS_ADV}")


def loads():
    out = []
    for sub in ("reloads", "cold"):
        for p in sorted(glob.glob(str(ROOT / "campaign_v144" / "final" / "latency" / sub / "?_*.json"))):
            r = json.load(open(p))
            if "settings" not in r:
                continue
            closure(p, r)
            z = np.load(p[:-5] + ".npz", allow_pickle=False)
            cols = [str(c) for c in z["columns"]]
            segc = [str(c) for c in z["seg_columns"]]
            row, seg = z["row_op"], z["seg_op"]
            out.append(dict(board=Path(p).name[0], label=r["settings"]["label"], L_50=r["points"]["op"]["L_50"]["mean"],
                            segs={k: r["points"]["op"]["segments"][k] for k in SEGS},
                            counter=r["points"]["op"]["counter"]["mean"], n=r["points"]["op"]["n_pulse"],
                            shot_L50=row[:, cols.index("L_50")], shot_segs=seg[:, [segc.index(k) for k in SEGS]]))
    return out


def groups():
    ls = loads()
    res = {}
    for name, sel in (("low", [x for x in ls if x["L_50"] < SPLIT_NS]), ("high", [x for x in ls if x["L_50"] >= SPLIT_NS])):
        res[name] = dict(n_loads=len(sel), n_A=sum(x["board"] == "A" for x in sel), n_B=sum(x["board"] == "B" for x in sel),
                         n_shots=sum(x["n"] for x in sel), L_50=float(np.mean([x["L_50"] for x in sel])),
                         L_50_min=min(x["L_50"] for x in sel), L_50_max=max(x["L_50"] for x in sel),
                         counter=float(np.mean([x["counter"] for x in sel])),
                         segs={k: float(np.mean([x["segs"][k] for x in sel])) for k in SEGS},
                         loads=[f"{x['board']}_{x['label']}" for x in sel],
                         shot_L50=np.concatenate([x["shot_L50"] for x in sel]),
                         shot_segs=np.concatenate([x["shot_segs"] for x in sel]))
    return res


if __name__ == "__main__":
    g = groups()
    keep = {k: {a: b for a, b in v.items() if not a.startswith("shot_")} for k, v in g.items()}
    keep["step_ns"] = keep["high"]["L_50"] - keep["low"]["L_50"]
    keep["logic_ns"] = {k: keep[k]["segs"]["S_nn"] + keep[k]["segs"]["S_cdc"] + keep[k]["segs"]["S_gate"] for k in ("low", "high")}
    json.dump(keep, open(HERE / "loop_groups_values_v144.json", "w"), indent=1)
    print(json.dumps(keep, indent=1))
