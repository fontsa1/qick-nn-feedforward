"""Writes paper_notes/v144_exempt.tsv: every number in the draft that is correct without coming from a v1.44 measurement, with why.
Written per paragraph, so an exemption cannot quietly cover the same number somewhere else. Numbers not listed stay UNCLASSED in
check_paper_v144.py. Class 'open' means it needs a person: it is listed so that it is visible, and the checker reports it separately.

"""
import csv
from pathlib import Path

DESIGN = "design constant or clock, not a measurement"
DATA = "property of the public dataset or source device, measured offline or cited"
OFF = "offline result on raw traces, no hardware involved"
CALC = "arithmetic from design constants"
ROUND = "rounded form of a ledger value"
SIZE = "sample size, row index or training-set size"
TEST = "statistical convention"
L = {
 "P1": {"176": ROUND}, "P3": {"96.01": OFF, "96.00": OFF, "96": "value reported by [16]", "16.3": DESIGN + ": 9 cycles, confirmed by S_NN and RTL sim", "176": ROUND, "1,340": ROUND},
 "P10": {"96": "value reported by [16]", "32.25": "value reported by [16]", "16": "value reported by [16]"}, "P11": {"32.25": "value reported by [16]"}, "P17": {"176": ROUND},
 "P24": {"0.2": DATA, "6.32": DATA, "1.5": DATA, "0.4": DATA, "32": DATA, "2.5": DATA, "770": DATA, "3.25": DATA, "4.50": DATA},
 "P39": {"16.3": DESIGN, "552.96": DESIGN}, "P40": {"1.808": DESIGN, "10": "the 10-90% definition", "90": "the 10-90% definition", "17": CALC + ": moving-sum rise"},
 "P49": {"96.06": "value reported by [41]"}, "P50": {"31": "value reported by [47]", "52": "value reported by [47]"},
 "P60": {"3.0": "PYNQ version (3.0.1) read from both boards 2026-11-07: campaign_v144/final/board_versions/board_A.json, board_B.json", "0.2": "QICK host software version (0.2.414), as installed", "7864.32": DESIGN, "4423.68": DESIGN, "552.96": DESIGN, "1.808": DESIGN},
 "P61": {"770": DATA, "11,000": DESIGN, "99": DESIGN, "3.25": DATA, "2.5": DATA, "1.808": DESIGN, "1.39": CALC}, "P62": {"500": DESIGN + ": calibration shots"},
 "P67": {"770": DATA, "95": TEST}, "P68": {"20": "qualitative band edge", "60": "qualitative band edge"},
 "P71": {"−56": "phase of the swap session, in the ledger as arg(k)", "−97": "phase of the swap session, in the ledger as arg(k)", "20": "phrst shots per program", "4,000": SIZE},
 "P76": {"900,000": DATA, "0.2": DATA, "1.808": DESIGN, "1.80": CALC, "3.87": OFF}, "P78": {"3.25": DATA, "865": DATA, "0.2": DATA, "3.80": OFF, "3.93": OFF},
 "P79": {"9,999": SIZE, "110,000": SIZE, "299,999": SIZE, "20": "validation split", "10,000": SIZE, "109,999": SIZE, "1,000": SIZE, "1,599": SIZE},
 "P80": {"576": DESIGN, "640": "window length compared"}, "P84": {"+0.32": OFF, "1.4": OFF, "2.0": OFF, "1,540": DATA, "4.0": OFF, "8.2": OFF, "3.80": OFF, "3.93": OFF, "3.86": OFF},
 "P85": {"10,000": SIZE, "50,000": SIZE, "30,000": SIZE, "70": ROUND + ": 0.36 of 0.49"}, "P86": {"+2.73": OFF, "−2.09": OFF, "1.65": OFF, "500": OFF},
 "P88": {"10,000": SIZE, "500": OFF, "770,": DATA, "770": DATA, "95": TEST, "50,000": SIZE, "96.05": OFF},
 "P92": {"770": DESIGN, "576": DESIGN}, "P93": {"±0.2": "seed spread at 10,000 rows, from the retrained networks", "10,000": SIZE, "10": "training hyperparameter: learning rate 10^-3", "32,": "training hyperparameter: batch size"},
 "P94": {"16,6": DESIGN, "32,20": DESIGN, "48,32": DESIGN, "40,8": DESIGN, "32,16": DESIGN, "18,8": DESIGN},
 "P95": {"576": DESIGN, "770": DESIGN, "2.5": DATA, "351": CALC + ": 194 samples at 1.808 ns", "1.87": CALC},
 "P97": {"95": TEST, "3.25": DATA, "576": DESIGN, "0.10": ROUND, "640": "window length compared", "100": "unit of the axis", "32": DATA, "544": "read off Fig. 5(b), ledger"},
 "P99": {"96.00": OFF, "100,": SIZE, "100": SIZE, "500": SIZE, "95": TEST, "95.78": OFF, "95.80": "value reported by [16]", "96.04": "value reported by [16]"},
 "P106": {"770": DESIGN, "1,540": DESIGN, "15.5": CALC + ": 1,540 cycles at 100 MHz plus inference", "15.4": CALC + ": 1,540 cycles at 100 MHz", "16.3": DESIGN, "14": "NN_axi DSP count, ledger table", "95.5": ROUND},
 "P107": {"770": DESIGN, "1,540": "NN_axi 1.41 core as built into v1.44: neural_net4_tlast/firmware defines.h, parameters.h, NN_axi.cpp", "2": "NN_axi 1.41 core as built into v1.44: neural_net4_tlast/firmware defines.h, parameters.h, NN_axi.cpp", "32,20": "NN_axi 1.41 core as built into v1.44: neural_net4_tlast/firmware defines.h, parameters.h, NN_axi.cpp", "48,32": "NN_axi 1.41 core as built into v1.44: neural_net4_tlast/firmware defines.h, parameters.h, NN_axi.cpp", "40,8": "NN_axi 1.41 core as built into v1.44: neural_net4_tlast/firmware defines.h, parameters.h, NN_axi.cpp", "18,8": "NN_axi 1.41 core as built into v1.44: neural_net4_tlast/firmware defines.h, parameters.h, NN_axi.cpp", "32": "NN_axi 1.41 core as built into v1.44: neural_net4_tlast/firmware defines.h, parameters.h, NN_axi.cpp", "16,384": "NN_axi 1.41 core as built into v1.44: neural_net4_tlast/firmware defines.h, parameters.h, NN_axi.cpp", "2023.1": "tool version", "10": DESIGN + ": HLS clock target", "15.5": CALC, "16.3": DESIGN},
 "P109": {"99.6": CALC, "1,152": DESIGN}, "P111": {"−1,0,": DESIGN}, "P120": {"16.3": DESIGN},
 "P122": {"576": DESIGN, "63": DESIGN + ": bit index and 31 cycles", "16": DESIGN + ": bit index", "15": DESIGN + ": bit index", "552.96": DESIGN, "491.52": DESIGN, "31": DESIGN, "16.3": DESIGN},
 "P124": {"576": DESIGN}, "P127": {"640": "equation layout artefact"}, "P130": {"0.5": DESIGN}, "P131": {"15": DESIGN, "16": DESIGN, "63": DESIGN, "512": DESIGN},
 "P133": {"176": ROUND, "25": CALC + ": 16.3 + 9.0 ns, both stated in Sec. VIII"}, "P135": {"11": DESIGN}, "P142": {"4423.68": DESIGN, "552.96": DESIGN, "16": DESIGN, "100.5": CALC, "11,": DESIGN, "−100": DESIGN, "16.3": DESIGN, "9.0": CALC},
 "P144": {"20": "shots in the capture", "−100": DESIGN, "11": DESIGN, "100": DESIGN}, "P145": {"16.3": DESIGN, "0.10": ROUND}, "P149": {"15.5": CALC, "552.96": DESIGN},
 "P154": {"160": ROUND + ": 337 minus 176", "176": ROUND}, "P155": {"576": DESIGN, "16.3": DESIGN, "63": DESIGN, "29": DESIGN, "16": DESIGN, "15": DESIGN, "491.52": DESIGN, "200": DESIGN, "256": DESIGN, "245.76": "QICK design constant: clk_core in firmware/projects/qick_tprocv2_4x2_standard/bd_2023-1.tcl at commit 779dffa", "204.8": "QICK design constant: clk_core in the same file on QICK main since 2026-09-16 (commit a24cc8e, merged in #432)", "552.96": DESIGN, "0.42": "QICK tProc IP alone, out-of-context place and route at c_clk 5.0 ns; a property of QICK's IP, not of a build of ours"},
 "P156": {"95": TEST}, "P82": {"23": "tools/verify_no_leakage.py re-run 2026-11-06, 23/23: campaign_v144/final/analysis/verify_no_leakage.log"},
 "P162": {"2.03": DESIGN + ": tProc tick", "176": ROUND, "1.4": "v1.44 latency study: campaign_v144/final/latency", "0.6": "v1.44 latency study, same entry", "0.2": "v1.44 latency study, same entry"}, "P166": {"10": "equation number"},
 "P171": {"100": "shots per load", "40": "traces per slack", "17": "traces with decision 1", "−66": "slack value"},
 "P172": {"1039.9": CALC, "575": CALC, "1.216": CALC, "1.22": CALC, "1872": CALC, "176": ROUND, "91": CALC, "2.05": CALC}, "P177": {"1.808": DESIGN, "3.25": DATA},
 "P179": {"16": CALC + ": 16 DAC samples per 491.52 MHz cycle", "491": DESIGN},
 "P181": {"552.96": DESIGN, "100": DESIGN, "0.3": "detectable size at this n"}, "P183": {"576": DESIGN, "500": OFF}, "P184": {"32": DATA, "4.5": "assignment error, rounded from the headline"},
 "P186": {"16.3": DESIGN, "1,340": ROUND, "176": ROUND},
}
OPEN = {
}
# numbers that match a v1.43 ledger value only by coincidence, and numbers in the v1.44 wording
_A = {
 "P40": {"50": "the 50% point, a definition"}, "P50": {"860": "value reported by [43]", "620": "value reported by [43]", "270": "value reported by [43]", "10": "value reported by [43]: 9.93% rounded"},
 "P61": {"50": DESIGN + ": 50 MHz carrier"}, "P71": {"−164": "phase of the swap session, in the ledger as arg(k)", "−143": "phase of the swap session, in the ledger as arg(k)"},
 "P85": {"60": ROUND + ": 0.34 of 0.56"}, "P93": {"50": "training hyperparameter: at most 50 epochs"}, "P104": {"600": SIZE},
 "P106": {"50": DESIGN + ": 5 cycles at 100 MHz"}, "P122": {"44": DESIGN + ": bit index"}, "P131": {"44": DESIGN + ": bit index"}, "P155": {"44": DESIGN + ": bit index"},
 "P144": {"50": DESIGN + ": 50 MHz pulse and the 50% crossing"}, "P149": {"50": DESIGN + ": 5 cycles at 100 MHz"},
 "P156": {"50": "the 50% point, a definition", "0.26": ROUND + ": 0.251% and 0.265%"}, "P168": {"50": "the 50% point, a definition"},
 "P162": {"50": "the 50% point, a definition", "10": "number of loads, ledger row name", "1.2": "v1.44 latency study, per-shot s.d. 1.24-1.48 ns over 14 loads (campaign_v144/final/latency)",
          "1.5": "same", "0.7": "v1.44 latency study, largest sensitivity shift 0.74 ns (campaign_v144/final/latency/cold/B_c2.json)"},
 "P171": {"10": "number of loads, ledger row name"}, "P184": {"4.4": "assignment error, rounded from the headline (100 - 95.60)"},
}
# Appendices A-C (P188-P200)
_CR = "v1.44 calibration repeats, 8 per board on one load: s.d. of arg k and of the probe phase (campaign_v144/final/cal_repeat/cal_repeat_<host>.json)"
_PH = "v1.44 phase across loads, board A (campaign_v144/final/board2/A_phrst_probe_v144.json)"
_B = {
 "P189": {"500": DESIGN + ": calibration shots", "127": _PH + ", ledger 'Phase across loads'", "−121": _PH + ", ledger 'Phase across loads'",
          "125": _PH + ", ledger 'Phase across loads'", "58": _PH + ", ledger 'Phase across loads'", "−34": _PH + ", ledger 'Phase across loads'",
          "0.33": _PH + ": per-shot s.d. per load without phrst, 0.33-0.40 deg", "0.40": _PH + ": per-shot s.d. per load without phrst, 0.33-0.40 deg",
          "32": CALC + ": separation of the two phrst clusters from the ledger means (-1.9 vs -33.5 deg)", "9": _PH + ": per-shot s.d. with phrst, 8.7-9.2 deg"},
 "P190": {"0.87": _CR, "1.00": _CR, "0.05": _CR, "0.10": _CR, "100": DESIGN + ": the 100 MHz mixing image",
          "0.086": ROUND + ": ledger 'Drift, 3 h' s.d.", "0.084": ROUND + ": ledger 'Drift, 3 h' s.d.", "0.26": "ledger 'Drift, 3 h', p of the paired comparison"},
 "P191": {"600": SIZE, "1,000": SIZE, "1,599": SIZE, "11": DESIGN + ": boxcar length M", "−0.25": "boxcar map delay s (calibration/box_domain_map.json)",
          "+0.37": "earlier build, offline: domain-map ablation of 2026-10-01 (analysis_scripts/analysis/phase0_calibration_ablation.py, results/phase0_calibration_ablation.json); said so in the text"},
 "P193": {"9,999": SIZE, "110,000": SIZE, "299,999": SIZE, "10,000": SIZE, "109,999": SIZE, "1,000": SIZE, "1,599": SIZE, "20": "validation split",
          "12": CALC + ": sum of the ledger's DAC-clipped shots, 7 + 5"},
 "P194": {"23": "tools/verify_no_leakage.py re-run 2026-11-06, 23/23: campaign_v144/final/analysis/verify_no_leakage.log", "1,000": SIZE, "1,599,": SIZE, "1,599": SIZE},
 "P195": {"95": TEST, "0.0030": ROUND + ": ledger 'Late-rate 95% upper bound'"},
 "P196": {"576": DESIGN, "640": "window length compared", "770": "window length compared"},
 "P198": {"200": "ledger 'Boxcar vs raw buffer', 200/200 per board", "400": "v1.44 tProc decision test, 400/400 per board (campaign_v144/final/tests/<board>_tproc_decision.json)",
          "0.4": DESIGN + ": fault trigger offset of the binding-fault test (binding_fault_check.py)", "25": "v1.44 binding-fault test counts (campaign_v144/final/tests/<board>_binding_fault.json)"},
 "P199": {"50": "the 50% point, a definition"}, "P200": {"21.5": "ledger 'Phase difference of the boards in that session'"},
}
for _d in (_A, _B):
    for _p, _x in _d.items():
        L.setdefault(_p, {}).update(_x)
root = Path(__file__).resolve().parents[3]
with open(root / "paper_notes" / "v144_exempt.tsv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, delimiter="\t", lineterminator="\n")
    f.write("# paragraph\tnumber\tclass\treason. Generated by campaign_v144/final/analysis/make_v144_exempt.py; edit that, not this.\n")
    for table, cls in ((L, "ok"), (OPEN, "open")):
        for p in sorted(table, key=lambda x: int(x[1:])):
            for n, why in table[p].items():
                w.writerow([p, n, cls, why])
print("exempt written")
