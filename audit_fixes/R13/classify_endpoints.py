"""Classify every failing setup endpoint of the v1.40L10 build (from a read-only Vivado dump).

Input : vivado_out/failing_setup_endpoints.csv (one worst path per failing endpoint pin)
Output: endpoints_classified.csv (one row per endpoint) and endpoints_summary.csv / .md (grouped).

Classes
  F-dec   functional: decision bit / result word delivered to the tProc or the DAC gate
  F-nn    functional: NN_live's internal datapath / control (feeds the decision and the record)
  F-in    functional: NN input data (readout 0 front end, boxcar, window gate) -- same value also goes to the
          reference buffer, so window==buffer cannot see it
  V       verification / acquisition capture only (NN record BRAM, reference buffer 1); an error would show as a
          mismatch in the bit-exact checks, never as a false match, and cannot reach the loop
  D       diagnostic only (timestamps, edge detectors, per-shot logs)
  Q       stock QICK, functional for QICK users but outside our loop (readout 0 FIR buffer)
  U       stock QICK block not used by our design on this wiring (readouts 1/2, weighted buffer, time tagger,
          DDR4 capture, their inputs)
Modifier (column `static_src`): source is a reset or a quasi-static configuration register that does not toggle in
operation; a setup miss on a path whose launch value does not change cannot capture a wrong value.
"""
import csv
import re
from collections import defaultdict
from pathlib import Path

D = Path(__file__).resolve().parent


def strip(p):
    return p.replace("OG_build_i/", "")


def block(p):
    return strip(p).split("/")[0]


def static_source(sp):
    s = strip(sp)
    if s.startswith("rst_"):
        return "reset"
    if re.search(r"boxcar_tap_0/inst/(cfg_rr?|sel|msel|shift|rnd)_reg", s):
        return "boxcar cfg"
    if re.search(r"seg_stamp_0/inst/(cfg|rf_thr|thr)_reg", s):
        return "seg cfg"
    if re.search(r"down_conversion_i/dds_ctrl_int_r_reg", s):
        return "readout DDS ctrl (changes only at a readout-config load, ~1/shot)"
    if re.search(r"down_conversion_i/dds_tvalid_r_reg", s):
        return "readout DDS tvalid (constant 1 after reset)"
    if re.search(r"dac_gate_0/inst/(cfg|delay_r)", s):
        return "gate cfg"
    return ""


def classify(ep, sp):
    e = strip(ep)
    b = block(ep)
    if b == "live_res_cdc_0":
        return "F-dec", "tProc input-3 word (hold/tog)", "shot field: tProc fresh 720/720; decision bit NOT logged"
    if b == "live_res_cdc_1":
        return "F-dec", "gate decision word (hold/tog)", "gate record vs NN record 358/358; 1 decision per shot 720/720"
    if b == "NN_live_0":
        if "sigmoid" in e:
            return "V", "sigmoid table (record word 6 only)", "record bit-exact 22,000/22,000"
        return "F-nn", "NN_live datapath/control", "record bit-exact 22,000/22,000 (A 20k + B 2k)"
    if b == "axi_bram_ctrl_live_bram":
        return "V", "NN record/capture BRAM write port", "is the bit-exact evidence; errors -> mismatch"
    if b == "live_gate_0":
        return "F-in", "window gate (NN input framing)", "window==buffer 22,000/22,000; frame errors 0"
    if b == "boxcar_tap_0":
        if re.search(r"/(s1[iq]|s2[iq]|b8[iq]|v13)", e):
            return "F-in", "boxcar adder tree (error heals after M)", "range/jump check only (no bit-exact ref)"
        if re.search(r"/(sr[iq]|new_|old_|dif_|acc_|vac)", e):
            return "F-in", "boxcar moving sum (error persists until cfg change)", "in-run step scan only"
        if re.search(r"/(rd_|sh_|rnd|m_box|vsh)", e):
            return "F-in", "boxcar round/shift/sat/output", "range/jump check only (no bit-exact ref)"
        return "F-in", "boxcar cfg/select", "static cfg"
    if b == "axis_avg_buffer_1":
        if "buffer_top_i" in e:
            return "V", "reference buffer 1, decimated memory (boxcar stream capture)", "window==buffer: errors -> mismatch"
        return "V", "buffer 1 accumulated-sum path (not used by any check or the loop)", "n/a"
    if b == "seg_stamp_0":
        if re.search(r"/cfg_reg", e):
            return "F-in", "seg_stamp cfg out -> boxcar cfg", "static after host write"
        if re.search(r"/(rf_thr|thr)_reg", e):
            return "D", "seg_stamp thresholds (rf_tap / edge detectors only)", "n/a"
        if re.search(r"/bram_(addr|we|en)", e):
            return "D", "seg record BRAM addr/we (could hit cfg word 3 if corrupted)", "none"
        return "D", "segment timestamps / edge detectors", "n/a"
    if b == "axi_bram_ctrl_seg_bram":
        return "D", "seg record BRAM", "n/a"
    if b == "rf_tap_0":
        return "D", "rf_tap hit detector", "n/a"
    if b == "axis_register_slice_2":
        return "F-in", "readout 0 input register slice (ADC -> readout 0)", "indirect (fidelity, T20)"
    if b == "axis_dyn_readout_v1_0":
        if "down_conversion_fir_i/fir_i/" in e:
            return "Q", "readout 0 FIR (buffer 0 only; not NN input on boxcar path)", "T11/T12/T20"
        if "dds_i/U0" in e:
            return "F-in", "readout 0 DDS phase accumulator (from DDS cfg register)", "indirect (T08, T20, fidelity)"
        if "readout_top_i/fifo_i/" in e:
            return "F-in", "readout 0 config FIFO", "indirect (T08, fidelity)"
        return "F-in", "readout 0 mixer input data (ADC -> m0 -> boxcar)", "indirect (fidelity, T20)"
    if b == "axis_cdcsync_v1_0":
        return "F-in", "tProc -> readout 0 config CDC FIFO (read side, popped by readout 0)", "indirect (T08, fidelity)"
    if b == "axis_avg_buffer_0":
        return "Q", "readout 0 FIR buffer", "T11/T12 exact"
    if b in ("axis_dyn_readout_v1_1", "axis_dyn_readout_v1_2", "axis_register_slice_3", "axis_register_slice_4",
             "axis_register_slice_5", "axis_weighted_buffer_0", "qick_time_tagger_0", "ddr4"):
        return "U", "stock QICK, unused on this wiring", "n/a"
    if b.startswith("rst_"):
        return "U", "reset synchroniser", "static"
    return "?", "unclassified", ""


def main():
    rows = list(csv.DictReader(open(D / "vivado_out" / "failing_setup_endpoints.csv")))
    out = []
    for r in rows:
        c, what, prot = classify(r["endpoint"], r["startpoint"])
        out.append(dict(slack=float(r["slack"]), block=block(r["endpoint"]), cls=c, what=what,
                        static_src=static_source(r["startpoint"]), protection=prot,
                        start_clock=r["start_clock"], end_clock=r["end_clock"], levels=r["logic_levels"],
                        startpoint=strip(r["startpoint"]), endpoint=strip(r["endpoint"])))
    out.sort(key=lambda x: x["slack"])
    with open(D / "endpoints_classified.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    g = defaultdict(list)
    for x in out:
        g[(x["block"], x["cls"], x["what"], x["static_src"], x["protection"])].append(x)
    summ = []
    for (b, c, wh, st, pr), xs in g.items():
        ws = min(xs, key=lambda x: x["slack"])
        summ.append(dict(cls=c, block=b, what=wh, static_src=st, n=len(xs), worst=ws["slack"],
                         tns=round(sum(x["slack"] for x in xs), 3), worst_start=ws["startpoint"],
                         worst_end=ws["endpoint"], protection=pr))
    order = {"F-dec": 0, "F-nn": 1, "F-in": 2, "V": 3, "D": 4, "Q": 5, "U": 6, "?": 7}
    summ.sort(key=lambda s: (order[s["cls"]], s["worst"]))
    with open(D / "endpoints_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summ[0].keys()))
        w.writeheader()
        w.writerows(summ)
    tot = defaultdict(int)
    for x in out:
        tot[x["cls"]] += 1
    print("total", len(out), dict(tot))
    print("static-source endpoints:", sum(1 for x in out if x["static_src"]))
    for s in summ:
        print(f"{s['cls']:5s} {s['block']:24s} n={s['n']:4d} worst={s['worst']:+.3f} static={s['static_src'] or '-':10s} {s['what']}")


if __name__ == "__main__":
    main()
