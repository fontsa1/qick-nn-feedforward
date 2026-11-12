# 4.5a / 4.5b — multi-classifier scaling feasibility: findings

Run 2026-10-27 (FINAL_TASKS 4.5). Offline and out-of-context (OOC) only; no board work, no full build.
Device xczu48dr-ffvg1517-2-e, Vivado 2023.1. All numbers below are from files in this folder.

## Files

| file | what |
|---|---|
| `resource_scaling.py` → `resource_scaling.json` | 4.5a: per-unit cost from the routed v1.41 reports, resource-limited N |
| `rtl/classifier_unit.v` | one qubit's unit, exactly the v1.41 blocks wired as in the BD: boxcar_tap → trig_sync → live_window_gate (N = 576) → NN_live 1.5 → live_res_cdc ×2 → dac_gate (GAT4), plus the record RAM |
| `rtl/live_record_ram.v` | NN_live's 8 × RAMB36 record memory **with the planned write-port register stage** (4.1m fix) |
| `rtl/ooc_top.v`, `ooc.xdc`, `run_ooc.tcl`, `launch_ooc.bat` | N-unit OOC harness: BUFG clocks, v1.41 clock periods and setup uncertainty, QICK's asynchronous clock groups, the scoped live_res_cdc bus-skew constraint; v1.41's place/phys_opt/route directives |
| `rtl/*_zl.v`, `run_ooc_zl.tcl` | **ZL variant (experiment only, not deployed):** gate output register removed, NN_live's `res_o` registered before the CDCs — same total latency |
| `runs/N<N>_<pb\|free>[_zl]/` | per run: utilization, timing, failing endpoints (unit-internal vs harness), bus skew, routed .dcp |
| `runs/SUMMARY.txt` | one line per run |

Harness fidelity: synthesized unit = routed v1.41 within 3% per block (NN_live 1,343 vs 1,352 LUT; boxcar 1,105 vs
1,071; dac_gate 888 vs 858; same 8 RAMB36 + 1 RAMB18 + 4 DSP). Differences from the deployed design: the gate's
debug taps to seg_stamp, seg_stamp/rf_tap and averaging buffer 1 (diagnostic/verification) are not included.
Both CDC instances bound the 2 ns bus-skew constraint in every run (met).

## 4.5a — resources

Per unit (routed v1.41): classifier (NN_live + boxcar + gate + trig_sync + 2 CDC) 2,437 LUT, 3,231 FF, 4 DSP,
0.5 BRAM tile; dac_gate 858 LUT, 2,388 FF; record memory (verification) 8 BRAM tiles. Per-tone front ends (QICK's):
DDS/mixer only (`axis_dyn_readout_v1_1` as routed — its FIR output is unconnected, so the FIR is trimmed) 1,921 LUT,
35 DSP, 16 tiles; full stock readout with FIR 6,219 LUT, 157 DSP, 16 tiles.

Largest N that fits in what v1.41 leaves free:

| per-tone front end | record memory | v1.41 as built | v1.41 − NN_axi − time tagger |
|---|---|---|---|
| none (unit only) | yes / no | 58 / 58 (LUT) | 68 (BRAM) / 101 (LUT) |
| DDS/mixer only | yes / no | 23 / 34 (BRAM) | 23 / 35 (BRAM) |
| full stock readout | yes / no | 20 / 20 (LUT) | 23 / 23 (BRAM, DSP) |

Our logic is never the binding cost once a per-tone front end is counted: QICK's readout front end (16 BRAM tiles
for its DDS tables; 35–157 DSP) is. Resources allow ~20–35 qubits on this device; the RFSoC4x2's converter count
limits feed-forward outputs long before that.

## 4.5b — timing vs N (OOC, setup at 552.96 MHz, clk_adc0_x2 1.809 ns)

`pb` = all units constrained to v1.41's pb_live (CLOCKREGION_X4Y3:X5Y4, next to the ADC tiles: 36,480 LUT,
96 BRAM tiles, so ≤ 11 units fit); `free` = whole device.

| N | placement | RTL | WNS (ns) | failing endpoints (unit / harness) | pblock LUT fill |
|---|---|---|---|---|---|
| 1 | pb | v1.41 | +0.171 | 0 / 0 | 9% |
| 2 | pb | v1.41 | +0.103 | 0 / 0 | |
| 4 | pb | v1.41 | +0.002 | 0 / 0 | 37% |
| 8 | pb | v1.41 | −0.011 | 9 / 0 (6 of 8 units) | ~72% |
| 16 | free | v1.41 | +0.002 | 0 / 0 | |
| 32 | free | v1.41 | −0.183 | 1,103 / 142 | device 26% LUT |
| 4 | pb | ZL | +0.038 | 0 / 0 | |
| 8 | pb | ZL | +0.006 | 0 / 0 | |
| 4 | pb | T | +0.063 | 0 / 0 | |
| 8 | pb | T | +0.029 | 0 / 0 | |

RTL variants (experiments only, nothing deployed changed):
* **v1.41** — the deployed blocks.
* **ZL** — gate output register removed, `res_o` registered in the unit before the CDCs. Same end-to-end cycle
  count as v1.41.
* **T** — NN_live 1.5 source unchanged, re-synthesized with HLS clock uncertainty 30% (`hls_tailreg/`): ACC_L depth
  11 → 12 at II = 1, i.e. one extra pipeline stage (kernel last sample → `res_o` 9 → 10 cycles), plus the
  register-free gate. **Same end-to-end latency as v1.41**: v1.41 = gate register (1) + kernel (9) = 10 cycles from
  the readout output to the result (the 18.1 ns measured on hardware); T = gate (0) + kernel (10) = 10 cycles. HLS
  estimated Fmax 668 MHz. Not functionally re-verified (csim source identical; RTL sim / bit-exact not run).

**Every near-critical and failing path is the same one: NN_live's fused tail.** The last iteration computes the
four rectified units, the signed sums and the saturating select in one cycle (`NN_live.cpp:188–209`; 4–6 logic
levels, 3–4 CARRY8, ~0.55–0.67 ns of logic) and drives both `res_o` — combinationally into the two CDC hold
registers — and the saturated-logit register `lr_1_reg` inside NN_live. Nothing in the failing set involves another
unit, the stream fan-out or the clock crossings themselves. Margin shrinks with packing density because routing
around the carry chains gets longer, not because units interact.

The same path family fails in the **integrated** v1.41 build at N = 1 (`live_res_cdc_0` hold_reg −0.042 ns, NN_live
internal −0.141), i.e. QICK's congestion costs ~0.2 ns against the OOC N = 1 margin of +0.171.

ZL removes the tail → CDC paths from the critical set; the remaining worst path is NN_live-internal (tail →
`lr_1_reg`). T removes the tail from the critical set altogether: at N = 8 in pb the worst path becomes the weight-ROM
lookup (loop counter → LUTRAM ROM output register, 4 LUT levels, +0.029), i.e. the next-slowest stage of NN_live.

N = 32 (whole device, 26% of the LUTs) is where it breaks broadly: 1,103 unit endpoints spread over all 32 units, 607 of them
in dac_gate's 256-bit DAC-domain path (2.035 ns), 193 in NN_live, ~180 in the CDC hold registers, 97 in the record
RAM. This case is pessimistic in one specific way: the harness feeds every unit from one shared ADC stream and one
shared DAC stream, so 32 units spread over the whole die share global buses; in a real multiplexed design each tone's
unit would sit beside its own readout and generator.

Unconstrained placement does not buy margin: at N = 16 over the whole device the worst path is again the tail
(+0.002). The tail's margin is a property of the kernel at 552.96 MHz (~0.0–0.17 ns depending on placement), not of
crowding alone.

## What this means for 4.5c (integrated N = 2–3 build)

Not built. With the v1.41 RTL the decision path of every unit sits at ~0 to +0.17 ns out of context, and the same
path already fails at N = 1 inside the real design (−0.042 ns; QICK's congestion costs ~0.2 ns). An integrated build
with 2–3 units would almost certainly show failing endpoints on each unit's decision path, which is the opposite of
the claim 4.5c is meant to support. The T variant gains 0.04–0.06 ns out of context at the same end-to-end latency,
still less than the ~0.2 ns integration penalty measured at N = 1, and it changes the deployed design (NN_live,
gate, re-verification), so it is a user decision.

## v1.43 recount (2026-11-07)

`SCALING_OUT=resource_scaling_v143.json python scaling_study/resource_scaling.py` (v141_reports now reads the v1.43 reports;
v1.43 has no time tagger, and its record memory includes live_rec_wreg_0). With a per-tone front end: 18-35 units
(full stock readout, as built: 18, LUT-bound; DDS/mixer only, without NN_axi and record memory: 35, BRAM-bound). The unit's RTL is unchanged since
these OOC runs (sources last committed 2026-10-21, NN_live 1.5), so the timing results above apply to v1.43 as they are.
