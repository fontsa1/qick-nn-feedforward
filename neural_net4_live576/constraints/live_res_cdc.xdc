## live_res_cdc.xdc -- timing constraints for the live_res_cdc hold-and-toggle crossing
## (v1.41, audit R.16 / D4). SCOPED TO THE MODULE: add it to constrs_1 with
##   add_files -fileset constrs_1 live_res_cdc.xdc
##   set_property SCOPED_TO_REF live_res_cdc [get_files live_res_cdc.xdc]
##   set_property PROCESSING_ORDER LATE      [get_files live_res_cdc.xdc]
##   set_property USED_IN_SYNTHESIS false    [get_files live_res_cdc.xdc]
##   set_property USED_IN_IMPLEMENTATION true [get_files live_res_cdc.xdc]
## so every instance of the module gets it (d_1_i/live_res_cdc_0: clk_adc0_x2 -> clk_core 200 MHz,
## d_1_i/live_res_cdc_1: clk_adc0_x2 -> clk_dac2 491.52 MHz). Object names below are relative to
## the instance. If live_res_cdc is ever packaged as an IP, ship this file in the IP with
## PROCESSING_ORDER LATE (same content).
##
## Why: the source writes `hold` and flips `tog` in the same s_aclk cycle; the destination
## samples `hold` into m_axis_tdata two destination edges after `tog` was first captured by
## tog_sync[0]. The scheme is correct only if the 64 data bits do not arrive much later than the
## toggle: at 491.52 MHz a data-vs-toggle skew of 4.0 ns gave 0/3000 corrupted results in sim,
## 4.5 ns gave 664/3000 (lane D, tb_cdc_skew). QICK's timing.xdc puts both clock pairs in
## asynchronous clock groups (timing.xdc lines 39-41, 51-53), so nothing bounds that skew today.
##
## set_bus_skew is NOT overridden by set_clock_groups / set_false_path (UG903), so it is the
## constraint that actually takes effect with the unmodified QICK timing.xdc. 2.0 ns = one
## clk_dac2 period minus margin (the faster of the two destinations; half the measured
## tolerance there and a fifth of it at 200 MHz). Check after routing with
##   report_bus_skew -warn_on_violation -file bus_skew_live_res_cdc.rpt
set_bus_skew -from [get_cells {tog_reg hold_reg[*]}] \
             -to   [get_cells {tog_sync_reg[0] m_axis_tdata_reg[*]}] 2.000

## The absolute bound (set_max_delay -datapath_only) is in live_res_cdc_optionB.xdc: it has no
## effect while QICK's asynchronous clock groups cover these clock pairs (they take precedence).
