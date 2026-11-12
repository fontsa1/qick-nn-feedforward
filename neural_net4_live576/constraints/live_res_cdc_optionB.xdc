## live_res_cdc_optionB.xdc -- OPTIONAL, only together with a timing.xdc change (BUILD_PLAN.md option B).
## Scoped like live_res_cdc.xdc (SCOPED_TO_REF live_res_cdc, PROCESSING_ORDER LATE, implementation only).
## Absolute bound on the crossing paths, as XPM_CDC does. set_clock_groups -asynchronous takes
## precedence over set_max_delay, so these lines only act if timing.xdc's clock groups
## clk_tproc_to_adc0_x2 (dac2<->adc0_x2) and clk_core_to_adc0_x2 are removed and every other path
## between those clocks gets its own exception (they are not all known to be safe: avg_buffer,
## readout and trigger crossings would then be timed). 1.8 ns = one clk_adc0_x2 (source) period.
set_max_delay -datapath_only -from [get_cells {hold_reg[*]}] -to [get_cells {m_axis_tdata_reg[*]}] 1.800
set_max_delay -datapath_only -from [get_cells {tog_reg}]     -to [get_cells {tog_sync_reg[0]}]     1.800
