## live_res_cdc_optionB.xdc: absolute max-delay bound on the live_res_cdc crossing (not used in the build).
## Scoped like live_res_cdc.xdc. Takes effect only if timing.xdc's asynchronous clock groups between clk_adc0_x2
## and clk_dac2 / clk_core are removed, since set_clock_groups takes precedence over set_max_delay; every other
## path between those clocks would then need its own exception. 1.8 ns = one clk_adc0_x2 period.
set_max_delay -datapath_only -from [get_cells {hold_reg[*]}] -to [get_cells {m_axis_tdata_reg[*]}] 1.800
set_max_delay -datapath_only -from [get_cells {tog_reg}]     -to [get_cells {tog_sync_reg[0]}]     1.800
