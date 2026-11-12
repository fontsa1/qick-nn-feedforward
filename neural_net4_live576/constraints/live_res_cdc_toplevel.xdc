## live_res_cdc_toplevel.xdc -- FALLBACK for live_res_cdc.xdc if SCOPED_TO_REF does not bind
## (e.g. the BD module-reference wrapper renames the module). Unscoped, PROCESSING_ORDER LATE,
## implementation only. Paths as in the v1.40L10 block design (d_1_i/live_res_cdc_0, _1); -hier
## patterns so the wrapper level (inst) does not matter. Same values and reasoning as live_res_cdc.xdc.
set_bus_skew -from [get_cells -hier -filter {NAME =~ *live_res_cdc_0/* && (NAME =~ *tog_reg || NAME =~ *hold_reg[*])}]              -to   [get_cells -hier -filter {NAME =~ *live_res_cdc_0/* && (NAME =~ *tog_sync_reg[0] || NAME =~ *m_axis_tdata_reg[*])}] 2.000
set_bus_skew -from [get_cells -hier -filter {NAME =~ *live_res_cdc_1/* && (NAME =~ *tog_reg || NAME =~ *hold_reg[*])}]              -to   [get_cells -hier -filter {NAME =~ *live_res_cdc_1/* && (NAME =~ *tog_sync_reg[0] || NAME =~ *m_axis_tdata_reg[*])}] 2.000
