## live_res_cdc_toplevel.xdc: unscoped version of live_res_cdc.xdc, for builds where SCOPED_TO_REF does not
## bind (e.g. a renamed module-reference wrapper). Not used in the build. Matches instances live_res_cdc_0 and
## live_res_cdc_1 by -hier pattern; same values as live_res_cdc.xdc.
set_bus_skew -from [get_cells -hier -filter {NAME =~ *live_res_cdc_0/* && (NAME =~ *tog_reg || NAME =~ *hold_reg[*])}]              -to   [get_cells -hier -filter {NAME =~ *live_res_cdc_0/* && (NAME =~ *tog_sync_reg[0] || NAME =~ *m_axis_tdata_reg[*])}] 2.000
set_bus_skew -from [get_cells -hier -filter {NAME =~ *live_res_cdc_1/* && (NAME =~ *tog_reg || NAME =~ *hold_reg[*])}]              -to   [get_cells -hier -filter {NAME =~ *live_res_cdc_1/* && (NAME =~ *tog_sync_reg[0] || NAME =~ *m_axis_tdata_reg[*])}] 2.000
