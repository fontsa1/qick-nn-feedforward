## live_res_cdc.xdc: bus-skew bound for the live_res_cdc hold-and-toggle crossing.
## Scoped to the module, so every instance gets it:
##   add_files -fileset constrs_1 live_res_cdc.xdc
##   set_property SCOPED_TO_REF live_res_cdc [get_files live_res_cdc.xdc]
##   set_property PROCESSING_ORDER LATE      [get_files live_res_cdc.xdc]
##   set_property USED_IN_SYNTHESIS false    [get_files live_res_cdc.xdc]
##   set_property USED_IN_IMPLEMENTATION true [get_files live_res_cdc.xdc]
## Object names are relative to the instance.
##
## The destination samples `hold` two edges after `tog` is captured, so the 64 data bits must not arrive much
## later than the toggle. QICK's timing.xdc puts both clock pairs in asynchronous groups; set_bus_skew still
## applies (UG903). 2.0 ns is about one clk_dac2 period, the faster destination. Check after routing with
##   report_bus_skew -warn_on_violation
set_bus_skew -from [get_cells {tog_reg hold_reg[*]}] \
             -to   [get_cells {tog_sync_reg[0] m_axis_tdata_reg[*]}] 2.000

## An absolute max-delay bound is in live_res_cdc_optionB.xdc (inactive with QICK's clock groups).
