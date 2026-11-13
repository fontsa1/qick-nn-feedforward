## Out-of-context timing study: clocks as in the deployed routed design.
## clk_adc0_x2 1.809 ns (552.94 MHz in the routed report), RFDAC2 2.035 ns, tProc core 5.001 ns.
create_clock -name clk_adc  -period 1.809 [get_ports clk_adc_p]
create_clock -name clk_dac  -period 2.035 [get_ports clk_dac_p]
create_clock -name clk_core -period 5.001 [get_ports clk_core_p]
create_clock -name clk_axi  -period 10.000 [get_ports clk_axi_p]
## Intra-clock uncertainty as in the routed design (MMCM jitter).
set_clock_uncertainty -setup 0.035 [get_clocks clk_adc]
set_clock_uncertainty -setup 0.051 [get_clocks clk_dac]
## QICK's timing.xdc puts these clock pairs in asynchronous groups.
set_clock_groups -asynchronous -group clk_adc -group clk_dac -group clk_core -group clk_axi
