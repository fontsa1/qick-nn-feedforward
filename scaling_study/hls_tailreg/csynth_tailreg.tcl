# 4.5b experiment (not deployed): NN_live 1.5 source, unchanged, synthesized with a tighter clock
# uncertainty so the scheduler splits the fused tail. Usage: vitis_hls -f csynth_tailreg.tcl -tclargs <uncertainty%> <prj>
set unc [lindex $argv end-1]
set prj [lindex $argv end]
open_project -reset $prj
set_top NN_live
add_files firmware/NN_live.cpp -cflags "-std=c++14"
open_solution -reset solution1 -flow_target vivado
set_part xczu48dr-ffvg1517-2-e
create_clock -period 1.808 -name default
set_clock_uncertainty ${unc}%
config_compile -name_max_length=80
csynth_design
puts "CSYNTH_TAILREG_DONE unc=$unc"
exit
