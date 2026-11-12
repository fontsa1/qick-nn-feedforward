source project.tcl
open_project -reset NN_live_prj
set_top NN_live
add_files firmware/NN_live.cpp -cflags "-std=c++14"
add_files -tb NN_live_test.cpp -cflags "-std=c++14 -Wno-unknown-pragmas"
add_files -tb tb_ref_nn.cpp -cflags "-std=c++14 -Wno-unknown-pragmas"
add_files -tb firmware/weights
add_files -tb tb_data
open_solution -reset solution1 -flow_target vivado
set_part $part
create_clock -period $clock_period -name default
set_clock_uncertainty $clock_uncertainty
config_compile -name_max_length 80
config_export -format ip_catalog -version $version
csim_design
exit
