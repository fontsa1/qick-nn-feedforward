source project.tcl
open_project -reset NN_axi_prj
set_top NN_axi
add_files firmware/NN_axi.cpp -cflags "-std=c++0x"
add_files firmware/NN.cpp -cflags "-std=c++0x"
add_files -tb NN_axi_test.cpp -cflags "-Wno-unknown-pragmas"
add_files -tb firmware/weights -cflags "-Wno-unknown-pragmas"
add_files -tb tb_data -cflags "-Wno-unknown-pragmas"
open_solution -reset solution1 -flow_target vivado
set_part xczu48dr-ffvg1517-2-e
create_clock -period 10 -name default
set_clock_uncertainty 12.5%
config_compile -name_max_length 80
config_export -format ip_catalog -version $version
csim_design
exit
