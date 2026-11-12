source project.tcl
open_project NN_axi_prj
set_top NN_axi
open_solution solution1
export_design -format ip_catalog -version $version
exit
