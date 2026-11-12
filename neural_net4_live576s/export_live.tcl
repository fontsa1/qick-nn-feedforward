source project.tcl
open_project NN_live_prj
set_top NN_live
open_solution solution1
export_design -format ip_catalog -version $version
exit
