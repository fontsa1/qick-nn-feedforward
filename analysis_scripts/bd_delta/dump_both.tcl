# 1) QICK's qick_tprocv2_4x2_standard project at 779dffa, built in this script's folder; 2) our v1.43 project, opened read-only.
# Edit these two paths. Keep this folder's path short on Windows (260-character limit).
set QICK_PROJ C:/qick/firmware/projects/qick_tprocv2_4x2_standard
set OUR_XPR C:/qick_projects/top/top.xpr
set here [file dirname [file normalize [info script]]]
source $here/dump_bd.tcl
set_param general.maxThreads 2
set ::origin_dir_loc $QICK_PROJ
cd $here
if {[catch {source $::origin_dir_loc/proj.tcl} e]} { puts "proj.tcl stopped (expected at the wrapper add): $e" }
open_bd_design [get_files d_1.bd]
dump_bd $here/qick_779dffa_d_1.tsv
puts "VANILLA_DUMPED"
close_project
open_project -read_only $OUR_XPR
open_bd_design [get_files OG_build.bd]
dump_bd $here/v143_OG_build.tsv
puts "OURS_DUMPED"
close_project
puts "DUMP_DONE"
