#################################################################
# v1.43 stage 1 of 2. No synthesis; build_v1.43.tcl builds it.
# Starting point: the v1.42b BD (bd_backups/OG_build_v1.42b.bd) after two GUI sessions on 2026-11-04 that only
# ran update_module_reference on live_res_cdc_0/1 and left the BD unvalidated (clock properties cleared).
#  1. update_module_reference on seg_stamp_0 (dead register wr_shot removed; lint 2026-11-04).
#  2. validate, which re-propagates the clock properties. check_bd_v1.43.py then requires the saved BD to match
#     v1.42b in content. No other BD change.
#################################################################
set_param general.maxThreads 2
# Pre-edit backups are made once; run_stage1_v1.43.ps1 restores from them before each attempt.
if {![file exists bd_backups/OG_build_pre_v1.43.bd]} { file copy top.srcs/sources_1/bd/OG_build/OG_build.bd bd_backups/OG_build_pre_v1.43.bd }
if {![file exists top_pre_v1.43.xpr]} { file copy top.xpr top_pre_v1.43.xpr }
puts "STEP: opening project"
open_project top.xpr
set ips [get_ips -quiet OG_build_seg_stamp_0_0]
if {[llength $ips] != 1} { puts "FATAL: seg_stamp IP not found"; exit 1 }
puts "STEP: update_module_reference $ips"
update_module_reference $ips
puts "STEP: opening BD design"
open_bd_design [get_files OG_build.bd]
foreach {p want} {EARLY_BLANK 1 SHOT_BIND 1 FAIL_SAFE 1 CFG_SAFE 0} {
    if {[get_property CONFIG.$p [get_bd_cells dac_gate_0]] != $want} { puts "FATAL: dac_gate_0 $p changed"; exit 1 }
}
if {[get_property CONFIG.N [get_bd_cells live_gate_0]] != 576} { puts "FATAL: live_gate_0 N changed"; exit 1 }
puts "STEP: validating BD design"
validate_bd_design -force
save_bd_design
write_bd_tcl -force v1.43_bd.tcl
close_bd_design [get_bd_designs OG_build]
file copy -force top.srcs/sources_1/bd/OG_build/OG_build.bd bd_backups/OG_build_v1.43.bd
close_project
file copy -force top.xpr top_v1.43.xpr
puts "BDEDIT_COMPLETE_v1.43"
exit
