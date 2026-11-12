#################################################################
# v1.43 stage 2 of 2. Stage 1: bdedit_v1.43.tcl (seg_stamp_0 module reference; BD content otherwise = v1.42b).
# Flow identical to build_v1.42.tcl. Differences:
#  1. seg_stamp.v: dead register wr_shot removed (no behaviour change; checked in the OOC log below).
#  2. Implementation-only over-constraint on the dac_gate blank path (setup 150 ps tighter than RFDAC2_CLK's
#     period during place and route, removed before the sign-off reports). v1.42 failed it by 1-2 ps.
#  3. Full report set after the bitstream (FINAL_TASKS 6.2): report_methodology, report_cdc for all clock pairs,
#     report_drc, report_clock_interaction.
#  4. Provenance also saves the uncommitted repo diff.
#################################################################

set TAG v1.43
set BUILD_T0 [clock seconds]

# ---- provenance (R.21): repo commit and dirty state, source hashes
set PROV [open ${TAG}_provenance.txt w]
puts $PROV "build $TAG started [clock format $BUILD_T0 -format {%Y-%m-%d %H:%M:%S}]"
puts $PROV "vivado [version -short]"
if {[catch {puts $PROV "qick_project commit [exec git -C . rev-parse HEAD]"} e]} { puts $PROV "qick_project commit: unavailable ($e)" }
if {[catch {puts $PROV "qick_project uncommitted files: [llength [split [string trim [exec git -C . status --porcelain]] 
]]"} e]} { puts $PROV "git status: unavailable" }
foreach f {neural_net4_live576/rtl/live_rec_wreg.v neural_net4_live576/rtl/dac_gate.v neural_net4_live576/rtl/boxcar_tap.v neural_net4_live576/rtl/seg_stamp.v neural_net4_live576/rtl/rf_tap.v neural_net4_live576/rtl/trig_sync.v neural_net4_live576/rtl/live_res_cdc.v neural_net4_live/rtl/live_window_gate.v neural_net4_live576/constraints/live_res_cdc.xdc neural_net4_tlast/NN_axi_prj/solution1/impl/ip/xilinx_com_hls_NN_axi_1_41.zip} {
    set p ./$f
    if {[file exists $p]} { puts $PROV "src [file size $p] [file mtime $p] $f" } else { puts $PROV "src MISSING $f" }
}
if {[catch {
    set dp [open ${TAG}_uncommitted.diff w]
    puts $dp [exec git -C . diff]
    close $dp
    puts $PROV "uncommitted diff saved to ${TAG}_uncommitted.diff"
} e]} { puts $PROV "git diff: unavailable ($e)" }
close $PROV
puts "STEP: provenance written to ${TAG}_provenance.txt"

set_param general.maxThreads 2
puts "STEP: maxThreads capped at [get_param general.maxThreads] (TIMING_CLOSURE_REFERENCE #11)"

puts "STEP: opening project"
open_project top.xpr

puts "STEP: opening BD design"
open_bd_design [get_files OG_build.bd]

foreach cell {live_rec_wreg_0 NN_live_0 live_gate_0 axi_bram_ctrl_live axi_bram_ctrl_live_bram live_res_cdc_0 seg_stamp_0 axi_bram_ctrl_seg axi_bram_ctrl_seg_bram clk_core dac_gate_0 live_res_cdc_1 axi_bram_ctrl_gate axi_bram_ctrl_gate_bram boxcar_tap_0 trig_sync_0 rf_tap_0} {
    if {[llength [get_bd_cells -quiet $cell]] == 0} { puts "FATAL: $cell missing from the BD"; exit 1 }
}
set vlnv [get_property VLNV [get_bd_cells NN_axi_0]]
puts "NN_axi_0 VLNV: $vlnv"
if {![string match "*:1.41" $vlnv]} { puts "FATAL: NN_axi_0 is $vlnv, expected 1.41 -- run bdedit_v1.42b.tcl first"; exit 1 }
foreach cell {qick_time_tagger_0 axi_dma_tt axis_register_slice_5} {
    if {[llength [get_bd_cells -quiet $cell]] != 0} { puts "FATAL: $cell still in the BD -- run bdedit_v1.42.tcl first"; exit 1 }
}
if {[get_bd_intf_nets -of_objects [get_bd_intf_pins NN_live_0/out_r_PORTA]] ne [get_bd_intf_nets -of_objects [get_bd_intf_pins live_rec_wreg_0/s_bram]]} { puts "FATAL: live_rec_wreg_0 not on NN_live_0/out_r_PORTA"; exit 1 }
if {[get_bd_intf_nets -of_objects [get_bd_intf_pins live_rec_wreg_0/m_bram]] ne [get_bd_intf_nets -of_objects [get_bd_intf_pins axi_bram_ctrl_live_bram/BRAM_PORTB]]} { puts "FATAL: live_rec_wreg_0 not on the record BRAM"; exit 1 }
puts "V142_BD: NN_axi_0 1.41, time tagger absent, live_rec_wreg_0 in place"
set lvlnv [get_property VLNV [get_bd_cells NN_live_0]]
puts "NN_live_0 VLNV: $lvlnv"
if {![string match "*:1.5" $lvlnv]} { puts "FATAL: NN_live_0 is $lvlnv, not 1.5"; exit 1 }
set g [get_bd_cells dac_gate_0]
foreach {p want} {EARLY_BLANK 1 SHOT_BIND 1 FAIL_SAFE 1 CFG_SAFE 0} {
    set got [get_property CONFIG.$p $g]
    puts "DAC_GATE_PARAM: $p = $got"
    if {$got != $want} { puts "FATAL: dac_gate_0 CONFIG.$p is '$got', expected $want -- run stage 1 first"; exit 1 }
}
set gn [get_property CONFIG.N [get_bd_cells live_gate_0]]
puts "live_gate_0 CONFIG.N: $gn"
if {$gn != 576} { puts "FATAL: live_gate_0 CONFIG.N is $gn, expected 576"; exit 1 }
set gt [get_bd_nets -of_objects [get_bd_pins qick_processor_0/trig_11_o]]
if {[get_bd_nets -of_objects [get_bd_pins trig_sync_0/trig_in]] ne $gt} { puts "FATAL: trig_sync_0/trig_in is not on trig_11"; exit 1 }
if {[get_bd_nets -quiet -of_objects [get_bd_pins -quiet dac_gate_0/cls_trig]] ne $gt} { puts "FATAL: dac_gate_0/cls_trig is not on trig_11"; exit 1 }
if {[get_bd_nets -quiet -of_objects [get_bd_pins boxcar_tap_0/cfg]] ne [get_bd_nets -quiet -of_objects [get_bd_pins seg_stamp_0/cfg]]} { puts "FATAL: boxcar_tap_0/cfg not driven by seg_stamp_0"; exit 1 }
set nin [get_property CONFIG.IN_PORT_QTY [get_bd_cells qick_processor_0]]
puts "qick_processor_0 IN_PORT_QTY: $nin"
if {$nin != 4} { puts "FATAL: qick_processor_0 IN_PORT_QTY is $nin, expected 4"; exit 1 }
set cdcx [get_files -quiet -of_objects [get_filesets constrs_1] *live_res_cdc.xdc]
if {[llength $cdcx] != 1} { puts "FATAL: live_res_cdc.xdc not in constrs_1 -- run stage 1 first"; exit 1 }
puts "CDC_XDC: $cdcx SCOPED_TO_REF=[get_property SCOPED_TO_REF $cdcx]"

puts "STEP: validating BD design"
validate_bd_design -force

puts "STEP: generate_target"
generate_target all [get_files OG_build.bd]
file copy -force top.gen/sources_1/bd/OG_build/hw_handoff/OG_build.hwh OG_build_${TAG}.hwh
puts "STEP: hwh snapshot written to OG_build_${TAG}.hwh"

puts "STEP: export_ip_user_files"
export_ip_user_files -of_objects [get_files OG_build.bd] -no_script -sync -force

puts "STEP: updating compile order"
update_compile_order -fileset sources_1

set OOC_RUNS {}
foreach c {live_rec_wreg_0 NN_axi_0 dac_gate_0 boxcar_tap_0 NN_live_0 live_res_cdc_0 live_res_cdc_1 live_gate_0 seg_stamp_0 rf_tap_0 trig_sync_0} {
    set r OG_build_${c}_0_synth_1
    if {[llength [get_runs -quiet $r]]} { puts "STEP: reset_run $r"; reset_run $r; lappend OOC_RUNS $r } else { puts "WARNING: no OOC run $r (synthesised in the top-level run)" }
}
puts "STEP: resetting and launching synth_1"
reset_run synth_1
launch_runs synth_1 -jobs 6
wait_on_run synth_1

set synth_status [get_property STATUS [get_runs synth_1]]
puts "STEP: synth_1 finished with status: $synth_status"
if {[string match "*ERROR*" $synth_status] || [string match "*Fail*" $synth_status]} { puts "FATAL: synth_1 failed -- $synth_status"; exit 1 }

foreach r $OOC_RUNS {
    set d [get_property DIRECTORY [get_runs $r]]
    set f [glob -nocomplain $d/*.dcp]
    if {[llength $f] != 1 || [file mtime [lindex $f 0]] < $BUILD_T0} { puts "FATAL: $r was not re-synthesized in this build ($f)"; exit 1 }
    puts "OOC_CHECK: $r re-synthesized ([clock format [file mtime [lindex $f 0]] -format %H:%M:%S])"
}
set segd [get_property DIRECTORY [get_runs OG_build_seg_stamp_0_0_synth_1]]
set sf [open $segd/runme.log r]; set slog [read $sf]; close $sf
if {[string first wr_shot $slog] >= 0} { puts "FATAL: seg_stamp OOC run still has wr_shot; the edit was not picked up"; exit 1 }
puts "SEG_CHECK: seg_stamp OOC log has no wr_shot"
puts "STEP: opening synth_1 run (open_run, not raw open_checkpoint -- avoids INBB-3 black-box bug)"
open_run synth_1
report_utilization -file ${TAG}_post_synth_utilization.rpt
report_utilization -hierarchical -file ${TAG}_post_synth_util_hier.rpt
puts "STEP: post-synth utilization written"

# ---- R.16 bus-skew constraint: did the scoped file bind? If not, apply it here.
proc cdc_cells {inst kind} {
    if {$kind eq "from"} {
        return [get_cells -quiet -hier -regexp ".*/${inst}/(inst/)?(tog_reg|hold_reg\\\[\[0-9\]+\\\])"]
    }
    return [get_cells -quiet -hier -regexp ".*/${inst}/(inst/)?(tog_sync_reg\\\[0\\\]|m_axis_tdata_reg\\\[\[0-9\]+\\\])"]
}
set bsk [report_bus_skew -return_string]
set nbound [regexp -all {live_res_cdc_[01]} $bsk]
puts "BUSSKEW_SCOPED: report_bus_skew mentions live_res_cdc $nbound time(s)"
if {$nbound == 0} {
    puts "STEP: scoped live_res_cdc.xdc did not bind; applying set_bus_skew in memory (option A fallback)"
    foreach inst {live_res_cdc_0 live_res_cdc_1} {
        set fr [cdc_cells $inst from]
        set to [cdc_cells $inst to]
        puts "BUSSKEW_FALLBACK: $inst from [llength $fr] cells, to [llength $to] cells"
        if {[llength $fr] == 0 || [llength $to] == 0} { puts "FATAL: bus-skew cells for $inst not found"; exit 1 }
        set_bus_skew -from $fr -to $to 2.000
    }
    set bsk [report_bus_skew -return_string]
    set nbound [regexp -all {live_res_cdc_[01]} $bsk]
    puts "BUSSKEW_FALLBACK: report_bus_skew now mentions live_res_cdc $nbound time(s)"
    if {$nbound == 0} { puts "FATAL: set_bus_skew on live_res_cdc did not take"; exit 1 }
}
foreach inst {live_res_cdc_0 live_res_cdc_1} {
    puts "BUSSKEW_CELLS: $inst from [llength [cdc_cells $inst from]] to [llength [cdc_cells $inst to]]"
}

set ila_ok 0
puts "STEP: no ILA in this build"

puts "STEP: pblock pb_live (clock regions X4Y3:X5Y4)"
create_pblock pb_live
resize_pblock pb_live -add {CLOCKREGION_X4Y3:CLOCKREGION_X5Y4}
set live_cells {}
foreach b {boxcar_tap_0 live_gate_0 NN_live_0 live_rec_wreg_0 axi_bram_ctrl_live_bram seg_stamp_0 axi_bram_ctrl_seg_bram trig_sync_0 live_res_cdc_0} {
    set c [get_cells -quiet OG_build_i/$b]
    if {[llength $c] != 1} { puts "FATAL: pblock cell OG_build_i/$b not found"; exit 1 }
    lappend live_cells $c
}
add_cells_to_pblock pb_live $live_cells
puts "PBLOCK: pb_live <- [join $live_cells { }]"
if {[catch {report_utilization -pblocks pb_live -file ${TAG}_pblock_post_synth_util.rpt} e]} { puts "WARNING: pblock utilization report failed ($e)" }

# ---- v1.43: tighter target on the dac_gate blank path during place and route only
set blank_cells [get_cells -hier -filter {NAME =~ *dac_gate_0/inst/blank_reg* && IS_SEQUENTIAL}]
if {[llength $blank_cells] == 0} { puts "FATAL: no dac_gate blank registers found"; exit 1 }
set dclk [get_clocks RFDAC2_CLK]
set blank_target [expr {[get_property PERIOD $dclk] - 0.150}]
set_max_delay -from $dclk -to $blank_cells $blank_target
puts "BLANK_CONSTRAINT: [llength $blank_cells] register(s), max delay $blank_target ns"

puts "STEP: opt_design"
opt_design
write_checkpoint -force ${TAG}_post_opt.dcp
report_utilization -file ${TAG}_post_opt_utilization.rpt
report_exceptions -ignored -file ${TAG}_post_opt_exceptions_ignored.rpt
report_bus_skew -file ${TAG}_post_opt_bus_skew.rpt

puts "STEP: place_design (default)"
place_design
write_checkpoint -force ${TAG}_placed.dcp
report_timing_summary -file ${TAG}_postplace_timing.rpt

puts "STEP: phys_opt_design -directive AggressiveExplore (post-place)"
phys_opt_design -directive AggressiveExplore
write_checkpoint -force ${TAG}_postplace_physopt.dcp
report_timing_summary -file ${TAG}_postplace_physopt_timing.rpt

puts "STEP: route_design -directive Default -tns_cleanup"
route_design -directive Default -tns_cleanup
write_checkpoint -force ${TAG}_routed_pre_physopt.dcp
report_timing_summary -file ${TAG}_routed_pre_physopt_timing.rpt
puts "STEP: phys_opt_design -directive AggressiveExplore (post-route)"
if {[catch {phys_opt_design -directive AggressiveExplore} e]} { puts "WARNING: post-route phys_opt failed ($e); keeping the routed design" }
# Put the true requirement (one RFDAC2_CLK period) back so the checkpoint and every sign-off report use it. Vivado
# has no command that deletes one exception, so -reset_path replaces it with the period. Replicas made by phys_opt
# share the name prefix, so the cells are looked up again here.
if {[catch {
    set blank_cells [get_cells -hier -filter {NAME =~ *dac_gate_0/inst/blank_reg* && IS_SEQUENTIAL}]
    set_max_delay -reset_path -from $dclk -to $blank_cells [get_property PERIOD $dclk]
    set bp [get_timing_paths -setup -max_paths 1 -to $blank_cells]
    puts "BLANK_TIMING: [llength $blank_cells] register(s); true slack [get_property SLACK $bp] requirement [get_property REQUIREMENT $bp] ([get_property STARTPOINT_PIN $bp] -> [get_property ENDPOINT_PIN $bp])"
} e]} { puts "WARNING: blank-path reset or report failed ($e)" }
write_checkpoint -force ${TAG}_routed.dcp
report_timing_summary -file ${TAG}_routed_timing.rpt
report_utilization -file ${TAG}_routed_utilization.rpt
report_utilization -hierarchical -file ${TAG}_routed_util_hier.rpt
catch {report_utilization -pblocks pb_live -file ${TAG}_routed_pblock_util.rpt}
report_bus_skew -file ${TAG}_routed_bus_skew.rpt
report_exceptions -ignored -file ${TAG}_routed_exceptions_ignored.rpt
if {[catch {
    set bsk [report_bus_skew -return_string]
    set viol [regexp -all -nocase {VIOLATED} $bsk]
    puts "BUSSKEW_ROUTED: live_res_cdc mentioned [regexp -all {live_res_cdc_[01]} $bsk] time(s); VIOLATED x$viol"
} e]} { puts "WARNING: bus-skew summary failed ($e)" }
report_timing -from [get_cells -hier -filter {NAME =~ *NN_live_0/*}] -max_paths 20 -file ${TAG}_routed_NN_live_paths.rpt
report_timing -to [get_cells -hier -filter {NAME =~ *NN_live_0/*}] -max_paths 20 -file ${TAG}_routed_NN_live_in_paths.rpt
report_timing -from [get_cells -hier -filter {NAME =~ *seg_stamp_0/*}] -max_paths 20 -file ${TAG}_routed_seg_paths.rpt
report_timing -through [get_cells -hier -filter {NAME =~ *live_res_cdc_0/*}] -max_paths 20 -file ${TAG}_routed_live_res_cdc_paths.rpt
report_timing -through [get_cells -hier -filter {NAME =~ *dac_gate_0/*}] -max_paths 30 -file ${TAG}_routed_dac_gate_paths.rpt
report_timing -through [get_cells -hier -filter {NAME =~ *live_res_cdc_1/*}] -max_paths 20 -file ${TAG}_routed_live_res_cdc1_paths.rpt
if {[catch {
    set gp [get_timing_paths -group [get_clocks RFDAC2_CLK] -max_paths 1 -setup]
    set gg [get_timing_paths -through [get_cells -hier -filter {NAME =~ *dac_gate_0/*}] -max_paths 1 -setup]
    puts "DAC2_CLK_TIMING: RFDAC2_CLK WNS [get_property SLACK $gp]; worst through dac_gate_0 [get_property SLACK $gg]"
} e]} { puts "WARNING: DAC2 timing summary failed ($e)" }
set cclk [get_clocks -of_objects [get_pins -hier -filter {NAME =~ OG_build_i/clk_core/clk_out1}]]
puts "tProc core clock: $cclk, period [get_property PERIOD $cclk] ns"
report_timing -group $cclk -max_paths 30 -nworst 1 -file ${TAG}_routed_core_clk_paths.rpt
report_timing -from [get_cells -hier -filter {NAME =~ *qick_processor_0/*}] -to [get_cells -hier -filter {NAME =~ *qick_processor_0/*}] -group $cclk -max_paths 20 -file ${TAG}_routed_tproc_core_paths.rpt
set aclk [get_clocks clk_out1_OG_build_clk_adc0_x2_0]
report_cdc -from $aclk -to $cclk -details -file ${TAG}_routed_cdc_adc0x2_to_core.rpt
if {[catch {report_cdc -from $aclk -to [get_clocks RFDAC2_CLK] -details -file ${TAG}_routed_cdc_adc0x2_to_dac2.rpt} e]} { puts "WARNING: report_cdc to RFDAC2_CLK failed ($e)" }
if {[catch {
    set wp [get_timing_paths -group $cclk -max_paths 1 -setup]
    set wh [get_timing_paths -group $cclk -max_paths 1 -hold]
    puts "CORE_CLK_TIMING: period [get_property PERIOD $cclk] WNS [get_property SLACK $wp] WHS [get_property SLACK $wh]"
} e]} { puts "WARNING: core clock timing summary failed ($e)" }

report_timing -through [get_cells -hier -filter {NAME =~ *boxcar_tap_0/*}] -max_paths 30 -file ${TAG}_routed_boxcar_paths.rpt
report_timing -through [get_cells -hier -filter {NAME =~ *rf_tap_0/*}] -max_paths 20 -file ${TAG}_routed_rf_tap_paths.rpt
report_timing -through [get_cells -hier -filter {NAME =~ *trig_sync_0/*}] -max_paths 20 -file ${TAG}_routed_trig_sync_paths.rpt
if {[catch {
    set rp [get_timing_paths -through [get_cells -hier -filter {NAME =~ *rf_tap_0/*}] -max_paths 1 -setup]
    set tp [get_timing_paths -from [get_cells -hier -filter {NAME =~ *trig_sync_0/*}] -max_paths 1 -setup]
    set np [get_timing_paths -through [get_cells -hier -filter {NAME =~ *NN_live_0/*}] -max_paths 1 -setup]
    puts "V141_TIMING: rf_tap_0 [get_property SLACK $rp]; trig_sync_0 out [get_property SLACK $tp]; NN_live_0 [get_property SLACK $np]"
} e]} { puts "WARNING: v1.41 timing summary failed ($e)" }
if {[catch {
    set bp [get_timing_paths -through [get_cells -hier -filter {NAME =~ *boxcar_tap_0/*}] -max_paths 1 -setup]
    puts "BOXCAR_TIMING: worst through boxcar_tap_0 [get_property SLACK $bp]"
    set ap [get_timing_paths -setup -max_paths 6000 -nworst 1 -unique_pins -slack_lesser_than 0 -group $aclk]
    array set nf {}
    foreach p $ap {
        set k [lindex [split [get_property NAME [get_property ENDPOINT_PIN $p]] /] 1]
        if {![info exists nf($k)]} { set nf($k) 0 }
        incr nf($k)
    }
    foreach k [lsort [array names nf]] { puts "ADC0X2_FAIL_SPLIT: $k $nf($k)" }
} e]} { puts "WARNING: boxcar/adc0_x2 summary failed ($e)" }
if {[catch {
    set ap1 [get_timing_paths -setup -max_paths 1 -group $aclk]
    set ah1 [get_timing_paths -hold -max_paths 1 -group $aclk]
    puts "ADC0X2_WNS: [get_property SLACK $ap1] WHS: [get_property SLACK $ah1]"
} e]} { puts "WARNING: adc0_x2 WNS summary failed ($e)" }

report_timing -through [get_cells -hier -filter {NAME =~ *live_rec_wreg_0/*}] -max_paths 20 -file ${TAG}_routed_live_rec_wreg_paths.rpt
report_timing -to [get_cells -hier -filter {NAME =~ *axi_bram_ctrl_live_bram/*}] -max_paths 20 -file ${TAG}_routed_live_bram_in_paths.rpt
report_timing -through [get_cells -hier -filter {NAME =~ *NN_axi_0/*}] -max_paths 20 -file ${TAG}_routed_NN_axi_paths.rpt
if {[catch {
    set wp [get_timing_paths -to [get_cells -hier -filter {NAME =~ *axi_bram_ctrl_live_bram/*}] -max_paths 1 -setup]
    set xp [get_timing_paths -through [get_cells -hier -filter {NAME =~ *NN_axi_0/*}] -max_paths 1 -setup]
    puts "V142_TIMING: worst into live record BRAM [get_property SLACK $wp]; worst through NN_axi_0 [get_property SLACK $xp]"
} e]} { puts "WARNING: v1.42 timing summary failed ($e)" }

puts "STEP: write_bitstream"
write_bitstream -force -file OG_build_${TAG}.bit
puts "BUILD_COMPLETE_${TAG}"

# ---- full report set (FINAL_TASKS 6.2), after the bitstream so a failure here cannot cost the build
foreach {name cmd} {
    methodology        {report_methodology -file ${TAG}_routed_methodology.rpt}
    cdc_summary        {report_cdc -file ${TAG}_routed_cdc_all.rpt}
    cdc_details        {report_cdc -details -file ${TAG}_routed_cdc_all_details.rpt}
    drc                {report_drc -file ${TAG}_routed_drc.rpt}
    clock_interaction  {report_clock_interaction -file ${TAG}_routed_clock_interaction.rpt}
} {
    puts "STEP: report $name"
    if {[catch {eval $cmd} e]} { puts "WARNING: report $name failed ($e)" }
}
puts "REPORTS_COMPLETE_${TAG}"
exit
