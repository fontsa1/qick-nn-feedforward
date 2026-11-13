# Out-of-context timing sweep of N classifier units.
# Usage: vivado -mode batch -source run_ooc.tcl -tclargs <N> <pb|free> [<N> <pb|free> ...]
# Each pair: synth ooc_top with NU=N (out_of_context), optional pblock pb_live (CLOCKREGION_X4Y3:X5Y4, next to the
# ADC tiles, as in the deployed build), then opt / place (Default) / phys_opt (AggressiveExplore) / route (Default).
# Writes runs/N<N>_<pb>/ : utilization, timing summary, failing-endpoint list, bus skew, SUMMARY line.
set HERE [file normalize [file dirname [info script]]]
set PART xczu48dr-ffvg1517-2-e
set HLS  [file normalize "$HERE/../neural_net4_live576s/NN_live_prj/solution1/impl/verilog"]
set RTL  [file normalize "$HERE/../neural_net4_live576/rtl"]
set_param general.maxThreads 4

proc run_one {N mode} {
    global HERE PART HLS RTL
    set out "$HERE/runs/N${N}_${mode}"
    file mkdir $out
    cd $out
    # HLS ROMs use $readmemh("./<file>.dat"): copy them into the run directory
    foreach f [glob $HLS/*.dat] { file copy -force $f $out }
    puts "STEP: N=$N mode=$mode synth [clock format [clock seconds]]"
    close_project -quiet
    create_project -in_memory -part $PART
    foreach f [glob $HLS/*.v] {
        if {[string match "*deadlock*" $f]} continue
        read_verilog $f
    }
    foreach m {boxcar_tap trig_sync live_window_gate live_res_cdc dac_gate} { read_verilog "$RTL/$m.v" }
    read_verilog "$HERE/rtl/live_record_ram.v"
    read_verilog "$HERE/rtl/classifier_unit.v"
    read_verilog "$HERE/rtl/ooc_top.v"
    read_xdc "$HERE/ooc.xdc"
    read_xdc -ref live_res_cdc "$RTL/../constraints/live_res_cdc.xdc"
    synth_design -top ooc_top -part $PART -mode out_of_context -generic NU=$N -flatten_hierarchy rebuilt
    set units [get_cells -quiet {g[*].u_unit}]
    puts "UNITS: [llength $units]"
    if {[llength $units] != $N} { puts "FATAL: expected $N units, found [llength $units]"; return }
    set_property DONT_TOUCH true $units
    report_utilization -hierarchical -hierarchical_depth 3 -file util_synth_hier.rpt
    if {$mode eq "pb"} {
        create_pblock pb_live
        resize_pblock pb_live -add {CLOCKREGION_X4Y3:CLOCKREGION_X5Y4}
        add_cells_to_pblock pb_live $units
    }
    puts "STEP: opt/place [clock format [clock seconds]]"
    opt_design
    place_design
    report_timing_summary -max_paths 10 -file timing_placed.rpt
    puts "STEP: phys_opt [clock format [clock seconds]]"
    phys_opt_design -directive AggressiveExplore
    puts "STEP: route [clock format [clock seconds]]"
    route_design
    puts "STEP: reports [clock format [clock seconds]]"
    report_utilization -file util_routed.rpt
    report_utilization -hierarchical -hierarchical_depth 3 -file util_routed_hier.rpt
    if {$mode eq "pb"} { catch {report_utilization -pblocks pb_live -file util_pblock.rpt} }
    report_timing_summary -max_paths 20 -report_unconstrained -file timing_routed.rpt
    catch {report_bus_skew -warn_on_violation -file bus_skew.rpt}

    # failing setup endpoints, split into unit-internal vs harness (start or end outside g[*].u_unit)
    set paths [get_timing_paths -setup -max_paths 100000 -nworst 1 -slack_lesser_than 0]
    set fh [open failing_endpoints.csv w]
    puts $fh "slack,group,class,startpoint,endpoint"
    set n_unit 0; set n_harn 0; set wu 1e9; set wh 1e9
    foreach p $paths {
        set s [get_property SLACK $p]
        set sp [get_property STARTPOINT_PIN $p]; set ep [get_property ENDPOINT_PIN $p]
        set g [get_property GROUP $p]
        set su [regexp {^g\[\d+\]\.u_unit/} $sp]; set eu [regexp {^g\[\d+\]\.u_unit/} $ep]
        set hreg [regexp {u_unit/(mix_r|mix_v|fir_r|fir_v|cfg_r|dac_r|dac_v)} $ep]
        if {$su && $eu && !$hreg} { set c unit; incr n_unit; if {$s < $wu} {set wu $s} } else { set c harness; incr n_harn; if {$s < $wh} {set wh $s} }
        puts $fh "$s,$g,$c,$sp,$ep"
    }
    close $fh
    set wns [get_property SLACK [get_timing_paths -setup -max_paths 1]]
    set whs [get_property SLACK [get_timing_paths -hold -max_paths 1]]
    set adc [get_timing_paths -setup -max_paths 1 -group clk_adc]
    set wadc [expr {[llength $adc] ? [get_property SLACK $adc] : "na"}]
    set lut [llength [get_cells -hier -filter {REF_NAME =~ LUT*}]]
    set line "SUMMARY N=$N mode=$mode WNS=$wns WHS=$whs WNS_clk_adc=$wadc failing_unit=$n_unit worst_unit=[expr {$n_unit ? $wu : "none"}] failing_harness=$n_harn worst_harness=[expr {$n_harn ? $wh : "none"}] luts=$lut"
    puts $line
    set f [open summary.txt w]; puts $f $line; close $f
    set f [open "$HERE/runs/SUMMARY.txt" a]; puts $f "[clock format [clock seconds]] $line"; close $f
    write_checkpoint -force routed.dcp
}

foreach {N mode} $argv {
    if {[catch {run_one $N $mode} e]} {
        puts "RUN_ERROR N=$N mode=$mode: $e"
        set f [open "$HERE/runs/SUMMARY.txt" a]; puts $f "[clock format [clock seconds]] RUN_ERROR N=$N mode=$mode: $e"; close $f
    }
}
puts "OOC_SWEEP_DONE"
