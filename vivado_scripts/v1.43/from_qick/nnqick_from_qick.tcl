#################################################################
# Builds the v1.43 project of the paper from QICK's own qick_tprocv2_4x2_standard project.
#
#   1. Build QICK's project as QICK documents it (openquantumhardware/qick at commit 779dffa):
#        cd firmware/projects/qick_tprocv2_4x2_standard
#        vivado -mode batch -source proj.tcl
#      This makes top/top.xpr with block design d_1. Building QICK's bitstream is not needed.
#   2. From this repository's root:
#        vivado -mode batch -source vivado_scripts/v1.43/from_qick/nnqick_from_qick.tcl \
#               -tclargs <path to QICK's top/top.xpr> <new project folder>
#
# The QICK project is not modified: it is saved as a new project in <new project folder>, and the changes go there:
#   - our two HLS cores (NN_live 1.5, NN_axi 1.41) are installed from their packaged zips into <new project folder>/nnqick_ip
#   - our RTL (module references in the block design) and the CDC constraint are added
#   - nnqick_bd_delta.tcl edits the block design: removes the time tagger, reconfigures six QICK blocks, adds our blocks,
#     rewires, and sets the v1.43 address map.
# Then implement with vivado_scripts/v1.43/build_v1.43.tcl (see the README for the two names it expects).
#################################################################
# Everything runs inside a proc: global variables in a sourced script can collide with Vivado's own BD scripts.
proc nnq_main {here argv} {
    set repo [file normalize $here/../../..]
    if {[llength $argv] != 2} { puts "usage: -tclargs <QICK top.xpr> <new project folder>"; exit 1 }
    lassign $argv qick_xpr outdir
    set outdir [file normalize $outdir]
    if {[file exists $outdir/nnqick.xpr]} { puts "FATAL: $outdir/nnqick.xpr already exists"; exit 1 }

    puts "STEP: opening QICK project $qick_xpr"
    open_project $qick_xpr
    set bds [lsearch -all -inline -not [get_files -quiet -of_objects [get_filesets sources_1] *.bd] */ip/*]   ;# top-level design only, not IP-internal BDs
    if {[llength $bds] != 1} { puts "FATAL: expected one block design in the QICK project, found: $bds"; exit 1 }
    puts "STEP: saving it as $outdir/nnqick.xpr (the QICK project is left as it was)"
    save_project_as nnqick $outdir
    # Open the block design now: in our tests, opening it after the IP install and RTL additions below failed inside
    # Vivado 2023.1's own block-automation scripts ("couldn't read file"), while opening it first works.
    set bds [lsearch -all -inline -not [get_files -quiet -of_objects [get_filesets sources_1] *.bd] */ip/*]
    open_bd_design [lindex $bds 0]

    puts "STEP: installing NN_live 1.5 and NN_axi 1.41"
    set iprepo $outdir/nnqick_ip
    file mkdir $iprepo
    set src [get_filesets sources_1]
    set_property ip_repo_paths [concat [get_property ip_repo_paths $src] [list $iprepo]] $src
    update_ip_catalog -rebuild
    foreach z {neural_net4_live576s/NN_live_prj/solution1/impl/ip/xilinx_com_hls_NN_live_1_5.zip
               neural_net4_tlast/NN_axi_prj/solution1/impl/ip/xilinx_com_hls_NN_axi_1_41.zip} {
        update_ip_catalog -add_ip $repo/$z -repo_path $iprepo
    }
    update_ip_catalog -rebuild
    foreach v {xilinx.com:hls:NN_live:1.5 xilinx.com:hls:NN_axi:1.41} {
        if {[llength [get_ipdefs -quiet $v]] != 1} { puts "FATAL: $v not in the IP catalog after install"; exit 1 }
    }

    puts "STEP: adding our RTL and constraint"
    set rtl {}
    foreach f {neural_net4_live/rtl/live_window_gate.v neural_net4_live576/rtl/trig_sync.v neural_net4_live576/rtl/rf_tap.v
               neural_net4_live576/rtl/dac_gate.v neural_net4_live576/rtl/boxcar_tap.v neural_net4_live576/rtl/live_rec_wreg.v
               neural_net4_live576/rtl/live_res_cdc.v neural_net4_live576/rtl/seg_stamp.v} {
        if {![file exists $repo/$f]} { puts "FATAL: $repo/$f not found"; exit 1 }
        lappend rtl $repo/$f
    }
    add_files -norecurse -fileset $src $rtl
    update_compile_order -fileset sources_1
    set xdc $repo/neural_net4_live576/constraints/live_res_cdc.xdc
    add_files -norecurse -fileset constrs_1 $xdc
    set xf [get_files $xdc]
    set_property USED_IN_SYNTHESIS false $xf
    set_property USED_IN_IMPLEMENTATION true $xf
    set_property SCOPED_TO_REF live_res_cdc $xf
    set_property PROCESSING_ORDER LATE $xf

    # In our Vivado 2023.1 tests, file reads sometimes failed for a short time after the RTL was added ("couldn't read
    # file" for files that exist), inside our script and inside Vivado's own. Wait until a read works again.
    for {set i 0} {$i < 60} {incr i} {
        if {![catch {set fh [open $here/nnqick_bd_delta.tcl r]; close $fh} e]} { break }
        puts "WAIT: file read failed ($::errorCode); retrying"
        after 2000
    }
    after 5000
    puts "STEP: editing the block design"
    uplevel #0 [list source $here/nnqick_bd_delta.tcl]
    if {[nnqick_apply_delta] != 0} { puts "FATAL: nnqick_apply_delta failed"; exit 1 }
    validate_bd_design -force
    save_bd_design
    close_bd_design [current_bd_design]
    close_project
    puts "NNQICK_PROJECT_DONE $outdir/nnqick.xpr"
}
nnq_main [file dirname [file normalize [info script]]] $argv
