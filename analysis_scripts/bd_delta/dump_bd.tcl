# Dumps the open block design as tab-separated lines for comparison: cells and their CONFIG, ports, nets, interface nets,
# address assignments. Usage (with a BD open): dump_bd <outfile>
proc _clean {s} { return [string map [list "\t" " " "\n" "\\n" "\r" ""] $s] }
proc dump_bd {out} {
    set f [open $out w]
    foreach c [lsort [get_bd_cells -hierarchical]] {
        set vlnv [get_property VLNV $c]
        set typ [get_property TYPE $c]
        set ref ""
        catch { set ref [get_property REFERENCE_NAME $c] }
        puts $f "CELL\t$c\t$typ\t$vlnv\t$ref"
        foreach p [lsort [list_property $c CONFIG.*]] {
            puts $f "PROP\t$c\t$p\t[_clean [get_property $p $c]]"
        }
    }
    foreach p [lsort [get_bd_ports]] {
        puts $f "PORT\t$p\t[get_property DIR $p]\t[get_property TYPE $p]\t[get_property LEFT $p]\t[get_property RIGHT $p]"
        foreach k [lsort [list_property $p CONFIG.*]] { puts $f "PPROP\t$p\t$k\t[_clean [get_property $k $p]]" }
    }
    foreach p [lsort [get_bd_intf_ports]] {
        puts $f "IPORT\t$p\t[get_property MODE $p]\t[get_property VLNV $p]"
        foreach k [lsort [list_property $p CONFIG.*]] { puts $f "IPPROP\t$p\t$k\t[_clean [get_property $k $p]]" }
    }
    foreach n [lsort [get_bd_nets -hierarchical]] {
        set ends [lsort [concat [get_bd_pins -quiet -of_objects $n] [get_bd_ports -quiet -of_objects $n]]]
        puts $f "NET\t$n\t[join $ends { }]"
    }
    foreach n [lsort [get_bd_intf_nets -hierarchical]] {
        set ends [lsort [concat [get_bd_intf_pins -quiet -of_objects $n] [get_bd_intf_ports -quiet -of_objects $n]]]
        puts $f "INET\t$n\t[join $ends { }]"
    }
    foreach s [lsort [get_bd_addr_segs -quiet -excluded]] { puts $f "AEXCL\t$s" }
    foreach s [lsort [get_bd_addr_segs]] {
        set off ""; set rng ""
        catch { set off [get_property OFFSET $s]; set rng [get_property RANGE $s] }
        if {$off ne ""} { puts $f "ADDR\t$s\t$off\t$rng" }
    }
    close $f
}
