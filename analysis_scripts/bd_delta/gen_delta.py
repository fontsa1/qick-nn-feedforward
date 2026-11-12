"""Generate nnqick_bd_delta.tcl: the edit that turns QICK's qick_tprocv2_4x2_standard block design (commit 779dffa)
into our v1.43 block design.

Cell creation and CONFIG come from the two write_bd_tcl scripts (user-set values only). Connectivity comes from the
dump_bd.tcl dumps, compared as endpoint sets. Addresses are copied from our export.
"""
import re, sys
from collections import OrderedDict

qick_tcl, our_tcl, vtsv, otsv, out = sys.argv[1:6]

CHUNK = re.compile(r"\n  # Create instance: (\S+), and set properties\n(.*?)(?=\n  # Create instance: |\n  # Create interface connections|\n  # Create port connections|\n  # Restore current instance)", re.S)

def chunks(path):
    s = open(path, encoding="utf-8").read()
    return OrderedDict((m.group(1), m.group(2)) for m in CHUNK.finditer(s)), s

def cell_config(chunk, var):
    """CONFIG values set on the cell itself (the set_property block that ends in ] $var)."""
    cfg = OrderedDict()
    for m in re.finditer(r"set_property -dict \[\s*list \\\n(.*?)\] \$" + re.escape(var) + r"\b", chunk, re.S):
        for ln in m.group(1).splitlines():
            ln = ln.strip().rstrip("\\").strip()
            if not ln.startswith("CONFIG."):
                continue
            k, v = ln.split(None, 1)
            cfg[k] = v
    for m in re.finditer(r"set_property (CONFIG\.\S+) (\{.*?\}|\S+) \$" + re.escape(var) + r"\b", chunk):
        cfg[m.group(1)] = m.group(2)
    return cfg

def load_tsv(p):
    cells, nets, inets, ports, iports = {}, [], [], set(), set()
    for ln in open(p, encoding="utf-8", errors="replace"):
        t = ln.rstrip("\n").split("\t")
        if t[0] == "CELL": cells[t[1]] = t[2]
        elif t[0] == "NET": nets.append(frozenset((t[2] if len(t) > 2 else "").split()))
        elif t[0] == "INET": inets.append(frozenset((t[2] if len(t) > 2 else "").split()))
        elif t[0] == "PORT": ports.add(t[1])
        elif t[0] == "IPORT": iports.add(t[1])
    return cells, nets, inets, ports, iports

qc, _ = chunks(qick_tcl)
oc, otext = chunks(our_tcl)
vcells, vnets, vinets, vports, viports = load_tsv(vtsv)
ocells, onets, oinets, oports, oiports = load_tsv(otsv)

AUTO = "/ps8_0_axi_periph/"          # axi_interconnect internals are generated from NUM_MI

def auto_internal(path):
    return path.startswith(AUTO) and path.count("/") >= 3

top_v = {c for c in vcells if not c.startswith(AUTO) and (vcells[c] != "hier" or c == "/ddr4")}
top_o = {c for c in ocells if not c.startswith(AUTO) and (ocells[c] != "hier" or c == "/ddr4")}
removed = sorted(top_v - top_o)
added = sorted(top_o - top_v)
assert all(c.count("/") == 1 for c in removed + added), (removed, added)

# path for a chunk name (chunk names are leaf names; unique in this design)
def path_of(name, cells):
    hits = [c for c in cells if c.rsplit("/", 1)[-1] == name and not c.startswith(AUTO)]
    assert len(hits) == 1, (name, hits)
    return hits[0]

notes = []
prop_lines = []
for name, ch in oc.items():
    if name not in qc:
        continue
    p = path_of(name, ocells)
    a, b = cell_config(qc[name], name), cell_config(ch, name)
    changed = OrderedDict((k, v) for k, v in b.items() if a.get(k) != v)
    gone = [k for k in a if k not in b]
    if gone:
        notes.append(f"{p}: set in QICK, not in ours (left at QICK's value unless the dump check says otherwise): {gone}")
    if changed:
        prop_lines.append(f"  # {p}: " + ", ".join(f"{k[7:]} {qc_v} -> {v}" for k, v, qc_v in ((k, v, a.get(k, '(default)')) for k, v in changed.items())))
        prop_lines.append("  set_property -dict [list " + " ".join(f"{k} {v}" for k, v in changed.items()) + f"] [get_bd_cells {p}]")

new_chunks = []
for c in added:
    name = c[1:]
    assert name in oc, name
    new_chunks.append(f"  # Create instance: {name}, and set properties\n" + oc[name].rstrip() + "\n")

def is_new(ep):
    return any(ep == c or ep.startswith(c + "/") for c in added)

def is_removed(ep):
    return any(ep.startswith(c + "/") for c in removed)

def norm(sets):
    out = set()
    for s in sets:
        if any(auto_internal(e) for e in s):
            continue
        s = frozenset(e for e in s if not is_removed(e))
        if len(s) >= 2:
            out.add(s)
    return out

def obj(ep, intf):
    port = ep.count("/") == 1
    if intf:
        return f"[get_bd_intf_ports {ep}]" if port else f"[get_bd_intf_pins {ep}]"
    return f"[get_bd_ports {ep}]" if port else f"[get_bd_pins {ep}]"

conn = []
for intf, V, O in ((True, vinets, oinets), (False, vnets, onets)):
    Vn, On = norm(V), norm(O)
    broken, made = sorted(Vn - On, key=sorted), sorted(On - Vn, key=sorted)
    kind = "intf_net" if intf else "net"
    conn.append(f"  # ---- {'interface' if intf else 'port'} connections: {len(broken)} QICK {kind}s change, {len(made)} {kind}s made")
    dis = sorted({e for s in broken for e in s if not is_new(e)})
    for e in dis:
        o = obj(e, intf)
        conn.append(f"  nnq_disconnect_{'intf' if intf else 'net'} {o}")
    for s in made:
        eps = sorted(s)
        if intf and len(eps) != 2:
            raise SystemExit(f"interface net with {len(eps)} endpoints: {eps}")
        conn.append(f"  connect_bd_{kind} " + " ".join(obj(e, intf) for e in eps))

# addresses: copy our export's assignment and exclusion sections
m = re.search(r"\n  # Create address segments\n(.*?)\n\n", otext, re.S)
addr = m.group(1).splitlines()
m2 = re.search(r"\n  # Exclude Address Segments\n(.*?)\n\n", otext, re.S)
excl = m2.group(1).splitlines() if m2 else []

hdr = f"""#################################################################
# nnqick_bd_delta.tcl: turns QICK's qick_tprocv2_4x2_standard block design (openquantumhardware/qick commit 779dffa,
# firmware/projects/qick_tprocv2_4x2_standard/bd_2023-1.tcl) into the v1.43 block design of the paper.
#
# Generated by analysis_scripts/bd_delta/gen_delta.py from QICK's bd_2023-1.tcl and our write_bd_tcl export of v1.43
# (OG_build_v1.43_bd.tcl). Do not hand-edit; regenerate. Sourced by nnqick_from_qick.tcl, which opens the QICK
# project, adds our RTL and HLS cores, and calls nnqick_apply_delta on the open block design.
#
# Removed from QICK's design: {', '.join(c[1:] for c in removed)}
# Added: {', '.join(c[1:] for c in added)}
# QICK blocks reconfigured: {', '.join(sorted({l.split(':')[0][4:] for l in prop_lines if l.startswith('  # ')}))}
#################################################################
"""
body = [hdr,
        "proc nnq_disconnect_net {o} { set n [get_bd_nets -quiet -of_objects $o]; if {$n ne \"\"} { disconnect_bd_net $n $o } }",
        "proc nnq_disconnect_intf {o} { set n [get_bd_intf_nets -quiet -of_objects $o]; if {$n ne \"\"} { disconnect_bd_intf_net $n $o } }",
        "",
        "proc nnqick_apply_delta {} {",
        "  current_bd_instance /",
        "  if {[llength [get_bd_cells -quiet qick_processor_0]] != 1 || [llength [get_bd_cells -quiet NN_live_0]] != 0} {",
        "    error \"nnqick_apply_delta: the open block design is not an unmodified qick_tprocv2_4x2_standard design\"",
        "  }",
        "",
        "  # ---- 1. QICK blocks this design does not use (time tagger and its DMA path)",
        "  delete_bd_objs [get_bd_cells {" + " ".join(c[1:] for c in removed) + "}]",
        "",
        "  # ---- 2. QICK blocks reconfigured",
        *prop_lines,
        "",
        "  # ---- 3. added blocks (from the v1.43 export, unchanged)",
        *new_chunks,
        *conn,
        "",
        "  # ---- address map (v1.43 assignment, from the export)",
        *[f"  {l.strip()}" for l in addr],
        *[f"  if {{[catch {{{l.strip()}}} e]}} {{ puts \"NOTE: $e\" }}" for l in excl],
        "  return 0",
        "}",
        ""]
open(out, "w", encoding="utf-8", newline="\n").write("\n".join(body))
print(f"removed {len(removed)}, added {len(added)}, prop edits {sum(1 for l in prop_lines if l.startswith('  set_'))}, "
      f"connection lines {len(conn)}, address lines {len(addr)}+{len(excl)}")
for n in notes:
    print("NOTE", n)
