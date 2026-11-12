"""Compare two dump_bd.tcl outputs. Cell paths compared as-is (BD name is not in them)."""
import sys
from collections import defaultdict


def load(p):
    d = dict(cell={}, prop=defaultdict(dict), port={}, pprop=defaultdict(dict), iport={}, ipprop=defaultdict(dict),
             net={}, inet={}, addr={}, aexcl=set())
    for ln in open(p, encoding="utf-8", errors="replace"):
        t = ln.rstrip("\n").split("\t")
        k = t[0]
        if k == "CELL": d["cell"][t[1]] = tuple(t[2:])
        elif k == "PROP": d["prop"][t[1]][t[2]] = t[3] if len(t) > 3 else ""
        elif k == "PORT": d["port"][t[1]] = tuple(t[2:])
        elif k == "PPROP": d["pprop"][t[1]][t[2]] = t[3] if len(t) > 3 else ""
        elif k == "IPORT": d["iport"][t[1]] = tuple(t[2:])
        elif k == "IPPROP": d["ipprop"][t[1]][t[2]] = t[3] if len(t) > 3 else ""
        elif k == "NET": d["net"][t[1]] = frozenset((t[2] if len(t) > 2 else "").split())
        elif k == "INET": d["inet"][t[1]] = frozenset((t[2] if len(t) > 2 else "").split())
        elif k == "ADDR": d["addr"][t[1]] = (t[2], t[3])
        elif k == "AEXCL": d["aexcl"].add(t[1])
    return d


def main(a, b, verbose=True):
    A, B = load(a), load(b)
    out = []
    pr = out.append
    ca, cb = set(A["cell"]), set(B["cell"])
    for c in sorted(ca - cb): pr(f"CELL removed  {c} {A['cell'][c]}")
    for c in sorted(cb - ca): pr(f"CELL added    {c} {B['cell'][c]}")
    for c in sorted(ca & cb):
        if A["cell"][c] != B["cell"][c]: pr(f"CELL changed  {c} {A['cell'][c]} -> {B['cell'][c]}")
        pa, pb = A["prop"][c], B["prop"][c]
        for k in sorted(set(pa) | set(pb)):
            if pa.get(k) != pb.get(k): pr(f"PROP {c} {k}: {pa.get(k)!r} -> {pb.get(k)!r}")
    for kind in ("port", "iport"):
        x, y = A[kind], B[kind]
        for p in sorted(set(x) | set(y)):
            if x.get(p) != y.get(p): pr(f"{kind.upper()} {p}: {x.get(p)} -> {y.get(p)}")
        px, py = A[kind[0:-4] + "pprop" if kind == "port" else "ipprop"], B["pprop" if kind == "port" else "ipprop"]
        px = A["pprop" if kind == "port" else "ipprop"]
        for p in sorted(set(px) | set(py)):
            for k in sorted(set(px[p]) | set(py[p])):
                if px[p].get(k) != py[p].get(k): pr(f"{kind.upper()}PROP {p} {k}: {px[p].get(k)!r} -> {py[p].get(k)!r}")
    for kind in ("net", "inet"):
        sa, sb = set(A[kind].values()), set(B[kind].values())
        for s in sorted(sa - sb, key=sorted): pr(f"{kind.upper()} removed {' '.join(sorted(s))}")
        for s in sorted(sb - sa, key=sorted): pr(f"{kind.upper()} added   {' '.join(sorted(s))}")
    for s in sorted(set(A["addr"]) | set(B["addr"])):
        if A["addr"].get(s) != B["addr"].get(s): pr(f"ADDR {s}: {A['addr'].get(s)} -> {B['addr'].get(s)}")
    for s in sorted(A["aexcl"] ^ B["aexcl"]): pr(f"AEXCL {s} {'removed' if s in A['aexcl'] else 'added'}")
    if verbose:
        print("\n".join(out))
    return out


if __name__ == "__main__":
    r = main(sys.argv[1], sys.argv[2])
    print(f"# {len(r)} differences", file=sys.stderr)
