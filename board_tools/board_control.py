"""
Board control helpers for switching bitstreams between on-chip test runs with no state carried across runs:

  1. `reload_board()` always goes through a kernel restart, which reloads the overlay and the bitstream; after it
     nn_status() reports out_offset=0 and trig_count=0. `nn_reset()` alone does not clear them, so a reload is the
     way to clear state between runs.

  2. `firmware_banner()` returns the build timestamp the running firmware reports; stamp every on-chip result
     with it.
"""
import re
import sys
import time

sys.path.insert(0, __file__.rsplit("\\", 1)[0] if "\\" in __file__ else ".")
import jupyter_remote as jr

import Pyro4
Pyro4.config.SERIALIZER = "pickle"
Pyro4.config.PICKLE_PROTOCOL_VERSION = 4

NS_HOST, NS_PORT, PROXY = "10.1.9.26", 8888, "myqick"
SERVER_NB = "start nameserver.ipynb"


def list_board_bitfiles():
    s, x = jr.login()
    out = []
    for entry in jr.list_contents(s, x, "")["content"]:
        n = entry.get("name", "")
        if n.endswith((".bit", ".hwh")):
            out.append((entry["path"], entry.get("last_modified", "")[:19],
                        entry.get("size")))
    return sorted(out)


def current_bitfile():
    """Read the bitfile path the server notebook is configured to load."""
    s, x = jr.login()
    nb = jr.get_notebook(s, x, SERVER_NB)
    for c in nb["content"]["cells"]:
        src = "".join(c["source"])
        m = re.search(r"bitfile\s*=\s*['\"]([^'\"]+\.bit)['\"]", src)
        if m:
            return m.group(1)
    return None


def set_bitfile(board_bit_path):
    """Rewrite the server notebook's bitfile= line. Preserves the directory
    prefix already present, so only the filename changes."""
    s, x = jr.login()
    nb = jr.get_notebook(s, x, SERVER_NB)
    changed = 0
    for c in nb["content"]["cells"]:
        src = "".join(c["source"])
        if "bitfile" not in src:
            continue
        new, n = re.subn(r"(bitfile\s*=\s*['\"])[^'\"]+\.bit(['\"])",
                         rf"\g<1>{board_bit_path}\g<2>", src)
        if n:
            c["source"] = new.splitlines(keepends=True)
            changed += n
    if not changed:
        raise RuntimeError("no bitfile= line found to rewrite")
    jr.put_notebook(s, x, SERVER_NB, nb["content"])
    return changed


def reload_board(timeout_s=240):
    """Restart the server kernel -> reloads overlay + bitstream -> wait for
    Pyro to come back. Returns the live proxy. This is the ONLY sanctioned
    way to clear NN state between runs."""
    s, x = jr.login()
    sessions = jr.list_sessions(s, x)
    if not sessions:
        raise RuntimeError("no running session on the board")
    kid = sessions[0]["kernel"]["id"]
    nb = jr.get_notebook(s, x, SERVER_NB)
    cell_src = None
    for c in nb["content"]["cells"]:
        src = "".join(c["source"])
        if "bitfile" in src and "start_server(" in src:
            cell_src = src
            break
    if cell_src is None:
        raise RuntimeError("could not find the start_server cell")

    print("  restarting kernel (reloads overlay + bitstream)...")
    jr.restart_kernel(s, x, kid)
    time.sleep(3)
    print("  re-executing start_server cell (never returns, by design)...")
    jr.execute_in_kernel(s, x, kid, cell_src, wait_for_idle=False)

    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            ns = Pyro4.locateNS(host=NS_HOST, port=NS_PORT)
            soc = Pyro4.Proxy(ns.lookup(PROXY))
            st = soc.nn_status()
            print(f"  board up after {time.time()-t0:.0f}s -> {st}")
            if st["out_offset"] != 0 or st["trig_count"] != 0:
                print(f"  ⚠️  WARNING: expected a fully reset core "
                      f"(out_offset=0, trig_count=0), got {st}. "
                      f"State may be carried over from a previous run.")
            return soc
        except Exception:
            time.sleep(3)
    raise TimeoutError(f"board did not come back within {timeout_s}s")


def firmware_banner(soc):
    """Build timestamp + board string the running firmware reports. Stamp
    every on-chip result with this."""
    from qick import QickConfig
    cfg = QickConfig(soc.get_cfg())
    txt = str(cfg)
    m = re.search(r"Firmware configuration \(built ([^)]+)\)", txt)
    board = re.search(r"QICK running on (\S+)", txt)
    return {
        "built": m.group(1) if m else "unknown",
        "board": board.group(1) if board else "unknown",
    }
