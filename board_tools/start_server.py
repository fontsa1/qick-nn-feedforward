"""Bring up the QICK Pyro server on a given board, from a cold kernel.

The `start_server(...)` call in `start nameserver.ipynb` blocks forever, so the
notebook's kernel sits "busy" while the board is usable and "idle" when it is
not. A board whose kernel reads idle has no nameserver, and every Pyro client
will fail with "Failed to locate the nameserver" / WinError 10061.

--restart shuts down a running server's kernel first (needed after a deploy when
the server runs in a kernel with no notebook session, so deploy cannot restart it).
"""
import sys
import time

sys.path.insert(0, "board_tools")
import jupyter_remote as jr
import Pyro4

Pyro4.config.SERIALIZER = "pickle"
Pyro4.config.PICKLE_PROTOCOL_VERSION = 4

SERVER_NB = "start nameserver.ipynb"

def start(host, timeout=300, restart=False):
    jr.BOARD_HOST = host
    jr.BASE = f"http://{host}:{jr.JUPYTER_PORT}"
    jr.WS_BASE = f"ws://{host}:{jr.JUPYTER_PORT}"
    s, x = jr.login()
    print(f"[{host}] jupyter login OK")

    # already up?
    try:
        if restart:
            raise RuntimeError("restart requested")
        ns = Pyro4.locateNS(host=host, port=8888)
        soc = Pyro4.Proxy(ns.lookup("myqick"))
        print(f"[{host}] already up -> {soc.nn_status()}")
        return soc
    except Exception:
        print(f"[{host}] {'restart requested' if restart else 'no nameserver'}; cold-starting")

    r = s.get(f"{jr.BASE}/api/kernels", headers=jr._headers(x))
    for k in r.json():
        print(f"[{host}] shutting down kernel {k['id'][:8]}")
        try:
            jr.shutdown_kernel(s, x, k["id"])
        except Exception as e:
            print("   ", e)
    time.sleep(3)

    nb = jr.get_notebook(s, x, SERVER_NB)
    cell_src = None
    for c in nb["content"]["cells"]:
        src = "".join(c["source"])
        if "bitfile" in src and "start_server(" in src:
            cell_src = src
            break
    if cell_src is None:
        raise SystemExit(f"[{host}] could not find the start_server cell")
    for line in cell_src.splitlines():
        if line.strip().startswith(("bitfile", "ns_host")):
            print(f"[{host}]   {line.strip()}")

    kid = jr.start_kernel(s, x)["id"]
    print(f"[{host}] started kernel {kid[:8]}")
    time.sleep(2)
    jr.execute_in_kernel(s, x, kid, cell_src, wait_for_idle=False)
    print(f"[{host}] start_server dispatched; waiting up to {timeout}s ...")

    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            ns = Pyro4.locateNS(host=host, port=8888)
            soc = Pyro4.Proxy(ns.lookup("myqick"))
            st = soc.nn_status()
            print(f"[{host}] UP after {time.time()-t0:.0f}s -> {st}")
            return soc
        except Exception:
            time.sleep(3)
    raise SystemExit(f"[{host}] did not come up within {timeout}s")

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--restart"]
    host = args[0] if args else "10.1.9.26"
    start(host, restart="--restart" in sys.argv[1:])
