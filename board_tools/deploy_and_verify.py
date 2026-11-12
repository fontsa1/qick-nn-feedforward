"""
Deploy a new bitstream to the board and verify it's actually live.

STATUS (2026-09-01 night): the upload/restart automation this docstring
used to call an "unconfirmed gap" is now REAL and TESTED --
`board_tools/jupyter_remote.py` implements it against the board's classic
Jupyter Notebook server (Tornado, cookie+XSRF auth, password "xilinx")
via its REST contents/kernels API plus the kernel-channels websocket for
code execution. Validated tonight end-to-end (login, file upload, kernel
restart actually clearing state, code execution with error capture) using
a disposable scratch kernel -- NOT yet exercised against the real
`start nameserver.ipynb` kernel, since that requires an actual new
bitstream to be worth the disruption. See jupyter_remote.py's own
docstring for the "first real use" checklist before pointing this at the
live server for the first time.

`deploy_new_bitstream()` below wires it all together: upload .bit/.hwh,
patch the notebook's bitfile-load cell, restart the kernel, re-run every
code cell in order (Restart & Run All semantics), leaving the final
`daemon.requestLoop()` cell running (fire-and-don't-wait, since it never
returns by design). Then call `verify_deployment(...)` as before.

"""
import argparse
import re
import sys
import time

import Pyro4

from jupyter_remote import (
    login,
    list_sessions,
    upload_file,
    get_notebook,
    put_notebook,
    restart_kernel,
    execute_in_kernel,
)

Pyro4.config.SERIALIZER = "pickle"
Pyro4.config.PICKLE_PROTOCOL_VERSION = 4

from qick import QickConfig  # noqa: E402

NS_HOST = __import__("os").environ.get("QICK_NS_HOST", "10.1.9.26")  # board B: 10.1.4.105
NS_PORT = 8888
PROXY_NAME = "myqick"

def connect(max_attempts=30, retry_delay_s=2.0):
    """Connect to the board's QICK Pyro4 server, retrying the WHOLE
    lookup (including locateNS) on every attempt -- not just the
    Proxy() call. A prior version of this pattern called locateNS()
    once, outside the retry loop, and hit a real race: right after a
    bitfile swap + kernel restart, the nameserver lookup can fail with
    ConnectionRefusedError if the board hasn't finished booting yet. See
    CLASSIFIER_DEBUGGING_REFERENCE.md #15 for the history of this bug.
    """
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            ns = Pyro4.locateNS(host=NS_HOST, port=NS_PORT)
            soc = Pyro4.Proxy(ns.lookup(PROXY_NAME))
            soccfg = QickConfig(soc.get_cfg())
            return soc, soccfg
        except Exception as e:  # noqa: BLE001 - deliberately broad, this is a connect-retry loop
            last_err = e
            print(f"[connect] attempt {attempt}/{max_attempts} failed: {e!r}")
            time.sleep(retry_delay_s)
    raise RuntimeError(
        f"Could not connect to {PROXY_NAME} after {max_attempts} attempts"
    ) from last_err

def verify_deployment(expect_banner_contains=None, onset_threshold=None,
                       max_connect_attempts=30):
    """Connect, print the firmware banner, and optionally verify it
    against an expected substring (e.g. the build timestamp you know you
    just deployed -- cross-check this against the .hwh file's own mtime
    on the build machine before trusting it, per the workflow doc).

    If onset_threshold is given, sets it via nn_configure-adjacent API
    and reads it back to confirm -- this resets to 0 on every fresh
    deploy and is NOT preserved (see CLASSIFIER_DEBUGGING_REFERENCE.md #7
    for the cost of forgetting this).
    """
    print(f"Connecting to {PROXY_NAME} at {NS_HOST}:{NS_PORT} ...")
    soc, soccfg = connect(max_attempts=max_connect_attempts)

    banner = str(soccfg)
    print("=== Firmware banner ===")
    print(banner)
    print("========================")

    if expect_banner_contains is not None:
        if expect_banner_contains not in banner:
            print(
                f"\n*** MISMATCH: expected banner to contain "
                f"{expect_banner_contains!r}, but it did not. ***\n"
                "This usually means either (a) the wrong build got "
                "uploaded, (b) the kernel didn't actually restart onto "
                "the new bitfile, or (c) PYNQ is serving stale cached "
                "overlay metadata for a byte-identical .bit -- see "
                "CLASSIFIER_DEBUGGING_REFERENCE.md #11. A full board "
                "reboot (not just a kernel restart) fixes (c)."
            )
            return False
        print(f"Banner check passed: contains {expect_banner_contains!r}")

    if onset_threshold is not None:
        # Raw MMIO is the reliable path for this register: PYNQ's register_map
        # ctypes getter crashes on some AXI-Lite registers, including this one.
        # Used when nn_set_onset_threshold()/nn_get_onset_threshold() are not
        # present on the NNQickSoc class the board's server is running.
        try:
            soc.nn_set_onset_threshold(onset_threshold)
            readback = soc.nn_get_onset_threshold()
        except AttributeError:
            print(
                "*** soc has no nn_set_onset_threshold/nn_get_onset_threshold "
                "-- check the currently-deployed NNQickSoc class definition "
                "(start nameserver.ipynb cell 0) for the current method "
                "names before assuming this API is missing. ***"
            )
            return False
        if readback != onset_threshold:
            print(
                f"*** onset_threshold readback mismatch: set {onset_threshold}, "
                f"read back {readback}. ***"
            )
            return False
        print(f"onset_threshold confirmed set to {onset_threshold}")

    print("\nDeployment verification PASSED.")
    return True

NAMESERVER_NB_PATH = "start nameserver.ipynb"

def deploy_new_bitstream(local_bit_path, local_hwh_path, board_name_stem):
    """Real, tested (as of tonight) implementation of the manual deploy
    procedure from qick-board-workflow-pipeline memory, steps 1-4:
    upload .bit/.hwh, patch the notebook, restart, re-run all cells.

    board_name_stem: e.g. "qick34" -- files go up as qick34.bit/qick34.hwh.
    """
    session, xsrf = login()
    print("Logged in to board Jupyter server.")

    bit_board_path = f"{board_name_stem}.bit"
    hwh_board_path = f"{board_name_stem}.hwh"
    upload_file(session, xsrf, local_bit_path, bit_board_path)
    upload_file(session, xsrf, local_hwh_path, hwh_board_path)

    nb = get_notebook(session, xsrf, NAMESERVER_NB_PATH)
    content = nb["content"]
    cells = content["cells"]

    # Confirmed by direct inspection (2026-09-01 night): the REAL notebook
    # is only 3 cells -- cell 0 alone does everything (BD 2.0 compat patch,
    # NNQickSoc class def, starts its own `pyro4-ns` nameserver subprocess,
    # then calls the blocking start_server(bitfile=..., ...)). Cell 1 is
    # inactive leftover code (an alternate `start_nameserver()` call that
    # would conflict -- port already in use -- if run after cell 0 already
    # started its own nameserver subprocess). Cell 2 is empty. So: patch
    # and re-run ONLY cell 0, never cell 1. Do NOT assume this generalizes
    # to "every code cell in order" without re-checking if this notebook's
    # structure ever changes.
    bitfile_cell_idx = None
    for i, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        if "bitfile" in src and "start_server(" in src:
            bitfile_cell_idx = i
            break
    if bitfile_cell_idx is None:
        raise RuntimeError(
            "Could not find the cell containing both 'bitfile' and "
            f"'start_server(' in {NAMESERVER_NB_PATH} -- its structure "
            "has apparently changed since this was last verified "
            "(2026-09-01 night). Inspect it manually before proceeding, "
            "don't guess."
        )

    old_src = "".join(cells[bitfile_cell_idx]["source"]) if isinstance(
        cells[bitfile_cell_idx]["source"], list) else cells[bitfile_cell_idx]["source"]
    print(f"--- current bitfile-load cell (index {bitfile_cell_idx}) ---")
    print(old_src)

    # Actual observed format: bitfile = '/home/xilinx/jupyter_notebooks/qick33.bit'
    # -- replace only the trailing filename, preserve the directory prefix.
    new_src, n_sub = re.subn(
        r"(bitfile\s*=\s*['\"])([^'\"]*/)?[\w.]+\.bit(['\"])",
        rf"\g<1>\g<2>{bit_board_path}\g<3>",
        old_src,
    )
    if n_sub == 0:
        raise RuntimeError(
            "Regex substitution matched nothing in the bitfile-load cell "
            "-- the notebook's actual 'bitfile = ...' line differs from "
            "what was last observed. Inspect old_src above and fix the "
            "regex before proceeding, rather than deploying blind."
        )
    print(f"--- new bitfile-load cell ---\n{new_src}")
    cells[bitfile_cell_idx]["source"] = new_src.splitlines(keepends=True)

    put_notebook(session, xsrf, NAMESERVER_NB_PATH, content)
    print(f"Patched {NAMESERVER_NB_PATH} to reference {bit_board_path}")

    sessions = list_sessions(session, xsrf)
    matches = [s for s in sessions if s["path"] == NAMESERVER_NB_PATH]
    if not matches:
        raise RuntimeError(
            f"No running session found for {NAMESERVER_NB_PATH} -- "
            "is the nameserver notebook actually open on the board?"
        )
    kernel_id = matches[0]["kernel"]["id"]
    print(f"Found kernel {kernel_id} for {NAMESERVER_NB_PATH}")

    restart_kernel(session, xsrf, kernel_id)
    print("Kernel restarted (all prior state cleared).")
    time.sleep(2)  # let the fresh kernel finish coming up before executing into it

    print(f"\n=== executing cell {bitfile_cell_idx} (the only cell that needs "
          f"to run -- contains the blocking start_server() call, so NOT "
          f"waiting for idle) ===")
    result = execute_in_kernel(session, xsrf, kernel_id, new_src,
                                wait_for_idle=False, timeout_s=30)
    if result["error"] is not None:
        raise RuntimeError(
            f"Cell raised an immediate error on redeploy: {result['error']}"
        )
    if not result["seen_busy"]:
        raise RuntimeError(
            "Never saw the kernel go 'busy' after firing the redeploy cell "
            "-- something is wrong (kernel didn't accept the execute "
            "request), don't assume the server is coming up."
        )
    print("Cell accepted and running (kernel busy, no immediate error). "
          "This blocks forever by design -- not waiting further.")

    print("\nDeploy sequence complete. Now call verify_deployment(...) "
          "with real retries -- a just-restarted server needs a few seconds.")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--deploy", nargs=3, metavar=("BIT_PATH", "HWH_PATH", "BOARD_NAME_STEM"),
        default=None,
        help="Upload+deploy a new bitstream before verifying, e.g. "
             "--deploy OG_build_v1.23.bit OG_build.hwh qick34",
    )
    parser.add_argument(
        "--expect-banner-contains", default=None,
        help="Substring the firmware banner should contain (e.g. a build timestamp)",
    )
    parser.add_argument(
        "--onset-threshold", type=int, default=None,
        help="Value to set and verify for onset_threshold after deploy",
    )
    args = parser.parse_args()

    if args.deploy:
        deploy_new_bitstream(*args.deploy)

    ok = verify_deployment(
        expect_banner_contains=args.expect_banner_contains,
        onset_threshold=args.onset_threshold,
    )
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
