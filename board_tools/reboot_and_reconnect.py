"""
Wait for the board to come back after a reboot, and confirm it genuinely
rebooted (not just that the network blipped).

STATUS: same split as deploy_and_verify.py -- the WAIT/VERIFY half below
is real, working code. TRIGGERING the reboot itself is not automated here
-- it requires executing a command on the board (`sudo reboot`, root has
passwordless sudo per this project's history), which needs whatever
board-side code-execution mechanism is available at the time (a Jupyter
notebook cell, direct terminal access, etc.) -- not something reachable
through the Pyro4 QICK connection this script otherwise uses.

WHEN TO USE A FULL REBOOT INSTEAD OF JUST A KERNEL RESTART
(see CLASSIFIER_DEBUGGING_REFERENCE.md #11 for the full story):
PYNQ can silently serve stale cached overlay metadata for a bitstream
whose BINARY CONTENT hasn't changed, even under a brand-new filename,
even across a Jupyter kernel restart. Symptom: a redeployed build's
driver-visible behavior doesn't match its just-verified-correct .hwh
content. If you hit that, don't keep re-verifying file bytes or trying
new filenames -- reboot.

MANUAL STEP: trigger the reboot yourself (fire a `sudo reboot` cell in a
board notebook, or however you currently have code-execution access),
THEN run wait_for_reboot_and_reconnect() below to confirm it's back and
genuinely fresh.

"""
import sys
import time

import Pyro4

Pyro4.config.SERIALIZER = "pickle"
Pyro4.config.PICKLE_PROTOCOL_VERSION = 4

from qick import QickConfig  # noqa: E402

NS_HOST = "10.1.9.26"
NS_PORT = 8888
PROXY_NAME = "myqick"

def wait_for_reboot_and_reconnect(
    total_timeout_s=600,
    poll_interval_s=10,
    settle_delay_s=15,
):
    """Poll for the QICK Pyro4 server coming back after a reboot you've
    already triggered elsewhere. A genuine reboot takes real wall-clock
    time (boot + the nameserver/daemon notebook needs to actually start
    running again, matching this project's own precedent -- a confirmed
    real reboot showed ~75s uptime shortly after coming back) -- don't
    treat the FIRST successful connection as necessarily meaning the
    reboot fully completed; a brief settle delay + one more reconnect
    attempt afterward is cheap insurance.

    Returns (soc, soccfg) on success, raises TimeoutError on failure.
    """
    deadline = time.time() + total_timeout_s
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        try:
            ns = Pyro4.locateNS(host=NS_HOST, port=NS_PORT)
            soc = Pyro4.Proxy(ns.lookup(PROXY_NAME))
            soccfg = QickConfig(soc.get_cfg())
            print(f"[wait_for_reboot] connected on attempt {attempt}")
            break
        except Exception as e:  # noqa: BLE001 - deliberately broad, this is a poll loop
            print(f"[wait_for_reboot] attempt {attempt}: not up yet ({e!r})")
            time.sleep(poll_interval_s)
    else:
        raise TimeoutError(
            f"Board did not come back within {total_timeout_s}s of polling"
        )

    print(
        f"Waiting {settle_delay_s}s and reconnecting once more, to avoid "
        "mistaking a partially-up server for a fully-booted one..."
    )
    time.sleep(settle_delay_s)
    ns = Pyro4.locateNS(host=NS_HOST, port=NS_PORT)
    soc = Pyro4.Proxy(ns.lookup(PROXY_NAME))
    soccfg = QickConfig(soc.get_cfg())

    print("=== Firmware banner after reboot ===")
    print(soccfg)
    print("=====================================")
    print(
        "\nNote: this script cannot independently confirm /proc/uptime is "
        "genuinely small (that requires board-side code execution, the "
        "same gap noted in this file's module docstring) -- if you need "
        "that specific confirmation, check it manually via whatever "
        "board-side access you used to trigger the reboot. A successful "
        "reconnect after your own reboot trigger is still strong evidence "
        "on its own."
    )
    return soc, soccfg

def main():
    try:
        wait_for_reboot_and_reconnect()
    except TimeoutError as e:
        print(f"\nFAILED: {e}")
        sys.exit(1)
    print("\nReconnect after reboot: PASSED.")
    sys.exit(0)

if __name__ == "__main__":
    main()
