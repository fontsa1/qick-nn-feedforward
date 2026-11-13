"""
Wait for the board to come back after a reboot, and confirm it actually rebooted (not just a network blip).

The reboot itself is not triggered here: run `sudo reboot` on the board (e.g. from a board notebook cell), then
wait_for_reboot_and_reconnect().

When to reboot instead of restarting the kernel: PYNQ can serve stale cached overlay metadata for a bitstream whose
binary content has not changed, even under a new filename and across a kernel restart. Symptom: a redeployed build's
driver-visible behaviour does not match its .hwh. A reboot clears it.
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
    already triggered elsewhere. The first successful connection may come
    before the server notebook is fully up, so after it the function waits
    settle_delay_s and reconnects once more.

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
        "\nNote: the reboot itself is not verified from the board side "
        "(/proc/uptime); check it there if needed."
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
