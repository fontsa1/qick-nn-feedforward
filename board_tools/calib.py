"""Board-scoped storage for the loopback predistortion coefficients (k, c).

WHY THIS EXISTS (task 0.6, 2026-10-05). `tools/run_calibration.py` historically wrote
bare `calibration/k.npy` / `calibration/c.npy` in the repo root with no record of which board they were
fitted against. The two-board test on 2026-10-04 showed what that costs: the two
RFSoCs differ by **119 degrees** in the phase of `k`, and running board A with
board B's coefficients scores **31%** instead of 94% -- below chance, silently,
with no error message anywhere. A rotated decision boundary does not look like a
configuration mistake, it looks like a broken classifier.

DESIGN. The coefficients live in `calib/k_<host>.npy` / `calib/c_<host>.npy`,
one pair per board. `calibration/k.npy` / `calibration/c.npy` in the root remain as the *active* pair so
that the ~25 existing scripts which load them by name keep working unchanged;
`calib/ACTIVE` records which host those two files currently hold. Any script
that goes through `load_kc()` is checked against that record and will refuse to
run on the wrong board rather than return a number that scores 31%.

USAGE

    from calib import load_kc, save_kc, activate            # board_tools on path

    k, c = load_kc()                    # host from QICK_NS_HOST, verified
    k, c = load_kc(host="10.1.4.105")   # explicit
    save_kc(k, c, host)                 # writes scoped pair + activates it

Switching boards from the shell, without recalibrating:

"""
import os
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CALDIR = ROOT / "calib"
ACTIVE = CALDIR / "ACTIVE"
LEGACY_K = ROOT / "calibration/k.npy"
LEGACY_C = ROOT / "calibration/c.npy"

DEFAULT_HOST = "10.1.9.26"          # board A -- every historical result
KNOWN = {"10.1.9.26": "board A", "10.1.4.105": "board B"}

def host_from_env(host=None):
    """The board this process is talking to: explicit argument, else
    QICK_NS_HOST, else board A (which is what every script defaults to)."""
    return host or os.environ.get("QICK_NS_HOST") or DEFAULT_HOST

def paths(host=None):
    h = host_from_env(host)
    return CALDIR / f"k_{h}.npy", CALDIR / f"c_{h}.npy"

def active_host():
    """Which board's coefficients `calibration/k.npy` / `calibration/c.npy` currently hold, or None if
    the marker is missing (an un-migrated tree)."""
    if ACTIVE.exists():
        s = ACTIVE.read_text(encoding="utf-8").strip()
        return s or None
    return None

def activate(host=None, quiet=False):
    """Copy a board's scoped coefficients into the root `calibration/k.npy` / `calibration/c.npy` and
    record the ownership, so legacy by-name readers see the right board."""
    h = host_from_env(host)
    kp, cp = paths(h)
    if not (kp.exists() and cp.exists()):
        raise FileNotFoundError(
            f"no stored calibration for {h} ({KNOWN.get(h, 'unknown board')}); "
            f"expected {kp.name} and {cp.name} in {CALDIR}. Run "
            f"tools/run_calibration.py against that board first.")
    CALDIR.mkdir(exist_ok=True)
    shutil.copy(kp, LEGACY_K)
    shutil.copy(cp, LEGACY_C)
    ACTIVE.write_text(h + "\n", encoding="utf-8")
    if not quiet:
        k, c = complex(np.load(kp)), complex(np.load(cp))
        print(f"[calib] active = {h} ({KNOWN.get(h, 'unknown board')})  "
              f"|k|={abs(k):.5f} arg(k)={np.degrees(np.angle(k)):+.2f} deg")
    return complex(np.load(kp)), complex(np.load(cp))

def save_kc(k, c, host=None):
    """Store a freshly fitted pair under the board's own name and make it
    active. Never clobbers another board's values."""
    h = host_from_env(host)
    CALDIR.mkdir(exist_ok=True)
    kp, cp = paths(h)
    np.save(kp, complex(k))
    np.save(cp, complex(c))
    print(f"[calib] saved {kp.name} / {cp.name}")
    return activate(h)

def load_kc(host=None, strict=True):
    """Return `(k, c)` for this board.

    `strict` (the default) refuses to proceed when the root `calibration/k.npy` / `calibration/c.npy`
    belong to a different board, because the 25 scripts that read those files by
    name would then silently use the wrong ones. Pass `strict=False` only for
    offline analysis that touches no hardware.
    """
    h = host_from_env(host)
    kp, cp = paths(h)

    if kp.exists() and cp.exists():
        k, c = complex(np.load(kp)), complex(np.load(cp))
        act = active_host()
        if act != h:
            if strict:
                raise SystemExit(
                    f"[calib] REFUSING TO RUN: this process targets {h} "
                    f"({KNOWN.get(h, 'unknown board')}) but the root calibration/k.npy/c.npy "
                    f"hold {act or 'an unrecorded board'}. Scripts that load "
                    f"calibration/k.npy by name would use the wrong coefficients -- the "
                    f"two boards differ by 119 degrees in phase, which scores "
                    f"31% instead of 94%. Fix with:\n"
                    f"    python board_tools/calib.py activate {h}")
            print(f"[calib] WARNING: root calibration/k.npy/c.npy hold {act}, not {h}")
        return k, c

    # No scoped pair yet. Fall back to the legacy files, but only if nothing
    # says they belong elsewhere.
    act = active_host()
    if act is not None and act != h and strict:
        raise SystemExit(
            f"[calib] REFUSING TO RUN: no stored calibration for {h}, and the "
            f"legacy calibration/k.npy/c.npy belong to {act}. Run tools/run_calibration.py "
            f"against {h} first.")
    if not (LEGACY_K.exists() and LEGACY_C.exists()):
        raise FileNotFoundError(
            f"no calibration for {h}: neither {kp} nor {LEGACY_K} exists. Run "
            f"tools/run_calibration.py.")
    print(f"[calib] no scoped pair for {h}; using legacy calibration/k.npy/c.npy "
          f"(unverified provenance)")
    return complex(np.load(LEGACY_K)), complex(np.load(LEGACY_C))

def _cli():
    if len(sys.argv) < 2 or sys.argv[1] not in ("activate", "list", "adopt"):
        raise SystemExit(__doc__.split("USAGE")[-1])
    cmd = sys.argv[1]
    if cmd == "activate":
        activate(sys.argv[2] if len(sys.argv) > 2 else None)
    elif cmd == "adopt":
        # Take the current root calibration/k.npy/c.npy and claim them for a named host.
        h = sys.argv[2]
        CALDIR.mkdir(exist_ok=True)
        kp, cp = paths(h)
        shutil.copy(LEGACY_K, kp)
        shutil.copy(LEGACY_C, cp)
        ACTIVE.write_text(h + "\n", encoding="utf-8")
        print(f"[calib] adopted root calibration/k.npy/c.npy as {h}")
    else:
        act = active_host()
        print(f"calibration store: {CALDIR}")
        print(f"active (root calibration/k.npy/c.npy): {act or 'UNRECORDED'}")
        print()
        print(f"{'host':14s} {'board':9s} {'|k|':>9s} {'arg(k)':>9s}  {'|c|':>9s}")
        found = False
        for kp in sorted(CALDIR.glob("k_*.npy")):
            h = kp.stem[2:]
            cp = CALDIR / f"c_{h}.npy"
            if not cp.exists():
                print(f"{h:14s} {KNOWN.get(h, '?'):9s}  (missing c_{h}.npy)")
                continue
            k, c = complex(np.load(kp)), complex(np.load(cp))
            mark = " <- ACTIVE" if h == act else ""
            print(f"{h:14s} {KNOWN.get(h, '?'):9s} {abs(k):9.5f} "
                  f"{np.degrees(np.angle(k)):+9.2f} {abs(c):9.5f}{mark}")
            found = True
        if not found:
            print("(store is empty -- run `calib.py adopt <host>` to claim the "
                  "current root calibration/k.npy/c.npy)")

if __name__ == "__main__":
    _cli()
