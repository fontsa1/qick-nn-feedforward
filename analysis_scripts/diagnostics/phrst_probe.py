"""Does a DDS phase reset (phrst) make the loopback phase -- and so
arg(k) -- repeat across bitstream loads?

Per load, a fixed const pulse (gen 0, 50 MHz) is read back on ro 0 and the
plateau's mean IQ phase is taken per shot. That phase is arg(k) up to a
constant (k is the complex gain of the same gen -> ro loop), so its spread
across loads is the arg(k) spread. Modes:
  none  no phrst (as the measurement programs): expected random per load
  gen   phrst on the pulse only
  ro    phrst on the readout config only
  both  phrst on both, at the same tProc time
Each mode runs as separate one-shot programs (--shots, as every acquisition
script does) and as one --reps program (rep-to-rep stability within a run).

Between loads the board's nameserver kernel is restarted and its server cell
re-run, which re-downloads the same bitstream (QickSoc -> PYNQ Overlay). The
banner is checked after each reload. Stored calibration (k/c) is NOT touched;
run tools/run_calibration.py afterwards.

"""
import argparse
import json
import sys
import time

import numpy as np
import Pyro4

Pyro4.config.SERIALIZER = "pickle"
Pyro4.config.PICKLE_PROTOCOL_VERSION = 4

sys.path.insert(0, "board_tools")
import calib                                    # noqa: E402
import jupyter_remote as J                      # noqa: E402
from qick import QickConfig                     # noqa: E402
from qick.asm_v2 import AveragerProgramV2       # noqa: E402

NS_HOST = __import__("os").environ.get("QICK_NS_HOST", "10.1.9.26")
SERVER_NB = "start nameserver.ipynb"
MODES = {"none": (None, None), "gen": (1, None), "ro": (None, 1), "both": (1, 1)}

def connect(tries=60, wait=3.0):
    last = None
    for _ in range(tries):
        try:
            ns = Pyro4.locateNS(host=NS_HOST, port=8888)
            soc = Pyro4.Proxy(ns.lookup("myqick"))
            cfg = QickConfig(soc.get_cfg())
            cfg["tprocs"][0]["output_pins"].append(("trig", 4, 0, "NN_trigger (trig_4_o)"))
            return soc, cfg
        except Exception as e:                  # board coming up
            last = e
            time.sleep(wait)
    raise RuntimeError(f"board never came up: {last!r}")

def banner(cfg):
    b = [l for l in str(cfg).splitlines() if "built" in l]
    return b[0].strip() if b else "?"

class PhProg(AveragerProgramV2):
    def __init__(s, sc, cf):
        super().__init__(sc, reps=cf["reps"], final_delay=cf["final_delay"], cfg=cf)

    def _initialize(s, cf):
        s.declare_gen(ch=0, nqz=1)
        s.declare_readout(ch=0, length=cf["ro_len"])
        kw = {} if cf["phrst_gen"] is None else {"phrst": cf["phrst_gen"]}
        s.add_pulse(ch=0, name="c", style="const", length=cf["p_len"], freq=cf["freq"],
                    phase=0.0, gain=cf["gain"], ro_ch=0, **kw)
        kw = {} if cf["phrst_ro"] is None else {"phrst": cf["phrst_ro"]}
        s.add_readoutconfig(ch=0, name="r", freq=cf["freq"], gen_ch=0, **kw)

    def _body(s, cf):
        s.send_readoutconfig(ch=0, name="r", t=0)
        s.pulse(ch=0, name="c", t=0)
        # pins [0, NN] keep NN_axi's input draining (tools/run_calibration.py's BD 2.0 note)
        s.trigger(ros=[0], pins=[0, len(s.soccfg["tprocs"][0]["output_pins"]) - 1],
                  t=cf["trig"], width=1.0)

def plateau_phase(iq, skip=40):
    """Mean IQ over the plateau (drop `skip` samples each end) -> phase (deg), |IQ|."""
    z = iq[..., 0] + 1j * iq[..., 1]
    m = z[..., skip:-skip].mean(axis=-1)
    return np.degrees(np.angle(m)), np.abs(m)

def circ(ph_deg):
    """Circular mean and std (deg)."""
    u = np.exp(1j * np.radians(ph_deg))
    R = np.abs(u.mean())
    return float(np.degrees(np.angle(u.mean()))), float(np.degrees(np.sqrt(-2 * np.log(max(R, 1e-12)))))

def probe(soc, cfg, a):
    out = {}
    base = dict(freq=50.0, gain=a.gain, p_len=a.p_len, ro_len=a.ro_len, trig=0.31,
                final_delay=10.0)
    for mode, (pg, pr) in MODES.items():
        cf = dict(base, phrst_gen=pg, phrst_ro=pr, reps=1)
        ph = []
        for _ in range(a.shots):
            iq = np.asarray(PhProg(cfg, cf).acquire_decimated(soc, progress=False))
            p, mag = plateau_phase(iq.reshape(-1, iq.shape[-2], 2)[0])
            ph.append(float(p))
        cfr = dict(cf, reps=a.reps)
        iq = np.asarray(PhProg(cfg, cfr).acquire_decimated(soc, progress=False))
        pr_, mag_r = plateau_phase(iq.reshape(-1, iq.shape[-2], 2))
        m1, s1 = circ(np.array(ph))
        mr, sr = circ(np.asarray(pr_))
        out[mode] = dict(shots=ph, reps=[float(v) for v in np.ravel(pr_)], mean=m1, std=s1,
                         reps_mean=mr, reps_std=sr, mag=float(np.mean(mag_r)))
        print(f"  {mode:>4}: one-shot programs {m1:8.2f} deg +- {s1:5.2f}   "
              f"reps program {mr:8.2f} +- {sr:5.2f}   |IQ| {np.mean(mag_r):.0f}", flush=True)
    return out

def reload_bitstream():
    """Restart the nameserver kernel and re-run its server cell (re-downloads the PL)."""
    s, x = J.login()
    nb = J.get_notebook(s, x, SERVER_NB)
    src = None
    for cell in nb["content"]["cells"]:
        t = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        if cell.get("cell_type") == "code" and "bitfile" in t and "start_server(" in t:
            src = t
            break
    if src is None:
        raise RuntimeError("server cell not found in " + SERVER_NB)
    sess = [q for q in J.list_sessions(s, x) if q["path"] == SERVER_NB]
    if not sess:
        raise RuntimeError("no running session for " + SERVER_NB)
    kid = sess[0]["kernel"]["id"]
    J.restart_kernel(s, x, kid)
    time.sleep(2)
    r = J.execute_in_kernel(s, x, kid, src, wait_for_idle=False, timeout_s=30)
    if r["error"] is not None or not r["seen_busy"]:
        raise RuntimeError(f"server cell failed: {r}")
    time.sleep(20)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loads", type=int, default=4, help="bitstream loads to probe (incl. the current one)")
    ap.add_argument("--shots", type=int, default=20, help="one-shot programs per mode per load")
    ap.add_argument("--reps", type=int, default=50, help="reps in the per-mode reps program")
    ap.add_argument("--gain", type=float, default=0.5)
    ap.add_argument("--p-len", type=float, default=0.6, help="pulse length, us")
    ap.add_argument("--ro-len", type=float, default=0.5, help="readout length, us")
    ap.add_argument("--out", default="campaign_v140L1/A_L2_phrst_probe.json")
    a = ap.parse_args()

    res = dict(args=vars(a), host=NS_HOST, loads=[])
    try:
        k0, _ = calib.load_kc(host=NS_HOST, strict=False)
        res["stored_argk_deg"] = float(np.degrees(np.angle(k0)))
        print(f"stored calibration arg(k) for this load: {res['stored_argk_deg']:.2f} deg")
    except Exception as e:                      # no stored calibration
        print("no stored k:", e)
    for L in range(a.loads):
        if L > 0:
            print(f"\n== reloading bitstream (load {L}) ==", flush=True)
            reload_bitstream()
        soc, cfg = connect()
        b = banner(cfg)
        print(f"\n== load {L}: {b} ==", flush=True)
        if L > 0 and b != res["loads"][0]["banner"]:
            raise SystemExit(f"banner changed across reloads: {b!r}")
        res["loads"].append(dict(banner=b, t=time.time(), modes=probe(soc, cfg, a)))
        json.dump(res, open(a.out, "w"), indent=1)

    print("\n== across loads: per-load mean phase (deg) ==")
    for mode in MODES:
        means = np.array([ld["modes"][mode]["mean"] for ld in res["loads"]])
        within = np.array([ld["modes"][mode]["std"] for ld in res["loads"]])
        m, s = circ(means)
        rng = np.ptp(np.unwrap(np.radians(means - m)) * 180 / np.pi)
        print(f"  {mode:>4}: " + " ".join(f"{v:8.2f}" for v in means)
              + f"   across-load spread: std {s:6.2f}, range {rng:6.2f}"
              + f"   within-load std (median) {np.median(within):5.2f}")
    json.dump(res, open(a.out, "w"), indent=1)
    print("saved", a.out)
    print("NEXT: tools/run_calibration.py (stored k/c belong to the load before this probe).")

if __name__ == "__main__":
    main()
