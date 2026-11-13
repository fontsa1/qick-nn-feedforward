"""Calibration drift within one bitstream load.

Every --interval seconds, on the board named by QICK_NS_HOST:
  phase   arg(k) from the phrst_probe fixed-pulse probe, mode "none" only (a
          phrst program would itself move arg(k); see phrst_probe.py): 20 one-shot programs
          + one 50-rep program. The probe phase equals the calibration's arg(k).
  gain    plateau |IQ| (proportional to |k|)
  offset  mean IQ of a window with a zero-gain pulse (the c term's analogue)
  temp    PL and PS die temperature (AMS sysmon), read in a scratch Jupyter
          kernel so the board's server kernel is never touched
--score N scores a fixed block (test20k_real_X[0:N]) with score_armed_fidelity.py
at the start and the end, both with the START calibration (stored k/c are not
touched), to measure the stale-calibration cost directly.

Both boards can run at once (one process each, QICK_NS_HOST picks the board).
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, "analysis_scripts/diagnostics")
sys.path.insert(0, "board_tools")
import phrst_probe as P                         # noqa: E402
import calib                                    # noqa: E402
import jupyter_remote as J                      # noqa: E402

HOST = P.NS_HOST
TEMP_CODE = r'''
d = "/sys/bus/iio/devices/iio:device0/"
def t(n):
    r = float(open(d + n + "_raw").read()); o = float(open(d + n + "_offset").read())
    s = float(open(d + n + "_scale").read())
    return (r + o) * s / 1000.0
print("TEMP %.3f %.3f" % (t("in_temp2_pl_temp"), t("in_temp0_ps_temp")))
'''

class Temps:
    """A scratch kernel on the board for AMS temperature reads."""

    def __init__(self):
        J.BOARD_HOST = HOST
        J.BASE = f"http://{HOST}:{J.JUPYTER_PORT}"
        J.WS_BASE = f"ws://{HOST}:{J.JUPYTER_PORT}"
        self.s, self.x = J.login()
        self.k = J.start_kernel(self.s, self.x)["id"]

    def read(self):
        try:
            r = J.execute_in_kernel(self.s, self.x, self.k, TEMP_CODE, timeout_s=30)
            for line in (r.get("stdout") or "").splitlines():
                if line.startswith("TEMP"):
                    pl, ps = line.split()[1:3]
                    return float(pl), float(ps)
        except Exception as e:                  # a missed temperature must not stop the run
            print("temp read failed:", e)
        return float("nan"), float("nan")

    def close(self):
        try:
            J.shutdown_kernel(self.s, self.x, self.k)
        except Exception:
            pass

def sample(soc, cfg, a):
    base = dict(freq=50.0, gain=a.gain, p_len=0.6, ro_len=0.5, trig=0.31, final_delay=10.0,
                phrst_gen=None, phrst_ro=None)
    ph, mags = [], []
    for _ in range(a.shots):
        iq = np.asarray(P.PhProg(cfg, dict(base, reps=1)).acquire_decimated(soc, progress=False))
        p, m = P.plateau_phase(iq.reshape(-1, iq.shape[-2], 2)[0])
        ph.append(float(p)); mags.append(float(m))
    iq = np.asarray(P.PhProg(cfg, dict(base, reps=a.reps)).acquire_decimated(soc, progress=False))
    pr, mr = P.plateau_phase(iq.reshape(-1, iq.shape[-2], 2))
    off = []
    for _ in range(5):
        iq = np.asarray(P.PhProg(cfg, dict(base, gain=0.0, reps=1)).acquire_decimated(soc, progress=False))
        z = iq.reshape(-1, iq.shape[-2], 2)[0]
        off.append(z[40:-40].mean(axis=0))
    off = np.mean(off, axis=0)
    m1, s1 = P.circ(np.array(ph))
    mr_, sr = P.circ(np.ravel(pr))
    return dict(phase=m1, phase_std=s1, phase_reps=mr_, phase_reps_std=sr,
                mag=float(np.mean(mags)), mag_reps=float(np.mean(mr)),
                off_i=float(off[0]), off_q=float(off[1]))

def score(n, tag, out_dir, chunk=2000, tries=3, chunk_timeout=780):
    """score_armed_fidelity.py on test20k_real_X[0:n] with the stored (start) calibration, as the campaign's NN_axi step:
    TLAST windows so no host alignment, dataset norm, one process per chunk with retries. Saved shots allow a paired
    start/end comparison. Returns the saved paths; a chunk that never completes is reported and skipped."""
    env = dict(os.environ, QICK_NS_HOST=HOST)
    saved = []
    for s in range(0, n, chunk):
        m = min(chunk, n - s)
        stem = os.path.join(out_dir, f"drift_{HOST}_score_{tag}_{s}")
        cmd = [sys.executable, "-u", "analysis_scripts/diagnostics/score_armed_fidelity.py", "--start", str(s), "--shots", str(m),
               "--chunk", str(m), "--norm", "dataset", "--no-align", "--save", stem + ".npz"]
        for attempt in range(1, tries + 1):
            with open(stem + ".log", "a") as f:
                try:
                    rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, check=False, timeout=chunk_timeout).returncode
                except subprocess.TimeoutExpired:
                    rc = "timeout"
            if os.path.exists(stem + ".npz"):
                saved.append(stem + ".npz")
                print(f"[{HOST}] score {tag} {s}+{m} done (try {attempt})", flush=True)
                break
            print(f"[{HOST}] score {tag} {s}+{m} try {attempt} failed, rc {rc}", flush=True)
            time.sleep(30)
        else:
            print(f"[{HOST}] score {tag} {s}+{m} MISSING", flush=True)
    return saved

def reconnect():
    """Wait out a lost connection. Prints as it waits, so the runner's silence watchdog sees a live process."""
    while True:
        try:
            soc, cfg = P.connect(tries=20)
            return soc, cfg
        except Exception as e:
            print(f"[{HOST}] still reconnecting at {time.strftime('%H:%M:%S')}: {e!r}", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=180)
    ap.add_argument("--interval", type=float, default=60, help="seconds between sample starts")
    ap.add_argument("--shots", type=int, default=20)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--gain", type=float, default=0.5)
    ap.add_argument("--score", type=int, default=0, help="shots to score at start and end (0 = skip)")
    ap.add_argument("--out-dir", default="campaign_v140L1")
    a = ap.parse_args()

    if sys.platform == "win32":                 # no idle sleep while this runs; released when the process exits
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    soc, cfg = P.connect()
    out = os.path.join(a.out_dir, f"drift_{HOST}.json")
    k0, c0 = calib.load_kc(host=HOST, strict=False)
    res = dict(host=HOST, board=calib.KNOWN.get(HOST, "?"), banner=P.banner(cfg), args=vars(a),
               stored_argk_deg=float(np.degrees(np.angle(k0))), stored_absk=float(abs(k0)),
               t0=time.time(), samples=[], gaps=[])
    print(f"[{HOST}] {res['banner']}; stored arg(k) {res['stored_argk_deg']:.2f} deg", flush=True)
    if a.score:
        print(f"[{HOST}] start score ({a.score} shots, stored calibration) ...", flush=True)
        res["score_start"] = score(a.score, "start", a.out_dir)
    temps = Temps()
    try:
        t_end = time.time() + 60 * a.minutes
        while True:
            t_s = time.time()
            pl, ps = temps.read()
            try:
                smp = sample(soc, cfg, a)
            except Exception as e:              # connection lost: same load, so the series continues after the gap
                print(f"[{HOST}] sample failed at {time.strftime('%H:%M:%S')}: {e!r}; reconnecting", flush=True)
                soc, cfg = reconnect()
                res["gaps"].append(dict(t_lost=t_s - res["t0"], t_back=time.time() - res["t0"], error=repr(e)))
                print(f"[{HOST}] reconnected after {time.time() - t_s:.0f} s", flush=True)
                temps.close()
                try:
                    temps = Temps()
                except Exception as e2:
                    print(f"[{HOST}] scratch kernel not restarted: {e2!r}", flush=True)
                if time.time() >= t_end:
                    break
                continue
            smp.update(t=t_s - res["t0"], pl_temp=pl, ps_temp=ps)
            res["samples"].append(smp)
            json.dump(res, open(out, "w"), indent=1)
            first = res["samples"][0]["phase"]
            d = (smp["phase"] - first + 180) % 360 - 180
            print(f"[{HOST}] t={smp['t'] / 60:6.1f} min  arg(k) {smp['phase']:8.3f} deg "
                  f"(d {d:+6.3f}, sd {smp['phase_std']:.2f})  |IQ| {smp['mag']:7.1f}  "
                  f"off ({smp['off_i']:+.2f},{smp['off_q']:+.2f})  PL {pl:5.1f} C  PS {ps:5.1f} C",
                  flush=True)
            if time.time() >= t_end:
                break
            time.sleep(max(0.0, a.interval - (time.time() - t_s)))
    finally:
        temps.close()
    if a.score:
        print(f"[{HOST}] end score ({a.score} shots, START calibration) ...", flush=True)
        res["score_end"] = score(a.score, "end", a.out_dir)
    res["t_end"] = time.time()
    json.dump(res, open(out, "w"), indent=1)
    print(f"[{HOST}] saved {out}", flush=True)

if __name__ == "__main__":
    main()
