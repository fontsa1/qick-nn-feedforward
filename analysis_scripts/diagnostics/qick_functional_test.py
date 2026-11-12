"""QICK functional test suite for our modified firmware (tProc v2).

Exercises the features documented in the upstream tProc v2 tutorials
(docs.qick.dev/latest/tutorials, notebooks 00-09; 10/14/15 need extra hardware) on
the running board, with a quantitative pass criterion per test instead of a plot.
The point is to show QICK still does what its documentation says after our
changes (NN_live tap, dac_gate on gen 0, tProc core clock 200 MHz, extra BRAMs).

Wiring on our boards: only gen 0 (DAC_B) -> readout 0 (ADC_D) is cabled. Tests that
need gen 1, readouts 1/2 or the time tagger's ADC report NOT WIRED. The dac_gate
must be disabled (it is at reset and after every probe script); the suite checks.

Statuses: PASS / WARN (works, but differs from QICK's own documentation) / FAIL (criterion
missed) / ERROR (exception) / NOT WIRED / INFO.

  ... --only T05,T12      run a subset;   --list   list tests
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "board_tools"))
import deploy_and_verify as D                                    # noqa: E402
from qick.asm_v2 import AveragerProgramV2, QickSweep1D, QickSpan, AsmV2   # noqa: E402

G, R = 0, 0                    # the cabled generator / readout
F = 300.0                      # default carrier (MHz), NQZ 1 for both DAC and ADC
soc = soccfg = None
LAT = None                     # tProc pulse time -> pulse edge in the readout window (us), from T03
TESTS = []

def test(tid, name, source):
    def reg(fn):
        TESTS.append((tid, name, source, fn))
        return fn
    return reg

# ---------------------------------------------------------------- helpers
def cplx(iq):
    iq = np.asarray(iq, dtype=float)
    return iq[..., 0] + 1j * iq[..., 1]

def dt_ro():
    """Decimated readout sample period (us)."""
    return 1.0 / soccfg['readouts'][R]['f_output']

def runs(mag, frac=0.5, min_len=4):
    """Pulses in a magnitude trace: list of (rise, fall) in fractional samples."""
    mag = np.asarray(mag, float)
    base = np.median(np.sort(mag)[:max(8, len(mag) // 10)])
    top = np.percentile(mag, 99)
    thr = base + frac * (top - base)
    above = mag > thr
    out, i, n = [], 0, len(mag)
    while i < n:
        if above[i]:
            j = i
            while j < n and above[j]:
                j += 1
            if j - i >= min_len:
                r = i - 1 + (thr - mag[i - 1]) / (mag[i] - mag[i - 1]) if i > 0 else float(i)
                f = j - 1 + (mag[j - 1] - thr) / (mag[j - 1] - mag[j]) if j < n else float(j)
                out.append((r, f))
            i = j
        else:
            i += 1
    return out

def plateau(z, rise, fall, guard=6):
    a, b = int(np.ceil(rise)) + guard, int(np.floor(fall)) - guard
    return z[a:b].mean() if b > a else np.nan

def ok(cond):
    return "PASS" if cond else "FAIL"

def load_dmem(data, addr):
    """soc.load_mem to dmem, tolerating a QICK 0.2.414 (and upstream main) software bug:
    tproc.load_mem's readback check reads from address 0 whatever `addr` was, so any
    load to addr != 0 raises "readback does not match". We verify the write ourselves."""
    data = np.asarray(data, np.int32)
    try:
        soc.load_mem(data, mem_sel='dmem', addr=addr)
    except Exception as e:
        if "readback does not match" not in str(e) or addr == 0:
            raise
    back = np.asarray(soc.read_mem(length=len(data), mem_sel='dmem', addr=addr)).astype(np.int32).reshape(-1)
    if not np.array_equal(back, data):
        raise RuntimeError(f"dmem write at {addr} did not read back")

class Base(AveragerProgramV2):
    """declare gen G + readout R (length cfg['ro_len']), readout config 'ro' at
    cfg['freq'] (frequency-matched), sent at t=0. Subclasses add pulses/body."""
    def _initialize(self, cfg):
        self.declare_gen(ch=G, nqz=1)
        self.declare_readout(ch=R, length=cfg['ro_len'])
        self.add_readoutconfig(ch=R, name="ro", freq=cfg.get('ro_freq', cfg['freq']), gen_ch=G,
                               phase=cfg.get('ro_phase', 0))
        self.send_readoutconfig(ch=R, name="ro", t=0)
        self.setup(cfg)

    def setup(self, cfg):
        pass

def const(prog, name, length, gain=0.5, phase=0, freq=None):
    prog.add_pulse(ch=G, name=name, style="const", freq=freq if freq is not None else prog.cfg['freq'],
                   ro_ch=R, length=length, phase=phase, gain=gain)

def dec(prog, rounds=1):
    return np.asarray(prog.acquire_decimated(soc, rounds=rounds, progress=False)[0])

# ---------------------------------------------------------------- 00 getting started
@test("T00", "Firmware/software configuration", "00 Getting Started §3")
def t00():
    t = soccfg['tprocs'][0]
    m = dict(sw=soccfg['sw_version'], n_gens=len(soccfg['gens']), n_ros=len(soccfg['readouts']),
             tproc=t['type'], rev=t['revision'], f_core=t['f_core'], f_time=t['f_time'],
             gen0_fs=soccfg['gens'][0]['fs'], ro0_fs=soccfg['readouts'][0]['fs'],
             ro0_fout=soccfg['readouts'][0]['f_output'], pins=len(t['output_pins']),
             has_lfsr=t['has_lfsr'], has_div=t['has_divider'], has_arith=t['has_arith'],
             ddr4=bool(soccfg['ddr4_buf']) if 'ddr4_buf' in soccfg._cfg else None,
             mr=bool(soccfg['mr_buf']) if 'mr_buf' in soccfg._cfg else None)
    good = (m['n_gens'] == 2 and m['n_ros'] == 3 and m['tproc'] == 'qick_processor' and m['f_core'] == 200.0
            and abs(m['f_time'] - 491.52) < 1e-6 and m['pins'] == 8)
    return ok(good), m, "expected: 2 gens, 3 readouts, tProc v2 core 200 MHz / time 491.52 MHz, 8 pins"

@test("T01", "Memory access: host and tProc", "00 §4 (read_mem); 05 (dmem); 04 (write_dmem)")
def t01():
    rng = np.random.default_rng(1)
    data = rng.integers(-2**31, 2**31 - 1, 512, dtype=np.int64).astype(np.int32)
    try:
        load_dmem(data, 8000)
        host_ok = True
    except RuntimeError:
        host_ok = False

    class P(Base):
        def setup(s, cfg):
            s.add_reg("radr"); s.add_reg("rval"); s.add_reg("rcnt")
        def _body(s, cfg):
            s.write_reg("radr", 9000); s.write_reg("rval", 7); s.write_reg("rcnt", 64)
            s.label("LOOP")
            s.write_dmem(addr="radr", src="rval")
            s.inc_reg("radr", 1); s.inc_reg("rval", 3); s.inc_reg("rcnt", -1)
            s.cond_jump("LOOP", "rcnt", "NZ")
            s.trigger(ros=[R], t=0.1)
    load_dmem(np.zeros(64, np.int32), 9000)
    P(soccfg, reps=1, final_delay=1.0, cfg=dict(freq=F, ro_len=0.1)).acquire(soc, rounds=1, progress=False)
    got = np.asarray(soc.read_mem(length=64, mem_sel='dmem', addr=9000)).astype(np.int64).reshape(-1)
    exp = 7 + 3 * np.arange(64)
    prog_ok = np.array_equal(got, exp)
    pc = int(soc.get_tproc_counter(addr=1))
    return ok(host_ok and prog_ok), dict(host_roundtrip_512=host_ok, tproc_loop_writes_64=prog_ok,
                                         first_tproc_values=got[:4].tolist(), tproc_counter=pc), \
        ("host load/read 512 random words; tProc counted loop writing 64 register-addressed words. "
         "NB: QICK's own load_mem readback check is broken for addr != 0 (upstream software bug, worked around)")

@test("T02", "Wiring matrix (gen -> readout)", "06 §3, 07 §10 (check loopback)")
def t02():
    class P(AveragerProgramV2):
        def _initialize(s, cfg):
            s.declare_gen(ch=cfg['g'], nqz=1)
            for r in range(3):
                s.declare_readout(ch=r, length=1.0)
                s.add_readoutconfig(ch=r, name=f"r{r}", freq=F, gen_ch=cfg['g'])
                s.send_readoutconfig(ch=r, name=f"r{r}", t=0)
            s.add_pulse(ch=cfg['g'], name="p", style="const", freq=F, ro_ch=0, length=0.4, phase=0, gain=0.5)
        def _body(s, cfg):
            s.trigger(ros=[0, 1, 2], t=0.0)
            s.pulse(ch=cfg['g'], name="p", t=0.2)
    mat = np.zeros((2, 3))
    for g in range(2):
        d = P(soccfg, reps=1, final_delay=1.0, cfg={'g': g}).acquire_decimated(soc, rounds=2, progress=False)
        for r in range(3):
            a = np.abs(cplx(d[r]))
            mat[g, r] = a.max() / max(np.median(a), 1e-9)
    wired = mat > 50
    m = dict(peak_over_median=np.round(mat, 1).tolist(), wired=wired.tolist())
    return ("PASS" if wired[0, 0] else "FAIL"), m, "gen0->ro0 must be wired; others reported (peak/median > 50 = wired)"

# ---------------------------------------------------------------- 01 basic sequencing
@test("T03", "Single const pulse: width, timing shift, repeatability", "01 §3; 00_Send_receive (rep-to-rep)")
def t03():
    global LAT

    class P(Base):
        def setup(s, cfg): const(s, "p", cfg['len'], gain=0.5)
        def _body(s, cfg):
            s.trigger(ros=[R], t=0.1)
            s.pulse(ch=G, name="p", t=cfg['tp'])
    cfg = dict(freq=F, ro_len=1.5, len=0.4, tp=0.3)
    zs, edges = [], []
    for _ in range(5):
        z = cplx(dec(P(soccfg, reps=1, final_delay=1.0, cfg=cfg)))
        r = runs(np.abs(z))
        if len(r) != 1:
            return "FAIL", dict(n_pulses=len(r)), "expected exactly one pulse in the window"
        edges.append(r[0]); zs.append(plateau(z, *r[0]))
    zs = np.array(zs); edges = np.array(edges)
    width = np.mean(edges[:, 1] - edges[:, 0]) * dt_ro()
    LAT = edges[:, 0].mean() * dt_ro() + 0.1 - cfg['tp']
    cfg2 = dict(cfg, tp=0.4)
    z2 = cplx(dec(P(soccfg, reps=1, final_delay=1.0, cfg=cfg2)))
    shift = (runs(np.abs(z2))[0][0] - edges[:, 0].mean()) * dt_ro()
    amp_cv = np.std(np.abs(zs)) / np.mean(np.abs(zs))
    ph_sd = np.degrees(np.std(np.angle(zs * np.conj(zs.mean()))))
    m = dict(width_us=round(width, 4), width_err_ns=round((width - 0.4) * 1e3, 2),
             t_shift_us=round(shift, 4), shift_err_ns=round((shift - 0.1) * 1e3, 2),
             latency_us=round(LAT, 4), amp_cv=round(amp_cv, 5), phase_sd_deg=round(ph_sd, 3),
             edge_sd_samples=round(float(edges[:, 0].std()), 3))
    good = abs(width - 0.4) < 3 * dt_ro() and abs(shift - 0.1) < 2 * dt_ro() and amp_cv < 0.01 and ph_sd < 1.0
    return ok(good), m, "width 0.4 us and a 0.1 us shift within 3/2 samples; 5 runs: amplitude CV < 1%, phase sd < 1 deg"

@test("T04", "Envelope shapes: gauss, flat_top, triangle, DRAG, custom", "01 §4; 06 §4; playing_pulses")
def t04():
    fsg = soccfg['gens'][G]['fs']                     # envelope sample rate (Msps)
    spc = soccfg['gens'][G]['samps_per_clk']

    class P(Base):
        def setup(s, cfg):
            k = cfg['kind']
            if k == "gauss":
                s.add_gauss(ch=G, name="e", sigma=0.025, length=0.2, even_length=True)
                s.add_pulse(ch=G, name="p", style="arb", envelope="e", freq=F, ro_ch=R, phase=0, gain=0.8)
            elif k == "flat_top":
                s.add_gauss(ch=G, name="e", sigma=0.02, length=0.16, even_length=True)
                s.add_pulse(ch=G, name="p", style="flat_top", envelope="e", freq=F, ro_ch=R, length=0.2, phase=0, gain=0.8)
            elif k == "triangle":
                s.add_triangle(ch=G, name="e", length=0.2)
                s.add_pulse(ch=G, name="p", style="arb", envelope="e", freq=F, ro_ch=R, phase=0, gain=0.8)
            elif k == "drag":
                s.add_DRAG(ch=G, name="e", sigma=0.025, length=0.2, delta=-50.0, alpha=0.5)
                s.add_pulse(ch=G, name="p", style="arb", envelope="e", freq=F, ro_ch=R, phase=0, gain=0.8)
            elif k == "custom":
                n = 96 * spc
                maxv = soccfg['gens'][G]['maxv'] * 0.9
                ramp = np.r_[np.linspace(0, maxv, n // 2), np.linspace(maxv, 0, n // 2) * 0.5]
                s.add_envelope(ch=G, name="e", idata=ramp, qdata=np.zeros(n))
                s.add_pulse(ch=G, name="p", style="arb", envelope="e", freq=F, ro_ch=R, phase=0, gain=0.8)
        def _body(s, cfg):
            s.trigger(ros=[R], t=0.1)
            s.pulse(ch=G, name="p", t=0.3)
    res, worst = {}, 0.0
    for k in ("gauss", "flat_top", "triangle", "drag", "custom"):
        prog = P(soccfg, reps=1, final_delay=1.0, cfg=dict(freq=F, ro_len=1.0, kind=k))
        mag = np.abs(cplx(dec(prog, rounds=8)))
        env = prog.envelopes[G]['envs']['e']['data'].astype(float)
        e = np.hypot(env[:, 0], env[:, 1])
        if k == "flat_top":                         # rise half, flat, fall half
            h = len(e) // 2
            nflat = int(round(0.2 * fsg))
            e = np.r_[e[:h], np.full(nflat, e.max()), e[h:]]
        te = np.arange(len(e)) / fsg                 # envelope time (us)
        t = np.arange(len(mag)) * dt_ro()
        base = np.median(mag[:20])
        y = mag - base
        best = None
        coarse = []
        for t0 in np.arange(-te[-1], t[-1], dt_ro() / 2):
            mdl = np.interp(t - t0, te, e, left=0, right=0)
            coarse.append((-(mdl @ y) / max(np.sqrt(mdl @ mdl), 1e-12), t0))
        t0c = min(coarse)[1]
        for t0 in np.arange(t0c - dt_ro(), t0c + dt_ro(), dt_ro() / 40):
            mdl = np.interp(t - t0, te, e, left=0, right=0)
            sc = (mdl @ y) / max(mdl @ mdl, 1e-12)
            rr = np.sqrt(np.mean((y - sc * mdl) ** 2)) / y.max()
            if best is None or rr < best[0]:
                best = (rr, t0, sc)
        res[k] = round(float(best[0]), 4)
        worst = max(worst, best[0])
    return ok(worst < 0.03), dict(nrms_residual=res), \
        "measured |envelope| vs the program's own envelope samples (best shift+scale): normalised RMS residual < 3% each"

# ---------------------------------------------------------------- 02 sweeps
def window_prog():
    """Long const pulse with the readout window inside its plateau (needs LAT)."""
    class P(Base):
        def setup(s, cfg):
            if 'loops' in cfg:
                for nm, n in cfg['loops']:
                    s.add_loop(nm, n)
            s.add_pulse(ch=G, name="p", style="const", freq=cfg['freq'], ro_ch=R, length=1.0,
                        phase=cfg.get('phase', 0), gain=cfg.get('gain', 0.5))
        def _body(s, cfg):
            s.pulse(ch=G, name="p", t=0.2)
            s.trigger(ros=[R], t=0.2 + LAT + 0.3)
    return P

@test("T05", "Gain sweep (QickSweep1D): linearity", "02 §3; 09 §3.3 (quantization)")
def t05():
    P = window_prog()
    g = QickSweep1D("g", 0.05, 0.95)
    prog = P(soccfg, reps=20, final_delay=1.0, cfg=dict(freq=F, ro_len=0.4, loops=[("g", 19)], gain=g))
    z = cplx(np.asarray(prog.acquire(soc, rounds=1, progress=False)[0])).reshape(-1)
    gains = np.asarray(prog.get_pulse_param("p", "gain", as_array=True)).reshape(-1)
    a = np.abs(z)
    k, c = np.polyfit(gains, a, 1)
    dev = np.max(np.abs(a - (k * gains + c))) / (k * gains.max())
    ph = np.degrees(np.angle(z * np.conj(z[-1])))
    m = dict(n=len(a), slope=round(k, 3), intercept_frac=round(c / (k * gains.max()), 5),
             max_dev_frac=round(dev, 5), phase_span_deg=round(float(ph.max() - ph.min()), 3),
             gains_first_last=[round(gains[0], 5), round(gains[-1], 5)])
    good = dev < 0.01 and abs(c) / (k * gains.max()) < 0.01 and (ph.max() - ph.min()) < 1.0
    return ok(good), m, "|IQ| vs actual gain: max deviation from a line < 1% FS, intercept < 1% FS, phase constant < 1 deg"

@test("T06", "Phase sweep: commanded vs measured", "02 §3; 06 §5 (IQ mixing)")
def t06():
    P = window_prog()
    ph = QickSweep1D("ph", 0, 330)
    prog = P(soccfg, reps=20, final_delay=1.0, cfg=dict(freq=F, ro_len=0.4, loops=[("ph", 12)], phase=ph))
    z = cplx(np.asarray(prog.acquire(soc, rounds=1, progress=False)[0])).reshape(-1)
    cmd = np.asarray(prog.get_pulse_param("p", "phase", as_array=True)).reshape(-1)
    meas = np.degrees(np.angle(z * np.conj(z[0])))
    err = (meas - (cmd - cmd[0]) + 180) % 360 - 180
    amp_cv = np.std(np.abs(z)) / np.mean(np.abs(z))
    m = dict(rms_err_deg=round(float(np.sqrt(np.mean(err ** 2))), 3), max_err_deg=round(float(np.max(np.abs(err))), 3),
             amp_cv=round(amp_cv, 5))
    return ok(m['max_err_deg'] < 1.0 and amp_cv < 0.005), m, "12 phases 0..330: |error| < 1 deg each, amplitude CV < 0.5%"

@test("T07", "Frequency matching: phase coherence across runs", "freq_matching; 01 §3 note")
def t07():
    P = window_prog()
    f = 487.1234567
    zs = []
    for _ in range(10):
        zs.append(cplx(np.asarray(P(soccfg, reps=20, final_delay=1.0, cfg=dict(freq=f, ro_len=0.4))
                              .acquire(soc, rounds=1, progress=False)[0])).reshape(-1)[0])
    zs = np.array(zs)
    sd = np.degrees(np.std(np.angle(zs * np.conj(zs.mean()))))
    return ok(sd < 1.0), dict(freq=f, phase_sd_deg=round(sd, 3), amp_cv=round(np.std(np.abs(zs)) / np.mean(np.abs(zs)), 5)), \
        "10 separately compiled runs at an odd frequency: phase sd < 1 deg"

@test("T08", "Dynamic readout frequency sweep (exec_before) + electrical delay", "07 §5, §5.1")
def t08():
    class P(AveragerProgramV2):
        def _initialize(s, cfg):
            s.declare_gen(ch=G, nqz=1)
            s.declare_readout(ch=R, length=0.3)
            s.add_readoutconfig(ch=R, name="ro", freq=cfg['freq'], gen_ch=G)
            lb = AsmV2()
            lb.send_readoutconfig(ch=R, name="ro", t=0)
            s.add_loop("f", cfg['steps'], exec_before=lb)
            s.add_pulse(ch=G, name="p", style="const", freq=cfg['freq'], ro_ch=R, length=0.8, phase=0, gain=0.5)
        def _body(s, cfg):
            s.pulse(ch=G, name="p", t=0.2)
            s.trigger(ros=[R], t=0.2 + LAT + 0.3)
    out = []
    for rep in range(2):
        prog = P(soccfg, reps=5, final_delay=1.0, cfg=dict(freq=QickSweep1D("f", 100, 1900), steps=3601))
        z = cplx(np.asarray(prog.acquire(soc, rounds=1, progress=False)[0])).reshape(-1)
        fr = np.asarray(prog.get_pulse_param("ro", "freq", as_array=True)).reshape(-1)
        ph = np.unwrap(np.angle(z)) / (2 * np.pi)
        k, c = np.polyfit(fr, ph, 1)
        resid = (ph - (k * fr + c)) * 360
        out.append(dict(delay_ns=k * 1e3, resid_rms=float(np.sqrt(np.mean(resid ** 2))),
                        resid_max=float(np.max(np.abs(resid))), amp_min_over_med=float(np.min(np.abs(z)) / np.median(np.abs(z)))))
    m = dict(delay_ns=[round(o['delay_ns'], 3) for o in out], resid_rms_deg=[round(o['resid_rms'], 2) for o in out],
             resid_max_deg=[round(o['resid_max'], 2) for o in out], amp_min_over_median=[round(o['amp_min_over_med'], 3) for o in out])
    good = abs(out[0]['delay_ns'] - out[1]['delay_ns']) < 0.1 and max(o['resid_rms'] for o in out) < 5 \
        and min(o['amp_min_over_med'] for o in out) > 0.2
    return ok(good), m, "3601 points 100-1900 MHz, twice: linear phase (residual RMS < 5 deg), delay repeatable to 0.1 ns, no dropouts"

# ---------------------------------------------------------------- 03 timing
@test("T09", "Timeline semantics: delay, delay_auto, wait, collisions, swept delay", "03 §3-§6; timing")
def t09():
    tol = 3 * dt_ro()

    class P(Base):
        def setup(s, cfg):
            const(s, "a", cfg['la'], gain=0.5); const(s, "b", 0.1, gain=0.5)
            if 'steps' in cfg:
                s.add_loop("d", cfg['steps'])
        def _body(s, cfg):
            s.trigger(ros=[R], t=0.05)
            s.pulse(ch=G, name="a", t=0.1)
            v = cfg['v']
            if v == "delay":
                s.delay(0.3)
            elif v == "delay_auto":
                s.delay_auto(0.1, ros=False)
            elif v == "wait":
                s.wait(0.8)
            elif v == "sweep":
                s.delay(QickSweep1D("d", 0.15, 0.55))
            s.pulse(ch=G, name="b", t=cfg.get('tb', 0.1))
    # a at t=0.1 (len la); b at t=tb (0.1) relative to the reference after the timing command:
    #   delay(0.3): ref 0.3 -> b at 0.4, a at 0.1: 0.3;  delay_auto(0.1): ref = end of a (0.35) + 0.1
    #   -> b at 0.55: 0.45;  collision: b requested at 0.4 inside a (ends 0.6) -> plays right after a
    cases = [("delay", dict(la=0.1), 0.3), ("delay_auto", dict(la=0.25), 0.45),
             ("collision", dict(la=0.5, tb=0.3 + 0.1), 0.5)]
    m, good = {}, True
    for v, extra, exp in cases:
        z = cplx(dec(P(soccfg, reps=1, final_delay=1.0, cfg=dict(freq=F, ro_len=1.6, v=v, **extra)), rounds=2))
        r = runs(np.abs(z))
        if len(r) != 2 and v != "collision":
            m[v] = f"{len(r)} pulses"; good = False; continue
        if v == "collision":                       # back-to-back: one long run of la + 0.1
            dmeas = (r[0][1] - r[0][0]) * dt_ro() - 0.1 if len(r) == 1 else (r[1][0] - r[0][0]) * dt_ro()
        else:
            dmeas = (r[1][0] - r[0][0]) * dt_ro()
        m[v] = dict(expected_us=exp, measured_us=round(dmeas, 4), err_ns=round((dmeas - exp) * 1e3, 2))
        good &= abs(dmeas - exp) < tol
    # wait(0.8): tProc pauses to ref+0.8; b (t=0.1 relative) is late and plays as soon as dispatched
    z = cplx(dec(P(soccfg, reps=1, final_delay=1.0, cfg=dict(freq=F, ro_len=1.6, v="wait", la=0.1)), rounds=2))
    r = runs(np.abs(z))
    dw = (r[1][0] - r[0][0]) * dt_ro() if len(r) == 2 else np.nan
    m['wait'] = dict(measured_us=round(dw, 4), note="b must start no earlier than wait time - a start (0.7 us)")
    good &= bool(dw >= 0.7 - tol)
    prog = P(soccfg, reps=1, final_delay=1.0, cfg=dict(freq=F, ro_len=1.6, v="sweep", la=0.1, steps=5))
    d = cplx(dec(prog, rounds=2))
    exp = np.linspace(0.15, 0.55, 5)
    meas = []
    for k in range(5):
        r = runs(np.abs(d[k]))
        meas.append((r[1][0] - r[0][0]) * dt_ro() if len(r) == 2 else np.nan)
    meas = np.array(meas)
    m['swept_delay'] = dict(expected_us=exp.round(3).tolist(), measured_us=meas.round(4).tolist())
    good &= bool(np.all(np.abs(meas - exp) < tol))
    return ok(good), m, f"pulse separations within 3 samples ({tol * 1e3:.1f} ns) of the documented timeline"

# ---------------------------------------------------------------- loops and averaging
@test("T10", "2D sweep (QickSpan), loop order, reps_innermost", "02 §4; 09 §5.1")
def t10():
    P = window_prog()
    gain = 0.1 + QickSpan("l1", 0.4) + QickSpan("l2", 0.3)
    prog = P(soccfg, reps=10, final_delay=1.0, cfg=dict(freq=F, ro_len=0.4, loops=[("l1", 3), ("l2", 4)], gain=gain))
    z = cplx(np.asarray(prog.acquire(soc, rounds=1, progress=False)[0]))
    gg = np.asarray(prog.get_pulse_param("p", "gain", as_array=True))
    a = np.abs(z).reshape(gg.shape)
    k = (a.ravel() @ gg.ravel()) / (gg.ravel() @ gg.ravel())
    rel = np.max(np.abs(a - k * gg) / (k * gg))
    m = dict(data_shape=list(z.shape), gain_shape=list(gg.shape), max_rel_dev=round(float(rel), 4))
    good = rel < 0.02 and tuple(gg.shape) == (3, 4)
    # loop order in per-shot raw data, both nestings
    for inner in (False, True):
        g1 = QickSweep1D("g", 0.2, 0.8)
        p2 = P(soccfg, reps=3, final_delay=1.0, cfg=dict(freq=F, ro_len=0.4, loops=[("g", 4)], gain=g1), reps_innermost=inner)
        p2.acquire(soc, rounds=1, progress=False)
        raw = np.asarray(p2.get_raw()[0])
        amp = np.abs(cplx(raw)).reshape(raw.shape[:-1]).squeeze()
        # the gain axis must be the one along which amplitude rises
        gax = 1 if amp.shape[0] == 3 else 0
        rising = np.all(np.diff(amp, axis=gax) > 0)
        m[f"raw_shape_inner{inner}"] = list(raw.shape)
        m[f"amp_rises_along_gain_inner{inner}"] = bool(rising)
        good &= bool(rising)
    return ok(good), m, "|IQ| proportional to the 3x4 QickSpan gain grid within 2%; per-shot raw data ordered as declared"

@test("T11", "Accumulated vs decimated buffer, per shot, exact", "01 §3; 00_Send_receive; 08 (buffers)")
def t11():
    class P(Base):
        def setup(s, cfg): const(s, "p", 0.3, gain=0.5)
        def _body(s, cfg):
            s.pulse(ch=G, name="p", t=0.2)
            s.trigger(ros=[R], t=0.2 + LAT - 0.05)
    n = 40
    prog = P(soccfg, reps=n, final_delay=1.0, cfg=dict(freq=F, ro_len=0.4))
    prog.acquire_decimated(soc, rounds=1, progress=False)
    L = prog.ro_chs[R]['length']
    decd = np.asarray(soc.get_decimated(ch=R, address=0, length=n * L)).reshape(-1, 2)
    acc = np.asarray(soc.get_accumulated(ch=R, address=0, length=n)).reshape(-1, 2)
    sums = decd.reshape(n, L, 2).sum(axis=1).astype(np.int64)
    exact = int(np.sum(np.all(sums == acc.astype(np.int64), axis=1)))
    m = dict(shots=n, samples_per_shot=int(L), exact_matches=exact, first_shot=dict(dec_sum=sums[0].tolist(), acc=acc[0].tolist()))
    return ok(exact == n), m, "sum of each shot's decimated samples == that shot's accumulated value, all shots"

# ---------------------------------------------------------------- 04 feedback
@test("T12", "Feedback: read_input == buffered accumulation, per shot (sweep read_wait)", "04 §3")
def t12():
    BASE = 10000

    class P(Base):
        def setup(s, cfg):
            const(s, "p", 0.1, gain=cfg['gain'])
            s.add_reg("radr")
            s.write_reg("radr", BASE)
        def _body(s, cfg):
            s.pulse(ch=G, name="p", t=0)
            s.trigger(ros=[R], t=LAT - 0.02)
            s.wait_auto(cfg['rw'])
            s.read_input(ro_ch=R)
            s.write_dmem(addr="radr", src="s_port_l"); s.inc_reg("radr", 1)
            s.write_dmem(addr="radr", src="s_port_h"); s.inc_reg("radr", 1)
    n, res, first_ok = 200, {}, None
    for rw in (0.0, 0.02, 0.05, 0.1, 0.2):
        load_dmem(np.zeros(2 * n, np.int32), BASE)
        prog = P(soccfg, reps=n, final_delay=1.0, cfg=dict(freq=F, ro_len=0.15, rw=rw, gain=0.4))
        prog.acquire(soc, rounds=1, progress=False)
        raw = np.asarray(prog.get_raw()[0]).reshape(n, 2).astype(np.int64)
        fb = np.asarray(soc.read_mem(length=2 * n, mem_sel='dmem', addr=BASE)).astype(np.int32).astype(np.int64).reshape(n, 2)
        k = int(np.sum(np.all(fb == raw, axis=1)))
        res[str(rw)] = k
        if k == n and first_ok is None:
            first_ok = rw
    m = dict(shots=n, exact_matches_by_read_wait_us=res, min_read_wait_all_exact=first_ok)
    # once the wait is long enough, every larger wait must also match exactly
    stable = first_ok is not None and all(v == n for kk, v in res.items() if float(kk) >= first_ok)
    return ok(stable), m, "with a sufficient read_wait, the tProc's read_input equals the buffer's per-shot sum (I and Q) on every shot"

@test("T13", "Conditional pulse (read_and_jump) decision consistency", "04 §4, §6")
def t13():
    # rotate the readout so the response lies on +I for positive gain
    P = window_prog()
    z0 = cplx(np.asarray(P(soccfg, reps=10, final_delay=1.0, cfg=dict(freq=F, ro_len=0.4)).acquire(soc, rounds=1, progress=False)[0])).reshape(-1)[0]
    rot = -np.degrees(np.angle(z0))

    class Q(Base):
        def setup(s, cfg):
            s.add_loop("g", cfg['steps'])
            const(s, "m", 0.1, gain=QickSweep1D("g", -1.0, 1.0))
            const(s, "c", 0.1, gain=0.8)
        def _body(s, cfg):
            s.pulse(ch=G, name="m", t=0)
            s.trigger(ros=[R], t=LAT - 0.02)
            # the read needs >= 0.2 us after the readout window (T12); with 0.1 the
            # tProc read the PREVIOUS shot's second readout on ~0.25% of shots
            # (2026-10-13). delay_auto must then clear the stalled processor time.
            s.wait_auto(0.25)
            s.delay_auto(0.5)
            s.read_and_jump(ro_ch=R, component='I', threshold=0, test='>=', label='NOCOND')
            s.pulse(ch=G, name="c", t=0)
            s.label('NOCOND')
            s.trigger(ros=[R], t=LAT - 0.02)
    steps, reps = 41, 5
    prog = Q(soccfg, reps=reps, final_delay=1.0, cfg=dict(freq=F, ro_len=0.15, ro_phase=rot, steps=steps))
    prog.acquire(soc, rounds=1, progress=False)
    raw = np.asarray(prog.get_raw()[0]).astype(np.int64)          # (reps, steps, 2 trig, 2)
    raw = raw.reshape(reps, steps, 2, 2)
    i1 = raw[:, :, 0, 0]
    a2 = np.abs(cplx(raw[:, :, 1, :]))
    played = a2 > 0.5 * np.median(a2[a2 > np.percentile(a2, 60)])
    expect = i1 < 0
    agree = int(np.sum(played == expect))
    m = dict(shots=reps * steps, agree=agree, n_played=int(played.sum()), n_expected=int(expect.sum()), ro_rotation_deg=round(rot, 2))
    return ok(agree == reps * steps), m, "the conditional pulse plays exactly on the shots whose first readout has I < 0 (same-shot raw value)"

# ---------------------------------------------------------------- 05 dynamic parameters
@test("T14", "Virtual-Z via read/write_wmem subroutine; subroutine timing; per-pulse gain", "05 §3, §5, §6")
def t14():
    class VZ(Base):
        def setup(s, cfg):
            s.add_gauss(ch=G, name="ramp", sigma=0.01, length=0.1, even_length=True)
            s.add_pulse(ch=G, name="p", style="flat_top", envelope="ramp", freq=F, ro_ch=R, length=0.05, phase=0, gain=0.8)
            sub = AsmV2()
            for w in s.list_pulse_waveforms("p"):
                sub.read_wmem(name=w); sub.inc_reg(dst='w_phase', src=s.deg2reg(90, gen_ch=G)); sub.write_wmem(name=w)
            s.add_subroutine("vz", sub)
        def _body(s, cfg):
            s.trigger(ros=[R], t=0.05)
            s.pulse(ch=G, name="p", t=0.1)
            s.call("vz")
            s.pulse(ch=G, name="p", t=0.35)
            s.pulse(ch=G, name="p", t=0.6)
            s.call("vz")
            s.pulse(ch=G, name="p", t=0.85)
            for w in s.list_pulse_waveforms("p"):
                s.read_wmem(name=w); s.write_reg(dst='w_phase', src=0); s.write_wmem(name=w)
    d = cplx(np.asarray(VZ(soccfg, reps=2, final_delay=1.0, cfg=dict(freq=F, ro_len=1.4)).acquire_decimated(soc, rounds=4, progress=False)[0]))
    m, good = {}, True
    for rep in range(2):
        z = d[rep]
        r = runs(np.abs(z), min_len=3)
        if len(r) != 4:
            m[f"rep{rep}"] = f"{len(r)} pulses"; good = False; continue
        c = np.array([z[int((a + b) / 2)] for a, b in r])
        ph = (np.degrees(np.angle(c * np.conj(c[0]))) + 360) % 360
        err = (ph - np.array([0, 90, 90, 180]) + 180) % 360 - 180
        m[f"rep{rep}_phases_deg"] = ph.round(2).tolist()
        good &= bool(np.max(np.abs(err)) < 2.0)

    class SUB(Base):
        def setup(s, cfg):
            const(s, "p", 0.1, gain=0.5)
            sub = AsmV2()
            sub.pulse(ch=G, name="p", t=0); sub.delay(0.15); sub.pulse(ch=G, name="p", t=0)
            s.add_subroutine("pair", sub)
        def _body(s, cfg):
            s.trigger(ros=[R], t=0.05)
            s.delay(0.1)
            s.call("pair"); s.delay(0.2); s.call("pair")
    z = cplx(dec(SUB(soccfg, reps=1, final_delay=1.0, cfg=dict(freq=F, ro_len=1.2)), rounds=2))
    r = runs(np.abs(z))
    st = np.array([a for a, b in r]) * dt_ro()
    exp = np.array([0, 0.15, 0.35, 0.50])
    m['subroutine_starts_us'] = (st - st[0]).round(4).tolist() if len(r) == 4 else f"{len(r)} pulses"
    good &= len(r) == 4 and bool(np.all(np.abs((st - st[0]) - exp) < 3 * dt_ro()))

    class GM(Base):
        def setup(s, cfg):
            for nm, g in (("f", 0.8), ("h", 0.4), ("q", 0.2)):
                const(s, nm, 0.12, gain=g)
        def _body(s, cfg):
            s.trigger(ros=[R], t=0.05)
            s.pulse(ch=G, name="f", t=0.1); s.pulse(ch=G, name="h", t=0.35); s.pulse(ch=G, name="q", t=0.6)
    z = cplx(dec(GM(soccfg, reps=1, final_delay=1.0, cfg=dict(freq=F, ro_len=1.0)), rounds=4))
    r = runs(np.abs(z), frac=0.1)
    amps = np.array([abs(plateau(z, a, b, guard=4)) for a, b in r])
    ratio = (amps / amps[0]).round(4).tolist() if len(amps) == 3 else f"{len(amps)} pulses"
    m['gain_ratios'] = ratio
    good &= len(amps) == 3 and bool(np.all(np.abs(amps / amps[0] - np.array([1, 0.5, 0.25])) < 0.01))
    return ok(good), m, "phases 0/90/90/180 within 2 deg on both reps (reset works); subroutine pulse starts within 3 samples; gain ratios 1/0.5/0.25 within 1%"

# ---------------------------------------------------------------- tProc arithmetic and branching
@test("T15", "tProc ALU, ARITH, DIV and conditional-jump tests", "05 (registers); tProc v2 TRM; 04_Reading_Math_Writing (v1)")
def t15():
    A, B = 123456, -789
    BASE = 700                     # literal dmem addresses are 10-bit (< 1024)
    ops = [("+", A + B), ("-", A - B), ("AND", A & B), ("OR", A | B), ("XOR", A ^ B)]
    shifts = [("SL", 3, (A << 3) & 0xFFFFFFFF), ("SR", 3, A >> 3), ("ASR", 3, B >> 3)]
    one = [("ABS", B, abs(B)), ("NOT", A, ~A)]
    tests = [("Z", 0, True), ("Z", 5, False), ("NZ", 5, True), ("S", -3, True), ("S", 3, False), ("NS", 3, True)]

    class P(Base):
        def setup(s, cfg):
            for r in ("ra", "rb", "rd", "rc", "rt"):
                s.add_reg(r)
        def _body(s, cfg):
            ra, rb, rd, rc, rt = (s._get_reg(x) for x in ("ra", "rb", "rd", "rc", "rt"))
            s.write_reg("ra", A); s.write_reg("rb", B)
            k = BASE
            for op, _ in ops:
                s.asm_inst({'CMD': 'REG_WR', 'DST': rd, 'SRC': 'op', 'OP': f'{ra} {op} {rb}'})
                s.write_dmem(addr=k, src="rd"); k += 1
            for op, n, _ in shifts:
                src = rb if op == "ASR" else ra
                s.asm_inst({'CMD': 'REG_WR', 'DST': rd, 'SRC': 'op', 'OP': f'{src} {op} #{n}'})
                s.write_dmem(addr=k, src="rd"); k += 1
            for op, v, _ in one:
                s.asm_inst({'CMD': 'REG_WR', 'DST': rd, 'SRC': 'op', 'OP': f'{op} {rb if v == B else ra}'})
                s.write_dmem(addr=k, src="rd"); k += 1
            # ARITH T: A*B -> s_arith_l ; DIV A/7 -> s_div_q, s_div_r
            s.write_reg("rc", 7)
            s.asm_inst({'CMD': 'ARITH', 'C_OP': 'T', 'R1': ra, 'R2': rb})
            for _ in range(10):
                s.nop()
            s.write_dmem(addr=k, src="s_arith_l"); k += 1
            s.asm_inst({'CMD': 'DIV', 'NUM': ra, 'DEN': rc})
            for _ in range(40):
                s.nop()
            s.write_dmem(addr=k, src="s_div_q"); k += 1
            s.write_dmem(addr=k, src="s_div_r"); k += 1
            for i, (tst, v, _) in enumerate(tests):
                s.write_reg("rt", v)
                s.write_dmem(addr=k, src=0)
                s.cond_jump(f"J{i}", "rt", tst)
                s.jump(f"N{i}")
                s.label(f"J{i}")
                s.write_dmem(addr=k, src=1)
                s.label(f"N{i}")
                k += 1
            s.trigger(ros=[R], t=0.1)
    n = len(ops) + len(shifts) + len(one) + 3 + len(tests)
    load_dmem(np.full(n, -12345, np.int32), BASE)
    P(soccfg, reps=1, final_delay=1.0, cfg=dict(freq=F, ro_len=0.1)).acquire(soc, rounds=1, progress=False)
    got = np.asarray(soc.read_mem(length=n, mem_sel='dmem', addr=BASE)).astype(np.int64).reshape(-1)
    exp = [v for _, v in ops] + [v for _, _, v in shifts] + [v for _, _, v in one] + [A * B, A // 7, A % 7] + [int(t) for _, _, t in tests]
    exp32 = [int(np.int64(v).astype(np.int32)) for v in exp]
    got32 = [int(np.int64(v).astype(np.int32)) for v in got]
    names = [o for o, _ in ops] + [f"{o}#{n_}" for o, n_, _ in shifts] + [o for o, _, _ in one] + ["ARITH A*B", "DIV q", "DIV r"] + \
        [f"jump {t} ({v})" for t, v, _ in tests]
    bad = {nm: dict(got=g, exp=e) for nm, g, e in zip(names, got32, exp32) if g != e}
    return ok(not bad), dict(n_checks=n, mismatches=bad), "every result written to dmem equals Python's (32-bit)"

# ---------------------------------------------------------------- 08 buffers
@test("T16", "DDR4 buffer: capture, shot period (final_delay), equality with decimated buffer", "08 §3; 09 §5.2")
def t16():
    class P(Base):
        def setup(s, cfg): const(s, "p", 0.2, gain=0.5)
        def _body(s, cfg):
            s.pulse(ch=G, name="p", t=0.2)
            s.trigger(ros=[R], t=0.2 + LAT - 0.1, ddr4=True)
    reps, fd = 6, 5.0
    nt = int(np.ceil(reps * (fd + 1.5) / dt_ro() / 256)) + 4      # the whole run, no stale tail
    soc.arm_ddr4(ch=R, nt=nt)
    prog = P(soccfg, reps=reps, final_delay=fd, cfg=dict(freq=F, ro_len=0.5))
    prog.acquire_decimated(soc, rounds=1, progress=False)
    ddr = np.asarray(soc.get_ddr4(nt)).reshape(-1, 2).astype(np.int64)
    L = prog.ro_chs[R]['length']
    lastdec = np.asarray(soc.get_decimated(ch=R, address=0, length=reps * L)).reshape(-1, 2).astype(np.int64)
    mag = np.abs(cplx(ddr))
    r = runs(mag)
    starts = np.array([a for a, b in r]) * dt_ro()
    per = np.diff(starts)
    # exact match of the last shot's decimated samples somewhere in the DDR4 stream
    tgt = lastdec[(reps - 1) * L:reps * L]
    hit = None
    for off in range(0, len(ddr) - L):
        if ddr[off, 0] == tgt[0, 0] and np.array_equal(ddr[off:off + L], tgt):
            hit = off
            break
    m = dict(ddr_samples=len(ddr), pulses_found=len(r), period_us=per.round(4).tolist() if len(per) else [],
             last_shot_found_exactly=hit is not None)
    good = len(r) == reps and np.ptp(per) < 3 * dt_ro() and per[0] > fd and hit is not None
    return ok(good), m, "every shot's pulse in DDR4 at a constant period (> final_delay); a shot's decimated samples found bit-exactly in the DDR4 stream"

@test("T17", "MR buffer: raw ADC capture, tone frequency", "08 §4, §6")
def t17():
    class P(Base):
        def setup(s, cfg): const(s, "p", 1.5, gain=0.5)
        def _body(s, cfg):
            s.pulse(ch=G, name="p", t=0.2)
            s.trigger(ros=[R], t=0.2 + LAT + 0.1, mr=True)
    soc.arm_mr(ch=R)
    P(soccfg, reps=1, final_delay=1.0, cfg=dict(freq=F, ro_len=1.0)).acquire_decimated(soc, rounds=1, progress=False)
    mr = np.asarray(soc.get_mr()).reshape(-1, 2).astype(float)
    fs_doc = soccfg['readouts'][R]['fs']          # what get_time_axis_mr() assumes
    res = {}
    for nm, fs in (("doc_fs", fs_doc), ("half_fs", fs_doc / 2)):
        x = mr[:, 0] - mr[:, 0].mean()
        sp = np.abs(np.fft.rfft(x * np.hanning(len(x))))
        fr = np.fft.rfftfreq(len(x), 1 / fs)
        res[nm] = float(fr[np.argmax(sp[5:]) + 5])
    zc = (mr[:, 0] - mr[:, 0].mean()) + 1j * (mr[:, 1] - mr[:, 1].mean())
    spc = np.abs(np.fft.fft(zc * np.hanning(len(zc))))
    frc = np.fft.fftfreq(len(zc), 2 / fs_doc)
    ipk = int(np.argmax(spc))
    pkc = float(frc[ipk])
    # quadrature check: the tone's mirror (-f) in the complex spectrum, relative to the tone.
    # (Was |corr(I, Q)| < 0.2, which depends on the tone's phase in a finite window and so on
    # the bitstream load: vanilla 0.14, L7 -0.01, L8 -0.23 -- 2026-10-13.)
    imir = int(np.argmin(np.abs(frc + pkc)))
    mirror_db = float(20 * np.log10(spc[max(imir - 2, 0):imir + 3].max() / spc[ipk]))
    quad = float(np.corrcoef(mr[:, 0], mr[:, 1])[0, 1])
    m = dict(rows=len(mr), peak_MHz_if_doc_fs=round(res['doc_fs'], 2), peak_MHz_if_half_fs=round(res['half_fs'], 2),
             complex_peak_MHz_half_fs=round(pkc, 2), mirror_dBc=round(mirror_db, 1), iq_correlation=round(quad, 3), played_MHz=F)
    bin_ = fs_doc / 2 / len(mr)
    data_ok = abs(res['half_fs'] - F) < 2 * bin_ and abs(abs(pkc) - F) < 2 * bin_ and mirror_db < -20
    status = "WARN" if data_ok else "FAIL"
    return status, m, ("MR data valid: two quadrature columns at fs/2 = %.2f Msps with the tone at %.0f MHz. QICK's "
                       "soccfg text (8192 samples = 1.852 us) and get_time_axis_mr() assume fs = %.2f Msps, so a user "
                       "following the tutorial sees the tone at 2x. Documentation/software mismatch, not our change" % (fs_doc / 2, F, fs_doc))

@test("T18", "Edge counter vs software Schmitt trigger", "09_photon_counting (v1 demo; buffer has edge counter)")
def t18():
    class P(AveragerProgramV2):
        def _initialize(s, cfg):
            s.declare_gen(ch=G, nqz=1)
            s.declare_readout(ch=R, length=2.0, edge_counting=True,
                              high_threshold=cfg['hi'], low_threshold=cfg['lo'])
            s.add_readoutconfig(ch=R, name="ro", freq=F, gen_ch=G)
            s.send_readoutconfig(ch=R, name="ro", t=0)
            const(s, "p", 0.06, gain=0.5)
        def _body(s, cfg):
            s.trigger(ros=[R], t=0.05)
            for k in range(cfg['n']):
                s.pulse(ch=G, name="p", t=0.1 + 0.25 * k)
    # rotate onto +I first: the counter looks at I
    W = window_prog()
    z0 = cplx(np.asarray(W(soccfg, reps=10, final_delay=1.0, cfg=dict(freq=F, ro_len=0.4)).acquire(soc, rounds=1, progress=False)[0])).reshape(-1)[0]
    out, good = {}, True
    for n in (0, 3, 5):
        cfg = dict(freq=F, n=n, hi=int(abs(z0) * 0.5), lo=int(abs(z0) * 0.2))

        class PR(P):
            def _initialize(s, cfg):
                s.declare_gen(ch=G, nqz=1)
                s.declare_readout(ch=R, length=2.0, edge_counting=True, high_threshold=cfg['hi'], low_threshold=cfg['lo'])
                s.add_readoutconfig(ch=R, name="ro", freq=F, gen_ch=G, phase=-np.degrees(np.angle(z0)))
                s.send_readoutconfig(ch=R, name="ro", t=0)
                const(s, "p", 0.06, gain=0.5)
        prog = PR(soccfg, reps=1, final_delay=1.0, cfg=cfg)
        d = np.asarray(prog.acquire_decimated(soc, rounds=1, progress=False)[0])
        cnt = int(np.asarray(prog.get_raw()[0]).reshape(-1)[0])
        i = d[:, 0]
        sw, armed = 0, True
        for v in i:
            if armed and v > cfg['hi']:
                sw += 1; armed = False
            elif not armed and v < cfg['lo']:
                armed = True
        out[f"n{n}"] = dict(hw=cnt, sw=sw)
        good &= cnt == n and sw == n
    return ok(good), out, "hardware edge count == number of pulses == software Schmitt count on the decimated I trace"

@test("T19", "Untestable on this wiring: gen 1, readouts 1/2 (weighted buffer), time tagger, PMOD pins", "06/07/08; time tagger")
def t19():
    return "NOT WIRED", dict(gen1="DAC_A not cabled", readout1="ADC_C not cabled", readout2_weighted="ADC_B not cabled",
                             time_tagger="ADC_A not cabled", pmod="needs a scope/logic analyser"), \
        "connect a loopback (e.g. gen 1 -> readout 1) to cover these"

# ---------------------------------------------------------------- integrity
CW_DF = 0.5                                        # readout detuned 0.5 MHz from the tone: slow rotation

def cw_prog():
    class P(AveragerProgramV2):
        def _initialize(s, cfg):
            s.declare_gen(ch=G, nqz=1)
            s.declare_readout(ch=R, length=cfg['ro_len'])
            s.add_readoutconfig(ch=R, name="ro", freq=F + CW_DF, gen_ch=G)
            s.send_readoutconfig(ch=R, name="ro", t=0)
            s.add_pulse(ch=G, name="cw", style="const", freq=F, ro_ch=R, length=0.5, phase=0, gain=0.5, mode="periodic")
            s.add_pulse(ch=G, name="stop", style="const", freq=F, ro_ch=R, length=0.05, phase=0, gain=0.0)
        def _body(s, cfg):
            s.pulse(ch=G, name="cw", t=0)
            s.trigger(ros=[R], t=2.0, ddr4=cfg['ddr4'])
            s.delay(cfg['hold'])
            s.pulse(ch=G, name="stop", t=0)
    return P

def phase_jumps(x):
    """Phase-continuity check of a detuned CW capture. Returns (samples, jump positions,
    implied missing samples per jump, amplitude dropouts, median step)."""
    a = np.abs(x)
    idx = np.flatnonzero(a > 0.5 * np.median(a))
    seg = x[idx[0] + 50: idx[-1] - 50]
    d = np.angle(seg[1:] * np.conj(seg[:-1]))
    med = np.median(d)
    sd = 1.4826 * np.median(np.abs(d - med))
    bad = np.flatnonzero(np.abs(d - med) > 10 * sd)
    miss = np.round(((d[bad] - med + np.pi) % (2 * np.pi) - np.pi) / med).astype(int)
    drop = int(np.sum(np.abs(seg) < 0.5 * np.median(np.abs(seg))))
    return len(seg), bad + idx[0] + 50, miss, drop, med

@test("T20", "Readout-chain integrity: CW tone in the decimated buffer, phase-continuity glitch search", "integrity check (avg_buffer decimated path)")
def t20():
    P = cw_prog()
    tot, nj, nd, med = 0, 0, 0, 0.0
    for _ in range(8):
        x = cplx(dec(P(soccfg, reps=1, final_delay=1.0, cfg=dict(ro_len=28.9, ddr4=False, hold=40.0))))
        n, b, miss, drop, med = phase_jumps(x)
        tot += n; nj += len(b); nd += drop
    exp = 2 * np.pi * CW_DF * dt_ro()
    m = dict(samples=tot, phase_jumps=nj, amp_dropouts=nd, step_rad=round(float(abs(med)), 6), step_expected=round(exp, 6))
    return ok(nj == 0 and nd == 0 and abs(abs(med) - exp) < 0.05 * exp), m, \
        "8 x 16k consecutive samples of a detuned CW tone through the readout chain: no phase jump > 10 sd, no dropout, rotation = detuning"

@test("T20b", "DDR4 stream continuity: CW tone through DDR4", "08 §3 (DDR4) + integrity check")
def t20b():
    P = cw_prog()
    nt = 2 ** 14                                     # 4.2 M samples = 7.6 ms per capture
    tot, pos, miss_all = 0, [], []
    for _ in range(3):
        soc.arm_ddr4(ch=R, nt=nt)
        P(soccfg, reps=1, final_delay=1.0, cfg=dict(ro_len=10.0, ddr4=True, hold=9000.0)).acquire_decimated(soc, rounds=1, progress=False)
        x = cplx(np.asarray(soc.get_ddr4(nt)).reshape(-1, 2))
        n, b, miss, drop, med = phase_jumps(x)
        tot += n; pos += list(b); miss_all += list(miss)
    pos, miss_all = np.array(pos), np.array(miss_all)
    m = dict(samples=tot, jumps=len(pos), per_million=round(len(pos) / tot * 1e6, 1),
             missing_samples_hist={int(k): int(v) for k, v in zip(*np.unique(miss_all, return_counts=True))} if len(pos) else {},
             jump_position_mod16={int(k): int(v) for k, v in zip(*np.unique(pos % 16, return_counts=True))} if len(pos) else {})
    return ok(len(pos) == 0), m, ("~12 M consecutive DDR4 samples: no discontinuity. A failure with jumps that each skip ~16 samples at "
                                  "16-aligned positions means the DDR4 write path drops whole 16-sample beats")

@test("T21", "Program size and timing limits", "09 §2")
def t21():
    class P(Base):
        def setup(s, cfg): const(s, "p", 0.012, gain=0.3)
        def _body(s, cfg):
            s.trigger(ros=[R], t=0.05)
            for k in range(cfg['n']):
                s.pulse(ch=G, name="p", t=0.1 + 0.028 * k)
    prog = P(soccfg, reps=1, final_delay=1.0, cfg=dict(freq=F, ro_len=29.0, n=1000))
    prog.compile()
    size = len(prog.binprog['pmem'])
    z = cplx(dec(prog, rounds=1))
    r = runs(np.abs(z), min_len=3)
    m = dict(pmem_words=size, pmem_size=soccfg['tprocs'][0]['pmem_size'], pulses_programmed=1000, pulses_seen=len(r))
    return ok(len(r) == 1000), m, "a ~1000-pulse program fits program memory and every pulse is played"

# ---------------------------------------------------------------- driver
def main():
    global soc, soccfg
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="comma-separated test ids")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out", default="campaign_v140L1/qick_functional_test.json")
    a = ap.parse_args()
    if a.list:
        for tid, name, src, _ in TESTS:
            print(f"{tid}  {name}   [{src}]")
        return
    soc, soccfg = D.connect()
    print(str(soccfg).split('\n')[2])
    has_gate = hasattr(soc, 'gate_present') and soc.gate_present()
    w0 = int(soc.gate_stats()[0]) if has_gate else 0
    if w0 & 1:
        raise SystemExit(f"dac_gate is ENABLED (word 0 = {w0:#x}); disable it first")
    only = set(a.only.split(",")) if a.only else None
    if only and "T03" not in only:
        only.add("T03")                          # provides the pulse latency other tests use
    results = []
    for tid, name, src, fn in TESTS:
        if only and tid not in only:
            continue
        t0 = time.time()
        try:
            status, metrics, crit = fn()
        except Exception as e:
            status, metrics, crit = "ERROR", dict(error=f"{type(e).__name__}: {e}", tb=traceback.format_exc()[-1500:]), ""
        dt = time.time() - t0
        results.append(dict(id=tid, name=name, source=src, status=status, criterion=crit, metrics=metrics, seconds=round(dt, 1)))
        print(f"[{status:9s}] {tid} {name} ({dt:.0f} s)\n            {json.dumps(metrics, default=str)[:600]}", flush=True)
    w1 = int(soc.gate_stats()[0]) if has_gate else 0
    summary = {s: sum(r['status'] == s for r in results) for s in ("PASS", "WARN", "FAIL", "ERROR", "NOT WIRED", "INFO")}
    print("\nSUMMARY", summary, f"| dac_gate word 0 before {w0:#x} after {w1:#x}")
    Path(ROOT / a.out).write_text(json.dumps(dict(firmware=str(soccfg).split('\n')[2], summary=summary, results=results),
                                             indent=1, default=str))
    print("saved", a.out)

if __name__ == "__main__":
    main()
