"""
Acquire the exact input window NN_axi classifies, with its on-chip sigmoid, using NN_axi's capture mode.

Arming is via the window_offset AXI-lite register (CAPTURE_ARM = 0xC0DE), which the datapath otherwise ignores:

    soc.nn_configure(window_size, CAPTURE_ARM, scaling_factor)   # arm
    <fire one shot>
    soc.nn_read_bram(i*4) for i in 0..770                         # read back

BRAM layout of one armed shot (NN_axi.cpp):
    words [0 .. 769] = the raw packed 32-bit stream words, carried through the float BRAM word as bits.
                       Unpack, do not float-decode:
                         in_local[2i]   = (w & 0xFFFF) * scaling_factor
                         in_local[2i+1] = (w >> 16)    * scaling_factor
    word  [770]      = the sigmoid NN() produced from those values (a float)

For each dataset it saves the captured inputs, the chip's sigmoid, the label and the waveform that was played,
so the received input can be compared with the one sent and the network recomputed offline from the capture.
"""
import os, sys, gc, struct, argparse
sys.path.insert(0, ".")
sys.path.insert(0, "tools")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import Pyro4
import decode_taps as DT      # tap slot map + bit decode

Pyro4.config.SERIALIZER = "pickle"
Pyro4.config.PICKLE_PROTOCOL_VERSION = 4

from qick import QickConfig
from qick.asm_v2 import AveragerProgramV2

CAPTURE_ARM = 0xC0DE          # must match NN_axi.h

# Layer taps (slots 771..788) exist only in firmware built with NN()'s debug ports; the deployed NN_axi has none,
# so `taps=True` raises unless QICK_TAPS_PRESENT=1. Slots 0..769 (capture) and 770 (sigmoid) are always written.
TAPS_PRESENT = os.environ.get("QICK_TAPS_PRESENT", "0") == "1"
N_WORDS = 770                 # N_IQ_WINDOW_IN packed 32-bit stream words
N_CAP = 2 * N_WORDS           # 1540 unpacked I/Q values
SIG_SLOT = N_WORDS            # sigmoid slot in capture mode
BRAM_WORDS = 2048

# BRAM layout (NN_axi.cpp): slots [0 .. 769] raw packed stream words carried as bits (read as integers,
# in_local[2i] = (word & 0xFFFF) * scaling_factor, in_local[2i+1] = (word >> 16) * scaling_factor);
# slot [770] the sigmoid (a float).

NS_HOST, NS_PORT, PROXY_NAME = os.environ.get("QICK_NS_HOST", "10.1.9.26"), 8888, "myqick"
ns = Pyro4.locateNS(host=NS_HOST, port=NS_PORT)
soc = Pyro4.Proxy(ns.lookup(PROXY_NAME))
soccfg = QickConfig(soc.get_cfg())
for line in str(soccfg).splitlines():
    if "built" in line:
        print("BANNER:", line.strip())

# NN_TRIGGER_PIN is fired with the readout to be classified (pins=[0, NN_TRIGGER_PIN]): trig_11 (PMOD0_7), the
# classify trigger that starts NN_live's window.
soccfg['tprocs'][0]['output_pins'].append(('trig', 11, 0, 'NN classify trigger (trig_11_o)'))
NN_TRIGGER_PIN = len(soccfg['tprocs'][0]['output_pins']) - 1

import onchip_test_harness as h

# drain_count_last packs the idle-flush count in bits 31:16; decode it, do not read it raw.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import onset_diag as OD

DATASETS = [
    ("x_50k",                   "x_50k.npy",                   "y_50k.npy"),
    ("x_50k_heldout",           "x_50k_heldout.npy",           "y_50k_heldout.npy"),
    ("x_50k_heldout_measnoise", "x_50k_heldout_measnoise.npy", "y_50k_heldout_measnoise.npy"),
    ("test20k_real_X",          "data/test20k_real_X.npy",          "data/test20k_real_y.npy"),
]


class Prog(AveragerProgramV2):
    def __init__(s, sc, cf):
        super().__init__(sc, reps=cf["reps"], final_delay=cf["final_delay"], cfg=cf)

    def _initialize(s, cf):
        s.declare_gen(ch=cf["gen_ch"], nqz=cf["nqz"])
        s.declare_readout(ch=cf["ro_ch"], length=cf["readout_length"])
        s.add_envelope(ch=cf["gen_ch"], name="e",
                       idata=cf["waveform_I"], qdata=cf["waveform_Q"])
        s.add_pulse(ch=cf["gen_ch"], name="p", style="arb", envelope="e",
                    gain=cf["gain"], freq=cf["freq"], phase=cf["phase"],
                    outsel=cf["outsel"], mode=cf["mode"], stdysel=cf["stdysel"],
                    ro_ch=cf["ro_ch"])
        s.add_readoutconfig(ch=cf["ro_ch"], name="r", freq=cf["freq"], gen_ch=cf["gen_ch"])

    def _body(s, cf):
        s.pulse(ch=cf["gen_ch"], name="p")
        s.send_readoutconfig(ch=cf["ro_ch"], name="r", t=0)
        s.trigger(ros=[cf["ro_ch"]], pins=[0, NN_TRIGGER_PIN],
                  t=cf["adc_trig_offset"], width=10)


CFG0 = {"gen_ch": 0, "ro_ch": 0, "nqz": 1, "reps": 1, "final_delay": 100,
        "readout_length": 1.393, "adc_trig_offset": 0.31, "length": 16,
        "freq": 50, "mode": "oneshot", "stdysel": "zero", "outsel": "product",
        "gain": 1, "phase": 0.0}


def word_to_float(raw):
    return struct.unpack("f", struct.pack("I", raw & 0xFFFFFFFF))[0]


def read_capture(stride=1, scaling_factor=1, taps=False):
    """Read back one armed shot: unpacked I/Q values + the sigmoid.

    Returns (values, prob) where `values` are the in_local entries the core loaded, i.e. (uint16 half *
    scaling_factor), interleaved I,Q,I,Q. With taps=True returns (values, prob, tapwords): the 18 raw tap words
    from slots 771..788, undecoded (decode_taps.decode does that).

    Each BRAM word is a separate Pyro round-trip, so a full record costs 771 reads, or 789 with taps.
    `stride` subsamples whole I/Q pairs for callers that only need a summary statistic.
    """
    idx = np.arange(0, N_WORDS, stride)
    vals = np.empty(2 * len(idx), dtype=np.float64)
    for n, i in enumerate(idx):
        w = soc.nn_read_bram(int(i) * 4) & 0xFFFFFFFF
        vals[2 * n] = (w & 0xFFFF) * scaling_factor
        vals[2 * n + 1] = ((w >> 16) & 0xFFFF) * scaling_factor
    prob = word_to_float(soc.nn_read_bram(SIG_SLOT * 4))
    if not taps:
        return vals, prob
    if not TAPS_PRESENT:
        raise SystemExit(
            "taps=True requested, but this firmware has no debug taps: "
            "slots 771+ are not tap data.\nScoring does not need them -- use "
            "score_armed_fidelity.py, which reads only slot 770.\nTaps need "
            "firmware built with NN()'s debug ports plus QICK_TAPS_PRESENT=1.")
    tw = np.array([soc.nn_read_bram((DT.DBG_BASE + i) * 4) & 0xFFFFFFFF
                   for i in range(DT.N_DBG_TOTAL)], dtype=np.uint32)
    # in_local readbacks (firmware with debug ports only): PRE = in_local[0..1001] after LOAD_L, read word for
    # word; RAW = LOAD_L's own read-back of a pair written RAW_LAG iterations earlier.
    itail = np.array([soc.nn_read_bram((DT.PRE_TAIL_BASE + i) * 4) & 0xFFFFFFFF
                      for i in range(DT.N_PRE_TAIL_WORDS)], dtype=np.uint32)
    ihead = np.array([soc.nn_read_bram((DT.PRE_HEAD_BASE + i) * 4) & 0xFFFFFFFF
                      for i in range(DT.N_PRE_HEAD_WORDS)], dtype=np.uint32)
    idense = np.array([soc.nn_read_bram((DT.DENSE_IN_BASE + i) * 4) & 0xFFFFFFFF
                       for i in range(DT.N_DENSE_IN)], dtype=np.uint32)
    iraw = np.array([soc.nn_read_bram((DT.RAW_BASE + i) * 4) & 0xFFFFFFFF
                     for i in range(DT.N_RAW_RB)], dtype=np.uint32)
    ipre0 = np.array([soc.nn_read_bram((DT.PRELOAD_BASE + i) * 4) & 0xFFFFFFFF
                      for i in range(DT.N_PRELOAD_WORDS)], dtype=np.uint32)
    return vals, prob, tw, itail, ihead, idense, iraw, ipre0


def to_signed(cap, sf):
    """Undo the uint16 bit-reinterpretation so amplitude is monotone again."""
    s = np.asarray(cap, dtype=np.float64) / max(sf, 1)
    return np.where(s >= 32768, s - 65536, s)


def measure_readback_lag(k, c, sf, n=6):
    """Measure, don't assume, WHICH fired shot each BRAM record belongs to.

    The scored path in tools/onchip_test_harness.py pairs pred[1:] against y[:-1] --
    a one-shot readback lag (harness lines 306/345). If that lag is also
    present here, then caps[i]/probs[i] describe the shot fired at i-1, and
    every comparison against raw_X[i], sibling[i] or labels[i] is silently
    off by one. (Test A -- offline NN reconstruction vs the chip's own
    sigmoid -- is immune either way, because LOAD_L's input mirror and
    STORE_L's sigmoid are written by the same TRIGGER_C pass.)

    Rather than inherit the harness's +1 on faith, fire ONE fixed waveform at
    a deliberately distinctive sequence of amplitudes. The captured window's
    signed mean|.| is then monotone in the amplitude of whichever shot it
    actually came from, so the lag falls straight out of the alignment.
    """
    amps = np.array([1.00, 0.30, 0.75, 0.15, 0.55, 0.95])[:n]
    X = np.load(DATASETS[0][1])[:1]
    base = h.prepare_chunk(X, k, c)[0]        # one fixed shape, amplitude varied

    print(f"\n=== readback-lag probe: {n} shots, amplitudes {amps.tolist()} ===")
    got = np.zeros(n)
    for i in range(n):
        soc.nn_configure(10000, CAPTURE_ARM, sf)
        wf = h.prepare_waveform(base * amps[i])
        cfg = dict(CFG0, waveform_I=wf[0::2], waveform_Q=wf[1::2])
        prog = Prog(soccfg, cfg)
        prog.acquire_decimated(soc)
        cap, _ = read_capture(stride=8, scaling_factor=sf)  # mean only
        got[i] = np.abs(to_signed(cap, sf)).mean()
        del prog, cfg, wf
        gc.collect()
        print(f"  fired amp={amps[i]:.2f}  captured mean|in|={got[i]:9.2f}")

    best, scores = None, {}
    for lag in (0, 1, 2):
        if n - lag < 3:
            continue
        a_, g_ = amps[:n - lag], got[lag:]
        if a_.std() == 0 or g_.std() == 0:
            continue
        r = float(np.corrcoef(a_, g_)[0, 1])
        scores[lag] = r
        print(f"  lag={lag}: corr(fired amp, captured mean) = {r:+.4f}")
        if best is None or r > scores[best]:
            best = lag
    if best is None:
        print("  -> lag probe INCONCLUSIVE (no usable correlation -- captured "
              "means may be constant). Falling back to the harness's assumed 1; "
              "treat alignment-dependent results from this run as provisional.")
        best = 1
    print(f"  -> READBACK LAG = {best} shot(s)"
          f"  (harness assumes 1; {'AGREES' if best == 1 else 'DISAGREES'})")
    return best, scores, amps, got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=10,
                    help="captured shots per dataset (each is 1541 Pyro reads)")
    ap.add_argument("--chunk", type=int, default=500,
                    help="rows fed to prepare_chunk before slicing to --shots. "
                         "prepare_chunk normalises by the peak over the whole "
                         "chunk, so this must equal the chunk size of the run "
                         "being compared (500).")
    ap.add_argument("--scaling-factor", type=int, default=1)
    ap.add_argument("--onset-threshold", type=int, default=40)
    ap.add_argument("--lag-probe-shots", type=int, default=6,
                    help="shots used to MEASURE the BRAM readback lag before "
                         "capturing anything (see measure_readback_lag)")
    ap.add_argument("--force-lag", type=int, default=None,
                    help="skip the measured lag and use this one "
                         "(onchip_test_harness.py assumes 1: pred[1:] vs y[:-1])")
    ap.add_argument("--out", default="classifier_input_captures.npz")
    a = ap.parse_args()

    k = complex(np.load("calibration/k.npy"))
    c = complex(np.load("calibration/c.npy"))
    print(f"k={k} c={c}  SF={a.scaling_factor}  onset={a.onset_threshold}")

    soc.nn_set_onset_threshold(a.onset_threshold)
    print("onset_threshold readback:", soc.nn_get_onset_threshold())

    store = {}
    lag, lag_scores, lag_amps, lag_got = measure_readback_lag(
        k, c, a.scaling_factor, n=a.lag_probe_shots)
    if a.force_lag is not None:
        print(f"  (overridden by --force-lag {a.force_lag})")
        lag = a.force_lag
    store["readback_lag"] = lag
    store["lag_probe_amps"] = lag_amps
    store["lag_probe_captured"] = lag_got
    store["lag_probe_scores"] = np.array(
        [[k_, v_] for k_, v_ in sorted(lag_scores.items())], dtype=float)

    for name, xf, yf in DATASETS:
        # fire `lag` extra shots so that after alignment we still have
        # a.shots records whose fired waveform / label are known
        n_fire = a.shots + lag
        Xfull = np.load(xf)[:a.chunk]
        # identical prep to the scored runs: normalise over the FULL chunk,
        # only then slice down to the shots we can afford to capture
        Dfull = h.prepare_chunk(Xfull, k, c)
        X = Xfull[:n_fire]
        Y = np.load(yf)[:n_fire].astype(int)
        D = Dfull[:n_fire]

        caps = np.zeros((len(D), N_CAP), dtype=np.float64)
        probs = np.zeros(len(D), dtype=np.float64)
        drains = np.zeros(len(D), dtype=np.int64)
        # the 18 raw tap words per shot, stored undecoded; decode_taps.decode() turns them into layer values
        tapw = np.zeros((len(D), DT.N_DBG_TOTAL), dtype=np.uint32)
        # in_local readback, stored undecoded
        tail = np.zeros((len(D), DT.N_PRE_TAIL_WORDS), dtype=np.uint32)
        head = np.zeros((len(D), DT.N_PRE_HEAD_WORDS), dtype=np.uint32)
        dense_in = np.zeros((len(D), DT.N_DENSE_IN), dtype=np.uint32)
        raw_rb = np.zeros((len(D), DT.N_RAW_RB), dtype=np.uint32)
        preload = np.zeros((len(D), DT.N_PRELOAD_WORDS), dtype=np.uint32)
        sibling = None   # what acquire_decimated() reads for the SAME shot

        print(f"\n=== {name}: capturing {len(D)} shots ===")
        for i in range(len(D)):
            # ARM capture for this shot
            soc.nn_configure(10000, CAPTURE_ARM, a.scaling_factor)
            wf = h.prepare_waveform(D[i])
            cfg = dict(CFG0, waveform_I=wf[0::2], waveform_Q=wf[1::2])
            prog = Prog(soccfg, cfg)
            iq = prog.acquire_decimated(soc)
            # Keep the sibling readout: NN_axi reads axis_avg_buffer_0/m1_axis through a broadcaster and FIFO,
            # and this DMA path reads the same source, so both can be compared for the same shot.
            rec = (np.concatenate([np.asarray(x).reshape(-1) for x in iq])
                   if isinstance(iq, list) else np.asarray(iq).reshape(-1))
            if sibling is None:
                sibling = np.zeros((len(D), rec.size), dtype=np.float64)
            if rec.size == sibling.shape[1]:
                sibling[i] = rec
            drains[i] = OD.read_onset_diag(soc)["drain"]
            (caps[i], probs[i], tapw[i], tail[i], head[i], dense_in[i],
             raw_rb[i], preload[i]) = read_capture(
                 scaling_factor=a.scaling_factor, taps=True)
            del prog, cfg, wf
            gc.collect()
            t = DT.decode(tapw[i])
            print(f"  shot {i+1}/{len(D)}: label={Y[i]} prob={probs[i]:.6f} "
                  f"drain={drains[i]} mean|in|={np.abs(caps[i]).mean():.1f} "
                  f"l2={np.round(t['l2'], 1).tolist()} "
                  f"l6={t['l6'][0]:.4f} l10={t['l10'][0]:.4f}")
            # Fail on the first shot if the taps are not present: layer10 and the sigmoid slot are written in the
            # same pass, so they must agree.
            if i == 0 and name == DATASETS[0][0]:
                want = 1.0 / (1.0 + np.exp(-t["l10"][0]))
                if abs(want - probs[i]) > 0.05:
                    raise SystemExit(
                        f"ABORT: tap slot layer10={t['l10'][0]:.4f} implies "
                        f"sigmoid {want:.4f} but the sigmoid slot holds "
                        f"{probs[i]:.6f}. The taps are not live -- is this "
                        f"actually the v1.28 bitstream?")

        # ALIGNMENT. caps[i]/probs[i] describe the shot fired at i-lag, so the
        # analysis must compare caps[lag:] against raw_X[:-lag] / labels[:-lag]
        # / fired[:-lag]. Store the raw unaligned arrays plus the lag, and the
        # aligned label vector ready to use, so nothing downstream has to
        # rediscover this.
        store[f"{name}__caps"] = caps
        store[f"{name}__probs"] = probs
        store[f"{name}__labels"] = Y
        store[f"{name}__drains"] = drains
        store[f"{name}__tapwords"] = tapw
        store[f"{name}__tail"] = tail
        store[f"{name}__head"] = head
        store[f"{name}__dense_in"] = dense_in
        store[f"{name}__raw_rb"] = raw_rb
        store[f"{name}__preload"] = preload
        store[f"{name}__raw_X"] = X
        # the waveform actually fired
        store[f"{name}__fired"] = D.astype(np.float32)
        # The taps are written right after the sigmoid in the same pass, so they carry the sigmoid's lag:
        # caps[i] pairs with probs[i+SIG_LAG] and tapwords[i+SIG_LAG]; the tapped layer10 must equal the sigmoid
        # slot's input.
        if lag:
            store[f"{name}__probs_aligned"] = probs[lag:]
            store[f"{name}__labels_aligned"] = Y[:-lag]
            store[f"{name}__caps_aligned"] = caps[lag:]
            store[f"{name}__tapwords_aligned"] = tapw[lag:]
            store[f"{name}__tail_aligned"] = tail[lag:]
            store[f"{name}__head_aligned"] = head[lag:]
            store[f"{name}__dense_in_aligned"] = dense_in[lag:]
            store[f"{name}__raw_rb_aligned"] = raw_rb[lag:]
            store[f"{name}__preload_aligned"] = preload[lag:]
        else:
            store[f"{name}__probs_aligned"] = probs
            store[f"{name}__labels_aligned"] = Y
            store[f"{name}__caps_aligned"] = caps
            store[f"{name}__tapwords_aligned"] = tapw
            store[f"{name}__tail_aligned"] = tail
            store[f"{name}__head_aligned"] = head
            store[f"{name}__dense_in_aligned"] = dense_in
            store[f"{name}__raw_rb_aligned"] = raw_rb
            store[f"{name}__preload_aligned"] = preload
        if sibling is not None:
            store[f"{name}__sibling"] = sibling

        # signed view: undo the uint16 bit-reinterpretation so the captured
        # window can be compared against the host-side int16 waveform
        signed = caps / max(a.scaling_factor, 1)
        signed = np.where(signed >= 32768, signed - 65536, signed)
        store[f"{name}__signed"] = signed
        print(f"  -> signed view: mean|.|={np.abs(signed).mean():.1f} "
              f"min={signed.min():.0f} max={signed.max():.0f}")

    # restore normal (non-capture) operation
    soc.nn_configure(10000, 0, a.scaling_factor)
    np.savez_compressed(a.out, scaling_factor=a.scaling_factor,
                        onset_threshold=a.onset_threshold, **store)
    print(f"\nsaved {a.out}")
    print("capture mode disarmed (window_offset back to 0)")


if __name__ == "__main__":
    main()
