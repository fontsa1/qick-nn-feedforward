"""Score NN_axi's on-chip fidelity through its capture mode, and NN_live's on the same shots.

Each shot arms capture and reads one BRAM word (the sigmoid, slot 770), instead of the full ~770-word capture, so a
few hundred shots per minute are affordable. Both label pairings (same shot and previous shot) are reported, so a
pipeline offset would show directly.
"""
import argparse
import sys
import numpy as np

sys.path.insert(0, "analysis_scripts/diagnostics")
sys.path.insert(0, "board_tools")
sys.path.insert(0, ".")
import calib                                    # noqa: E402  board-scoped k/c
import acquire_classifier_input_captures as A   # noqa: E402  (brings up soc)

soc, soccfg = A.soc, A.soccfg
h = A.h
Prog, CFG0 = A.Prog, A.CFG0
CAPTURE_ARM = A.CAPTURE_ARM
SIG_SLOT = A.SIG_SLOT


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    hw = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - hw), 100 * (c + hw))


def raw_peak(X, k, c, block=1000):
    """Peak |sample| prepare_chunk would produce for X before its DAC-fill
    scaling. Mirrors onchip_test_harness.prepare_chunk; blocked for memory."""
    from scipy.interpolate import interp1d
    m = 0.0
    xn = np.linspace(0, 1, 11000)
    for s in range(0, len(X), block):
        x = np.asarray(X[s:s + block])
        zs = (x[:, 0::2] + 1j * x[:, 1::2] - c) / k
        xo = np.linspace(0, 1, zs.shape[1])
        for part in (zs.real, zs.imag):
            m = max(m, float(np.abs(interp1d(xo, part, axis=1, kind="linear",
                                             fill_value="extrapolate")(xn)).max()))
    return m


# ---------------------------------------------------------------------------
# Stream-phase alignment. NN_axi's window grid sits at some phase R relative to each shot's buffer copy, so it
# can classify a circular rotation of the shot, chip[n] = shot[(n - R) mod 770]. R persists across runs and slips by
# one word whenever DRAIN_L drains an extra word below onset_threshold. Host-side handling:
#   * onset_threshold 0, so DRAIN_L never drains extra words;
#   * R is measured by circular cross-correlation of captured reference shots against loopback20k_real's
#     recording of the same waveforms;
#   * one get_decimated(delta) moves the grid by delta words, giving R = 0. Transfers are rounded up to an even
#     number of words (64-bit beats), so an odd delta lands at R = 1, within the +-1 tolerance.
# ---------------------------------------------------------------------------
REF_ROWS = (19990, 19991, 19992, 19993, 19994, 19995)


def _unpack_words(words):
    w = np.asarray(words, dtype=np.uint32)
    i = (w & 0xFFFF).astype(np.uint16).view(np.int16).astype(np.float64)
    q = (w >> 16).astype(np.uint16).view(np.int16).astype(np.float64)
    return i + 1j * q


def _rotation(zc, zl):
    """R such that zc[n] ~ zl[n - R] (circular), and the peak correlation."""
    a, b = zc - zc.mean(), zl - zl.mean()
    xc = np.fft.ifft(np.fft.fft(a) * np.conj(np.fft.fft(b)))
    m = np.abs(xc) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return int(m.argmax()), float(m.max())


def measure_rotation(k, c, n=3):
    """Fire n reference shots (test20k_real_X rows with a loopback20k_real
    recording) with capture armed; return [(R, peak_corr), ...]."""
    X = np.load("data/test20k_real_X.npy", mmap_mode="r")
    LB = np.load("data/loopback20k_real.npy", mmap_mode="r")
    rows = list(REF_ROWS[:n])
    D = h.prepare_chunk(np.asarray(X[rows]), k, c)
    L = soccfg.us2cycles(CFG0["readout_length"], ro_ch=CFG0["ro_ch"])
    out = []
    for j, r in enumerate(rows):
        soc.nn_configure(10000, CAPTURE_ARM, 1)
        wf = h.prepare_waveform(D[j])
        Prog(soccfg, dict(CFG0, waveform_I=wf[0::2], waveform_Q=wf[1::2])).acquire_decimated(soc)
        soc.get_decimated(CFG0["ro_ch"], 0, L)
        w = [soc.nn_read_bram(i * 4) & 0xFFFFFFFF for i in range(770)]
        zl = np.asarray(LB[r], dtype=float)
        out.append(_rotation(_unpack_words(w), zl[0::2] + 1j * zl[1::2]))
    return out


def _phase(m):
    good = [R for R, pk in m if pk > 0.3]
    if not good:
        return None
    # circular median: unwrap around the first value
    ref = good[0]
    d = [((R - ref + 385) % 770) - 385 for R in good]
    return int((ref + int(np.median(d))) % 770)


def align_stream(k, c, tries=3):
    """Bring the window grid to R = 0 (+-1). Returns (R_before, R_after)."""
    R0 = None
    for _ in range(tries):
        R = _phase(measure_rotation(k, c, 3))
        if R is None:
            continue
        if R0 is None:
            R0 = R
        if R in (769, 0, 1):
            return R0, R
        delta = (770 - R) % 770
        if delta < 8:                    # avoid tiny DMAs; same phase mod 770
            delta += 770
        soc.get_decimated(CFG0["ro_ch"], 0, delta)
    R = _phase(measure_rotation(k, c, 3))
    return R0, R


LIVE_MAGIC = 0x4E4E4C31          # neural_net4_live/firmware/NN_live.h


def report_live(Y, probs, hdr, rec, words=None, dec=None):
    """NN_live's per-shot records against the labels and NN_axi (same shots)."""
    n = len(Y)
    cnt = hdr[:, 0]
    logit = (rec[:, 5] & 0xFFFF).astype(np.uint16).view(np.int16) / 1024.0
    prob = (rec[:, 6] & 0x3FFFF) / 1024.0            # result_t, always >= 0
    dec_bit = (rec[:, 7] & 1).astype(int)
    fresh = rec[:, 0] == cnt - 1
    steps = np.diff(cnt)
    print("\nNN_live (live readout stream)")
    print(f"  magic {'ok' if (hdr[:, 2] == LIVE_MAGIC).all() else 'WRONG'}; newest record read "
          f"{int(fresh.sum())}/{n}; frame errors {int(hdr[0, 1])} -> {int(hdr[-1, 1])} over the run")
    if len(steps):
        u, k = np.unique(steps, return_counts=True)
        print(f"  live windows between reads: {dict(zip(u.tolist(), k.tolist()))} "
              "(1 per shot; each re-align check adds 3)")
    print(f"  decision bit == (probability > 0.5): {int((dec_bit == (prob > 0.5)).sum())}/{n}; "
          f"logit range {logit.min():.3f} .. {logit.max():.3f}")
    print(f"  vs NN_axi on the same shot: probability identical {int((probs == prob).sum())}/{n}, "
          f"decision agrees {int(((probs > 0.5) == (prob > 0.5)).sum())}/{n}")
    ok = dec_bit == Y
    g, e = Y == 0, Y == 1
    ga, ea = 100 * ok[g].mean(), 100 * ok[e].mean()
    lo, hi = wilson(int(ok.sum()), n)
    print(f"  FIDELITY (live) {0.5 * (ga + ea):6.2f}%   overall {100 * ok.mean():.2f}% "
          f"[{lo:.2f}, {hi:.2f}]   P(1|0) {100 - ga:.2f}%   P(0|1) {100 - ea:.2f}%")
    if words is not None:
        wi = (words & 0xFFFF).astype(np.uint16).view(np.int16).astype(np.int32)
        wq = (words >> 16).astype(np.uint16).view(np.int16).astype(np.int32)
        dec = dec[:, :words.shape[1]]           # NN_live may frame fewer samples (576)
        exact = ((wi == dec[:, :, 0]) & (wq == dec[:, :, 1])).all(axis=1)
        print(f"  live window == acquire_decimated() window, sample for sample: {int(exact.sum())}/{n}")
        best = []
        for s in np.flatnonzero(~exact):
            score = {d: int(((np.roll(wi[s], -d) == dec[s, :, 0]) &
                             (np.roll(wq[s], -d) == dec[s, :, 1]))[8:-8].sum())
                     for d in range(-6, 7)}
            best.append(max(score, key=score.get))
        if best:
            u, k = np.unique(best, return_counts=True)
            print(f"  mismatched shots, best shift (live index - buffer index): "
                  f"{dict(zip(u.tolist(), k.tolist()))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=300)
    ap.add_argument("--chunk", type=int, default=500)
    ap.add_argument("--dataset", default="test20k_real_X")
    ap.add_argument("--scaling-factor", type=int, default=1)
    # 0: any other value lets DRAIN_L drain extra words below threshold, and each one slips the window grid by
    # one word (see stream-phase alignment above). With 0 DRAIN_L takes exactly one word.
    ap.add_argument("--onset-threshold", type=int, default=0)
    ap.add_argument("--no-align", dest="align", action="store_false",
                    help="skip the stream-phase alignment (diagnostics only: "
                         "scores a rotated window, -0.2 to -1.2 points)")
    # the grid can still slip occasionally at threshold 0, so re-check periodically (~5 s per check)
    ap.add_argument("--realign-every", type=int, default=500,
                    help="re-measure R every N shots and re-align if off (0 = never)")
    # On by default. NN_axi is fed from axis_avg_buffer_0.m1_axis (the host-DMA readback), not the live readout
    # stream. The extra get_decimated() streams a second copy of the shot (770 words, a whole window, so it does
    # not move the phase) so the BRAM read always follows a complete window of the current shot.
    ap.add_argument("--extra-dma", dest="extra_dma", action="store_true",
                    default=True,
                    help="(default, kept for compatibility) advance past the "
                         "one-shot host-DMA readback lag")
    ap.add_argument("--no-extra-dma", dest="extra_dma", action="store_false",
                    help="suppress the lag fix and score the PREVIOUS shot; "
                         "for lag diagnostics only")
    ap.add_argument("--start", type=int, default=0,
                    help="first dataset row; scores rows [start, start+chunk)")
    # prepare_chunk scales every chunk so its loudest sample fills 99% of the DAC, so a shot's played amplitude
    # depends on the rest of its chunk (a 500-row chunk of test20k_real_X plays up to ~19% louder than the whole
    # file; board- and calibration-specific). "dataset" scales every chunk by the whole file's peak, so every shot
    # is played identically however the run is split, and split runs can be pooled. "chunk" scales per chunk.
    ap.add_argument("--norm", choices=("chunk", "dataset"), default="chunk")
    ap.add_argument("--capture", action="store_true",
                    help="also read back the 770 raw packed input words NN() "
                         "consumed for each shot (771 Pyro reads/shot; slow). "
                         "Saved as `words` in --save.")
    ap.add_argument("--save", default=None,
                    help="write per-shot idx/probs/labels to this .npz")
    # NN_live classifies every shot straight off the readout stream, beside NN_axi. Its per-shot record
    # (NN_live.h) is read after each shot and compared with NN_axi's answer for the same shot.
    ap.add_argument("--live", action="store_true",
                    help="also read NN_live's record for every shot (needs a "
                         "v1.40L1+ bitstream and the patched NNQickSoc)")
    ap.add_argument("--live-n", type=int, default=770,
                    help="samples NN_live frames per window (live_window_gate N): "
                         "770 through v1.40L2, 576 from v1.40L3")
    ap.add_argument("--live-capture", action="store_true",
                    help="with --live: also read the 770 words NN_live captured "
                         "and compare them with acquire_decimated()'s samples "
                         "for the same shot (one extra Pyro call per shot)")
    a = ap.parse_args()
    if a.live_capture:
        a.live = True

    # Use the canonical mapping from the acquisition script rather than
    # deriving the label filename by string surgery: "x_50k_heldout" has no
    # "_X" to swap, so a derive-by-replace silently loads the X file as labels
    # and fails with a shape error 300 shots later.
    DATASETS = {name: (xf_, yf_) for name, xf_, yf_ in A.DATASETS}
    if a.dataset not in DATASETS:
        raise SystemExit(f"unknown dataset {a.dataset!r}; "
                         f"known: {sorted(DATASETS)}")
    xf, yf = DATASETS[a.dataset]

    # Board-scoped k/c for QICK_NS_HOST. strict=False: this script never reads the root calibration/k.npy/c.npy
    # that the ACTIVE guard protects, and the guard would block concurrent runs on both boards.
    k, c = calib.load_kc(strict=False)
    soc.nn_set_onset_threshold(a.onset_threshold)
    print(f"dataset={a.dataset} shots={a.shots} SF={a.scaling_factor} "
          f"onset={a.onset_threshold}")

    X_all = np.load(xf, mmap_mode="r")
    Xfull = np.asarray(X_all[a.start:a.start + a.chunk])
    Dfull = h.prepare_chunk(Xfull, k, c)
    data_peak = None
    if a.norm == "dataset":
        data_peak = raw_peak(X_all, k, c)
        Dfull *= raw_peak(Xfull, k, c) / data_peak
        print(f"norm=dataset: whole-file peak {data_peak:.1f}, chunk played at "
              f"{np.max(np.abs(Dfull)) / (h.ENVELOPE_LIMIT * h.TARGET_FRACTION):.3f}"
              f" of the per-chunk amplitude")
    n = min(a.shots, len(Dfull))
    Y = np.load(yf)[a.start:a.start + n].astype(int)
    D = Dfull[:n]

    R_before = R_start = R_end = None
    if a.align:
        R_before, R_start = align_stream(k, c)
        print(f"stream phase: R = {R_before} before alignment -> {R_start} "
              f"({'ALIGNED' if R_start in (769, 0, 1) else 'ALIGNMENT FAILED'})")
        if R_start not in (769, 0, 1):
            raise SystemExit("could not align the NN_axi window grid; refusing to "
                             "score a rotated window")

    if a.live and not soc.live_present():
        raise SystemExit("--live: no axi_bram_ctrl_live on this bitstream (or "
                         "NNQickSoc is unpatched); deploy v1.40L1+ and run "
                         "board_tools/patch_nameserver_live.py")

    probs = np.zeros(n)
    words = np.zeros((n, 770), dtype=np.uint32) if a.capture else None
    live_hdr = np.zeros((n, 3), dtype=np.int64) if a.live else None    # shots, frame_err, magic
    live_rec = np.zeros((n, 8), dtype=np.int64) if a.live else None    # NN_live.h record
    live_words = np.zeros((n, a.live_n), dtype=np.uint32) if a.live_capture else None
    dec_iq = np.zeros((n, 770, 2), dtype=np.int32) if a.live_capture else None
    realigns = []                      # (shot index, R measured, R after)
    for i in range(n):
        if a.align and a.realign_every and i > 0 and i % a.realign_every == 0:
            Rm = _phase(measure_rotation(k, c, 3))
            Ra = Rm
            if Rm is not None and Rm not in (769, 0, 1):
                _, Ra = align_stream(k, c)
                print(f"  re-align at shot {i}: R = {Rm} -> {Ra}")
            realigns.append((i, -1 if Rm is None else Rm, -1 if Ra is None else Ra))
        soc.nn_configure(10000, CAPTURE_ARM, a.scaling_factor)
        wf = h.prepare_waveform(D[i])
        cfg = dict(CFG0, waveform_I=wf[0::2], waveform_Q=wf[1::2])
        prog = Prog(soccfg, cfg)
        iq = prog.acquire_decimated(soc)
        if a.extra_dma:
            soc.get_decimated(CFG0["ro_ch"], 0,
                              soccfg.us2cycles(CFG0["readout_length"],
                                               ro_ch=CFG0["ro_ch"]))
        probs[i] = A.word_to_float(soc.nn_read_bram(SIG_SLOT * 4))
        if a.capture:
            words[i] = [soc.nn_read_bram(j * 4) & 0xFFFFFFFF for j in range(770)]
        if a.live:
            lv = soc.live_latest()
            live_hdr[i], live_rec[i] = lv[:3], lv[3:]
            if a.live_capture:
                live_words[i] = soc.live_read_block(16, a.live_n)
                d0 = np.asarray(iq[0] if isinstance(iq, list) else iq).reshape(-1, 2)
                dec_iq[i] = np.rint(d0[:770]).astype(np.int32)
        del prog, cfg, wf, iq
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{n}")

    soc.nn_configure(10000, 0, a.scaling_factor)   # disarm

    if a.align:
        R_end = _phase(measure_rotation(k, c, 3))
        print(f"stream phase at end of run: R = {R_end} "
              f"({'held' if R_end in (769, 0, 1) else 'SLIPPED -- some shots were scored rotated'})")
        soc.nn_configure(10000, 0, a.scaling_factor)

    if a.save:
        np.savez(a.save, idx=np.arange(a.start, a.start + n), probs=probs, y=Y,
                 k=k, c=c, norm=a.norm, data_peak=data_peak or np.nan,
                 host=A.NS_HOST, dataset=a.dataset, extra_dma=a.extra_dma,
                 onset_threshold=a.onset_threshold, align=a.align,
                 R_before=-1 if R_before is None else R_before,
                 R_start=-1 if R_start is None else R_start,
                 R_end=-1 if R_end is None else R_end,
                 realigns=np.array(realigns, dtype=int).reshape(-1, 3),
                 **({"words": words} if a.capture else {}),
                 **({"live_hdr": live_hdr, "live_rec": live_rec} if a.live else {}),
                 **({"live_words": live_words, "dec_iq": dec_iq, "live_n": a.live_n}
                    if a.live_capture else {}))
        print(f"saved per-shot results to {a.save}")

    if not a.extra_dma:
        print("\nNOTE: --no-extra-dma was passed, so the chip reports the PREVIOUS "
              "shot and the working pairing\n      will be y[i-1]. That is the "
              "known DMA-readback lag, not a classifier fault.\n      Re-run "
              "without the flag for same-shot numbers.")
    print(f"\nprob range: {probs.min():.6f} -> {probs.max():.6f}")
    print(f"predicted |1>: {int((probs > 0.5).sum())}/{n}")
    if probs.min() == probs.max():
        print("DEGENERATE: every probability identical -- the readback path "
              "is not delivering results; the fidelity number below is "
              "meaningless.")

    print("\npairing        overall     ground    excited   FIDELITY   95% CI")
    for off in (0, -1, 1):
        pi = [s for s in range(n) if 0 <= s + off < n]
        if not pi:
            continue
        pred = (probs[pi] > 0.5).astype(int)
        lab = Y[[s + off for s in pi]]
        ok = pred == lab
        g, e = lab == 0, lab == 1
        ga = 100 * ok[g].mean() if g.sum() else float("nan")
        ea = 100 * ok[e].mean() if e.sum() else float("nan")
        lo, hi = wilson(int(ok.sum()), len(ok))
        tag = {0: "pred[i]/y[i]", -1: "pred[i]/y[i-1]", 1: "pred[i]/y[i+1]"}[off]
        print(f"{tag:15s} {100*ok.mean():6.2f}%   {ga:6.2f}%   {ea:6.2f}%   "
              f"{0.5*(ga+ea):6.2f}%   [{lo:.2f}, {hi:.2f}]")

    if a.live:
        report_live(Y, probs, live_hdr, live_rec, live_words, dec_iq)


if __name__ == "__main__":
    main()
