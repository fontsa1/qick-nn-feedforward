"""
2026-09-01: on-chip test harness with contamination guards.

Written in response to CLASSIFIER_DEBUGGING_REFERENCE.md #17 UPDATE 2, in
which `loopback_nn_fidelity_500_smoketest.ipynb` silently resumed from
leftover chunk files, fired ZERO shots, and reported a stale qick33 result
as v1.23's on-chip fidelity. Every guard below exists to make a specific
observed failure impossible.

PROCESS FIXES IMPLEMENTED (each maps to a real failure):

 G1. NO RESUME, EVER. There is no "skip if file exists" path. If the output
     directory already exists the run ABORTS unless --force is passed, and
     --force wipes it first. The old notebook's resume guard is the direct
     cause of the misattribution.

 G2. FIRMWARE BANNER STAMPED ON EVERY RESULT. The manifest records the
     build timestamp the running firmware reports plus the bitfile path the
     server actually loaded. A number can no longer be attributed to the
     wrong bitstream, because the bitstream identity travels with it.

 G3. FRESH-CORE ASSERTION. After the bitstream reload the harness requires
     nn_status() == out_offset 0, trig_count 0. `nn_reset()` does NOT
     achieve this (measured: 499 before and after -- bug #3), so a reload
     is the only trustworthy clear. Anything else aborts the run.

 G4. WRITE-COUNT INTEGRITY CHECK. out_offset is recorded before and after
     firing. It MUST advance by exactly the number of shots fired. A
     mismatch means the NN dropped or duplicated writes, which would
     silently corrupt the read-to-shot pairing -- the run is flagged.

 G5. UNDERFLOW DETECTION. Counts BRAM words written as exactly 0.0. At an
     underflowed scaling_factor ~40% of shots write 0.0, and the old
     classify_state() skipped zeros and returned an OLDER shot's value,
     making the read lag variable. This harness reads the raw slot at
     out_offset-1 and never skips, so the mapping stays unambiguous; any
     run with zero-writes is still flagged as unreliable.

 G6. SCALING_FACTOR CALIBRATED PER BITSTREAM, ON DISJOINT SHOTS. The
     correct scaling_factor differs between builds (qick33-era data was
     properly scaled at 1; v1.23 underflows at 1 and needs ~8). Calibration
     uses heldout shots [CAL_LO:CAL_HI], the scored test uses [0:N] -- no
     overlap, so the operating point is not tuned on the reported data.

"""
import argparse
import gc
import json
import shutil
import struct
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

sys.path.insert(0, "board_tools")
import board_control as bc

from qick import QickConfig
from qick.asm_v2 import AveragerProgramV2

BASE = Path(".")
RUNS = BASE / "onchip_runs"
ENVELOPE_LIMIT, TARGET_FRACTION = 32766, 0.99
SF_CANDIDATES = [1, 2, 4, 6, 8, 12, 16]
CAL_LO, CAL_HI = 2000, 2120          # disjoint from the scored range [0:N]

def prepare_chunk(x_chunk, k, c):
    z = x_chunk[:, 0::2] + 1j * x_chunk[:, 1::2]
    zs = (z - c) / k
    xo = np.linspace(0, 1, zs.shape[1])
    xn = np.linspace(0, 1, 11000)
    I = interp1d(xo, zs.real, axis=1, kind="linear", fill_value="extrapolate")(xn)
    Q = interp1d(xo, zs.imag, axis=1, kind="linear", fill_value="extrapolate")(xn)
    out = np.empty((x_chunk.shape[0], 22000))
    out[:, 0::2] = I
    out[:, 1::2] = Q
    return out * (ENVELOPE_LIMIT * TARGET_FRACTION / np.max(np.abs(out)))

def prepare_waveform(row):
    a = np.asarray(row, dtype=float).reshape(-1)
    pad = (-a.size) % 32
    return np.pad(a, (0, pad)) if pad else a

def wilson(kk, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = kk / n
    d = 1 + z ** 2 / n
    ctr = (p + z ** 2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return (ctr - h, ctr + h)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bitstream", required=True)
    ap.add_argument("--shots", type=int, default=500)
    ap.add_argument("--force", action="store_true")
    # G7: dataset and calibration are EXPLICIT and recorded. Using qick34's
    # k/c with the qick33 bitstream (which differ by ~193 deg in arg(k) --
    # essentially a sign flip) silently invalidates the predistortion, and
    # that is exactly what the first qick33 comparison run did.
    ap.add_argument("--dataset", default="x_50k_heldout",
                    help="basename: x_50k or x_50k_heldout")
    ap.add_argument("--onset-threshold", type=int, default=None,
                    help="if given, calls nn_set_onset_threshold() before firing. "
                         "2026-09-02: the register reads 0 by default after a fresh "
                         "reload/reset -- nothing in this harness or its predecessors "
                         "ever set it -- which means DRAIN_L's onset test (mag > "
                         "*onset_threshold) fires on essentially the FIRST probed word "
                         "of real ADC noise, not genuine pulse onset. Leave unset to "
                         "reproduce that (default/historical) condition.")
    ap.add_argument("--calfile", default=None,
                    help="basename of a DEDICATED scaling_factor calibration set "
                         "(<name>_X.npy/<name>_y.npy), disjoint from the scored set")
    ap.add_argument("--kfile", default="calibration/k.npy")
    ap.add_argument("--cfile", default="calibration/c.npy")
    ap.add_argument("--tag", default=None,
                    help="output subdir name; defaults to the bitstream name")
    ap.add_argument("--skip-reload", action="store_true",
                    help="board already reloaded+calibrated in this session; "
                         "skips set_bitfile/reload but still asserts G2 and "
                         "records the banner")
    ap.add_argument("--force-sf", type=int, default=None,
                    help="skip the G6 auto-select sweep entirely and fire the "
                         "scored run at this scaling_factor. 2026-09-04: G6's "
                         "own selection logic disqualifies any sf with >=1 "
                         "zero-write out of 200 cal shots even when that sf is "
                         "the clear best performer, and more fundamentally, "
                         "'best cal fidelity' can pick an sf that domain-"
                         "mismatches an SF=1-trained model. Use this to test "
                         "an SF=1-trained model at its own native sf=1.")
    a = ap.parse_args()

    outdir = RUNS / (a.tag or a.bitstream)
    if outdir.exists():
        if not a.force:
            sys.exit(
                "ABORT (G1): %s already exists. This harness never resumes -- "
                "a stale-resume is exactly what caused the 2026-09-01 "
                "misattribution. Pass --force to wipe and rerun." % outdir)
        print("[G1] --force given: wiping %s" % outdir)
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True)

    print("=" * 74)
    print("ON-CHIP TEST -- bitstream %s, %d shots" % (a.bitstream, a.shots))
    print("=" * 74)

    board_bit = "/home/xilinx/jupyter_notebooks/%s.bit" % a.bitstream
    if a.skip_reload:
        print("[setup] --skip-reload: reusing the already-loaded board")
        import Pyro4
        ns = Pyro4.locateNS(host=bc.NS_HOST, port=bc.NS_PORT)
        soc = Pyro4.Proxy(ns.lookup(bc.PROXY))
    else:
        print("[setup] pointing server notebook at %s" % board_bit)
        bc.set_bitfile(board_bit)
        print("[setup] reloading board (kernel restart -> overlay + bitstream)")
        soc = bc.reload_board()

    banner = bc.firmware_banner(soc)
    loaded = bc.current_bitfile()
    print("[G2] firmware built : %s" % banner["built"])
    print("[G2] board          : %s" % banner["board"])
    print("[G2] bitfile loaded : %s" % loaded)
    if a.bitstream not in (loaded or ""):
        sys.exit("ABORT (G2): server notebook says %s, expected %s"
                 % (loaded, a.bitstream))

    if a.onset_threshold is not None:
        soc.nn_set_onset_threshold(a.onset_threshold)
        rb = soc.nn_get_onset_threshold()
        print("[G7] onset_threshold set to %d (readback %d)" % (a.onset_threshold, rb))
        assert rb == a.onset_threshold, "onset_threshold readback mismatch"
    else:
        print("[G7] onset_threshold LEFT AS-IS: currently %d" % soc.nn_get_onset_threshold())

    st0 = soc.nn_status()
    if a.skip_reload:
        # A calibration pass between reload and test legitimately advances
        # out_offset. G4's write-count check still catches dropped writes,
        # so the run stays trustworthy without a zero-offset requirement.
        print("[G3] --skip-reload: fresh-core assertion waived; core state %s" % st0)
    elif st0["out_offset"] != 0 or st0["trig_count"] != 0:
        sys.exit("ABORT (G3): core not fresh after reload: %s" % st0)
    else:
        print("[G3] fresh core confirmed: %s" % st0)

    soccfg = QickConfig(soc.get_cfg())
    soccfg["tprocs"][0]["output_pins"].append(("trig", 11, 0, "NN classify trigger (trig_11_o)"))  # v1.40L5: NN_live gate on trig_11
    PIN = len(soccfg["tprocs"][0]["output_pins"]) - 1
    BW = soc.nn_bram_word_count()
    print("[setup] NN_TRIGGER_PIN=%d  BRAM_WORDS=%d" % (PIN, BW))

    # G7: explicit dataset + calibration, both recorded in the manifest
    k = complex(np.load(BASE / a.kfile))
    c = complex(np.load(BASE / a.cfile))
    if a.dataset == "x_50k":
        xf, yf, pf = "x_50k.npy", "y_50k.npy", "Phi.npy"
    elif a.dataset == "x_50k_heldout":
        xf, yf, pf = "x_50k_heldout.npy", "y_50k_heldout.npy", "Phi_heldout.npy"
    elif a.dataset.endswith("_X"):
        xf, yf, pf = a.dataset + ".npy", a.dataset[:-2] + "_y.npy", None
    else:
        xf, yf, pf = a.dataset + ".npy", a.dataset.replace("x_", "y_") + ".npy", "Phi.npy"
    print("[G7] dataset     : %s / %s / %s" % (xf, yf, pf))
    print("[G7] calibration : %s k=%s" % (a.kfile, k))
    print("[G7]               %s c=%s" % (a.cfile, c))
    X = np.load(BASE / xf)
    Y = np.load(BASE / yf).astype(int)
    PH = np.load(BASE / pf) if pf else None   # unused: phase contract is 0

    cfg0 = {"gen_ch": 0, "ro_ch": 0, "nqz": 1, "reps": 1, "final_delay": 100,
            "readout_length": 1.393, "adc_trig_offset": 0.31, "length": 16,
            "freq": 50, "mode": "oneshot", "stdysel": "zero", "outsel": "product"}

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
                        outsel=cf["outsel"], mode=cf["mode"],
                        stdysel=cf["stdysel"], ro_ch=cf["ro_ch"])
            s.add_readoutconfig(ch=cf["ro_ch"], name="r", freq=cf["freq"],
                                gen_ch=cf["gen_ch"])

        def _body(s, cf):
            s.pulse(ch=cf["gen_ch"], name="p")
            s.send_readoutconfig(ch=cf["ro_ch"], name="r", t=0)
            s.trigger(ros=[cf["ro_ch"]], pins=[0, PIN],
                      t=cf["adc_trig_offset"], width=10)

    def read_latest():
        off = soc.nn_status()["out_offset"]
        raw = soc.nn_read_bram(((off - 1) % BW) * 4)
        return off, struct.unpack("f", struct.pack("I", raw & 0xFFFFFFFF))[0]

    def fire_block(D, phis, tag):
        """Fire len(D) shots, reading the word at out_offset-1 each time.
        Deliberately does NOT skip zeros -- reading the raw slot keeps the
        shot<->read mapping unambiguous (G5)."""
        recs = []
        t0 = time.time()
        for i in range(len(D)):
            w = prepare_waveform(D[i])
            cf = dict(cfg0, waveform_I=w[0::2], waveform_Q=w[1::2],
                      gain=1, phase=phis[i])
            pr = Prog(soccfg, cf)
            pr.acquire_decimated(soc, progress=False)
            off, val = read_latest()
            recs.append((i, off, val))
            del pr, cf, w
            if i % 50 == 0:
                gc.collect()
                if tag == "test":
                    print("    %d/%d (%.0fs)" % (i + 1, len(D), time.time() - t0))
        return recs

    # ---------------- G6: scaling_factor calibration on DISJOINT shots -----
    if a.force_sf is not None:
        SF = a.force_sf
        caltab = None
        print("\n[G6] SKIPPED (--force-sf given): firing scored run directly "
              "at scaling_factor=%d, no auto-select sweep" % SF)
    else:
        print("\n[G6] calibrating scaling_factor on heldout[%d:%d] "
              "(disjoint from the scored range [0:%d])" % (CAL_LO, CAL_HI, a.shots))
        if a.calfile:
            Xc = np.load(BASE / (a.calfile + "_X.npy"))
            yc = np.load(BASE / (a.calfile + "_y.npy")).astype(int)
            print("[G6] calibrating on DEDICATED file %s (%d shots), disjoint from "
                  "the scored set by construction" % (a.calfile, len(yc)))
        else:
            Xc, yc = X[CAL_LO:CAL_HI], Y[CAL_LO:CAL_HI]
            if CAL_LO < a.shots:
                sys.exit("ABORT (G6): calibration range [%d:%d] overlaps the scored "
                         "range [0:%d]. Pass --calfile with a disjoint set."
                         % (CAL_LO, CAL_HI, a.shots))
        Dcal = prepare_chunk(Xc, k, c)
        ycal = yc
        phcal = [0.0] * len(Dcal)          # phase baked into the waveform
        caltab = []
        print("    %4s %9s %9s %6s %6s %9s"
              % ("sf", "maxprob", "meanprob", ">0.5", "zeros", "fidelity"))
        for sf in SF_CANDIDATES:
            soc.nn_configure(10000, 0, sf)
            recs = fire_block(Dcal, phcal, "cal")
            v = np.array([r[2] for r in recs])
            pred = (v > 0.5).astype(int)
            ps, ts = pred[1:], ycal[:-1]
            gi, ei = np.where(ts == 0)[0], np.where(ts == 1)[0]
            ga = np.mean(ps[gi] == ts[gi]) if len(gi) else np.nan
            ea = np.mean(ps[ei] == ts[ei]) if len(ei) else np.nan
            f = 0.5 * (ga + ea)
            nz = int((v == 0.0).sum())
            caltab.append(dict(sf=sf, maxprob=float(v.max()), meanprob=float(v.mean()),
                               n_above=int((v > 0.5).sum()), zeros=nz,
                               fidelity=float(f)))
            print("    %4d %9.5f %9.5f %6d %6d %8.2f%%"
                  % (sf, v.max(), v.mean(), int((v > 0.5).sum()), nz, f * 100))
        ok = [r for r in caltab if r["zeros"] == 0]
        best = max(ok or caltab, key=lambda r: r["fidelity"])
        SF = best["sf"]
        print("[G6] chosen scaling_factor = %d (cal fidelity %.2f%%, zero-writes %d)"
              % (SF, best["fidelity"] * 100, best["zeros"]))

    # ---------------- scored run ------------------------------------------
    print("\n[run] firing %d scored shots at scaling_factor=%d" % (a.shots, SF))
    soc.nn_configure(10000, 0, SF)
    off_before = soc.nn_status()["out_offset"]
    D = prepare_chunk(X[:a.shots], k, c)
    recs = fire_block(D, [0.0] * a.shots, "test")   # phase contract: 0
    off_after = soc.nn_status()["out_offset"]

    advanced = off_after - off_before
    integrity_ok = (advanced == a.shots)
    print("[G4] out_offset %d -> %d (advanced %d, expected %d) %s"
          % (off_before, off_after, advanced, a.shots,
             "OK" if integrity_ok else "*** MISMATCH ***"))

    v = np.array([r[2] for r in recs])
    y = Y[:a.shots]
    nz = int((v == 0.0).sum())
    print("[G5] zero-writes: %d/%d%s" % (nz, a.shots,
          "" if nz == 0 else "   *** underflow: read pairing UNRELIABLE ***"))

    pred = (v > 0.5).astype(int)
    unc = float(np.mean(pred == y))
    ps, ts = pred[1:], y[:-1]
    overall = float(np.mean(ps == ts))
    gi, ei = np.where(ts == 0)[0], np.where(ts == 1)[0]
    ga = float(np.mean(ps[gi] == ts[gi]))
    ea = float(np.mean(ps[ei] == ts[ei]))
    fid = 0.5 * (ga + ea)
    lo, hi = wilson(int(np.sum(ps == ts)), len(ts))

    print("\n" + "=" * 74)
    print("RESULTS -- %s (firmware built %s)" % (a.bitstream, banner["built"]))
    print("=" * 74)
    print("  prob range               : %.6f -> %.6f" % (v.min(), v.max()))
    print("  predicted |1>            : %d/%d" % (int(pred.sum()), a.shots))
    print("  uncorrected (pred[i] vs true[i])  : %6.2f%%" % (unc * 100))
    print("  CORRECTED   (pred[i] vs true[i-1]): %6.2f%% (%d/%d)  95%% CI [%.2f, %.2f]"
          % (overall * 100, int(np.sum(ps == ts)), len(ts), lo * 100, hi * 100))
    print("    ground  |0>            : %6.2f%%" % (ga * 100))
    print("    excited |1>            : %6.2f%%" % (ea * 100))
    print("    FIDELITY 0.5*(g+e)     : %6.2f%%" % (fid * 100))
    print("=" * 74)

    np.save(outdir / "raw_records.npy", np.array(recs, dtype=object))
    np.save(outdir / "probs.npy", v)
    np.save(outdir / "labels.npy", y)
    manifest = dict(
        bitstream=a.bitstream, bitfile_loaded=loaded,
        firmware_built=banner["built"], board=banner["board"],
        timestamp=datetime.now().isoformat(timespec="seconds"),
        n_shots=a.shots, scaling_factor=SF, window_size=10000, window_offset=0,
        calibration=caltab, calibration_range=[CAL_LO, CAL_HI],
        out_offset_before=int(off_before), out_offset_after=int(off_after),
        out_offset_advanced=int(advanced), write_integrity_ok=bool(integrity_ok),
        zero_writes=nz, pairing_reliable=bool(nz == 0),
        prob_min=float(v.min()), prob_max=float(v.max()),
        n_pred_1=int(pred.sum()),
        fidelity_uncorrected=unc, fidelity_overall_corrected=overall,
        ground_acc=ga, excited_acc=ea, fidelity=fid, ci95=[lo, hi],
        dataset="%s[:%d]" % (xf, a.shots),
        dataset_files=[xf, yf, pf],
        kfile=a.kfile, cfile=a.cfile,
        k=[k.real, k.imag], c=[c.real, c.imag],
        skip_reload=bool(a.skip_reload),
        guards=("G1 no-resume, G2 banner-stamped, G3 fresh-core, "
                "G4 write-count, G5 underflow, G6 disjoint-calibration, "
                "G7 explicit-dataset-and-calibration"),
    )
    with open(outdir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print("wrote %s" % (outdir / "manifest.json"))

if __name__ == "__main__":
    main()
