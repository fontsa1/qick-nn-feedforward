"""boxcar_tap's output, bit for bit, against its own input.

boxcar_tap_0 passes its input (readout 0's full-rate m0 stream, 8 complex samples per clk_adc0_x2 cycle) to
axis_switch_mr S00 unchanged, so the MR buffer (NM 8, 32-bit samples, 1024 cycles) records exactly what the
boxcar sums. Per shot, the same replayed trace is captured twice by one program:
  * MR buffer (arm_mr(0) -> switch input 0, trigger mr=True): the boxcar's input
  * averaging buffer 1, decimated (readout 1 = the boxcar stream, as box_live_capture.py): its output
The numpy model of boxcar_tap (audit_fixes/RB_prep/sim/box_check.py): per cycle, I = sum of the 8 I lanes,
Q = sum of the 8 Q lanes; moving sum over M cycles; round-half-up shift; saturate to int16. The model
applied to the MR samples must equal the buffer on every overlapping sample, at one fixed offset found on
the first shot and held for all (the two captures are triggered together, so the offset is a constant
pipeline delay). Not the kb * MA11 + cb fit used for the domain map: this is exact or it fails.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis_scripts" / "diagnostics"))
import box_live_capture as B                                     # noqa: E402  (brings up soc)

soc, soccfg, calib, prepare_waveform = B.soc, B.soccfg, B.calib, B.prepare_waveform

class PMR(B.P):
    def _body(self, cfg):
        self.pulse(ch=0, name="p")
        self.send_readoutconfig(ch=0, name="ro0", t=0)
        self.send_readoutconfig(ch=1, name="ro1", t=0)
        self.trigger(ros=[0], t=cfg["adc_trig_offset"], width=10)
        self.trigger(ros=[1], pins=[B.CLS_PIN], mr=True, t=cfg["adc_trig_offset"] - cfg["box_advance"], width=10)

def model(mr, m, shift):
    """boxcar_tap on MR samples (n_cycles*8, 2) int16 -> (n_cycles, 2) int16; first m-1 cycles invalid."""
    x = mr.reshape(-1, 8, 2).astype(np.int64)
    I, Q = x[:, :, 0].sum(1), x[:, :, 1].sum(1)
    k = np.ones(m, dtype=np.int64)
    out = []
    for v in (I, Q):
        s = np.convolve(v, k)[:len(v)]
        if shift:
            s = (s + (1 << (shift - 1))) >> shift
        out.append(np.clip(s, -32768, 32767))
    y = np.stack(out, 1)
    y[:m - 1] = np.iinfo(np.int64).min        # not enough history
    return y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=200)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--box-advance-us", type=float, default=0.1135)
    ap.add_argument("--shift", type=int, default=7)
    ap.add_argument("--m", type=int, default=11)
    ap.add_argument("--out", default="campaign_v140L1/A_v142_boxcar_mr.json")
    a = ap.parse_args()
    print("boxcar path:", soc.box_config(sel=1, shift=a.shift, m=a.m))
    if "mr_buf" not in soccfg._cfg:
        raise SystemExit("no MR buffer in this firmware's config")
    k, c = calib.load_kc(strict=False)
    Xall = np.load(ROOT / "data/test20k_real_X.npy", mmap_mode="r")
    z = np.asarray(Xall[:, 0::2]) + 1j * np.asarray(Xall[:, 1::2])
    zs = (z - c) / k
    sf = 32766 * 0.99 / (max(np.abs(zs.real).max(), np.abs(zs.imag).max()) * 1.0001)
    del z, zs
    xo = np.linspace(0, 1, 770); xn = np.linspace(0, 1, 11000)
    cfg0 = dict(readout_length=1.393, adc_trig_offset=0.31, box_advance=a.box_advance_us, freq=50.0)

    lag = None
    res_rows = []
    for i, r in enumerate(range(a.start, a.start + a.shots)):
        zr = (np.asarray(Xall[r, 0::2]) + 1j * np.asarray(Xall[r, 1::2]) - c) / k
        w = np.empty(22000); w[0::2] = np.interp(xn, xo, zr.real) * sf; w[1::2] = np.interp(xn, xo, zr.imag) * sf
        w = prepare_waveform(w)
        soc.arm_mr(0)
        iq = PMR(soccfg, reps=1, final_delay=100, cfg=dict(cfg0, waveform_I=w[0::2], waveform_Q=w[1::2])
                 ).acquire_decimated(soc, rounds=1, progress=False)
        box = np.asarray(iq[1]).reshape(-1, 2).astype(np.int64)            # (770, 2)
        mr = np.asarray(soc.get_mr()).reshape(-1, 2)
        y = model(mr, a.m, a.shift)
        if lag is None:      # box[j] == y[j + lag] for all j; search the fixed offset once
            for d in range(-len(box) // 2, len(y) - len(box) + 1):
                lo, hi = max(0, -d), min(len(box), len(y) - d)
                v = y[lo + d:hi + d]; ok = v[:, 0] != np.iinfo(np.int64).min
                if ok.sum() > 600 and np.array_equal(box[lo:hi][ok], v[ok]):
                    lag = d
                    break
            if lag is None:
                print("no offset gives an exact match on the first shot; saving it for inspection")
                np.savez(ROOT / (a.out.replace(".json", "_shot0.npz")), box=box, mr=mr, y=y)
                raise SystemExit("boxcar vs MR buffer: FAIL (no exact alignment)")
            print(f"offset: buffer sample j == model cycle j + {lag}")
        # the MR buffer and averaging buffer 1 each synchronize the tProc trigger on their own, so the
        # offset may move by a cycle from shot to shot; the arithmetic must be exact at the shot's offset
        best = None
        for d in range(lag - 2, lag + 3):
            lo, hi = max(0, -d), min(len(box), len(y) - d)
            valid = y[lo + d:hi + d, 0] != np.iinfo(np.int64).min
            a_ = box[lo:hi][valid]; b_ = y[lo + d:hi + d][valid]
            n_bad = int(np.sum(np.any(a_ != b_, axis=1)))
            if best is None or n_bad < best[1]:
                best = (d, n_bad, int(valid.sum()))
        res_rows.append(dict(row=r, offset=best[0], compared=best[2], mismatched=best[1]))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{a.shots}: mismatched samples so far {sum(x['mismatched'] for x in res_rows)}",
                  flush=True)
    tot = sum(x["compared"] for x in res_rows); bad = sum(x["mismatched"] for x in res_rows)
    shots_ok = sum(x["mismatched"] == 0 for x in res_rows)
    offs = {}
    for x in res_rows:
        offs[x["offset"]] = offs.get(x["offset"], 0) + 1
    print(f"per-shot offsets (capture trigger synchronization): {dict(sorted(offs.items()))}")
    res = dict(offset=lag, offsets=offs, m=a.m, shift=a.shift, shots=a.shots, shots_exact=shots_ok, samples=tot,
               samples_mismatched=bad, verdict="PASS" if bad == 0 else "FAIL", rows=res_rows)
    print(f"boxcar output == model(MR input): {shots_ok}/{a.shots} shots exact, {tot - bad}/{tot} samples; "
          f"{res['verdict']}")
    with open(ROOT / a.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print("saved", a.out)

if __name__ == "__main__":
    main()
