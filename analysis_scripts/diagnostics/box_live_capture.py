"""NN_live on the boxcar readout path: live capture for the bit-exact check.

Per shot (test20k_real_X rows, the run_loopback_box.py recipe and dataset-wide DAC scale):
readout 0 (FIR) triggered at adc_trig_offset; readout 1 (avg buffer 1 = the boxcar stream)
and NN_live's classify trigger (trig_11) together at adc_trig_offset - box_advance. Reads
NN_live's record (NN_live.h) and its captured window (live BRAM words 16..16+N), and the
boxcar buffer. Writes an npz in campaign_v140L1/check_live_bitexact.py's format (live_words, live_rec, y)
plus box/fir and the window == buffer comparison.

"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "board_tools"))
sys.path.insert(0, str(ROOT / "analysis_scripts" / "acquisition"))
import calib                                                     # noqa: E402
import deploy_and_verify as D                                    # noqa: E402
from run_loopback_box import prepare_waveform                    # noqa: E402
from qick.asm_v2 import AveragerProgramV2                        # noqa: E402

N_LIVE = 576
LIVE_MAGIC = 0x4E4E4C31

soc, soccfg = D.connect()
soccfg['tprocs'][0]['output_pins'].append(('trig', 11, 0, 'NN classify trigger (trig_11_o)'))
CLS_PIN = len(soccfg['tprocs'][0]['output_pins']) - 1

class P(AveragerProgramV2):
    def _initialize(self, cfg):
        self.declare_gen(ch=0, nqz=1)
        for ch in (0, 1):
            self.declare_readout(ch=ch, length=cfg["readout_length"])
            self.add_readoutconfig(ch=ch, name=f"ro{ch}", freq=cfg["freq"], gen_ch=0)
        self.add_envelope(ch=0, name="env", idata=cfg["waveform_I"], qdata=cfg["waveform_Q"])
        self.add_pulse(ch=0, name="p", style="arb", envelope="env", gain=1, freq=cfg["freq"],
                       phase=0.0, outsel="product", mode="oneshot", stdysel="zero", ro_ch=0)

    def _body(self, cfg):
        self.pulse(ch=0, name="p")
        self.send_readoutconfig(ch=0, name="ro0", t=0)
        self.send_readoutconfig(ch=1, name="ro1", t=0)
        self.trigger(ros=[0], t=cfg["adc_trig_offset"], width=10)
        self.trigger(ros=[1], pins=[CLS_PIN], t=cfg["adc_trig_offset"] - cfg["box_advance"], width=10)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=200)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--box-advance-us", type=float, default=0.1135)
    ap.add_argument("--shift", type=int, default=7)
    ap.add_argument("--m", type=int, default=11)
    ap.add_argument("--out", default="campaign_v140L1/A_L8_box_live_capture.npz")
    ap.add_argument("--x", default="data/test20k_real_X.npy", help="input traces; --start/--shots index its rows")
    ap.add_argument("--y", default="data/test20k_real_y.npy")
    ap.add_argument("--scale-ref", default="data/test20k_real_X.npy",
                    help="traces that set the dataset-wide DAC scale (kept at the 20k set so every campaign "
                         "replays a given trace identically)")
    ap.add_argument("--clip-dac", action="store_true",
                    help="cap envelope samples at the DAC full scale instead of failing (traces whose peak exceeds the "
                         "scale-ref maximum); the per-shot count of capped samples is saved as dac_clipped")
    a = ap.parse_args()
    print("boxcar path:", soc.box_config(sel=1, shift=a.shift, m=a.m))
    k, c = calib.load_kc(strict=False)
    Xall = np.load(ROOT / a.x, mmap_mode="r")
    yall = np.load(ROOT / a.y).astype(int)
    Xref = np.load(ROOT / a.scale_ref, mmap_mode="r")
    z = np.asarray(Xref[:, 0::2]) + 1j * np.asarray(Xref[:, 1::2])
    zs = (z - c) / k
    sf = 32766 * 0.99 / (max(np.abs(zs.real).max(), np.abs(zs.imag).max()) * 1.0001)
    del z, zs
    xo = np.linspace(0, 1, 770); xn = np.linspace(0, 1, 11000)
    cfg0 = dict(readout_length=1.393, adc_trig_offset=0.31, box_advance=a.box_advance_us, freq=50.0)

    rows = np.arange(a.start, a.start + a.shots)
    n = len(rows)
    live_words = np.zeros((n, N_LIVE), np.uint32); live_rec = np.zeros((n, 8), np.int64)
    hdr = np.zeros((n, 3), np.int64); box = np.zeros((n, 1540), np.int16); fir = np.zeros((n, 1540), np.int16)
    maxv = soccfg["gens"][0]["maxv"]; clipped = np.zeros(n, np.int64)
    for i, r in enumerate(rows):
        zr = (np.asarray(Xall[r, 0::2]) + 1j * np.asarray(Xall[r, 1::2]) - c) / k
        w = np.empty(22000); w[0::2] = np.interp(xn, xo, zr.real) * sf; w[1::2] = np.interp(xn, xo, zr.imag) * sf
        w = prepare_waveform(w)
        if a.clip_dac:
            clipped[i] = int(np.sum(np.abs(w) > maxv))
            w = np.clip(w, -maxv, maxv)
        iq = P(soccfg, reps=1, final_delay=100, cfg=dict(cfg0, waveform_I=w[0::2], waveform_Q=w[1::2])
               ).acquire_decimated(soc, rounds=1, progress=False)
        fir[i] = np.asarray(iq[0]).reshape(-1)[:1540]; box[i] = np.asarray(iq[1]).reshape(-1)[:1540]
        lv = soc.live_latest()
        hdr[i], live_rec[i] = lv[:3], lv[3:]
        live_words[i] = soc.live_read_block(16, N_LIVE)
    li = (live_words & 0xFFFF).astype(np.uint16).view(np.int16)
    lq = (live_words >> 16).astype(np.uint16).view(np.int16)
    same = np.array([np.array_equal(li[i], box[i, 0:2 * N_LIVE:2]) and np.array_equal(lq[i], box[i, 1:2 * N_LIVE:2])
                     for i in range(n)])
    y = yall[rows]
    dec = (live_rec[:, 7] & 1).astype(int)
    g, e = y == 0, y == 1
    print(f"magic {'ok' if (hdr[:, 2] == LIVE_MAGIC).all() else 'WRONG'}; frame errors {hdr[0, 1]} -> {hdr[-1, 1]}")
    print(f"live window == boxcar buffer (ch1), sample for sample: {int(same.sum())}/{n}")
    print(f"NN_live fidelity (boxcar path, {n} shots): {100 * 0.5 * (np.mean(dec[g] == 0) + np.mean(dec[e] == 1)):.2f}%")
    if a.clip_dac:
        print(f"DAC clipping: {int((clipped > 0).sum())} shots, {int(clipped.sum())} samples capped at {maxv}")
    np.savez(ROOT / a.out, live_words=live_words, live_rec=live_rec, live_hdr=hdr, y=y, rows=rows,
             box=box, fir=fir, window_eq_buffer=same, live_n=N_LIVE, dac_clipped=clipped)
    print("saved", a.out)

if __name__ == "__main__":
    main()
