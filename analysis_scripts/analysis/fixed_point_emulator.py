"""Bit-accurate fixed-point emulation of the deployed classifier, at scale.

Runs the hls4ml C++ model (hls_model.compile(), which compiles the same NN.cpp / nnet_utils / ap_fixed types
that synthesis uses) over all 20,000 shots of loopback20k_real, to measure the fixed-point cost against the float
Keras model on the same shots.

The model is rebuilt from the deployed recipe (the float32 path reproduces seed 20260902 = 95.34% bit for bit) and
converted with the same make_hls_config() the export uses. `--check-firmware` asserts the emulator's
weights are byte-identical to neural_net4/firmware/weights, i.e. that this is
the model the bitstream carries.

"""
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
import filecmp
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis_scripts" / "analysis"))
sys.path.insert(0, str(ROOT / "analysis_scripts" / "acquisition"))

def train(n_train, seed):
    import tensorflow as tf
    import task33_three_way_comparison as T
    k2 = complex(np.load(ROOT / "calibration/k2_loopback.npy"))
    c2 = complex(np.load(ROOT / "calibration/c2_loopback.npy"))
    X = np.load(ROOT / "0528_X_train_0_770.npy", mmap_mode="r")[:n_train]
    X = T.hw_raw_transform(T.apply_affine(np.asarray(X).astype(np.float32), k2, c2))
    y = np.load(ROOT / "0528_y_train_0_770.npy")[:n_train].astype(np.float32)
    perm = np.random.default_rng(seed).permutation(len(y))
    m = T.build_model(X.shape[1], seed)
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                          restore_best_weights=True)
    h = m.fit(X[perm], y[perm], validation_split=0.2, epochs=50, batch_size=32,
              verbose=0, callbacks=[es])
    return m, float(min(h.history["val_loss"]))

MSYS_BIN = Path(r"C:\msys64\ucrt64\bin")

def _patch_windows_compile():
    """hls4ml runs `./build_lib.sh` through cmd.exe, which cannot execute it.
    Run it through bash instead, with BASEDIR as a Windows path (`pwd -W`):
    NN.cpp loads weights/*.txt at RUNTIME from WEIGHTS_DIR under
    `#ifndef __SYNTHESIS__`, and the native MSYS2 g++ build cannot open a
    /c/... path. The .so links libstdc++ from MSYS2, so register that DLL dir."""
    import subprocess
    from hls4ml.backends.fpga.fpga_backend import FPGABackend
    if getattr(FPGABackend, "_win_patched", False):
        return
    os.add_dll_directory(str(MSYS_BIN))

    def compile(self, model):
        odir = Path(model.config.get_output_dir())
        sh = odir / "build_lib.sh"
        s = sh.read_text()
        s = s.replace('&& pwd)"', '&& pwd -W)"')
        sh.write_text(s, newline="\n")
        env = dict(os.environ, PATH=str(MSYS_BIN) + os.pathsep + os.environ["PATH"])
        r = subprocess.run(["bash", "build_lib.sh"], cwd=odir, env=env, text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            print(r.stdout)
            raise RuntimeError("build_lib.sh failed")
        return "{}/firmware/{}-{}.so".format(
            model.config.get_output_dir(), model.config.get_project_name(),
            model.config.get_config_value("Stamp"))

    FPGABackend.compile = compile
    FPGABackend._win_patched = True

def fixed_point_model(model, outdir):
    import hls4ml
    from export_signed_domain_model import make_hls_config, FPGA_PART, CLOCK_PERIOD_NS
    _patch_windows_compile()
    hm = hls4ml.converters.convert_from_keras_model(
        model=model, hls_config=make_hls_config(model), output_dir=str(outdir),
        part=FPGA_PART, io_type="io_parallel", clock_period=CLOCK_PERIOD_NS,
        backend="Vivado", project_name="NN")
    hm.compile()
    return hm

def stats(p, y):
    pred = (p > 0.5).astype(int)
    g, e = y == 0, y == 1
    p10 = float(np.mean(pred[g] != y[g])); p01 = float(np.mean(pred[e] != y[e]))
    return dict(fid=100 * (1 - 0.5 * (p01 + p10)), acc=100 * float(np.mean(pred == y)),
                p10=100 * p10, p01=100 * p01)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260902)
    ap.add_argument("--n-eval", type=int, default=20000)
    ap.add_argument("--outdir", default=str(Path(os.environ.get("TEMP", ".")) /
                                            "hls4ml_fixed_point_emulator"))
    ap.add_argument("--model", default=None,
                    help="load this exported Keras .h5 instead of retraining "
                         "(--n-train/--seed then only label the output dir)")
    ap.add_argument("--check-firmware", action="store_true",
                    help="assert weights match neural_net4/firmware/weights")
    ap.add_argument("--save", default="data/fixed_point_emulator_loopback20k.npz")
    ap.add_argument("--window", type=int, default=770,
                    help="score on samples [0, window) (a --window model from "
                         "export_signed_domain_model.py)")
    ap.add_argument("--eval", default="data/loopback20k_real.npy",
                    help="acquired shots to score: an (N, 1540) .npy (labels in "
                         "<name>_y.npy), or a run_loopback_box.py .npz (task 4.1h; "
                         "array --eval-key, labels 'y')")
    ap.add_argument("--eval-key", default="box", help="fir or box, for an .npz --eval")
    a = ap.parse_args()

    import task33_three_way_comparison as T
    t0 = time.time()
    if a.model:
        from qkeras.utils import load_qmodel
        model, vl = load_qmodel(str(ROOT / a.model), compile=False), float("nan")
    else:
        model, vl = train(a.n_train, a.seed)
    if a.eval.endswith(".npz"):
        ev = np.load(ROOT / a.eval)
        Xraw, yev = ev[a.eval_key][:a.n_eval], ev["y"][:a.n_eval].astype(int)
    else:
        Xraw = np.load(ROOT / a.eval, mmap_mode="r")[:a.n_eval]
        yev = np.load(ROOT / a.eval.replace(".npy", "_y.npy"))[:a.n_eval].astype(int)
    print(f"eval set: {a.eval}{'[' + a.eval_key + ']' if a.eval.endswith('.npz') else ''} ({len(yev)} shots)")
    Xev = T.hw_raw_transform(np.asarray(Xraw).astype(np.float64))[:, :2 * a.window]
    p_float = model.predict(Xev, verbose=0).reshape(-1)
    print(f"trained n={a.n_train} seed={a.seed} val_loss={vl:.5f} "
          f"({time.time()-t0:.0f}s)")

    outdir = Path(a.outdir) / (Path(a.model).stem if a.model else f"n{a.n_train}_s{a.seed}")
    hm = fixed_point_model(model, outdir)
    if a.check_firmware:
        fw = ROOT / "neural_net4" / "firmware" / "weights"
        bad = [f.name for f in sorted((outdir / "firmware" / "weights").iterdir())
               if not filecmp.cmp(f, fw / f.name, shallow=False)]
        assert not bad, f"emulator weights differ from deployed firmware: {bad}"
        print("weights byte-identical to neural_net4/firmware/weights")

    t1 = time.time()
    p_fixed = np.asarray(hm.predict(np.ascontiguousarray(Xev, dtype=np.float32))).reshape(-1)
    print(f"fixed-point predict on {len(yev)} shots: {time.time()-t1:.0f}s")

    sf, sx = stats(p_float, yev), stats(p_fixed, yev)
    agree = 100 * float(np.mean((p_float > 0.5) == (p_fixed > 0.5)))
    print(f"float  Keras : FIDELITY {sf['fid']:.2f}%  P(1|0) {sf['p10']:.2f}%  P(0|1) {sf['p01']:.2f}%")
    print(f"fixed  hls4ml: FIDELITY {sx['fid']:.2f}%  P(1|0) {sx['p10']:.2f}%  P(0|1) {sx['p01']:.2f}%")
    print(f"quantization effect: {sx['fid']-sf['fid']:+.2f} points; decision agreement {agree:.2f}%")
    print(f"fixed prob range {p_fixed.min():.6f} -> {p_fixed.max():.6f}; "
          f"exactly 0.0: {int((p_fixed == 0).sum())}, exactly max: {int((p_fixed == p_fixed.max()).sum())}")
    np.savez(ROOT / a.save, p_float=p_float, p_fixed=p_fixed, y=yev,
             n_train=a.n_train, seed=a.seed)
    print(f"saved {a.save}")

if __name__ == "__main__":
    main()
