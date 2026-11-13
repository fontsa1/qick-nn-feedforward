"""
Train and export the deployed classifier (ternary 2N -> 4 -> 1, QKeras, hls4ml precision config) on the
loopback-domain-calibrated training set, in the signed ADC domain.

hw_raw_transform reads each int16 sample as signed, multiplied by scaling_factor, as NN_axi's LOAD_L and NN_live do;
the firmware encoding and this transform must agree. Measured offline on 0528_X_train_0_770_loopback_calibrated.npy
[:20000] (3 seeds each): unsigned reading 92.35%, signed 94.42%, linear ceiling on the same data 96.07%.

The training set uses the label-free global domain map. Training runs several seeds and keeps the lowest val_loss,
computed on the held-out 20% of the training pool (no test or acquired shot is used). Seed-to-seed spread on the
acquired eval set is about +/-0.24 points.
"""
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_USE_LEGACY_KERAS"] = "1"

SEED = 20260901        # historical default; --seeds overrides, and the
                       # selected seed is recorded in results/last_export_seed.txt

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Input, Activation
from qkeras.qlayers import QDense
from qkeras.quantizers import ternary
from pathlib import Path
from datetime import datetime

tf.keras.utils.set_random_seed(SEED)

base = Path(".")
SCALING_FACTOR = 1

def hw_raw_transform(X, scaling_factor):
    """Signed int16 sample * scaling_factor, as NN_axi's LOAD_L.

    ap_fixed<32,20> holds +/-524288, and |32768 * SF<=8| = 262144, so the signed range fits.
    """
    X_int16 = np.clip(np.round(X), -32768, 32767).astype(np.int16)
    return X_int16.astype(np.float32) * scaling_factor

def build_model(input_dim):
    L2 = tf.keras.regularizers.l2(1e-3)
    model = tf.keras.Sequential([
        Input(shape=(input_dim,), name="input_1"),
        QDense(4, kernel_quantizer=ternary(), bias_quantizer=ternary(), name="q_dense_1"),
        BatchNormalization(name="batchnorm1", gamma_regularizer=L2, beta_regularizer=L2),
        Activation("relu", name="relu1"),
        QDense(1, kernel_quantizer=ternary(), bias_quantizer=ternary(),
               kernel_regularizer=L2, bias_regularizer=L2, name="q_output"),
        Activation("sigmoid", name="sigmoid"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model

FPGA_PART = "xczu48dr-ffvg1517-2-e"
CLOCK_PERIOD_NS = 10.0

def make_hls_config(model):
    """The deployed hls4ml precision config. Shared with the bit-accurate
    emulator (analysis_scripts/analysis/fixed_point_emulator.py) so the two
    cannot drift apart."""
    import hls4ml
    hls_config = hls4ml.utils.config_from_keras_model(model, granularity="name")
    hls_config["Model"]["Precision"] = "ap_fixed<16,6>"
    hls_config["Model"]["ReuseFactor"] = 1
    hls_config["Model"]["Strategy"] = "Resource"

    def ensure_precision_dict(layer_cfg):
        if "Precision" not in layer_cfg or not isinstance(layer_cfg["Precision"], dict):
            layer_cfg["Precision"] = {}
        return layer_cfg["Precision"]

    def set_result_precision(layer_name, precision):
        if layer_name in hls_config["LayerName"]:
            p = ensure_precision_dict(hls_config["LayerName"][layer_name])
            p["result"] = precision
            hls_config["LayerName"][layer_name]["Trace"] = True

    def set_accum_precision(layer_name, precision):
        if layer_name in hls_config["LayerName"]:
            hls_config["LayerName"][layer_name]["accum_t"] = precision

    def set_precision_field(layer_name, field, precision):
        if layer_name in hls_config["LayerName"]:
            p = ensure_precision_dict(hls_config["LayerName"][layer_name])
            p[field] = precision

    INPUT_PRECISION = "ap_fixed<32,20>"
    for name in ["input_1", "input_1_input", "q_dense_1_input"]:
        if name in hls_config["LayerName"]:
            hls_config["LayerName"][name]["Precision"] = INPUT_PRECISION

    set_result_precision("q_dense_1", "ap_fixed<48,32>")
    set_accum_precision("q_dense_1", "ap_fixed<48,32>")
    set_result_precision("q_dense_2", "ap_fixed<24,10>")
    set_accum_precision("q_dense_2", "ap_fixed<24,10>")
    set_result_precision("q_output", "ap_fixed<32,16>")
    set_accum_precision("q_output", "ap_fixed<32,16>")

    for name in ["batchnorm1", "batchnorm2", "relu1", "relu2"]:
        set_result_precision(name, "ap_fixed<32,16>")
    set_result_precision("sigmoid", "ap_fixed<18,8>")

    for name in list(hls_config["LayerName"].keys()):
        if "alpha" in name or "linear" in name:
            set_result_precision(name, "ap_fixed<24,10>")

    set_result_precision("q_dense_1_linear", "ap_fixed<48,32>")
    set_result_precision("q_output_linear", "ap_fixed<32,16>")

    set_precision_field("batchnorm1", "scale", "ap_fixed<40,8>")
    set_precision_field("batchnorm1", "bias", "ap_fixed<40,8>")
    set_precision_field("batchnorm2", "scale", "ap_fixed<40,8>")
    set_precision_field("batchnorm2", "bias", "ap_fixed<40,8>")
    return hls_config

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[20260901, 20260902, 20260903, 20260904, 20260905],
                    help="train one model per seed and keep the one with the "
                         "lowest val_loss. Pass a single seed to reproduce the "
                         "historical fixed-seed behaviour.")
    ap.add_argument("--n-train", type=int, default=10000,
                    help="training rows from the front of the pool (v1.39: "
                         "10000; task 3.3 measured +0.3 points at 50k)")
    ap.add_argument("--rows", default=None,
                    help="training row ranges, e.g. '0:10000,110000:300000' "
                         "(overrides --n-train). Use this to skip "
                         "0528_X_train_0_770[10000:110000], the synthetic "
                         "generator's fit pool (tools/verify_no_leakage.py E1/E4).")
    ap.add_argument("--window", type=int, default=770,
                    help="classify on samples [0, window) only (task 4.1g; "
                         "window_tradeoff.py chose 576). The model input is "
                         "2*window; outputs get a _w<window> suffix unless 770.")
    ap.add_argument("--xcal", default="0528_X_train_0_770_loopback_calibrated.npy",
                    help="domain-mapped training pool (task 4.1h: the boxcar-path "
                         "map, build_box_training_set.py, rows in --rows order)")
    ap.add_argument("--tag", default="",
                    help="extra output suffix (e.g. _box), so a variant never "
                         "overwrites the deployed model's artifacts")
    ap.add_argument("--no-sanity", action="store_true",
                    help="skip the k2/c2 offline check (it applies the FIR-path map)")
    a = ap.parse_args()
    sfx = ("" if a.window == 770 else f"_w{a.window}") + a.tag
    if a.rows:
        ranges = [tuple(int(v) for v in r.split(":")) for r in a.rows.split(",")]
    else:
        ranges = [(0, a.n_train)]
    rows = np.concatenate([np.arange(lo, hi) for lo, hi in ranges])

    print("=" * 70)
    print("v1.37 SIGNED-domain training on the loopback-domain-calibrated set")
    print(f"training rows {ranges} ({len(rows)} shots)")
    print(f"training pool: {a.xcal}")
    Xcal = np.load(base / a.xcal, mmap_mode="r")
    X_train = np.concatenate([Xcal[lo:hi] for lo, hi in ranges]).astype("float32")
    if a.xcal == "0528_X_train_0_770_loopback_calibrated.npy":
        y_train = np.load(base / "0528_y_train_0_770.npy")[rows].astype("float32")
    else:
        # a mapped subset stores its rows in order, with their pool indices beside it
        src_rows = np.load(base / (Path(a.xcal).stem + "_rows.npy"))
        y_train = np.load(base / "0528_y_train_0_770.npy")[src_rows[rows]].astype("float32")
    X_train = hw_raw_transform(X_train, SCALING_FACTOR)[:, :2 * a.window]
    print(f"window [0, {a.window}) -> {X_train.shape[1]} inputs")
    print(f"input range after transform: [{X_train.min():.0f}, {X_train.max():.0f}]"
          f"  (negative present: {bool((X_train < 0).any())})")

    # Seed selection by validation loss: seed-to-seed spread on the acquired eval set (+/-0.24 points) is
    # comparable to the domain map's benefit (+0.37). val_loss is computed on the held-out 20% of the training
    # pool, so the selection uses no test or acquired shot.
    print(f"\nTraining {len(a.seeds)} seed(s); keeping the lowest val_loss")
    best = None
    for seed in a.seeds:
        tf.keras.utils.set_random_seed(seed)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(y_train))
        m = build_model(X_train.shape[1])
        es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                             restore_best_weights=True)
        h = m.fit(X_train[perm], y_train[perm], validation_split=0.2,
                  epochs=50, batch_size=32, verbose=0, callbacks=[es])
        vl = float(min(h.history["val_loss"]))
        va = float(h.history["val_accuracy"][int(np.argmin(h.history["val_loss"]))])
        print(f"  seed {seed}  val_loss {vl:.5f}  val_acc {va*100:.2f}%  "
              f"({len(h.history['val_loss'])} epochs)")
        if best is None or vl < best[0]:
            if best is not None:
                tf.keras.backend.clear_session()
            best = (vl, seed, m)
        else:
            tf.keras.backend.clear_session()
    val_loss, SEED_USED, model = best
    print(f"  -> selected seed {SEED_USED} (val_loss {val_loss:.5f})")

    print("\n" + "=" * 70)
    print("Converting through hls4ml")
    import hls4ml

    hls_config = make_hls_config(model)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    PROJECT_DIR = Path(r"C:\hls4ml_projects") / f"hls4ml_qubit_prj{sfx}_{stamp}"
    print("Using output_dir:", PROJECT_DIR)

    hls_model = hls4ml.converters.convert_from_keras_model(
        model=model,
        hls_config=hls_config,
        output_dir=str(PROJECT_DIR),
        part=FPGA_PART,
        io_type="io_parallel",
        clock_period=CLOCK_PERIOD_NS,
        backend="Vivado",
        project_name="NN",
    )
    hls_model.write()
    print("Wrote HLS project to:", PROJECT_DIR)
    print("firmware dir:", PROJECT_DIR / "firmware")

    with open(base / f"last_hls4ml_export_dir_signed{sfx}.txt", "w") as f:
        f.write(str(PROJECT_DIR))

    model.save(base / f"models/model_signed_domain_seed20260901{sfx}.h5")
    print(f"Saved Keras model to models/model_signed_domain_seed20260901{sfx}.h5")
    (base / f"last_export_seed{sfx}.txt").write_text(
        f"{SEED_USED}\nval_loss={val_loss:.6f}\nrows={ranges}\n", encoding="utf-8")
    print(f"Recorded the selected seed in results/last_export_seed.txt ({SEED_USED})")

    if a.no_sanity:
        print("\n=== DONE (offline k2/c2 check skipped) ===")
        return
    # Offline sanity check on the SCORED (untouched, non-leaked) range.
    print("\n=== Offline sanity check (test20k_real_X[:500], via k2/c2) ===")
    k2 = complex(np.load(base / "calibration/k2_loopback.npy"))
    c2 = complex(np.load(base / "calibration/c2_loopback.npy"))
    X_test = np.load(base / "data/test20k_real_X.npy")[:500].astype("float32")
    y_test = np.load(base / "data/test20k_real_y.npy")[:500].astype(int)
    z_test = X_test[:, 0::2] + 1j * X_test[:, 1::2]
    z_test_matched = k2 * z_test + c2
    X_test_matched = np.empty_like(X_test)
    X_test_matched[:, 0::2] = z_test_matched.real
    X_test_matched[:, 1::2] = z_test_matched.imag
    Xt = hw_raw_transform(X_test_matched, SCALING_FACTOR)[:, :2 * a.window]
    p = model.predict(Xt, verbose=0).reshape(-1)
    pred = (p > 0.5).astype(int)
    g, e = y_test == 0, y_test == 1
    ga = float(np.mean(pred[g] == y_test[g])) if g.sum() else float("nan")
    ea = float(np.mean(pred[e] == y_test[e])) if e.sum() else float("nan")
    print(f"overall={np.mean(pred==y_test)*100:.2f}%  ground={ga*100:.2f}%  "
          f"excited={ea*100:.2f}%  fidelity={0.5*(ga+ea)*100:.2f}%")
    print(f"prob range: {p.min():.4f} -> {p.max():.4f}")

    print("\n=== DONE ===")

if __name__ == "__main__":
    main()
