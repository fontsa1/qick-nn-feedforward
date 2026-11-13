"""Fit the affine domain map (k2, c2) and apply it to the training set.

The map is fit on the domain-calibration slice (`data/domain_calib_raw_X.npy` / `data/domain_calib_acquired.npy`,
rows [1000:1600] of `test20k_real_X`, excluded from every scored set) and applied to `0528_X_train_0_770.npy`, so the
classifier trains on data in the domain the loopback chain delivers.

--fit global (default) is label-free: one complex affine fit to the mean trajectory over all shots. --fit perclass
stacks the two class-mean trajectories and uses `data/domain_calib_y.npy` (test-file labels); it is kept only to
reproduce earlier artifacts. Measured with phase0_calibration_ablation.py (3 arms x 2 windows x 3 seeds, on 20,000
real shots played through the DAC and back through the ADC):

    raw, no map at all                    94.76 +/- 0.15
    per-class map (uses test labels)      95.12 +/- 0.18
    global map (label-free)               95.13 +/- 0.11

So the labels add nothing (+0.01 points), and the map as a whole adds +0.37 +/- 0.19 points.
"""
import argparse
import numpy as np

OUT_X = "0528_X_train_0_770_loopback_calibrated.npy"

def fit_global(z_raw, z_acquired):
    """Label-free complex affine fit: mean trajectory over ALL shots, lstsq.

    Mirrors `tools/run_calibration.py`'s own fit. Four real degrees of freedom (a
    complex gain and a complex offset), applied identically to every shot
    regardless of class, so it cannot carry class-discriminative information.
    """
    zr = z_raw.mean(axis=0)
    za = z_acquired.mean(axis=0)
    A = np.vstack([zr, np.ones_like(zr)]).T
    sol, *_ = np.linalg.lstsq(A, za, rcond=None)
    return complex(sol[0]), complex(sol[1])

def fit_perclass(z_raw, z_acquired, y):
    """Per-class fit: uses test-file labels (`y`).

    Stacks the two class-mean trajectories so the joint regression keeps the state-dependent signal.
    """
    z_raw_0 = np.mean(z_raw[y == 0], axis=0)
    z_raw_1 = np.mean(z_raw[y == 1], axis=0)
    z_acq_0 = np.mean(z_acquired[y == 0], axis=0)
    z_acq_1 = np.mean(z_acquired[y == 1], axis=0)
    A = np.concatenate([z_raw_0, z_raw_1])
    b = np.concatenate([z_acq_0, z_acq_1])
    A_matrix = np.vstack([A, np.ones_like(A)]).T
    sol, *_ = np.linalg.lstsq(A_matrix, b, rcond=None)
    return complex(sol[0]), complex(sol[1])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", choices=["global", "perclass"], default="global",
                    help="global (default, LABEL-FREE, no F1 leakage) or "
                         "perclass (historical; consumes test-file labels)")
    a = ap.parse_args()

    acquired = np.load("data/domain_calib_acquired.npy")
    raw_calib = np.load("data/domain_calib_raw_X.npy")
    z_acquired = acquired[:, 0::2] + 1j * acquired[:, 1::2]
    z_raw = raw_calib[:, 0::2] + 1j * raw_calib[:, 1::2]

    if a.fit == "global":
        k2, c2 = fit_global(z_raw, z_acquired)
        print("fit = GLOBAL (label-free). data/domain_calib_y.npy is NOT read; no "
              "test-set label touches any fitted quantity.")
    else:
        y_calib = np.load("data/domain_calib_y.npy").astype(int)
        k2, c2 = fit_perclass(z_raw, z_acquired, y_calib)
        print("fit = PERCLASS. *** THIS CONSUMES TEST-FILE LABELS (leakage "
              "F1). Historical reproduction only. ***")

    print(f"k2 = {k2!r}")
    print(f"c2 = {c2!r}")
    print(f"|k2| = {abs(k2):.5f}   arg(k2) = {np.degrees(np.angle(k2)):+.2f} deg")

    X_train_raw = np.load("0528_X_train_0_770.npy").astype("float32")
    z_train = X_train_raw[:, 0::2] + 1j * X_train_raw[:, 1::2]
    z_train_matched = k2 * z_train + c2

    X_train_calibrated = np.empty_like(X_train_raw)
    X_train_calibrated[:, 0::2] = z_train_matched.real
    X_train_calibrated[:, 1::2] = z_train_matched.imag

    np.save(OUT_X, X_train_calibrated)
    np.save("calibration/k2_loopback.npy", k2)
    np.save("calibration/c2_loopback.npy", c2)
    np.save(f"k2_loopback_{a.fit}.npy", k2)
    np.save(f"c2_loopback_{a.fit}.npy", c2)

    print(f"\nsaved {OUT_X} {X_train_calibrated.shape}")
    print(f"saved calibration/k2_loopback.npy / calibration/c2_loopback.npy (and the "
          f"_{a.fit}-tagged copies)")
    print("pre-transform peak: %.2f  post-transform peak: %.2f" %
          (np.max(np.abs(z_train)), np.max(np.abs(z_train_matched))))
    print("sanity: acquired domain calib slice peak=%.2f mean|.|=%.2f" %
          (np.max(np.abs(z_acquired)), np.mean(np.abs(z_acquired))))
    print("sanity: transformed training data mean|.|=%.2f (should be in the "
          "same ballpark as the acquired slice, not the raw training scale)" %
          np.mean(np.abs(z_train_matched)))

if __name__ == "__main__":
    main()
