"""
2026-09-01: generate a new holdout dataset using the MEASURED noise model.

Replaces the white-noise generator used for x_50k_heldout.npy. See
tools/qubit_synth_lib.py's docstring for the characterisation this is based on
(char_real_readout_noise.py / char_real_readout_t1.py): the real 0528 noise
is coloured (autocorr +0.32, -0.20 at lags 1,2; sharp Nyquist cutoff) and
carries a per-shot common-mode offset that does not average down. Both are
reproduced by shaping the noise with the measured per-class PSD.

Fit pool is unchanged from generate_data_4_holdout.ipynb: the LAST 100,000
shots of each class, disjoint from the FIRST 400,000/class used to train the
classifier -- so the physics model that defines the test domain shares zero
rows with the classifier's training data (2026-08-31 audit finding #3).

ACCEPTANCE TEST: the generated set's matched-filter ceiling must land near
the real data's d = 3.885 sigma / 95.65%. The previous white-noise
generator produces data materially easier than that.

"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(".").resolve()))
from qubit_synth_lib import (fit_cavity_and_noise_model, generate_dataset,
                             matched_filter_ceiling)

RAW_BASE = Path(__import__("os").environ.get("RAW_DATA_DIR", "."))   # folder holding 0528_X_all_*.npy
N_TRAIN_PER_CLASS = 400000
N_SHOTS = 50000
SEED = 20260901

FULL_POOL_WARM_START = [
    1.01720706e+07, -4.54101244e+04, 3.64588334e+06, -3.07607069e-02,
    2.98501165e-02, 2.70625462e-02, -4.07594194e-02, 9.14256911e-02,
    3.18694928e-02, 6.96860703e-06, 3.61911904e+03,
]

def main():
    validate_only = "--validate-only" in sys.argv

    X_all_0 = np.load(RAW_BASE / "0528_X_all_0.npy", mmap_mode="r")
    X_all_1 = np.load(RAW_BASE / "0528_X_all_1.npy", mmap_mode="r")
    print("X_all_0", X_all_0.shape, " X_all_1", X_all_1.shape)

    X_h0 = np.asarray(X_all_0[N_TRAIN_PER_CLASS:])
    X_h1 = np.asarray(X_all_1[N_TRAIN_PER_CLASS:])
    print("Held-out fit pool:", X_h0.shape, X_h1.shape)

    print("\n=== REAL-DATA REFERENCE (the target the generator must match) ===")
    Xr = np.concatenate([X_h0[:6000], X_h1[:6000]])
    yr = np.concatenate([np.zeros(6000, int), np.ones(6000, int)])
    dr, fr = matched_filter_ceiling(Xr, yr)
    print(f"  real held-out pool: d = {dr:.3f} sigma, matched-filter fidelity {fr*100:.2f}%")

    print("\n=== FITTING CAVITY + NOISE MODEL ===")
    model = fit_cavity_and_noise_model(X_h0, X_h1,
                                       initial_guess=FULL_POOL_WARM_START)

    print("\n=== VALIDATION: 6000 shots, both noise models, side by side ===")
    for mode in ["white", "measured"]:
        Xv, yv, _ = generate_dataset(model, 6000, add_noise=True, seed=SEED,
                                     noise_mode=mode, verbose=False)
        d, f = matched_filter_ceiling(Xv, yv)
        z = Xv[:, 0::2] + 1j * Xv[:, 1::2]
        r = z - np.array([z[yv == 0].mean(axis=0), z[yv == 1].mean(axis=0)])[yv]
        rn = r / np.std(r)
        ac = [float(np.mean((rn[:, :-k] * np.conj(rn[:, k:])).real)) for k in (1, 2, 3)]
        cm = np.std(r.mean(axis=1))
        print(f"  {mode:9s} d = {d:6.3f} sigma   fidelity {f*100:6.2f}%   "
              f"autocorr {ac[0]:+.3f} {ac[1]:+.3f} {ac[2]:+.3f}   "
              f"per-shot common-mode std {cm:6.2f}")
    print(f"  {'REAL':9s} d = {dr:6.3f} sigma   fidelity {fr*100:6.2f}%   "
          f"autocorr +0.320 -0.197 +0.068   per-shot common-mode std ~64-79")

    if validate_only:
        print("\n--validate-only: stopping before generating the full dataset.")
        return

    print(f"\n=== GENERATING {N_SHOTS} SHOTS (measured noise model) ===")
    X, y, Phi = generate_dataset(model, N_SHOTS, add_noise=True, seed=SEED,
                                 noise_mode="measured")
    np.save("x_50k_heldout_measnoise.npy", X)
    np.save("y_50k_heldout_measnoise.npy", y)
    np.save("Phi_heldout_measnoise.npy", Phi)
    print("Saved x/y/Phi_50k_heldout_measnoise.npy", X.shape, y.shape, Phi.shape)
    print("Class balance:", dict(zip(*np.unique(y, return_counts=True))))

if __name__ == "__main__":
    main()
