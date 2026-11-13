"""
Generate the synthetic holdout (x/y/Phi_synth_v2.npy) from a generator fit pool disjoint from the real test set
and from the classifier's training data (the pool 0528_X_all_{0,1}[400000:] contains the real test shots, so it is
not used).

Three-way split of the 1,000,000 real shots:
    classifier training : 0528_X_train_0_770.npy[:10000]
    generator fit pool  : 0528_X_train_0_770.npy[10000:110000]   <-- here
    real test set       : 0528_X_test_0_770.npy                  (disjoint file)

Noise model: full real-valued [I,Q] covariance (noise_mode="measured"), the
only variant whose matched-filter ceiling matches the real data.

"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, ".")
from qubit_synth_lib import (fit_cavity_and_noise_model, generate_dataset,
                             matched_filter_ceiling)
from generate_holdout_measured_noise import FULL_POOL_WARM_START

N_SHOTS = 50000
SEED = 20260902
FIT_LO, FIT_HI = 10000, 110000

def main():
    X = np.load("0528_X_train_0_770.npy", mmap_mode="r")
    y = np.load("0528_y_train_0_770.npy").astype(int)
    sl = slice(FIT_LO, FIT_HI)
    Xp = np.asarray(X[sl]).astype(np.float64)
    yp = y[sl]
    X_h0, X_h1 = Xp[yp == 0], Xp[yp == 1]
    print("fit pool rows [%d:%d] -> class0 %s  class1 %s"
          % (FIT_LO, FIT_HI, X_h0.shape, X_h1.shape))

    print("\n=== REAL reference on the SAME pool ===")
    nr = min(6000, len(X_h0), len(X_h1))
    Xr = np.concatenate([X_h0[:nr], X_h1[:nr]])
    yr = np.concatenate([np.zeros(nr, int), np.ones(nr, int)])
    dr, fr = matched_filter_ceiling(Xr, yr)
    print("  real fit pool: d = %.3f sigma, matched-filter fidelity %.2f%%" % (dr, fr * 100))

    print("\n=== FITTING ===")
    model = fit_cavity_and_noise_model(X_h0, X_h1, initial_guess=FULL_POOL_WARM_START)

    print("\n=== GENERATING %d shots (measured noise) ===" % N_SHOTS)
    Xg, yg, Phi = generate_dataset(model, N_SHOTS, add_noise=True, seed=SEED,
                                   noise_mode="measured")
    np.save("x_synth_v2.npy", Xg)
    np.save("y_synth_v2.npy", yg)
    np.save("Phi_synth_v2.npy", Phi)
    print("saved x/y/Phi_synth_v2.npy", Xg.shape)
    print("balance:", dict(zip(*np.unique(yg, return_counts=True))))

    # Phi consistency: the recorded phase must match the traces' rotation
    z = Xg[:, 0::2] + 1j * Xg[:, 1::2]
    a = np.zeros(len(z))
    for s in (0, 1):
        m = z[yg == s].mean(0)
        a[yg == s] = np.angle((z[yg == s] * np.conj(m)).sum(1))
    P = Phi[:, 0]
    sl_ = np.polyfit(P - P.mean(), a - a.mean(), 1)[0]
    print("  Phi consistency: regression slope = %.3f (want ~1.0)" % sl_)

if __name__ == "__main__":
    main()
