"""
Characterise the noise structure of the real qubit readout data
(0528_X_train_0_770.npy), to decide how it should actually be modelled.

The synthetic generator's noise model should match the real data: fidelity depends strongly on how hard the
data is (the on-chip classifier scores 100% on the white-noise x_50k but 72% on x_50k_heldout).

Structure of the real set: 900,000 shots x 770 complex samples, balanced,
CLASS-BLOCKED in 180 alternating runs of 5000. Sampling is therefore done
block-wise and spread across the whole file, so slow drift is not aliased
into the noise estimate.

Questions, each with a concrete modelling consequence:

 Q1. Is the noise state-dependent? Additive white Gaussian noise (AWGN) is
     state-independent by construction. In real dispersive readout the |1>
     trajectory carries extra variance because the qubit can DECAY during
     the readout window. If sigma_1 > sigma_0, an AWGN generator is
     structurally wrong, not merely mistuned.

 Q2. Is it stationary along the trace, or does the ring-up transient carry
     different statistics from the steady state?

 Q3. Is it white, or correlated sample-to-sample? The readout chain has
     finite bandwidth, so correlated ("coloured") noise is expected. This
     matters because integration gain is sqrt(N/rho*), not sqrt(N).

 Q4. Is it Gaussian? T1 decay produces heavy tails and bimodality in the
     residual, which no Gaussian model reproduces.

 Q5. Is it isotropic in I/Q?

 Q6. THE DECISIVE ONE -- are there mid-trace state transitions? Project
     each shot onto the discriminant axis w = mean1 - mean0 and track it
     versus time. A |1> shot that decays partway through starts at the
     excited level and ends at the ground level. The fraction of such shots
     and the distribution of their transition times give T1 directly (the
     transition times should be exponential). This is the single feature
     that separates a realistic readout model from AWGN.

"""
import numpy as np
from scipy import stats
from pathlib import Path

BASE = Path(".")
BLOCK = 5000
N_BLOCKS_USED = 12          # spread across the file
PER_BLOCK = 1200            # shots taken from each sampled block

def hdr(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)

def main():
    X = np.load(BASE / "0528_X_train_0_770.npy", mmap_mode="r")
    y = np.load(BASE / "0528_y_train_0_770.npy").astype(int)
    n_blocks = X.shape[0] // BLOCK
    print("dataset: %s, %d blocks of %d (alternating class)" % (X.shape, n_blocks, BLOCK))

    # sample blocks evenly across the file, keeping both classes
    idx0, idx1 = [], []
    picks = np.linspace(0, n_blocks - 2, N_BLOCKS_USED).astype(int)
    for b in picks:
        lo = b * BLOCK
        cls = y[lo]
        rows = np.arange(lo, lo + PER_BLOCK)
        (idx0 if cls == 0 else idx1).append(rows)
    idx0 = np.concatenate(idx0) if idx0 else np.array([], int)
    idx1 = np.concatenate(idx1) if idx1 else np.array([], int)
    print("sampled %d ground / %d excited shots from %d blocks spread across the file"
          % (len(idx0), len(idx1), N_BLOCKS_USED))

    def load(idx):
        return np.asarray(X[idx]).astype(np.float64)

    A0, A1 = load(idx0), load(idx1)
    z0 = A0[:, 0::2] + 1j * A0[:, 1::2]
    z1 = A1[:, 0::2] + 1j * A1[:, 1::2]
    del A0, A1
    m0, m1 = z0.mean(axis=0), z1.mean(axis=0)
    T = z0.shape[1]

    # ---------------------------------------------------------------- Q1
    hdr("Q1. Is the noise STATE-DEPENDENT?  (AWGN assumes it is not)")
    r0, r1 = z0 - m0, z1 - m1
    s0 = np.std(r0), np.std(r1)
    print("  residual std, ground  |0> : %10.3f" % np.std(r0))
    print("  residual std, excited |1> : %10.3f" % np.std(r1))
    print("  ratio sigma_1 / sigma_0   : %10.4f" % (np.std(r1) / np.std(r0)))
    print("  signal separation |m1-m0| : %10.3f  (mean over trace)"
          % np.mean(np.abs(m1 - m0)))
    if np.std(r1) / np.std(r0) > 1.15:
        print("  => STATE-DEPENDENT. The excited state is noisier, which AWGN")
        print("     cannot represent. Expect T1 decay during readout (see Q6).")
    else:
        print("  => roughly state-independent on this statistic.")

    # ---------------------------------------------------------------- Q2
    hdr("Q2. Is the noise STATIONARY along the trace?")
    st0 = np.std(r0, axis=0)
    st1 = np.std(r1, axis=0)
    seg = T // 5
    print("  %-14s %10s %10s %12s" % ("trace segment", "sigma_0", "sigma_1", "|m1-m0|"))
    for i in range(5):
        a, b = i * seg, min((i + 1) * seg, T)
        print("  samples %3d-%3d %10.3f %10.3f %12.3f"
              % (a, b, st0[a:b].mean(), st1[a:b].mean(),
                 np.abs(m1 - m0)[a:b].mean()))
    print("  sigma_0 across trace: min %.2f max %.2f (ratio %.2f)"
          % (st0.min(), st0.max(), st0.max() / st0.min()))

    # ---------------------------------------------------------------- Q3
    hdr("Q3. Is the noise WHITE or COLOURED?")
    def acorr(r, lags=(1, 2, 3, 5, 10, 20, 50)):
        rn = r / np.std(r)
        out = []
        for k in lags:
            a, b = rn[:, :-k], rn[:, k:]
            out.append(float(np.mean((a * np.conj(b)).real)))
        return out
    lags = (1, 2, 3, 5, 10, 20, 50)
    a0, a1 = acorr(r0), acorr(r1)
    print("  lag      " + "".join("%8d" % k for k in lags))
    print("  rho |0>  " + "".join("%8.3f" % v for v in a0))
    print("  rho |1>  " + "".join("%8.3f" % v for v in a1))
    rho_star = 1 + 2 * sum((1 - k / T) * v for k, v in zip(lags, a0))
    print("  effective integration gain sqrt(N/rho*) = %.2f  vs naive sqrt(%d) = %.2f"
          % (np.sqrt(T / max(rho_star, 1e-6)), T, np.sqrt(T)))

    # ---------------------------------------------------------------- Q4
    hdr("Q4. Is the noise GAUSSIAN?")
    for lab, r in [("|0>", r0), ("|1>", r1)]:
        f = r.real.ravel() / np.std(r.real)
        f = f[::max(1, len(f) // 200000)]
        print("  %s  skew %+7.4f   excess kurtosis %+8.4f   KS(D) %.4f"
              % (lab, stats.skew(f), stats.kurtosis(f),
                 stats.kstest(f[::7], "norm").statistic))
    print("  (excess kurtosis >> 0 = heavy tails, the signature of jump events)")

    # ---------------------------------------------------------------- Q5
    hdr("Q5. Is the I/Q noise ISOTROPIC?")
    for lab, r in [("|0>", r0), ("|1>", r1)]:
        si, sq = np.std(r.real), np.std(r.imag)
        rho = np.corrcoef(r.real.ravel()[::97], r.imag.ravel()[::97])[0, 1]
        print("  %s  std_I %9.3f  std_Q %9.3f  ratio %.4f  corr(I,Q) %+.4f"
              % (lab, si, sq, si / sq, rho))

    # ---------------------------------------------------------------- Q6
    hdr("Q6. MID-TRACE STATE TRANSITIONS (the T1 signature)")
    w = (m1 - m0)
    w = w / (np.vdot(w, w).real ** 0.5)
    # projection of each shot onto the discriminant axis, per time sample
    p0 = (z0 * np.conj(w)).real
    p1 = (z1 * np.conj(w)).real
    lvl0 = (m0 * np.conj(w)).real
    lvl1 = (m1 * np.conj(w)).real
    mid = 0.5 * (lvl0.mean() + lvl1.mean())
    print("  ground level %.4f, excited level %.4f, midpoint %.4f"
          % (lvl0.mean(), lvl1.mean(), mid))

    # smooth each shot's projection so single-sample noise doesn't dominate
    K = 40
    ker = np.ones(K) / K
    def smooth(p):
        return np.apply_along_axis(lambda v: np.convolve(v, ker, mode="valid"), 1, p)
    sp0, sp1 = smooth(p0), smooth(p1)
    above = lvl1.mean() > lvl0.mean()
    def frac_excited(sp):
        return (sp > mid) if above else (sp < mid)
    e0, e1 = frac_excited(sp0), frac_excited(sp1)
    print("\n  fraction of the (smoothed) trace spent at the EXCITED level:")
    print("    first 10%% of trace : |0> %.3f   |1> %.3f"
          % (e0[:, :e0.shape[1] // 10].mean(), e1[:, :e1.shape[1] // 10].mean()))
    print("    last  10%% of trace : |0> %.3f   |1> %.3f"
          % (e0[:, -e0.shape[1] // 10:].mean(), e1[:, -e1.shape[1] // 10:].mean()))

    # per-shot: started excited but ended ground => decayed mid-readout
    start1 = e1[:, :e1.shape[1] // 10].mean(axis=1) > 0.5
    end1 = e1[:, -e1.shape[1] // 10:].mean(axis=1) > 0.5
    decayed = start1 & ~end1
    never = ~start1
    print("\n  of %d |1> shots:" % len(e1))
    print("    started AND ended excited (no decay) : %6.2f%%"
          % (100 * np.mean(start1 & end1)))
    print("    started excited, ENDED GROUND (decay): %6.2f%%" % (100 * np.mean(decayed)))
    print("    never looked excited at all          : %6.2f%%" % (100 * np.mean(never)))
    start0 = e0[:, :e0.shape[1] // 10].mean(axis=1) > 0.5
    end0 = e0[:, -e0.shape[1] // 10:].mean(axis=1) > 0.5
    print("  of %d |0> shots: ended excited (thermal/mis-prep): %6.2f%%"
          % (len(e0), 100 * np.mean(end0)))

    if decayed.sum() > 20:
        # transition time = first crossing below midpoint
        cross = []
        for row in sp1[decayed]:
            ex = frac_excited(row[None, :])[0]
            idxs = np.flatnonzero(~ex)
            if len(idxs):
                cross.append(idxs[0])
        cross = np.array(cross, float)
        print("\n  decay transition times (samples into the smoothed trace):")
        print("    n=%d  median %.0f  mean %.0f  of %d samples"
              % (len(cross), np.median(cross), cross.mean(), sp1.shape[1]))
        # exponential? fit rate on the observed window
        if len(cross) > 30:
            lam = 1.0 / max(cross.mean(), 1e-9)
            ks = stats.kstest(cross, "expon", args=(0, 1 / lam))
            print("    exponential fit: mean lifetime %.0f samples "
                  "(KS D=%.3f, p=%.3g)" % (1 / lam, ks.statistic, ks.pvalue))
            print("    => if exponential, this IS T1 measured in readout samples")

    hdr("SUMMARY OF WHAT THE MODEL MUST REPRODUCE")
    print("  sigma_1/sigma_0 ......... %.3f" % (np.std(r1) / np.std(r0)))
    print("  excess kurtosis |1> ..... %+.3f" % stats.kurtosis(r1.real.ravel()[::13]))
    print("  lag-1 autocorrelation ... %+.3f" % a0[0])
    print("  I/Q anisotropy |1> ...... %.4f" % (np.std(r1.real) / np.std(r1.imag)))
    print("  mid-readout decay rate .. %.2f%% of |1> shots" % (100 * np.mean(decayed)))

if __name__ == "__main__":
    main()
