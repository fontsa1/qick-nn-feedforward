"""
2026-09-01 follow-up: char_real_readout_noise.py's Q6 (mid-trace state
transitions) was underpowered -- per-sample SNR is only ~0.2, so a K=40
smoothing window gives ~1.3 sigma and the resulting "state per window"
assignments were dominated by noise (54.7% of |1> shots "never looked
excited", exponential fit rejected at p=1.7e-55). Those numbers measured
the estimator, not the qubit.

This redoes it with a matched filter and an explicit null model.

Method:
  w(t) = m1(t) - m0(t), the optimal linear discriminant per sample.
  Split the trace in half. For each shot compute the matched-filter score
  of each half, standardised so the |0>/|1> class means are at -1/+1.
  A shot that decays mid-readout scores EXCITED in half 1 and GROUND in
  half 2.

  The null (no decay, pure Gaussian noise) predicts a specific rate of
  (excited, ground) outcomes purely from noise. We generate that null by
  simulation using the MEASURED per-half noise, and compare. The excess of
  half1-excited/half2-ground over the null -- and its ASYMMETRY against the
  reverse pattern (half1-ground/half2-excited, which decay cannot produce)
  -- is the T1 signal. The asymmetry is the key control: noise produces
  both patterns equally, decay produces only one.

Also reports the OPTIMAL achievable single-shot fidelity on the real data
(full-trace matched filter, cross-validated), which is the ceiling any
classifier on this dataset can reach and the number our synthetic data
should be tuned to reproduce.

"""
import numpy as np
from scipy import stats
from scipy.special import ndtr
from pathlib import Path

BASE = Path(".")
BLOCK = 5000
N_BLOCKS = 16
PER_BLOCK = 1500

def hdr(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)

def main():
    X = np.load(BASE / "0528_X_train_0_770.npy", mmap_mode="r")
    y = np.load(BASE / "0528_y_train_0_770.npy").astype(int)
    nb = X.shape[0] // BLOCK
    i0, i1 = [], []
    for b in np.linspace(0, nb - 2, N_BLOCKS).astype(int):
        lo = b * BLOCK
        (i0 if y[lo] == 0 else i1).append(np.arange(lo, lo + PER_BLOCK))
    i0 = np.concatenate(i0)
    i1 = np.concatenate(i1)
    A0 = np.asarray(X[i0]).astype(np.float64)
    A1 = np.asarray(X[i1]).astype(np.float64)
    z0 = A0[:, 0::2] + 1j * A0[:, 1::2]
    z1 = A1[:, 0::2] + 1j * A1[:, 1::2]
    del A0, A1
    print("ground %d shots, excited %d shots" % (len(z0), len(z1)))

    # split-half means so the filter isn't fit on the shots it scores
    h0, h1 = len(z0) // 2, len(z1) // 2
    m0 = z0[:h0].mean(axis=0)
    m1 = z1[:h1].mean(axis=0)
    ev0, ev1 = z0[h0:], z1[h1:]
    w = m1 - m0
    T = len(w)

    def mf(z, sl):
        """matched-filter score over a slice of the trace"""
        return (z[:, sl] * np.conj(w[sl])).real.sum(axis=1)

    # ------------------------------------------------ optimal fidelity
    hdr("CEILING: optimal single-shot fidelity on the REAL data (full trace)")
    full = slice(0, T)
    s0, s1 = mf(ev0, full), mf(ev1, full)
    mu0, mu1 = s0.mean(), s1.mean()
    sd = np.sqrt(0.5 * (s0.var() + s1.var()))
    d = abs(mu1 - mu0) / sd
    thr = 0.5 * (mu0 + mu1)
    acc = 0.5 * (np.mean(s0 < thr) + np.mean(s1 > thr)) if mu1 > mu0 \
        else 0.5 * (np.mean(s0 > thr) + np.mean(s1 < thr))
    print("  class separation d = %.3f sigma   (Gaussian-optimal fidelity %.2f%%)"
          % (d, ndtr(d / 2) * 100))
    print("  MEASURED matched-filter fidelity  = %.2f%%  (held-out half)" % (acc * 100))
    print("  per-sample SNR |m1-m0|/sigma      = %.4f"
          % (np.mean(np.abs(w)) / np.std(ev0 - m0)))
    print("\n  => this is the ceiling. Any classifier on this dataset, on-chip or")
    print("     offline, is bounded by it, and our synthetic generator should be")
    print("     tuned so its own ceiling matches.")

    # ------------------------------------------------ half-split T1 test
    hdr("T1 TEST: half-split matched filter, with a noise null and an asymmetry control")
    A, B = slice(0, T // 2), slice(T // 2, T)
    def std_scores(z):
        a, b = mf(z, A), mf(z, B)
        return a, b
    a0, b0 = std_scores(ev0)
    a1, b1 = std_scores(ev1)
    # standardise each half so |0> mean = -1, |1> mean = +1
    def norm(x, r0, r1):
        c, s = 0.5 * (r0.mean() + r1.mean()), 0.5 * abs(r1.mean() - r0.mean())
        return (x - c) / s
    A0n, A1n = norm(a0, a0, a1), norm(a1, a0, a1)
    B0n, B1n = norm(b0, b0, b1), norm(b1, b0, b1)
    nA, nB = np.std(A1n), np.std(B1n)
    print("  half A: |0> mean %.3f  |1> mean %.3f  noise sd %.3f  (d=%.2f sigma)"
          % (A0n.mean(), A1n.mean(), nA, abs(A1n.mean() - A0n.mean()) / nA))
    print("  half B: |0> mean %.3f  |1> mean %.3f  noise sd %.3f  (d=%.2f sigma)"
          % (B0n.mean(), B1n.mean(), nB, abs(B1n.mean() - B0n.mean()) / nB))

    # observed patterns among |1> shots
    exA, exB = A1n > 0, B1n > 0
    obs_decay = np.mean(exA & ~exB)     # excited then ground  <- T1 can cause
    obs_rise = np.mean(~exA & exB)      # ground then excited  <- T1 CANNOT cause
    print("\n  among %d held-out |1> shots:" % len(A1n))
    print("    half A excited, half B ground  (decay-like) : %6.3f%%" % (100 * obs_decay))
    print("    half A ground,  half B excited (reverse)    : %6.3f%%" % (100 * obs_rise))
    print("    asymmetry (decay - reverse)                 : %+6.3f pp"
          % (100 * (obs_decay - obs_rise)))

    # null: no decay, both halves excited-mean with measured noise
    rng = np.random.default_rng(0)
    NSIM = 200
    nd, nr = [], []
    for _ in range(NSIM):
        sa = A1n.mean() + rng.normal(0, nA, len(A1n))
        sb = B1n.mean() + rng.normal(0, nB, len(B1n))
        nd.append(np.mean((sa > 0) & (sb <= 0)))
        nr.append(np.mean((sa <= 0) & (sb > 0)))
    nd, nr = np.array(nd), np.array(nr)
    print("\n  Gaussian null (no decay, measured noise, %d sims):" % NSIM)
    print("    decay-like : %6.3f%% +/- %.3f" % (100 * nd.mean(), 100 * nd.std()))
    print("    reverse    : %6.3f%% +/- %.3f" % (100 * nr.mean(), 100 * nr.std()))
    z_asym = ((obs_decay - obs_rise) - (nd.mean() - nr.mean())) / \
             np.sqrt(nd.var() + nr.var() + 1e-18)
    print("\n  EXCESS asymmetry over null: %+.3f pp  (z = %+.2f)"
          % (100 * ((obs_decay - obs_rise) - (nd.mean() - nr.mean())), z_asym))
    if abs(z_asym) < 3:
        print("  => NO significant decay signature. Noise alone explains the")
        print("     observed half-to-half flips. A jump/T1 term is NOT needed.")
    else:
        print("  => SIGNIFICANT decay asymmetry: a T1 jump process IS present")
        print("     and must be modelled; AWGN alone cannot reproduce it.")

    # ------------------------------------------------ noise spectrum
    hdr("NOISE SPECTRUM (for generating correctly-coloured noise)")
    r = ev0 - m0
    R = np.fft.fft(r, axis=1)
    psd = (np.abs(R) ** 2).mean(axis=0)
    psd = psd / psd.mean()
    f = np.fft.fftfreq(T)
    k = np.argsort(f)
    print("  normalised PSD at selected normalised frequencies:")
    for frac in [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
        i = np.argmin(np.abs(f[k] - frac))
        print("    f = %+.2f  ->  PSD %.3f" % (f[k][i], psd[k][i]))
    ac = []
    rn = r / np.std(r)
    for lag in (1, 2, 3, 4, 5):
        ac.append(float(np.mean((rn[:, :-lag] * np.conj(rn[:, lag:])).real)))
    print("  autocorrelation lags 1-5: " + " ".join("%+.4f" % v for v in ac))
    print("\n  A generator can reproduce this by filtering white complex Gaussian")
    print("  noise with an FIR whose autocorrelation matches the above, or by")
    print("  sampling directly from this measured PSD.")

    hdr("SIGNAL SHAPE (the ring-up transient the generator must include)")
    sep = np.abs(w)
    for i in range(10):
        a, b = i * T // 10, (i + 1) * T // 10
        print("    samples %3d-%3d  |m1-m0| = %8.2f" % (a, b, sep[a:b].mean()))

if __name__ == "__main__":
    main()
