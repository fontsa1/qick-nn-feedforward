"""
Cavity-response fit and synthetic-shot generation for the synthetic readout datasets.

fit_cavity_and_noise_model() fits the cavity response (joint least squares) and per-class noise models to real
shots; generate_dataset() draws synthetic shots. noise_mode:
  * "white": independent per-timestep noise with a per-class covariance (Cholesky factor per timestep).
  * "measured" (default): the empirical class mean, rotated by the shot's phase, plus a draw from the full
    real-valued covariance in the interleaved [I, Q] space, which captures colouring, common mode,
    signal-aligned power and I/Q non-circularity. The fit also stores the measured per-class PSD and std profile.
The real 0528 noise is coloured (autocorrelation at lags 1-5: +0.3195, -0.1967, +0.0678, +0.0019, -0.0576; PSD
cutoff near Nyquist), and each shot carries a common-mode offset that does not average down (37-49% of the
integrated variance is shared between trace halves). A per-shot constant offset is a DC component of the PSD:
predicted per-shot offset std sigma*sqrt(PSD_norm[0]/n) = 1185*sqrt(2.243/770) = 63.95, measured 63.98.
Measured and not modelled because absent: a T1/jump process (half-split decay asymmetry z = +2.30), heavy tails
(excess kurtosis -0.017), state-dependent variance (sigma_1/sigma_0 = 1.002), I/Q anisotropy (std_I/std_Q = 0.9996).

Acceptance test: matched_filter_ceiling() reproduces the ceiling the real data supports, d = 3.885 sigma, 95.65%
held-out matched-filter fidelity. A generator whose ceiling is above that produces data easier than the real data.
"""
import numpy as np
from scipy.optimize import least_squares

N_LOWRANK = 8   # eigen-components sampled explicitly (carry signal-aligned power)


def cavity_response_fit(params, state, offset_I, offset_Q, n=770, dt=2e-9):
    (
        kappa, delta0, delta1,
        ass0_real, ass0_imag,
        ass1_real, ass1_imag,
        a0_real, a0_imag,
        phase, gain,
    ) = params

    delta = delta0 if state == 0 else delta1
    t = np.arange(n) * dt

    if state == 0:
        a_ss = ass0_real + 1j * ass0_imag
    else:
        a_ss = ass1_real + 1j * ass1_imag

    a0 = a0_real + 1j * a0_imag
    a = a_ss + (a0 - a_ss) * np.exp((-kappa / 2 + 1j * delta) * t)

    z = gain * a * np.exp(1j * phase)
    z += offset_I + 1j * offset_Q
    return z


def cavity_response_shot(params, state, offset_I, offset_Q, n=770, dt=2e-9,
                         rng=None, phase_sigma=0.1):
    """Returns (z, phi, shot_phase).

    phi is this shot's phase replicated across the trace (the Phi.npy row layout, [[p0]*n, [p1]*n, ...]);
    shot_phase is the same number as a scalar, so the caller applies the same rotation to its deterministic part.
    Drawn from the passed rng.
    """
    z = cavity_response_fit(params, state, offset_I, offset_Q, n, dt)
    r = rng if rng is not None else np.random
    shot_phase = r.normal(0, phase_sigma)
    phi = np.zeros(n) + params[9] + shot_phase
    z = z * np.exp(1j * shot_phase)
    return z, phi, shot_phase


_INITIAL = [
    1 / (100e-9), -2 * np.pi * 2e6, 2 * np.pi * 2e6,
    1, 0, 1, 0, 1, 0, 0, 3000,
]
_LOWER = [1e3, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.pi, 0]
_UPPER = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.pi, np.inf]


def fit_cavity_and_noise_model(X_all_0, X_all_1, n=770, verbose=True, initial_guess=None):
    """
    Cavity fit (joint least squares, offset_I/offset_Q) plus separate per-timestep noise covariances for each
    class, returned as Cholesky factors for vectorized per-shot noise generation.

    X_all_0 / X_all_1: (N0, 2*n) and (N1, 2*n) interleaved [I0,Q0,I1,Q1,...] real-shot arrays for class 0 / class 1.
    N0 and N1 need not be equal or the full pool.
    """
    if verbose:
        print(f"Fitting on {X_all_0.shape[0]} class-0 shots and {X_all_1.shape[0]} class-1 shots")

    mean_real0 = np.mean(X_all_0, axis=0)
    mean_real1 = np.mean(X_all_1, axis=0)
    z0_real = mean_real0[0::2] + 1j * mean_real0[1::2]
    z1_real = mean_real1[0::2] + 1j * mean_real1[1::2]

    offset_I = np.mean(np.concatenate([z0_real.real[-100:], z1_real.real[-100:]]))
    offset_Q = np.mean(np.concatenate([z0_real.imag[-100:], z1_real.imag[-100:]]))

    def residual_joint(params):
        weights = np.exp(-np.arange(n) / 200)
        z0_fit = cavity_response_fit(params, 0, offset_I, offset_Q, n=n)
        z1_fit = cavity_response_fit(params, 1, offset_I, offset_Q, n=n)
        r0 = weights * (z0_fit - z0_real)
        r1 = weights * (z1_fit - z1_real)
        return np.concatenate([r0.real, r0.imag, r1.real, r1.imag])

    x0 = _INITIAL if initial_guess is None else initial_guess
    fit = least_squares(residual_joint, x0, bounds=(_LOWER, _UPPER))
    params0 = fit.x
    params1 = fit.x
    if verbose:
        print("Fitted cavity params:", params0)
        print(f"least_squares status={fit.status} cost={fit.cost:.1f}")
        z0_fit_check = cavity_response_fit(params0, 0, offset_I, offset_Q, n=n)
        z1_fit_check = cavity_response_fit(params1, 1, offset_I, offset_Q, n=n)
        print(f"  z0 mean abs residual: {np.mean(np.abs(z0_real - z0_fit_check)):.2f}"
              f" (signal scale {np.mean(np.abs(z0_real)):.2f})")
        print(f"  z1 mean abs residual: {np.mean(np.abs(z1_real - z1_fit_check)):.2f}"
              f" (signal scale {np.mean(np.abs(z1_real)):.2f})")

    # per-class per-timestep noise covariance
    shots0 = X_all_0[:, 0::2] + 1j * X_all_0[:, 1::2]
    shots1 = X_all_1[:, 0::2] + 1j * X_all_1[:, 1::2]
    noise0 = shots0 - z0_real
    noise1 = shots1 - z1_real

    def per_timestep_cov(noise):
        cov_t = np.empty((n, 2, 2))
        for t in range(n):
            iq = np.column_stack([noise[:, t].real, noise[:, t].imag])
            cov_t[t] = np.cov(iq, rowvar=False)
        return cov_t

    cov_t0 = per_timestep_cov(noise0)
    cov_t1 = per_timestep_cov(noise1)

    # Small diagonal jitter (negligible vs. real noise magnitude) so
    # near-singular timesteps don't break the Cholesky factorization.
    def safe_cholesky(cov_t):
        L = np.empty_like(cov_t)
        for t in range(cov_t.shape[0]):
            c = cov_t[t] + np.eye(2) * 1e-6 * max(np.trace(cov_t[t]), 1e-6)
            try:
                L[t] = np.linalg.cholesky(c)
            except np.linalg.LinAlgError:
                # fall back to an eigen-decomposition square root if the
                # jittered matrix is still not positive-definite
                w, v = np.linalg.eigh(c)
                w = np.clip(w, 0, None)
                L[t] = v @ np.diag(np.sqrt(w))
        return L

    L0 = safe_cholesky(cov_t0)
    L1 = safe_cholesky(cov_t1)

    if verbose:
        print("Mean per-timestep noise std (class 0):", np.sqrt(np.mean(np.diagonal(cov_t0, axis1=1, axis2=2))))
        print("Mean per-timestep noise std (class 1):", np.sqrt(np.mean(np.diagonal(cov_t1, axis1=1, axis2=2))))

    # ---- measured noise model: per-class PSD + std profile ----
    # The PSD carries both the colouring and the per-shot common-mode offset (a DC component); see the module
    # docstring for the numerical check.
    def measure_psd(noise):
        R = np.fft.fft(noise, axis=1)
        psd = (np.abs(R) ** 2).mean(axis=0)
        return psd / psd.mean()          # normalised to unit average power

    def measure_std_profile(noise):
        s = np.std(noise, axis=0)        # per-timestep complex std
        return s / s.mean(), float(np.std(noise))

    psd0, psd1 = measure_psd(noise0), measure_psd(noise1)
    prof0, sig0 = measure_std_profile(noise0)
    prof1, sig1 = measure_std_profile(noise1)

    # ---- full real-valued covariance in the interleaved [I, Q] space ----
    # Ceilings reached against the real-data target d = 3.84 sigma: white per-timestep 9.12, stationary PSD 5.65,
    # PSD + top-8 eigenvectors 4.64, full complex covariance E[r r^H] 4.76. The complex covariance models only the
    # circular part, while the matched filter's real-part inner product also depends on the pseudo-covariance
    # E[r r^T]; the real 2n-dimensional covariance captures every second-order structure.
    def real_cov_chol(noise_iq, sub=40000):
        r = noise_iq[:sub] if noise_iq.shape[0] > sub else noise_iq
        C = np.cov(r, rowvar=False)
        C = 0.5 * (C + C.T)
        C += np.eye(C.shape[0]) * 1e-9 * np.trace(C) / C.shape[0]
        try:
            return np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            w, V = np.linalg.eigh(C)
            return V @ np.diag(np.sqrt(np.clip(w, 0, None)))

    mean_iq0 = np.mean(X_all_0, axis=0)
    mean_iq1 = np.mean(X_all_1, axis=0)
    Lr0 = real_cov_chol(np.asarray(X_all_0[:40000]) - mean_iq0)
    Lr1 = real_cov_chol(np.asarray(X_all_1[:40000]) - mean_iq1)
    if verbose:
        print(f"  real-valued covariance Cholesky: {Lr0.shape} per class")

    return {
        "params0": params0,
        "params1": params1,
        "offset_I": offset_I,
        "offset_Q": offset_Q,
        "z0_real": z0_real,
        "z1_real": z1_real,
        "cov_t0": cov_t0,
        "cov_t1": cov_t1,
        "L0": L0,
        "L1": L1,
        # measured-noise-model fields
        "psd0": psd0, "psd1": psd1,
        "prof0": prof0, "prof1": prof1,
        "sigma0": sig0, "sigma1": sig1,
        # empirical class means -- used INSTEAD of the fitted cavity response
        # in noise_mode="measured". The parametric fit overshoots the true
        # class-mean difference by ~13% (|m1-m0| 7922 fitted vs 6992 real),
        # which alone inflates d by that factor. For a test set, fidelity to
        # the measured trajectory beats parametric tidiness.
        "mean0": z0_real, "mean1": z1_real,
        # full real-valued [I,Q] covariance factors (measured noise model)
        "Lr0": Lr0, "Lr1": Lr1,
        "mean_iq0": mean_iq0, "mean_iq1": mean_iq1,
        "n": n,
    }


def matched_filter_ceiling(X, y, n=770, seed=0):
    """
    ACCEPTANCE TEST. Optimal single-shot fidelity achievable on a dataset,
    via a full-trace matched filter w = m1 - m0 fit on one half and scored
    on the other (so the filter is never fit on the shots it scores).

    Returns (d_sigma, fidelity). On the REAL 0528 data this gives
    d = 3.885 sigma / 95.65%. A synthetic generator should reproduce that;
    a materially higher ceiling means the synthetic data is easier than
    reality, which is exactly the defect x_50k and x_50k_heldout have.
    """
    z = X[:, 0::2] + 1j * X[:, 1::2]
    y = np.asarray(y).astype(int)
    rng = np.random.default_rng(seed)
    i0 = np.flatnonzero(y == 0)
    i1 = np.flatnonzero(y == 1)
    rng.shuffle(i0)
    rng.shuffle(i1)
    h0, h1 = len(i0) // 2, len(i1) // 2
    m0 = z[i0[:h0]].mean(axis=0)
    m1 = z[i1[:h1]].mean(axis=0)
    w = m1 - m0
    s0 = (z[i0[h0:]] * np.conj(w)).real.sum(axis=1)
    s1 = (z[i1[h1:]] * np.conj(w)).real.sum(axis=1)
    sd = np.sqrt(0.5 * (s0.var() + s1.var()))
    d = abs(s1.mean() - s0.mean()) / sd
    thr = 0.5 * (s0.mean() + s1.mean())
    if s1.mean() > s0.mean():
        fid = 0.5 * (np.mean(s0 < thr) + np.mean(s1 > thr))
    else:
        fid = 0.5 * (np.mean(s0 > thr) + np.mean(s1 < thr))
    return float(d), float(fid)


def _coloured_noise(psd, profile, sigma, n, rng):
    """
    Complex noise with the MEASURED power spectrum. Shaping white complex
    Gaussian by sqrt(PSD) reproduces the measured autocorrelation AND, via
    the elevated DC bin, the per-shot common-mode offset that does not
    average down under integration. `profile` re-imposes the (mild, ~3%)
    per-timestep amplitude variation; `sigma` sets the overall scale.
    """
    w = rng.normal(size=n) + 1j * rng.normal(size=n)
    w /= np.sqrt(2.0)                       # unit total complex variance
    shaped = np.fft.ifft(np.fft.fft(w) * np.sqrt(psd))
    shaped = shaped / np.std(shaped)        # renormalise; shaping changes scale
    return shaped * sigma * profile


def generate_shot(state, model, add_noise, n=770, noise_mode="measured", rng=None):
    """
    noise_mode:
      "measured" (default) -- coloured noise matching the real 0528 PSD,
          which also reproduces the per-shot common-mode offset.
      "white"    -- independent noise per timestep (per-timestep Cholesky
          factor); generates x_50k_noisy and x_50k_heldout.
    """
    params = model["params0"] if state == 0 else model["params1"]
    if rng is None:
        rng = np.random.default_rng()
    z, phi, shot_phase = cavity_response_shot(
        params, state, model["offset_I"], model["offset_Q"], n=n, rng=rng)

    if add_noise:
        if noise_mode == "measured":
            # Deterministic part = the empirical class mean (the parametric cavity fit overshoots |m1-m0| by 13%),
            # rotated by this shot's phase, plus a draw from the full real-valued [I, Q] covariance (additive).
            miq = model["mean_iq0"] if state == 0 else model["mean_iq1"]
            L = model["Lr0"] if state == 0 else model["Lr1"]
            m = miq[0::2] + 1j * miq[1::2]
            noise = L @ rng.normal(size=L.shape[0])
            z = m * np.exp(1j * shot_phase) + (noise[0::2] + 1j * noise[1::2])
        elif noise_mode == "white":
            L = model["L0"] if state == 0 else model["L1"]  # (n, 2, 2)
            raw = np.random.normal(size=(n, 2))
            noise = np.einsum("tij,tj->ti", L, raw)
            z = z + (noise[:, 0] + 1j * noise[:, 1])
        else:
            raise ValueError("noise_mode must be 'measured' or 'white'")

    iq = np.empty(2 * len(z))
    iq[0::2] = z.real
    iq[1::2] = z.imag
    return iq, phi


def generate_dataset(model, n_shots, add_noise, n=770, seed=None,
                     noise_mode="measured", verbose=True):
    """Returns (X, y, Phi) matching generate_data_4.ipynb's x_50k/y_50k/Phi shapes."""
    if seed is not None:
        np.random.seed(seed)
    rng = np.random.default_rng(seed)

    X = np.empty((n_shots, 2 * n))
    y = np.empty(n_shots, dtype=int)
    Phi = np.empty((n_shots, n))

    for i in range(n_shots):
        state = np.random.randint(0, 2)
        iq, phi = generate_shot(state, model, add_noise, n=n,
                               noise_mode=noise_mode, rng=rng)
        X[i] = iq
        y[i] = state
        Phi[i] = phi
        if verbose and n_shots >= 10000 and i % 10000 == 0 and i:
            print(f"    {i}/{n_shots} shots")

    if verbose:
        d, fid = matched_filter_ceiling(X, y, n=n)
        print(f"  ACCEPTANCE TEST: matched-filter ceiling d = {d:.3f} sigma, "
              f"fidelity {fid*100:.2f}%")
        print(f"    real 0528 data target:            d = 3.885 sigma, "
              f"fidelity 95.65%")
        if fid > 0.985:
            print("    *** WARNING: generated data is EASIER than the real data. ***")

    return X, y, Phi
