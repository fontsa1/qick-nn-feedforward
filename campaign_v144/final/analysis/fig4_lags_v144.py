"""Fig. 4 data: the replay apparatus. Run once; writes fig4_data.npz and fig4_values.json beside this file.
(a) fidelity of a FIXED linear discriminator vs a rotation of the replayed traces (2-fold on the 19,400 headline
    rows of loopback20k_real: fit on one half, score the other half rotated by theta, then swap), plus a gain sweep;
(b) per-shot probe phase (= arg k) on 4 bitstream loads, no phrst vs phrst on generator and readout
    (campaign_v140L1/A_L2_phrst_probe.json, task 4.4);
(c) arg k over 3 h on both boards, relative to the first sample (campaign_v140L1/drift_<host>.json, task 4.3);
(d) replay channel |H(f)|: per-FFT-bin LTI fit of the replayed traces against their lag-aligned, domain-mapped
    sources (the fit in campaign_v140L1/arch_aug.py, same 4,000 shots);
(e) raw vs replayed fidelity per model, same 20k shots (campaign_v140L1/arch_sweep2_src.json)."""
import json
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent / "fig4"
ROOT = HERE.parents[3]
CAMP = ROOT / "campaign_v140L1"
EXCL = slice(1000, 1600)           # domain-map rows, excluded from every headline


def cplx(X):
    return X[:, 0::2] + 1j * X[:, 1::2]


def fidelity(y, p):
    return 100 * (1 - 0.5 * (np.mean(p[y == 1] == 0) + np.mean(p[y == 0] == 1)))


# ---------------- (a) rotation and gain sweeps ----------------
X = np.load(ROOT / "campaign_v144/final/analysis/replay20k_A.npy").astype(np.float64)   # v1.44, board A, NN_axi session
y = np.load(ROOT / "data/test20k_real_y.npy").astype(int)
keep = np.ones(len(y), bool); keep[EXCL] = False
Z, y = cplx(X[keep]), y[keep]
del X
n = len(y)
fold = np.arange(n) % 2


RIDGES = (1e-2, 1e-1, 1.0, 10.0, 100.0)


def lda_w(F, yy, ridge):
    m0, m1 = F[yy == 0].mean(0), F[yy == 1].mean(0)
    S = np.cov(np.vstack([F[yy == 0] - m0, F[yy == 1] - m1]).T)
    S += ridge * np.trace(S) / len(S) * np.eye(len(S))
    return np.linalg.solve(S, m1 - m0)


def best_thr(s, yy):
    cand = np.quantile(s, np.linspace(0.01, 0.99, 981))
    return cand[np.argmax([fidelity(yy, (s >= c).astype(int)) for c in cand])]


def fit_lda(z, yy):
    """Ridge chosen by an inner 80/20 split of the fitting half only; then refit on the whole half."""
    F = np.hstack([z.real, z.imag])
    cut = int(0.8 * len(yy))
    val = []
    for r in RIDGES:
        w = lda_w(F[:cut], yy[:cut], r)
        t = best_thr(F[:cut] @ w, yy[:cut])
        val.append(fidelity(yy[cut:], ((F[cut:] @ w) >= t).astype(int)))
    r = RIDGES[int(np.argmax(val))]
    w = lda_w(F, yy, r)
    return w, best_thr(F @ w, yy), r


def score(z, yy, w, t):
    return fidelity(yy, ((np.hstack([z.real, z.imag]) @ w) >= t).astype(int))


fits = [fit_lda(Z[fold == k], y[fold == k]) for k in (0, 1)]
theta = np.arange(-180, 181, 1)
rot = np.array([np.mean([score(Z[fold == 1 - k] * np.exp(1j * np.radians(th)), y[fold == 1 - k], *fits[k][:2])
                         for k in (0, 1)]) for th in theta])
ridges = [f[2] for f in fits]
print(f"(a) ridges {ridges}; 0 deg {rot[theta == 0][0]:.2f}%, 20 deg {rot[theta == 20][0]:.2f}, "
      f"-99 deg {rot[theta == -99][0]:.2f}, -119 deg {rot[theta == -119][0]:.2f}")
del Z

# ---------------- (b) phrst probe ----------------
P = json.loads((ROOT / "campaign_v144/final/board2/A_phrst_probe_v144.json").read_text())
phrst = {m: [l["modes"][m]["shots"] for l in P["loads"]] for m in ("none", "both")}

# ---------------- (c) 3 h drift ----------------
drift = {}
for host in ("10.1.9.26", "10.1.4.105"):
    D = json.loads((ROOT / f"campaign_v144/final/drift_full/drift_{host}.json").read_text())
    t = np.array([s["t"] for s in D["samples"]]); ph = np.array([s["phase"] for s in D["samples"]])
    ph = np.degrees(np.unwrap(np.radians(ph)))
    drift[D["board"]] = dict(t_min=(t - t[0]) / 60, dphase=ph - ph[0],
                             sem=np.array([s["phase_std"] for s in D["samples"]]) / np.sqrt(D["args"]["shots"]))

# ---------------- (d) replay channel |H(f)| (arch_aug.py fit, N_FIT = 4000) ----------------
N_FIT, LAGS = 4000, range(-8, 9)
k2 = complex(np.load(ROOT / "calibration/k2_loopback.npy")); c2 = complex(np.load(ROOT / "calibration/c2_loopback.npy"))
zs = k2 * cplx(np.load(ROOT / "data/test20k_real_X.npy", mmap_mode="r")[:N_FIT].astype(np.float64)) + c2
zl = cplx(np.load(ROOT / "campaign_v144/final/analysis/replay20k_A.npy", mmap_mode="r")[:N_FIT].astype(np.float64))
ds, dl = zs - zs.mean(0), zl - zl.mean(0)
C = np.array([[np.abs(np.vdot(np.roll(ds[i], l)[10:-10], dl[i][10:-10])) for l in LAGS] for i in range(N_FIT)])
lag = np.array(LAGS)[C.argmax(1)]


def shift(z, d):
    if d == 0:
        return z
    out = np.empty_like(z); out[d:] = z[:-d]; out[:d] = z[:1]
    return out


al = np.stack([shift(zs[i], int(lag[i])) for i in range(N_FIT)])
S, L = np.fft.fft(al, axis=1), np.fft.fft(zl, axis=1)
H = (np.conj(S) * L).sum(0) / (np.abs(S) ** 2).sum(0)
f = np.fft.fftfreq(len(H), 1 / 552.96)
order = np.argsort(f)
fpos = np.abs(f)
bands = {f"{lo}-{hi}": float(np.abs(H[(fpos >= lo) & (fpos < hi)]).mean())
         for lo, hi in ((0, 5), (5, 20), (20, 60), (60, 150), (150, 277))}
print("(d) |H| bands:", {k: round(v, 2) for k, v in bands.items()},
      "lags:", dict(sorted(Counter(lag.tolist()).items())))

# ---------------- (e) raw vs replayed, per model ----------------
A2 = json.loads((ROOT / "campaign_v144/final/analysis/arch/arch_sweep2_src_v144.json").read_text())
NAMES = {"lda_ref": "LDA", "cnn1d": "1-D CNN", "seg_mlp": "segment MLP", "bin_mlp": "binned MLP",
         "seg_hgb": "boosted trees", "bin_gru": "binned GRU"}
models_e = [dict(key=k, name=NAMES[k], raw=A2[k]["src20k"], replayed=A2[k]["loop20k"]) for k in NAMES]

json.dump(dict(lags={str(k_): v_ for k_, v_ in sorted(Counter(lag.tolist()).items())}), open(ROOT / "campaign_v144/final/analysis/fig2_lags_v144.json", "w"))
