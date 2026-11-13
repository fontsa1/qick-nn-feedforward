"""Statistics helpers: balanced fidelity, CIs, exact McNemar."""
from math import erf, sqrt

import numpy as np
from scipy.stats import binomtest

CAL = (1000, 1600)  # domain-map calibration rows (FIR k2/c2 and boxcar kb/s/cb)
Z = 1.959964


def wilson(k, n):
    p, d = k / n, 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    hw = Z * np.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return [100 * (c - hw), 100 * (c + hw)]


def fstats(pred, y):
    pred = np.asarray(pred).astype(int); y = np.asarray(y).astype(int)
    g, e = y == 0, y == 1
    n0, n1 = int(g.sum()), int(e.sum())
    p10 = float(np.mean(pred[g] != 0)); p01 = float(np.mean(pred[e] != 1))
    F = 1 - 0.5 * (p10 + p01)
    se = 0.5 * sqrt(p10 * (1 - p10) / n0 + p01 * (1 - p01) / n1)
    ok = int((pred == y).sum())
    return dict(n=int(len(y)), n0=n0, n1=n1, fid=100 * F, fid_ci=[100 * (F - Z * se), 100 * (F + Z * se)],
                acc=100 * ok / len(y), acc_wilson=wilson(ok, len(y)), p10=100 * p10, p01=100 * p01)


def paired(predA, predB, y):
    """A vs B on the same shots. b = A-only-correct, c = B-only-correct."""
    y = np.asarray(y).astype(int)
    okA, okB = np.asarray(predA).astype(int) == y, np.asarray(predB).astype(int) == y
    b = int((okA & ~okB).sum()); c = int((okB & ~okA).sum())
    p_exact = float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0
    z = (b - c) / sqrt(b + c) if b + c else 0.0
    p_norm = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
    # balanced-fidelity difference and its paired variance (per class)
    d, var = 0.0, 0.0
    for cl in (0, 1):
        m = y == cl; n = int(m.sum())
        dA = (okA[m].astype(int) - okB[m].astype(int))
        d += 0.5 * dA.mean(); var += 0.25 * dA.var() / n
    se = sqrt(var)
    return dict(n=int(len(y)), A_only=b, B_only=c, p_exact=p_exact, z=z, p_normal=p_norm,
                dF=100 * d, dF_ci=[100 * (d - Z * se), 100 * (d + Z * se)])


def calmask(rows):
    rows = np.asarray(rows)
    return ~((rows >= CAL[0]) & (rows < CAL[1]))


