"""Synthetic-vs-real training study: arm table and paired differences with CIs.

Inputs: results/*.json + *_probs.npz written by r9_train_eval.py.
Scored sets: 19,400 rows of loopback20k_real (rows 1000:1600 excluded; headline) and 20k (footnote).

Per group: mean balanced fidelity over replicate runs; sd over runs.
Replicate unit: a seed for single-draw groups; the 5-seed mean of a data draw / generator pool for
multi-draw groups (draw-to-draw variance is then included; df = n_draws - 1).

Difference A - B (paired: every run is scored on the same test shots):
  SE_train : sqrt(var_unitsA/nA + var_unitsB/nB)    (training randomness, test set fixed)
  SE_test  : sd of D under a class-stratified bootstrap of the test rows, all runs fixed
             (2,000 resamples, seed 20261020; paired: the same resampled rows for both arms)
  95% CI   : D +/- sqrt((t_{df,0.975} SE_train)^2 + (1.96 SE_test)^2), df Welch-Satterthwaite
  p_train  : Welch t-test on replicate units (conditional on the test set)
  McNemar  : exact two-sided, on the val_loss-selected run of each group (deployment protocol)
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
yev = np.load(REPO / "data/loopback20k_real_y.npy").astype(int)
keep = np.ones(len(yev), bool)
keep[1000:1600] = False

runs = {}
for jf in sorted((__import__("pathlib").Path(__import__("os").environ.get("R9_RESULTS", str(HERE / "results")))).glob("*.json")):
    if jf.stem.startswith("check"):
        continue
    P = np.load(str(jf)[:-5] + "_probs.npz")
    for r in json.loads(jf.read_text()):
        r["correct"] = ((P[f"{r['arm']}|{r['seed']}"] > 0.5).astype(int) == yev)
        runs[(r["arm"], r["seed"])] = r

S5 = list(range(20260901, 20260906))
S10 = list(range(20260901, 20260911))


def sel(arm, seeds):
    return [(arm, s) for s in seeds if (arm, s) in runs]


# group -> list of draws, each draw a list of run keys
GROUPS = {
    "real10k_d0": dict(label="real 10k (rows 0:10k)", fit="-", draws=[sel("real10k_d0", S10)]),
    "v2_10k_d0": dict(label="synthetic 10k (x_synth_v2[0:10k])", fit="100k",
                      draws=[sel("v2_10k_d0", S10)]),
    "real10k_d0_s5": dict(label="real 10k (rows 0:10k), first 5 seeds", fit="-",
                          draws=[sel("real10k_d0", S5)]),
    "v2_10k_d0_s5": dict(label="synthetic 10k (x_synth_v2[0:10k]), first 5 seeds", fit="100k",
                         draws=[sel("v2_10k_d0", S5)]),
    "real50k_s5": dict(label="real 50k, first 5 seeds", fit="-", draws=[sel("real50k", S5)]),
    "v2_50k_s5": dict(label="synthetic 50k, first 5 seeds", fit="100k", draws=[sel("v2_50k", S5)]),
    "real10k_3d": dict(label="real 10k, 3 row blocks", fit="-",
                       draws=[sel("real10k_d0", S5), sel("real10k_d1", S5), sel("real10k_d2", S5)]),
    "v2_10k_3d": dict(label="synthetic 10k, 3 slices of x_synth_v2", fit="100k",
                      draws=[sel("v2_10k_d0", S5), sel("v2_10k_d1", S5), sel("v2_10k_d2", S5)]),
    "fit30k": dict(label="synthetic 10k, generator fit on 30k real (3 pools)", fit="30k",
                   draws=[sel(f"fit30k_p{i}", S5) for i in range(3)]),
    "fit10k": dict(label="synthetic 10k, generator fit on 10k real (3 pools)", fit="10k",
                   draws=[sel(f"fit10k_p{i}", S5) for i in range(3)]),
    "own10k_synth10k": dict(label="synthetic 10k, generator fit on the real-10k rows", fit="10k (own)",
                            draws=[sel("own10k_synth10k", S10)]),
    "real50k": dict(label="real 50k (rows 0:50k)", fit="-", draws=[sel("real50k", S10)]),
    "v2_50k": dict(label="synthetic 50k (all x_synth_v2)", fit="100k", draws=[sel("v2_50k", S10)]),
    "own10k_synth50k": dict(label="synthetic 50k, generator fit on the real-10k rows", fit="10k (own)",
                            draws=[sel("own10k_synth50k", S10)]),
    "real10k_plus_own50k": dict(label="real 10k + 50k synthetic from those rows", fit="10k (own)",
                                draws=[sel("real10k_plus_own50k", S10)]),
}
COMPARISONS = [
    ("v2_10k_d0_s5", "real10k_d0_s5"),
    ("v2_50k_s5", "real50k_s5"),
    ("v2_10k_d0", "real10k_d0"),
    ("v2_10k_3d", "real10k_3d"),
    ("fit30k", "real10k_3d"),
    ("fit10k", "real10k_3d"),
    ("fit30k", "v2_10k_3d"),
    ("fit10k", "v2_10k_3d"),
    ("own10k_synth10k", "real10k_d0"),
    ("v2_50k", "real50k"),
    ("own10k_synth50k", "real10k_d0"),
    ("real10k_plus_own50k", "real10k_d0"),
    ("own10k_synth50k", "own10k_synth10k"),
    ("real50k", "real10k_d0"),
    ("own10k_synth50k", "real50k"),
    ("own10k_synth50k", "v2_50k"),
    ("real10k_plus_own50k", "own10k_synth50k"),
]

rng = np.random.default_rng(20261020)
B = 2000


def make_weights(mask):
    """class-stratified bootstrap weights over the rows in mask: (n_mask, B) per class."""
    idx = np.flatnonzero(mask)
    W = {}
    for c in (0, 1):
        ic = np.flatnonzero(yev[idx] == c)
        Wc = np.zeros((len(idx), B), np.float32)
        for b in range(B):
            Wc[ic, b] = np.bincount(rng.integers(0, len(ic), len(ic)), minlength=len(ic))
        W[c] = Wc
    return idx, W


def run_F(keys, idx):
    C = np.stack([runs[k]["correct"][idx] for k in keys]).astype(np.float64)
    y = yev[idx]
    return 0.5 * (C[:, y == 0].mean(1) + C[:, y == 1].mean(1))


def boot_F(keys, idx, W):
    """(n_runs, B) bootstrap balanced fidelities."""
    C = np.stack([runs[k]["correct"][idx] for k in keys]).astype(np.float32)
    acc = [(C @ W[c]) / W[c].sum(0) for c in (0, 1)]
    return 0.5 * (acc[0] + acc[1])


def group_units(g, idx):
    """replicate-unit values (seed values, or per-draw means) and the full run list."""
    draws = [d for d in GROUPS[g]["draws"] if d]
    if len(draws) == 1:
        return run_F(draws[0], idx), draws
    return np.array([run_F(d, idx).mean() for d in draws]), draws


def group_mean_boot(draws, idx, W):
    # equal weight per draw (each draw has the same number of seeds anyway)
    return np.mean([boot_F(d, idx, W).mean(0) for d in draws], axis=0)


def mcnemar_exact(ca, cb):
    b = int(np.sum(ca & ~cb)); c = int(np.sum(~ca & cb))
    p = min(1.0, 2 * stats.binom.cdf(min(b, c), b + c, 0.5)) if b + c else 1.0
    return b, c, p


def best_run(draws):
    return min((k for d in draws for k in d), key=lambda k: runs[k]["val_loss"])


out = {}
for setname, mask in (("19400", keep), ("20k", np.ones(len(yev), bool))):
    idx, W = make_weights(mask)
    table, comps = {}, []
    boot_cache = {}
    for g, G in GROUPS.items():
        units, draws = group_units(g, idx)
        if not draws:
            continue
        allF = run_F([k for d in draws for k in d], idx)
        n_u = len(units)
        hw = stats.t.ppf(0.975, n_u - 1) * units.std(ddof=1) / np.sqrt(n_u) if n_u > 1 else np.nan
        bm = group_mean_boot(draws, idx, W)
        boot_cache[g] = (units, draws, bm)
        se_test = bm.std(ddof=1)
        tot = np.sqrt(hw ** 2 + (1.96 * se_test) ** 2)
        bk = best_run(draws)
        Fb = run_F([bk], idx)[0]
        table[g] = dict(label=G["label"], fit_pool=G["fit"], n_train=runs[draws[0][0]]["n_train"],
                        n_runs=int(len(allF)), n_units=int(n_u), mean=100 * allF.mean(),
                        sd_runs=100 * allF.std(ddof=1), unit_values=(100 * units).tolist(),
                        ci95=[100 * (allF.mean() - tot), 100 * (allF.mean() + tot)],
                        ci95_train_only_halfwidth=100 * hw, se_test=100 * se_test,
                        val_selected=dict(arm=bk[0], seed=bk[1], F=100 * Fb))
    for a, b in COMPARISONS:
        if a not in boot_cache or b not in boot_cache:
            continue
        ua, da, bma = boot_cache[a]
        ub, db, bmb = boot_cache[b]
        D = ua.mean() - ub.mean()
        va, vb = ua.var(ddof=1) / len(ua), ub.var(ddof=1) / len(ub)
        se_tr = np.sqrt(va + vb)
        df = (va + vb) ** 2 / (va ** 2 / (len(ua) - 1) + vb ** 2 / (len(ub) - 1))
        se_te = (bma - bmb).std(ddof=1)
        hw = np.sqrt((stats.t.ppf(0.975, df) * se_tr) ** 2 + (1.96 * se_te) ** 2)
        p_tr = stats.ttest_ind(ua, ub, equal_var=False).pvalue
        ka, kb = best_run(da), best_run(db)
        mb, mc, mp = mcnemar_exact(runs[ka]["correct"][idx], runs[kb]["correct"][idx])
        comps.append(dict(A=a, B=b, diff=100 * D, ci95=[100 * (D - hw), 100 * (D + hw)],
                          se_train=100 * se_tr, df_train=float(df), se_test=100 * se_te,
                          p_welch_train=float(p_tr),
                          mcnemar_valsel=dict(A=list(ka), B=list(kb),
                                              F_A=100 * run_F([ka], idx)[0], F_B=100 * run_F([kb], idx)[0],
                                              A_only_correct=mb, B_only_correct=mc, p_exact=mp)))
    out[setname] = dict(n_rows=int(len(idx)), groups=table, comparisons=comps)

(__import__("pathlib").Path(__import__("os").environ.get("R9_STATS", str(HERE / "r9_stats.json")))).write_text(json.dumps(out, indent=1))
for setname in out:
    print(f"\n=== scored set {setname} (n = {out[setname]['n_rows']})")
    print(f"{'group':22s} {'fit':>9s} {'n_tr':>6s} {'runs':>4s} {'mean':>7s} {'sd':>5s} {'95% CI':>17s}  val-sel")
    for g, t in out[setname]["groups"].items():
        print(f"{g:22s} {t['fit_pool']:>9s} {t['n_train']:6d} {t['n_runs']:4d} {t['mean']:7.3f} "
              f"{t['sd_runs']:5.3f} [{t['ci95'][0]:.3f}, {t['ci95'][1]:.3f}]  {t['val_selected']['F']:.3f}")
    print(f"\n{'A - B':42s} {'diff':>7s} {'95% CI':>18s} {'SEtr':>6s} {'SEte':>6s} {'p_tr':>7s}  McNemar(val-sel) b/c p")
    for c in out[setname]["comparisons"]:
        m = c["mcnemar_valsel"]
        print(f"{c['A'] + ' - ' + c['B']:42s} {c['diff']:+7.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}] "
              f"{c['se_train']:6.3f} {c['se_test']:6.3f} {c['p_welch_train']:7.4f}  "
              f"{m['A_only_correct']}/{m['B_only_correct']} p={m['p_exact']:.3g}")
