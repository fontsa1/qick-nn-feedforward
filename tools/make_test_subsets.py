"""
Build shuffled, class-balanced test subsets for the loopback runs.

0528_X_test_0_770.npy is block-ordered in runs of 5000, so firing it in file order would confound drift during
the acquisition with the class label. Shuffling with a recorded seed turns that confound into noise.

Writes, for each subset:  <tag>_X.npy, <tag>_y.npy, <tag>_manifest.json
The manifest records the seed and the source row indices, so any result can be
traced back to exact rows in the source file.

"""
import argparse, json, hashlib
import numpy as np
from pathlib import Path

BASE = Path(".")

def build(tag, xsrc, ysrc, n, seed):
    X = np.load(BASE / xsrc, mmap_mode="r")
    y = np.load(BASE / ysrc).astype(int)
    rng = np.random.default_rng(seed)

    # stratified: exactly n/2 per class, then shuffle so class is decorrelated
    # from firing order
    idx = []
    for s in (0, 1):
        pool = np.where(y == s)[0]
        idx.append(rng.choice(pool, size=n // 2, replace=False))
    idx = np.concatenate(idx)
    rng.shuffle(idx)
    idx = np.sort(idx) if False else idx          # keep shuffled order

    Xs = np.asarray(X[idx]).astype(np.float64)
    ys = y[idx]

    np.save(BASE / f"{tag}_X.npy", Xs)
    np.save(BASE / f"{tag}_y.npy", ys)

    # order-confound check: correlation between firing position and label
    pos = np.arange(len(ys))
    r = float(np.corrcoef(pos, ys)[0, 1])
    runs = int(1 + np.sum(np.diff(ys) != 0))
    man = dict(tag=tag, source_X=xsrc, source_y=ysrc, n=int(n), seed=int(seed),
               balance={int(k): int(v) for k, v in zip(*np.unique(ys, return_counts=True))},
               order_label_corr=r, label_runs=runs,
               source_indices_sha256=hashlib.sha256(idx.tobytes()).hexdigest(),
               source_indices=idx.tolist())
    json.dump(man, open(BASE / f"{tag}_manifest.json", "w"), indent=1)

    print(f"  {tag}: {Xs.shape} balance {man['balance']}")
    print(f"     order/label corr = {r:+.4f} (want ~0)   label runs = {runs} of {n} "
          f"(file order would give ~{n//5000})")
    return Xs, ys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260902)
    ap.add_argument("--which", default="both", choices=["real", "synth", "both"])
    a = ap.parse_args()

    print("=== building shuffled, balanced test subsets (n=%d, seed=%d) ===" % (a.n, a.seed))
    if a.which in ("real", "both"):
        build("data/test20k_real", "0528_X_test_0_770.npy", "0528_y_test_0_770.npy", a.n, a.seed)
    if a.which in ("synth", "both"):
        build("data/test20k_synth", "x_synth_v2.npy", "y_synth_v2.npy",
              a.n, a.seed + 1)

if __name__ == "__main__":
    main()
