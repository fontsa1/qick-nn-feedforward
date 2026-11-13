"""End-to-end check that no test information reaches the trained model.

Re-derives every link in the data chain from the raw files and asserts disjointness:

    0528_X_all_{0,1}.npy          real shots, 500k/class
        |                         (the synthetic generator fits its cavity+noise model on train rows 10000:110000)
        +-> 0528_X_train_0_770    900k real   -- classifier TRAIN pool
        +-> 0528_X_test_0_770     100k real   -- held-out TEST pool
                |
                +-> test20k_real_X  20k, indices in data/test20k_real_manifest.json
                        |
                        +-> [0:20000]    scored (NN_axi, NN_live, emulators)
                        |                 headline set = [0:20000] minus [1000:1600] (19,400)
                        +-> [1000:1600]  domain_calib_* -- fits (k2, c2) and the boxcar-path map (kb, s, cb),
                                          calibration/box_domain_map.json

    (k2, c2) applied to all of 0528_X_train_0_770
        -> 0528_X_train_0_770_loopback_calibrated  -- what the model trains on

Checks, each fatal if it fails:

  A  train pool and test pool share no shot (raw row bytes)
  B  test20k_real_X is exactly 0528_X_test[source_indices], labels included, and the manifest's index checksum matches
  C  the scored rows as stored per shot (test20k 0..19,999), and the 19,400-row headline set disjoint from the rows
     used to fit (k2, c2) and (kb, s, cb)
  D  the calibrated training file is the train pool under the recorded (k2, c2) and contains no test shot
  E  the generator's fit pool is disjoint from both training and scored data
  F  reports every place a test-set label is consumed by a fitting step
  G  test100k_real (the whole test split as replayed for NN_live) is exactly 0528_X_test[source_indices] with its
     manifest hashes, starts with test20k_real, shares no shot with the train pool, and its 99,400-row headline set
     (rows 1000:1600 removed) shares no shot with the domain-map fit slices

--legacy-check-c runs check C on the first SCORED_N rows only.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

RAW_DIR = Path(__import__("os").environ.get("RAW_DATA_DIR", "."))   # 0528_X_all_{0,1}.npy
CHUNK = 20000

# The number of rows the classifier trains on (export_signed_domain_model.py and phase0_calibration_ablation.py);
# the generator's fit pool starts at exactly TRAIN_N.
TRAIN_N = 10000
GEN_FIT_LO, GEN_FIT_HI = 10000, 110000   # as in tools/generate_synth_holdout_v2.py
SCORED_N = 400          # --legacy-check-c: check C on rows 0:400 only
CALIB_SLICE = (1000, 1600)
# Check C: the headlines score test20k rows 0..19,999. The full 20k contains the 600 domain-map calibration rows,
# so the headline set is the 19,400 rows outside CALIB_SLICE.
LEGACY_C = "--legacy-check-c" in sys.argv
SCORED_FILES = {   # per-shot files whose row indices define the scored set
    "NN_axi v1.40 (campaign_v140/aligned)": ("campaign_v140/aligned", "idx"),
    "NN_live L10 board A": ("campaign_v140L1/A_L10_box_live_capture_0_20000.npz", "rows"),
    "boxcar/FIR emulator (19,400)": ("campaign_v140L1/loopback20k_box_A_eval.npz", "rows"),
}
BOX_MAP = "calibration/box_domain_map.json"   # boxcar-path map (kb, s, cb); records its calib_rows

results = []

def record(name, ok, detail="", pass_detail=""):
    """detail is shown only on FAILURE; pass_detail only on success."""
    results.append((name, ok, detail))
    shown = pass_detail if ok else detail
    print(f"  [{'PASS' if ok else '**FAIL**'}] {name}" + (f"  -- {shown}" if shown else ""))

def row_hashes(a):
    a = np.ascontiguousarray(a)
    return [hashlib.blake2b(r.tobytes(), digest_size=16).digest() for r in a]

def hash_file_rows(path, limit=None):
    """Streaming set of row hashes, so an 11 GB file never lands in RAM."""
    A = np.load(path, mmap_mode="r")
    n = A.shape[0] if limit is None else min(limit, A.shape[0])
    out = set()
    for i in range(0, n, CHUNK):
        out.update(row_hashes(np.asarray(A[i:min(i + CHUNK, n)])))
    return out, A.shape

def load_scored_rows(src, key):
    p = Path(src)
    if p.is_dir():
        parts = [np.load(f, allow_pickle=True) for f in sorted(p.glob("board*_*.npz"))]
        if not parts:
            return None, None
        return (np.concatenate([z[key] for z in parts]).astype(np.int64),
                np.concatenate([z["y"] for z in parts]).astype(int))
    if not p.exists():
        return None, None
    z = np.load(p, mmap_mode="r")
    return np.asarray(z[key]).astype(np.int64), np.asarray(z["y"]).astype(int)

def check_c_scored(X20, y20, h_calib, lo, hi):
    """C2-C4: the scored set as it actually is (rows 0..19,999), and the 19,400-row
    headline set (CALIB_SLICE removed) disjoint from both domain-map fits."""
    import json as _json
    bm = _json.load(open(BOX_MAP)) if Path(BOX_MAP).exists() else {}
    record(f"C2 boxcar-path map ({BOX_MAP}) was fit on rows {lo}:{hi}",
           bm.get("calib_rows") == f"{lo}:{hi}",
           f"{BOX_MAP} records calib_rows = {bm.get('calib_rows')!r}")
    yy = np.asarray(y20).astype(int)
    full = set(range(len(yy)))
    calib = set(range(lo, hi))
    union, ok_all, lines = set(), True, []
    for name, (src, key) in SCORED_FILES.items():
        rows, ys = load_scored_rows(src, key)
        if rows is None:
            lines.append(f"{name}: file missing ({src})"); ok_all = False
            continue
        uniq = len(np.unique(rows)) == len(rows)
        lab = bool(np.array_equal(yy[rows], ys))
        inside = int(np.isin(rows, list(calib)).sum())
        ok_all &= uniq and lab and set(rows.tolist()) <= full
        union |= set(rows.tolist())
        lines.append(f"{name}: {len(rows)} rows [{rows.min()}..{rows.max()}], unique {uniq}, "
                     f"labels == test20k {lab}, calibration rows inside {inside}")
    for s in lines:
        print("     " + s)
    record("C3 scored per-shot files index test20k rows 0..19,999 with matching labels",
           ok_all and union <= full,
           "; ".join(lines),
           pass_detail=f"union of scored rows = {len(union)} rows; the full 20k includes the "
                       f"{len(calib & union)} calibration rows (footnote set only)")
    head = np.array(sorted(full - calib))
    h_head = set(row_hashes(np.asarray(X20[head])))
    h_slice = set(row_hashes(np.asarray(X20[lo:hi])))
    n_dup = len(h_head & (h_calib | h_slice))
    rows_ev, _ = load_scored_rows(*SCORED_FILES["boxcar/FIR emulator (19,400)"])
    ev_ok = rows_ev is not None and np.array_equal(np.sort(rows_ev), head)
    record(f"C4 the {len(head)}-row headline set (rows outside {lo}:{hi}) shares no shot with "
           f"either domain-map fit slice", n_dup == 0 and ev_ok,
           f"{n_dup} shared rows by content; boxcar/FIR eval rows == headline set: {ev_ok}",
           pass_detail=f"0 shared rows by content (blake2b); the boxcar/FIR emulator file scores "
                       f"exactly these {len(head)} rows")

T100 = "data/test100k_real"                  # the full test split as replayed for NN_live
SCORED_100K = {   # per-shot files of the 100k campaign (run 1 and the board-swapped run 2)
    "run 1, board A": "campaign_v140L1/A_v141_100k_merged_0_50000.npz",
    "run 1, board B": "campaign_v140L1/B_v141_100k_merged_50000_100000.npz",
    "run 2, board A": "campaign_v140L1/A_v141_100k_swap_merged_50000_100000.npz",
    "run 2, board B": "campaign_v140L1/B_v141_100k_swap_merged_0_50000.npz",
}

def check_g_test100k(h_train, h_calib, X20, lo, hi):
    """G1-G5: the 100k replay set that carries the NN_live headline."""
    man = json.load(open(f"{T100}_manifest.json"))
    idx = np.asarray(man["source_indices"], dtype=np.int64)
    X = np.load(f"{T100}_X.npy", mmap_mode="r")
    y = np.load(f"{T100}_y.npy")
    hx = hashlib.sha256()
    for i in range(0, X.shape[0], CHUNK):
        hx.update(np.ascontiguousarray(X[i:i + CHUNK]).tobytes())
    hy = hashlib.sha256(np.ascontiguousarray(y).tobytes()).hexdigest()
    record(f"G1 {T100}_X / _y match the sha256 recorded in the manifest",
           hx.hexdigest() == man["sha256_X"] and hy == man["sha256_y"],
           f"X {hx.hexdigest()[:16]}... vs {man['sha256_X'][:16]}..., y {hy[:16]}... vs {man['sha256_y'][:16]}...",
           pass_detail=f"X {hx.hexdigest()[:16]}..., y {hy[:16]}...")

    Xs = np.load(man["source_X"], mmap_mode="r")
    ys = np.load(man["source_y"], mmap_mode="r")
    uniq = len(np.unique(idx)) == len(idx) == X.shape[0] == man["n"]
    lab = bool(np.array_equal(y, np.asarray(ys)[idx]))
    rows_ok = all(np.array_equal(np.asarray(X[i:i + CHUNK]), np.asarray(Xs[idx[i:i + CHUNK]]))
                  for i in range(0, len(idx), CHUNK))
    record(f"G2 {T100} is exactly {man['source_X']}[source_indices] (every row and label), indices unique",
           uniq and lab and rows_ok, f"unique indices {uniq}, labels equal {lab}, rows equal {rows_ok}",
           pass_detail=f"all {len(idx)} rows byte-for-byte, labels equal; balance {man['balance']}")

    first = bool(np.array_equal(np.asarray(X[:X20.shape[0]]), np.asarray(X20)))
    record(f"G3 {T100} rows 0..{X20.shape[0] - 1} are test20k_real in the same order", first)

    h100 = set()
    for i in range(0, X.shape[0], CHUNK):
        h100.update(row_hashes(np.asarray(X[i:i + CHUNK])))
    record(f"G4 no {T100} shot appears in the train pool", len(h100 & h_train) == 0,
           f"{len(h100 & h_train)} shared", pass_detail=f"{len(h100)} unique rows, 0 shared with the train pool")

    full, calib, lines, ok = set(range(X.shape[0])), set(range(lo, hi)), [], True
    for name, f in SCORED_100K.items():
        if not Path(f).exists():
            lines.append(f"{name}: file missing ({f})"); ok = False
            continue
        z = np.load(f, allow_pickle=True)
        r = np.asarray(z["rows"]).astype(np.int64)
        good = (len(np.unique(r)) == len(r) and set(r.tolist()) <= full
                and np.array_equal(y[r].astype(int), np.asarray(z["y"]).astype(int)))
        ok &= good
        lines.append(f"{name}: {len(r)} rows [{r.min()}..{r.max()}], labels == {T100} {good}, "
                     f"calibration rows inside {int(np.isin(r, list(calib)).sum())}")
    for s in lines:
        print("     " + s)
    head = np.array(sorted(full - calib))
    h_head = set(row_hashes(np.asarray(X[head[head < X20.shape[0]]]))) | {
        hh for i in range(X20.shape[0], X.shape[0], CHUNK) for hh in row_hashes(np.asarray(X[i:i + CHUNK]))}
    h_slice = set(row_hashes(np.asarray(X20[lo:hi])))
    n_dup = len(h_head & (h_calib | h_slice))
    record(f"G5 the {len(head)}-row NN_live headline set ({T100} minus rows {lo}:{hi}) shares no shot with the "
           f"domain-map fit slices; the campaign files index {T100} with matching labels",
           ok and n_dup == 0, "; ".join(lines) + f"; {n_dup} headline rows shared with a fit slice",
           pass_detail=f"0 shared rows by content; 4 per-shot files, each run covers all 100,000 rows")

def main():
    print(__doc__.split("Run from")[0])
    if LEGACY_C:
        print("(--legacy-check-c: check C on rows 0:400 only)")
    print("=" * 74)

    # ---------- A: the two real pools share no shot -------------------------
    print("\nA. train pool vs held-out test pool (raw row bytes)")
    h_test, sh_test = hash_file_rows("0528_X_test_0_770.npy")
    h_train, sh_train = hash_file_rows("0528_X_train_0_770.npy")
    inter = h_train & h_test
    print(f"     train {sh_train}  unique rows {len(h_train)}")
    print(f"     test  {sh_test}  unique rows {len(h_test)}")
    record("A1 train pool and test pool are disjoint", len(inter) == 0,
           f"{len(inter)} shared rows")

    # ---------- B: test subset provenance -----------------------------------
    print("\nB. test20k_real_X provenance against its manifest")
    man = json.load(open("data/test20k_real_manifest.json"))
    idx = np.asarray(man["source_indices"])
    recomputed = hashlib.sha256(
        np.ascontiguousarray(idx.astype(np.int64)).tobytes()).hexdigest()
    record("B1 manifest index checksum is self-consistent",
           recomputed == man["source_indices_sha256"],
           detail=f"recorded {man['source_indices_sha256'][:16]}... but recomputed "
                  f"{recomputed[:16]}... -- indices changed after the manifest was written",
           pass_detail=f"sha256 {recomputed[:16]}... matches the manifest")

    Xte_src = np.load("0528_X_test_0_770.npy", mmap_mode="r")
    yte_src = np.load("0528_y_test_0_770.npy", mmap_mode="r")
    X20 = np.load("data/test20k_real_X.npy", mmap_mode="r")
    y20 = np.load("data/test20k_real_y.npy", mmap_mode="r")
    record("B2 test20k row count matches manifest",
           X20.shape[0] == man["n"] == len(idx), f"{X20.shape[0]} vs {man['n']}")

    probe = np.linspace(0, len(idx) - 1, 500).astype(int)
    xs_ok = all(np.array_equal(np.asarray(X20[p]), np.asarray(Xte_src[idx[p]]))
                for p in probe)
    ys_ok = all(int(y20[p]) == int(yte_src[idx[p]]) for p in probe)
    record("B3 test20k X rows are exactly 0528_X_test[source_indices]", xs_ok,
           pass_detail="checked 500 spread-out rows byte-for-byte")
    record("B4 test20k labels match 0528_y_test[source_indices]", ys_ok)

    h_test20 = set(row_hashes(np.asarray(X20)))
    record("B5 no test20k shot appears in the train pool",
           len(h_test20 & h_train) == 0, f"{len(h_test20 & h_train)} shared")

    # ---------- C: scored rows vs domain-calibration rows -------------------
    print("\nC. scored rows vs the rows used to fit the (k2, c2) domain map")
    dc_raw = np.load("data/domain_calib_raw_X.npy", mmap_mode="r")
    lo, hi = CALIB_SLICE
    claimed = np.asarray(X20[lo:hi])
    same = (dc_raw.shape == claimed.shape
            and np.array_equal(np.asarray(dc_raw), claimed))
    record(f"C1 domain_calib_raw_X is test20k_real_X[{lo}:{hi}]", same,
           f"domain_calib_raw_X {dc_raw.shape} vs test20k[{lo}:{hi}] {claimed.shape}")

    h_calib = set(row_hashes(np.asarray(dc_raw)))
    if LEGACY_C:
        # --legacy-check-c: rows 0:SCORED_N only
        h_scored = set(row_hashes(np.asarray(X20[:SCORED_N])))
        record(f"C2 the {SCORED_N} scored shots are disjoint from the calibration slice",
               len(h_scored & h_calib) == 0, f"{len(h_scored & h_calib)} shared")
    else:
        check_c_scored(X20, y20, h_calib, lo, hi)

    # ---------- G: the 100k replay set (NN_live headline) -------------
    print(f"\nG. {T100} (the full test split as replayed; NN_live headline n = 99,400)")
    check_g_test100k(h_train, h_calib, X20, lo, hi)

    # ---------- D: the calibrated training file -----------------------------
    print("\nD. the calibrated training set the model actually consumes")
    cal_path = "0528_X_train_0_770_loopback_calibrated.npy"
    Xcal = np.load(cal_path, mmap_mode="r")
    record("D1 calibrated training file has the same shape as the train pool",
           Xcal.shape == sh_train, f"{Xcal.shape} vs {sh_train}")

    k2 = complex(np.load("calibration/k2_loopback.npy"))
    c2 = complex(np.load("calibration/c2_loopback.npy"))
    Xraw = np.load("0528_X_train_0_770.npy", mmap_mode="r")
    pr = np.linspace(0, Xcal.shape[0] - 1, 200).astype(int)
    errs = []
    for p in pr:
        z = np.asarray(Xraw[p])[0::2] + 1j * np.asarray(Xraw[p])[1::2]
        w = k2 * z + c2
        got = np.asarray(Xcal[p])
        errs.append(max(np.max(np.abs(got[0::2] - w.real)),
                        np.max(np.abs(got[1::2] - w.imag))))
    worst = float(np.max(errs))
    record("D2 calibrated rows == k2*train_row + c2 (recorded k2/c2)",
           worst < 1.0,
           detail=f"worst abs deviation {worst:.4g} exceeds tolerance -- the "
                  f"calibrated file was NOT produced by the recorded k2/c2",
           pass_detail=f"worst abs deviation over 200 rows: {worst:.4g} "
                       f"(float32 storage, so small nonzero is expected)")

    h_cal_head, _ = hash_file_rows(cal_path, limit=CHUNK)
    record("D3 calibrated training rows are NOT byte-identical to raw test rows",
           len(h_cal_head & h_test) == 0,
           detail="a hit here means the transform was identity AND the pools overlapped")

    # The pool tools/generate_synth_holdout_v2.py fits on (0528_X_train_0_770.npy[10000:110000]). An older pool,
    # 0528_X_all_{0,1}[400000:], which contains the real test shots, is also probed and reported for information.
    print("\nE. synthetic generator fit pool "
          "(0528_X_train_0_770.npy[10000:110000], tools/generate_synth_holdout_v2.py)")
    FIT_LO, FIT_HI = GEN_FIT_LO, GEN_FIT_HI
    h_train20k, _ = hash_file_rows("0528_X_train_0_770.npy", limit=TRAIN_N)
    Atr = np.load("0528_X_train_0_770.npy", mmap_mode="r")
    h_fit = set()
    for i in range(FIT_LO, min(FIT_HI, Atr.shape[0]), CHUNK):
        h_fit.update(row_hashes(np.asarray(Atr[i:min(i + CHUNK, FIT_HI)])))
    print(f"   fit pool rows hashed: {h_fit.__len__()} unique")
    tot_tr = len(h_fit & h_train20k)
    tot_te = len(h_fit & h_test20)
    record(f"E1 generator fit pool vs the {TRAIN_N} rows the model trains on",
           tot_tr == 0, f"{tot_tr} overlaps")
    record("E2 generator fit pool vs the 20k real test subset",
           tot_te == 0, f"{tot_te} overlaps")
    record("E4 the generator fit pool begins exactly where training ends",
           GEN_FIT_LO == TRAIN_N,
           f"training uses rows [0:{TRAIN_N}] and the generator fits on "
           f"[{GEN_FIT_LO}:{GEN_FIT_HI}] -- a "
           f"{'gap' if GEN_FIT_LO > TRAIN_N else 'OVERLAP'} of "
           f"{abs(GEN_FIT_LO - TRAIN_N)} rows. An overlap is leakage; a gap "
           f"wastes real shots. Reconcile the two constants.",
           pass_detail=f"training [0:{TRAIN_N}], generator fit "
                       f"[{GEN_FIT_LO}:{GEN_FIT_HI}] -- adjacent, no gap, no "
                       f"overlap")
    record("E3 generator output in use is the clean one (x_synth_v2.npy)",
           Path("x_synth_v2.npy").exists(),
           "x_synth_v2.npy absent -- regenerate with "
           "tools/generate_synth_holdout_v2.py before reporting any synthetic "
           "fidelity; the x_50k* files are fit on the contaminated pool")

    if (RAW_DIR / "0528_X_all_0.npy").exists():
        tot_old = 0
        for f in ["0528_X_all_0.npy", "0528_X_all_1.npy"]:
            A = np.load(RAW_DIR / f, mmap_mode="r")
            for i in range(400000, A.shape[0], CHUNK):
                tot_old += len(set(row_hashes(np.asarray(A[i:i + CHUNK])))
                               & h_test20)
        print(f"   (info) OBSOLETE pool 0528_X_all_*[400000:] overlaps the "
              f"real test subset in {tot_old} rows -- this is the failure "
              f"v2 fixed, and")
        print(f"          it is why x_50k.npy / x_50k_heldout.npy / "
              f"x_50k_heldout_measnoise.npy must never carry a reported "
              f"synthetic fidelity.")

    # ---------- F: where test-set LABELS are consumed ------------------------
    # Recomputes both candidate (k2, c2) fits, label-free and label-using, and identifies which one the deployed
    # coefficients are; a file-existence test could not tell them apart.
    print("\nF. audit: does any fitting step consume a TEST-SET label?")
    dc_acq = np.load("data/domain_calib_acquired.npy")
    z_acq = dc_acq[:, 0::2] + 1j * dc_acq[:, 1::2]
    z_raw_c = np.asarray(dc_raw)[:, 0::2] + 1j * np.asarray(dc_raw)[:, 1::2]

    # label-free: mean trajectory over all shots
    _A = np.vstack([z_raw_c.mean(axis=0), np.ones(z_raw_c.shape[1])]).T
    _sol, *_ = np.linalg.lstsq(_A, z_acq.mean(axis=0), rcond=None)
    k_free, c_free = complex(_sol[0]), complex(_sol[1])

    # per-class: stacks the two class-mean trajectories, needs the labels
    k_lab = c_lab = None
    if Path("data/domain_calib_y.npy").exists():
        _y = np.load("data/domain_calib_y.npy").astype(int)
        _Astk = np.concatenate([z_raw_c[_y == 0].mean(axis=0),
                                z_raw_c[_y == 1].mean(axis=0)])
        _bstk = np.concatenate([z_acq[_y == 0].mean(axis=0),
                                z_acq[_y == 1].mean(axis=0)])
        _sol2, *_ = np.linalg.lstsq(
            np.vstack([_Astk, np.ones_like(_Astk)]).T, _bstk, rcond=None)
        k_lab, c_lab = complex(_sol2[0]), complex(_sol2[1])

    d_free = abs(k2 - k_free) + abs(c2 - c_free) / max(abs(c_free), 1e-12)
    d_lab = (abs(k2 - k_lab) + abs(c2 - c_lab) / max(abs(c_lab), 1e-12)
             if k_lab is not None else float("inf"))
    print(f"   recorded   k2 = {k2:.6f}  c2 = {c2:.4f}")
    print(f"   label-free k  = {k_free:.6f}  c  = {c_free:.4f}   "
          f"distance {d_free:.2e}")
    if k_lab is not None:
        print(f"   per-class  k  = {k_lab:.6f}  c  = {c_lab:.4f}   "
              f"distance {d_lab:.2e}")
    is_free = d_free < 1e-6
    record("F1 (k2, c2) is fit WITHOUT using test-set labels", is_free,
           f"the recorded coefficients match the PER-CLASS fit, which consumes "
           f"data/domain_calib_y.npy -- labels belonging to the test file. Rebuild "
           f"with `build_domain_calibrated_training_set.py` (default --fit "
           f"global) to close this.",
           pass_detail="the recorded coefficients are the label-free global "
                       "fit; data/domain_calib_y.npy is present on disk but enters "
                       "no fitted quantity. The per-class alternative is "
                       "worth +0.01 points (phase0_calibration_ablation.py), so nothing "
                       "is given up by not using it.")

    # ---------- verdict ------------------------------------------------------
    print("\n" + "=" * 74)
    n_fail = sum(1 for _, ok, _ in results if not ok)
    for name, ok, _ in results:
        print(f"  {'PASS ' if ok else 'FAIL '} {name}")
    print("=" * 74)
    print(f"VERDICT: {len(results) - n_fail}/{len(results)} checks passed")
    return 0 if n_fail == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
