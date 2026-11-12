variable project_name
set project_name "NN_axi"
variable backend
set backend "vivado"
variable part
set part "xczu48dr-ffvg1517-2-e"
variable clock_period
set clock_period 10.0
variable clock_uncertainty
set clock_uncertainty 12.5%
variable version
# v1.25: 2026-09-04, direct user instruction -- "calibrate the training
# dataset such that when we run a loopback program the classifier will be
# calibrated to the waveforms played during loopback." v1.24's SF=8
# domain-matched retrain fixed the training/deployment SCALING_FACTOR
# mismatch (confirmed bit-accurately) but did NOT fix the live board's
# ~50-76% on-chip ceiling -- root cause traced instead to a genuine
# amplitude/phase difference between what test20k_real_X.npy represents
# on disk and what the real DAC->loopback->ADC chain actually delivers
# (a ~3.8x attenuation confirmed to persist even after refitting the k/c
# predistortion against test20k_real_X itself -- ruling out "wrong
# calibration reference" and pointing at the training DOMAIN itself).
#
# Fix: fit a per-class affine domain map (k2, c2) from a REAL loopback
# acquisition of test20k_real_X rows [1000:1600] -- disjoint from the
# scored [0:500] range, avoiding the exact leakage pattern the
# 2026-08-31 audit found and rejected in the old downconvert_dataset.ipynb
# (which fit its own k2/c2 against the same rows/labels it later scored).
# Applied k2,c2 to 0528_X_train_0_770.npy before training (same
# architecture/seed/procedure as v1.23/v1.24 otherwise). Offline sanity
# check on the untouched, non-leaked test20k_real_X[:500] (via the SAME
# k2/c2, applied to the test data, never its labels) scored 93.84%
# fidelity -- see export_loopback_calibrated_model.py and
# build_domain_calibrated_training_set.py. Unlike v1.24 (only s4.h/
# BatchNorm changed), this retrain changed w2/b4/b6/s4 -- a genuinely
# different decision function, not just a rescale, since the input
# DOMAIN (not just its scale) shifted. See CURRENT_BEST_RESULTS.md's
# 2026-09-04 entry for the full writeup.
set version "1.41"
# v1.40 (2026-10-06): RETRAIN ONLY -- firmware/weights/* and parameters.h's
# dense1 n_zeros/n_nonzeros (2722/3438 -> 3320/2840) change; NN_axi.cpp,
# NN.cpp, nnet_utils are byte-identical to v1.39. The model is trained on
# 200,000 real shots (rows [0:10000] + [110000:300000], skipping the synthetic
# generator's fit pool) instead of 10,000; 5 seeds, val_loss selected 20260904.
# Bit-accurate fixed-point emulation on the 20,000 acquired loopback shots:
# 95.61% (v1.39: 95.31%), 100.00% decision agreement with float, no overflow.
# 17% fewer nonzero ternary weights in dense1 => smaller adder tree, which is
# where v1.39's 17 failing clk_pl_0 endpoints were. v1.39 files kept in
# firmware/v1.39_backup_before_v1.40_retrain/.
#
# --- v1.39 notes follow ---
# v1.39 (2026-10-05): THE LATENCY BUILD. Tasks 2.1-2.4 of FINAL_TASKS.md in a
# single HLS cycle. Four changes, three of which are deletions.
#
# 1. reuse_factor 1540 -> 1 (task 2.2). The 2026-09-29 csynth sweep over eight
#    variants showed partial unroll is a dead end: any rf > 1 pins trip x II at
#    ~770 cycles, because nnet_dense_resource.h:33 forces the weight ROM into
#    BRAM whenever reuse_factor > 1. Only rf = 1 escapes, where that pragma is
#    skipped and the 6160 ternary weights become LUT constants. dense1 goes
#    1541 -> 2 cycles. Delivered by a fresh hls4ml export, which also restores
#    "ARRAY_RESHAPE variable=input_1 complete dim=0" in NN.cpp.
#
# 2. The v1.28-v1.36 debug taps are REMOVED (task 2.1): NN()'s six debug
#    ports, the dbg2/4/5/6/10 + dbg_in + raw_rb register files, and the
#    PRELOAD_L / RAW_L / PRE_TAIL_L / PRE_HEAD_L / DENSE_IN_L / DBG_L drains.
#    DENSE_IN_TAP alone was 96 of NN()'s 107 cycles. Every investigation that
#    needed them is closed. The armed capture keeps exactly what the scorer
#    reads: out[0..769] (the shot's input) and out[770] (the sigmoid).
#
# 3. LOAD_L REVERTS TO v1.32's SHIFT REGISTER, and this is the risk in this
#    build. ARRAY_RESHAPE complete propagates outward and makes in_local[] one
#    49,280-bit REGISTER again; a runtime-indexed write into it is defect
#    #19-#22, which measured ~50% on-chip fidelity because the high-order
#    lanes of HLS's four 49,280-bit barrel shifters do not survive synthesis.
#    v1.33's plain "in_local[j] = v0" was correct only while rf=1540 made
#    in_local a BRAM; returning to rf=1 withdraws that permission.
#    RECONSTRUCTED FROM THE SPEC: the hardware-verified v1.32 source is in
#    neither git nor firmware/*backup*, so the index arithmetic (newest pair at
#    1538/1539, shift down two per iteration, word i ends at 2i) is the part to
#    review before trusting this build.
#
# 4. Window stays the FULL 770 (task 2.3), reversing the 2026-09-29 truncation
#    recommendation -- on ACQUIRED data the [100,500) gain collapses and the NN
#    scores 0.5-0.8 points WORSE. Cost: full-unroll LUT is 163,668 (38%) at
#    1540 inputs rather than 89,495 (21%) at 800.
#
# Weights are NEW (task 1.2): trained on the LABEL-FREE domain map (task 1.1,
# closing leakage F1) with 5-seed val_loss selection. Selected seed 20260902,
# scoring 95.34% on 20,000 real acquired loopback shots against a 95.28%
# matched-filter ceiling on the same set. verify_no_leakage.py is 16/16.
#
# EXPECTED: ~11 cycles for NN() against the reference work's 10. Gate on the
# csynth PORT LIST (zero out_r_*_B) and on routed timing, not the HLS WNS
# estimate, which reads -0.44 ns for every variant including ones that routed
# at +0.002 ns with zero failing endpoints.
#
# (superseded) v1.38:
# v1.38 (2026-09-21): ONE-SHOT LAG FIX. The chip's answer described the
# PREVIOUS shot (pred[i]/y[i] at chance, correct pairing y[i-1]) -- no
# accuracy cost, but it blocks real-time feed-forward and puts an asterisk on
# every reported number. Root-caused on qick49/v1.37 this session:
#
#   * lag is exactly ONE FIRE (model-free cross-correlation over a 400-fire
#     interleaved sequence: lag0 55.6%, lag1 94.5%, lag3+ chance)
#   * NOT a host readback race (value constant over 260ms of polling;
#     trig_count advances exactly +1 per fire)
#   * NOT trigger timing (NN trigger swept 0.31us -> 80us: no change)
#   * NOT fixable with a second flush trigger (tProc delivers only one NN
#     trigger per program; trig_count +1 per fire out to 300us separation)
#   * stream is GATED at exactly 770 words/shot with no idle filler, so
#     production == consumption and the queue depth is pinned at exactly one
#     stale shot forever
#   * DRAIN_L is LEVEL-triggered, so it latches that stale pulse every time
#     (drain_count_last == 1 on every shot). Stale and fresh pulses have
#     identical amplitude -> no threshold can separate them, which is why
#     every onset_threshold sweep in NN_axi.h's history failed.
#
# Fix: IDLE_FLUSH_L keeps the input queue empty BETWEEN shots (bounded
# non-blocking reads in the not-triggered branch), so a trigger can only ever
# see its own shot's data. Flushing inside TRIGGER_C was considered and
# rejected: by then the queue holds stale AND in-flight fresh data, and at
# 100MHz the core cannot outrun a 552.96MHz producer.
#
# Hardware workaround already proven on v1.37 (interleaving a blank fire):
# y[i] = 94.51% [90.37, 96.89], vs y[i-1] = 50.21%. That is the number this
# build must reproduce WITHOUT the extra fire.
# v1.37: 2026-09-21 -- THE FIDELITY FIX. LOAD_L read each ADC word as
# ap_uint<16>, so the network received two's-complement samples reinterpreted
# as unsigned: -1 arrived as 65535. 46.6% of samples are negative, putting a
# 65536-wide cliff through the middle of the input distribution. DRAIN_L
# already read the same word SIGNED (`ap_int<16> pi`) for onset detection, so
# the design was internally inconsistent. Now both read signed.
#
# Retrained in the matching signed domain
# (analysis_scripts/acquisition/export_signed_domain_model.py, same seed, same
# architecture, same loopback-calibrated training set -- only the input
# transform changed). Offline on test20k_real_X[:2000] via k2/c2:
#     deployed unsigned model  93.84%      new signed model  95.74%
# against a linear ceiling of ~96.07% on the same data; the reference paper
# reports 95.80-96%. Measured on-chip before this change: 90.23%.
#
# The batchnorm scale s4 moved ~1e-6 -> ~1e-4, the expected ~100x, since the
# dense sums no longer carry the +65536 offsets.
#
# v1.36: 2026-09-11 -- probes BOTH open bugs, on the hypothesis that they are
# ONE bug. Datapath unchanged from v1.33/v1.34 (rf=1540, no register file).
#
# THE HYPOTHESIS. v1.34 verified 69.4% of in_local word-for-word (PRE covered
# [0..1001], DENSE_IN sampled the rest at stride 8) and found ZERO differing
# words on every shot of every dataset -- while the l2 residual was unchanged.
# The 472 words never observed are exactly in_local[1002..1539], the TAIL, and
# every shot whose residual fits a single-index model (21/21) has a candidate
# there. Separately, #25 proved the in_local commit is deferred by one trigger
# (LOAD_L cannot read back its own writes). If that deferral is PARTIAL rather
# than a clean whole-shot offset -- head from shot N-1, tail from shot N -- the
# classifier is fed a SPLICED input every shot, which would explain the
# residual, its ~65000 scale (arbitrary shot-to-shot deltas in the uint16
# view), and why 69% coverage saw nothing. One bug, not two.
#
# PROBES:
#  1. PRE re-pointed at the TAIL: in_local[1002..1539], all 538 words, the only
#     region never observed. Plus a 128-word HEAD control on the same shots so
#     a tail difference cannot be blamed on something global.
#  2. RAW lag probe becomes FOUR BANKS (lags 1, 32, 256, 700 iterations, 32
#     samples each). v1.34 used a single lag of 16 and got the previous shot
#     100.00%; this asks whether the write becomes visible at ANY distance
#     inside the trigger. The i-windows are disjoint so only one read fires per
#     iteration -- still a single extra read port.
#
# READING THE RESULT:
#   tail == caps[s-1] (like head) -> lag is clean, residual is arithmetic
#   tail == caps[s]   (current)   -> spliced input, ONE bug, fix the commit
#   lag 1 current, rest previous  -> HLS forwards the store; memory never
#                                    updates in-trigger
#   a later bank flips to current -> the commit has a measurable latency
#
# BRAM: caps 0..769, sig 770, taps 771..788, TAIL 789..1326, HEAD 1327..1454,
# dense-in 1455..1646, RAW 1647..1774. Total 1775 of 2048.

# v1.34: PREPPED 2026-09-10, NOT YET BUILT. Sources and testbench are staged
# and csim-validated; nothing has been synthesised, exported or placed. The
# version is bumped NOW only so the documented stale-version trap (v1.31's
# "built the wrong content under the previous IP version") cannot fire when
# this is green-lit. To build: run_csynth_only.tcl -> build_prj.tcl csim=0 ->
# build_v1.34_full.tcl (copy build_v1.33_full.tcl, s/1.33/1.34/).
#
# PURPOSE: settle #24's l2 residual. The stride-6 readback sees only 128 of 770
# pairs (17%) and matches the capture 256/256 EXACTLY on every shot -- so the
# words responsible for the residual are either at unobserved positions or the
# residual is not an input difference at all. v1.34 replaces the scattered
# sample with a CONTIGUOUS one: PRE word k is in_local[k] for k=0..1001, i.e.
# capture pairs 0..500, 65% of the window observed word-for-word.
#   - If the wide readback finds differing words -> the residual IS an input
#     difference, and their positions/values say what kind.
#   - If 1002/1002 still match -> the input is right and the error is in the
#     dense accumulation, which is a different investigation entirely.
#
# LAYOUT (2048-word BRAM): capture 0..769, sigmoid 770, taps 771..788,
# in_local PRE 789..1790 (1002 words), dense-input tap 1791..2046. Total 2047.
# The v1.29 POST gather is DROPPED -- it agreed with PRE 256/256 on every build
# from v1.30 on, so it measured nothing PRE did not, and its 256 words buy four
# times the PRE coverage.
#
# TWO TRAPS DELIBERATELY AVOIDED: (1) no staging array -- v1.30's PRE_GATHER
# built pre_sel[256] ARRAY_PARTITION complete, which at 1002 entries would
# rebuild the large register file v1.33 just removed; PRE_STRIDE==1 makes the
# index simply k, so PRE_L reads in_local[k] and writes out[] directly.
# (2) ONE out[] write per iteration (the v1.26 single-port lesson) -- the loop
# runs over WORDS, not pairs. The csynth port list (no out_r_*_B) remains the
# gate, and csim cannot check it.
#
# csim (armed) PASSED: 501 contiguous pairs, 65.1% of window, 0 mismatches vs
# capture; dense-input 0 mismatches, 0/84 differing vs PRE on the overlap.
#
# (superseded) v1.33:
# v1.33: 2026-09-10 -- ARCHITECTURAL: reuse_factor 1 -> 1540, removing the
# 49,280-bit register file entirely. Direct user instruction after v1.32's
# hardware results ("we currently don't have any major time constraints, so a
# potentially time expensive architectural change is not a big deal").
#
# WHY, given v1.32 already worked. v1.32 fixed the WRITE side of the defect
# (LOAD_L shift register, verified on hardware: in_local went from frozen above
# index ~426 to tracking the capture out to 1525, 0/256 slots frozen, fidelity
# ~50% -> 71.15%). But the 49,280-bit register ITSELF survived, so the class of
# construct that caused #19-#22 was still present. v1.33 deletes it.
#
# WHY reuse_factor is the lever, and why io_stream was NOT. The register exists
# because config2 had reuse_factor=1, which makes DenseResource_rf_leq_nin fully
# unroll all 6160 MACs and consume every one of the 1540 inputs in the SAME
# cycle. Nothing about the I/O type changes that: the io_stream export
# (hls4ml_qubit_iostream_20260910_103745, weights verified bit-identical) types
# the input as `nnet::array<ap_fixed<32,20>, 1540>`, i.e. ONE 49,280-bit stream
# beat -- the same monolith with a valid/ready handshake. Reviewing that
# generated C++ is what ruled Fix B out.
#
# At rf = n_in = 1540 the kernel arithmetic collapses to one input per cycle:
# multfactor = MIN(n_in, rf) = 1540, multiplier_limit = block_factor =
# ceil(6160/1540) = 4, multscale = 1. Inside ReuseLoop, `in_index += rufactor`
# immediately trips `if (in_index >= nin) in_index = ir`, so all four unrolled
# MACs in a cycle read the SAME data[ir]; w_index = ir + k*1540 with out_index
# = k, and w2 is stored [n_out][n_in], so the cycle computes
# acc[k] += data[ir]*w[k][ir] -- the correct dense, walked input-by-input.
# Both kernel asserts hold (rufactor >= nin; multiplier_limit == block_factor).
#
# CHANGES: config2 reuse_factor 1 -> 1540 (other layers stay at 1 -- q_output
# has n_in=4); `ARRAY_RESHAPE variable=input_1 complete dim=0` REMOVED from
# NN.cpp so input_1 becomes a BRAM; LOAD_L REVERTED from v1.32's shift register
# to the plain `in_local[j] = v0` (correct against a BRAM, and a shift register
# into a BRAM would be 1540 reads + 1540 writes per iteration); PRE_GATHER,
# INLOC_GATHER and DENSE_IN_TAP changed UNROLL -> PIPELINE, since a BRAM cannot
# serve 128/256 simultaneous reads.
#
# COST: the dense goes from ~1 cycle to 1540 (~15.4us at 100MHz), and rf>1 moves
# the 6160 ternary weights into BRAM (RESOURCE core=ROM_nP_BRAM inside the
# kernel). Expected to be much SMALLER in LUT/FF than any build since v1.27.
#
# All v1.28-v1.31 taps retained, so this build verifies itself the same way
# v1.32 did.
#
# (superseded) v1.32:
# v1.32: 2026-09-09 -- THE FIX. Not a probe build: this is the repair for the
# on-chip fidelity defect that v1.26-v1.31's capture/tap builds localised.
#
# WHAT WAS BROKEN. LOAD_L wrote in_local[] through a RUNTIME index
# (in_local[j], j = 0,2,4,...) into an array that ARRAY_RESHAPE complete
# (propagated in from NN()'s input_1 port) makes a SINGLE 49,280-bit register.
# Vitis HLS lowered that to a read-modify-write built from four 49,280-bit
# VARIABLE barrel shifters per iteration -- two masks (0xFFFFFFFF << j*32 and
# << (j*32|32)) and two data shifts -- and costed all four at 33,355 LUT
# (NN_axi_Pipeline_LOAD_L_csynth.rpt) when one correct 49,280-bit shifter is
# ~790K LUT-equivalents on a 425,280-LUT device. On real silicon the low-order
# lanes of that network work and the high-order ones do not: measured on
# hardware, in_local[] tracks the BRAM capture only below index ~426 of 1540
# (highest varying 421, lowest permanently-static 432), while the out[] capture
# written in the SAME LOAD_L iterations is 770/770 complete -- proving the loop
# ran and only its in_local[] writes were lost. The dense layer was summing
# ~426 live words and ~1114 stale ones, which is exactly what the v1.28 layer
# taps saw: q_dense_1 the only broken layer (l2 0/384 exact), and every layer
# downstream bit-exact when given the chip's own upstream values (l4 384/384,
# l5 384/384, l6 96/96, l10 96/96, sigmoid 96/96 -- this exonerates the
# ap_fixed<16,6> wrap, the wide BatchNorm multiply and the sigmoid LUT, which
# were the leading hypotheses before the taps existed).
#
# Ruled out before settling on this: port truncation (in_local_out and NN's
# p_read are both [49279:0]), a LOAD_L/NN start race (FSM state8 waits on
# LOAD_L's ap_done before state10 starts NN), and register trimming (the sole
# "trimmed from" warning is unrelated; the zext_ln609/610_reg[6:15] merge is a
# correct optimisation). This is TIMING_CLOSURE_REFERENCE #4 -- massive fanout
# from runtime-variable addressing, the project's worst recurring failure --
# appearing as a FUNCTIONAL fault instead of a timing one, which is why routed
# timing (zero failing clk_pl_0 endpoints on all four tapped builds) never
# caught it, and why csim always passed: csim has one plain array and no
# register file, so it cannot express this failure mode.
#
# THE FIX ("fix D"). LOAD_L never needed random access -- j is strictly
# sequential; the demux existed only because the C expressed it as an index.
# LOAD_L is now a SHIFT REGISTER: each iteration does in_local[m] = in_local[m+2]
# for all m (fully unrolled, every index a compile-time constant) and writes the
# new pair at the top. Fixed register-to-register wiring with one common enable:
# no shifter, no demux, no mask, no Expression LUT. Shifting DOWN toward index 0
# preserves sample order EXACTLY -- word i lands at 1538-2*(769-i) = 2i, where
# the old in_local[j] put it -- so weights, the offline emulator, decode_taps.py
# and every capture slot index are unchanged. This is a bit-identical
# replacement, not a reversal.
#
# Also removed: RESET_IN_LOCAL_L. It was a second write site into in_local[],
# and the v1.13 note in NN_axi.cpp records that a second write site makes HLS
# materialise extra full-width 49,280-bit staging/merge copies. LOAD_L now
# overwrites all 1540 entries on every shot regardless of prior contents, so
# the zeroing was dead code buying a 2:1 mux on 49,280 bits.
#
# All v1.28-v1.31 taps are KEPT so this build verifies itself: the PRE / POST /
# DENSE_IN readbacks (slots 789/1045/1301) must now agree with the capture
# (slots 0..769) on 256/256 words instead of failing above ~426.
#
# (superseded) v1.31:
# v1.31: 2026-09-09 -- tap the FIRST DENSE'S OWN INPUT, from inside NN().
# Slots 1301..1556 hold the 128 I/Q pairs (same indices/stride as the NN_axi
# gather) read through the very pointer nnet::dense dereferences.
#
# WHY, and a RETRACTION. v1.29/v1.30 sampled in_local[] in NN_axi -- before and
# after the NN() call -- and both disagreed with the capture on ~99.5% of
# words, identically (1528/1536 each, identical per shot). That cleanly killed
# the "DATAFLOW channel storage" explanation, since the mismatch is present
# before NN() ever touches in_local. But the reading itself does NOT survive an
# internal consistency check and is RETRACTED:
#   * the sampled values are near-constant across shots -- 185/256 slots
#     bit-identical over four shots, per-slot sigma 1613 vs the capture's 13230
#   * yet the measured l2 varies with CV 0.66-1.05 and correlates +0.44..+0.88
#     with the true l2
# A constant input cannot produce a varying, input-correlated l2. So the NN_axi
# gather is not reading the copy of in_local that the kernel consumes.
# NN_axi.cpp's own v1.13 note says why this is possible: HLS creates multiple
# 49,280-bit copies of in_local[] plus a staging/merge register.
# Also checked and rejected: the corruption is not a fixed permutation either
# (0/256 slots have a consistent source index across 5 shots).
#
# The only sample that cannot be the wrong copy comes from the pointer the
# kernel itself dereferences. That needs NO tee and NO 49K-FF pass-through
# copy: #pragma HLS DATAFLOW constrains channels BETWEEN processes, not reads
# WITHIN one, so dense1_with_tap() both taps and computes in a single process
# and `data` still has exactly one consumer at the region level.
#
# DENSE_IN matches the capture -> the kernel IS fed correct data, and the
#   defect is the weight ROM indexing or the 1540-term adder tree.
# DENSE_IN mismatches -> the kernel really is fed wrong data.
# Weights UNCHANGED from v1.25; observability only.
#
# (superseded) v1.30:
# v1.30: 2026-09-08 -- in_local[] sampled PRE and POST NN(). Slots 789..1044
# are the same 128 constant-index I/Q pairs read BETWEEN LOAD_L and NN();
# 1045..1300 are those pairs read AFTER NN() returns (v1.29's measurement).
#
# WHY: v1.29 measured the POST-NN() readback disagreeing with the capture on
# 99.32% of words (1017/1024 over 4 shots). That is not a readback artefact --
# repeated reads of one static record varied in 0/1045 words -- and not the
# 20ps clk_pl_0 margin for the same reason. The values keep the right I/Q
# structure (large I, small Q) but vary ~8x less shot-to-shot than real data
# (per-slot sigma 1613 vs 13230) and 185/256 slots are bit-identical across
# four different shots. They also do NOT match another shot's capture at these
# indices (0-13/256, i.e. chance), so it is not a plain one-shot lag.
#
# BUT a post-NN() read alone cannot separate:
#   (a) LOAD_L's writes into in_local[] are corrupted -> NN() is fed bad data,
#       which would be the root cause of the whole q_dense_1 divergence; from
#   (b) in_local[] is CORRECT when NN() reads it, and the post-NN() read sees
#       different storage because input_1 is a DATAFLOW channel that HLS may
#       buffer, ping-pong or reallocate once consumed -- in which case the
#       defect is inside the dense kernel RTL and v1.29's result is an
#       artefact of WHERE the sample was taken.
# Reporting (a) without excluding (b) would be exactly the kind of premature
# root-cause call this project's history (#12 closed on a premise #17 later
# retracted) shows to be expensive. So sample both sides of the call.
#
# PRE matches capture and POST does not -> (b). PRE also mismatches -> (a).
# Weights UNCHANGED from v1.25; observability only.
#
# (superseded) v1.29:
# v1.29: 2026-09-08 -- in_local[] READBACK. Adds 256 words (slots 789..1044)
# holding in_local at 128 constant-index I/Q pairs, stride 6 pairs.
#
# WHY: v1.28's taps answered the localisation question definitively. Given its
# OWN inputs, every layer downstream of the first dense is BIT-EXACT on
# silicon -- batchnorm1 384/384, relu1 384/384, q_output 96/96,
# q_output_alpha 96/96, sigmoid 96/96, max|d| = 0 on every one. Only
# q_dense_1 diverges: 0/384, at every caps/taps offset from -2 to +3, so it is
# not a pairing artifact. The three long-standing suspects (the wide l2*s4
# multiply, the ap_fixed<16,6> wrap, the sigmoid LUT) are all EXONERATED.
#
# The defect is also not a binary-point shift: per-neuron chip/emul ratios are
# ragged (0.109, 0.172, -0.159, 0.207 -- note neuron 2 is sign-inverted), not
# clean powers of two. And it is not a fixed wrong weight matrix either: under
# 6-fold CV a fitted linear map reaches only ~75% of the ceiling that the SAME
# ridge procedure achieves on a target known to be exactly linear (0.70/0.75/
# 0.22/0.67 observed vs 0.95/0.94/0.30/0.93 ceiling; shuffled control ~0.00).
# So part of what the trees sum does not come from the captured words at all.
#
# That points at in_local[] being corrupted between LOAD_L writing it and NN()
# reading it -- and that is precisely what the capture path CANNOT see, since
# out[i] and in_local[2i]/[2i+1] come from the same `word` in the same
# iteration but out[] is a plain BRAM write while in_local[] goes through a
# runtime-indexed demux over a fully ARRAY_RESHAPE'd 1540-entry register file.
#
# Reading in_local back is cheap ONLY at constant indices (a wire per read,
# not the 1540:1 mux the v1.26 header warned about); the gather is fully
# unrolled with compile-time indices and only a small 256-entry staging array
# is addressed at runtime. Sampled as whole I/Q PAIRS so it cannot repeat
# v1.26's one-channel sampling mistake. Weights UNCHANGED from v1.25.
#
# (superseded) v1.28:
# v1.28: 2026-09-08 -- DATAFLOW-TAP build. NN() now exports every intermediate
# layer (layer2/4/5/6/10) as raw fixed-point BITS into capture slots 771..788,
# alongside v1.27's exact input capture (0..769) and sigmoid (770).
#
# WHY: v1.27's capture answered the question the v1.26 header posed. Replaying
# the exact captured in_local[] through the offline emulator (bit-exact vs
# csim, 499/500 shots) disagrees with the chip on 95/96 shots across four
# datasets. The chip's logit correlates +0.9085 with the emulator's but spans
# only [-8.00, -2.14] where the emulator spans [-17.24, +12.07] -- compressed
# ~5x and pinned onto the g->0 bias floor (1 + relu(b4_0) - relu(b4_1) -
# relu(b4_3) = -5.5113, which is exactly the measured all-zero-input
# response). Every offline hypothesis was tested and REJECTED: a global gain
# on the BatchNorm multiply (19/96 within 1 LSB), independent per-neuron gains
# (23/96, and it picks three DIFFERENT exponents, so no systematic
# binary-point shift), I/Q swap (4/96), signed-vs-uint16 (1/96), single-channel
# masking (7/96), prefix/suffix truncation (14/96), all 770 whole-pair input
# rotations (12/96), and a 6-parameter per-neuron affine distortion of l2
# (15/96). Baseline as-captured is 9/96. The distortion is therefore NOT a
# static function of the captured input and cannot be identified from the
# output -- the intermediates must be observed. Weights are UNCHANGED from
# v1.25; this build adds observability only.
#
# HOW, WITHOUT REMOVING DATAFLOW: #10/#13 record that structural closeness to
# a known-good reference is this design's dominant timing lever, and v1.20
# removed DATAFLOW and broke timing catastrophically. A DATAFLOW channel
# permits only ONE producer and ONE consumer, so a second reader of
# layer2_out is illegal (and HLS sometimes answers by silently SERIALISING the
# region -- DATAFLOW removal by accident). Instead each tap is a tee PROCESS
# spliced into the chain: dense -> layer2_raw -> tap_l2 -> {layer2_out, dbg2}.
# Every channel keeps one producer and one consumer; every dbg array is a
# port written by exactly one process. The region is untouched, and so is the
# scheduling. Taps write raw .range() bits rather than floats specifically to
# avoid synthesising 14 fixed-to-float converters.
#
# COST, MEASURED post-opt vs v1.27 (the pre-build estimate of "~800 FFs and no
# arithmetic" was WRONG -- off ~20x on registers):
#     CLB LUTs      223,439 (52.5%) -> 256,085 (60.2%)   +32,646  (+14.6%)
#     CLB Registers 210,986         -> 227,999           +17,013   (+8.1%)
#     BRAM 500.5 / DSP 648          -> unchanged
# Flat BRAM/DSP confirms no multipliers or memory were added, so the four
# fully-unrolled 1540-term adder trees really are untouched. The growth is
# from five new DATAFLOW channels (layer_raw -> tee -> layer_out) plus the
# loss of dense/normalize fusion. At 60.2% LUT occupancy congestion is a real
# timing risk: gate on clk_pl_0 having zero failing endpoints (#3), and treat
# this as a DIAGNOSTIC bitstream, not a shippable one.
#
# DBG_L makes exactly ONE write to out[] per iteration (an 18:1 mux over
# registers), so the v1.27 single-port lesson still holds: the gate for this
# change is the csynth PORT LIST (no out_r_*_B), not csim.
#
# (superseded) v1.27:
# v1.27: 2026-09-08 -- FIXES v1.26's capture, which was measured broken on
# real hardware (qick38). v1.26 wrote BOTH halves of each stream word in one
# LOAD_L iteration; HLS answered by inferring a DUAL-PORT `out` interface
# (csynth shows out_r_EN_A/WEN_A and out_r_EN_B/WEN_B), but the BD gives
# NN_axi only one port -- `out` is /axi_bram_ctrl_0_bram, a 2048x32
# blk_mem_gen whose second port belongs to axi_bram_ctrl_0 for PS readback.
# Half the writes went nowhere: on hardware every EVEN slot read back as the
# bit-identical constant 0x3B800000 while odd slots held correct data.
# csim passed v1.26's capture check regardless, because C simulation does not
# model BRAM port counts -- the standing lesson is that the csynth PORT LIST
# (absence of out_r_*_B), not csim, is the gate for an interface change.
# v1.27 writes the raw packed word ONCE per iteration through a
# unsigned/float union, so I and Q still come from the same shot, losslessly.
#
# (superseded) v1.26: 2026-09-07 DIAGNOSTIC INPUT-CAPTURE build (user instruction: stop
# running black-box tests, observe what the classifier actually sees).
# Weights are v1.25's, UNCHANGED -- this build adds observability only.
#
# Adds: LOAD_L mirrors every value it writes into in_local[] out to the
# BRAM when armed via window_offset == CAPTURE_ARM (0xC0DE), and STORE_L
# puts that shot's sigmoid at the fixed slot 2*N_IQ_WINDOW_IN. One armed
# shot therefore yields a self-contained 1541-word BRAM record: the exact
# 1540-value NN input plus the probability the chip computed from it.
#
# Chosen over ILA/JTAG probes deliberately: the 2026-08-31 diag2 build
# (deployed as qick33) already put 10 ILA probes in this design and its
# capture was blocked by a JTAG driver issue, which is why that
# investigation pivoted to RTL reconstruction. The BRAM path is the one
# readback channel proven working on every single run since.
#
# Reuses the dead window_offset AXI-lite register as the arm signal, so
# this needs NO BD change and NO change to the deployed NNQickSoc driver
# class. NN() itself is deliberately UNTOUCHED -- tapping its intermediate
# arrays would need #pragma HLS DATAFLOW removed (second consumer per
# channel), which would perturb the very behaviour being measured. Layer
# internals are instead reconstructed offline from the captured input,
# which is exact because the weights are known; if that reconstruction
# disagrees with the chip's own sigmoid, THEN the fault is inside NN() and
# an invasive internals build is justified.
#
# Motivated by Phase-0 register diagnostics the same day, which ruled out
# the leading "wrong capture window" hypothesis: drain_count_last == 1 on
# every dataset, the 100%-scoring and the 50%-scoring alike.
variable maximum_size
set maximum_size 4096
