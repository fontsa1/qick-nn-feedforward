#ifndef NN_AXI_H_
#define NN_AXI_H_

#include "NN.h"
#include "hls_stream.h"
#include "ap_axi_sdata.h"

#include <iostream>

static const unsigned N_IQ_IN = 770; // 770 (I,Q)
static const unsigned N_IQ_WINDOW_IN = 770; // full window, matches current model's 1540 (=770*2) input
static const unsigned N_OUT = 1; // single sigmoid output
static const unsigned N_CONFIGS = 2;
static const unsigned BUFFER_SIZE = N_OUT * 16384; // 1 * 16384 * 4B = 64KB

// v1.28 capture record layout (one armed shot, self-contained):
//   [0                 .. 769]  packed I/Q stream words = the exact NN input
//   [770]                       the sigmoid the chip produced from it
//   [771               .. 788]  raw fixed-point bits of every NN() internal
// = 789 words, comfortably inside the 2048-word physical BRAM. Nothing else
// is written during an armed shot, so the record cannot be clobbered.
static const unsigned CAP_SIG_SLOT = N_IQ_WINDOW_IN;          // 770
// v1.39 (2026-10-05): every other CAP_* / N_DBG_* / RAW_* / PRELOAD_*
// constant is REMOVED along with the drains that used them (tasks 2.1/2.2).
// The armed capture record is now just two things: words [0 .. 769], the
// shot's exact NN input, written by LOAD_L; and word [770] = CAP_SIG_SLOT,
// the sigmoid STORE_L computed from it. That is the channel
// score_armed_fidelity.py reads. The removed slots indexed the v1.28-v1.36
// layer taps and in_local readbacks -- see project.tcl's version history and
// CURRENT_BEST_RESULTS.md for what each one established. Restoring any of
// them means restoring NN()'s debug ports too, which costs 96 cycles for
// DENSE_IN_TAP alone against the dense kernel's 2.
// N_DBG_* / N_DBG_TOTAL come from NN.h so both sides share one definition.

// v1.29 in_local READBACK (2026-09-08). v1.28's taps proved q_dense_1 is the
// ONLY broken layer (batchnorm1/relu1/q_output/q_output_alpha/sigmoid are all
// bit-exact given their own inputs, max|d| = 0), and that chip l2 reaches only
// ~75% of the achievable held-out R^2 of a fitted linear map -- so it is not a
// fixed wrong weight matrix either. That leaves in_local[] being corrupted
// between LOAD_L writing it and NN() reading it.
//
// This is the one thing the capture path is STRUCTURALLY BLIND TO: out[i] and
// in_local[2i]/[2i+1] are written in the SAME LOAD_L iteration from the SAME
// `word`, but out[] is a plain BRAM write while in_local[] goes through a
// runtime-indexed demux over a fully ARRAY_RESHAPE'd 1540-entry register file
// (~33K LUT per the v1.13 note; TIMING_CLOSURE_REFERENCE #4 calls this
// pattern the project's most severe historical failure). A dropped or
// misdirected write there corrupts in_local while leaving the capture
// perfect, which is exactly the observed signature.
//
// Read back CONSTANT indices only. The v1.26 header warned that reading
// in_local costs a 1540:1 mux -- true for a RUNTIME index, but a constant
// index into a partitioned array is just a wire, so a fully-unrolled
// constant-index gather is free. Sampled as 128 whole I/Q PAIRS at stride 12
// (in_local[12k] and [12k+1], k = 0..127, covering 0..1525) rather than a
// flat stride, so it can never accidentally sample one channel -- the exact
// mistake that produced v1.26's bogus "all zeros" reading.
//
// INLOC_STRIDE is counted in I/Q PAIRS, not words. in_local holds 770 pairs
// (1540 words), so sampling 128 of them needs stride 770/128 -> 6. The gather
// reads word indices 2*STRIDE*k and 2*STRIDE*k+1, i.e. 12k and 12k+1, and the
// last pair is k=127 -> words 1524/1525, just inside 1540. (Getting this
// wrong the other way -- treating the stride as words -- indexes 3048 and
// runs off the end of the array.)
static const unsigned N_INLOC_PAIRS = 128;
static const unsigned N_INLOC = 2 * N_INLOC_PAIRS;            // 256 words
static const unsigned INLOC_STRIDE = 6;                       // in PAIRS

// v1.30 PRE/POST split (2026-09-08). v1.29 measured the POST-NN() readback
// disagreeing with the capture on 99.32% of words, with the right I/Q
// structure but ~8x less shot-to-shot variance than real data and 185/256
// slots bit-identical across four different shots. The readback path itself
// is deterministic (0/1045 words varied over repeated reads of one static
// record), so that is not the 20ps clk_pl_0 margin.
//
// But a POST-NN() read alone cannot distinguish the two candidate causes:
//   (a) LOAD_L's writes into in_local[] are corrupted, so NN() is fed bad
//       data -- the actual defect; or
//   (b) in_local[] is CORRECT when NN() reads it, and the post-NN() read
//       observes different storage, because input_1 is a DATAFLOW channel
//       that HLS may buffer, ping-pong or reallocate once consumed.
// Ruled out already: the values do not match another shot's capture at these
// indices (0-13/256, chance), so it is not a plain one-shot lag.
//
// So sample the SAME 128 pairs twice -- once between LOAD_L and NN(), once
// after NN() returns:
//   PRE  matches capture, POST does not  -> (b): the input is fine, the
//        defect is inside the dense kernel's own RTL, and the POST read is an
//        artefact of dataflow channel storage.
//   PRE  also mismatches                 -> (a): LOAD_L genuinely does not
//        put the captured words into in_local, and that is the root cause.
// v1.34 PREPPED (not yet built): WIDE, CONTIGUOUS in_local readback.
//
// WHY. #24's residual -- chip l2 differing from emulator(caps[s-1]) by exact
// integers clustered near multiples of 65536 -- must come from in_local words
// the readback never looks at: the stride-6 sample covers only 128 of 770
// pairs (17%), and it matches the capture 256/256 EXACTLY on every shot and
// every dataset. Either the differing words systematically avoid every-6th
// pair (no plausible mechanism), or the difference is not in in_local at all.
// A different, much wider sample settles it: if the residual is an input
// difference, 65% coverage will catch it; if the wide sample STILL matches
// perfectly, the error is in the dense accumulation, not the input.
//
// LAYOUT. The 2048-word BRAM is the budget:
//     [0    .. 769 ]  capture              770
//     [770        ]   sigmoid                1
//     [771 .. 788 ]   layer taps            18
//     [789 .. 1790]   in_local PRE        1002   <- was 256, stride 6
//     [1791 .. 2046]  dense-input tap      256
//                                        -----
//                                         2047  (fits 2048)
// POST (v1.29's post-NN() gather) is DROPPED. It has agreed with PRE 256/256
// on every build since v1.30, so it is pure redundancy, and its 256 words buy
// four times as much PRE coverage.
//
// PRE_STRIDE is 1, so PRE reads in_local[0..1001] CONTIGUOUSLY -- pairs 0..500,
// 65% of the window, every word in that range observed rather than sampled.
// Contiguous beats scattered here: the suspect words are few and their
// positions are unknown, so what matters is how many words are seen at all.
//
// NOTE the two traps this layout deliberately avoids:
//   1. NO staging array. v1.30's PRE_GATHER built `pre_sel[256]` with
//      ARRAY_PARTITION complete; at 1002 entries that would rebuild a large
//      register file -- exactly the construct v1.33 just removed. With
//      PRE_STRIDE == 1 the index is simply k, so PRE_L reads in_local[k] and
//      writes out[] directly, no staging.
//   2. ONE write to out[] per iteration (the v1.26 single-port lesson): the
//      loop runs over WORDS, not pairs, so it never writes two slots in one
//      trip. The csynth port list (no out_r_*_B) is still the gate.
// v1.35: PRE is re-pointed at the TAIL, which is the only part of in_local
// nothing has ever observed.
//
// WHY. v1.34 verified 69.4% of the window word-for-word -- PRE covered
// in_local[0..1001] contiguously and DENSE_IN sampled the rest at stride 8 --
// and found ZERO differing words on every shot of every dataset, while the l2
// residual was unchanged. The 472 words never looked at are exactly
// in_local[1002..1539], i.e. capture pairs 501..769: the TAIL. Every shot
// whose residual fits a single-index model (21/21) has a candidate there.
//
// THE HYPOTHESIS THIS TESTS, which would make the two open bugs ONE bug: the
// deferred in_local commit (#25 -- LOAD_L cannot even read back its own writes
// within a trigger) may be PARTIAL rather than a clean whole-shot offset. If
// the head holds shot N-1 while the tail holds shot N, the classifier is fed a
// SPLICED input on every shot. That predicts exactly what is seen: head
// matching caps[s-1] perfectly, tail differing, and differences of arbitrary
// shot-to-shot size -- which in the uint16 view are frequently ~65000, the
// observed residual scale.
//
//   tail matches caps[s-1] too      -> the lag IS clean; the residual is in
//                                      the dense arithmetic after all
//   tail matches caps[s] (current)  -> spliced input confirmed, one bug not
//                                      two, and the fix is the commit
// Head control: re-confirm on the SAME shots that the head still matches, so a
// tail difference cannot be blamed on a global change.

// v1.34: CAP_INLOC_POST_BASE removed -- POST dropped, see the note above.

// v1.31: the DENSE KERNEL'S OWN view of its input, tapped from inside NN()
// (see dense1_with_tap in NN.cpp). The PRE/POST samples above are taken in
// NN_axi and were shown NOT to be the copy the kernel consumes -- they are
// near-constant across shots while the measured l2 varies with CV 0.66-1.05
// and correlates +0.44..+0.88 with the true l2, which a constant input cannot
// do. This slot holds the one sample that cannot be the wrong copy, since it
// comes from the very pointer the kernel dereferences.
//
// Compare DENSE_IN against the capture:
//   matches   -> the kernel IS fed the right data; the defect is the weight
//                ROM indexing or the 1540-term adder tree, and NN_axi's
//                in_local gather was simply reading a different copy.
//   mismatches-> the kernel really is fed wrong data, and PRE/POST were
//                right about the corruption even though they sampled a
//                different copy of it.

// v1.34 LAG PROBE (CLASSIFIER_DEBUGGING_REFERENCE #25). The one-shot lag is
// measured and is NOT a host/board race (0 triggers execute during the
// readback, and flipping the read order changes nothing), yet the RTL looks
// correct: one in_local BRAM, LOAD_L @state7 -> PRE_L @state11 -> NN @state12.
// The untested question is whether LOAD_L's writes COMMIT during the trigger
// that issues them.
//
// RAW_L answers it. Inside LOAD_L, iteration i reads in_local back at a pair
// written RAW_LAG iterations EARLIER and stashes it. The lag matters: reading
// the word just written would let HLS store-to-load forward the value and the
// probe would report "correct" no matter what the memory holds. 16 iterations
// of separation is far beyond any forwarding window, so this is a genuine
// memory read.
//
//   reads back THIS shot's value -> the write commits, and something between
//                                   LOAD_L and PRE_L aliases the memory
//   reads back the PREVIOUS shot's -> the writes are not landing in the
//                                     issuing trigger; the whole lag follows
//
// Costs one extra read port on in_local (LOAD_L already only writes it), so it
// may push LOAD_L's II from 2 to 3 -- pure latency, irrelevant here. It does
// NOT modify in_local, so it cannot perturb the classification being measured.
// v1.35: the lag probe becomes FOUR BANKS at different read-back distances.
// v1.34 used a single lag of 16 and got the previous shot 100.00%. The open
// question is whether the write becomes visible at ANY distance inside the
// trigger. The four banks read at lags 1, 32, 256 and 700 iterations; their
// iteration windows are disjoint so only ONE read fires per iteration and the
// cost stays at a single extra read port.
//   lag 1 current, rest previous -> HLS is forwarding the store, the memory
//                                   itself is never updated in-trigger
//   all four previous            -> writes never land during their own trigger
//   later lags current           -> the commit has a latency, and its length
//                                   is the number between the banks that flip

// ---- v1.36 PRELOAD probe -------------------------------------------------
// Read in_local[] BEFORE LOAD_L overwrites it, at the SAME indices PRE_TAIL
// re-reads afterwards. #27 proved in_local[0..1531] holds the PREVIOUS shot
// while in_local[1532..1539] holds the CURRENT one; PRELOAD vs PRE_TAIL says
// directly whether LOAD_L's writes become visible inside their own trigger,
// and for which indices. 64 words covers the 8 anomalous ones plus 56 normal
// neighbours at the same end of the array.
// full record = 1301 + 256 = 1557 words, inside the 2048-word physical BRAM

// v1.11 fidelity-ceiling fix (2026-08-15): live JTAG capture via the
// pre-existing system_ila_1 core (no rebuild needed to observe this --
// see CONTEXT_DUMP.md's 2026-08-15 updates) showed NN_axi's raw stream
// input is mostly near-zero noise floor with a brief, positionally-
// variable real-signal burst somewhere inside the 770-word window it
// blindly reads starting at trigger time -- unlike axis_avg_buffer_0,
// which does its own internal trigger-gated capture and is why software's
// acquire_decimated() sees a clean, correctly-windowed signal. Root cause:
// axis_data_fifo_0 (HAS_TREADY=0 upstream, no backpressure) accumulates
// backlog from the continuously-flowing decimated ADC stream during the
// (typically ~100us) gap between shots; LOAD_L's fixed-count read starting
// exactly at trigger time drains through whatever's queued -- old backlog
// first, the actual fresh pulse response landing wherever in the 770-word
// budget the "catch up to live" point happens to fall. An empirical
// trigger-delay sweep (software-only, no rebuild) confirmed timing
// alignment is a REAL factor but not reliably fixable with a single fixed
// offset -- results were inconsistent across different shot batches,
// consistent with genuine per-shot jitter in how much backlog accumulates.
//
// Fix: NN_axi now does its own bounded "drain until onset" phase before
// the real N_IQ_WINDOW_IN-word capture, using a magnitude threshold to
// detect where real signal (vs. noise floor) begins -- adaptive per shot,
// not a fixed delay. MAX_DRAIN_WORDS bounds this by axis_data_fifo_0's own
// FIFO_DEPTH (1024): with HAS_TREADY=0 upstream, no possible backlog can
// exceed the FIFO's own capacity (writes beyond that are either dropped or
// overwrite the oldest content, depending on the FIFO's overflow
// behavior -- either way, 1024 words is a hard structural upper bound on
// how much could ever need to be drained to reach live data). If no
// onset is found within that bound (a genuinely quiet/spurious trigger),
// LOAD_L falls back to reading the remaining words from wherever the
// drain left off -- same total word budget either way, so the existing
// unconditional trig_count/out_offset commit logic (see NN_axi.cpp's
// extensive comment on the ORIGINAL k/trig_count bug) is untouched and
// not put at risk of reintroducing that class of hazard.
//
// v1.14 (2026-08-20): ROOT-CAUSE FOUND for the qick18/qick19 ~75-80%
// fidelity ceiling. The "bounded by FIFO_DEPTH" reasoning above is WRONG:
// axis_data_fifo_0 has HAS_TREADY=0 upstream (confirmed in the BD JSON,
// FIFO_DEPTH=1024) -- a continuously-flowing producer with no
// backpressure. FIFO_DEPTH bounds how much can be BUFFERED at one instant,
// not how much can be DRAINED over time from a stream that keeps
// flowing -- draining past the FIFO's nominal depth just means reading
// progressively fresher live samples, not running out of data. Confirmed
// via two independent lines of evidence:
//   1. Live hardware telemetry (2026-08-20, new max_onset_magnitude_last/
//      magnitude_at_drain_end probes + the existing drain_count_last):
//      EVERY tested shot on qick19 (30/30 across two different shots)
//      hit drain_count_last == MAX_DRAIN_WORDS exactly, every time,
//      regardless of shot identity -- DRAIN_L was NEVER finding a real
//      onset, always falling back.
//   2. csim against the real acquired dataset (loopback_nn_acquired_500.
//      npy, via NN_axi_test.cpp's synthetic-backlog-prefix mechanism):
//      100.00% accuracy (0 misses) for synthetic backlog up to 900 words,
//      collapsing to 43-53% (onset_miss_count 167-250/500) for backlog
//      >=1000 words -- a sharp cliff right at the old MAX_DRAIN_WORDS=1024
//      boundary, for BOTH the current v1.13 model (fresh data) and the
//      original v1.11 model (old data, found independently in this
//      project's csim log history) -- same underlying mechanism, unchanged
//      since v1.12's LOAD_L addressing fix.
// A software-only fix was tried first and ruled out: lowering
// onset_threshold (so DRAIN_L crosses it sooner) made real hardware
// fidelity WORSE, not better (48-53% vs the already-degraded 73-80%
// baseline) -- lower thresholds trigger on backlog noise before reaching
// the real signal, they don't fix the underlying "not enough drain budget
// to reach live data" problem.
//
// Fix: MAX_DRAIN_WORDS raised 1024->8192 (8x). This only affects the
// WORST-CASE fallback latency (when no onset is found at all, or one is
// found late) -- DRAIN_L is #pragma HLS PIPELINE'd at ~1 cycle/word, so
// the added worst-case latency is ~7168 extra cycles (~71.7us at NN_axi's
// 100MHz clock_period), a genuinely free-running, tProc-decoupled cost
// (NN_axi processes its own AXI stream independently of tProc's next-shot
// timing) -- not expected to interact with the >=100us final_delay
// already used between shots in every established test procedure. Value
// chosen with real margin beyond the ~2000-word backlog already observed
// failing in csim (not just barely clearing the smallest failing case),
// pending real-hardware confirmation of the actual worst-case backlog via
// the new probes at this larger budget.
//
// REVERTED 8192->1024, 2026-08-23: the "pending real-hardware confirmation"
// above was never actually completed -- every build carrying 8192 (qick20
// through qick28, spanning timing quality from thousands of failing
// endpoints down to a clean ddr4/93-endpoint incremental-compile build,
// WNS=-0.012ns, better than qick19 itself) has shown the IDENTICAL ~50%
// coin-flip real-hardware signature, independent of timing quality --
// ruling out timing as the cause. Direct on-board `onset_threshold` sweep
// (0/40/100/300/1000/3000/10000/30000/60000/70000, repeated-identical-shot
// diagnostic) found: (a) `onset_threshold` had been silently left at its
// hardware POR default of 0 in every test this session (a testing-harness
// gap, not a firmware bug -- see CURRENT_BEST_RESULTS.md); (b) restoring
// the correct calibrated value (40) alone did not fix fidelity (still
// ~50%) -- low thresholds trigger on backlog-noise before reaching real
// signal, exactly as this file's own comment above already documented from
// an earlier hardware test; (c) forcing threshold high enough to guarantee
// DRAIN_L always exhausts the full budget (matching qick18/19's own
// always-exhausts-1024 behavior) produces PERFECTLY reproducible output
// (prob_std=0.0000, matching qick18's own signature) but the SAME constant
// probability for two different shots with different true labels and
// different real input data -- i.e. by word 8192 the real per-shot pulse
// has completely left the stream and DRAIN_L is reading generic idle time,
// not backlog. **8192 overshoots past the real signal entirely on real
// hardware** -- the ~2000-word csim-observed backlog this value was sized
// against does not reflect the real board's actual behavior. qick18/19's
// own 1024-word deterministic fallback is the ONLY configuration ever
// empirically confirmed to produce real, varying, correct-correlated
// fidelity (73-80%) on this hardware -- reverting to it here, combined
// with the incremental-compile placement technique (qick28/v1.16, ddr4
// fully clean, WNS=-0.012ns) that this specific value was never tested
// against before. This does NOT re-close the investigation into what the
// true safe backlog value is (that remains open -- see
// CURRENT_BEST_RESULTS.md's "onset_threshold sweep" section for the raw
// data, including a reproducible period-2 52/205-word pattern at low
// thresholds that is not yet understood) -- it returns to the one
// known-good value while that investigation continues.
//
// v1.19 (2026-08-25): ROOT CAUSE FOUND for why DRAIN_L never finds real
// onset at MAX_DRAIN_WORDS=1024 (drain_count_last pegged at exactly 1024
// on every real-hardware shot, every threshold ever tried, qick18 through
// qick29 -- see CURRENT_BEST_RESULTS.md). Live JTAG/ILA capture of
// axis_data_fifo_0's M_AXIS (system_ila_1/SLOT_0_AXIS, qick29 live) during
// genuine idle time confirmed the "backlog" is not quiet noise, it is a
// CONTINUOUS, TVALID=1 stream of exact zero words (0x00000000, every
// sample, across two independent captures) -- and `axis_data_fifo_0`'s own
// BD-configured FIFO_DEPTH is exactly 1024, i.e. IDENTICAL to
// MAX_DRAIN_WORDS. That equality is the bug: FIFO_DEPTH is a hard
// structural bound on how much of this constant-zero backlog can ever be
// queued ahead of the real pulse, but DRAIN_L's entire scan budget was set
// to EXACTLY that bound, leaving zero words of margin to actually search
// INTO the real 770-word signal once the backlog is cleared -- in the
// worst case (FIFO fully backlogged), DRAIN_L's budget is consumed
// entirely by zeros and it never even starts looking at real data, always
// landing on the fixed, misaligned fallback read (this is the direct,
// mechanistic explanation for the ~20-23% of shots that come back
// classified with a probability smeared toward 0.5 rather than
// confidently correct -- confirmed via Python replay of real acquired
// data through the actual trained network: the SAME data classifies
// 100.00% correctly off-chip, so the shortfall is entirely a capture-
// alignment problem, not a signal-quality or model problem).
//
// Fix: MAX_DRAIN_WORDS raised 1024->1152 (FIFO_DEPTH's hard 1024-word
// worst case + 128 words of margin to actually search into the real
// signal). This is deliberately NOT the previous 8192 attempt (8x
// FIFO_DEPTH) that was already proven, via a direct on-hardware
// onset_threshold sweep (2026-08-23/24), to overshoot past the real
// per-shot pulse entirely -- 128 words is chosen from the real,
// already-observed onset position within genuine acquired data (this
// session's own csim check found onset at word 64 into the real 770-word
// signal; the pre-existing NN_axi_test.cpp comment independently
// documents "peak-magnitude position within the first ~13 real samples of
// essentially every shot") -- comfortable headroom above the observed
// range without repeating the previous large overshoot. csim-validated
// against real acquired data at synthetic backlog padding up to 1024
// (FIFO_DEPTH's own hard maximum) before this change went in -- see
// csim_v119_pad*.log. RTL cosimulation was attempted as an additional
// cross-check but is NOT available for this design: Vitis HLS's
// `cosim_design` only supports `ap_ctrl_none` designs that are
// combinational, II=1-pipelined, or all-stream-interface -- NN_axi's
// `out` (BRAM) and AXI-lite scalar ports disqualify it on all three
// counts (confirmed via a direct `cosim_design` attempt, error COSIM
// 212-345, not a limitation of this specific build). This is a genuine,
// previously-untested tooling constraint for this design, not something
// this fix could have passed or failed -- csim against real acquired
// data and (pending a future session's time/lab access) real-hardware
// confirmation remain the available verification paths.
static const unsigned MAX_DRAIN_WORDS = 1152;

// v1.38 (2026-09-21): how many stale words IDLE_FLUSH_L may discard per
// FOREVER_L iteration. See NN_axi.cpp's IDLE_FLUSH_L comment for the full root
// cause of the one-shot lag; in short, the queue holds exactly one stale shot
// at every trigger (pinned there because the gated stream produces exactly as
// many words per shot as the core consumes per trigger), DRAIN_L is
// level-triggered, and so it latches the stale pulse every time.
//
// This is a PER-ITERATION budget, not a drain-to-empty. FOREVER_L must keep
// sampling `trigger` promptly -- the v1.9.x history in NN_axi.cpp is a long
// lesson in what happens when this loop's control flow drifts away from that
// read -- so the flush is spread across iterations instead of blocking one.
// 64 words/iteration clears a full 770-word stale window in ~12 trips, orders
// of magnitude inside the ~100us between shots, while keeping the worst-case
// added latency before the trigger is next sampled at ~64 cycles (0.64us at
// 100MHz).
//
// Deliberately NOT sized against axis_data_fifo_0's FIFO_DEPTH (1024): the
// v1.14 note above records that reasoning from FIFO_DEPTH was wrong once
// already. This loop cannot overshoot into live data whatever the depth is --
// it stops on `*trigger` going high, and on the first empty cycle.
static const unsigned IDLE_FLUSH_PER_ITER = 64;

// v1.26 (2026-09-07): diagnostic input-capture arming value.
//
// Written to the `window_offset` AXI-lite register to arm a full dump of
// the classifier's ACTUAL input window into the BRAM for one shot. Chosen
// deliberately as a distinctive non-zero magic rather than a plain 1, so
// that a stale or accidentally-nonzero window_offset can never silently
// put the core into capture mode and corrupt a normal scored run.
//
// Why window_offset: it is declared and s_axilite-wired but has NEVER been
// read by the datapath (dead port since v1.x -- confirmed by direct source
// inspection 2026-09-07). Reusing it means the capture build needs no BD
// change and no change to the deployed NNQickSoc driver class -- the
// existing soc.nn_configure(window_size, window_offset, scaling_factor)
// arms it and the existing soc.nn_read_bram() reads the result back.
//
// Motivation: as of 2026-09-07 every on-chip fidelity investigation has
// been black-box (fire shots, read one sigmoid, infer). Phase-0 register
// diagnostics that same day ruled out the leading "wrong capture window"
// hypothesis -- drain_count_last == 1 on every dataset, 100%-scoring and
// 50%-scoring alike -- which leaves the window CONTENTS as the thing that
// actually has to be observed rather than inferred.
static const unsigned CAPTURE_ARM = 0xC0DE;

// v1.27 (2026-09-08): bit-transport for the capture path.
//
// v1.26 captured by writing BOTH halves of each stream word in one LOAD_L
// iteration (out[j] = v0; out[j+1] = v1). That is TWO writes to the `out`
// BRAM interface per iteration, and HLS responded by inferring a DUAL-PORT
// interface for it -- the v1.26 csynth report shows out_r_EN_A/WEN_A *and*
// out_r_EN_B/WEN_B. But the BD gives NN_axi only ONE port: `out` lands on
// /axi_bram_ctrl_0_bram, a 2048x32 blk_mem_gen whose other port belongs to
// axi_bram_ctrl_0 for PS readback. So every port-B write went nowhere.
//
// Measured on real hardware (qick38, 2026-09-08): every EVEN capture slot
// read back as the bit-identical constant 0x3B800000 while every ODD slot
// held correct data -- exactly half the writes silently discarded. csim
// could never catch this: C simulation does not model BRAM port counts,
// which is why v1.26's "CAPTURE VERIFICATION PASSED" was worthless here.
//
// Fix: write the RAW 32-bit stream word once per iteration instead of two
// derived halves. One write => one port => no dual-port inference. The
// float BRAM word carries the bits verbatim via this union rather than a
// numeric conversion, so nothing is rounded and I/Q still come from the
// SAME shot. The host unpacks I = w & 0xFFFF, Q = w >> 16 and multiplies by
// scaling_factor to recover in_local exactly -- and because the chip's own
// sigmoid is captured alongside, an offline replay that matches it confirms
// the scaling multiply too, so nothing is lost by capturing pre-scale.
union capture_word_t {
    unsigned u;
    float f;
};

#ifndef __SYNTHESIS__
// Simulation-only bound on NN_axi's free-running loop (see NN_axi.cpp). Not used in hardware.
// Bumped from 20 to 500 to replay the full acquired smoketest dataset
// (real_packed_words.bin / real_labels.bin, see NN_axi_test.cpp) through
// csim/cosim -- guarded by __SYNTHESIS__ so this has zero effect on the
// actual synthesized hardware, which really does loop forever.
static const unsigned SIM_MAX_ITERS = 500;
#endif

// v1.41 (task 3.4, 2026-10-06): the input carries TLAST. axis_avg_buffer_0's
// readback (m1_axis) marks the end of every DMA transfer with TLAST (the S2MM
// DMA needs it), and axis_broadcaster_clean / axis_data_fifo_clean already pass
// it through (HAS_TLAST=1). Through v1.40 NN_axi ignored it, so its 770-word
// windows were cut back to back from the stream at an arbitrary phase -- the
// "rotated window" of CURRENT_BEST_RESULTS 2026-10-06. LOAD_L now restarts its
// window whenever TLAST arrives before the window is full, so every window
// begins at a transfer start. hls::axis<..,0,0,0> also carries TKEEP/TSTRB
// (unused here); the FIFO has none, so IP integrator ties them off.
typedef hls::axis<ap_uint<32>, 0, 0, 0> in_word_t;
typedef hls::stream< in_word_t > input_axi_t;
typedef float output_axi_t;
typedef float output_t;

// v1.15 (2026-08-22): the v1.14 diagnostic probes (max_onset_magnitude_last/
// magnitude_at_drain_end) that briefly lived here have been REMOVED again.
// They were purely observational -- never required for the actual fidelity
// fix below -- and their addition happened to grow NN_axi's AXI-lite
// address space from 7 to 8 bits (s_axi_config_ARADDR/AWADDR), the only
// thing that changed outside NN_axi itself between the last-known-good
// v1.13 netlist (qick19, 106 failing endpoints) and v1.14 (which placed
// catastrophically worse across 3 independent placement directives despite
// a near-identical netlist -- 3259-7745 failing endpoints, all outside
// NN_axi). Device-wide utilization and interconnect cell counts were
// checked and ruled out as the mechanism (negligible deltas), leaving the
// address-width bump's effect on net topology/fanout (not caught by a
// cell-count check) as the remaining untested hypothesis. Removing the
// probes reverts the interface to v1.13's exact original signature/width
// while keeping the actual fix (MAX_DRAIN_WORDS 1024->8192, above) --
// isolates whether the interface change was the placement-fragility
// trigger, and ships a smaller, strictly-necessary diff either way.
void NN_axi(input_axi_t &in, output_axi_t out[BUFFER_SIZE], volatile bool *trigger, volatile unsigned *window_size, volatile unsigned *window_offset, volatile unsigned *scaling_factor, volatile unsigned *out_reset, unsigned *out_offset, unsigned *trig_count, volatile unsigned *onset_threshold, unsigned *drain_count_last, unsigned *onset_miss_count);

#endif
