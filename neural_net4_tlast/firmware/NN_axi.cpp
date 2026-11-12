#include "NN_axi.h"

void NN_axi(input_axi_t &in, output_axi_t out[BUFFER_SIZE], volatile bool *trigger, volatile unsigned *window_size, volatile unsigned *window_offset, volatile unsigned *scaling_factor, volatile unsigned *out_reset, unsigned *out_offset, unsigned *trig_count, volatile unsigned *onset_threshold, unsigned *drain_count_last, unsigned *onset_miss_count) {
    // window_size/window_offset/scaling_factor/out_reset are all polled inputs
    // read fresh every FOREVER_L iteration -- volatile is required here.
    // Without it, Vitis HLS's default scheduling is free to (and, confirmed
    // via direct RTL inspection of the non-volatile build, DID) hoist the
    // read+compare of a polled scalar into a one-time "preheader" FSM state
    // that the loop-back edge never revisits, permanently freezing the
    // comparison at whatever the port's value was on the very first pass
    // (typically 0 at cold start). This is exactly what happened to
    // OUT_RESET_C's "(*out_reset)==255" check: cmp_reg_713 was registered
    // only in ap_ST_fsm_state2, a state entered once at reset and never
    // reached again via the state10->state3 loop-back, so nn_reset() could
    // never work regardless of what value software wrote. out_offset is
    // excluded -- NN_axi only ever WRITES it, never reads it back, so there
    // is no stale-read risk for that port.
    //
    // `trigger` needs the SAME treatment (2026-08-13, v1.8.0 regression
    // hunt): v1.8.0's fix (see `got_trigger` below) reliably fixed the
    // k/trig_count silent-overwrite bug -- confirmed via direct RTL
    // inspection, trig_count_local_fu_164/k_fu_168's commits are gated only
    // on their own FSM states, no `trigger` reference in that path. But
    // qick14 (the resulting bitstream) then showed a NEW, worse failure:
    // total non-response, trig_count/out_offset never increment even on a
    // single manually-fired shot. Live JTAG capture (internal FSM probes,
    // u_ila_v18diag) showed the idle poll loop cycling completely normally,
    // `trigger` correctly going high inside NN_axi on-schedule -- then the
    // FSM enters TRIGGER_C/LOAD_L (state6) and FREEZES there permanently,
    // never completing. Direct RTL re-inspection of the regenerated
    // NN_axi.v found the mechanism: `trigger` being a plain (non-volatile)
    // `bool` parameter left Vitis HLS free to treat every textual read of
    // it as interchangeable -- and it inserted a SECOND, independent live
    // read of the raw `trigger` net (`trigger_read_read_fu_184_p2 =
    // trigger`, purely combinational) to gate
    // grp_NN_axi_Pipeline_LOAD_L_fu_292_ap_start_reg (LOAD_L's own start
    // pulse) and several downstream state9 conditions -- despite
    // `got_trigger` already having sampled `trigger` once, explicitly, right
    // below. This is the exact same class of hazard the k/trig_count fix
    // targeted, just relocated to a different point in the schedule that
    // survived because only the COMMIT logic was hardened, not the
    // parameter's own C++ type. `volatile` is the correct fix: it forbids
    // HLS from eliminating, duplicating, or re-deriving any read of a live
    // port beyond exactly the ones written in the source -- the single
    // `bool got_trigger = trigger;` below becomes the ONLY authorized
    // hardware read of this port, and every other use must go through
    // `got_trigger` (an ordinary, non-volatile local, which HLS is free to
    // register/carry normally without collapsing it back to the live wire).
    //
    // v1.9.0 (`volatile bool trigger`, deployed as qick15) turned out to be
    // NECESSARY BUT NOT SUFFICIENT (2026-08-14, THIRD regression hunt).
    // qick15 showed the exact same total-non-response symptom as qick14.
    // External JTAG capture on ila_trig again confirmed the physical signal
    // path is clean (trig_4_o/trig_out both pulse correctly). A fresh
    // diagnostic bitstream (u_ila_v190diag2) with probes on the FULL
    // trigger-observation chain caught the smoking gun directly: the single
    // `got_trigger = trigger` read (`trigger_assign_fu_180` in the
    // regenerated RTL) was gated ONLY on `ap_CS_fsm_state1` -- and
    // exhaustive analysis of the FSM's next-state case statement proved
    // state1 is reachable ONLY via the power-on reset assignment
    // (`ap_CS_fsm <= ap_ST_fsm_state1`); nothing in the loop-back path ever
    // transitions back into it. So `volatile` correctly produced exactly
    // ONE authorized read -- but Vitis HLS scheduled that one read into a
    // one-time "prologue" state that never recurs, instead of the genuine
    // per-iteration loop body (proven, by the same transition table, to be
    // states 2->3->4->5->(6->7->8->)9->2->...). The live port was sampled
    // exactly once, at cold boot (reading 0, since the trigger line is idle
    // at power-on), and `got_trigger_reg_770` faithfully re-copied that
    // frozen 0 every single loop pass thereafter, forever.
    //
    // This is the SAME general failure class described in the
    // window_size/out_reset comment above (a polled read hoisted into a
    // one-time preheader) -- but concretely, THIS build's RTL shows
    // out_reset's own comparison (`icmp_ln105_fu_333_p2`, the `(*out_reset)
    // == 255` check on line ~105 below) IS correctly gated on `state3`,
    // which the transition table confirms DOES recur every loop iteration
    // -- so out_reset/nn_reset() is fine; the bug is specific to `trigger`.
    //
    // THREE follow-up fix attempts -- reordering the read before
    // OUT_RESET_C; wrapping it in an explicitly `#pragma HLS PIPELINE`'d
    // 1-trip loop; making `got_trigger` loop-carried (declared above
    // FOREVER_L, matching `k`/`trig_count_local`'s demonstrably-working
    // shape) -- were EACH RTL-verified (cheap HLS-only csynth pass, no
    // Vivado needed each time) to NOT change the scheduling: `trigger_
    // assign_fu_180 <= trigger;` stayed gated on `ap_CS_fsm_state1` alone,
    // completely unchanged, across all three. This is strong evidence the
    // C++ loop-body shape was never the actual lever -- `trigger_assign_
    // fu_180` looks like an artifact of how Vitis HLS synthesizes a plain
    // scalar `ap_none` PARAMETER's interface protocol for an `ap_ctrl_none`
    // design (sampled once per "invocation", and this design only ever
    // "invokes" once), not something loop restructuring can reach.
    //
    // v1.9.1 attempt #4: `#pragma HLS LATENCY min=1` removed, to test
    // whether it was forcing this single-invocation-style scheduling for
    // `trigger`'s read. RTL-verified: NO CHANGE -- `trigger_assign_fu_180`
    // still gated on `ap_CS_fsm_state1` alone (out_reset's state3-gated
    // scheduling was unaffected either way, confirming it never depended on
    // this pragma). Restored below since it made no difference and there's
    // no reason to leave it removed.
    //
    // Six independent C-level restructurings (v1.9.1 attempts #1-6: reorder;
    // explicit PIPELINE wrapper; loop-carried variable; removing this
    // pragma; the `ap_none register` INTERFACE keyword; merging the read
    // into OUT_RESET_C's else branch) ALL produced the exact same result --
    // a one-time-only capture in whichever early FSM state HLS chose,
    // always unreachable after the first pass. That convergence was itself
    // the real finding: this was never a loop-scheduling accident reachable
    // by C-source shape, and not fixable by touching the CONSUMER side of
    // `trigger`'s value at all.
    //
    // RESOLVED (v1.9.2, 2026-08-14): root-caused via a fast standalone
    // minimal-repro HLS project (no NN, no BRAM/AXI complexity --
    // ~20s/synthesis instead of ~25-30min), which reproduced the identical
    // bug with a trivial trigger-polling loop PLUS a genuine AXI-stream
    // read (forcing the same rolled multi-state FSM this design needs).
    // The fix: change `trigger` from a plain by-value `volatile bool`
    // PARAMETER to a `volatile bool *` POINTER (dereferenced as
    // `*trigger`, see the read site below, now inside OUT_RESET_C's else
    // branch). This alone changes how Vitis HLS treats the read -- from
    // "marshal the parameter into a captured register at invocation" (the
    // state1-only bug) to a genuine combinational `assign` wire, freely
    // re-evaluable wherever referenced -- the exact same category of
    // access `out_reset` already gets via `*out_reset`. Confirmed via the
    // minimal repro's RTL: the read becomes a plain `assign ... = trigger;`
    // (no register at all), consumed directly inside the recurring loop-
    // body state, with the SAME `ap_none` interface mode preserved at the
    // RTL boundary (RTGEN log unchanged: "Setting interface mode on port
    // 'trigger' to 'ap_none'") -- no BD/pin-level change needed. `volatile`
    // still protects `*trigger` from elimination/duplication/re-derivation,
    // same as before, just expressed through pointer semantics.
    #pragma HLS LATENCY min=1

    // Unregistered axis
	#pragma HLS INTERFACE axis off port=in
	// Registered axis
	//#pragma HLS INTERFACE axis register both port=in
    // depth must be a literal here -- Vitis HLS doesn't reliably resolve a
    // named constant (e.g. BUFFER_SIZE) in this pragma field, and silently
    // falls back to depth=1 instead of erroring. Keep this in sync with
    // BUFFER_SIZE in NN_axi.h by hand.
    #pragma HLS INTERFACE bram depth=16384 latency=1 port=out
    // v1.9.1 attempt #5 (2026-08-14): explicit `register` keyword added.
    // All four prior C-level restructuring attempts (see the top-of-
    // function comment) left `trigger_assign_fu_180`'s gating completely
    // unchanged -- always `ap_CS_fsm_state1` alone -- pointing at interface
    // synthesis, not loop scheduling, as the actual lever. Testing whether
    // explicitly requesting a registered ap_none port changes how/when
    // Vitis HLS clocks that register (as opposed to whatever default mode
    // was already implicitly in effect).
    #pragma HLS INTERFACE ap_none register port=trigger
	#pragma HLS INTERFACE s_axilite register port=window_size bundle=config
	#pragma HLS INTERFACE s_axilite register port=window_offset bundle=config
    #pragma HLS INTERFACE s_axilite register port=scaling_factor bundle=config
	#pragma HLS INTERFACE s_axilite register port=out_reset bundle=config
    #pragma HLS INTERFACE s_axilite register port=out_offset bundle=config
    #pragma HLS INTERFACE s_axilite register port=trig_count bundle=config
    // v1.11: software-tunable onset-detection magnitude threshold (see the
    // MAX_DRAIN_WORDS comment in NN_axi.h). Same proven pointer/s_axilite
    // pattern as scaling_factor/out_reset -- adding it to the EXISTING
    // `config` AXI-lite bundle needs no new BD-level wiring, just a wider
    // address map on the interface that's already there.
    #pragma HLS INTERFACE s_axilite register port=onset_threshold bundle=config
    // v1.11 diagnostics -- same write-only pattern as out_offset/trig_count
    // (see those pragmas/comments above): no stale-read risk, so no
    // volatile needed on these two.
    #pragma HLS INTERFACE s_axilite register port=drain_count_last bundle=config
    #pragma HLS INTERFACE s_axilite register port=onset_miss_count bundle=config
    // v1.15 (2026-08-22): the v1.14 max_onset_magnitude_last/
    // magnitude_at_drain_end probe pragmas that lived here have been
    // removed -- see NN_axi.h for why (address-width-bump isolation test).
    #pragma HLS INTERFACE ap_ctrl_none port=return


	// I/O buffers of the hls4ml NN module
    input_t in_local[N_IQ_WINDOW_IN*2];
    // v1.36 THE FIX. Without this pragma HLS infers `array_partition cyclic
    // factor=3` from LOAD_L's PIPELINE (see the HLS log, [HLS 214-270]) --
    // a NON-power-of-two, which forces it to instantiate
    // NN_axi_urem_11ns_3ns_2_15_1, a 15-CYCLE integer remainder computing
    // j % 3 for the bank select, inside a loop running at II=2. 15 cycles at
    // II=2 is ~7.5 iterations, and #27 measured the anomaly at exactly the
    // last 4 iterations / 8 words: in_local[1532..1539] holds the CURRENT
    // shot while in_local[0..1531] holds the PREVIOUS one, splicing the
    // dense's input on every shot.
    //
    // 1540 = 4 * 385, so cyclic factor=4 makes the bank select j & 3 and the
    // address j >> 2 -- both free, no divider at all. Four banks also still
    // cover the three concurrent readers (the dense's data[ir] plus
    // DENSE_IN_TAP's two) that made HLS want a partition in the first place.
    #pragma HLS ARRAY_PARTITION variable=in_local cyclic factor=4 dim=1
    result_t out_local[N_OUT];
    // v1.39: the v1.28-v1.36 debug register files (dbg2/dbg4/dbg5/dbg6/dbg10,
    // dbg_in, raw_rb) are GONE, along with NN()'s six debug ports. They did
    // their job -- the layer taps localised the fault to q_dense_1 and
    // exonerated the BatchNorm multiply, the ap_fixed<16,6> wrap and the
    // sigmoid LUT; the in_local readbacks found the 49,280-bit register's
    // lost writes; the RAW probe settled the one-shot lag. All of those
    // investigations are closed, and DENSE_IN_TAP alone cost 96 cycles
    // against the dense kernel's 2.
    // v1.13 footprint fix (2026-08-17): investigated ARRAY_PARTITION as a
    // way to shrink LOAD_L's dynamic-index write logic (six ~49,280-bit
    // and/or/xor/shl operations, ~33K LUT, confirmed via
    // NN_axi_Pipeline_LOAD_L_csynth.rpt) for writing in_local[j]/[j+1]
    // inside its loop. Three variants tested empirically (csim+csynth,
    // each RTL-verified, no Vivado needed):
    //   1. `complete` dim=1: DID cut LOAD_L's own write-side LUT (33463->
    //      22798) and FF (147954->46729), but pushed Vitis HLS's resource
    //      binder to replicate the scaling multiply into 724 dedicated
    //      DSP48 instances instead of sharing ~2 across iterations, AND
    //      introduced a ~13.4K-LUT "Multiplexer" cost elsewhere (the
    //      reassembly network gathering 1540 now-individually-partitioned
    //      registers back into the single bus NN()'s input port expects)
    //      that almost exactly canceled the write-side savings.
    //   2. `cyclic factor=2` dim=1 (matching LOAD_L's actual 2-elements/
    //      cycle write granularity): avoided the DSP replication (stayed
    //      at 2-4) but gave essentially NO LUT benefit (33463->33483) --
    //      the write-side cost turned out to scale with the NUMBER of
    //      possible target positions per bank (770, unchanged by a 2-way
    //      split), not the bit-width being shifted.
    //   3. `complete` dim=1 + an explicit ALLOCATION limit on the multiply
    //      (forcing the binder back to sharing ~2 instances): fixed the
    //      DSP problem, but the ~13.4K-LUT reassembly-network cost turned
    //      out to be inherent to `complete` partitioning itself, not the
    //      DSP replication -- so it was present here too, and the NET
    //      top-level LUT change across all three was negligible (174634->
    //      174511, -0.07%) despite real internal restructuring.
    // CONCLUSION: for this design (one array, written piecewise across a
    // pipelined loop, then read all-at-once by a single monolithic
    // consumer), ARRAY_PARTITION does not deliver a net LUT reduction --
    // it only relocates the "pack into one bus" cost from write-time to
    // read-time. NOT keeping any ARRAY_PARTITION/ALLOCATION pragma here as
    // a result. The merge-elimination restructuring above (moving the
    // onset-word write into LOAD_L's own loop) remains a genuine,
    // side-effect-free win on its own: FF 254023->~155000 (confirmed via
    // the cyclic=2 variant, which left LOAD_L's own synthesis essentially
    // unchanged from unpartitioned, isolating fix 1's effect), zero LUT/
    // DSP cost, csim-verified 100.00% (500/500) unchanged throughout.

    // Index of the output buffer over AXI-lite / MMIO
    unsigned k = 0;

    // Latched trigger observation (2026-08-14 regression hunt -- see the
    // full story at the read site below, currently inside OUT_RESET_C's
    // else branch). Declared here, outside FOREVER_L and only ASSIGNED
    // (never re-declared) inside the loop, matching `k`/`trig_count_local`'s
    // shape -- this alone did NOT fix the scheduling (v1.9.1 attempt #3),
    // but is kept since it's still the semantically correct shape for a
    // loop-carried observation, regardless of which attempt ends up fixing
    // the actual FSM-state placement.
    bool got_trigger = false;

    // Diagnostic-only counter (2026-08-10 trigger catch-rate investigation):
    // counts every time FOREVER_L observes trigger==1 and enters TRIGGER_C's
    // body, REGARDLESS of whether that body goes on to complete a
    // classification. Compared against out_offset (which only advances on a
    // *completed* classification), this splits the catch-rate mystery in
    // two: if trig_count tracks out_offset closely, whatever's being lost is
    // lost BEFORE reaching this port (trig_4_o generation, the CDC crossing
    // in trig_edge_oneshot, or the physical trigger() call upstream). If
    // trig_count is much higher than out_offset, NN_axi IS seeing most/all
    // triggers, and the loss is happening downstream, inside this function
    // (most likely LOAD_L's in.read() stalling on an AXI stream that doesn't
    // have a full window ready). Kept as a LOCAL variable and only ever
    // WRITTEN to *trig_count -- never read back -- for the same reason
    // out_offset does this (see the volatile comment above): a write-only
    // port has no stale-read risk regardless of scheduling.
    unsigned trig_count_local = 0;

    // v1.11 diagnostic-only counter (2026-08-15 onset-detection fix, see
    // MAX_DRAIN_WORDS comment in NN_axi.h and DRAIN_L below): counts shots
    // where DRAIN_L exhausted MAX_DRAIN_WORDS without ever crossing
    // *onset_threshold -- the fallback path. Compare against trig_count
    // after a run to see what fraction of shots needed it (ideally 0, or
    // very low). Same write-only, no-stale-read-risk reasoning as
    // trig_count_local above.
    unsigned onset_miss_count_local = 0;
    unsigned tlast_restart_total = 0;   // v1.41: windows restarted on an early TLAST

    // v1.38 diagnostic: how many stale words IDLE_FLUSH_L discarded since the
    // last shot. Loop-carried (declared here, only ASSIGNED inside FOREVER_L),
    // the same shape as trig_count_local/k -- see the v1.9.x story above for
    // why anything conditionally written deep inside TRIGGER_C and merged back
    // at the loop-back point is the shape to avoid. Reported in the upper half
    // of drain_count_last and cleared per shot.
    unsigned idle_flush_total = 0;

#ifndef __SYNTHESIS__
    // Simulation-only: bound the free-running loop so csim can return and the
    // testbench can inspect results. __SYNTHESIS__ is defined by Vitis HLS
    // during synthesis, so real hardware still gets the true infinite loop.
    unsigned sim_iters = 0;
#endif

    // Always active
    FOREVER_L: do {

        // Sample the live `trigger` port EXACTLY ONCE per FOREVER_L
        // iteration. `volatile` (function signature) guarantees this is the
        // ONLY authorized hardware read of `trigger` -- but it does NOT by
        // itself guarantee WHICH pipeline state that read lands in.
        //
        // v1.9.1 attempt #1 (2026-08-14): moving this line to be the
        // textually-first statement in the loop (ahead of OUT_RESET_C,
        // hoping to mirror out_reset's demonstrably-correct state3
        // scheduling) did NOT help -- re-verified via a cheap HLS-only
        // csynth pass (no Vivado needed): still gated solely on
        // `ap_CS_fsm_state1`.
        //
        // v1.9.1 attempt #2: wrapping the read in an explicitly
        // `#pragma HLS PIPELINE` 1-trip loop, on the theory that a
        // pipelined construct can't be treated as invocation-only, ALSO
        // did not help -- same result, still state1-only. (Most likely
        // explanation in hindsight: a compile-time trip count of 1 is
        // trivially collapsible, so HLS's front-end probably eliminated
        // the loop structure -- and the pragma along with it -- before
        // scheduling ever saw it.)
        //
        // v1.9.1 attempt #3: comparing directly against `k`/`trig_count_
        // local` -- which DO get correctly, verifiably refreshed every
        // single loop iteration -- found they're LOOP-CARRIED (declared
        // once, above FOREVER_L, each iteration's value a function of the
        // previous). Making `got_trigger` loop-carried too (see
        // `bool got_trigger = false;` near `k`'s declaration) did NOT help
        // either -- RTL-verified, still state1-only.
        //
        // v1.9.1 attempt #4: removed `#pragma HLS LATENCY min=1` (the one
        // remaining function-level lever) -- no change either. Restored.
        //
        // v1.9.1 attempt #5: added the `register` keyword to `trigger`'s
        // `ap_none` INTERFACE pragma. This DID change the generated RTL
        // (a genuinely new always-live shadow register, `trigger_0_data_
        // reg`, appeared) -- but the actual snapshot into the downstream
        // pipeline still only happened once, now in `state2` instead of
        // `state1` -- and state2 turned out to be JUST as unreachable
        // after the first pass (transition table: state1->state2 is the
        // only path in, and the loop-back target is state3, not state2).
        // Kept the `register` keyword anyway (harmless, arguably cleaner),
        // but it alone doesn't fix the actual problem.
        //
        // v1.9.1 attempt #6 (this one): out_reset's comparison is the only
        // proven-working example in this whole file, and the one thing
        // it has that `got_trigger` never did across 5 attempts is that
        // it's evaluated DIRECTLY INSIDE the branch that consumes it,
        // recomputed fresh at the exact point of use, not cached ahead of
        // time and referenced later. Restructuring `got_trigger`'s
        // assignment to live INSIDE the same if/else as OUT_RESET_C --
        // sharing its condition and scheduling context directly, instead
        // of being a separate statement merely adjacent to it -- is the
        // next thing to try.
    	// Reset output buffer over AXI-lite / MMIO
        OUT_RESET_C: if ((*out_reset) == 255) {
            k = 0;
            *out_offset = 0;
            trig_count_local = 0;
            *trig_count = 0;
            onset_miss_count_local = 0;
            tlast_restart_total = 0;
            *onset_miss_count = 0;
            *drain_count_last = 0;
            // No trigger processing during a reset cycle -- out[]/in_local[]
            // are actively being cleared below, so any pulse arriving mid-
            // reset shouldn't be acted on this iteration anyway.
            got_trigger = false;
            // v1.32: RESET_IN_LOCAL_L removed. LOAD_L (now a shift register --
            // see its comment below) is the SOLE writer of in_local[] and
            // overwrites all 1540 entries every shot, so zeroing here changed
            // no observable behaviour; it only added a second write site,
            // which the v1.13 note below shows costs full-width staging/merge
            // copies of a 49,280-bit register.
            RESET_OUT_LOCAL_L: for (unsigned i = 0; i < N_OUT; i++) {
                #pragma HLS UNROLL
                out_local[i] = 0;
            }
            // Actually clear the output BRAM on reset (previously commented
            // out -- see catch-rate/fidelity investigation 2026-08-08).
            // Software's classify_state()-style readback logic searches
            // backward for "the most recent nonzero slot" as a proxy for
            // "did a fresh classification happen"; with this loop disabled,
            // out[] never gets zeroed, so old floats from unrelated past
            // sessions/shots sit here indefinitely and get misread as a
            // fresh (but actually stale/unrelated) result whenever the
            // low-catch-rate trigger path misses the current shot. No
            // UNROLL here -- out[] is a 16384-entry BRAM-mapped array, not
            // registers; fully unrolling would demand 16384 parallel BRAM
            // write ports. A plain pipelined sequential loop clears it in
            // ~BUFFER_SIZE cycles (~164us at 100MHz), a one-time cost paid
            // only when nn_reset() is actually called, not on the hot path.
            RESET_OUT_L: for (unsigned i = 0; i < BUFFER_SIZE; i++) {
				#pragma HLS PIPELINE
            	out[i] = 0;
            }
        } else {
            // v1.9.1 attempt #6 (merging this assignment into OUT_RESET_C's
            // else branch) did NOT fix it either -- RTL-verified, still a
            // one-time-only capture (state2 this time, equally unreachable
            // after reset). Kept this structural position anyway since it's
            // harmless and semantically sensible (no trigger processing
            // during a reset cycle), but it wasn't the actual fix.
            //
            // v1.9.2 attempt #7 (THE fix, found via a fast standalone
            // minimal-repro HLS project -- ~20s/attempt instead of
            // ~25-30min on the full design): changed `trigger` from a
            // plain by-value `volatile bool` parameter to a
            // `volatile bool *` POINTER, dereferenced here as `*trigger`.
            // Confirmed via the minimal repro's RTL: this alone changes
            // Vitis HLS's treatment of the read from "marshal into a
            // captured register at invocation" (the state1-only bug, which
            // reproduced IDENTICALLY in the minimal case) to a genuine
            // combinational `assign ... = trigger;` -- the exact same
            // category of live, freely-re-evaluable access `out_reset`
            // already gets via `*out_reset`. Confirmed gated on the
            // recurring loop-body state in the minimal repro (not a
            // one-time state), with the same `ap_none` interface mode
            // preserved at the RTL boundary (no BD/pin-level change
            // needed). `volatile` on the pointee (`volatile bool *`, not
            // `bool * volatile`) still means each `*trigger` dereference is
            // a distinct, unelidable access -- same protection as before,
            // just expressed through pointer semantics instead of by-value
            // parameter semantics.
            got_trigger = *trigger;

            // v1.38 ONE-SHOT LAG FIX (2026-09-21). Keep the input queue EMPTY
            // while idle, so a trigger can only ever see its own shot's data.
            //
            // Root cause, measured on qick49/v1.37 this session:
            //   * the readable result lags by exactly ONE FIRE -- model-free
            //     cross-correlation over a 400-fire interleaved sequence:
            //     lag 0 = 55.6%, lag 1 = 94.5%, lag 3+ = chance.
            //   * NOT a host readback race: the value is constant across
            //     260ms of polling and trig_count advances exactly +1 per
            //     fire, so the core really did compute the previous shot. (An
            //     earlier session "ruled out" a race with a determinism check;
            //     that check cannot see a SYSTEMATICALLY early read, which
            //     returns the stale value every time and passes it.)
            //   * NOT fixable by trigger timing: sweeping the NN trigger from
            //     0.31us to 80us (80% of the inter-shot period) leaves
            //     drain_count_last pinned at 1 and y[i] at chance throughout.
            //   * NOT fixable with a second, earlier "flush" trigger: the
            //     tProc only ever delivers one NN trigger per program --
            //     trig_count advanced by exactly 1 per fire at every tested
            //     separation out to 300us.
            //   * the stream is GATED at exactly N_IQ_WINDOW_IN words per
            //     shot with no idle filler -- the armed capture's 770 words
            //     are one clean contiguous pulse (14-25 quiet words, longest
            //     quiet run 1-4, zero leading quiet). Production (770/shot)
            //     therefore EQUALS consumption (770/trigger), which is exactly
            //     why the queue depth is pinned at one shot forever instead of
            //     draining or growing.
            //
            // So at every trigger the FIFO already holds the PREVIOUS shot's
            // complete window, and DRAIN_L is LEVEL-triggered: it breaks on
            // the first word above *onset_threshold, which is that stale
            // pulse's first sample (hence drain_count_last == 1 on every
            // shot). Stale and fresh pulses have IDENTICAL amplitude, so no
            // threshold can separate them -- the direct explanation for every
            // onset_threshold sweep in NN_axi.h's history failing, at every
            // value from 0 to 70000.
            //
            // Draining here, in the not-triggered branch, is the one place
            // where "what is queued" is unambiguously stale: no shot is in
            // flight, so anything present is a leftover. FOREVER_L spins
            // continuously, so across the ~100us between shots this holds the
            // queue at empty. Bounded per iteration (not drained to empty in
            // one pass) so a single FOREVER_L trip stays short and the
            // trigger keeps being sampled promptly -- the v1.9.x history
            // above is a long lesson in not letting this loop's control flow
            // drift away from the trigger read. read_nb() returns false as
            // soon as the FIFO is empty, so the common case costs one cycle.
            IDLE_FLUSH_L: for (unsigned f = 0; f < IDLE_FLUSH_PER_ITER; f++) {
                #pragma HLS PIPELINE
                in_word_t stale_word;  // v1.41: TLAST-carrying stream word
                if (*trigger)          // a shot is starting -- stop immediately
                    break;             // so we cannot eat its leading samples
                if (!in.read_nb(stale_word))
                    break;             // queue already empty: nothing to do
                idle_flush_total++;
            }
        }

        // Root-cause note (2026-08-11): a first attempt just sampled
        // `trigger` into `got_trigger` above and branched on that instead
        // of `trigger` directly -- direct RTL inspection showed Vitis HLS
        // saw through the copy (proved it dataflow-equivalent to `trigger`
        // itself) and still re-referenced the LIVE port a second time, far
        // later, at the FSM's loop-back state -- to decide whether to
        // commit k/trig_count_local's incremented value into their
        // persistent, loop-carried registers. Since trig_edge_oneshot's
        // pulse is only ~200 cycles (~2us) wide -- far shorter than the
        // ~15-20us LOAD_L+NN+STORE_L takes -- the pulse had always already
        // ended by the time of that second read, so the commit almost
        // never happened: the AXI-lite write (gated on the FIRST read,
        // immediately below) fired correctly every time, but the internal
        // "base value used for next time's +1" silently failed to advance
        // on most triggers -- discarding that increment permanently and,
        // worse, meaning STORE_L's write below landed on the SAME output
        // BRAM slot as the previous classification, silently overwriting a
        // fresh, valid result.
        //
        // The actual fix: stop giving the scheduler ANY variable that's
        // conditionally written deep inside a ~1500-cycle branch and only
        // merged back into its loop-carried register at the far-away
        // loop-back point -- that shape is exactly what created room for
        // the bug, regardless of what condition gated the branch. Both
        // `k` and `trig_count_local` are now updated via a single,
        // unconditional, same-cycle select (add 0 or add 1) sitting right
        // next to `got_trigger`'s own definition -- there is no later
        // merge point left for anything to go stale at.
        unsigned trig_count_next = trig_count_local + (got_trigger ? 1u : 0u);
        unsigned k_next = k + (got_trigger ? 1u : 0u);
        if (k_next * N_OUT >= BUFFER_SIZE)
            k_next = 0;
        // Capture the slot THIS shot's result belongs in before committing
        // k_next -- STORE_L must write to the pre-increment slot.
        unsigned k_for_this_write = k;
        trig_count_local = trig_count_next;
        k = k_next;
        *trig_count = trig_count_local;
        *out_offset = k;

        // Trigger for readout data
        TRIGGER_C: if (got_trigger) {

        	// NOTE: an optional ap_wait_n(*window_offset) delay was removed here --
        	// ap_utils.h wasn't available in this Vitis HLS install, and the wait
        	// was documented as optional ("if you need"). window_offset is left
        	// as an unused AXI-Lite register for now.

            // v1.11 (2026-08-15): DRAIN_L discards raw stream words until one
            // crosses *onset_threshold (a magnitude estimate, |I|+|Q|), i.e.
            // until it finds where real signal actually starts in the
            // continuously-flowing decimated stream -- see the MAX_DRAIN_WORDS
            // comment in NN_axi.h for the full root-cause story (this
            // realigns LOAD_L's capture window instead of trusting a fixed
            // trigger-relative offset, which live-capture + an empirical
            // delay sweep both showed isn't reliable shot-to-shot).
            //
            // Deliberately kept UNCONDITIONAL on total word consumption: this
            // loop always executes up to MAX_DRAIN_WORDS iterations (bounded,
            // not data-dependent beyond the early `break`), and LOAD_L below
            // always reads exactly enough additional words to bring the
            // total to N_IQ_WINDOW_IN, regardless of whether an onset was
            // found. This mirrors the k/trig_count_local fix's own lesson
            // (see the ORIGINAL K/TRIG_COUNT BUG comment above and the
            // trigger volatile-pointer story): don't make a downstream
            // commit conditional on a value computed deep inside a long,
            // variable-latency branch. Here the "downstream commit" is
            // trig_count/out_offset/k, which are already committed
            // unconditionally on got_trigger, above -- this loop's outcome
            // (onset found or not) never feeds back into that decision, only
            // into where LOAD_L starts writing into in_local.
            // v1.38 ONE-SHOT LAG FIX (2026-09-21). Root cause, measured on
            // qick49/v1.37 this session:
            //
            //   * the readable result lags by exactly ONE FIRE -- model-free
            //     cross-correlation over a 400-fire interleaved sequence:
            //     lag 0 = 55.6%, lag 1 = 94.5%, lag 3+ = chance.
            //   * NOT a host readback race: the value is constant across
            //     260ms of polling and trig_count advances exactly +1 per
            //     fire, so the core really did compute the previous shot.
            //     (An earlier session "ruled out" a race with a determinism
            //     check; that check cannot see a SYSTEMATICALLY early read,
            //     which returns the stale value every time and passes it.)
            //   * NOT fixable by trigger timing: sweeping the NN trigger from
            //     0.31us to 80us (80% of the inter-shot period) leaves
            //     drain_count_last pinned at 1 and y[i] at chance throughout.
            //   * the stream is GATED at exactly N_IQ_WINDOW_IN words per
            //     shot with no idle filler -- the armed capture's 770 words
            //     are one clean contiguous pulse (14-25 quiet words, longest
            //     quiet run 1-4, zero leading quiet). Production (770/shot)
            //     therefore EQUALS consumption (770/trigger), which is
            //     precisely why the queue depth is pinned at exactly one shot
            //     forever instead of draining or growing.
            //
            // So at every trigger the FIFO already holds the PREVIOUS shot's
            // complete 770-word window, and DRAIN_L below is LEVEL-triggered:
            // it breaks on the first word above *onset_threshold, which is
            // that stale pulse's first sample (hence drain_count_last == 1 on
            // every shot). Stale and fresh pulses have IDENTICAL amplitude,
            // so no threshold value can separate them -- which is the direct
            // explanation for every onset_threshold sweep in NN_axi.h's
            // history failing, at every value from 0 to 70000.
            //
            // Fix: discard whatever is already queued before looking for
            // onset. Because the stream is gated, "queue non-empty at trigger
            // time" means exactly "stale data present", so a non-blocking
            // drain to empty is both sufficient and safe: after it, DRAIN_L's
            // blocking read below waits for THIS shot's words and consumes
            // them live as they arrive. That is self-aligning -- it cannot
            // drift back out of phase the way the host-side workarounds did
            // (a single priming fire was tested and does NOT hold) -- and it
            // is what makes the core genuinely real-time rather than one shot
            // behind.
            //
            // The flush CANNOT live here, inside TRIGGER_C -- that was tried
            // and rejected by analysis before it was ever built. By the time
            // the trigger is observed, the queue holds the stale window AND
            // this shot's incoming burst, and a drain-to-empty cannot tell
            // them apart: the fabric produces at f_fabric = 552.96MHz while
            // NN_axi consumes at 100MHz, so a flush can never outrun an
            // in-flight burst and would simply eat both, leaving DRAIN_L to
            // block for the NEXT shot -- the same lag, or worse.
            //
            // It belongs in the IDLE path instead (see IDLE_FLUSH_L in
            // OUT_RESET_C's else branch): keep the queue empty BETWEEN shots,
            // so that when the trigger does arrive the only thing DRAIN_L can
            // read is this shot's own data, consumed live as it streams in.
            // That is also what makes the core genuinely real-time rather
            // than one shot behind.
            // v1.41 (task 3.4): NO ONSET SKIPPING. The readback stream holds
            // only buffered samples, so a magnitude threshold carries no onset
            // information (CURRENT_BEST_RESULTS 2026-09-22 / 2026-10-06), and
            // with TLAST alignment it is actively harmful: every word skipped
            // here pushes the window across the transfer boundary, where LOAD_L
            // would restart onto the NEXT transfer. The window's first word is
            // simply the first word available. onset_threshold stays in the
            // AXI-lite map (driver compatibility) but is no longer read.
            bool found_onset = true;
            in_word_t onset_w;
            in.read(onset_w);
            unsigned drain_words_used = 1;

            // Diagnostic-only, write-only outputs (same no-stale-read-risk
            // reasoning as trig_count/out_offset -- see that comment above):
            // drain_count_last reports how many words THIS shot's DRAIN_L
            // consumed (software can correlate this with a manually-fired
            // shot to directly see backlog depth, no JTAG capture needed);
            // onset_miss_count_local counts shots where MAX_DRAIN_WORDS was
            // exhausted without finding onset (the fallback path). Committed
            // immediately adjacent to found_onset's own computation, still a
            // single unconditional select -- same shape as the trig_count/k
            // fix above, just necessarily inside TRIGGER_C since found_onset
            // isn't known until DRAIN_L actually runs.
            // v1.38: pack how many words IDLE_FLUSH_L discarded since the
            // previous shot into the upper half, rather than adding an
            // AXI-lite port -- a new port means a BD/wrapper change and a full
            // re-integration, for a diagnostic. Both fields are bounded well
            // under 16 bits (drain <= 1152; the idle flush is saturated at
            // 0xFFFF below). Host decode:
            //     idle_flush = v >> 16;   drain = v & 0xFFFF
            // Pre-v1.38 readers see drain unchanged in the low half, so the
            // existing diagnostics still hold provided they mask.
            //
            // This is the direct success signal for the fix: with the queue
            // held empty between shots, idle_flush should settle at ~0 in
            // steady state and drain should stop being pinned at 1.
            *drain_count_last = ((idle_flush_total > 0xFFFFu ? 0xFFFFu
                                                            : idle_flush_total) << 16)
                              | (drain_words_used & 0xFFFFu);
            idle_flush_total = 0;   // per-shot, not cumulative
            onset_miss_count_local = onset_miss_count_local + (found_onset ? 0u : 1u);
            // v1.41: *onset_miss_count is written after LOAD_L, packed with the
            // TLAST restart count (upper half).

            // Read readout data: if DRAIN_L found an onset word, it's the
            // first sample of the real capture (consumed from the stream
            // already, so LOAD_L below reads the remaining N_IQ_WINDOW_IN-1).
            // If no onset was found within MAX_DRAIN_WORDS (a genuinely
            // quiet/spurious trigger), fall back to reading a full
            // N_IQ_WINDOW_IN starting from wherever DRAIN_L left off --
            // same total word budget either way.
            //
            // v1.12 timing fix (2026-08-16): the previous version varied
            // LOAD_L's own loop-carried start index/write-offset
            // (`i = start_i, j = start_i*2`) at runtime. That forced HLS to
            // generate a RUNTIME-COMPUTED base address for in_local[]'s
            // writes on every iteration, instead of the fully static,
            // compile-time-known per-iteration write target v1.10 had
            // (j always starts at 0, fixed +2/iteration). Confirmed via
            // report_high_fanout_nets on the routed v1.11 checkpoint: a
            // single control signal feeding in_local[]'s ~1540-element
            // write-enable network had 43,968 fanout, routed through a
            // dedicated global clock buffer (BUFGCE) just to physically
            // reach it -- and NN_axi_0's cells ended up scattered across
            // nearly the entire die (X0-118,Y16-479) chasing that
            // distribution network. Two full place/route attempts
            // (AltSpreadLogic_medium, then _high) both failed to recover
            // timing (WNS stuck around -2.5 to -3.0ns, TNS unchanged
            // between attempts) because the problem wasn't congestion
            // intensity -- it was a structurally different (dynamic vs.
            // static) addressing scheme for the same huge register array.
            //
            // Fix: keep LOAD_L's loop bounds AND j's increment pattern
            // fully fixed/compile-time-constant (i=0..N_IQ_WINDOW_IN-1,
            // j=0,2,4,... -- identical to v1.10's original addressing).
            // The only thing DRAIN_L's onset word changes is whether
            // iteration 0 performs a stream read at all -- a single-bit,
            // compile-time-localized skip on i==0 only, not a variable
            // base offset threaded through every iteration's indexing.
            // Functionally identical word-budget behavior to before
            // (N_IQ_WINDOW_IN-1 stream words read when found_onset, full
            // N_IQ_WINDOW_IN otherwise) -- this only changes how the RTL
            // expresses it, not what it does.
            //
            // v1.13 footprint fix (2026-08-17, shrinking NN_axi for timing):
            // the onset-word write used to happen HERE, before LOAD_L, as a
            // separate write site into in_local[0]/in_local[1]. Per direct
            // post-synthesis RTL/resource-report inspection
            // (NN_axi_csynth.rpt + NN_axi_Pipeline_LOAD_L_csynth.rpt),
            // splitting the write into two sites (this one, plus LOAD_L's
            // own loop) forced HLS to materialize TWO extra full-width
            // (49,280-bit) copies of in_local[] on top of LOAD_L's own
            // working copy -- one top-level "settled" register
            // (in_local_0_fu_196) and one top-level "staging/merge" register
            // (in_local_12_reg_287) that reconciled the two write sites
            // before handing off to LOAD_L. Folding the onset-word write
            // into LOAD_L's own loop body (as its i==0 case, below) makes
            // LOAD_L the SOLE writer of in_local[] in every case, so there's
            // nothing left to merge or stage -- functionally identical
            // (same word budget, same values written), purely a scheduling
            // restructure. `found_onset` is used directly in place of the
            // old separate `skip_first_read` flag (they were always
            // identical -- skip_first_read was only ever set when
            // found_onset was true).
            // v1.26 DIAGNOSTIC CAPTURE (2026-09-07): when capture mode is
            // armed, LOAD_L additionally mirrors every value it writes into
            // in_local[] out to the BRAM, so software can read back the
            // EXACT input the classifier saw for this shot. Armed via the
            // window_offset AXI-lite register, which is declared and wired
            // but has never been read by the datapath (dead port since
            // v1.x) -- reusing it means this build needs NO BD change and
            // NO change to the deployed NNQickSoc driver class:
            // soc.nn_configure(w, CAPTURE_ARM, sf) arms it and the existing
            // soc.nn_read_bram() reads it back.
            //
            // The mirror writes deliberately reuse the SAME values and the
            // SAME static j-indexing LOAD_L already computes, rather than
            // reading back out of in_local[] afterwards. Reading the
            // finished in_local[] would mean a 1540:1 mux over a
            // fully-ARRAY_RESHAPE'd register array -- structurally the same
            // dynamic-addressing / huge-fanout pattern that cost this
            // project two failed place/route attempts in v1.11 (see the
            // v1.12 timing-fix note above). Mirroring in-loop keeps
            // addressing static (+2/iteration) and adds only a BRAM write
            // port, not another copy of in_local[].
            //
            // in_local values are integers: as of v1.37 range(15,0) is read
            // SIGNED (ap_int<16>, -32768..32767) times scaling_factor, so at
            // the deployed SF<=8 the magnitude is <= 262144 < 2^24 -- still
            // exactly representable in the float32 BRAM word. The capture is
            // therefore LOSSLESS, not an approximation. NOTE for host-side
            // decoders: in_local now carries NEGATIVE values, so the old
            // uint16 reinterpretation (`to_signed`, which subtracted 65536
            // above 32768) must NOT be applied to v1.37+ captures.
            //
            // v1.32 FUNCTIONAL FIX (2026-09-09) -- "fix D", the shift
            // register. THIS IS THE FIX FOR THE ON-CHIP FIDELITY DEFECT, not
            // another probe. Everything above about in_local[]'s write cost
            // was framed as a TIMING/AREA problem; v1.28-v1.31's taps proved
            // it is also a CORRECTNESS problem, and this loop is the site.
            //
            // What v1.31 measured: in_local[] readback (PRE, POST, and the
            // dense kernel's own pointer -- all three agree, 256/256) matches
            // the BRAM capture only below index ~426 of 1540. Highest varying
            // word index 421, lowest permanently-static 432. The out[] capture
            // written in the SAME iterations is complete, 770/770 fresh, so
            // LOAD_L demonstrably ran every iteration -- it simply failed to
            // land its in_local[] writes above ~426. The dense kernel was
            // therefore summing ~426 live words plus ~1114 stale ones, which
            // is exactly the signature the layer taps found: q_dense_1 is the
            // ONLY broken layer (l2 0/384 exact), while every layer downstream
            // of it is bit-exact when fed the chip's own upstream values
            // (l4 384/384, l5 384/384, l6 96/96, l10 96/96, sigmoid 96/96).
            //
            // The RTL mechanism, read directly out of
            // NN_axi_NN_axi_Pipeline_LOAD_L.v: `in_local` is ONE 49,280-bit
            // register (ARRAY_RESHAPE complete dim=0 propagates in from
            // NN()'s input_1 port), and the old `in_local[j] = v0` used a
            // RUNTIME index. HLS therefore emitted a read-modify-write built
            // from FOUR 49,280-bit VARIABLE barrel shifters per iteration:
            //     shl_ln2_fu_256_p3   = {j, 5'd0}                    // j*32
            //     shl_ln609_fu_268_p2 = 49280'd4294967295 << offset  // mask j
            //     shl_ln610_fu_284_p2 = ...          << (offset|32)  // mask j+1
            //     shl_ln609_1 / shl_ln610_1                          // data
            //     in_local_fu_92 <= (in_local & ~masks) | data
            // A correct 49,280-bit variable shifter is ~16 stages x 49,280
            // muxes, ~790K LUT-equivalents EACH, on a 425,280-LUT device. HLS
            // budgeted 33,355 LUT for all four (NN_axi_Pipeline_LOAD_L_csynth
            // .rpt) and the whole IP synthesised to 168,485 LUT. The low-order
            // lanes of that network survive synthesis; the high-order ones do
            // not. Ruled out along the way: port truncation (in_local_out and
            // NN's p_read are both [49279:0]), a LOAD_L/NN start race (FSM
            // state8 waits on LOAD_L's ap_done before state10 starts NN), and
            // register trimming (the only "trimmed from" warning is unrelated;
            // the zext_ln609/610_reg[6:15] merge is a correct optimisation).
            // This is TIMING_CLOSURE_REFERENCE #4 -- massive fanout from
            // runtime-variable addressing -- showing up as a FUNCTIONAL
            // failure rather than a timing one, which is why routed timing
            // (zero failing clk_pl_0 endpoints) never flagged it, and why
            // csim always passed: csim has one plain array and no register
            // file, so it cannot express this failure mode at all.
            //
            // THE FIX: LOAD_L never actually needs random access. j only ever
            // goes 0, 2, 4, ... strictly sequentially; the demux existed purely
            // because the code EXPRESSED it as an index. Rewritten as a shift
            // register, every index below is a compile-time constant, so the
            // hardware is fixed register-to-register wiring with one common
            // enable -- no shifter, no demux, no mask, zero Expression LUT.
            //
            // ORDERING IS PRESERVED EXACTLY -- this is a bit-identical
            // replacement, NOT a reversal, so no weight, emulator, decoder or
            // tap-index change is needed anywhere. The shift runs DOWN toward
            // index 0 with the newest pair inserted at the TOP: word i is
            // written at 1538 and shifted down twice per subsequent iteration,
            // so after all 770 iterations it sits at 1538 - 2*(769-i) = 2i --
            // precisely where the old in_local[j], j=2i, put it. (Shifting the
            // other way would have been a time reversal needing w2's input
            // axis reversed; it is not necessary and is not done.)
            //
            // RESET_IN_LOCAL_L (which used to zero in_local[] on nn_reset) is
            // deliberately GONE -- see OUT_RESET_C above. It was a SECOND write
            // site into in_local[], and the v1.13 note above records what a
            // second write site costs here: HLS materialises extra full-width
            // 49,280-bit staging/merge copies to reconcile them. With the shift
            // register, all 1540 entries are overwritten by the stream on every
            // single shot regardless of prior contents, so the zeroing was dead
            // code that only bought a 2:1 mux on all 49,280 bits.
            // v1.36 PRELOAD probe -- see CAP_PRELOAD_BASE in NN_axi.h. Reads
            // in_local BEFORE LOAD_L writes it, at the same indices PRE_TAIL
            // re-reads after. PRELOAD == PRE_TAIL means the write never became
            // visible in its own trigger; PRELOAD != PRE_TAIL localises
            // exactly which indices did commit in-pass.
            // v1.39: PRELOAD_L removed with the rest of the v1.36 probes.
            const bool do_capture = (*window_offset == CAPTURE_ARM);
            // v1.41 (task 3.4): TLAST-ALIGNED WINDOWS. If a transfer ends
            // (TLAST) before the window is full, the partial window is
            // abandoned and counting restarts at the next word -- the first
            // word of the next transfer. Every window therefore starts at a
            // DMA transfer boundary, whatever the host reads in between
            // (odd-length transfers, idle-flushed fragments, a stale phase).
            // The shift register needs no clearing: a complete window shifts
            // in 770 fresh pairs, which pushes any abandoned partial out.
            // A transfer LONGER than 770 words yields a window of its first
            // 770; the remainder restarts at its own TLAST.
            unsigned i = 0;
            bool use_onset = found_onset;
            unsigned restarts_this_window = 0;
            LOAD_L: while (i < N_IQ_WINDOW_IN) {
                #pragma HLS PIPELINE
                #pragma HLS LOOP_TRIPCOUNT min=770 max=770
                input_t v0, v1;
                in_word_t w;
                if (use_onset) {
                    w = onset_w;
                    use_onset = false;
                } else {
                    in.read(w);
                }
                ap_uint<32> word = w.data;
                // v1.37 FIDELITY FIX (2026-09-21). These two reads used to be
                // `word.range(15,0) * *scaling_factor`, i.e. ap_uint<16>, which
                // fed the network the ADC's two's-complement samples
                // REINTERPRETED AS UNSIGNED -- a sample of -1 arrived as 65535.
                // 46.6% of samples are negative, so a 65536-wide cliff sat in
                // the middle of the input distribution. DRAIN_L six hundred
                // lines up already read the same word SIGNED (`ap_int<16> pi`)
                // for onset detection, so the design was internally
                // inconsistent. Measured offline on the deployed training
                // domain, ternary 1540->4->1: unsigned 92.35% vs signed 94.42%
                // (3 seeds each), against a linear ceiling of 96.07%.
                // The paper's "raw 14-bit unsigned" is genuine offset binary,
                // which is monotone; reinterpreting two's complement is not.
                // scaling_factor MUST be cast: it is `volatile unsigned`, and
                // signed*unsigned promotes to unsigned in C++, which would
                // silently undo the sign.
                ap_int<16> samp_i = word.range(15,0);
                ap_int<16> samp_q = word.range(31,16);
                v0 = samp_i * (int)(*scaling_factor);
                v1 = samp_q * (int)(*scaling_factor);
                // v1.39: BACK TO v1.32's SHIFT REGISTER, and this is
                // mandatory, not a preference. config2 is at reuse_factor=1
                // again, so NN.cpp carries `ARRAY_RESHAPE variable=input_1
                // complete dim=0` once more, and that pragma propagates
                // outward to make in_local[] a single 49,280-bit REGISTER.
                // A RUNTIME-indexed write into that register is defect
                // #19-#22 exactly: HLS lowers it to four 49,280-bit VARIABLE
                // barrel shifters per iteration, budgets all four at 33,355
                // LUT when one correct shifter is ~790K LUT-equivalents on a
                // 425,280-LUT device, and the high-order lanes do not survive
                // synthesis. Measured consequence on hardware: in_local
                // tracked the stream only below index ~426 of 1540, the dense
                // summed ~426 live words and ~1114 stale ones, and on-chip
                // fidelity sat near 50%. v1.33's plain `in_local[j] = v0` was
                // correct ONLY because rf=1540 had made in_local a BRAM.
                // Returning to rf=1 withdraws that permission.
                //
                // Every index below is a COMPILE-TIME constant, so this is
                // fixed register-to-register wiring with one common enable:
                // no shifter, no demux, no mask, zero Expression LUT.
                //
                // ORDERING IS PRESERVED EXACTLY -- a bit-identical
                // replacement, not a reversal. The newest pair enters at the
                // TOP and the contents shift DOWN two positions per
                // iteration, so the word read at iteration i lands at
                // 1538 - 2*(769 - i) = 2i after all 770 iterations, which is
                // where `in_local[j]`, j = 2i, put it. Weights, the offline
                // emulator, decode_taps.py and every capture slot index are
                // therefore unchanged.
                SHIFT_L: for (int m = 0; m < (int)(2*N_IQ_WINDOW_IN) - 2; m++) {
                    #pragma HLS UNROLL
                    in_local[m] = in_local[m+2];
                }
                in_local[2*N_IQ_WINDOW_IN - 2] = v0;
                in_local[2*N_IQ_WINDOW_IN - 1] = v1;
                if (do_capture) {
                    // v1.27: EXACTLY ONE write to out[] per iteration. Two
                    // writes here made HLS infer a dual-port BRAM interface
                    // whose port B the BD never wires, silently dropping half
                    // the samples on real hardware -- see capture_word_t in
                    // NN_axi.h for the full measurement. Transport the raw
                    // stream word's BITS through the float BRAM word; the
                    // host unpacks and rescales.
                    capture_word_t cvt;
                    cvt.u = (unsigned) word;
                    out[i] = cvt.f;
                }
                if (w.last && i != N_IQ_WINDOW_IN - 1) {
                    i = 0;                       // transfer ended early: restart
                    restarts_this_window++;
                } else {
                    i++;
                }
            }
            tlast_restart_total += restarts_this_window;
            // Host decode: restarts = v >> 16 (cumulative, saturating);
            // onset misses = v & 0xFFFF (always 0 since v1.41).
            *onset_miss_count = ((tlast_restart_total > 0xFFFFu ? 0xFFFFu
                                                                : tlast_restart_total) << 16)
                              | (onset_miss_count_local & 0xFFFFu);

            // v1.39: PRE_TAIL_L / PRE_HEAD_L removed. They answered #24 and
            // #25 (the in_local commit is deferred by one trigger; the tail
            // was never spliced) and nothing downstream reads their slots.

            // hls4ml NN module
        	NN(in_local, out_local);

        	// Output logit (single sigmoid probability). v1.26: in capture
        	// mode the result goes to a FIXED slot just past the captured
        	// window instead of the rolling slot, so one captured shot is a
        	// deterministic, self-contained BRAM layout:
        	//   words [0 .. 2*N_IQ_WINDOW_IN-1] = the exact NN input
        	//   word  [2*N_IQ_WINDOW_IN]        = the sigmoid the chip
        	//                                     produced from that input
        	// = 1541 words, inside the physical 2048-word BRAM. Nothing else
        	// is written during a captured shot, so the capture cannot be
        	// clobbered by the rolling output stream. out_offset still
        	// advances either way, so G3/G4's fresh-core and write-count
        	// guards keep working unchanged.
        	STORE_L: for(unsigned i = 0; i < N_OUT; i++){
            	#pragma HLS PIPELINE
            	if (do_capture) {
            		// v1.27: the capture region is now N_IQ_WINDOW_IN words of
            		// packed I/Q (not 2*N), so the sigmoid moves down to slot
            		// N_IQ_WINDOW_IN. Whole record is 771 words, well inside
            		// the 2048-word physical BRAM.
            		out[CAP_SIG_SLOT + i] = out_local[i];
            	} else {
            		out[k_for_this_write*N_OUT + i] = out_local[i];
            	}
        	}

        	// v1.28: drain NN()'s taps into the capture record. ONE write to
        	// out[] per iteration -- an 18:1 mux over registers, not two write
        	// sites -- so HLS still infers a SINGLE-PORT out[] interface. This
        	// is the v1.26 lesson applied: the csynth port list (no out_r_*_B)
        	// is the gate for this change, and csim cannot check it.
        	// v1.29: read in_local[] BACK, at constant indices, and mirror it
        	// out. Compared against out[0..769] -- written from the same
        	// `word` in the same LOAD_L iteration -- this says directly
        	// whether NN() was fed what the capture recorded. See NN_axi.h.
        	//
        	// The gather is fully unrolled with COMPILE-TIME indices, so each
        	// read is a wire off one register, not a 1540:1 mux. Only the
        	// small 256-entry staging array is addressed at runtime, by the
        	// pipelined drain loop below, which is a cheap 256:1 mux.
        	// v1.34: the v1.29 POST gather is REMOVED. It agreed with PRE
        	// 256/256 on every build from v1.30 on, so it measured nothing PRE
        	// did not, and its 256 BRAM words now go to PRE's wider coverage.

        	// v1.31: drain the dense kernel's own input view. One write to
        	// out[] per iteration, same as every other capture loop, so the
        	// single-port out_r inference (the v1.26 lesson) still holds.
        	// v1.39: RAW_L / DENSE_IN_L / DBG_L removed. The armed capture now
        	// writes exactly two things -- the 770-word input record (out[i]
        	// in LOAD_L) and the sigmoid at CAP_SIG_SLOT (STORE_L). That is
        	// the whole channel score_armed_fidelity.py reads, so scoring is
        	// unaffected; what is lost is the layer-level observability, and
        	// every investigation that needed it is closed.
        }

#ifndef __SYNTHESIS__
        if (++sim_iters >= SIM_MAX_ITERS)
            break;
#endif
    } while (true);
}
