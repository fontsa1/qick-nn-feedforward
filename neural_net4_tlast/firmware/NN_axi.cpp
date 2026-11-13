// NN_axi: the deployed classifier (hls4ml NN(), 2 x 770 inputs) on QICK's buffered readout stream, for comparison
// with NN_live. Runs free (ap_ctrl_none) on the 100 MHz PL clock.
//
// Per trigger: one window of N_IQ_WINDOW_IN words is read from `in` (readout 0's decimated buffer readback,
// TLAST at the end of each transfer), each {Q, I} word multiplied by scaling_factor into the network input, and
// NN() evaluated after the whole window has arrived. The sigmoid output goes to out[out_offset], a ring of
// BUFFER_SIZE slots. An early TLAST restarts the window, so windows start at a transfer boundary.
// Between triggers, up to IDLE_FLUSH_PER_ITER stale words per loop iteration are discarded.
//
// AXI-lite registers (bundle config):
//   out_reset         255 clears the counters and the output buffer
//   window_offset     CAPTURE_ARM arms capture mode: the window's raw words go to out[0..769] and the sigmoid to
//                     out[CAP_SIG_SLOT]
//   scaling_factor    input multiplier
//   out_offset        slot of the next result      trig_count   triggers seen
//   drain_count_last  [31:16] words flushed before the last trigger, [15:0] always 1
//   onset_miss_count  [31:16] TLAST restarts, total, [15:0] 0
//   window_size, onset_threshold: registers present but not used by the datapath.
#include "NN_axi.h"

void NN_axi(input_axi_t &in, output_axi_t out[BUFFER_SIZE], volatile bool *trigger, volatile unsigned *window_size, volatile unsigned *window_offset, volatile unsigned *scaling_factor, volatile unsigned *out_reset, unsigned *out_offset, unsigned *trig_count, volatile unsigned *onset_threshold, unsigned *drain_count_last, unsigned *onset_miss_count) {
    #pragma HLS LATENCY min=1

	#pragma HLS INTERFACE axis off port=in
    #pragma HLS INTERFACE bram depth=16384 latency=1 port=out
    #pragma HLS INTERFACE ap_none register port=trigger
	#pragma HLS INTERFACE s_axilite register port=window_size bundle=config
	#pragma HLS INTERFACE s_axilite register port=window_offset bundle=config
    #pragma HLS INTERFACE s_axilite register port=scaling_factor bundle=config
	#pragma HLS INTERFACE s_axilite register port=out_reset bundle=config
    #pragma HLS INTERFACE s_axilite register port=out_offset bundle=config
    #pragma HLS INTERFACE s_axilite register port=trig_count bundle=config
    #pragma HLS INTERFACE s_axilite register port=onset_threshold bundle=config
    #pragma HLS INTERFACE s_axilite register port=drain_count_last bundle=config
    #pragma HLS INTERFACE s_axilite register port=onset_miss_count bundle=config
    #pragma HLS INTERFACE ap_ctrl_none port=return

    // network input, filled by a shift register (SHIFT_L) so every write goes to a constant index
    input_t in_local[N_IQ_WINDOW_IN*2];
    #pragma HLS ARRAY_PARTITION variable=in_local cyclic factor=4 dim=1
    result_t out_local[N_OUT];

    unsigned k = 0;

    bool got_trigger = false;

    unsigned trig_count_local = 0;

    unsigned onset_miss_count_local = 0;
    unsigned tlast_restart_total = 0;   // windows restarted on an early TLAST

    unsigned idle_flush_total = 0;

#ifndef __SYNTHESIS__
    unsigned sim_iters = 0;
#endif

    FOREVER_L: do {

        // host reset of the counters and the output buffer
        OUT_RESET_C: if ((*out_reset) == 255) {
            k = 0;
            *out_offset = 0;
            trig_count_local = 0;
            *trig_count = 0;
            onset_miss_count_local = 0;
            tlast_restart_total = 0;
            *onset_miss_count = 0;
            *drain_count_last = 0;
            got_trigger = false;
            RESET_OUT_LOCAL_L: for (unsigned i = 0; i < N_OUT; i++) {
                #pragma HLS UNROLL
                out_local[i] = 0;
            }
            RESET_OUT_L: for (unsigned i = 0; i < BUFFER_SIZE; i++) {
				#pragma HLS PIPELINE
            	out[i] = 0;
            }
        } else {
            got_trigger = *trigger;

            // no trigger: discard stale words, a bounded number per iteration so the trigger is sampled promptly
            IDLE_FLUSH_L: for (unsigned f = 0; f < IDLE_FLUSH_PER_ITER; f++) {
                #pragma HLS PIPELINE
                in_word_t stale_word;
                if (*trigger)          // a shot is starting: keep its samples
                    break;
                if (!in.read_nb(stale_word))
                    break;             // queue empty
                idle_flush_total++;
            }
        }

        // trigger count and output slot, advanced when a trigger is seen
        unsigned trig_count_next = trig_count_local + (got_trigger ? 1u : 0u);
        unsigned k_next = k + (got_trigger ? 1u : 0u);
        if (k_next * N_OUT >= BUFFER_SIZE)
            k_next = 0;
        unsigned k_for_this_write = k;
        trig_count_local = trig_count_next;
        k = k_next;
        *trig_count = trig_count_local;
        *out_offset = k;

        TRIGGER_C: if (got_trigger) {

            bool found_onset = true;
            in_word_t onset_w;
            in.read(onset_w);
            unsigned drain_words_used = 1;

            *drain_count_last = ((idle_flush_total > 0xFFFFu ? 0xFFFFu
                                                            : idle_flush_total) << 16)
                              | (drain_words_used & 0xFFFFu);
            idle_flush_total = 0;   // per-shot, not cumulative
            onset_miss_count_local = onset_miss_count_local + (found_onset ? 0u : 1u);

            const bool do_capture = (*window_offset == CAPTURE_ARM);
            unsigned i = 0;
            bool use_onset = found_onset;
            unsigned restarts_this_window = 0;
            // read the window; an early TLAST restarts it
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
                ap_int<16> samp_i = word.range(15,0);
                ap_int<16> samp_q = word.range(31,16);
                v0 = samp_i * (int)(*scaling_factor);
                v1 = samp_q * (int)(*scaling_factor);
                SHIFT_L: for (int m = 0; m < (int)(2*N_IQ_WINDOW_IN) - 2; m++) {
                    #pragma HLS UNROLL
                    in_local[m] = in_local[m+2];
                }
                in_local[2*N_IQ_WINDOW_IN - 2] = v0;
                in_local[2*N_IQ_WINDOW_IN - 1] = v1;
                if (do_capture) {
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
            *onset_miss_count = ((tlast_restart_total > 0xFFFFu ? 0xFFFFu
                                                                : tlast_restart_total) << 16)
                              | (onset_miss_count_local & 0xFFFFu);

        	// evaluate the network after the whole window has arrived
        	NN(in_local, out_local);

        	STORE_L: for(unsigned i = 0; i < N_OUT; i++){
            	#pragma HLS PIPELINE
            	if (do_capture) {
            		out[CAP_SIG_SLOT + i] = out_local[i];
            	} else {
            		out[k_for_this_write*N_OUT + i] = out_local[i];
            	}
        	}

        }

#ifndef __SYNTHESIS__
        if (++sim_iters >= SIM_MAX_ITERS)
            break;
#endif
    } while (true);
}
