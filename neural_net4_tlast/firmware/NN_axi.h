#ifndef NN_AXI_H_
#define NN_AXI_H_

#include "NN.h"
#include "hls_stream.h"
#include "ap_axi_sdata.h"

#include <iostream>

static const unsigned N_IQ_IN = 770;          // unused
static const unsigned N_IQ_WINDOW_IN = 770;   // window length; the model has 2 x 770 inputs
static const unsigned N_OUT = 1; // single sigmoid output
static const unsigned N_CONFIGS = 2;          // unused
static const unsigned BUFFER_SIZE = N_OUT * 16384; // result ring, 64 KB

// Capture mode (window_offset == CAPTURE_ARM): out[0..769] = the window's raw words, out[CAP_SIG_SLOT] = sigmoid.
static const unsigned CAP_SIG_SLOT = N_IQ_WINDOW_IN;          // 770

// N_INLOC_PAIRS, N_INLOC, INLOC_STRIDE, MAX_DRAIN_WORDS: unused.
static const unsigned N_INLOC_PAIRS = 128;
static const unsigned N_INLOC = 2 * N_INLOC_PAIRS;            // 256 words
static const unsigned INLOC_STRIDE = 6;                       // in PAIRS

static const unsigned MAX_DRAIN_WORDS = 1152;

// stale words discarded per loop iteration while no trigger is high
static const unsigned IDLE_FLUSH_PER_ITER = 64;

// window_offset value that arms capture mode
static const unsigned CAPTURE_ARM = 0xC0DE;

// carries a raw 32-bit stream word through the float output buffer unchanged
union capture_word_t {
    unsigned u;
    float f;
};

#ifndef __SYNTHESIS__
// csim only: bound on the free-running loop
static const unsigned SIM_MAX_ITERS = 500;
#endif

// input word with TLAST (end of a buffer readback transfer)
typedef hls::axis<ap_uint<32>, 0, 0, 0> in_word_t;
typedef hls::stream< in_word_t > input_axi_t;
typedef float output_axi_t;
typedef float output_t;

void NN_axi(input_axi_t &in, output_axi_t out[BUFFER_SIZE], volatile bool *trigger, volatile unsigned *window_size, volatile unsigned *window_offset, volatile unsigned *scaling_factor, volatile unsigned *out_reset, unsigned *out_offset, unsigned *trig_count, volatile unsigned *onset_threshold, unsigned *drain_count_last, unsigned *onset_miss_count);

#endif
