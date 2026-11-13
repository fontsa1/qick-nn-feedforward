#ifndef NN_LIVE_H_
#define NN_LIVE_H_

#include "ap_axi_sdata.h"
#include "ap_int.h"
#include "hls_stream.h"

#include "NN_live_rom.h"
#include "defines.h"

// One readout sample per word, as axis_dyn_readout_v1.m1_axis emits it:
// bits [15:0] = I, [31:16] = Q, both signed. live_window_gate frames exactly
// N_SAMP of them per trigger and sets TLAST on the last.
typedef hls::axis<ap_uint<32>, 0, 0, 0> in_word_t;
typedef hls::stream<in_word_t> live_stream_t;

// Per-shot result, a one-cycle ap_vld strobe as soon as the logit exists (it does not wait for the sigmoid):
//   [15:0]  logit: the sigmoid's input, q_output_alpha_result_t raw bits (ap_fixed<16,6>)
//   [44:16] shot index, low 29 bits
//   [63]    decision: logit >= DECISION_LOGIT_MIN
// The probability (sigmoid output) goes to the BRAM record a few cycles later.
//
// DECISION_LOGIT_MIN reproduces the scorer's rule, sigmoid output > 0.5, on
// hls4ml's table sigmoid: index = floor(logit * 64) + 512 and table[513] is
// the first entry above 0.5, so the rule is logit >= 1/64. NN_live_test.cpp
// checks the equivalence over all 65,536 logit values.
#define DECISION_LOGIT_MIN 0.015625

#define N_SAMP 770

// Output BRAM, 32-bit words, read by the host through an AXI BRAM controller.
#define HDR_SHOTS     0     // completed windows; written after each record
#define HDR_FRAME_ERR 1     // windows whose TLAST was not on word N_SAMP-1
#define HDR_MAGIC     2
#define CAP_BASE      16    // raw words of the most recent window
#define REC_BASE      1024  // ring of per-shot records
#define REC_WORDS     8     // shot, acc0..acc3, sigmoid input, sigmoid output, decision
#define N_REC         512
#define OUT_WORDS     (REC_BASE + N_REC * REC_WORDS)   // 5120
#define LIVE_MAGIC    0x4E4E4C31                       // "NNL1"

// First-layer sums: integers bounded by gen_live_rom.py, so ACC_W bits are exact.
typedef ap_int<ACC_W> acc_t;
// Not used by NN_live.cpp.
typedef ap_fixed<ACC_W, ACC_W> l2n_t;

void NN_live(live_stream_t &in, ap_uint<64> *res_o, ap_uint<32> out[OUT_WORDS]);

#endif
