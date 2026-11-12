// NN_live: the v1.40 classifier, fed from the live readout stream (task 4.1).
//
// NN_live 1.5 (v1.41, audit R.17 / C1): the logit (q_output_alpha_result_t) saturates instead
// of wrapping: ap_fixed<16,6,AP_TRN,AP_SAT> in defines.h (was ap_fixed<16,6>, wrap, so a true
// logit at or beyond +-32 came out modulo 64 with a random sign, ~4x the deployed amplitude).
// Same 16-bit res_o / record layout; identical logits for every input whose logit is in range.
// The tail of the fast path uses exact narrow types (bounds checked in csim, below) so that the
// saturation costs no extra pipeline stage: batchnorm1 in ap_fixed<28,12>, the pre-cast logit in
// ap_fixed<30,14>, and the decision taken from the pre-cast sum (st >= 1/64 is the same test as
// sat(floor(st)) >= 1/64, since 1/64 lies on the logit's 2^-10 grid).
//
// NN_live 1.3 (v1.40L9, task 4.1h): the tail is retimed. batchnorm1's scale is
// applied per sample instead of once at the end: each sample's ternary
// contribution d is multiplied by K = s4 (raw, units of 2^-32) and added to a
// scaled accumulator that starts at K*b2 + b4 (raw). All of it is integer
// arithmetic, so the final scaled sum equals s4*acc + b4 exactly, and
// batchnorm1's output is its bits [47:16] (layer4_t: 16 fraction bits, floor,
// wrap), as before. The last sample then reaches relu1 through one
// multiply-accumulate instead of add, add, multiply, add. The weight ROM has a
// 1-cycle read. The unscaled sums are still kept for the record.
//
// NN_axi buffers all 770 samples and then evaluates the network, so its
// latency starts when the window ends. Here the first layer (1540 -> 4,
// ternary) is accumulated as the samples arrive instead. It is linear, so
// each sample's contribution can be added the moment it exists: per sample
// and per hidden unit, add, subtract or skip I and Q. When the last sample
// arrives the four sums are already final, and only the 4-unit tail
// (batchnorm1, relu1, q_output, q_output_alpha, sigmoid) is left. The tail
// runs hls4ml's own layer functions with the deployed weights, and the sums
// are exact integers, so the output is bit-identical to NN() on the same
// window (checked shot by shot in NN_live_test.cpp).
//
// Clocking: runs on the readout clock (clk_adc0_x2, 552.96 MHz) beside
// axis_dyn_readout_v1_0, so no clock crossing is needed on the input.
//
// Framing: live_window_gate (rtl/) cuts exactly N_SAMP samples per tProc
// trigger (trig_0_o, the same trigger and the same capture rules as
// axis_avg_buffer_0's decimated buffer) and sets TLAST on the last. The
// readout has no backpressure, so ACC_L must take a word every cycle (II=1);
// between windows the gate holds TVALID low and this kernel waits.

#include "NN_live.h"
#include "parameters.h"
#ifndef __SYNTHESIS__
#include <cmath>
#include <cstdlib>
#endif

// one sample's contribution to one hidden unit: +-I +-Q, |d| <= 2^16
typedef ap_int<18> d_t;
// scaled first-layer sum, units of 2^-32 (batchnorm1's scale and bias raw LSB)
typedef ap_int<48> accs_t;

static inline d_t tern(ap_uint<2> code, ap_int<16> x) {
    #pragma HLS INLINE
    // 2-bit two's complement, as packed by gen_live_rom.py: 01 = +1, 11 = -1.
    d_t v = x;
    return (code == 1) ? v : (code == 3) ? d_t(-v) : d_t(0);
}

// raw integer of batchnorm1's scale / bias (ap_fixed<40,8>: 32 fraction bits)
static inline accs_t scale_raw(batchnorm1_scale_t s) {
    #pragma HLS INLINE
    ap_int<40> r = s.range(39, 0);
    return r;
}
static inline accs_t bias_raw(batchnorm1_bias_t b) {
    #pragma HLS INLINE
    ap_int<40> r = b.range(39, 0);
    return r;
}

void NN_live(live_stream_t &in, ap_uint<64> *res_o, ap_uint<32> out[OUT_WORDS]) {
    #pragma HLS INTERFACE ap_ctrl_none port=return
    // No input register slice: live_window_gate's outputs are already registered.
    #pragma HLS INTERFACE axis port=in register_mode=off
    #pragma HLS INTERFACE ap_vld port=res_o
    #pragma HLS INTERFACE bram port=out storage_type=ram_1p

#ifndef __SYNTHESIS__
    // csim only: the weight headers are uninitialized outside synthesis
    // (same mechanism as NN.cpp).
    static bool loaded_weights = false;
    if (!loaded_weights) {
        nnet::load_weights_from_txt<bias2_t, 4>(b2, "b2.txt");
        nnet::load_weights_from_txt<batchnorm1_scale_t, 4>(s4, "s4.txt");
        nnet::load_weights_from_txt<batchnorm1_bias_t, 4>(b4, "b4.txt");
        nnet::load_weights_from_txt<weight6_t, 4>(w6, "w6.txt");
        nnet::load_weights_from_txt<bias6_t, 1>(b6, "b6.txt");
        nnet::load_exponent_weights_from_txt<exponent_scale10_t, 1>(s10, "s10.txt");
        nnet::load_weights_from_txt<bias10_t, 1>(b10, "b10.txt");
        loaded_weights = true;
        // 1.5: the narrow tail below is exact only if, for every int16 window,
        // |batchnorm1| < 2^11 and the pre-cast logit lies within +-2^13. Check the
        // worst case from the weights: |acc_u| <= nnz_u * 32768 + |b2_u|.
        double lt_hi = (double)b6[0].to_int() + b10[0].to_int(), lt_lo = lt_hi;
        for (int u = 0; u < 4; u++) {
            long nnz = 0;
            for (int n = 0; n < N_SAMP; n++) {
                ap_uint<16> r = WROM[n];
                nnz += (r.range(4 * u + 1, 4 * u) != 0) + (r.range(4 * u + 3, 4 * u + 2) != 0);
            }
            double l4max = s4[u].to_double() * (nnz * 32768.0 + 1) + std::fabs(b4[u].to_double());
            if (!(l4max < 2048.0)) { std::printf("NN_live: batchnorm1 bound %.1f >= 2048, narrow tail invalid\n", l4max); std::exit(1); }
            if (w6[u] == 1) lt_hi += l4max;
            if (w6[u] == -1) lt_lo -= l4max;
        }
        if (!(lt_hi < 8192.0 && lt_lo > -8192.0)) { std::printf("NN_live: logit bound invalid\n"); std::exit(1); }
    }
#endif

    #pragma HLS BIND_STORAGE variable=WROM type=rom_1p impl=lutram latency=1

    static ap_uint<32> shots = 0;
    static ap_uint<32> frame_err = 0;

    acc_t acc0 = b2[0], acc1 = b2[1], acc2 = b2[2], acc3 = b2[3];
    // scaled sums (units of 2^-32): K*acc + b4, K = s4 raw. |acc| < 2^25 and
    // K < 2^20, b4 < 2^40 (raw), so 48 bits hold them exactly.
    accs_t k0 = scale_raw(s4[0]), k1 = scale_raw(s4[1]), k2 = scale_raw(s4[2]), k3 = scale_raw(s4[3]);
    accs_t as0 = k0 * b2[0] + bias_raw(b4[0]);
    accs_t as1 = k1 * b2[1] + bias_raw(b4[1]);
    accs_t as2 = k2 * b2[2] + bias_raw(b4[2]);
    accs_t as3 = k3 * b2[3] + bias_raw(b4[3]);
    bool early_last = false;
    bool last_ok = false;
    q_output_alpha_result_t logit = 0;
    ap_uint<1> decision = 0;

ACC_L:
    for (int n = 0; n < N_SAMP; n++) {
        #pragma HLS PIPELINE II=1 style=frp
        in_word_t w = in.read();
        ap_int<16> si = w.data.range(15, 0);
        ap_int<16> sq = w.data.range(31, 16);
        ap_uint<16> r = WROM[n];
        d_t d0 = tern(r.range(1, 0), si) + tern(r.range(3, 2), sq);
        d_t d1 = tern(r.range(5, 4), si) + tern(r.range(7, 6), sq);
        d_t d2 = tern(r.range(9, 8), si) + tern(r.range(11, 10), sq);
        d_t d3 = tern(r.range(13, 12), si) + tern(r.range(15, 14), sq);
        acc0 += d0; acc1 += d1; acc2 += d2; acc3 += d3;
        as0 += k0 * d0; as1 += k1 * d1; as2 += k2 * d2; as3 += k3 * d3;
        out[CAP_BASE + n] = w.data;
        bool final_word = (n == N_SAMP - 1);
        if (!final_word && w.last) early_last = true;
        if (final_word) {
            last_ok = w.last;
            // The tail runs inside the last iteration, so it starts in the
            // cycle the final sums exist, with no call handshakes between layers.
            q_output_alpha_result_t l10[1];
            bool dec_v;
            // The two normalize layers are written as nnet::normalize's own
            // statement (n_filt == -1), calling the same product classes,
            // rather than through normalize itself: its
            // "ALLOCATION instances=mul limit=multiplier_limit" pragma, once
            // inlined into this loop, applies to the whole loop, and config10's
            // limit of 1 made batchnorm1's four multipliers share one (II = 4).
            // batchnorm1 = (s4*acc + b4) cast to layer4_t (floor, wrap) = bits [47:16]
            // of the scaled sum (1.2 formed the product here, after the last add).
            if (s10[0].sign == 1 && s10[0].weight == 0) {
                // q_output_alpha is then x * 2^0 + b10, so b10 joins q_output's
                // bias and the four ternary terms are summed as a tree, in
                // config6::accum_t (ap_fixed<32,16>). 1.5: the final cast saturates,
                // so the sum must be exact, not just exact modulo 64: over every int16
                // window it lies within about +-3,500 (relu outputs <= ~1,900 each,
                // sat_check.cpp prints the bound), far inside +-32768, so it never wraps
                // and the saturating cast sees the true value.
                // Ternary weights: the +1 terms are added and the -1 terms
                // subtracted (no negation stage).
                // 1.5: exact narrow types (bounds checked in csim at load time):
                // batchnorm1 = bits [43:16] of the scaled sum (|value| < 2^11, so bits
                // [47:44] are sign copies and layer4_t's value is unchanged), relu on
                // its sign bit, and the sum in ap_fixed<30,14> (|value| < 2^13).
                typedef ap_fixed<28, 12> l4n_t;
                typedef ap_fixed<30, 14> stn_t;
                l4n_t n4[4];
                n4[0].range(27, 0) = as0.range(43, 16);
                n4[1].range(27, 0) = as1.range(43, 16);
                n4[2].range(27, 0) = as2.range(43, 16);
                n4[3].range(27, 0) = as3.range(43, 16);
                stn_t pos[4], neg[4];
                for (int u = 0; u < 4; u++) {
                    #pragma HLS UNROLL
                    stn_t r = n4[u][27] ? stn_t(0) : stn_t(n4[u]);              // relu1
                    pos[u] = (w6[u] == 1) ? r : stn_t(0);
                    neg[u] = (w6[u] == -1) ? r : stn_t(0);
                }
                // The saturating cast and the decision as sign bits of four parallel
                // sums, st - c for c in {0, 1/64, 32, -32}, with c folded into the second
                // operand (same depth as st itself), so only a select follows the last add:
                //   decision  = st >= 1/64           (1/64 is on the logit's 2^-10 grid)
                //   saturate+ = st >= 32  (floor(st * 2^10) > 32767)
                //   saturate- = st < -32  (floor(st * 2^10) < -32768)
                // The result equals the AP_SAT cast of st bit for bit.
                const stn_t cb = stn_t(b6[0]) + stn_t(b10[0]);
                const stn_t cd = cb - stn_t(DECISION_LOGIT_MIN);
                const stn_t ch = cb - stn_t(32);
                const stn_t cl = cb + stn_t(32);
                stn_t sa  = stn_t(pos[0] + pos[1]) - neg[0] - neg[1];
                stn_t sb  = stn_t(pos[2] + pos[3] + cb) - neg[2] - neg[3];
                stn_t sbd = stn_t(pos[2] + pos[3] + cd) - neg[2] - neg[3];
                stn_t sbh = stn_t(pos[2] + pos[3] + ch) - neg[2] - neg[3];
                stn_t sbl = stn_t(pos[2] + pos[3] + cl) - neg[2] - neg[3];
                stn_t st = sa + sb, sd = sa + sbd, sh = sa + sbh, sl = sa + sbl;
                bool ovp = !sh[29], ovn = sl[29];
                ap_fixed<16, 6> lw = st;         // st's bits [21:6] (floor; exact when not saturated)
                ap_uint<16> lr = ovp ? ap_uint<16>(0x7FFF) : ovn ? ap_uint<16>(0x8000) : ap_uint<16>(lw.range(15, 0));
                l10[0].range(15, 0) = lr;        // = the AP_SAT cast of st
                dec_v = !sd[29];
            } else {
                layer4_t l4[4];
                layer5_t l5[4];
                layer6_t l6[1];
                l4[0].range(31, 0) = as0.range(47, 16);
                l4[1].range(31, 0) = as1.range(47, 16);
                l4[2].range(31, 0) = as2.range(47, 16);
                l4[3].range(31, 0) = as3.range(47, 16);
                for (int u = 0; u < 4; u++) {
                    #pragma HLS UNROLL
                    l5[u] = l4[u][31] ? layer5_t(0) : layer5_t(l4[u]);
                }
                nnet::dense<layer5_t, layer6_t, config6>(l5, l6, w6, b6);                      // q_output
                l10[0] = config10::template product<layer6_t, config10::scale_t>::product(l6[0], s10[0]) + b10[0];  // q_output_alpha
                const q_output_alpha_result_t dmin = DECISION_LOGIT_MIN;
                dec_v = (l10[0] >= dmin);
            }
            logit = l10[0];
            decision = dec_v ? 1 : 0;
            if (w.last && !early_last) {
                ap_uint<64> r64 = 0;
                r64.range(15, 0) = l10[0].range(15, 0);
                r64.range(44, 16) = shots.range(28, 0);
                r64[63] = decision;
                *res_o = r64;
            }
        }
    }

    if (early_last || !last_ok) {
        // A word was lost (a short window) or this pass started mid-window.
        // Count it, drop the result, and if the window's TLAST is still
        // ahead, consume up to it so the next pass starts on a window.
        frame_err++;
        out[HDR_FRAME_ERR] = frame_err;
        if (!last_ok) {
        DRAIN_L:
            while (!in.read().last) {
                #pragma HLS PIPELINE II=1
            }
        }
        return;
    }

    // The probability, for the host's record only; the decision above does
    // not wait for it.
    q_output_alpha_result_t l10[1];
    result_t l8[1];
    l10[0] = logit;
    nnet::sigmoid<q_output_alpha_result_t, result_t, sigmoid_config8>(l10, l8);       // sigmoid

    int base = REC_BASE + (int)(shots % N_REC) * REC_WORDS;
    ap_uint<32> r_l10 = 0, r_l8 = 0;
    r_l10.range(15, 0) = logit.range(15, 0);
    r_l8.range(17, 0) = l8[0].range(17, 0);
    out[base + 0] = shots;
    out[base + 1] = (ap_uint<32>)(ap_int<32>)acc0;
    out[base + 2] = (ap_uint<32>)(ap_int<32>)acc1;
    out[base + 3] = (ap_uint<32>)(ap_int<32>)acc2;
    out[base + 4] = (ap_uint<32>)(ap_int<32>)acc3;
    out[base + 5] = r_l10;
    out[base + 6] = r_l8;
    out[base + 7] = decision;
    shots++;
    out[HDR_SHOTS] = shots;
    out[HDR_FRAME_ERR] = frame_err;
    out[HDR_MAGIC] = LIVE_MAGIC;
}
