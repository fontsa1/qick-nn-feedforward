// Reference model for NN_live_test.cpp: the hls4ml NN() of the deployed model,
// compiled under renamed symbols so it links next to NN_live's own copies of
// the weight arrays. It loads its weights from the same weights/*.txt files.
#define NN ref_NN
#define w2 ref_w2
#define b2 ref_b2
#define s4 ref_s4
#define b4 ref_b4
#define w6 ref_w6
#define b6 ref_b6
#define s10 ref_s10
#define b10 ref_b10
#include "firmware/NN.cpp"

// The four-unit tail on hls4ml's own types (layer2_t, not NN_live's narrowed
// l2n_t), from given first-layer sums: the logit and the probability.
void ref_tail(const long acc[4], int *logit_raw, int *prob_raw) {
    layer2_t l2[4];
    for (int u = 0; u < 4; u++) l2[u] = acc[u];
    layer4_t l4[4];
    layer5_t l5[4];
    layer6_t l6[1];
    q_output_alpha_result_t l10[1];
    result_t l8[1];
    nnet::normalize<layer2_t, layer4_t, config4>(l2, l4, s4, b4);
    nnet::relu<layer4_t, layer5_t, relu_config5>(l4, l5);
    nnet::dense<layer5_t, layer6_t, config6>(l5, l6, w6, b6);
    nnet::normalize<layer6_t, q_output_alpha_result_t, config10>(l6, l10, s10, b10);
    nnet::sigmoid<q_output_alpha_result_t, result_t, sigmoid_config8>(l10, l8);
    *logit_raw = l10[0].range(15, 0).to_int();
    *prob_raw = l8[0].range(17, 0).to_int();
}

// Exhaustive check of NN_live's decision rule: over every ap_fixed<16,6>
// logit, (hls4ml sigmoid output > 0.5) must equal (logit >= thr).
long ref_decision_rule_mismatches(double thr) {
    const q_output_alpha_result_t t = thr;
    const result_t half = 0.5;
    long bad = 0;
    for (int raw = -32768; raw <= 32767; raw++) {
        q_output_alpha_result_t x[1];
        result_t y[1];
        x[0].range(15, 0) = raw;
        nnet::sigmoid<q_output_alpha_result_t, result_t, sigmoid_config8>(x, y);
        if ((y[0] > half) != (x[0] >= t)) bad++;
    }
    return bad;
}
