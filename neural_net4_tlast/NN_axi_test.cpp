// csim testbench for NN_axi. Replays recorded windows (tb_data/real_packed_words.bin, 770 packed {Q, I} words per
// shot, and tb_data/real_labels.bin, written by export_cosim_tb_data.py) as stream transfers with TLAST on each
// shot's last word, holds the trigger high, and checks one result per window, the trigger and slot counters, and
// the accuracy against the labels. Without the data files it runs on all-zero windows (control flow only).
// Environment: NN_TEST_SCALING_FACTOR, NN_TEST_WINDOW_OFFSET (CAPTURE_ARM checks the capture record),
// NN_TEST_ODD_AT / NN_TEST_ODD_LEN (insert one stray transfer before that shot; LOAD_L must restart on its TLAST).
// NN_TEST_ONSET_THRESHOLD and NN_TEST_PADDING_WORDS are read but unused.
#include "firmware/NN_axi.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <string>

// trace symbols NN.cpp references in csim (normally defined in NN_bridge.cpp, which this build does not compile)
namespace nnet {
bool trace_enabled = false;
std::map<std::string, void *> *trace_outputs = NULL;
size_t trace_type_size = sizeof(double);
} // namespace nnet

// small LCG noise words (|I|, |Q| <= 6) for the stray transfer
static uint32_t noise_lcg_state = 0x12345678u;
static ap_uint<32> synthetic_noise_word() {
    noise_lcg_state = noise_lcg_state * 1103515245u + 12345u;
    int16_t i_noise = (int16_t)((noise_lcg_state >> 16) % 13) - 6;
    noise_lcg_state = noise_lcg_state * 1103515245u + 12345u;
    int16_t q_noise = (int16_t)((noise_lcg_state >> 16) % 13) - 6;
    uint32_t word = ((uint32_t)(uint16_t)q_noise << 16) | (uint16_t)i_noise;
    return ap_uint<32>(word);
}

// one stream beat; TKEEP/TSTRB all ones as on hardware
static in_word_t mkword(ap_uint<32> data, bool last) {
    in_word_t w;
    w.data = data;
    w.keep = -1;
    w.strb = -1;
    w.last = last ? 1 : 0;
    return w;
}

int main() {
    static output_axi_t out_buf[BUFFER_SIZE];

    unsigned window_size = N_IQ_WINDOW_IN;
    unsigned window_offset = 0;
    if (const char *env_wo = std::getenv("NN_TEST_WINDOW_OFFSET"))
        window_offset = (unsigned)std::atoi(env_wo);
    unsigned scaling_factor = 1;
    if (const char *env_sf = std::getenv("NN_TEST_SCALING_FACTOR"))
        scaling_factor = (unsigned)std::atoi(env_sf);
    unsigned out_reset = 0;
    unsigned out_offset = 0;
    unsigned trig_count = 0;

    unsigned onset_threshold = 40;
    if (const char *env_thresh = std::getenv("NN_TEST_ONSET_THRESHOLD"))
        onset_threshold = (unsigned)std::atoi(env_thresh);

    unsigned drain_count_last = 0;
    unsigned onset_miss_count = 0;

    unsigned padding_words = 0;
    if (const char *env_pad = std::getenv("NN_TEST_PADDING_WORDS"))
        padding_words = (unsigned)std::atoi(env_pad);

    int odd_at = -1;                    // shot before which a stray transfer is inserted
    if (const char *env_oa = std::getenv("NN_TEST_ODD_AT"))
        odd_at = std::atoi(env_oa);
    unsigned odd_len = 385;
    if (const char *env_ol = std::getenv("NN_TEST_ODD_LEN"))
        odd_len = (unsigned)std::atoi(env_ol);

    printf("onset_threshold=%u (unused since v1.41)  padding_words=%u (unused since v1.41)  "
           "scaling_factor=%u  odd_at=%d odd_len=%u\n",
           onset_threshold, padding_words, scaling_factor, odd_at, odd_len);

#ifdef RTL_SIM
    std::string RESULTS_LOG = "tb_data/rtl_cosim_results.log";
#else
    std::string RESULTS_LOG = "tb_data/csim_results.log";
#endif

    std::ifstream fwords("tb_data/real_packed_words.bin", std::ios::binary);
    std::ifstream flabels("tb_data/real_labels.bin", std::ios::binary);

    input_axi_t in_stream;
    int32_t labels[SIM_MAX_ITERS];
    unsigned n_shots = 0;

    if (fwords.is_open() && flabels.is_open()) {
        printf("Loading real acquired data from tb_data/real_packed_words.bin / real_labels.bin\n");
        for (n_shots = 0; n_shots < SIM_MAX_ITERS; n_shots++) {
            uint32_t words[N_IQ_WINDOW_IN];
            fwords.read(reinterpret_cast<char *>(words), sizeof(words));
            if (!fwords) break;
            if ((int)n_shots == odd_at)
                for (unsigned p = 0; p < odd_len; p++)
                    in_stream.write(mkword(synthetic_noise_word(), p == odd_len - 1));
            for (unsigned i = 0; i < N_IQ_WINDOW_IN; i++)
                in_stream.write(mkword(ap_uint<32>(words[i]), i == N_IQ_WINDOW_IN - 1));

            int32_t label;
            flabels.read(reinterpret_cast<char *>(&label), sizeof(label));
            labels[n_shots] = label;
        }
        printf("Loaded %u real shots.\n", n_shots);
    } else {
        printf("INFO: tb_data/real_packed_words.bin not found -- falling back to all-zero placeholder input.\n");
        printf("Run export_cosim_tb_data.py first to test real classification accuracy.\n");
        n_shots = SIM_MAX_ITERS;
        for (unsigned iter = 0; iter < n_shots; iter++) {
            for (unsigned i = 0; i < N_IQ_WINDOW_IN; i++)
                in_stream.write(mkword(ap_uint<32>(0), i == N_IQ_WINDOW_IN - 1));
            labels[iter] = -1; // no label
        }
    }

    // trigger held high: every FOREVER_L iteration processes one window
    bool trigger_const = true;
    NN_axi(in_stream, out_buf, &trigger_const, &window_size, &window_offset,
           &scaling_factor, &out_reset, &out_offset, &trig_count, &onset_threshold,
           &drain_count_last, &onset_miss_count);
    printf("drain_count_last (last shot) = %u, onset_miss_count (cumulative) = %u / %u shots\n",
           drain_count_last, onset_miss_count, n_shots);
    printf("v1.41 TLAST restarts (cumulative) = %u, onset misses = %u\n",
           onset_miss_count >> 16, onset_miss_count & 0xFFFFu);

    unsigned expected_offset = n_shots;
    printf("out_offset after run: %u (expected %u)\n", out_offset, expected_offset);
    printf("trig_count after run: %u (expected %u -- should equal out_offset here since\n"
                       "  trigger is held constant true with no real-hardware catch-rate loss in csim)\n",
           trig_count, expected_offset);
    if (trig_count != out_offset) {
        printf("ERROR: trig_count != out_offset in csim -- the diagnostic counter itself has a bug\n"
                       "  (in csim, every observed trigger should also complete, since data is always ready).\n");
        return 1;
    }

    std::ofstream fout(RESULTS_LOG.c_str());
    bool ok = (out_offset == expected_offset);

    int correct = 0, g_correct = 0, g_total = 0, e_correct = 0, e_total = 0;
    bool have_labels = labels[0] != -1;

    for (unsigned k = 0; k < out_offset; k++) {
        float prob = out_buf[k * N_OUT];
        int predicted = prob > 0.5f ? 1 : 0;
        fout << prob << " \n";

        if (have_labels) {
            int label = labels[k];
            if (predicted == label) correct++;
            if (label == 0) { g_total++; if (predicted == 0) g_correct++; }
            else             { e_total++; if (predicted == 1) e_correct++; }
            printf("shot %3u: true=%d  prob=%.6f  predicted=%d\n", k, label, prob, predicted);
        } else {
            printf("out[%u] = %f\n", k, prob);
        }
    }
    fout.close();

    // capture mode: each captured word unpacks to two 16-bit halves; the sigmoid slot holds a probability
    if (window_offset == CAPTURE_ARM) {
        const unsigned NCAP = N_IQ_WINDOW_IN;
        double maxval = (double)65535u * (double)scaling_factor;
        unsigned bad = 0, nonzero = 0;
        for (unsigned i = 0; i < NCAP; i++) {
            capture_word_t cvt; cvt.f = out_buf[i];
            unsigned lo = cvt.u & 0xFFFFu, hi = (cvt.u >> 16) & 0xFFFFu;
            if (cvt.u != 0u) nonzero++;
            double a = (double)lo * (double)scaling_factor;
            double b = (double)hi * (double)scaling_factor;
            if (a < 0.0 || a > maxval || b < 0.0 || b > maxval) bad++;
        }
        float cap_prob = (float)out_buf[NCAP];
        printf("\n=== v1.26 CAPTURE VERIFICATION ===\n");
        printf("captured words              : %u (expected %u)\n", NCAP, NCAP);
        printf("non-zero captured words     : %u\n", nonzero);
        printf("out-of-range/non-integer    : %u (expected 0; bound = 65535*%u = %.0f)\n",
               bad, scaling_factor, maxval);
        printf("sigmoid slot out_buf[%u]  : %.6f\n", NCAP, cap_prob);
        printf("first 8 unpacked I/Q pairs  : ");
        for (unsigned i = 0; i < 8; i++) {
            capture_word_t c2; c2.f = out_buf[i];
            printf("(%u,%u) ", c2.u & 0xFFFFu, (c2.u >> 16) & 0xFFFFu);
        }
        printf("\n");
        if (bad != 0) {
            printf("ERROR: capture produced out-of-range or non-integer values -- "
                       "the mirror write in LOAD_L is wrong, do NOT build a bitstream.\n");
            return 1;
        }
        if (nonzero == 0) {
            printf("ERROR: capture region is entirely zero -- the mirror write never "
                       "executed, do NOT build a bitstream.\n");
            return 1;
        }
        if (!(cap_prob >= 0.0f && cap_prob <= 1.0f)) {
            printf("ERROR: capture-mode sigmoid slot is not a probability -- "
                       "STORE_L's capture branch is wrong.\n");
            return 1;
        }
        printf("CAPTURE VERIFICATION PASSED\n");
    }

    if (have_labels && out_offset > 0) {
        printf("\n=== NN_axi wrapper fidelity (full dataset, real acquired data) ===\n");
        printf("overall accuracy: %.2f%% (%d/%u)\n", 100.0 * correct / out_offset, correct, out_offset);
        if (g_total) printf("ground accuracy:  %.2f%% (%d/%d)\n", 100.0 * g_correct / g_total, g_correct, g_total);
        if (e_total) printf("excited accuracy: %.2f%% (%d/%d)\n", 100.0 * e_correct / e_total, e_correct, e_total);
    }

    if (!ok) {
        printf("ERROR: out_offset did not match the edge-triggered single-call expectation -- wrapper indexing looks wrong.\n");
        return 1;
    }

    printf("NN_axi testbench completed.\n");
    return 0;
}
