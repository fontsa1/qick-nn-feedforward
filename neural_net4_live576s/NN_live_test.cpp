// csim testbench for NN_live (task 4.1).
//
// For every shot in tb_data/live_words.bin it streams the N_SAMP words into
// NN_live with TLAST on the last, then checks, bit for bit:
//   * the sigmoid output against the deployed hls4ml NN() (tb_ref_nn.cpp)
//     run on the window NN_axi would have built from the same words;
//   * the logit (res_o) against hls4ml's tail on its own layer2_t type;
//   * the four first-layer sums against a direct dot product with w2;
//   * the sigmoid output against the v1.40 bit-accurate emulator, where
//     tb_data/live_meta.bin carries one;
//   * the res_o word (and that it is written only for scored windows), the
//     record ring, the header counters and the capture.
// Then it injects a short window and a window without TLAST and checks that
// each costs exactly one frame error and the next window is scored correctly,
// and checks the decision rule (logit >= DECISION_LOGIT_MIN equals sigmoid
// output > 0.5) over all 65,536 logits.
//
// NN_LIVE_N=<n> limits the run to the first n shots.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

#include "firmware/NN_live.h"

void ref_NN(input_t input_1[2 * N_SAMP], result_t layer8_out[1]);
extern weight2_t ref_w2[8 * N_SAMP];
extern bias2_t ref_b2[4];
void ref_tail(const long acc[4], int *logit_raw, int *prob_raw);
long ref_decision_rule_mismatches(double thr);

static ap_uint<32> out_mem[OUT_WORDS];
static const ap_uint<64> NO_RESULT = 0xDEADBEEFDEADBEEFULL;   // res_o untouched

static void push_window(live_stream_t &s, const uint32_t *w, int n, bool last_on_final) {
    for (int k = 0; k < n; k++) {
        in_word_t x;
        x.data = w[k];
        x.keep = -1;
        x.strb = -1;
        x.last = (last_on_final && k == n - 1) ? 1 : 0;
        s.write(x);
    }
}

int main() {
    std::ifstream fw("tb_data/live_words.bin", std::ios::binary);
    std::ifstream fm("tb_data/live_meta.bin", std::ios::binary);
    if (!fw || !fm) {
        std::printf("ERROR: tb_data missing; run make_tb_data.py\n");
        return 1;
    }
    int32_t n_all = 0, n_meta = 0;
    fw.read((char *)&n_all, 4);
    fm.read((char *)&n_meta, 4);
    if (n_all != n_meta) {
        std::printf("ERROR: words/meta disagree (%d vs %d)\n", n_all, n_meta);
        return 1;
    }
    std::vector<uint32_t> words((size_t)n_all * N_SAMP);
    std::vector<int32_t> labels(n_all);
    std::vector<float> emu(n_all);
    fw.read((char *)words.data(), words.size() * 4);
    fm.read((char *)labels.data(), labels.size() * 4);
    fm.read((char *)emu.data(), emu.size() * 4);

    int n = n_all;
    if (const char *e = std::getenv("NN_LIVE_N")) n = std::min(n, std::atoi(e));

    live_stream_t in("in");
    ap_uint<64> res = NO_RESULT;
    for (int k = 0; k < OUT_WORDS; k++) out_mem[k] = 0;

    long bad_ref = 0, bad_acc = 0, bad_emu = 0, n_emu = 0, bad_res = 0, bad_cap = 0;
    long correct = 0, correct_emu = 0;
    for (int s = 0; s < n; s++) {
        const uint32_t *w = &words[(size_t)s * N_SAMP];
        push_window(in, w, N_SAMP, true);
        res = NO_RESULT;
        NN_live(in, &res, out_mem);

        // reference: the window NN_axi's LOAD_L builds (I at 2n, Q at 2n+1, sf = 1)
        input_t x[2 * N_SAMP];
        for (int k = 0; k < N_SAMP; k++) {
            ap_int<16> si = (ap_int<16>)(w[k] & 0xFFFF);
            ap_int<16> sq = (ap_int<16>)(w[k] >> 16);
            x[2 * k] = si * 1;
            x[2 * k + 1] = sq * 1;
        }
        result_t ref[1];
        ref_NN(x, ref);

        int base = REC_BASE + (s % N_REC) * REC_WORDS;
        result_t got;
        got.range(17, 0) = out_mem[base + 6].range(17, 0);
        if (got != ref[0] || out_mem[base + 0] != (unsigned)s) bad_ref++;

        long sums[4];
        for (int u = 0; u < 4; u++) {
            long sum = (long)ref_b2[u].to_int();
            for (int i = 0; i < 2 * N_SAMP; i++) sum += (long)ref_w2[u * 2 * N_SAMP + i].to_int() * x[i].to_int();
            sums[u] = sum;
            long chip = (long)(int32_t)(uint32_t)out_mem[base + 1 + u];
            if (sum != chip) bad_acc++;
        }
        int ref_logit = 0, ref_prob = 0;
        ref_tail(sums, &ref_logit, &ref_prob);
        if ((ref_prob & 0x3FFFF) != (int)(ref[0].range(17, 0).to_int())) bad_ref++;

        if (res == NO_RESULT) {
            bad_res++;
        } else {
            ap_uint<64> r = res;
            bool dec = (ref[0] > (result_t)0.5);
            if ((int)r.range(15, 0) != (ref_logit & 0xFFFF) || r[63] != dec ||
                r.range(44, 16) != (unsigned)(s & 0x1FFFFFFF) || out_mem[base + 7] != (unsigned)dec ||
                (int)out_mem[base + 5].range(15, 0) != (ref_logit & 0xFFFF))
                bad_res++;
        }
        for (int k = 0; k < N_SAMP; k += 97)
            if (out_mem[CAP_BASE + k] != w[k]) bad_cap++;

        int pred = (got > (result_t)0.5) ? 1 : 0;
        correct += (pred == labels[s]);
        if (std::isfinite(emu[s])) {
            n_emu++;
            if (got.to_float() != emu[s]) bad_emu++;
            correct_emu += ((emu[s] > 0.5f ? 1 : 0) == labels[s]);
        }
        if (s < 3 || s % 5000 == 0)
            std::printf("shot %5d  live %.6f  ref %.6f  emu %.6f  acc [%d %d %d %d]\n", s, got.to_float(),
                        ref[0].to_float(), emu[s], (int)(int32_t)(uint32_t)out_mem[base + 1],
                        (int)(int32_t)(uint32_t)out_mem[base + 2], (int)(int32_t)(uint32_t)out_mem[base + 3],
                        (int)(int32_t)(uint32_t)out_mem[base + 4]);
    }
    bool hdr_ok = out_mem[HDR_SHOTS] == (unsigned)n && out_mem[HDR_FRAME_ERR] == 0 &&
                  out_mem[HDR_MAGIC] == LIVE_MAGIC;

    // Framing faults. The stream is filled before each call so no read ever
    // finds it empty, as on hardware where the next window always follows.
    long fault_bad = 0;
    const uint32_t *w0 = &words[0];
    result_t ref0;
    {
        result_t r[1];
        input_t x[2 * N_SAMP];
        for (int k = 0; k < N_SAMP; k++) {
            x[2 * k] = (ap_int<16>)(w0[k] & 0xFFFF);
            x[2 * k + 1] = (ap_int<16>)(w0[k] >> 16);
        }
        ref_NN(x, r);
        ref0 = r[0];
    }
    // (1) a window one word short (TLAST on word 768), then two good windows.
    //     Pass 1 reads the short window plus word 0 of good window A, sees the
    //     early TLAST and no TLAST on word 769, counts one error and drains A.
    //     Pass 2 scores good window B.
    unsigned fe0 = out_mem[HDR_FRAME_ERR], sh0 = out_mem[HDR_SHOTS];
    push_window(in, w0, N_SAMP - 1, true);
    push_window(in, w0, N_SAMP, true);
    push_window(in, w0, N_SAMP, true);
    res = NO_RESULT;
    NN_live(in, &res, out_mem);
    bool p1 = out_mem[HDR_FRAME_ERR] == fe0 + 1 && res == NO_RESULT && out_mem[HDR_SHOTS] == sh0;
    NN_live(in, &res, out_mem);
    bool p2 = res != NO_RESULT && in.empty();
    if (p2) {
        result_t rv;
        rv.range(17, 0) = out_mem[REC_BASE + (sh0 % N_REC) * REC_WORDS + 6].range(17, 0);
        p2 = (rv == ref0) && out_mem[HDR_SHOTS] == sh0 + 1;
    }
    std::printf("short window      : frame_err %u -> %u, dropped pass %s, next window scored %s\n", fe0,
                (unsigned)out_mem[HDR_FRAME_ERR], p1 ? "ok" : "WRONG", p2 ? "ok" : "WRONG");
    if (!p1 || !p2) fault_bad++;

    // (2) a window with no TLAST on word 769, its TLAST 10 words later, then a
    //     good window. Pass 1 counts one error and drains the 10 words.
    unsigned fe1 = out_mem[HDR_FRAME_ERR], sh1 = out_mem[HDR_SHOTS];
    std::vector<uint32_t> tail(10, 0x00010001u);
    push_window(in, w0, N_SAMP, false);
    push_window(in, tail.data(), 10, true);
    push_window(in, w0, N_SAMP, true);
    res = NO_RESULT;
    NN_live(in, &res, out_mem);
    bool q1 = out_mem[HDR_FRAME_ERR] == fe1 + 1 && res == NO_RESULT && out_mem[HDR_SHOTS] == sh1;
    NN_live(in, &res, out_mem);
    bool q2 = res != NO_RESULT && in.empty();
    if (q2) {
        result_t rv;
        rv.range(17, 0) = out_mem[REC_BASE + (sh1 % N_REC) * REC_WORDS + 6].range(17, 0);
        q2 = (rv == ref0) && out_mem[HDR_SHOTS] == sh1 + 1;
    }
    std::printf("missing TLAST     : frame_err %u -> %u, dropped pass %s, next window scored %s\n", fe1,
                (unsigned)out_mem[HDR_FRAME_ERR], q1 ? "ok" : "WRONG", q2 ? "ok" : "WRONG");
    if (!q1 || !q2) fault_bad++;

    long rule_bad = ref_decision_rule_mismatches(DECISION_LOGIT_MIN);

    std::printf("\n=== NN_live csim: %d shots ===\n", n);
    std::printf("decision rule over all logits  : %ld mismatches (of 65536)\n", rule_bad);
    std::printf("sigmoid output vs hls4ml NN()  : %ld mismatches\n", bad_ref);
    std::printf("first-layer sums vs w2 . x     : %ld mismatches (of %d)\n", bad_acc, 4 * n);
    std::printf("sigmoid output vs emulator     : %ld mismatches (of %ld)\n", bad_emu, n_emu);
    std::printf("res_o logit / decision / record: %ld mismatches\n", bad_res);
    std::printf("capture region                 : %ld mismatches\n", bad_cap);
    std::printf("header counters                : %s\n", hdr_ok ? "ok" : "WRONG");
    std::printf("framing fault handling         : %s\n", fault_bad ? "FAILED" : "ok");
    std::printf("fidelity (live)                : %.2f%%\n", 100.0 * correct / n);
    if (n_emu) std::printf("fidelity (emulator, same shots): %.2f%%\n", 100.0 * correct_emu / n_emu);
    int fail = bad_ref || bad_acc || bad_emu || bad_res || bad_cap || !hdr_ok || fault_bad || rule_bad;
    std::printf("RESULT: %s\n", fail ? "FAIL" : "PASS");
    return fail;
}
