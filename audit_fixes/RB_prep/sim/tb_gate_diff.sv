// tb_gate_diff: rtl/dac_gate.v against the earlier version (ref/dac_gate_ref.v, GAT3), same stimulus on
// every cycle:
//   LEG = dac_gate #(EARLY_BLANK 0, SHOT_BIND 0, FAIL_SAFE 0, CFG_SAFE 0): must equal REF in every phase (stream
//         on every cycle; record words 1-9, 10[3:0], 12-18).
//   NEW = dac_gate #(EARLY_BLANK 1, SHOT_BIND 1, FAIL_SAFE 0, CFG_SAFE 0).
// Decisions come from NN-like result strobes on the readout clock through live_res_cdc into clk_dac2; the shot
// field counts results as NN_live does. Phases:
//   1 normal operation: one cls_trig and one result per shot, DELAY = word 7 - DELAY_LEAD (4..12), the pulse
//     inside the window, zero samples outside pulses, slack -40..+200 cycles (late decisions included),
//     random policy and ENABLE between shots. NEW must equal REF exactly; word 19 = 0.
//   2 as 1 with small noise outside pulses: NEW may differ from REF only by zeroing the window's first two words.
//   3 abnormal (LEG vs REF only): lost results, cls_trig closer than the NN latency, DELAY_LEAD 0..3, ENABLE or
//     policy changes inside windows, second markers, noise.
`timescale 1ps / 1ps
module tb_gate_diff;
    localparam integer NS = 16, D_PS = 2034, S_PS = 1808;
    parameter integer N1 = 1500, N2 = 600, N3 = 1500;
    reg aclk = 0, s_aclk = 0, aresetn = 0;
    always #(D_PS/2) aclk = ~aclk;
    initial begin #(777); forever #(S_PS/2) s_aclk = ~s_aclk; end

    // ---- decision path (shared)
    reg  [63:0] res_o = 0; reg res_vld = 0;
    wire [63:0] dec_tdata; wire dec_tvalid;
    live_res_cdc cdc (.s_aclk(s_aclk), .s_aresetn(aresetn), .res_o(res_o), .res_o_ap_vld(res_vld),
                      .m_aclk(aclk), .m_aresetn(aresetn), .m_axis_tdata(dec_tdata), .m_axis_tvalid(dec_tvalid), .m_axis_tready(1'b1));

    reg  [16*NS-1:0] s_tdata = 0;
    reg marker = 0, cls = 0;
    wire [16*NS-1:0] o_r, o_l, o_n; wire v_r, v_l, v_n;
    wire ben_r, ben_l, ben_n; wire [3:0] bwe_r, bwe_l, bwe_n; wire [31:0] ba_r, ba_l, ba_n, bd_r, bd_l, bd_n;
    reg  [31:0] bq_r = 0, bq_l = 0, bq_n = 0;
    reg  [31:0] mem_r [0:2047], mem_l [0:2047], mem_n [0:2047];

    `define GATE_PORTS(O, V, BEN, BWE, BA, BD, BQ) (.aclk(aclk), .aresetn(aresetn), \
        .s_axis_tdata(s_tdata), .s_axis_tvalid(1'b1), .s_axis_tready(), .m_axis_tdata(O), .m_axis_tvalid(V), .m_axis_tready(1'b1), \
        .dec_axis_tdata(dec_tdata), .dec_axis_tvalid(dec_tvalid), .dec_axis_tready(), .marker(marker), .cls_trig(cls), \
        .bram_clk(), .bram_rst(), .bram_en(BEN), .bram_we(BWE), .bram_addr(BA), .bram_din(BD), .bram_dout(BQ))
    dac_gate_ref #(.NS(NS)) REF `GATE_PORTS(o_r, v_r, ben_r, bwe_r, ba_r, bd_r, bq_r);
    dac_gate #(.NS(NS), .EARLY_BLANK(0), .SHOT_BIND(0), .FAIL_SAFE(0), .CFG_SAFE(0)) LEG `GATE_PORTS(o_l, v_l, ben_l, bwe_l, ba_l, bd_l, bq_l);
    dac_gate #(.NS(NS)) NEW `GATE_PORTS(o_n, v_n, ben_n, bwe_n, ba_n, bd_n, bq_n);

    integer i;
    initial for (i = 0; i < 2048; i = i + 1) begin mem_r[i] = 0; mem_l[i] = 0; mem_n[i] = 0; end
    always @(posedge aclk) if (ben_r) begin if (bwe_r == 4'hF) begin mem_r[ba_r[12:2]] <= bd_r; bq_r <= bd_r; end else bq_r <= mem_r[ba_r[12:2]]; end
    always @(posedge aclk) if (ben_l) begin if (bwe_l == 4'hF) begin mem_l[ba_l[12:2]] <= bd_l; bq_l <= bd_l; end else bq_l <= mem_l[ba_l[12:2]]; end
    always @(posedge aclk) if (ben_n) begin if (bwe_n == 4'hF) begin mem_n[ba_n[12:2]] <= bd_n; bq_n <= bd_n; end else bq_n <= mem_n[ba_n[12:2]]; end
    task wcfg(input [31:0] w); begin mem_r[0] = w; mem_l[0] = w; mem_n[0] = w; end endtask

    // ---- generator model: pulse of PLEN words GEN_LAT cycles after the marker rise
    integer cyc = 0, p_start = -100000, PLEN = 40, GEN_LAT = 40, noise = 0;
    reg marker_q = 0;
    always @(posedge aclk) begin
        cyc <= cyc + 1;
        marker_q <= marker;
        if (marker && !marker_q) p_start <= cyc + GEN_LAT;
    end
    wire in_pulse = (cyc >= p_start) && (cyc < p_start + PLEN);
    integer j; reg [15:0] smp;
    always @(posedge aclk)
        for (j = 0; j < NS; j = j + 1) begin
            smp = 16'd1000 + ($urandom % 20000);
            s_tdata[16*j +: 16] <= in_pulse ? (($urandom % 2) ? smp : -smp) : (noise ? ($urandom % 7) - 3 : 16'd0);
        end

    // ---- per-cycle stream comparison (sampled at negedge, after the edge settles)
    integer phase = 0;
    integer d_leg = 0, d_new = 0, d_new_ok = 0, d_new_bad = 0, ncyc = 0;
    reg wo1 = 0, wo2 = 0;
    always @(negedge aclk) if (aresetn) begin
        ncyc = ncyc + 1;
        if (o_l !== o_r || v_l !== v_r) begin d_leg = d_leg + 1; if (d_leg < 5) $display("LEG STREAM DIFF cyc %0d phase %0d", cyc, phase); end
        if ((o_n !== o_r || v_n !== v_r) && phase < 3) begin
            d_new = d_new + 1;
            // allowed only as the early blank: NEW zero, in the window's first two words
            if (o_n == 0 && v_n === v_r && (wo1 || wo2)) d_new_ok = d_new_ok + 1;
            else begin d_new_bad = d_new_bad + 1; if (d_new_bad < 5) $display("NEW STREAM DIFF cyc %0d phase %0d ref %h new %h", cyc, phase, o_r[31:0], o_n[31:0]); end
        end
        wo2 = wo1; wo1 = NEW.win_open;
    end

    // ---- record comparison, 60 cycles after each REF window closes
    integer r_leg = 0, r_new = 0, n_rec = 0, n_mis_new = 0, w;
    task cmp_records; begin
        n_rec = n_rec + 1;
        for (w = 1; w <= 18; w = w + 1) begin
            if (w == 11) continue;
            if (w == 10) begin
                if (mem_l[10][3:0] !== mem_r[10][3:0]) r_leg = r_leg + 1;
                if (phase < 3 && mem_n[10][3:0] !== mem_r[10][3:0]) begin r_new = r_new + 1; if (r_new < 5) $display("NEW REC DIFF w10 rec %0d", n_rec); end
            end else begin
                if (mem_l[w] !== mem_r[w]) begin r_leg = r_leg + 1; if (r_leg < 5) $display("LEG REC DIFF w%0d rec %0d: %h vs %h", w, n_rec, mem_l[w], mem_r[w]); end
                if (phase < 3 && mem_n[w] !== mem_r[w]) begin r_new = r_new + 1; if (r_new < 5) $display("NEW REC DIFF w%0d rec %0d: %h vs %h", w, n_rec, mem_n[w], mem_r[w]); end
            end
        end
        if (phase < 3 && (mem_n[19] != 0 || mem_n[10][4])) n_mis_new = n_mis_new + 1;
    end endtask
    always @(posedge aclk) if (aresetn && REF.win_close) fork begin repeat (60) @(posedge aclk); cmp_records; end join_none

    // ---- NN result model
    integer nn_cnt = 0;
    task automatic nn_result(input integer lat, input bit dbit);
        begin
            repeat (lat) @(posedge aclk);
            @(posedge s_aclk);
            res_o <= {dbit, 18'd0, 29'(nn_cnt), 16'($urandom)};
            nn_cnt = nn_cnt + 1;
            res_vld <= 1'b1;
            @(posedge s_aclk); res_vld <= 1'b0;
        end
    endtask

    integer mp, LEAD, DELAY, s, lnn, slack, mw, period, wcls, nshots = 0;
    reg en, p1, p0, ps; reg lose, close_trig, mid_cfg, second_mk;
    function [31:0] cfgw(input bit e, input bit a, input bit b, input bit c, input integer d);
        cfgw = {16'd500, 8'(d), 4'b0, c, b, a, e};
    endfunction
    task calibrate; begin
        wcfg(cfgw(0, 1, 0, 0, 0)); repeat (20) @(posedge aclk);
        @(posedge aclk) cls <= 1; repeat (5) @(posedge aclk); cls <= 0;
        fork nn_result(500, 1); join_none
        repeat (600) @(posedge aclk);
        @(posedge aclk) marker <= 1; repeat (200) @(posedge aclk); marker <= 0; repeat (300) @(posedge aclk);
        mp = mem_r[7];
    end endtask

    task one_shot(input integer ph); begin
        nshots = nshots + 1;
        // policy for this shot (written between shots, as the host does)
        en = ($urandom % 8) != 0; p1 = ($urandom % 6) != 0; p0 = ($urandom % 6) == 0; ps = ($urandom % 10) == 0;
        LEAD = (ph == 3) ? ($urandom % 13) : 4 + ($urandom % 9);
        DELAY = mp - LEAD; if (DELAY < 0) DELAY = 0;
        wcfg(cfgw(en, p1, p0, ps, DELAY));
        repeat (20) @(posedge aclk);
        lose = (ph == 3) && ($urandom % 6 == 0);
        close_trig = (ph == 3) && ($urandom % 8 == 0);
        mid_cfg = (ph == 3) && ($urandom % 6 == 0);
        second_mk = (ph == 3) && ($urandom % 10 == 0);
        lnn = 515 + $urandom % 30;                     // trigger -> result at the gate, ~ window + NN + CDC
        slack = $urandom % 241; slack = slack - 40;
        PLEN = 8 + $urandom % 60;
        mw = PLEN + LEAD + ($urandom % 30);            // window covers the pulse (gate_probe: cp_len + win_extra)
        wcls = 1 + $urandom % 400;
        // shot
        @(posedge aclk) cls <= 1;
        fork begin repeat (wcls) @(posedge aclk); cls <= 0; end join_none
        if (!lose) fork nn_result(lnn, $urandom % 2); join_none
        if (close_trig) begin                          // another classify trigger before the result
            repeat (100 + $urandom % 300) @(posedge aclk);
            @(posedge aclk) cls <= 0; @(posedge aclk) cls <= 1; @(posedge aclk) cls <= 0;
            fork nn_result(lnn, $urandom % 2); join_none
        end
        // window opens ~ marker + 4 + DELAY; aim it at result + slack
        repeat ((lnn + slack - DELAY - 4 > 1) ? lnn + slack - DELAY - 4 : 1) @(posedge aclk);
        @(posedge aclk) marker <= 1;
        if (mid_cfg) fork begin repeat (DELAY + 6 + $urandom % 20) @(posedge aclk);
                                wcfg(cfgw($urandom, $urandom, $urandom, $urandom, DELAY)); end join_none
        repeat (mw) @(posedge aclk); marker <= 0;
        if (second_mk) begin repeat (50 + $urandom % 100) @(posedge aclk); marker <= 1; repeat (mw) @(posedge aclk); marker <= 0; end
        period = 300 + $urandom % 1500;
        repeat (period) @(posedge aclk);
    end endtask

    initial begin
        #(30000) aresetn = 1;
        repeat (50) @(posedge aclk);
        noise = 0; phase = 0; calibrate;
        $display("calibration: word7 marker->pulse = %0d (GEN_LAT %0d)", mp, GEN_LAT);
        phase = 1; noise = 0;
        for (s = 0; s < N1; s = s + 1) one_shot(1);
        $display("phase 1 (normal, clean): shots %0d, stream cycles %0d | LEG diffs %0d rec %0d | NEW diffs %0d (early-blank %0d, other %0d) rec %0d, binding rejects %0d",
                 N1, ncyc, d_leg, r_leg, d_new, d_new_ok, d_new_bad, r_new, n_mis_new);
        if (d_new != 0) $display("  ERROR: NEW differs from REF in normal operation");
        phase = 2; noise = 1;
        for (s = 0; s < N2; s = s + 1) one_shot(2);
        $display("phase 2 (normal, noise):  cumulative | LEG diffs %0d rec %0d | NEW diffs %0d (early-blank %0d, other %0d) rec %0d, binding rejects %0d",
                 d_leg, r_leg, d_new, d_new_ok, d_new_bad, r_new, n_mis_new);
        phase = 3; noise = 1;
        for (s = 0; s < N3; s = s + 1) one_shot(3);
        repeat (500) @(posedge aclk);
        $display("phase 3 (abnormal):       cumulative | LEG diffs %0d rec %0d (NEW not compared)", d_leg, r_leg);
        $display("records compared %0d; REF windows %0d passed %0d blanked %0d stale %0d late %0d cls %0d; NEW binding rejects total %0d",
                 n_rec, mem_r[1], mem_r[2], mem_r[3], mem_r[4], mem_r[12], mem_r[14], mem_n[19]);
        if (d_leg == 0 && r_leg == 0 && d_new_bad == 0 && r_new == 0 && n_mis_new == 0) $display("RESULT: PASS");
        else $display("RESULT: FAIL");
        $finish;
    end
endmodule
