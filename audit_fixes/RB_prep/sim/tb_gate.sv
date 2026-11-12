// RTL sim of dac_gate (v1.40L6b) with its decision path: NN_live result strobes on
// the readout clock (552.96 MHz) -> live_res_cdc -> dac_gate on clk_dac2
// (491.52 MHz), against a BRAM model (READ_LATENCY 1).
// Generator model: a pulse of PLEN stream words (all 16 samples non-zero) starts
// GEN_LAT cycles after the marker's rising edge (both at the gate input); otherwise small noise (|x| <= 3).
// Shots:
//   1 decision 1, PASS1           -> pulse passed, unchanged
//   2 decision 0, PASS0 = 0       -> pulse blanked (all-zero output over it)
//   3 no decision                 -> stale, blanked (PASS_STALE = 0)
//   4 decision 0, gate disabled   -> pulse passed; statistics still kept
//   5 decision 1 arrives after the window opened -> late: stale, blanked, counted
//   6 no new decision             -> stale, blanked (v1.40L6 used shot 5's late decision)
//   7 decision 1 arrives after shot 6's window closed, before shot 7's classify
//     trigger -> late, discarded; shot 7 stale, blanked
//   8 decision 1, PASS1           -> passed again (recovery)
// Every shot starts with the classify trigger (cls_trig), as on hardware.
// Everywhere outside a blanked window: output == input one cycle earlier.
// Checks the statistics words after each shot (margin, marker -> pulse, flags).
// v1.40L9 (GAT3): words 15-18 are absolute counter values (decision, marker,
// window open, classify trigger); window open - decision must equal the margin
// (word 6) on fresh shots, the decision word must be 0xFFFFFFFF on stale ones,
// and cls < marker < window open.
`timescale 1ps / 1ps

module tb_gate;
    localparam integer NS = 16;
    localparam integer D_PS = 2034, S_PS = 1808;
    localparam integer GEN_LAT = 40, PLEN = 50, DELAY = 30, MK_W = 120;

    reg aclk = 0, s_aclk = 0, aresetn = 0;
    always #(D_PS / 2) aclk = ~aclk;
    initial begin #(777); forever #(S_PS / 2) s_aclk = ~s_aclk; end

    // decision path
    reg  [63:0] res_o = 0;
    reg         res_vld = 0;
    wire [63:0] dec_tdata;
    wire        dec_tvalid;
    live_res_cdc cdc (.s_aclk(s_aclk), .s_aresetn(aresetn), .res_o(res_o), .res_o_ap_vld(res_vld),
                      .m_aclk(aclk), .m_aresetn(aresetn),
                      .m_axis_tdata(dec_tdata), .m_axis_tvalid(dec_tvalid), .m_axis_tready(1'b1));

    // stream + marker
    reg  [16*NS-1:0] s_tdata = 0;
    wire [16*NS-1:0] m_tdata;
    wire             m_tvalid;
    reg              marker = 0;
    reg              cls = 0;
    wire bclk, brst, ben; wire [3:0] bwe; wire [31:0] baddr, bdin; reg [31:0] bdout = 0;

    dac_gate #(.NS(NS)) dut (.aclk(aclk), .aresetn(aresetn),
        .s_axis_tdata(s_tdata), .s_axis_tvalid(1'b1), .s_axis_tready(),
        .m_axis_tdata(m_tdata), .m_axis_tvalid(m_tvalid), .m_axis_tready(1'b1),
        .dec_axis_tdata(dec_tdata), .dec_axis_tvalid(dec_tvalid), .dec_axis_tready(),
        .marker(marker), .cls_trig(cls),
        .bram_clk(bclk), .bram_rst(brst), .bram_en(ben), .bram_we(bwe),
        .bram_addr(baddr), .bram_din(bdin), .bram_dout(bdout));

    reg [31:0] mem [0:2047];
    integer i;
    initial for (i = 0; i < 2048; i = i + 1) mem[i] = 0;
    always @(posedge bclk) if (ben) begin
        if (bwe == 4'hF) begin mem[baddr[12:2]] <= bdin; bdout <= bdin; end
        else bdout <= mem[baddr[12:2]];
    end

    // generator model, on clk_dac2
    integer cyc = 0, mk_cyc = -100000, p_start = -100000;
    always @(posedge aclk) cyc <= cyc + 1;
    reg marker_q = 0;
    always @(posedge aclk) begin
        marker_q <= marker;
        if (marker && !marker_q) p_start <= cyc + GEN_LAT;
    end
    wire in_pulse = (cyc >= p_start) && (cyc < p_start + PLEN);
    integer j;
    always @(posedge aclk) begin
        for (j = 0; j < NS; j = j + 1)
            s_tdata[16*j +: 16] <= in_pulse ? (16'd1000 + j + (cyc[7:0] << 4)) : ($random % 4);
    end

    // checker: output vs input one cycle earlier; blank expectation per shot
    reg in_pulse_q = 0;
    reg pulse_word_d = 0;
    always @(posedge aclk) pulse_word_d <= in_pulse_q;
    // cycles around a window, where a blanked window may also zero the noise words
    wire near_window = (cyc >= mk_cyc) && (cyc < mk_cyc + MK_W + DELAY + 10);
    reg [16*NS-1:0] s_prev = 0;
    reg  expect_blank = 0;           // the current shot's pulse should be blanked
    integer errors = 0, n_pulse_out = 0, n_pulse_nonzero = 0, n_mismatch = 0;
    always @(posedge aclk) begin
        s_prev     <= s_tdata;
        in_pulse_q <= in_pulse;
        // m_tdata now holds the word that was on s_tdata one cycle ago (s_prev);
        // that word was a pulse word iff in_pulse was true two cycles ago
        if (aresetn) begin
            if (pulse_word_d) begin
                n_pulse_out = n_pulse_out + 1;
                if (m_tdata != 0) n_pulse_nonzero = n_pulse_nonzero + 1;
                if (!expect_blank && m_tdata !== s_prev) n_mismatch = n_mismatch + 1;
                if (expect_blank && m_tdata != 0) n_mismatch = n_mismatch + 1;
            end else if (!near_window && m_tdata !== s_prev) begin
                n_mismatch = n_mismatch + 1;
            end
        end
    end

    task send_decision(input bit dbit, input [28:0] shot);
        begin
            @(posedge s_aclk);
            res_o   <= {dbit, 18'd0, shot[28:0] , 16'h0123};    // [63] bit, [44:16] shot, [15:0] logit
            res_vld <= 1'b1;
            @(posedge s_aclk);
            res_vld <= 1'b0;
        end
    endtask

    task fire_cls;
        begin
            @(posedge aclk); cls <= 1'b1;
            repeat (10) @(posedge aclk); cls <= 1'b0;
            repeat (20) @(posedge aclk);
        end
    endtask

    task fire_marker;
        begin
            @(posedge aclk);
            marker <= 1'b1; mk_cyc = cyc;
            repeat (MK_W) @(posedge aclk);
            marker <= 1'b0;
            repeat (400) @(posedge aclk);
        end
    endtask

    task check_shot(input integer shot, input bit exp_pass, input bit exp_stale, input bit exp_en);
        reg [31:0] f;
        begin
            f = mem[10];
            $display("shot %0d: windows %0d passed %0d blanked %0d stale %0d decisions %0d margin %0d marker->pulse %0d flags %h | pulse words out %0d, non-zero %0d, mismatches %0d",
                     shot, mem[1], mem[2], mem[3], mem[4], mem[5], $signed(mem[6]), mem[7], f, n_pulse_out, n_pulse_nonzero, n_mismatch);
            if (mem[1] != shot) begin errors = errors + 1; $display("  ERROR window count"); end
            if (f[0] != exp_pass || f[1] != exp_stale || f[3] != exp_en) begin errors = errors + 1; $display("  ERROR flags"); end
            if (mem[7] != GEN_LAT - 2) begin errors = errors + 1; $display("  ERROR marker->pulse %0d (exp %0d)", mem[7], GEN_LAT - 2); end
            if (!exp_stale && ($signed(mem[6]) <= 0)) begin errors = errors + 1; $display("  ERROR margin"); end
            $display("  L9 stamps: dec-cls %0d  mk-cls %0d  open-cls %0d  open-dec %0d",
                     $signed(mem[15] - mem[18]), mem[16] - mem[18], mem[17] - mem[18], $signed(mem[17] - mem[15]));
            if (!exp_stale && (mem[17] - mem[15] != mem[6])) begin errors = errors + 1; $display("  ERROR open - dec != margin"); end
            if (exp_stale && mem[15] != 32'hFFFFFFFF) begin errors = errors + 1; $display("  ERROR stale decision stamp"); end
            if (!($signed(mem[16] - mem[18]) > 0 && $signed(mem[17] - mem[16]) > 0)) begin errors = errors + 1; $display("  ERROR stamp order"); end
            if (n_mismatch != 0) begin errors = errors + 1; $display("  ERROR %0d output mismatches", n_mismatch); end
            if (n_pulse_out != PLEN) begin errors = errors + 1; $display("  ERROR pulse words seen %0d", n_pulse_out); end
            // flags record the gate's decision; with the gate disabled the output still passes
            if ((exp_pass || !exp_en) && n_pulse_nonzero != PLEN) begin errors = errors + 1; $display("  ERROR pulse not passed"); end
            if (!(exp_pass || !exp_en) && n_pulse_nonzero != 0) begin errors = errors + 1; $display("  ERROR pulse not blanked"); end
            n_pulse_out = 0; n_pulse_nonzero = 0; n_mismatch = 0;
        end
    endtask

    // config word: [0] EN [1] PASS1 [2] PASS0 [3] PASS_STALE [15:8] DELAY [31:16] NZ_THR
    function [31:0] cfgw(input bit en);
        cfgw = {16'd100, 8'(DELAY), 4'b0, 1'b0, 1'b0, 1'b1, en};
    endfunction

    initial begin
        #(30000) aresetn = 1;
        mem[0] = cfgw(1);
        repeat (5) @(posedge aclk);
        n_mismatch = 0;                      // the first cycle out of reset
        #(50000);
        // passthrough before any window
        repeat (300) @(posedge aclk);
        if (n_mismatch != 0) begin errors = errors + 1; $display("ERROR passthrough before windows: %0d", n_mismatch); end
        n_mismatch = 0;

        expect_blank = 0; fire_cls; send_decision(1, 1); repeat (20) @(posedge aclk); fire_marker; check_shot(1, 1, 0, 1);
        expect_blank = 1; fire_cls; send_decision(0, 2); repeat (20) @(posedge aclk); fire_marker; check_shot(2, 0, 0, 1);
        expect_blank = 1; fire_cls;            repeat (20) @(posedge aclk); fire_marker; check_shot(3, 0, 1, 1);
        mem[0] = cfgw(0); repeat (10) @(posedge aclk);
        expect_blank = 0; fire_cls; send_decision(0, 4); repeat (20) @(posedge aclk); fire_marker; check_shot(4, 0, 0, 0);
        mem[0] = cfgw(1); repeat (10) @(posedge aclk);
        // 5: the decision comes after the window opened
        expect_blank = 1; fire_cls; repeat (20) @(posedge aclk);
        fork fire_marker; begin repeat (DELAY + 12) @(posedge aclk); send_decision(1, 5); end join
        check_shot(5, 0, 1, 1);
        if (mem[12] != 1) begin errors = errors + 1; $display("  ERROR late count %0d (exp 1)", mem[12]); end
        $display("  late decisions %0d, last late by %0d cycles", mem[12], mem[13]);
        // 6: no new decision: must not use shot 5's late one
        expect_blank = 1; fire_cls; repeat (20) @(posedge aclk); fire_marker; check_shot(6, 0, 1, 1);
        // 7: decision between shot 6's window and shot 7's classify trigger
        send_decision(1, 7); repeat (40) @(posedge aclk);
        expect_blank = 1; fire_cls; repeat (20) @(posedge aclk); fire_marker; check_shot(7, 0, 1, 1);
        if (mem[12] != 2) begin errors = errors + 1; $display("  ERROR late count %0d (exp 2)", mem[12]); end
        // 8: normal again
        expect_blank = 0; fire_cls; send_decision(1, 8); repeat (20) @(posedge aclk); fire_marker; check_shot(8, 1, 0, 1);
        if (mem[14] != 8) begin errors = errors + 1; $display("ERROR classify triggers %0d (exp 8)", mem[14]); end
        if (mem[11] != 32'h47415433) begin errors = errors + 1; $display("ERROR magic"); end
        if (errors == 0) $display("RESULT: PASS"); else $display("RESULT: FAIL (%0d errors)", errors);
        $finish;
    end
endmodule
