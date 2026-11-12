// (neural_net4_live576s copy, v1.40L10: N = 576, vectors from this project's
// tb_data; run_tb.sh <HLS project dir> runs it on NN_live 1.3 or 1.4.)
// RTL testbench: live_window_gate + the HLS-generated NN_live, on a
// continuous 552.96 MHz stream with an asynchronous tProc-style trigger.
//
// Test A (alignment): the stream carries its own sample index. Every trigger
// is seen by the gate and by QICK's real buffer.sv (axis_avg_buffer's
// decimated-buffer FSM, behind the same 2-flop synchronizer avg_buffer.v
// uses). The words the gate passes must equal the words buffer.sv stores,
// 770 per trigger, TLAST on the last.
// Test B (inference): the source places real windows exactly where the gate
// captures. Each result must equal the v1.40 emulator's output and each
// stored first-layer sum the integer dot product (expected.hex), with no
// word ever offered while TREADY is low. Latency is counted from the window's
// last sample to res_o_ap_vld.
// With +define+FEEDBACK (run_sim_fb.bat, v1.40L2): res_o also goes through
// live_res_axis and the generated axis_clock_converter (OG_build_axis_cc_live_0,
// 552.96 -> 100 MHz) into a model of the tProc input register
// (qproc_inport_reg: latch tdata on tvalid, one c_clk register; DPORT_RD reads
// it). Each result must arrive there intact, and the latency is counted from
// the gate's TLAST beat to that register's update, in ns.

`timescale 1ps / 1ps

module tb_live;
    localparam integer N = 576;
    localparam integer K = 24;          // test B windows (make_rtl_vectors.py)
    localparam integer KA = 12;         // test A triggers
    localparam integer GAP = 1500;      // aclk cycles between triggers (~2.7 us)

    reg aclk = 0, tclk = 0;
    always #904 aclk = ~aclk;           // 1.808 ns: clk_adc0_x2, 552.96 MHz
    always #1017 tclk = ~tclk;          // 2.034 ns: tProc t_clk, 491.52 MHz
    reg aresetn = 0;

    integer cyc = 0;
    always @(posedge aclk) cyc <= cyc + 1;

    // ---------------- stream source ----------------
    reg  [31:0] s_tdata = 0;
    reg         s_tvalid = 0;
    reg  [31:0] idx = 0;
    integer     mode = 0;               // 0: index stream, 1: real windows
    integer     win_sel = 0, src_cnt = 0;
    reg  [31:0] win [0:K*N-1];

    // predicts the gate's next state from its pre-edge state, so window word 0
    // is on the bus during the gate's first MEMW cycle
    wire will_memw = (gate.state == 2'd0 && gate.trig_r) || (gate.state == 2'd1 && !gate.done);

    always @(posedge aclk) begin
        idx <= idx + 1;
        s_tvalid <= aresetn;
        if (mode == 0) begin
            s_tdata <= idx;
        end else if (will_memw) begin
            s_tdata <= win[win_sel * N + src_cnt];
            src_cnt <= src_cnt + 1;
        end else begin
            s_tdata <= {idx[15:0], ~idx[15:0]};   // filler between windows
            src_cnt <= 0;
        end
    end

    // ---------------- trigger (tProc timing clock) ----------------
    reg trig = 0;
    task fire;
        begin
            @(posedge tclk);
            trig <= 1;
            repeat (20) @(posedge tclk);
            trig <= 0;
        end
    endtask

    // ---------------- DUT: gate + kernel ----------------
    wire [31:0] g_tdata;
    wire        g_tvalid, g_tlast, g_tready;
    wire [3:0]  g_tkeep, g_tstrb;
    live_window_gate #(.N(N)) gate (
        .aclk(aclk), .aresetn(aresetn), .trigger(trig),
        .s_axis_tdata(s_tdata), .s_axis_tvalid(s_tvalid),
        .m_axis_tdata(g_tdata), .m_axis_tvalid(g_tvalid), .m_axis_tlast(g_tlast),
        .m_axis_tkeep(g_tkeep), .m_axis_tstrb(g_tstrb), .m_axis_tready(g_tready));

    wire [63:0] res_o;
    wire        res_vld;
    wire [31:0] b_addr, b_din;
    wire        b_en, b_clk, b_rst;
    wire [3:0]  b_we;
    reg  [31:0] b_dout = 0;
    NN_live dut (
        .ap_clk(aclk), .ap_rst_n(aresetn),
        .in_r_TDATA(g_tdata), .in_r_TVALID(g_tvalid), .in_r_TREADY(g_tready),
        .in_r_TKEEP(g_tkeep), .in_r_TSTRB(g_tstrb), .in_r_TLAST(g_tlast),
        .res_o(res_o), .res_o_ap_vld(res_vld),
        .out_r_Addr_A(b_addr), .out_r_EN_A(b_en), .out_r_WEN_A(b_we),
        .out_r_Din_A(b_din), .out_r_Dout_A(b_dout), .out_r_Clk_A(b_clk), .out_r_Rst_A(b_rst));


    reg [31:0] mem [0:8191];
    always @(posedge aclk) if (b_en) begin
        if (b_we != 0) mem[b_addr[14:2]] <= b_din;
        b_dout <= mem[b_addr[14:2]];
    end

    // ---------------- reference: QICK buffer.sv ----------------
    (* ASYNC_REG = "TRUE" *) reg [1:0] ref_sync = 0;
    always @(posedge aclk) ref_sync <= aresetn ? {ref_sync[0], trig} : 2'b00;
    wire        ref_we;
    wire [9:0]  ref_addr;
    wire [31:0] ref_di;
    buffer #(.N(10), .B(16)) ref_buf (
        .rstn(aresetn), .clk(aclk), .trigger_i(ref_sync[1]),
        .din_valid_i(s_tvalid), .din_i(s_tdata),
        .mem_we_o(ref_we), .mem_addr_o(ref_addr), .mem_di_o(ref_di),
        .START_REG(1'b1), .ADDR_REG(10'd0), .LEN_REG(10'd576));

    // ---------------- monitors ----------------
    reg [31:0] ref_q  [0:KA*N-1];
    reg [31:0] gate_q [0:KA*N-1];
    integer n_ref = 0, n_gate = 0, gate_words_total = 0, tlast_bad = 0, drops = 0;
    integer t_last [0:63];
    time    t_last_ps [0:63];           // same event as t_last, in ps (FEEDBACK latency)
    integer last_mode [0:63], last_win [0:63];
    integer n_last = 0, n_res = 0, bad_val = 0, lat_min = 1 << 30, lat_max = 0;
    reg [17:0] exp_sig [0:K-1];
    reg [63:0] res_log [0:63];
    reg [31:0] exp_sum [0:K-1][0:3];

    always @(posedge aclk) begin
        if (ref_we && mode == 0 && n_ref < KA * N) begin
            ref_q[n_ref] = ref_di;
            n_ref = n_ref + 1;
        end
        if (g_tvalid) begin
            if (!g_tready) drops = drops + 1;
            if (mode == 0 && n_gate < KA * N) begin
                gate_q[n_gate] = g_tdata;
                n_gate = n_gate + 1;
            end
            if (g_tlast != ((gate_words_total % N) == N - 1)) tlast_bad = tlast_bad + 1;
            gate_words_total = gate_words_total + 1;
            if (g_tlast) begin
                t_last[n_last] = cyc;
                t_last_ps[n_last] = $time;
                last_mode[n_last] = mode;
                last_win[n_last] = win_sel;
                n_last = n_last + 1;
            end
        end
        if (res_vld) begin : chk
            integer lat;
            lat = cyc - t_last[n_res];
            if (lat < lat_min) lat_min = lat;
            if (lat > lat_max) lat_max = lat;
            res_log[n_res] = res_o;
            if (last_mode[n_res] == 1) begin
                if (res_o[63] !== (exp_sig[last_win[n_res]] > 18'd512) || res_o[44:16] !== n_res) begin
                    bad_val = bad_val + 1;
                    $display("MISMATCH window %0d: res %h expected sigmoid %h", last_win[n_res], res_o, exp_sig[last_win[n_res]]);
                end
            end
            if (n_res < 3 || last_mode[n_res] == 1 && last_win[n_res] < 3)
                $display("result %0d (mode %0d win %0d): logit raw %0d decision %0d, latency %0d cycles after the gate's TLAST beat",
                         n_res, last_mode[n_res], last_win[n_res], $signed(res_o[15:0]), res_o[63], lat);
            n_res = n_res + 1;
        end
    end

`ifdef FEEDBACK
    // ---------------- feedback path to the tProc (v1.40L2) ----------------
    reg cclk = 0;
    always #5000 cclk = ~cclk;          // 10 ns: pl_clk0, the tProc core clock
    reg cresetn = 0;
    always @(posedge cclk) cresetn <= aresetn;
    wire [63:0] ra_tdata, cc_tdata;
    wire        ra_tvalid, ra_tready, cc_tvalid;
    live_res_axis res_axis (
        .aclk(aclk), .aresetn(aresetn), .res_o(res_o), .res_o_ap_vld(res_vld),
        .m_axis_tdata(ra_tdata), .m_axis_tvalid(ra_tvalid), .m_axis_tready(ra_tready));
    OG_build_axis_cc_live_0 cc (
        .s_axis_aresetn(aresetn), .m_axis_aresetn(cresetn),
        .s_axis_aclk(aclk), .s_axis_tvalid(ra_tvalid), .s_axis_tready(ra_tready), .s_axis_tdata(ra_tdata),
        .m_axis_aclk(cclk), .m_axis_tvalid(cc_tvalid), .m_axis_tready(1'b1), .m_axis_tdata(cc_tdata));
    reg [63:0] port_dt_r = 0;           // qproc_inport_reg
    reg        port_new = 0;
    always @(posedge cclk) if (cc_tvalid) begin port_dt_r <= cc_tdata; port_new <= 1; end
    integer n_fb = 0, bad_fb = 0, ra_drop = 0;
    real    fb_min = 1e9, fb_max = 0;
    always @(posedge aclk) if (ra_tvalid && !ra_tready) ra_drop = ra_drop + 1;
    always @(posedge cclk) if (cc_tvalid) begin : fbchk
        real ns;
        reg [63:0] exp;
        #1;                              // after the register update at this edge
        ns = ($time - 1 - t_last_ps[n_fb]) / 1000.0;
        if (ns < fb_min) fb_min = ns;
        if (ns > fb_max) fb_max = ns;
        exp = {res_log[n_fb][63], 2'b00, res_log[n_fb][44:16], {16{res_log[n_fb][15]}}, res_log[n_fb][15:0]};
        if (port_dt_r !== exp) begin
            bad_fb = bad_fb + 1;
            $display("FEEDBACK MISMATCH result %0d: port %h expected %h", n_fb, port_dt_r, exp);
        end
        if (n_fb < 3)
            $display("feedback %0d: tProc port 3 = %h (s_port_l %0d = logit %.4f, shot %0d, decision %0d), %.1f ns after the gate's TLAST beat",
                     n_fb, port_dt_r, $signed(port_dt_r[31:0]), $signed(port_dt_r[31:0]) / 1024.0,
                     port_dt_r[60:32], port_dt_r[63], ns);
        n_fb = n_fb + 1;
    end
`endif

    // ---------------- sequence ----------------
    integer k, u, fd, r, bad_rec, bad_cap;
    reg [31:0] s0, s1, s2, s3;
    reg [17:0] sg;
    initial begin
        $readmemh("windows.hex", win);
        fd = $fopen("expected.hex", "r");
        for (k = 0; k < K; k = k + 1) begin
            r = $fscanf(fd, "%h %h %h %h %h\n", sg, s0, s1, s2, s3);
            exp_sig[k] = sg;
            exp_sum[k][0] = s0; exp_sum[k][1] = s1; exp_sum[k][2] = s2; exp_sum[k][3] = s3;
        end
        $fclose(fd);

        repeat (40) @(posedge aclk);
        aresetn = 1;
        repeat (200) @(posedge aclk);

        // Test A: index stream, triggers at drifting phases (t_clk vs aclk)
        mode = 0;
        for (k = 0; k < KA; k = k + 1) begin
            #(k * 137);                 // extra phase walk
            fire();
            repeat (GAP) @(posedge aclk);
        end

        // Test B: real windows
        mode = 1;
        for (k = 0; k < K; k = k + 1) begin
            win_sel = k;
            fire();
            repeat (GAP) @(posedge aclk);
        end
        repeat (200) @(posedge aclk);

        // ---- checks ----
        bad_cap = 0;
        for (k = 0; k < KA * N; k = k + 1)
            if (gate_q[k] !== ref_q[k]) bad_cap = bad_cap + 1;
        bad_rec = 0;
        for (k = 0; k < K; k = k + 1) begin : rec
            integer shot, base;
            shot = KA + k;
            base = 1024 + (shot % 512) * 8;
            if (mem[base] !== shot || mem[base + 6] !== {14'd0, exp_sig[k]} ||
                mem[base + 5] !== {16'd0, res_log[shot][15:0]} || mem[base + 7] !== {31'd0, res_log[shot][63]})
                bad_rec = bad_rec + 1;
            for (u = 0; u < 4; u = u + 1)
                if (mem[base + 1 + u] !== exp_sum[k][u]) bad_rec = bad_rec + 1;
        end

        $display("");
        $display("=== tb_live ===");
        $display("A  gate vs buffer.sv words      : %0d/%0d captured each side, %0d mismatches", n_gate, n_ref, bad_cap);
        $display("   TLAST placement errors       : %0d (of %0d gate words)", tlast_bad, gate_words_total);
        $display("B  decisions vs emulator        : %0d mismatches (of %0d)", bad_val, K);
        $display("   records (sums, probability, logit = res_o, shot): %0d mismatches", bad_rec);
        $display("   words offered with TREADY low: %0d", drops);
        $display("   header: shots %0d (expect %0d), frame_err %0d, magic %h", mem[0], KA + K, mem[1], mem[2]);
        $display("   results seen %0d of %0d windows", n_res, n_last);
        $display("LATENCY gate TLAST beat -> res_o_ap_vld (logit + decision): %0d..%0d cycles = %.1f..%.1f ns at 552.96 MHz",
                 lat_min, lat_max, lat_min * 1.80845, lat_max * 1.80845);
        $display("        (the gate adds 1 register cycle after the last sample leaves the readout)");
`ifdef FEEDBACK
        $display("FEEDBACK results at tProc input 3: %0d of %0d, %0d mismatches, %0d beats refused by the converter",
                 n_fb, n_res, bad_fb, ra_drop);
        $display("LATENCY gate TLAST beat -> tProc input register (100 MHz core): %.1f..%.1f ns", fb_min, fb_max);
        if (n_fb != n_res || bad_fb != 0 || ra_drop != 0) $display("RESULT: FAIL (feedback)");
`endif
        if (bad_cap == 0 && n_gate == KA * N && n_ref == KA * N && tlast_bad == 0 && bad_val == 0 &&
            bad_rec == 0 && drops == 0 && mem[0] == KA + K && mem[1] == 0 && n_res == n_last && n_last == KA + K)
            $display("RESULT: PASS");
        else
            $display("RESULT: FAIL");
        $finish;
    end
endmodule
