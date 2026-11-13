// tb_seg: trig_sync + live_window_gate (N = 576) + rf_tap + seg_stamp, with a BRAM model (READ_LATENCY 1),
// on a continuous 552.96 MHz stream.
// Per shot: an asynchronous trigger; an input pulse (|I|+|Q| = 3000) from stream index P1 after the trigger
// and a second one (the conditional pulse) from P2; a fake NN_live result RES_LAT cycles after TLAST. A raw
// 8-sample tile stream carries the same pulses RF_LEAD cycles earlier into rf_tap. Shots 1-2 are 3000 cycles
// apart (record closed by the next trigger); shot 3 is closed by the CLOSE_CYC timeout.
// Checks per record: last - first = N-1, res - last = RES_LAT, edges against the stimulus, edge3 - edge0 =
// P2 - P1, edge1 - edge0 = edge2 - edge3 = RF_LEAD - 3, word count, flags, header, magic, rf_thr readback.
`timescale 1ps / 1ps

module tb_seg;
    localparam integer N = 576;
    localparam integer RES_LAT = 14;
    localparam integer P1 = 100, P2 = 900, PLEN = 50;
    localparam integer RF_LEAD = 20;      // tile output leads the readout output
    localparam integer CLK_PS = 1808;

    reg aclk = 0, aresetn = 0;
    always #(CLK_PS / 2) aclk = ~aclk;

    reg  trigger = 0;
    reg  [31:0] s_tdata = 0;
    wire [31:0] m_tdata; wire m_tvalid, m_tlast; wire [3:0] m_tkeep, m_tstrb;
    wire dbg_trig, dbg_first, dbg_last, dbg_dvalid; wire [31:0] dbg_data;

    wire trig_s;
    trig_sync tsync (.aclk(aclk), .trig_in(trigger), .trig_out(trig_s));

    live_window_gate #(.N(N)) gate (
        .aclk(aclk), .aresetn(aresetn), .trigger(trig_s),
        .s_axis_tdata(s_tdata), .s_axis_tvalid(1'b1),
        .m_axis_tdata(m_tdata), .m_axis_tvalid(m_tvalid), .m_axis_tlast(m_tlast),
        .m_axis_tkeep(m_tkeep), .m_axis_tstrb(m_tstrb), .m_axis_tready(1'b1),
        .dbg_trig(dbg_trig), .dbg_first(dbg_first), .dbg_last(dbg_last),
        .dbg_data(dbg_data), .dbg_dvalid(dbg_dvalid));

    // fake NN_live result strobe
    reg [RES_LAT:0] lsr = 0;
    always @(posedge aclk) lsr <= {lsr[RES_LAT-1:0], m_tlast & m_tvalid};
    wire res_vld = lsr[RES_LAT-1];

    // raw tile stream: 8 real samples per cycle
    reg  [127:0] rf_tdata = 0;
    wire [127:0] rf_m_tdata; wire rf_m_tvalid, rf_s_tready, rf_hit; wire [15:0] rf_thr;
    rf_tap #(.NS(8)) rft (
        .aclk(aclk), .s_axis_tdata(rf_tdata), .s_axis_tvalid(1'b1), .s_axis_tready(rf_s_tready),
        .m_axis_tdata(rf_m_tdata), .m_axis_tvalid(rf_m_tvalid), .m_axis_tready(1'b1),
        .thr(rf_thr), .hit(rf_hit));

    wire bclk, brst, ben; wire [3:0] bwe; wire [31:0] baddr, bdin; reg [31:0] bdout = 0;
    seg_stamp #(.N_REC(64), .CLOSE_CYC(4095)) dut (
        .aclk(aclk), .aresetn(aresetn),
        .dbg_trig(dbg_trig), .dbg_first(dbg_first), .dbg_last(dbg_last),
        .dbg_data(dbg_data), .dbg_dvalid(dbg_dvalid), .res_vld(res_vld),
        .rf_hit(rf_hit), .rf_thr(rf_thr),
        .bram_clk(bclk), .bram_rst(brst), .bram_en(ben), .bram_we(bwe),
        .bram_addr(baddr), .bram_din(bdin), .bram_dout(bdout));

    reg [31:0] mem [0:2047];
    integer k;
    initial for (k = 0; k < 2048; k = k + 1) mem[k] = 0;
    always @(posedge bclk) if (ben) begin
        if (bwe == 4'hF) begin mem[baddr[12:2]] <= bdin; bdout <= bdin; end   // write-first
        else bdout <= mem[baddr[12:2]];
    end

    // stream: index counts cycles since the gate's first taken word would be
    // impossible to know here, so pulses are placed relative to the cycle the
    // trigger was asserted, and the expected edges are computed from dbg_first.
    integer cyc = 0, trig_cyc = -100000;
    always @(posedge aclk) cyc <= cyc + 1;
    always @(posedge aclk) begin
        if ((cyc - trig_cyc >= P1 && cyc - trig_cyc < P1 + PLEN) ||
            (cyc - trig_cyc >= P2 && cyc - trig_cyc < P2 + PLEN))
            s_tdata <= {16'd1000, 16'hF830};      // Q = 1000, I = -2000 -> |I|+|Q| = 3000
        else
            s_tdata <= {16'd3, 16'hFFFE};         // small noise, |I|+|Q| = 5
        // the tile stream: one sample of the word at -1500 (|x| > 1000), the rest noise
        if ((cyc - trig_cyc >= P1 - RF_LEAD && cyc - trig_cyc < P1 - RF_LEAD + PLEN) ||
            (cyc - trig_cyc >= P2 - RF_LEAD && cyc - trig_cyc < P2 - RF_LEAD + PLEN))
            rf_tdata <= {16'd7, 16'hFFF9, 16'd2, 16'hFA24, 16'd5, 16'hFFFB, 16'd1, 16'd0};
        else
            rf_tdata <= {16'd7, 16'hFFF9, 16'd2, 16'hFFFE, 16'd5, 16'hFFFB, 16'd1, 16'd0};
        if (rf_m_tdata !== rf_tdata) $display("ERROR rf_tap passthrough");
    end

    task shot(input integer gap_after);
        begin
            #(CLK_PS * 3 + 517);                 // asynchronous to aclk
            trigger = 1; trig_cyc = cyc;
            #(CLK_PS * 20);
            trigger = 0;
            #(CLK_PS * gap_after);
        end
    endtask

    integer errors = 0;
    integer fe0 = -1;
    task check(input integer r, input integer exp_shot);
        integer b; reg [31:0] w [0:15]; integer j;
        begin
            b = 16 + 16 * r;
            for (j = 0; j < 16; j = j + 1) w[j] = mem[b + j];
            $display("rec %0d: shot %0d trig %0d first %0d last %0d res %0d edge1 %0d edge2 %0d words %0d flags %02x edge0 %0d edge3 %0d",
                     r, w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7][23:8], w[7][7:0], w[8], w[9]);
            if (w[0] != exp_shot) begin errors = errors + 1; $display("  ERROR shot"); end
            if (w[3] - w[2] != N - 1) begin errors = errors + 1; $display("  ERROR last-first %0d", w[3] - w[2]); end
            if (w[4] - w[3] != RES_LAT) begin errors = errors + 1; $display("  ERROR res-last %0d", w[4] - w[3]); end
            if (w[7][7:0] != 8'h7F) begin errors = errors + 1; $display("  ERROR flags"); end
            if (w[9] - w[8] != P2 - P1) begin errors = errors + 1; $display("  ERROR edge3-edge0 %0d", w[9] - w[8]); end
            if (w[5] - w[8] != w[6] - w[9]) begin errors = errors + 1; $display("  ERROR front end differs: %0d vs %0d", w[5] - w[8], w[6] - w[9]); end
            if (fe0 < 0) fe0 = w[5] - w[8];
            else if (w[5] - w[8] != fe0) begin errors = errors + 1; $display("  ERROR front end not constant"); end
            for (j = 10; j < 16; j = j + 1) if (w[j] != 0) begin errors = errors + 1; $display("  ERROR reserved word %0d", j); end
            if (w[7][23:8] != N) begin errors = errors + 1; $display("  ERROR words %0d", w[7][23:8]); end
            // edges: the stream sample at trigger cycle + P lands in mag two
            // cycles later (dbg_data register + mag register), stamped with its
            // own counter value; first - trig is the sync + gate latency.
            if (w[5] > w[2] + P1 || w[5] + 8 < w[2] + P1 - 8) begin errors = errors + 1; $display("  ERROR edge1"); end
            if (w[6] <= w[3]) begin errors = errors + 1; $display("  ERROR edge2 not after last"); end
            if (w[6] - w[5] != P2 - P1) begin errors = errors + 1; $display("  ERROR edge2-edge1 %0d (exp %0d)", w[6] - w[5], P2 - P1); end
        end
    endtask

    initial begin
        #(CLK_PS * 10) aresetn = 1;
        #(CLK_PS * 5);
        mem[1] = 32'd2000;                       // host writes the threshold
        mem[3] = 32'h00000A31;                   // and the boxcar_tap cfg word
        mem[4] = 32'd1000;                       // and the rf_tap threshold
        #(CLK_PS * 10);
        shot(3000);
        shot(3000);
        shot(6000);                              // closed by the timeout
        #(CLK_PS * 100);
        $display("header: records %0d, thr %0d, magic %08x", mem[0], mem[1], mem[2]);
        if (mem[0] != 3) begin errors = errors + 1; $display("ERROR record count"); end
        if (mem[2] != 32'h53454733) begin errors = errors + 1; $display("ERROR magic"); end
        if (rf_thr != 1000) begin errors = errors + 1; $display("ERROR rf_thr not read (%0d)", rf_thr); end
        if (dut.thr != 2000) begin errors = errors + 1; $display("ERROR thr not read (%0d)", dut.thr); end
        if (dut.cfg != 32'h00000A31) begin errors = errors + 1; $display("ERROR cfg not read (%08x)", dut.cfg); end
        $display("cfg read back: %08x", dut.cfg);
        check(0, 1); check(1, 2); check(2, 3);
        $display("front end (edge1 - edge0, cycles): %0d (tile lead %0d)", fe0, RF_LEAD);
        if (fe0 != RF_LEAD - 3) begin errors = errors + 1; $display("ERROR front end %0d, expected %0d", fe0, RF_LEAD - 3); end
        if (errors == 0) $display("RESULT: PASS"); else $display("RESULT: FAIL (%0d errors)", errors);
        $finish;
    end
endmodule
