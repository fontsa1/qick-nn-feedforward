// tb_probe_gate: edge cases of dac_gate with the decision input driven cycle-exactly on aclk, BRAM model
// READ_LATENCY 1, generator pulse of PLEN words (first RAMP words at |x| = LOWAMP < NZ_THR, the rest at FULL).
// Cases: reset output, calibration (word 7), blank onset vs DELAY_LEAD, a sub-threshold leading edge, late and
// same-cycle decisions, decision arrival scanned around window open, two markers per shot, the next cls_trig
// inside a window, gate disabled mid-window.
`timescale 1ps / 1ps
module tb_probe_gate;
    localparam integer NS = 16, D_PS = 2034;
    localparam integer GEN_LAT = 40, PLEN = 40, MK_W = 120, NZ = 2000;
    reg aclk = 0, aresetn = 0;
    always #(D_PS/2) aclk = ~aclk;

    reg  [16*NS-1:0] s_tdata = 0;
    wire [16*NS-1:0] m_tdata; wire m_tvalid;
    reg  [63:0] dec_tdata = 0; reg dec_tvalid = 0;
    reg  marker = 0, cls = 0;
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

    integer cyc = 0, p_start = -100000, ramp = 0;
    reg marker_q = 0;
    always @(posedge aclk) cyc <= cyc + 1;
    always @(posedge aclk) begin
        marker_q <= marker;
        if (marker && !marker_q) p_start <= cyc + GEN_LAT;
    end
    wire in_pulse = (cyc >= p_start) && (cyc < p_start + PLEN);
    wire in_ramp  = in_pulse && (cyc < p_start + ramp);
    integer j;
    always @(posedge aclk)
        for (j = 0; j < NS; j = j + 1)
            s_tdata[16*j +: 16] <= in_ramp ? 16'd500 : in_pulse ? 16'd12000 : 16'd0;

    // which pulse words leave the gate non-zero: index of the word in the pulse
    // (m_tdata at cycle c carries s_tdata of cycle c-1, i.e. generator word of c-2)
    integer leak_first = -1, leak_count = 0, leak_amp_low = 0, leak_amp_full = 0;
    integer widx;
    always @(posedge aclk) begin
        widx = cyc - 2 - p_start;              // index of the word now on m_tdata
        if (aresetn && widx >= 0 && widx < PLEN && m_tdata != 0) begin
            if (leak_first < 0) leak_first = widx;
            leak_count = leak_count + 1;
            if (widx < ramp) leak_amp_low = leak_amp_low + 1; else leak_amp_full = leak_amp_full + 1;
        end
    end

    task cfg(input bit en, input bit p1, input bit p0, input bit ps, input [7:0] d);
        begin mem[0] = {16'(NZ), d, 4'b0, ps, p0, p1, en}; repeat (12) @(posedge aclk); end
    endtask
    task fire_cls;  begin @(posedge aclk); cls <= 1; repeat (10) @(posedge aclk); cls <= 0; repeat (5) @(posedge aclk); end endtask
    task decision(input bit b, input [28:0] shot);
        begin @(posedge aclk); dec_tdata <= {b, 2'b0, shot, 32'h0}; dec_tvalid <= 1; @(posedge aclk); dec_tvalid <= 0; end
    endtask
    task fire_marker; begin @(posedge aclk); marker <= 1; repeat (MK_W) @(posedge aclk); marker <= 0; repeat (300) @(posedge aclk); end endtask
    task clear_leak; begin leak_first = -1; leak_count = 0; leak_amp_low = 0; leak_amp_full = 0; end endtask
    task show(input [8*40-1:0] name);
        begin
            $display("%0s: flags passed=%0d stale=%0d bit=%0d | windows %0d passed %0d blanked %0d stale %0d dec %0d late %0d cls %0d | margin %0d word7 %0d | pulse words out non-zero: %0d (first idx %0d; low-amp %0d, full-amp %0d)",
                name, mem[10][0], mem[10][1], mem[10][2], mem[1], mem[2], mem[3], mem[4], mem[5], mem[12], mem[14],
                $signed(mem[6]), mem[7], leak_count, leak_first, leak_amp_low, leak_amp_full);
        end
    endtask

    integer mp, L;
    initial begin
        #(20000); aresetn = 1;
        // ---- X / reset check: output right after reset
        repeat (3) @(posedge aclk);
        if (^m_tdata === 1'bx) $display("RESET: m_tdata has X 3 cycles after reset"); else $display("RESET: m_tdata X-free 3 cycles after reset");
        cfg(0, 1, 0, 0, 0);
        // ---- calibration as gate_probe.py: gate disabled, word 7 = marker -> pulse
        ramp = 0; fire_cls; fire_marker; mp = mem[7]; $display("CAL (no ramp): word7 marker->pulse = %0d (GEN_LAT %0d)", mp, GEN_LAT);
        // ---- A: blank onset vs DELAY_LEAD, decision 0 (pulse must be fully blanked)
        for (L = 0; L <= 7; L = L + 1) begin
            cfg(1, 1, 0, 0, mp - L);
            ramp = 0; clear_leak; fire_cls; decision(0, 1); repeat (10) @(posedge aclk); fire_marker;
            $display("A  DELAY_LEAD %0d (DELAY %0d), rect pulse, decision 0: non-zero pulse words out %0d (first idx %0d)", L, mp - L, leak_count, leak_first);
        end
        // ---- A2: pulse with a 3-word sub-threshold leading edge; calibrate with it
        ramp = 3; cfg(0, 1, 0, 0, 0); fire_cls; fire_marker; mp = mem[7];
        $display("CAL (3-word ramp below NZ_THR): word7 = %0d", mp);
        for (L = 4; L <= 7; L = L + 1) begin
            cfg(1, 1, 0, 0, mp - L);
            clear_leak; fire_cls; decision(0, 1); repeat (10) @(posedge aclk); fire_marker;
            $display("A2 DELAY_LEAD %0d, ramped pulse, decision 0: non-zero pulse words out %0d (low-amp %0d, full-amp %0d)", L, leak_count, leak_amp_low, leak_amp_full);
        end
        ramp = 0; cfg(0, 1, 0, 0, 0); fire_cls; fire_marker; mp = mem[7];
        cfg(1, 1, 0, 0, mp - 4);
        // ---- B: decision of shot k arrives after shot k+1's classify trigger
        clear_leak; fire_cls; /* shot k: no marker yet, NN still busy */ fire_cls; /* shot k+1 */
        decision(1, 77); /* k's decision, late */ repeat (10) @(posedge aclk); fire_marker;
        show("B  shot-k decision after shot-k+1 cls");
        // ---- C: decision in the same cycle as the classify trigger rise (+2 sync)
        clear_leak;
        fork
            begin @(posedge aclk); cls <= 1; repeat (10) @(posedge aclk); cls <= 0; end
            begin repeat (3) @(posedge aclk); dec_tdata <= {1'b1, 63'h0}; dec_tvalid <= 1; @(posedge aclk); dec_tvalid <= 0; end
        join
        repeat (10) @(posedge aclk); fire_marker;
        show("C  decision coincident with cl_rise");
        // ---- D: decision arrival scanned around win_open
        for (L = -4; L <= 3; L = L + 1) begin
            clear_leak; fire_cls;
            fork
                begin @(posedge aclk); marker <= 1; repeat (MK_W) @(posedge aclk); marker <= 0; repeat (300) @(posedge aclk); end
                begin repeat (1 + 3 + (mp - 4) + 2 + L) @(posedge aclk); dec_tdata <= {1'b1, 63'h0}; dec_tvalid <= 1; @(posedge aclk); dec_tvalid <= 0; end
            join
            $display("D  offset %0d: passed=%0d stale=%0d margin %0d late_total %0d late_by %0d | pulse words out %0d", L, mem[10][0], mem[10][1], $signed(mem[6]), mem[12], mem[13], leak_count);
        end
        // ---- E: two markers in one shot (fresh decision 1): second window
        clear_leak; fire_cls; decision(1, 5); repeat (10) @(posedge aclk); fire_marker; show("E1 first window");
        clear_leak; fire_marker; show("E2 second window, no new cls/decision");
        // ---- F: classify trigger of the next shot in the middle of a passing window
        clear_leak; fire_cls; decision(1, 6); repeat (10) @(posedge aclk);
        fork fire_marker; begin repeat (mp + 10) @(posedge aclk); cls <= 1; repeat (10) @(posedge aclk); cls <= 0; end join
        show("F  cls mid-window (pulse should still pass)");
        // ---- G: gate disabled mid-window of a blanked pulse
        clear_leak; fire_cls; decision(0, 7); repeat (10) @(posedge aclk);
        fork fire_marker; begin repeat (mp + 10) @(posedge aclk); mem[0][0] = 1'b0; end join
        show("G  ENABLE cleared mid-window (decision 0)");
        $finish;
    end
endmodule
