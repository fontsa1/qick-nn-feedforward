// tb_gate_fix: four gates on the same stimulus, decision input driven cycle-exactly on aclk, BRAM model
// READ_LATENCY 1, generator pulse of PLEN words GEN_LAT cycles after the marker (first RAMP words at |x| = 500
// < NZ_THR, the rest at 12000):
//   0 REF = earlier version (ref/dac_gate_ref.v)    1 NEW = rtl/dac_gate.v defaults (EARLY_BLANK, SHOT_BIND)
//   2 FS  = NEW + FAIL_SAFE = 1                      3 CS  = NEW + CFG_SAFE = 1
// Shot fields follow NN_live (one per result, consecutive). Cases B1-B5 (shot binding), C3 (ENABLE cleared
// mid-window), D1-D3 (corrupted or mid-window writes of word 0).
`timescale 1ps / 1ps
module tb_gate_fix;
    localparam integer NS = 16, D_PS = 2034;
    localparam integer GEN_LAT = 40, PLEN = 40, MK_W = 120, NZ = 2000;
    reg aclk = 0, aresetn = 0;
    always #(D_PS/2) aclk = ~aclk;

    reg  [16*NS-1:0] s_tdata = 0;
    reg  [63:0] dec_tdata = 0; reg dec_tvalid = 0;
    reg  marker = 0, cls = 0;
    wire [16*NS-1:0] o [0:3];
    wire ben [0:3]; wire [3:0] bwe [0:3]; wire [31:0] ba [0:3], bd [0:3]; reg [31:0] bq [0:3];
    reg  [31:0] mem [0:3][0:63];
    reg  glitch = 0;                                  // one corrupted read of word 0
    `define GATE(K) (.aclk(aclk), .aresetn(aresetn), .s_axis_tdata(s_tdata), .s_axis_tvalid(1'b1), .s_axis_tready(), \
        .m_axis_tdata(o[K]), .m_axis_tvalid(), .m_axis_tready(1'b1), .dec_axis_tdata(dec_tdata), .dec_axis_tvalid(dec_tvalid), \
        .dec_axis_tready(), .marker(marker), .cls_trig(cls), .bram_clk(), .bram_rst(), .bram_en(ben[K]), .bram_we(bwe[K]), \
        .bram_addr(ba[K]), .bram_din(bd[K]), .bram_dout(bq[K]))
    dac_gate_ref #(.NS(NS)) G0 `GATE(0);
    dac_gate     #(.NS(NS)) G1 `GATE(1);
    dac_gate     #(.NS(NS), .FAIL_SAFE(1)) G2 `GATE(2);
    dac_gate     #(.NS(NS), .CFG_SAFE(1))  G3 `GATE(3);
    genvar g;
    generate for (g = 0; g < 4; g = g + 1) begin : bram
        always @(posedge aclk) if (ben[g]) begin
            if (bwe[g] == 4'hF) begin mem[g][ba[g][7:2]] <= bd[g]; bq[g] <= bd[g]; end
            else bq[g] <= (glitch && ba[g][7:2] == 0) ? (mem[g][0] ^ 32'h0000_8000) : mem[g][ba[g][7:2]];
        end
    end endgenerate
    integer i, k;
    initial begin for (k = 0; k < 4; k = k + 1) begin for (i = 0; i < 64; i = i + 1) mem[k][i] = 0; bq[k] = 0; end end
    task wcfg(input [31:0] w); begin for (k = 0; k < 4; k = k + 1) mem[k][0] = w; end endtask

    integer cyc = 0, p_start = -100000, ramp = 0;
    reg marker_q = 0;
    always @(posedge aclk) cyc <= cyc + 1;
    always @(posedge aclk) begin marker_q <= marker; if (marker && !marker_q) p_start <= cyc + GEN_LAT; end
    wire in_pulse = (cyc >= p_start) && (cyc < p_start + PLEN);
    wire in_ramp  = in_pulse && (cyc < p_start + ramp);
    integer j;
    always @(posedge aclk) for (j = 0; j < NS; j = j + 1) s_tdata[16*j +: 16] <= in_ramp ? 16'd500 : in_pulse ? 16'd12000 : 16'd0;

    // non-zero pulse words at each gate output; m_tdata at cycle c carries generator word c-2
    integer nz [0:3], nz_low [0:3], nz_any [0:3];
    integer widx;
    always @(negedge aclk) begin
        widx = cyc - 2 - p_start;
        for (k = 0; k < 4; k = k + 1) begin
            if (aresetn && widx >= 0 && widx < PLEN && o[k] != 0) begin
                nz[k] = nz[k] + 1; if (widx < ramp) nz_low[k] = nz_low[k] + 1;
            end
            if (aresetn && o[k] != 0) nz_any[k] = nz_any[k] + 1;
        end
    end
    task clr; begin for (k = 0; k < 4; k = k + 1) begin nz[k] = 0; nz_low[k] = 0; nz_any[k] = 0; end end endtask

    integer nn = 0;                                   // NN_live's result counter
    integer errors = 0;
    function [31:0] cfgw(input bit en, input bit p1, input bit p0, input bit ps, input integer d);
        cfgw = {16'(NZ), 8'(d), 4'b0, ps, p0, p1, en};
    endfunction
    task cfg(input [31:0] w); begin wcfg(w); repeat (14) @(posedge aclk); end endtask
    task fire_cls;  begin @(posedge aclk); cls <= 1; repeat (10) @(posedge aclk); cls <= 0; repeat (5) @(posedge aclk); end endtask
    task decision_f(input bit b, input integer f);
        begin @(posedge aclk); dec_tdata <= {b, 2'b0, 29'(f), 32'h0}; dec_tvalid <= 1; @(posedge aclk); dec_tvalid <= 0; end
    endtask
    task decision(input bit b); begin decision_f(b, nn); nn = nn + 1; end endtask
    task fire_marker; begin @(posedge aclk); marker <= 1; repeat (MK_W) @(posedge aclk); marker <= 0; repeat (300) @(posedge aclk); end endtask
    task show(input string name, input integer exp_ref, input integer exp_new);
        begin
            $display("%0s: pulse words out REF %0d | NEW %0d | FS %0d | CS %0d   (NEW flags %03x, rejects %0d, expected field %0d)",
                     name, nz[0], nz[1], nz[2], nz[3], mem[1][10][11:0], mem[1][19], mem[1][20]);
            if (exp_ref >= 0 && nz[0] != exp_ref) begin errors = errors + 1; $display("  ERROR REF expected %0d", exp_ref); end
            if (exp_new >= 0 && nz[1] != exp_new) begin errors = errors + 1; $display("  ERROR NEW expected %0d", exp_new); end
        end
    endtask

    integer mp, mpr, L, off, rej0, rej3;
    initial begin
        #(20000); aresetn = 1;
        repeat (3) @(posedge aclk);
        cfg(cfgw(0, 1, 0, 0, 0));
        // calibration as gate_probe.py (gate disabled): word 7 = marker -> first |x| > NZ_THR
        ramp = 0; fire_cls; decision(1); fire_marker; mp = mem[0][7];
        $display("CAL rect pulse: word7 = %0d (all gates agree: %0d)", mp, mem[0][7] == mem[1][7] && mem[1][7] == mem[3][7]);

        $display("\n== A: blank onset vs DELAY_LEAD (DELAY = word7 - DELAY_LEAD), fresh decision 0, rect pulse of %0d words", PLEN);
        for (L = 0; L <= 7; L = L + 1) begin
            cfg(cfgw(1, 1, 0, 0, mp - L)); clr;
            fire_cls; decision(0); repeat (10) @(posedge aclk); fire_marker;
            show($sformatf("A  DELAY_LEAD %0d", L), (L < 4) ? 4 - L : 0, (L < 2) ? 2 - L : 0);
        end
        $display("\n== A1: decision 1 (pass), DELAY_LEAD 2..7: the whole pulse must still pass");
        for (L = 2; L <= 7; L = L + 1) begin
            cfg(cfgw(1, 1, 0, 0, mp - L)); clr;
            fire_cls; decision(1); repeat (10) @(posedge aclk); fire_marker;
            show($sformatf("A1 DELAY_LEAD %0d, decision 1", L), PLEN, PLEN);
        end

        $display("\n== A2: pulse with a 3-word leading edge below NZ_THR, calibrated on it (as on hardware)");
        ramp = 3; cfg(cfgw(0, 1, 0, 0, 0)); fire_cls; decision(1); fire_marker; mpr = mem[0][7];
        $display("CAL ramped pulse: word7 = %0d", mpr);
        for (L = 2; L <= 8; L = L + 1) begin
            cfg(cfgw(1, 1, 0, 0, mpr - L)); clr;
            fire_cls; decision(0); repeat (10) @(posedge aclk); fire_marker;
            $display("A2 DELAY_LEAD %0d, ramped pulse, decision 0: pulse words out REF %0d (low-amp %0d) | NEW %0d (low-amp %0d)",
                     L, nz[0], nz_low[0], nz[1], nz_low[1]);
        end
        ramp = 0;

        $display("\n== A3: decision arriving at every offset around window open (DELAY_LEAD 4), bit 0 and bit 1");
        cfg(cfgw(1, 1, 0, 0, mp - 4));
        for (off = -4; off <= 2; off = off + 1) begin
            clr; fire_cls;
            fork
                begin @(posedge aclk); marker <= 1; repeat (MK_W) @(posedge aclk); marker <= 0; repeat (300) @(posedge aclk); end
                begin repeat (1 + 3 + (mp - 4) + 2 + off) @(posedge aclk); dec_tdata <= {1'b0, 2'b0, 29'(nn), 32'h0}; dec_tvalid <= 1; @(posedge aclk); dec_tvalid <= 0; end
            join
            nn = nn + 1;
            $display("A3 bit 0, arrival offset %0d: REF margin %0d stale %0d -> pulse words out %0d | NEW margin %0d stale %0d -> %0d",
                     off, $signed(mem[0][6]), mem[0][10][1], nz[0], $signed(mem[1][6]), mem[1][10][1], nz[1]);
            if (nz[1] != 0) begin errors = errors + 1; $display("  ERROR NEW leaked"); end
            clr; fire_cls;
            fork
                begin @(posedge aclk); marker <= 1; repeat (MK_W) @(posedge aclk); marker <= 0; repeat (300) @(posedge aclk); end
                begin repeat (1 + 3 + (mp - 4) + 2 + off) @(posedge aclk); dec_tdata <= {1'b1, 2'b0, 29'(nn), 32'h0}; dec_tvalid <= 1; @(posedge aclk); dec_tvalid <= 0; end
            join
            nn = nn + 1;
            $display("A3 bit 1, arrival offset %0d: REF fresh %0d -> pulse words out %0d | NEW fresh %0d -> %0d",
                     off, !mem[0][10][1], nz[0], !mem[1][10][1], nz[1]);
            if (nz[1] != nz[0]) begin errors = errors + 1; $display("  ERROR NEW pass differs from REF"); end
        end

        $display("\n== B: shot binding (DELAY_LEAD 4, PASS1 only)");
        cfg(cfgw(1, 1, 0, 0, mp - 4));
        clr; fire_cls; decision(1); repeat (10) @(posedge aclk); fire_marker;
        show("B0 normal shot, decision 1", PLEN, PLEN);
        // B1: shot k's decision arrives after shot k+1's classify trigger
        rej0 = mem[1][19];
        clr; fire_cls; /* shot k, its NN result still pending */ fire_cls; /* shot k+1 */
        decision(1); /* = shot k's result */ repeat (10) @(posedge aclk); fire_marker;
        show("B1 shot-k decision (bit 1) after shot k+1's trigger", PLEN, 0);
        if (mem[1][19] != rej0 + 1 || !mem[1][10][4]) begin errors = errors + 1; $display("  ERROR NEW did not count the rejection"); end
        // B2: same, and shot k+1's own decision arrives before its window -> used
        clr; fire_cls; fire_cls; decision(0); /* shot k's (bit 0) */ repeat (20) @(posedge aclk); decision(1); /* k+1's */
        repeat (10) @(posedge aclk); fire_marker;
        show("B2 shot k's result (bit 0) then shot k+1's own (bit 1)", PLEN, PLEN);
        // B2b: reversed bits: shot k's result says pass, shot k+1's own says blank
        clr; fire_cls; fire_cls; decision(1); repeat (20) @(posedge aclk); decision(0);
        repeat (10) @(posedge aclk); fire_marker;
        show("B2b shot k's result (bit 1) then shot k+1's own (bit 0)", 0, 0);
        // B3: a shot whose result never comes; the next shot is rejected once, then in step
        clr; fire_cls; repeat (50) @(posedge aclk); fire_marker;
        show("B3a no result (stale)", 0, 0);
        clr; fire_cls; decision(1); repeat (10) @(posedge aclk); fire_marker;
        show("B3b next shot, decision 1 (cannot be told from B3a's late one)", PLEN, 0);
        clr; fire_cls; decision(1); repeat (10) @(posedge aclk); fire_marker;
        show("B3c shot after (binding back in step)", PLEN, PLEN);
        // B4: a decision with a wrong field (e.g. stale readout) arriving alone
        clr; fire_cls; decision_f(1, nn + 5); repeat (10) @(posedge aclk); fire_marker;
        show("B4 decision with field +5", PLEN, 0);
        nn = nn + 0;                                  // (B4's result was not an NN result)
        clr; fire_cls; decision(1); repeat (10) @(posedge aclk); fire_marker;
        show("B4b next shot", PLEN, -1);
        $display("   (B4b: after a foreign field the binding expects field+1 of it; see REPORT)");
        // B5: marker with no classify trigger, after a shot whose window never came (program contract)
        clr; fire_cls; decision(1); repeat (300) @(posedge aclk); /* no marker for this shot */
        fire_marker; show("B5 marker without its own classify trigger (contract: not covered)", PLEN, PLEN);

        $display("\n== C: gate disabled (fail-open vs FAIL_SAFE)");
        cfg(cfgw(0, 1, 0, 0, mp - 4)); clr; fire_cls; decision(0); repeat (10) @(posedge aclk); fire_marker;
        $display("C1 ENABLE = 0, decision 0: pulse words out REF %0d NEW %0d FS %0d CS %0d", nz[0], nz[1], nz[2], nz[3]);
        if (nz[2] != 0 || nz[1] != PLEN) begin errors = errors + 1; $display("  ERROR"); end
        cfg(cfgw(0, 1, 0, 0, mp - 4)); clr; fire_cls; decision(1); repeat (10) @(posedge aclk); fire_marker;
        $display("C2 ENABLE = 0, decision 1: pulse words out REF %0d NEW %0d FS %0d CS %0d", nz[0], nz[1], nz[2], nz[3]);
        // C3: ENABLE cleared in the middle of a blanked window
        cfg(cfgw(1, 1, 0, 0, mp - 4)); clr; fire_cls; decision(0); repeat (10) @(posedge aclk);
        fork fire_marker; begin repeat (mp + 10) @(posedge aclk); wcfg(cfgw(0, 1, 0, 0, mp - 4)); end join
        $display("C3 ENABLE cleared mid-window, decision 0: pulse words out REF %0d NEW %0d FS %0d CS %0d", nz[0], nz[1], nz[2], nz[3]);
        if (nz[2] != 0 || nz[3] != 0) begin errors = errors + 1; $display("  ERROR FS/CS leaked"); end

        $display("\n== D: configuration filtering (CFG_SAFE)");
        // D1: one corrupted read of word 0 (DELAY bits flipped) while the gate sits idle
        cfg(cfgw(1, 1, 0, 0, mp - 4)); clr;
        @(posedge aclk) glitch <= 1; @(posedge aclk) glitch <= 0; repeat (20) @(posedge aclk);
        $display("D1 glitched read of word 0, no shot: non-zero output words REF %0d NEW %0d CS %0d; DELAY now REF %0d CS %0d",
                 nz_any[0], nz_any[1], nz_any[3], G0.delay_r, G3.delay_r);
        // D2: a glitched read while a marker is in the delay line
        clr; fire_cls; decision(0); repeat (10) @(posedge aclk);
        rej0 = mem[0][1]; rej3 = mem[3][1];
        fork fire_marker; begin repeat (mp + 10) @(posedge aclk); glitch <= 1; @(posedge aclk) glitch <= 0; end join
        $display("D2 glitched read (DELAY +128 for one read) mid-window, decision 0: pulse words out REF %0d NEW %0d CS %0d; windows this shot REF %0d CS %0d",
                 nz[0], nz[1], nz[3], mem[0][1] - rej0, mem[3][1] - rej3);
        if (nz[3] != 0 || mem[3][1] - rej3 != 1) begin errors = errors + 1; $display("  ERROR CS affected by the glitch"); end
        if (nz[3] != 0) begin errors = errors + 1; $display("  ERROR CS leaked"); end
        // D3: the host rewrites DELAY (+128) in the middle of a blanked window
        cfg(cfgw(1, 1, 0, 0, mp - 4)); clr; fire_cls; decision(0); repeat (10) @(posedge aclk); rej0 = mem[0][1]; rej3 = mem[3][1];
        fork fire_marker; begin repeat (mp + 10) @(posedge aclk); wcfg(cfgw(1, 1, 0, 0, mp - 4 + 128)); end join
        $display("D3 host rewrites DELAY mid-window, decision 0: pulse words out REF %0d NEW %0d CS %0d; windows this shot REF %0d CS %0d",
                 nz[0], nz[1], nz[3], mem[0][1] - rej0, mem[3][1] - rej3);
        if (nz[3] != 0) begin errors = errors + 1; $display("  ERROR CS leaked"); end
        cfg(cfgw(1, 1, 0, 0, mp - 4)); repeat (400) @(posedge aclk);

        $display("\n== E: gate reset while NN_live keeps counting (first shot rejected, then in step)");
        aresetn = 0; repeat (5) @(posedge aclk); aresetn = 1; repeat (5) @(posedge aclk);
        cfg(cfgw(1, 1, 0, 0, mp - 4));
        clr; fire_cls; decision(1); repeat (10) @(posedge aclk); fire_marker; show("E1 first shot after gate reset", PLEN, 0);
        clr; fire_cls; decision(1); repeat (10) @(posedge aclk); fire_marker; show("E2 second shot", PLEN, PLEN);

        $display("\nmagic REF %08x NEW %08x; param flags [11:8] NEW %h FS %h CS %h", mem[0][11], mem[1][11], mem[1][10][11:8], mem[2][10][11:8], mem[3][10][11:8]);
        if (errors == 0) $display("RESULT: PASS (all expectations met)"); else $display("RESULT: FAIL (%0d)", errors);
        $finish;
    end
endmodule
