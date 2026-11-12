// RTL sim of boxcar_tap (v1.40L7, task 4.1h).
// Random 8-lane {Q,I} input every cycle; a reference model computes
//   box8[t] = sum of the 8 lanes, msum[t] = sum_{j<M} box8[t-j],
//   out[t]  = sat16(floor((msum[t] + 2^(shift-1)) / 2^shift))   (shift 0: msum)
// and the test checks the DUT output equals ref[t - LAT] for a fixed LAT
// (found on the first config, then required for all), over several (M, shift)
// configs including a saturating segment. Also: m_nn follows cfg[0] (FIR vs
// boxcar) combinationally, and m_mr_axis is s_axis unchanged.

`timescale 1ps / 1ps

module tb_boxcar;
    localparam integer CLK_PS = 1808;
    localparam integer T = 30000;

    reg aclk = 0, aresetn = 0;
    always #(CLK_PS / 2) aclk = ~aclk;

    reg  [255:0] s_tdata = 0;
    reg  [31:0]  fir = 0;
    reg  [31:0]  cfg = 0;
    wire [255:0] mr_tdata; wire mr_tvalid, s_tready;
    wire [31:0]  nn_tdata, box_tdata; wire nn_tvalid, box_tvalid;

    boxcar_tap dut (
        .aclk(aclk), .aresetn(aresetn),
        .s_axis_tdata(s_tdata), .s_axis_tvalid(1'b1), .s_axis_tready(s_tready),
        .m_mr_axis_tdata(mr_tdata), .m_mr_axis_tvalid(mr_tvalid), .m_mr_axis_tready(1'b1),
        .s_fir_tdata(fir), .s_fir_tvalid(1'b1),
        .m_nn_tdata(nn_tdata), .m_nn_tvalid(nn_tvalid),
        .m_box_axis_tdata(box_tdata), .m_box_axis_tvalid(box_tvalid), .m_box_axis_tready(1'b1),
        .cfg(cfg));

    // input history
    integer b8i [0:T-1], b8q [0:T-1];
    integer cfg_at [0:T-1];
    reg [31:0] outv [0:T-1];
    reg        outok [0:T-1];
    integer cyc = 0;
    integer big = 0;                    // 1: saturating stimulus
    integer errors = 0;

    function automatic integer s16(input integer x); s16 = (x > 32767) ? 32767 : (x < -32768) ? -32768 : x; endfunction

    integer k, si, sq, li, lq;
    always @(posedge aclk) begin
        // drive the next input word
        si = 0; sq = 0;
        for (k = 0; k < 8; k = k + 1) begin
            if (big) begin li = 30000 + ($urandom % 2768); lq = -30000 - ($urandom % 2768); end
            else begin li = $signed($urandom % 4096) - 2048; lq = $signed($urandom % 4096) - 2048; end
            s_tdata[32*k +: 16]      <= li[15:0];
            s_tdata[32*k + 16 +: 16] <= lq[15:0];
            si = si + li; sq = sq + lq;
        end
        fir <= $urandom;
        if (cyc < T) begin
            b8i[cyc] = si; b8q[cyc] = sq; cfg_at[cyc] = cfg;
            outv[cyc] = box_tdata; outok[cyc] = box_tvalid;
        end
        cyc <= cyc + 1;
        // combinational select check (sel is cfg registered 3 cycles: skip near changes)
        if (cyc > 10 && cfg_at[cyc-4] == cfg) begin
            if (nn_tdata !== (cfg[0] ? box_tdata : fir) || nn_tvalid !== (cfg[0] ? box_tvalid : 1'b1)) begin
                errors = errors + 1; if (errors < 10) $display("ERROR m_nn select at %0d", cyc);
            end
        end
        if (mr_tdata !== s_tdata || mr_tvalid !== 1'b1 || s_tready !== 1'b1) begin
            errors = errors + 1; if (errors < 10) $display("ERROR MR pass-through at %0d", cyc);
        end
    end

    function automatic [31:0] refout(input integer t, input integer c);
        integer M, sh, j, ai, aq, ri, rq;
        begin
            M = c[12:8] + 1; sh = c[7:4]; ai = 0; aq = 0;
            for (j = 0; j < M; j = j + 1) begin ai = ai + b8i[t - j]; aq = aq + b8q[t - j]; end
            if (sh == 0) begin ri = ai; rq = aq; end
            else begin ri = (ai + (1 <<< (sh - 1))) >>> sh; rq = (aq + (1 <<< (sh - 1))) >>> sh; end
            ri = s16(ri); rq = s16(rq);
            refout = {rq[15:0], ri[15:0]};
        end
    endfunction

    // segments: [start, end) with one cfg; check outputs from start + 60 on
    integer seg_s [0:7], seg_e [0:7], seg_c [0:7], seg_big [0:7];
    integer LAT = -1, n_checked = 0, s, t, L, good;
    initial begin
        seg_c[0] = 32'h0000_0A31; seg_big[0] = 0;   // M = 11, shift 3, sel boxcar
        seg_c[1] = 32'h0000_0000; seg_big[1] = 0;   // M = 1, shift 0, FIR selected
        seg_c[2] = 32'h0000_0F61; seg_big[2] = 0;   // M = 16, shift 6
        seg_c[3] = 32'h0000_0451; seg_big[3] = 0;   // M = 5, shift 5
        seg_c[4] = 32'h0000_0AF1; seg_big[4] = 0;   // M = 11, shift 15
        seg_c[5] = 32'h0000_0F01; seg_big[5] = 1;   // M = 16, shift 0: saturates high I / low Q
        seg_c[6] = 32'h0000_0731; seg_big[6] = 1;   // M = 8, shift 3, still saturating
        seg_c[7] = 32'h0000_0A61; seg_big[7] = 0;   // M = 11, shift 6
        #(CLK_PS * 10) aresetn = 1;
        for (s = 0; s < 8; s = s + 1) begin
            @(negedge aclk);
            cfg = seg_c[s]; big = seg_big[s];
            seg_s[s] = cyc;
            repeat (3000) @(posedge aclk);
            seg_e[s] = cyc;
        end
        repeat (40) @(posedge aclk);
        // latency from segment 0
        for (L = 4; L < 30 && LAT < 0; L = L + 1) begin
            good = 1;
            for (t = seg_s[0] + 100; t < seg_e[0]; t = t + 1)
                if (outv[t] !== refout(t - L, seg_c[0])) good = 0;
            if (good) LAT = L;
        end
        $display("latency m0 word -> m_box_axis: %0d cycles", LAT);
        if (LAT < 0) begin errors = errors + 1; $display("ERROR no latency matches segment 0"); end
        else for (s = 0; s < 8; s = s + 1) begin
            good = 0;
            for (t = seg_s[s] + 60; t < seg_e[s]; t = t + 1) begin
                n_checked = n_checked + 1;
                if (outv[t] !== refout(t - LAT, seg_c[s]) || outok[t] !== 1'b1) begin
                    good = good + 1;
                    if (good < 4) $display("ERROR seg %0d t %0d got %08x exp %08x", s, t, outv[t], refout(t - LAT, seg_c[s]));
                end
            end
            $display("segment %0d cfg %08x: %0d mismatches", s, seg_c[s], good);
            errors = errors + good;
        end
        $display("checked %0d outputs", n_checked);
        if (errors == 0) $display("RESULT: PASS"); else $display("RESULT: FAIL (%0d errors)", errors);
        $finish;
    end
endmodule
