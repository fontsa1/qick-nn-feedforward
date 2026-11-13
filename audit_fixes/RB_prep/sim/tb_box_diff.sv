// tb_box_diff: rtl/boxcar_tap.v (non-recursive moving sum) against the earlier recursive version
// (ref/boxcar_tap_ref.v), same stimulus, every output compared on every cycle. Stimulus: full-range, rail,
// alternating-rail and realistic inputs, random s_axis_tvalid gaps, random cfg (M 1..16 and the aliased
// encodings 17..32, shift 0..15, sel), 1-2-cycle cfg glitches, and resets.
`timescale 1ps / 1ps
module tb_box_diff;
    localparam integer CLK_PS = 1808;
    parameter integer NCYC = 400000;
    reg aclk = 0, aresetn = 0;
    always #(CLK_PS/2) aclk = ~aclk;
    reg [255:0] s_tdata = 0; reg s_tvalid = 1; reg [31:0] cfg = 0; reg [31:0] fir = 0; reg fir_v = 1;
    wire [31:0] box_r, box_n, nn_r, nn_n; wire bv_r, bv_n, nnv_r, nnv_n, rdy_r, rdy_n, mrv_r, mrv_n;
    wire [255:0] mr_r, mr_n;
    reg mr_rdy = 1;
    boxcar_tap_ref R (.aclk(aclk), .aresetn(aresetn), .s_axis_tdata(s_tdata), .s_axis_tvalid(s_tvalid), .s_axis_tready(rdy_r),
        .m_mr_axis_tdata(mr_r), .m_mr_axis_tvalid(mrv_r), .m_mr_axis_tready(mr_rdy), .s_fir_tdata(fir), .s_fir_tvalid(fir_v),
        .m_nn_tdata(nn_r), .m_nn_tvalid(nnv_r), .m_box_axis_tdata(box_r), .m_box_axis_tvalid(bv_r), .m_box_axis_tready(1'b1), .cfg(cfg));
    boxcar_tap N (.aclk(aclk), .aresetn(aresetn), .s_axis_tdata(s_tdata), .s_axis_tvalid(s_tvalid), .s_axis_tready(rdy_n),
        .m_mr_axis_tdata(mr_n), .m_mr_axis_tvalid(mrv_n), .m_mr_axis_tready(mr_rdy), .s_fir_tdata(fir), .s_fir_tvalid(fir_v),
        .m_nn_tdata(nn_n), .m_nn_tvalid(nnv_n), .m_box_axis_tdata(box_n), .m_box_axis_tvalid(bv_n), .m_box_axis_tready(1'b1), .cfg(cfg));

    integer mode = 4, gaps = 0, cyc = 0, mism = 0, mism_box = 0, nchk = 0, nvalid = 0, k;
    reg [15:0] v;
    always @(posedge aclk) begin
        for (k = 0; k < 16; k = k + 1) begin
            case (mode)
              0: v = $urandom;
              1: v = 16'h8000;
              2: v = 16'h7FFF;
              3: v = ($urandom % 2) ? 16'h8000 : 16'h7FFF;
              default: v = $signed($urandom % 801) - 400;
            endcase
            s_tdata[16*k +: 16] <= v;
        end
        s_tvalid <= gaps ? (($urandom % 8) != 0) : 1'b1;
        fir <= $urandom; fir_v <= ($urandom % 4) != 0; mr_rdy <= $urandom;
        cyc <= cyc + 1;
    end
    // compare after the edge settles
    always @(negedge aclk) begin
        nchk = nchk + 1;
        if (bv_r) nvalid = nvalid + 1;
        if (box_r !== box_n || bv_r !== bv_n) begin
            mism_box = mism_box + 1;
            if (mism_box < 6) $display("MISMATCH box at cyc %0d: ref %08x/%0d new %08x/%0d cfg %08x", cyc, box_r, bv_r, box_n, bv_n, cfg);
        end
        if (nn_r !== nn_n || nnv_r !== nnv_n || mr_r !== mr_n || mrv_r !== mrv_n || rdy_r !== rdy_n) mism = mism + 1;
    end
    integer seg = 0, M, sh, len;
    initial begin
        #(CLK_PS*10) aresetn = 1;
        while (cyc < NCYC) begin
            @(negedge aclk);
            seg = seg + 1;
            case ($urandom % 10)
              0: begin aresetn = 0; repeat (1 + $urandom % 5) @(negedge aclk); aresetn = 1; end   // reset
              1: begin cfg = $urandom; end                                                     // any value, incl. M field 17..32
              2: begin cfg = cfg ^ (1 << ($urandom % 13)); @(negedge aclk); cfg = cfg ^ (1 << ($urandom % 13)); end  // 1-cycle glitch
              default: begin
                  M = 1 + $urandom % 16; sh = $urandom % 16;
                  cfg = {19'd0, 5'(M - 1), 4'(sh), 3'd0, 1'($urandom)};
              end
            endcase
            mode = $urandom % 5; gaps = ($urandom % 3) == 0;
            len = ($urandom % 4 == 0) ? (1 + $urandom % 20) : (20 + $urandom % 2000);
            repeat (len) @(negedge aclk);
        end
        $display("tb_box_diff: %0d cycles, %0d segments, %0d valid box outputs; box mismatches %0d; other-port mismatches %0d",
                 nchk, seg, nvalid, mism_box, mism);
        if (mism_box == 0 && mism == 0) $display("RESULT: PASS"); else $display("RESULT: FAIL");
        $finish;
    end
endmodule
