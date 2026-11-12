// R.16 fix test: one corrupted capture (bit 12 flipped for one cycle) in a moving-sum
// register. Old (recursive, boxcar_tap_ref) vs new (non-recursive, boxcar_tap), each
// compared with a fault-free twin of itself. Reports, per injection, how many outputs
// differ and the last differing output (cycles after the injection), and whether the
// output is still wrong 2,000 cycles later.
`timescale 1ps / 1ps
module tb_box_fault;
    localparam integer CLK_PS = 1808;
    reg aclk = 0, aresetn = 0;
    always #(CLK_PS/2) aclk = ~aclk;
    reg [255:0] s_tdata = 0; reg [31:0] cfg = 32'h0000_0A71;   // M = 11, shift 7, boxcar to NN (deployed)
    wire [31:0] oc, of, nc, nf;
    `define BOX(MOD, INST, OUT) MOD INST (.aclk(aclk), .aresetn(aresetn), .s_axis_tdata(s_tdata), .s_axis_tvalid(1'b1), .s_axis_tready(), \
        .m_mr_axis_tdata(), .m_mr_axis_tvalid(), .m_mr_axis_tready(1'b1), .s_fir_tdata(32'h0), .s_fir_tvalid(1'b1), \
        .m_nn_tdata(), .m_nn_tvalid(), .m_box_axis_tdata(OUT), .m_box_axis_tvalid(), .m_box_axis_tready(1'b1), .cfg(cfg));
    `BOX(boxcar_tap_ref, OC, oc)
    `BOX(boxcar_tap_ref, OF, of)
    `BOX(boxcar_tap,     NC, nc)
    `BOX(boxcar_tap,     NF, nf)

    integer k, cyc = 0;
    always @(posedge aclk) begin
        for (k = 0; k < 16; k = k + 1) s_tdata[16*k +: 16] <= $signed($urandom % 6001) - 3000;
        cyc <= cyc + 1;
    end
    reg [31:0] cfg_sel;
    integer t0, nd_o, nd_n, last_o, last_n, tail_o, tail_n;
    always @(negedge aclk) if (cyc > t0) begin
        if (oc !== of) begin nd_o = nd_o + 1; last_o = cyc - t0; if (cyc - t0 > 2000) tail_o = tail_o + 1; end
        if (nc !== nf) begin nd_n = nd_n + 1; last_n = cyc - t0; if (cyc - t0 > 2000) tail_n = tail_n + 1; end
    end
    task start; begin
        // re-align both faulted twins with their clean twins: a cfg change clears the sums
        // a cfg change clears all four moving sums in the same cycle, re-aligning the twins
        cfg = 32'h0000_0001; repeat (6) @(negedge aclk); cfg = cfg_sel; repeat (100) @(negedge aclk);
        t0 = cyc; nd_o = 0; nd_n = 0; last_o = -1; last_n = -1; tail_o = 0; tail_n = 0;
    end endtask
    task finish_rep(input [8*24-1:0] name, input integer is_old);
        begin
            repeat (2100) @(negedge aclk);
            if (is_old) $display("OLD %0s: outputs differing %0d, last at +%0d cycles, still wrong after +2000: %0d", name, nd_o, last_o, tail_o);
            else        $display("NEW %0s: outputs differing %0d, last at +%0d cycles, still wrong after +2000: %0d", name, nd_n, last_n, tail_n);
            t0 = 1 << 30;
        end
    endtask
    integer m;
    initial begin
        t0 = 1 << 30;
        #(CLK_PS*10) aresetn = 1;
        for (m = 0; m < 3; m = m + 1) begin
            cfg_sel = (m == 0) ? 32'h0000_0A71 : (m == 1) ? 32'h0000_0F01 : 32'h0000_0071;  // M = 11/s7, 16/s0, 1/s7
            $display("---- cfg %08x", cfg_sel);
            // old design registers
            start; OF.acc_i  = OF.acc_i  ^ 23'h1000; finish_rep("acc_i", 1);
            start; OF.dif_i  = OF.dif_i  ^ 20'h1000; finish_rep("dif_i", 1);
            start; OF.new_q  = OF.new_q  ^ 19'h1000; finish_rep("new_q", 1);
            start; OF.old_i  = OF.old_i  ^ 19'h1000; finish_rep("old_i", 1);
            start; OF.sri[4] = OF.sri[4] ^ 19'h1000; finish_rep("sri[4]", 1);
            start; OF.rd_i   = OF.rd_i   ^ 24'h1000; finish_rep("rd_i (output stage)", 1);
            // new design registers
            start; NF.acc_i  = NF.acc_i  ^ 23'h1000; finish_rep("acc_i", 0);
            start; NF.h1_i   = NF.h1_i   ^ 20'h1000; finish_rep("h1_i", 0);
            start; NF.h2_q   = NF.h2_q   ^ 20'h1000; finish_rep("h2_q", 0);
            start; NF.g_i    = NF.g_i    ^ 23'h1000; finish_rep("g_i", 0);
            start; NF.l3i[1] = NF.l3i[1] ^ 22'h1000; finish_rep("l3i[1]", 0);
            start; NF.l2q[0] = NF.l2q[0] ^ 21'h1000; finish_rep("l2q[0]", 0);
            start; NF.l1i[1] = NF.l1i[1] ^ 20'h1000; finish_rep("l1i[1]", 0);
            start; NF.l1i[5] = NF.l1i[5] ^ 20'h1000; finish_rep("l1i[5]", 0);
            start; NF.sri[0] = NF.sri[0] ^ 19'h1000; finish_rep("sri[0]", 0);
            start; NF.sri[12]= NF.sri[12]^ 19'h1000; finish_rep("sri[12]", 0);
            start; NF.mk     = NF.mk     ^ 16'h0400; finish_rep("mk (mask bit 10)", 0);
            start; NF.rd_i   = NF.rd_i   ^ 24'h1000; finish_rep("rd_i (output stage)", 0);
        end
        $finish;
    end
endmodule
