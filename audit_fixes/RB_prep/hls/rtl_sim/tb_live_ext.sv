// tb_live_ext: the HLS-generated NN_live on extreme inputs (ext_windows.hex from make_ext_vectors.py),
// streamed back to back (40 idle cycles between windows, TLAST on word 576). res_o must equal the exact model:
// logit = saturated value (wrapped value with +define+EXPECT_WRAP), decision = logit >= 1/64, shot field =
// window index. Latency: TLAST -> res_o_ap_vld.
`timescale 1ps / 1ps
module tb_live_ext;
    localparam integer N = 576, CLK_PS = 1808, MAXW = 256;
    reg aclk = 0, aresetn = 0;
    always #(CLK_PS/2) aclk = ~aclk;
    reg [31:0] tdata = 0; reg tvalid = 0, tlast = 0; wire tready;
    wire [63:0] res_o; wire res_vld;
    wire [31:0] b_addr, b_din; wire b_en, b_clk, b_rst; wire [3:0] b_we; reg [31:0] b_dout = 0;
    NN_live dut (.ap_clk(aclk), .ap_rst_n(aresetn),
        .in_r_TDATA(tdata), .in_r_TVALID(tvalid), .in_r_TREADY(tready), .in_r_TKEEP(4'hF), .in_r_TSTRB(4'hF), .in_r_TLAST(tlast),
        .res_o(res_o), .res_o_ap_vld(res_vld),
        .out_r_Addr_A(b_addr), .out_r_EN_A(b_en), .out_r_WEN_A(b_we), .out_r_Din_A(b_din), .out_r_Dout_A(b_dout), .out_r_Clk_A(b_clk), .out_r_Rst_A(b_rst));
    reg [31:0] mem [0:8191];
    always @(posedge aclk) if (b_en) begin if (b_we != 0) mem[b_addr[14:2]] <= b_din; b_dout <= mem[b_addr[14:2]]; end

    reg [31:0] win [0:MAXW*N-1];
    reg [15:0] e_sat [0:MAXW-1], e_wrap [0:MAXW-1]; reg e_dec [0:MAXW-1];
    integer nw = 0, fd, r, a, b, c;
    integer n_res = 0, bad_logit = 0, bad_dec = 0, bad_shot = 0, lat, lat_min = 1 << 30, lat_max = 0, t_last = 0, cyc = 0, offered_low = 0;
    reg [15:0] exp_l;
    always @(posedge aclk) cyc <= cyc + 1;
    always @(posedge aclk) if (tvalid && tlast && tready) t_last <= cyc;
    always @(posedge aclk) if (tvalid && !tready) offered_low = offered_low + 1;
    always @(posedge aclk) if (res_vld) begin
`ifdef EXPECT_WRAP
        exp_l = e_wrap[n_res];
`else
        exp_l = e_sat[n_res];
`endif
        if (res_o[15:0] !== exp_l) begin bad_logit = bad_logit + 1; if (bad_logit < 6) $display("logit window %0d: got %h exp %h", n_res, res_o[15:0], exp_l); end
        if (res_o[63] !== e_dec[n_res]) bad_dec = bad_dec + 1;
        if (res_o[44:16] !== n_res) bad_shot = bad_shot + 1;
        lat = cyc - t_last; if (lat < lat_min) lat_min = lat; if (lat > lat_max) lat_max = lat;
        n_res = n_res + 1;
    end
    integer k, j;
    initial begin
        $readmemh("ext_windows.hex", win);
        fd = $fopen("ext_expected.hex", "r");
        while (!$feof(fd)) begin r = $fscanf(fd, "%h %h %h\n", a, b, c); if (r == 3) begin e_sat[nw] = a; e_wrap[nw] = b; e_dec[nw] = c; nw = nw + 1; end end
        $fclose(fd);
        for (k = 0; k < 8192; k = k + 1) mem[k] = 0;
        repeat (10) @(posedge aclk); aresetn <= 1; repeat (10) @(posedge aclk);
        for (k = 0; k < nw; k = k + 1) begin
            for (j = 0; j < N; j = j + 1) begin
                tdata <= win[k * N + j]; tvalid <= 1; tlast <= (j == N - 1);
                @(posedge aclk);
            end
            tvalid <= 0; tlast <= 0;
            repeat (40) @(posedge aclk);
        end
        repeat (100) @(posedge aclk);
`ifdef EXPECT_WRAP
        $display("tb_live_ext (expect WRAP, NN_live 1.4): windows %0d results %0d | logit != wrap model %0d | decision != exact sign %0d | shot field errors %0d | words offered with TREADY low %0d | latency %0d..%0d cycles",
                 nw, n_res, bad_logit, bad_dec, bad_shot, offered_low, lat_min, lat_max);
`else
        $display("tb_live_ext (expect SAT): windows %0d results %0d | logit != sat model %0d | decision != exact sign %0d | shot field errors %0d | words offered with TREADY low %0d | latency %0d..%0d cycles",
                 nw, n_res, bad_logit, bad_dec, bad_shot, offered_low, lat_min, lat_max);
`endif
        $finish;
    end
endmodule
