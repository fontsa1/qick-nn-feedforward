`timescale 1ns / 1ps
// classifier_unit_zl: one qubit's streaming classifier and feed-forward gate for the out-of-context scaling study: the
// deployed blocks (boxcar_tap, trig_sync, live_window_gate_zl, NN_live, live_res_cdc x2, dac_gate) wired as in the
// block design. One unit per readout tone. Starts at the readout's mixed stream (axis_dyn_readout m0); the
// per-tone mixer is QICK's and is not included, nor are readout 1's capture buffer, seg_stamp and rf_tap.
// Input registers model the readout's registered output.
// Variant: the window gate's output register is replaced by a register on NN_live's result, before both
// CDCs; same end-to-end latency. Not deployed.
module classifier_unit_zl (
    input  wire         clk_adc,     // clk_adc0_x2, 552.96 MHz
    input  wire         rstn_adc,
    input  wire         clk_core,    // tProc core clock, 200 MHz
    input  wire         rstn_core,
    input  wire         clk_dac,     // RFDAC2 clock, 491.52 MHz
    input  wire         rstn_dac,
    input  wire         clk_axi,     // host side of the record RAM

    input  wire [255:0] s_mix_tdata,
    input  wire         s_mix_tvalid,
    input  wire [31:0]  s_fir_tdata,
    input  wire         s_fir_tvalid,
    input  wire [31:0]  box_cfg,
    input  wire         trig_async,  // tProc classify trigger (asynchronous to clk_adc)

    input  wire [255:0] s_dac_tdata, // generator stream through the gate
    input  wire         s_dac_tvalid,
    input  wire         marker,
    input  wire         cls_trig,
    input  wire [31:0]  gate_bram_dout,

    input  wire [12:0]  rec_addrb,
    output wire [31:0]  rec_doutb,

    output wire [63:0]  core_tdata,  // decision word at the tProc
    output wire         core_tvalid,
    output wire [255:0] dac_tdata,   // gated stream to the DAC
    output wire         dac_tvalid,
    output wire [31:0]  gate_bram_xor
);
    // local reset and input registers (model the readout's output register)
    reg [1:0] rst_a, rst_c, rst_d;
    always @(posedge clk_adc)  rst_a <= {rst_a[0], rstn_adc};
    always @(posedge clk_core) rst_c <= {rst_c[0], rstn_core};
    always @(posedge clk_dac)  rst_d <= {rst_d[0], rstn_dac};
    wire ra = rst_a[1], rc = rst_c[1], rd = rst_d[1];

    reg [255:0] mix_r;  reg mix_v;
    reg [31:0]  fir_r;  reg fir_v;
    reg [31:0]  cfg_r;
    always @(posedge clk_adc) begin
        mix_r <= s_mix_tdata;  mix_v <= s_mix_tvalid;
        fir_r <= s_fir_tdata;  fir_v <= s_fir_tvalid;
        cfg_r <= box_cfg;
    end
    reg [255:0] dac_r; reg dac_v;
    always @(posedge clk_dac) begin dac_r <= s_dac_tdata; dac_v <= s_dac_tvalid; end

    // boxcar -> gate -> NN_live
    wire [31:0] nn_d;  wire nn_v;
    boxcar_tap u_box (
        .aclk(clk_adc), .aresetn(ra),
        .s_axis_tdata(mix_r), .s_axis_tvalid(mix_v), .s_axis_tready(),
        .m_mr_axis_tdata(), .m_mr_axis_tvalid(), .m_mr_axis_tready(1'b1),
        .s_fir_tdata(fir_r), .s_fir_tvalid(fir_v),
        .m_nn_tdata(nn_d), .m_nn_tvalid(nn_v),
        .m_box_axis_tdata(), .m_box_axis_tvalid(), .m_box_axis_tready(1'b1),
        .cfg(cfg_r));

    wire trig_s;
    trig_sync u_ts (.aclk(clk_adc), .trig_in(trig_async), .trig_out(trig_s));

    wire [31:0] g_d; wire g_v, g_l, nn_rdy; wire [3:0] g_k, g_s;
    live_window_gate_zl #(.N(576)) u_gate (
        .aclk(clk_adc), .aresetn(ra), .trigger(trig_s),
        .s_axis_tdata(nn_d), .s_axis_tvalid(nn_v),
        .m_axis_tdata(g_d), .m_axis_tvalid(g_v), .m_axis_tlast(g_l),
        .m_axis_tkeep(g_k), .m_axis_tstrb(g_s), .m_axis_tready(nn_rdy));

    wire [63:0] res; wire res_v;
    wire [31:0] o_addr, o_din; wire o_en; wire [3:0] o_we;
    NN_live u_nn (
        .ap_clk(clk_adc), .ap_rst_n(ra),
        .in_r_TDATA(g_d), .in_r_TVALID(g_v), .in_r_TREADY(nn_rdy),
        .in_r_TKEEP(g_k), .in_r_TSTRB(g_s), .in_r_TLAST(g_l),
        .res_o(res), .res_o_ap_vld(res_v),
        .out_r_Addr_A(o_addr), .out_r_EN_A(o_en), .out_r_WEN_A(o_we), .out_r_Din_A(o_din),
        .out_r_Dout_A(32'd0), .out_r_Clk_A(), .out_r_Rst_A());

    // register NN_live's result before both CDCs
    reg [63:0] res_r; reg res_vr;
    always @(posedge clk_adc) begin
        res_r  <= res;
        res_vr <= ra & res_v;
    end

    live_record_ram u_rec (
        .clka(clk_adc), .addra(o_addr), .ena(o_en), .wea(o_we), .dina(o_din),
        .clkb(clk_axi), .addrb(rec_addrb), .doutb(rec_doutb));

    // decision -> tProc (core) and -> gate (DAC)
    live_res_cdc u_cdc0 (
        .s_aclk(clk_adc), .s_aresetn(ra), .res_o(res_r), .res_o_ap_vld(res_vr),
        .m_aclk(clk_core), .m_aresetn(rc),
        .m_axis_tdata(core_tdata), .m_axis_tvalid(core_tvalid), .m_axis_tready(1'b1));

    wire [63:0] dec_d; wire dec_v;
    live_res_cdc u_cdc1 (
        .s_aclk(clk_adc), .s_aresetn(ra), .res_o(res_r), .res_o_ap_vld(res_vr),
        .m_aclk(clk_dac), .m_aresetn(rd),
        .m_axis_tdata(dec_d), .m_axis_tvalid(dec_v), .m_axis_tready(1'b1));

    wire g_bclk, g_brst, g_ben; wire [3:0] g_bwe; wire [31:0] g_baddr, g_bdin;
    dac_gate #(.NS(16), .EARLY_BLANK(1), .SHOT_BIND(1), .FAIL_SAFE(1), .CFG_SAFE(0)) u_dg (
        .aclk(clk_dac), .aresetn(rd),
        .s_axis_tdata(dac_r), .s_axis_tvalid(dac_v), .s_axis_tready(),
        .m_axis_tdata(dac_tdata), .m_axis_tvalid(dac_tvalid), .m_axis_tready(1'b1),
        .dec_axis_tdata(dec_d), .dec_axis_tvalid(dec_v), .dec_axis_tready(),
        .marker(marker), .cls_trig(cls_trig),
        .bram_clk(g_bclk), .bram_rst(g_brst), .bram_en(g_ben), .bram_we(g_bwe),
        .bram_addr(g_baddr), .bram_din(g_bdin), .bram_dout(gate_bram_dout));
    // keep the gate's record-port writes observable (a BRAM in the real design)
    reg [31:0] gx;
    always @(posedge clk_dac) gx <= g_bdin ^ g_baddr ^ {27'd0, g_ben, g_bwe};
    assign gate_bram_xor = gx;
endmodule
