`timescale 1ns / 1ps
// ooc_top_t: NU copies of classifier_unit_t for the out-of-context timing study.
// Clocks enter through BUFGs. Shared input buses are registered once and fanned out to every unit, as the RF
// data converter's stream fans out to one readout per tone. Units are DONT_TOUCH (run_ooc_t.tcl) so identical
// units are not merged. Outputs are XOR-reduced per clock domain so nothing is trimmed. Harness registers are
// outside u_unit[*], so unit-internal and harness endpoints are reported separately.
module ooc_top_t #(
    parameter integer NU = 1
) (
    input  wire         clk_adc_p, clk_core_p, clk_dac_p, clk_axi_p,
    input  wire         rstn_adc, rstn_core, rstn_dac,
    input  wire [255:0] s_mix_tdata,
    input  wire         s_mix_tvalid,
    input  wire [31:0]  s_fir_tdata,
    input  wire         s_fir_tvalid,
    input  wire [31:0]  box_cfg,
    input  wire [NU-1:0] trig_async,
    input  wire [255:0] s_dac_tdata,
    input  wire         s_dac_tvalid,
    input  wire         marker,
    input  wire         cls_trig,
    input  wire [31:0]  gate_bram_dout,
    input  wire [12:0]  rec_addrb,
    output reg  [31:0]  rec_x,
    output reg  [63:0]  core_x,
    output reg  [255:0] dac_x,
    output reg  [31:0]  gate_x
);
    wire clk_adc, clk_core, clk_dac, clk_axi;
    BUFG b0 (.I(clk_adc_p),  .O(clk_adc));
    BUFG b1 (.I(clk_core_p), .O(clk_core));
    BUFG b2 (.I(clk_dac_p),  .O(clk_dac));
    BUFG b3 (.I(clk_axi_p),  .O(clk_axi));

    reg [255:0] mix_h; reg mix_vh; reg [31:0] fir_h; reg fir_vh; reg [31:0] cfg_h;
    always @(posedge clk_adc) begin
        mix_h <= s_mix_tdata; mix_vh <= s_mix_tvalid; fir_h <= s_fir_tdata; fir_vh <= s_fir_tvalid; cfg_h <= box_cfg;
    end
    reg [255:0] dac_h; reg dac_vh; reg mk_h, ct_h; reg [31:0] gd_h;
    always @(posedge clk_dac) begin
        dac_h <= s_dac_tdata; dac_vh <= s_dac_tvalid; mk_h <= marker; ct_h <= cls_trig; gd_h <= gate_bram_dout;
    end
    reg [12:0] ra_h;
    always @(posedge clk_axi) ra_h <= rec_addrb;

    wire [31:0]  rec_o  [0:NU-1];
    wire [63:0]  core_o [0:NU-1];
    wire         core_v [0:NU-1];
    wire [255:0] dac_o  [0:NU-1];
    wire         dac_v  [0:NU-1];
    wire [31:0]  gate_o [0:NU-1];

    genvar u;
    generate for (u = 0; u < NU; u = u + 1) begin : g
        (* keep_hierarchy = "yes", DONT_TOUCH = "yes" *)
        classifier_unit_t u_unit (
            .clk_adc(clk_adc), .rstn_adc(rstn_adc), .clk_core(clk_core), .rstn_core(rstn_core),
            .clk_dac(clk_dac), .rstn_dac(rstn_dac), .clk_axi(clk_axi),
            .s_mix_tdata(mix_h), .s_mix_tvalid(mix_vh), .s_fir_tdata(fir_h), .s_fir_tvalid(fir_vh),
            .box_cfg(cfg_h), .trig_async(trig_async[u]),
            .s_dac_tdata(dac_h), .s_dac_tvalid(dac_vh), .marker(mk_h), .cls_trig(ct_h),
            .gate_bram_dout(gd_h),
            .rec_addrb(ra_h), .rec_doutb(rec_o[u]),
            .core_tdata(core_o[u]), .core_tvalid(core_v[u]),
            .dac_tdata(dac_o[u]), .dac_tvalid(dac_v[u]),
            .gate_bram_xor(gate_o[u]));
    end endgenerate

    // per-domain XOR reduction across units, registered
    integer k;
    reg [31:0] rx; reg [63:0] cx; reg [255:0] dx; reg [31:0] gx;
    always @* begin
        rx = 0; cx = 0; dx = 0; gx = 0;
        for (k = 0; k < NU; k = k + 1) begin
            rx = rx ^ rec_o[k];
            cx = cx ^ (core_o[k] & {64{core_v[k]}});
            dx = dx ^ (dac_o[k] & {256{dac_v[k]}});
            gx = gx ^ gate_o[k];
        end
    end
    always @(posedge clk_axi)  rec_x  <= rx;
    always @(posedge clk_core) core_x <= cx;
    always @(posedge clk_dac)  begin dac_x <= dx; gate_x <= gx; end
endmodule
