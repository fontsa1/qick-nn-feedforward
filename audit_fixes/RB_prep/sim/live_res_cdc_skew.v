// live_res_cdc_skew: live_res_cdc with simulation-only delays: SKEW_PS on `hold` and TOG_PS on `tog` before
// the destination samples them. Used by tb_cdc_skew. See neural_net4_live576/rtl/live_res_cdc.v.
`timescale 1ns / 1ps

module live_res_cdc_skew #(parameter integer SKEW_PS = 0, parameter integer TOG_PS = 0) (
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 s_aclk CLK" *)
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_RESET s_aresetn" *)
    input  wire        s_aclk,
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 s_aresetn RST" *)
    (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input  wire        s_aresetn,

    input  wire [63:0] res_o,
    input  wire        res_o_ap_vld,

    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 m_aclk CLK" *)
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF m_axis, ASSOCIATED_RESET m_aresetn" *)
    input  wire        m_aclk,
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 m_aresetn RST" *)
    (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input  wire        m_aresetn,

    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 m_axis TDATA" *)
    output reg  [63:0] m_axis_tdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 m_axis TVALID" *)
    output reg         m_axis_tvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 m_axis TREADY" *)
    input  wire        m_axis_tready
);

    // ---- source domain
    reg [63:0] hold;
    reg        tog;
    always @(posedge s_aclk) begin
        if (!s_aresetn) begin
            hold <= 64'd0;
            tog  <= 1'b0;
        end else if (res_o_ap_vld) begin
            hold <= {res_o[63], 2'b00, res_o[44:16], {16{res_o[15]}}, res_o[15:0]};
            tog  <= ~tog;
        end
    end

    wire [63:0] hold_d; wire tog_d;
    assign #(SKEW_PS/1000.0) hold_d = hold;
    assign #(TOG_PS/1000.0) tog_d = tog;
    // ---- destination domain
    (* ASYNC_REG = "TRUE" *) reg [1:0] tog_sync;
    reg tog_seen;
    wire change = tog_sync[1] ^ tog_seen;
    always @(posedge m_aclk) begin
        if (!m_aresetn) begin
            tog_sync      <= 2'b00;
            tog_seen      <= 1'b0;
            m_axis_tdata  <= 64'd0;
            m_axis_tvalid <= 1'b0;
        end else begin
            tog_sync      <= {tog_sync[0], tog_d};
            tog_seen      <= tog_sync[1];
            m_axis_tvalid <= change;
            if (change)
                m_axis_tdata <= hold_d;
        end
    end

endmodule
