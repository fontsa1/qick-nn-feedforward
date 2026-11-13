// live_res_cdc: NN_live's per-shot result as one 64-bit AXIS beat, moved from the readout clock (s_aclk,
// clk_adc0_x2) into another clock domain (m_aclk) by a hold-and-toggle handshake (~3 destination cycles).
// Two instances: to the tProc core clock (tProc input port) and to clk_dac2 (dac_gate).
//
// Word (from res_o, NN_live.h):
//   [31:0]  logit, sign-extended (ap_fixed<16,6>, Q.10)
//   [60:32] shot count (29 bits)
//   [63]    decision bit (logit >= 1/64)
//
// Source: on res_o_ap_vld the word is written to `hold` and `tog` flips in the same cycle.
// Destination: `tog` passes a 2-flop synchronizer; a change registers `hold` into m_axis_tdata and pulses
// m_axis_tvalid for one cycle. `hold` is stable for at least one destination cycle before it is sampled, and
// results are >= ~1 us apart. The two clocks are asynchronous groups in QICK's timing.xdc; the data-vs-toggle
// skew is bounded by constraints/live_res_cdc.xdc. m_axis_tready is ignored.
`timescale 1ns / 1ps

module live_res_cdc (
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
            tog_sync      <= {tog_sync[0], tog};
            tog_seen      <= tog_sync[1];
            m_axis_tvalid <= change;
            if (change)
                m_axis_tdata <= hold;
        end
    end

endmodule
