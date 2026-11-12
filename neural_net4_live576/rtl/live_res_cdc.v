// live_res_cdc: NN_live's per-shot result (res_o, ap_vld) as one 64-bit AXIS
// beat for a tProc_v2 input port, crossing from the readout clock into the
// tProc core clock (v1.40L5). Replaces live_res_axis + axis_cc_live (an async
// FIFO, ~6 core cycles) with a hold-and-toggle handshake (~3 core cycles).
//
// Word layout as live_res_axis.v (NN_live.h):
//   tdata[31:0]  = logit sign-extended to 32 bits (ap_fixed<16,6> raw, Q.10)
//   tdata[60:32] = shot count (29 bits), to detect a stale read
//   tdata[63]    = decision bit (logit >= 1/64)
//
// Source (s_aclk, clk_adc0_x2): on res_o_ap_vld the formatted word is written
// to `hold` and `tog` flips, in the same cycle.
// Destination (m_aclk, the tProc core clock): `tog` passes a two-flop ASYNC_REG
// synchronizer; a change at its output registers `hold` into m_axis_tdata and
// pulses m_axis_tvalid for one cycle.
// Why sampling `hold` is safe: it was written in the same source cycle as the
// toggle, and is sampled only after the toggle has passed two destination flops,
// i.e. at least one full destination cycle later. It does not change again until
// the next result, and results arrive at most once per shot (>= ~1 us apart, vs
// 3 core cycles of latency).
// The two clocks are asynchronous clock groups in timing.xdc (clk_core vs
// clk_adc0_x2), so these paths are not timed; the argument above is the guarantee.
// m_axis_tready is ignored: the tProc's input ports latch on tvalid.
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
