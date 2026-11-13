// live_res_axis: NN_live's per-shot result (res_o, ap_vld) as a 64-bit AXIS beat for a tProc_v2 input port,
// on the readout clock. Same word as live_res_cdc without the clock crossing; not used in the v1.44 design.
//
// res_o (NN_live.h): [15:0] logit, ap_fixed<16,6> (Q.10), [44:16] shot, [63] decision. Output word, which
// DPORT_RD loads as s_port_l = [31:0], s_port_h = [63:32]:
//   [31:0]  logit, sign-extended (TEST s_port_l - #thr; JUMP NS branches on logit >= thr/1024)
//   [60:32] shot count (29 bits)
//   [63]    decision bit (logit >= 1/64)
// One register stage. tvalid pulses once per shot; tready is ignored.
`timescale 1ns / 1ps

module live_res_axis (
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 aclk CLK" *)
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF m_axis, ASSOCIATED_RESET aresetn" *)
    input  wire        aclk,
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 aresetn RST" *)
    (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input  wire        aresetn,

    input  wire [63:0] res_o,
    input  wire        res_o_ap_vld,

    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 m_axis TDATA" *)
    output reg  [63:0] m_axis_tdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 m_axis TVALID" *)
    output reg         m_axis_tvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 m_axis TREADY" *)
    input  wire        m_axis_tready
);

    always @(posedge aclk) begin
        if (!aresetn) begin
            m_axis_tvalid <= 1'b0;
            m_axis_tdata  <= 64'd0;
        end else begin
            m_axis_tvalid <= res_o_ap_vld;
            if (res_o_ap_vld)
                m_axis_tdata <= {res_o[63], 2'b00, res_o[44:16],
                                 {16{res_o[15]}}, res_o[15:0]};
        end
    end

endmodule
