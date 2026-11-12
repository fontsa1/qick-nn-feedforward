// live_res_axis: NN_live's per-shot result (res_o, ap_vld) as a 64-bit AXIS beat
// for a tProc_v2 input port (task 4.1 milestone 2, v1.40L2).
//
// res_o (NN_live.h): [15:0] logit, ap_fixed<16,6> raw (Q.10), [44:16] shot,
// [63] decision. The tProc's DPORT_RD loads tdata[31:0] into s_port_l and
// tdata[63:32] into s_port_h, so:
//   tdata[31:0]  = logit sign-extended to 32 bits: a program branches with
//                  TEST s_port_l - #thr ; JUMP NS  (logit >= thr/1024),
//                  i.e. QICK's read_and_jump(component='I'), any threshold
//   tdata[60:32] = shot count (29 bits), to detect a stale read
//   tdata[63]    = decision bit (logit >= 1/64)
// One registered stage on the readout clock. tvalid pulses once per shot; the
// downstream axis_clock_converter is never full at that rate, so tready is
// ignored (the tProc's own inputs tie tready high).
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
