// live_rec_wreg: one register stage on NN_live's record-memory write port (v1.42, FINAL_TASKS 4.1m / 4.5b).
//
// In v1.41, NN_live's out_r_PORTA drives axi_bram_ctrl_live_bram (8 x RAMB36) directly, and the path
// NN_live FSM -> 3 LUT -> 8 RAMB36 write inputs fails setup in clk_adc0_x2 (42 endpoints). This block sits
// between the two: Addr/EN/WEN/Din go through one register, so the RAM inputs launch from flops placed
// next to it. Clk, Rst and Dout pass straight through.
//
// Safe because NN_live never reads out_r (out_r_Dout_A is unused in NN_live.v): delaying its writes by
// one cycle changes nothing it can observe, and the host reads the record long after the shot. The same
// stage was timed out of context in scaling_study/rtl/live_record_ram.v (clean in every 4.5b run).
//
// Verified by rtl_sim/unit/tb_u_live_rec_wreg.sv.
`timescale 1ns / 1ps

module live_rec_wreg (
    // from NN_live_0/out_r_PORTA (BRAM master)
    (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 s_bram CLK" *)
    input  wire        s_clk,
    (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 s_bram RST" *)
    input  wire        s_rst,
    (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 s_bram ADDR" *)
    input  wire [31:0] s_addr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 s_bram EN" *)
    input  wire        s_en,
    (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 s_bram WE" *)
    input  wire [3:0]  s_we,
    (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 s_bram DIN" *)
    input  wire [31:0] s_din,
    (* X_INTERFACE_PARAMETER = "MASTER_TYPE BRAM_CTRL, MEM_WIDTH 32, MEM_SIZE 4, MEM_ECC NONE, READ_LATENCY 1" *)
    (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 s_bram DOUT" *)
    output wire [31:0] s_dout,

    // to axi_bram_ctrl_live_bram/BRAM_PORTB (BRAM master again)
    (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 m_bram CLK" *)
    output wire        m_clk,
    (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 m_bram RST" *)
    output wire        m_rst,
    (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 m_bram ADDR" *)
    output reg  [31:0] m_addr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 m_bram EN" *)
    output reg         m_en,
    (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 m_bram WE" *)
    output reg  [3:0]  m_we,
    (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 m_bram DIN" *)
    output reg  [31:0] m_din,
    (* X_INTERFACE_PARAMETER = "MASTER_TYPE BRAM_CTRL, MEM_WIDTH 32, MEM_SIZE 4, MEM_ECC NONE, READ_LATENCY 1" *)
    (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 m_bram DOUT" *)
    input  wire [31:0] m_dout
);

    assign m_clk  = s_clk;
    assign m_rst  = s_rst;
    assign s_dout = m_dout;

    initial begin m_addr = 32'd0; m_en = 1'b0; m_we = 4'd0; m_din = 32'd0; end

    always @(posedge s_clk) begin
        m_addr <= s_addr;
        m_en   <= s_en;
        m_we   <= s_we;
        m_din  <= s_din;
    end

endmodule
