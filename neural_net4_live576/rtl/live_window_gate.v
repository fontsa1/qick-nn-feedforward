// live_window_gate: frames the live readout stream into classifier windows.
//
// axis_dyn_readout_v1_0.m1_axis (via axis_broadcaster_0.M02_AXIS) is a
// continuous stream with no TLAST and no backpressure. This block passes
// exactly N samples per tProc readout trigger and marks the last with TLAST,
// using the same rules as axis_avg_buffer_0's decimated buffer (buffer.sv),
// so NN_live sees the samples acquire_decimated() returns for the same shot:
//   * the trigger crosses into aclk through a 2-flop synchronizer
//     (avg_buffer.v: synchronizer_n #(.N(2)));
//   * TRIGGER -> MEMW on the synchronized trigger, level-sensitive;
//   * in MEMW, every valid input word is taken, starting with the word
//     present in the first MEMW cycle, until N have been taken;
//   * WAIT_TRIGGER until the synchronized trigger is low, then re-arm.
// (buffer.sv's START_REG enable is omitted: this gate is always armed.)
//
// The output is registered, one cycle behind; the content is unchanged.
// m_axis_tready is ignored: the source cannot be stalled, and any word the
// consumer misses surfaces there as a window without TLAST on word N-1.

`timescale 1ns / 1ps

module live_window_gate #(
    parameter integer N = 770
) (
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 aclk CLK" *)
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF s_axis:m_axis, ASSOCIATED_RESET aresetn" *)
    input  wire        aclk,
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 aresetn RST" *)
    (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input  wire        aresetn,

    input  wire        trigger,

    input  wire [31:0] s_axis_tdata,
    input  wire        s_axis_tvalid,

    output reg  [31:0] m_axis_tdata,
    output reg         m_axis_tvalid,
    output reg         m_axis_tlast,
    output wire [3:0]  m_axis_tkeep,
    output wire [3:0]  m_axis_tstrb,
    input  wire        m_axis_tready
);

    (* ASYNC_REG = "TRUE" *) reg [1:0] trig_sync;
    wire trig_r = trig_sync[1];

    localparam [1:0] S_TRIG = 2'd0, S_MEMW = 2'd1, S_WAIT = 2'd2;
    reg [1:0]  state;
    reg [15:0] cnt;

    wire take = (state == S_MEMW) && s_axis_tvalid;
    wire done = take && (cnt == N - 1);

    always @(posedge aclk) begin
        if (!aresetn) begin
            trig_sync     <= 2'b00;
            state         <= S_TRIG;
            cnt           <= 16'd0;
            m_axis_tdata  <= 32'd0;
            m_axis_tvalid <= 1'b0;
            m_axis_tlast  <= 1'b0;
        end else begin
            trig_sync <= {trig_sync[0], trigger};

            m_axis_tdata  <= s_axis_tdata;
            m_axis_tvalid <= take;
            m_axis_tlast  <= done;

            case (state)
                S_TRIG: if (trig_r) state <= S_MEMW;
                S_MEMW: if (done) state <= S_WAIT;
                S_WAIT: if (!trig_r) state <= S_TRIG;
                default: state <= S_TRIG;
            endcase

            if (state == S_MEMW) begin
                if (s_axis_tvalid) cnt <= cnt + 16'd1;
            end else begin
                cnt <= 16'd0;
            end
        end
    end

    assign m_axis_tkeep = 4'hF;
    assign m_axis_tstrb = 4'hF;

endmodule
