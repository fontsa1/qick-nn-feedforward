// live_window_gate_zl: 4.5b timing experiment only (not deployed). live_window_gate with the output
// register removed: m_axis_* are combinational from take / done / s_axis_tdata, so NN_live sees each
// word one cycle earlier. Paired with a register on NN_live's res_o in classifier_unit_zl, the total
// readout-output -> registered-decision latency is unchanged (gate reg moved to the result side).
// Framing rules identical to live_window_gate (buffer.sv semantics).

`timescale 1ns / 1ps

module live_window_gate_zl #(
    parameter integer N = 770
) (
    input  wire        aclk,
    input  wire        aresetn,
    input  wire        trigger,
    input  wire [31:0] s_axis_tdata,
    input  wire        s_axis_tvalid,
    output wire [31:0] m_axis_tdata,
    output wire        m_axis_tvalid,
    output wire        m_axis_tlast,
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
            trig_sync <= 2'b00;
            state     <= S_TRIG;
            cnt       <= 16'd0;
        end else begin
            trig_sync <= {trig_sync[0], trigger};
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

    assign m_axis_tdata  = s_axis_tdata;
    assign m_axis_tvalid = take;
    assign m_axis_tlast  = done;
    assign m_axis_tkeep  = 4'hF;
    assign m_axis_tstrb  = 4'hF;

endmodule
