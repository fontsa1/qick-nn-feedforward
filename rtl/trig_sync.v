// trig_sync: 2-flop synchronizer plus one register for the classify trigger (tProc output) into clk_adc0_x2.
//
// Its one output drives both the NN_live window gate and readout 1's capture buffer, whose own synchronizers
// then see a signal already synchronous to their clock. Both therefore take the trigger on the same cycle, and
// NN_live's window holds the same samples as acquire_decimated() on readout 1. Adds 3 cycles (5.4 ns) between
// the trigger and both captures.
`timescale 1ns / 1ps

module trig_sync (
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 aclk CLK" *)
    input  wire aclk,
    input  wire trig_in,
    output wire trig_out
);

    (* ASYNC_REG = "TRUE" *) reg [1:0] s = 2'b00;
    reg o = 1'b0;

    always @(posedge aclk) begin
        s <= {s[0], trig_in};
        o <= s[1];
    end

    assign trig_out = o;

endmodule
