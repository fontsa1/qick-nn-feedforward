// trig_sync: one synchronizer for the classify trigger, shared by NN_live's
// window and the boxcar capture buffer (v1.40L9, task 4.1i).
//
// Before v1.40L9, live_gate_0 (classify trigger, tProc trig_11) and
// axis_avg_buffer_1 (readout 1 trigger, tProc trig_1) each brought an
// asynchronous tProc trigger into clk_adc0_x2 through their own 2-flop
// synchronizer. timing.xdc false-paths the tProc's trig_*_o, so the two routes
// are untimed and the two synchronizers could take the same edge on different
// cycles: NN_live's window and acquire_decimated()'s buffer then differed by one
// sample on 0-40% of shots, depending on the build's placement.
//
// Here the classify trigger is synchronized once (2 ASYNC_REG flops) and
// registered; that one signal drives both live_gate_0/trigger and
// axis_avg_buffer_1/trigger. Each block's own 2-flop synchronizer now sees an
// input that is already synchronous to its clock, so both take every edge on
// the same cycle (a fixed 2-cycle delay each), and the window equals the buffer
// by construction. Cost: 3 cycles (5.4 ns) between the trigger and both
// captures, i.e. the same samples are captured if the trigger is advanced by
// that much; nothing in the loop from the window's last sample onward changes.
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
