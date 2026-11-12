// rf_tap: pulse detector on the RFdc ADC output (v1.40L9, task 4.1h split).
//
// Sits on usp_rf_data_converter_0/m00_axis (ADC tile 0 channel 0, readout 0's
// input; 8 real 16-bit samples per clk_adc0_x2 cycle) in front of
// axis_register_slice_2. The stream passes through combinationally (no
// register, no change), so QICK's readout timing is untouched.
//
// Monitor: hit = 1 when any of the word's 8 samples has |x| > thr. Four
// register stages (|x|, compare, OR of each half-word, OR), so hit is 4 cycles
// behind the word at the tile output (v1.40L10; 3 in v1.40L9, whose single
// 8-input OR missed timing at 552.96 MHz). seg_stamp_0 timestamps the first hit after the classify
// trigger (edge0: the window's own pulse at the tile) and the first armed hit
// after the window (edge3: the conditional pulse coming back). Against
// seg_stamp's readout-output edges these give the readout front end, and
// against dac_gate_0's timestamps, DAC tile + cable + ADC tile.
// thr comes from seg_stamp_0 BRAM word 4 (0 at reset: every nonzero sample hits).
`timescale 1ns / 1ps

module rf_tap #(
    parameter integer NS = 8
) (
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 aclk CLK" *)
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF s_axis:m_axis" *)
    input  wire             aclk,

    input  wire [16*NS-1:0] s_axis_tdata,
    input  wire             s_axis_tvalid,
    output wire             s_axis_tready,

    output wire [16*NS-1:0] m_axis_tdata,
    output wire             m_axis_tvalid,
    input  wire             m_axis_tready,

    input  wire [15:0]      thr,
    output reg              hit
);

    assign m_axis_tdata  = s_axis_tdata;
    assign m_axis_tvalid = s_axis_tvalid;
    assign s_axis_tready = m_axis_tready;

    reg [15:0]   absv [0:NS-1];
    reg [NS-1:0] over;
    reg [1:0]    half;
    reg [15:0]   thr_r;
    reg          v1;
    integer k;

    initial hit = 1'b0;

    always @(posedge aclk) begin
        thr_r <= thr;
        v1    <= s_axis_tvalid;
        for (k = 0; k < NS; k = k + 1) begin
            absv[k] <= s_axis_tdata[16*k+15] ? (~s_axis_tdata[16*k +: 16] + 16'd1)
                                             : s_axis_tdata[16*k +: 16];
            over[k] <= v1 && (absv[k] > thr_r);
        end
        half <= {|over[NS-1:NS/2], |over[NS/2-1:0]};
        hit  <= |half;
    end

endmodule
