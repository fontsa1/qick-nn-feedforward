// boxcar_tap: low-latency readout path for NN_live (task 4.1h, v1.40L7).
//
// axis_dyn_readout_v1_0 mixes the ADC stream down with its DDS (m0_axis, 8
// complex samples per 552.96 MHz clock, sample k in bits [32k+31:32k] as
// {Q[15:0], I[15:0]}) and then decimates it 8x through a 120-tap systolic FIR
// (m1_axis, one sample per clock), which takes ~121 ns (C_LATENCY 67). This
// block builds a second decimated stream from m0 without the FIR:
//   box8  sum of the clock's 8 mixed samples (decimate by 8; 3-level adder tree)
//   msum  moving sum of the last M box8 values, M = 1..16 (cfg). The readout
//         mixes a REAL ADC signal, so the stream also carries the 2f image (at
//         -100 MHz for our 50 MHz readout) at full strength; box8 alone passes it
//         at -0.5 dB, while M = 11 puts a null at 552.96/11 * 2 = 100.5 MHz.
//   out   round(msum / 2^shift), saturated to 16 bits, {Q, I} like m1_axis.
// Latency from m0 to the output: 9 cycles (16.3 ns) + the moving sum's group
// delay (M-1)/2 samples.
//
// Ports:
//   s_axis      <- readout m0 (256 bits); passed straight through to m_mr_axis,
//                  so the MR buffer path is unchanged (combinational, no delay).
//   s_fir       <- readout m1 via axis_broadcaster_0 (32 bits; the FIR stream).
//   m_nn        -> live_gate_0: cfg[0] ? boxcar : FIR, combinational (no added
//                  latency on the FIR path; live_gate_0 registers its input).
//   m_box_axis  -> axis_broadcaster_1 (avg buffer 1 + DDR4 ch 1): the boxcar
//                  stream, so acquire_decimated() on readout 1 returns exactly
//                  what NN_live sees with cfg[0] = 1 (training data, bit-exact check).
//   cfg         <- seg_stamp_0 word 3: [0] NN input select, [7:4] shift,
//                  [12:8] M - 1. Any change clears the moving sum (16 cycles to refill).

`timescale 1ns / 1ps

module boxcar_tap (
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 aclk CLK" *)
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF s_axis:m_mr_axis:s_fir:m_nn:m_box_axis, ASSOCIATED_RESET aresetn" *)
    input  wire         aclk,
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 aresetn RST" *)
    (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input  wire         aresetn,

    input  wire [255:0] s_axis_tdata,
    input  wire         s_axis_tvalid,
    output wire         s_axis_tready,

    output wire [255:0] m_mr_axis_tdata,
    output wire         m_mr_axis_tvalid,
    input  wire         m_mr_axis_tready,

    input  wire [31:0]  s_fir_tdata,
    input  wire         s_fir_tvalid,

    output wire [31:0]  m_nn_tdata,
    output wire         m_nn_tvalid,

    output reg  [31:0]  m_box_axis_tdata,
    output reg          m_box_axis_tvalid,
    input  wire         m_box_axis_tready,

    input  wire [31:0]  cfg
);

    // ---- MR pass-through
    assign m_mr_axis_tdata  = s_axis_tdata;
    assign m_mr_axis_tvalid = s_axis_tvalid;
    assign s_axis_tready    = m_mr_axis_tready;

    // ---- configuration (registered; replicated select)
    reg [31:0] cfg_r, cfg_rr;
    (* max_fanout = 8 *)  reg       sel;
    (* max_fanout = 16 *) reg [3:0] msel;      // M - 1
    reg [3:0] shift;
    reg       clr;                             // cfg changed: clear the moving sum
    always @(posedge aclk) begin
        if (!aresetn) begin
            cfg_r <= 0; cfg_rr <= 0; sel <= 0; msel <= 0; shift <= 0; clr <= 1'b1;
        end else begin
            cfg_r  <= cfg;
            cfg_rr <= cfg_r;
            sel    <= cfg_rr[0];
            msel   <= cfg_rr[11:8];
            shift  <= cfg_rr[7:4];
            clr    <= (cfg_rr[12:0] != cfg_r[12:0]);
        end
    end

    // ---- stage 1-3: adder tree over the 8 lanes (I and Q separately)
    reg signed [16:0] s1i [0:3], s1q [0:3];
    reg signed [17:0] s2i [0:1], s2q [0:1];
    reg signed [18:0] b8i, b8q;
    reg [2:0] v13;
    integer k;
    always @(posedge aclk) begin
        for (k = 0; k < 4; k = k + 1) begin
            s1i[k] <= $signed(s_axis_tdata[64*k +: 16])      + $signed(s_axis_tdata[64*k + 32 +: 16]);
            s1q[k] <= $signed(s_axis_tdata[64*k + 16 +: 16]) + $signed(s_axis_tdata[64*k + 48 +: 16]);
        end
        for (k = 0; k < 2; k = k + 1) begin
            s2i[k] <= s1i[2*k] + s1i[2*k+1];
            s2q[k] <= s1q[2*k] + s1q[2*k+1];
        end
        b8i <= s2i[0] + s2i[1];
        b8q <= s2q[0] + s2q[1];
        v13 <= aresetn ? {v13[1:0], s_axis_tvalid} : 3'd0;
    end

    // ---- stage A-C: moving sum of M box8 values
    //   sr[j] holds box8 from j+1 cycles ago (relative to b8 now)
    //   A: new_r <= b8(n), old_r <= b8(n-M) = sr[M-1]
    //   B: diff <= new_r - old_r          C: acc <= acc + diff
    reg signed [18:0] sri [0:15], srq [0:15];
    reg signed [18:0] new_i, new_q, old_i, old_q;
    reg signed [19:0] dif_i, dif_q;
    reg signed [22:0] acc_i, acc_q;
    reg [2:0] vac;
    always @(posedge aclk) begin
        if (clr) begin
            for (k = 0; k < 16; k = k + 1) begin sri[k] <= 0; srq[k] <= 0; end
            new_i <= 0; new_q <= 0; old_i <= 0; old_q <= 0;
            dif_i <= 0; dif_q <= 0; acc_i <= 0; acc_q <= 0; vac <= 0;
        end else begin
            sri[0] <= b8i; srq[0] <= b8q;
            for (k = 1; k < 16; k = k + 1) begin sri[k] <= sri[k-1]; srq[k] <= srq[k-1]; end
            new_i <= b8i;        new_q <= b8q;
            old_i <= sri[msel];  old_q <= srq[msel];
            dif_i <= new_i - old_i;
            dif_q <= new_q - old_q;
            acc_i <= acc_i + dif_i;
            acc_q <= acc_q + dif_q;
            vac   <= {vac[1:0], v13[2]};
        end
    end

    // ---- stage D-F: round, shift, saturate
    reg signed [22:0] rnd;
    reg signed [23:0] rd_i, rd_q;
    reg signed [23:0] sh_i, sh_q;
    reg [2:0] vsh;
    function [15:0] sat16(input signed [23:0] x);
        sat16 = (x > 24'sd32767) ? 16'h7FFF : (x < -24'sd32768) ? 16'h8000 : x[15:0];
    endfunction
    always @(posedge aclk) begin
        rnd  <= (shift == 0) ? 23'sd0 : (23'sd1 <<< (shift - 1));
        rd_i <= acc_i + rnd;
        rd_q <= acc_q + rnd;
        sh_i <= rd_i >>> shift;
        sh_q <= rd_q >>> shift;
        m_box_axis_tdata  <= {sat16(sh_q), sat16(sh_i)};
        vsh               <= aresetn ? {vsh[1:0], vac[2]} : 3'd0;
        m_box_axis_tvalid <= aresetn && vsh[1];
    end

    // ---- NN_live input select (combinational)
    assign m_nn_tdata  = sel ? m_box_axis_tdata  : s_fir_tdata;
    assign m_nn_tvalid = sel ? m_box_axis_tvalid : s_fir_tvalid;

endmodule
