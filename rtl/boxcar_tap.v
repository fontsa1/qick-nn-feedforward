// boxcar_tap: low-latency readout path for NN_live
// v1.41 / R.16: non-recursive moving sum, same output bit for bit and cycle for cycle).
//
// axis_dyn_readout_v1_0 mixes the ADC stream down with its DDS (m0_axis, 8
// complex samples per 552.96 MHz clock, sample k in bits [32k+31:32k] as
// {Q[15:0], I[15:0]}) and then decimates it 8x through a 120-tap systolic FIR
// (m1_axis, one sample per clock), which takes ~121 ns (C_LATENCY 67). This
// block builds a second decimated stream from m0 without the FIR:
//   box8  sum of the clock's 8 mixed samples (decimate by 8; 3-level adder tree)
//   msum  moving sum of the last M box8 values, M = 1..16 (cfg; a masked adder
//         tree since v1.41, so it has no state that outlives M cycles). The readout
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

    integer k;

    // ---- configuration (registered; replicated select)
    reg [31:0] cfg_r, cfg_rr;
    (* max_fanout = 8 *)  reg       sel;
    (* max_fanout = 16 *) reg [15:0] mk;       // mk[j] = (j < M) (v1.41; was msel = M - 1)
    reg [3:0] shift;
    reg       clr;                             // cfg changed: clear the moving sum
    always @(posedge aclk) begin
        if (!aresetn) begin
            cfg_r <= 0; cfg_rr <= 0; sel <= 0; mk <= 16'h0001; shift <= 0; clr <= 1'b1;
        end else begin
            cfg_r  <= cfg;
            cfg_rr <= cfg_r;
            sel    <= cfg_rr[0];
            for (k = 0; k < 16; k = k + 1) mk[k] <= (k <= cfg_rr[11:8]);
            shift  <= cfg_rr[7:4];
            clr    <= (cfg_rr[12:0] != cfg_r[12:0]);
        end
    end

    // ---- stage 1-3: adder tree over the 8 lanes (I and Q separately)
    reg signed [16:0] s1i [0:3], s1q [0:3];
    reg signed [17:0] s2i [0:1], s2q [0:1];
    reg signed [18:0] b8i, b8q;
    reg [2:0] v13;
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

    // ---- stage A-C: moving sum of M box8 values (v1.41: non-recursive)
    //   acc(t+3) = sum_{j<M} b8(t-j), the same value and the same cycle as the
    //   v1.40L7 recursive form (acc += b8(n) - b8(n-M)), but computed as a masked
    //   16-term adder tree with no feedback: a corrupted capture anywhere in it
    //   (sr, h, the tree, acc) affects at most M + 3 outputs and then washes out,
    //   instead of leaving a permanent offset until the next cfg change (D5).
    //   sr[j] holds box8 from j+1 cycles ago (relative to b8 now). Terms b8(t),
    //   b8(t-1) go through h1/h2; terms b8(t-2..t-15) enter the tree two cycles
    //   earlier (they already exist), pairwise, masked by j < M:
    //     L1 (pairs) -> L2 -> L3 -> G  (G(t+2) = sum_{2<=j<M} b8(t-j))
    //     acc(t+3) = h2(t+2) + G(t+2),  h2(t+2) = h1(t+1) = b8(t) + [M>1] b8(t-1)
    //   Bit growth: 20/21/22/23 bits; 23-bit acc holds 16*8*(-32768) exactly.
    reg signed [18:0] sri [0:12], srq [0:12];
    reg signed [19:0] h1_i, h1_q, h2_i, h2_q;
    reg signed [19:0] l1i [1:7], l1q [1:7];   // pair j = (2i, 2i+1)
    reg signed [20:0] l2i [0:3], l2q [0:3];
    reg signed [21:0] l3i [0:1], l3q [0:1];
    reg signed [22:0] g_i, g_q;
    reg signed [22:0] acc_i, acc_q;
    reg [2:0] vac;
    always @(posedge aclk) begin
        if (clr) begin
            for (k = 0; k < 13; k = k + 1) begin sri[k] <= 0; srq[k] <= 0; end
            h1_i <= 0; h1_q <= 0; h2_i <= 0; h2_q <= 0;
            for (k = 1; k < 8; k = k + 1) begin l1i[k] <= 0; l1q[k] <= 0; end
            for (k = 0; k < 4; k = k + 1) begin l2i[k] <= 0; l2q[k] <= 0; end
            for (k = 0; k < 2; k = k + 1) begin l3i[k] <= 0; l3q[k] <= 0; end
            g_i <= 0; g_q <= 0; acc_i <= 0; acc_q <= 0; vac <= 0;
        end else begin
            sri[0] <= b8i; srq[0] <= b8q;
            for (k = 1; k < 13; k = k + 1) begin sri[k] <= sri[k-1]; srq[k] <= srq[k-1]; end
            // terms 0 and 1
            h1_i <= b8i + (mk[1] ? sri[0] : 19'sd0);
            h1_q <= b8q + (mk[1] ? srq[0] : 19'sd0);
            h2_i <= h1_i;  h2_q <= h1_q;
            // terms 2..15
            // (in the cycle that forms L1 for output t, b8(t-2) is b8 itself and
            // b8(t-j), j >= 3, is sr[j-3])
            l1i[1] <= (mk[2] ? b8i : 19'sd0) + (mk[3] ? sri[0] : 19'sd0);
            l1q[1] <= (mk[2] ? b8q : 19'sd0) + (mk[3] ? srq[0] : 19'sd0);
            for (k = 2; k < 8; k = k + 1) begin
                l1i[k] <= (mk[2*k] ? sri[2*k-3] : 19'sd0) + (mk[2*k+1] ? sri[2*k-2] : 19'sd0);
                l1q[k] <= (mk[2*k] ? srq[2*k-3] : 19'sd0) + (mk[2*k+1] ? srq[2*k-2] : 19'sd0);
            end
            l2i[0] <= l1i[1] + l1i[2];  l2q[0] <= l1q[1] + l1q[2];
            l2i[1] <= l1i[3] + l1i[4];  l2q[1] <= l1q[3] + l1q[4];
            l2i[2] <= l1i[5] + l1i[6];  l2q[2] <= l1q[5] + l1q[6];
            l2i[3] <= l1i[7];           l2q[3] <= l1q[7];
            l3i[0] <= l2i[0] + l2i[1];  l3q[0] <= l2q[0] + l2q[1];
            l3i[1] <= l2i[2] + l2i[3];  l3q[1] <= l2q[2] + l2q[3];
            g_i    <= l3i[0] + l3i[1];  g_q    <= l3q[0] + l3q[1];
            acc_i  <= h2_i + g_i;
            acc_q  <= h2_q + g_q;
            vac    <= {vac[1:0], v13[2]};
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
