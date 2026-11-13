// boxcar_tap with one deliberate fault: term 2 of the moving sum (l1i[1]) is masked by mk[3] instead of mk[2],
// so I is wrong for M = 3. Used to confirm that tb_box_diff detects a wrong mask. Otherwise identical to
// neural_net4_live576/rtl/boxcar_tap.v.

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

    // ---- configuration
    reg [31:0] cfg_r, cfg_rr;
    (* max_fanout = 8 *)  reg       sel;
    (* max_fanout = 16 *) reg [15:0] mk;       // mk[j] = (j < M)
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

    // ---- stages 1-3: box8, adder tree over the 8 samples (I and Q separately)
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

    // ---- stages A-C: acc(t+3) = sum_{j<M} b8(t-j), a masked 16-term adder tree without feedback, so a
    //   corrupted register affects at most M + 3 outputs.
    //   sr[j] = b8 from j+1 cycles ago. Terms j = 0, 1 go through h1/h2; terms j = 2..15 enter the tree
    //   pairwise, masked by j < M:
    //     L1 (pairs) -> L2 -> L3 -> G,  G(t+2) = sum_{2<=j<M} b8(t-j)
    //     acc(t+3) = h2(t+2) + G(t+2),  h2(t+2) = h1(t+1) = b8(t) + [M>1] b8(t-1)
    //   23-bit acc holds 16*8*(-32768) exactly.
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
            l1i[1] <= (mk[3] ? b8i : 19'sd0) + (mk[3] ? sri[0] : 19'sd0);
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

    // ---- stages D-F: round, shift, saturate
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

    // ---- NN_live input select
    assign m_nn_tdata  = sel ? m_box_axis_tdata  : s_fir_tdata;
    assign m_nn_tvalid = sel ? m_box_axis_tvalid : s_fir_tvalid;

endmodule
