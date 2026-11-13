// boxcar_tap: earlier boxcar_tap with a recursive moving sum (acc += b8(n) - b8(n-M)). Same ports, output values
// and timing as neural_net4_live576/rtl/boxcar_tap.v, but a corrupted accumulator persists until the next cfg
// change. Reference model for tb_box_diff and tb_box_fault.
//
// Function: box8 = sum of the 8 mixed samples per clock; msum = moving sum of the last M box8 values (M = 1..16);
// out = round(msum / 2^shift), saturated to 16 bits. cfg: [0] NN input select (1 = boxcar), [7:4] shift,
// [12:8] M - 1.

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

    // ---- configuration
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

    // ---- stages 1-3: box8, adder tree over the 8 samples (I and Q separately)
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

    // ---- stages A-C: moving sum of M box8 values
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
