`timescale 1ps / 1ps
module tb_cdc_skew;
    parameter integer M_PS = 2034, SKEW_PS = 0, TOG_PS = 0, N = 3000;
    localparam integer S_PS = 1808;
    reg s_aclk = 0, m_aclk = 0, rstn = 0;
    integer ph;
    initial begin ph = 0; forever #(S_PS/2) s_aclk = ~s_aclk; end
    initial begin #(313); forever #(M_PS/2) m_aclk = ~m_aclk; end
    reg [63:0] res = 0; reg vld = 0;
    wire [63:0] q; wire qv;
    live_res_cdc_skew #(.SKEW_PS(SKEW_PS), .TOG_PS(TOG_PS)) dut (.s_aclk(s_aclk), .s_aresetn(rstn), .res_o(res), .res_o_ap_vld(vld),
        .m_aclk(m_aclk), .m_aresetn(rstn), .m_axis_tdata(q), .m_axis_tvalid(qv), .m_axis_tready(1'b1));
    reg [63:0] exp_q [0:N-1];
    integer k = 0, got = 0, bad = 0, r;
    function [63:0] fmt(input [63:0] x); fmt = {x[63], 2'b00, x[44:16], {16{x[15]}}, x[15:0]}; endfunction
    always @(posedge m_aclk) if (qv) begin
        if (q !== exp_q[got]) bad = bad + 1;
        got = got + 1;
    end
    initial begin
        #(50000) rstn = 1; #(20000);
        for (k = 0; k < N; k = k + 1) begin
            r = $urandom % 7; repeat (r) @(posedge s_aclk);
            #($urandom % S_PS);                      // decorrelate phase
            @(posedge s_aclk);
            res <= {$urandom, $urandom}; vld <= 1;
            @(posedge s_aclk); vld <= 0; exp_q[k] = fmt(res);
            repeat (40) @(posedge s_aclk);           // >> 3 dest cycles between results
        end
        repeat (40) @(posedge m_aclk);
        $display("M_PS %0d SKEW_PS %0d TOG_PS %0d: sent %0d received %0d corrupted %0d", M_PS, SKEW_PS, TOG_PS, N, got, bad);
        $finish;
    end
endmodule
