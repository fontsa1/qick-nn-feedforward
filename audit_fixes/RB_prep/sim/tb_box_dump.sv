`timescale 1ps / 1ps
module tb_box_dump;
    localparam integer CLK_PS = 1808, SEGLEN = 1500, NSEG = 24;
    reg aclk = 0, aresetn = 0;
    always #(CLK_PS/2) aclk = ~aclk;
    reg [255:0] s_tdata = 0; reg [31:0] cfg = 0;
    wire [31:0] box; wire bv, nnv; wire [31:0] nn; wire [255:0] mr; wire mrv, str;
    boxcar_tap dut (.aclk(aclk), .aresetn(aresetn), .s_axis_tdata(s_tdata), .s_axis_tvalid(1'b1), .s_axis_tready(str),
        .m_mr_axis_tdata(mr), .m_mr_axis_tvalid(mrv), .m_mr_axis_tready(1'b1), .s_fir_tdata(32'h0), .s_fir_tvalid(1'b1),
        .m_nn_tdata(nn), .m_nn_tvalid(nnv), .m_box_axis_tdata(box), .m_box_axis_tvalid(bv), .m_box_axis_tready(1'b1), .cfg(cfg));
    integer fd, k, mode = 0, cyc = 0, s, M, sh;
    reg [15:0] v;
    always @(posedge aclk) begin
        for (k = 0; k < 16; k = k + 1) begin
            case (mode)
              0: v = $urandom;                                  // full-range uniform
              1: v = 16'h8000;                                  // all -32768
              2: v = 16'h7FFF;                                  // all +32767
              3: v = ($urandom % 2) ? 16'h8000 : 16'h7FFF;      // alternating extremes
              default: v = $signed($urandom % 801) - 400;       // realistic small
            endcase
            s_tdata[16*k +: 16] <= v;
        end
        $fdisplay(fd, "%0d %0d %h %h %0d %h", cyc, aresetn, s_tdata, box, bv, cfg);
        cyc <= cyc + 1;
    end
    initial begin
        fd = $fopen("box_dump.txt", "w");
        #(CLK_PS*10) aresetn = 1;
        for (s = 0; s < NSEG; s = s + 1) begin
            @(negedge aclk);
            if (s == 0) begin M = 11; sh = 7; mode = 4; end
            else if (s == 1) begin M = 11; sh = 7; mode = 0; end
            else if (s < 6) begin M = 16; sh = s - 2; mode = s - 1; end   // modes 1..4, shift 0..3
            else begin M = 1 + ($urandom % 16); sh = $urandom % 16; mode = $urandom % 5; end
            cfg = {19'd0, 5'(M - 1), 4'(sh), 3'd0, 1'b1};
            repeat (SEGLEN) @(posedge aclk);
        end
        $fclose(fd); $finish;
    end
endmodule
