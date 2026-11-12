// live_window_gate (built copy) vs QICK buffer.sv with adversarial trigger patterns
`timescale 1ps / 1ps
module tb_lwg;
    localparam integer CLK_PS = 1808, N = 576;
    reg clk = 0, rstn = 0; always #(CLK_PS/2) clk = ~clk;
    reg trig = 0; reg [31:0] din = 0; reg dv = 1;
    wire [31:0] g_d; wire g_v, g_l;
    live_window_gate #(.N(N)) g (.aclk(clk), .aresetn(rstn), .trigger(trig), .s_axis_tdata(din), .s_axis_tvalid(dv),
        .m_axis_tdata(g_d), .m_axis_tvalid(g_v), .m_axis_tlast(g_l), .m_axis_tkeep(), .m_axis_tstrb(), .m_axis_tready(1'b1),
        .dbg_trig(), .dbg_first(), .dbg_last(), .dbg_data(), .dbg_dvalid());
    (* ASYNC_REG = "TRUE" *) reg [1:0] bs = 0;
    always @(posedge clk) bs <= rstn ? {bs[0], trig} : 2'b00;
    wire we; wire [9:0] addr; wire [31:0] di;
    buffer #(.N(10), .B(16)) b (.rstn(rstn), .clk(clk), .trigger_i(bs[1]), .din_valid_i(dv), .din_i(din),
        .mem_we_o(we), .mem_addr_o(addr), .mem_di_o(di), .START_REG(1'b1), .ADDR_REG(10'd0), .LEN_REG(10'(N)));
    // gate output is one cycle behind: compare g(t) with buffer(t-1)
    reg we_q = 0; reg [31:0] di_q = 0; reg [9:0] addr_q = 0;
    integer wcnt = 0, cyc = 0, mism = 0, nwin_b = 0, nwin_g = 0, nlast = 0, nwords = 0, bad_last = 0;
    always @(posedge clk) begin
        cyc <= cyc + 1;
        din <= din + 1; dv <= ($urandom % 16) != 0;        // occasional invalid cycles
        we_q <= we; di_q <= di; addr_q <= addr;
        if (rstn) begin
            if (g_v !== we_q || (g_v && g_d !== di_q)) mism = mism + 1;
            if (we && addr == 0) nwin_b = nwin_b + 1;
            if (g_v) nwords = nwords + 1;
            if (g_v) wcnt = wcnt + 1; if (g_l) begin nlast = nlast + 1; if (wcnt != N) bad_last = bad_last + 1; wcnt = 0; end
        end
    end
    integer k, w, gap;
    initial begin
        #(CLK_PS*20) rstn = 1; repeat (2000) @(posedge clk);
        for (k = 0; k < 3000; k = k + 1) begin
            case (k % 5)
                0: begin w = 1; gap = 700 + $urandom % 200; end          // single-cycle trigger, normal spacing
                1: begin w = 1 + $urandom % 1200; gap = $urandom % 50; end  // long trigger, short gap (retrigger mid-window)
                2: begin w = 2; gap = 2 + $urandom % 5; end              // back-to-back pulses
                3: begin w = 5 + $urandom % 30; gap = 560 + $urandom % 40; end  // re-arm right at window end
                default: begin w = 1 + $urandom % 10; gap = $urandom % 1500; end
            endcase
            @(posedge clk); #(37 + $urandom % 1700) trig = 1;        // asynchronous edge
            repeat (w) @(posedge clk); #(11 + $urandom % 1700) trig = 0;
            repeat (gap) @(posedge clk);
        end
        repeat (2000) @(posedge clk);
        $display("windows (buffer) %0d, TLASTs %0d, words %0d (= %0d x N? %0d), word/valid mismatches vs buffer.sv %0d, windows whose TLAST is not on word N: %0d",
                 nwin_b, nlast, nwords, nlast, nwords == nlast * N, mism, bad_last);
        $finish;
    end
endmodule
