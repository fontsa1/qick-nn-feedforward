// seg_stamp: copy of neural_net4_live576/rtl/seg_stamp.v used by tb_seg. Same function; it also declares an
// unused register (wr_shot). See that file for the event and BRAM description.

`timescale 1ns / 1ps

module seg_stamp #(
    parameter integer N_REC     = 64,
    parameter integer CLOSE_CYC = 4095,
    parameter integer LOW_N     = 16
) (
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 aclk CLK" *)
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_RESET aresetn" *)
    input  wire        aclk,
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 aresetn RST" *)
    (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input  wire        aresetn,

    input  wire        dbg_trig,
    input  wire        dbg_first,
    input  wire        dbg_last,
    input  wire [31:0] dbg_data,
    input  wire        dbg_dvalid,
    input  wire        res_vld,
    input  wire        rf_hit,
    output reg  [31:0] cfg,
    output reg  [15:0] rf_thr,

    (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_PORT CLK" *)
    (* X_INTERFACE_PARAMETER = "MASTER_TYPE BRAM_CTRL, MEM_SIZE 8192, MEM_WIDTH 32, MEM_ECC NONE, READ_LATENCY 1" *)
    output wire        bram_clk,
    (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_PORT RST" *)
    output wire        bram_rst,
    (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_PORT EN" *)
    output reg         bram_en,
    (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_PORT WE" *)
    output reg  [3:0]  bram_we,
    (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_PORT ADDR" *)
    output reg  [31:0] bram_addr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_PORT DIN" *)
    output reg  [31:0] bram_din,
    (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_PORT DOUT" *)
    input  wire [31:0] bram_dout
);

    localparam [31:0] NONE  = 32'hFFFFFFFF;
    localparam [31:0] MAGIC = 32'h53454733;

    assign bram_clk = aclk;
    assign bram_rst = ~aresetn;


    // input registers
    reg        i_trig = 1'b0, i_first = 1'b0, i_last = 1'b0, i_dvalid = 1'b0, i_res = 1'b0, i_rf = 1'b0;
    reg [31:0] i_data = 32'd0;
    always @(posedge aclk) begin
        i_trig <= dbg_trig;  i_first <= dbg_first; i_last <= dbg_last;
        i_data <= dbg_data;  i_dvalid <= dbg_dvalid;
        i_res  <= res_vld;   i_rf <= rf_hit;
    end

    reg [31:0] t;                      // free-running counter
    reg        trig_d;
    wire       trig_rise = i_trig && !trig_d;

    // |I| + |Q| of the registered stream sample, one more register stage
    wire signed [15:0] si = i_data[15:0];
    wire signed [15:0] sq = i_data[31:16];
    wire [15:0] ai = si[15] ? (~si + 16'd1) : si;
    wire [15:0] aq = sq[15] ? (~sq + 16'd1) : sq;
    reg  [16:0] mag;
    reg         mag_v;
    reg  [31:0] t_mag;                 // counter value of the sample in mag
    reg  [31:0] thr;                   // from BRAM word 1

    // shot state
    reg        active;                 // between trig and the record write
    reg [31:0] shot, t_trig;
    reg [31:0] d_first, d_last, d_res, d_e1, d_e2, d_e0, d_e3;
    reg        rf_d;                   // i_rf, one cycle back (rising-edge detect)
    reg        e3_arm;                 // LOW_N consecutive cycles without i_rf after last
    reg [7:0]  rf_low;
    reg [7:0]  seen;
    reg [15:0] nwords;
    reg [12:0] close_cnt;
    reg        in_win;
    reg        e2_arm;                 // LOW_N consecutive samples below thr seen after last
    reg [7:0]  low_cnt;
    reg        post_last;              // the sample in mag came after last (t_mag > last)

    // writer: the 16-word record (10 fields, zero padded), then the record count (word 0), then the magic (word 2)
    reg        wr_busy;
    reg [4:0]  wr_k;
    reg [31:0] wr_shot, wr_rec_base;
    reg [31:0] rec [0:9];
    reg [31:0] n_written;
    reg [3:0]  rd_cnt;                 // read phase: words 1 (thr), 3 (cfg), 4 (rf_thr), 1

    integer i;

    always @(posedge aclk) begin
        if (!aresetn) begin
            t <= 0; trig_d <= 0; mag <= 0; mag_v <= 0; t_mag <= 0; thr <= 32'd1000;
            active <= 0; shot <= 0; t_trig <= 0;
            d_first <= NONE; d_last <= NONE; d_res <= NONE; d_e1 <= NONE; d_e2 <= NONE;
            d_e0 <= NONE; d_e3 <= NONE; rf_d <= 0; e3_arm <= 0; rf_low <= 0;
            seen <= 0; nwords <= 0; close_cnt <= 0; in_win <= 0; e2_arm <= 0; low_cnt <= 0; post_last <= 0;
            wr_busy <= 0; wr_k <= 0; wr_shot <= 0; wr_rec_base <= 0; n_written <= 0;
            bram_en <= 0; bram_we <= 0; bram_addr <= 0; bram_din <= 0; rd_cnt <= 0; cfg <= 0; rf_thr <= 0;
            for (i = 0; i < 10; i = i + 1) rec[i] <= 0;
        end else begin
            t      <= t + 32'd1;
            trig_d <= i_trig;
            mag    <= {1'b0, ai} + {1'b0, aq};
            mag_v  <= i_dvalid;
            t_mag  <= t;
            rf_d   <= i_rf;

            // ---- close the current shot: on timeout, or when the next trigger comes
            if (active && (trig_rise || (seen[1] && close_cnt == CLOSE_CYC[12:0])) && !wr_busy) begin
                rec[0] <= shot;     rec[1] <= t_trig;
                rec[2] <= d_first;  rec[3] <= d_last;
                rec[4] <= d_res;    rec[5] <= d_e1;
                rec[6] <= d_e2;     rec[7] <= {8'd0, nwords, seen};
                rec[8] <= d_e0;     rec[9] <= d_e3;
                wr_busy     <= 1'b1;
                wr_k        <= 5'd0;
                wr_rec_base <= 32'd64 + (((shot - 32'd1) % N_REC) << 6);   // byte address of word 16 + 16k
                active      <= 1'b0;
            end

            // ---- start a shot (a trigger that also closed the previous one starts the next)
            if (trig_rise) begin
                active  <= 1'b1;
                shot    <= shot + 32'd1;
                t_trig  <= t;
                d_first <= NONE; d_last <= NONE; d_res <= NONE; d_e1 <= NONE; d_e2 <= NONE;
                d_e0    <= NONE; d_e3 <= NONE; e3_arm <= 0; rf_low <= 0;
                seen    <= 0;
                nwords  <= 0;
                close_cnt <= 0;
                in_win  <= 0;
                e2_arm  <= 0;
                low_cnt <= 0;
                post_last <= 0;
            end else if (active) begin
                if (i_first && !seen[0]) begin d_first <= t - t_trig; seen[0] <= 1'b1; in_win <= 1'b1; nwords <= 16'd1; end
                if (in_win && i_dvalid) nwords <= nwords + 16'd1;
                if (i_last && !seen[1])  begin d_last  <= t - t_trig; seen[1] <= 1'b1; in_win <= 1'b0; end
                if (i_res  && !seen[2])  begin d_res   <= t - t_trig; seen[2] <= 1'b1; end
                if (mag_v && mag >= thr[16:0] && !seen[3]) begin d_e1 <= t_mag - t_trig; seen[3] <= 1'b1; end
                // seen[1] is set the cycle after last (t = T), when mag holds the sample of t_mag = T; the first
                // later sample is in mag one cycle after, when post_last turns on
                post_last <= seen[1];
                if (mag_v && post_last) begin
                    if (mag >= thr[16:0]) low_cnt <= 8'd0;
                    else if (low_cnt != 8'hFF) low_cnt <= low_cnt + 8'd1;
                    if (mag < thr[16:0] && low_cnt >= LOW_N - 1) e2_arm <= 1'b1;
                end
                if (mag_v && mag >= thr[16:0] && post_last && e2_arm && !seen[4])
                    begin d_e2 <= t_mag - t_trig; seen[4] <= 1'b1; end
                // i_rf (tile output): edge0 = first hit of the shot; edge3 = first
                // rising hit after last, once LOW_N hit-free cycles have passed
                if (i_rf && !seen[5]) begin d_e0 <= t - t_trig; seen[5] <= 1'b1; end
                if (post_last) begin
                    if (i_rf) rf_low <= 8'd0;
                    else if (rf_low != 8'hFF) rf_low <= rf_low + 8'd1;
                    if (!i_rf && rf_low >= LOW_N - 1) e3_arm <= 1'b1;
                end
                if (i_rf && !rf_d && post_last && e3_arm && !seen[6])
                    begin d_e3 <= t - t_trig; seen[6] <= 1'b1; end
                if (seen[1] && close_cnt != CLOSE_CYC[12:0]) close_cnt <= close_cnt + 13'd1;
            end

            // ---- BRAM port: write the record, then header + magic; otherwise read thr
            bram_en <= 1'b1;
            if (wr_busy) begin
                bram_we <= 4'hF;
                rd_cnt  <= 4'd0;
                if (wr_k < 5'd16) begin
                    bram_addr <= wr_rec_base + {wr_k, 2'b00};
                    bram_din  <= (wr_k < 5'd10) ? rec[wr_k[3:0]] : 32'd0;
                    wr_k      <= wr_k + 5'd1;
                end else if (wr_k == 5'd16) begin
                    bram_addr <= 32'd0;
                    bram_din  <= n_written + 32'd1;
                    n_written <= n_written + 32'd1;
                    wr_k      <= 5'd17;
                end else begin
                    bram_addr <= 32'd8;
                    bram_din  <= MAGIC;
                    wr_busy   <= 1'b0;
                end
            end else begin
                bram_we   <= 4'h0;
                // phases 0-3 address word 1 (thr), 4-7 word 3 (cfg), 8-11 word 4 (rf_thr), 12-15 word 1.
                // dout shows the word addressed at phase r from phase r + 2, so each is taken at its group's
                // last phase
                bram_addr <= (rd_cnt[3:2] == 2'd1) ? 32'd12 : (rd_cnt[3:2] == 2'd2) ? 32'd16 : 32'd4;
                rd_cnt    <= rd_cnt + 4'd1;
                if (rd_cnt == 4'd3 || rd_cnt == 4'd15) thr <= bram_dout;
                if (rd_cnt == 4'd7)  cfg <= bram_dout;
                if (rd_cnt == 4'd11) rf_thr <= bram_dout[15:0];
            end
        end
    end

endmodule
