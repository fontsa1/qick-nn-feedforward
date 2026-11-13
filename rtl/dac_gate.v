// dac_gate: passes or blanks generator 0's DAC stream according to NN_live's decision for the shot.
//
// Sits between generator 0's output pipeline and the RFdc's s00_axis, on clk_dac2 (491.52 MHz, 16 x 16-bit
// samples per cycle). The tProc schedules the conditional pulse unconditionally and raises `marker` with it.
// A window opens DELAY cycles after the marker rises (to line up with the generator pipeline) and lasts as
// long as the marker. Inside it the stream is passed or zeroed, so the decision acts on the DAC without a
// tProc read and branch.
//   * Outside windows the stream passes with a fixed 1-cycle delay.
//   * The policy is latched when the window opens and held for the whole window:
//       fresh decision, bit 1 -> pass if PASS1;  bit 0 -> pass if PASS0;  none fresh -> pass if PASS_STALE.
//     With EARLY_BLANK the policy covers the window's first two words too (computed from registered state the
//     cycle before; a decision arriving in that cycle blanks them). The blanked span ends one cycle after the
//     window.
//   * Shots: a rising cls_trig (the trigger that starts NN_live's window) starts a shot. A decision is fresh
//     if it arrives after this shot's cls_trig and before its window opens. One arriving after the window
//     opened is late: counted and discarded.
//   * SHOT_BIND: NN_live numbers its results (shot field [60:32]). The field expected for this shot is the
//     latest received field plus the cls_trig rises since it; a decision with another field is counted
//     (word 19) and not used. After a shot whose decision never arrives, the next shot is also rejected.
//     After reset the gate expects field 0.
//   * FAIL_SAFE: 0 = windows pass while ENABLE = 0; 1 = windows blank while ENABLE = 0.
//   * CFG_SAFE: 1 = word 0 is taken only when two consecutive reads agree and no marker or window is in
//     flight (one cycle later than with 0).
//   * Statistics are kept whether or not the gate is enabled.
//
// dec_axis: NN_live's 64-bit result in clk_dac2 (from live_res_cdc): [31:0] logit (Q.10, sign-extended),
// [60:32] shot, [63] decision.
//
// BRAM (32-bit words; PS side via axi_bram_ctrl_gate):
//   word 0  config, written by the host: [0] ENABLE [1] PASS1 [2] PASS0 [3] PASS_STALE
//           [15:8] DELAY (cycles, marker -> window) [31:16] NZ_THR
//   words 1-20, written at each window's end:
//   1  windows          2  passed          3  blanked          4  stale          5  decisions received
//   6  window open - decision arrival, cycles (signed; 0x7FFFFFFF if stale)
//   7  marker rise -> first sample with |x| > NZ_THR at the gate input, cycles (0xFFFFFFFF if none),
//      excluding the marker synchronizer's 2 cycles
//   8  decision [31:0]     9  decision [63:32]
//   10 flags: [0] passed [1] stale [2] decision bit [3] enabled [4] a decision with the wrong shot field
//      arrived in this shot; build parameters [8] EARLY_BLANK [9] SHOT_BIND [10] FAIL_SAFE [11] CFG_SAFE
//   11 magic 0x47415434 ("GAT4")
//   12 late decisions, total               13 last late decision: arrival - window open, cycles
//   14 cls_trig rises
//   15-18 cycle-counter values (mod 2^32) of: the fresh decision's arrival (0xFFFFFFFF if stale), the marker
//      rise, the window opening (the gated word reaches the RFdc one cycle later), the cls_trig rise
//   19 decisions rejected by the shot binding, total
//   20 shot field this shot's decision had to carry
`timescale 1ns / 1ps

module dac_gate #(
    parameter integer NS          = 16,        // samples per stream word
    parameter integer EARLY_BLANK = 1,         // policy applies from the window's first cycle (0: from the third)
    parameter integer SHOT_BIND   = 1,         // use a decision only if its shot field is this shot's
    parameter integer FAIL_SAFE   = 0,         // 1 = windows blank while ENABLE = 0 (0: pass)
    parameter integer CFG_SAFE    = 0          // 1 = word 0 double-read and only taken between windows
) (
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 aclk CLK" *)
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF s_axis:m_axis:dec_axis, ASSOCIATED_RESET aresetn" *)
    input  wire                aclk,
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 aresetn RST" *)
    (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input  wire                aresetn,

    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 s_axis TDATA" *)
    input  wire [16*NS-1:0]    s_axis_tdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 s_axis TVALID" *)
    input  wire                s_axis_tvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 s_axis TREADY" *)
    output wire                s_axis_tready,

    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 m_axis TDATA" *)
    output reg  [16*NS-1:0]    m_axis_tdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 m_axis TVALID" *)
    output reg                 m_axis_tvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 m_axis TREADY" *)
    input  wire                m_axis_tready,

    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 dec_axis TDATA" *)
    input  wire [63:0]         dec_axis_tdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 dec_axis TVALID" *)
    input  wire                dec_axis_tvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 dec_axis TREADY" *)
    output wire                dec_axis_tready,

    input  wire                marker,
    input  wire                cls_trig,

    (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_PORT CLK" *)
    (* X_INTERFACE_PARAMETER = "MASTER_TYPE BRAM_CTRL, MEM_SIZE 8192, MEM_WIDTH 32, MEM_ECC NONE, READ_LATENCY 1" *)
    output wire                bram_clk,
    (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_PORT RST" *)
    output wire                bram_rst,
    (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_PORT EN" *)
    output reg                 bram_en,
    (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_PORT WE" *)
    output reg  [3:0]          bram_we,
    (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_PORT ADDR" *)
    output reg  [31:0]         bram_addr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_PORT DIN" *)
    output reg  [31:0]         bram_din,
    (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_PORT DOUT" *)
    input  wire [31:0]         bram_dout
);

    localparam [31:0] MAGIC = 32'h47415434;  // "GAT4"
    localparam [31:0] NONE  = 32'hFFFFFFFF;
    localparam integer N_REC = 20;             // words 1..20
    localparam [2:0]   RD_OK = (CFG_SAFE != 0) ? 3'd6 : 3'd5;
    localparam [3:0]   PARAM_FLAGS = {CFG_SAFE != 0, FAIL_SAFE != 0, SHOT_BIND != 0, EARLY_BLANK != 0};

    assign s_axis_tready   = 1'b1;
    assign dec_axis_tready = 1'b1;
    assign bram_clk = aclk;
    assign bram_rst = ~aresetn;

    // ---- config (BRAM word 0)
    reg  [31:0] cfg, cfg_raw;
    wire        en         = cfg[0];
    wire        pass1      = cfg[1];
    wire        pass0      = cfg[2];
    wire        pass_stale = cfg[3];
    wire [15:0] nz_thr     = cfg[31:16];
    reg  [7:0]  delay_r;                       // registered copy of cfg[15:8]

    reg [31:0] t;                              // free-running cycle counter

    // ---- marker and cls_trig: 2-flop synchronizers (QICK's timing.xdc false-paths the tProc trigger outputs);
    // marker delay line, tapped through a two-stage mux
    (* ASYNC_REG = "TRUE" *) reg [1:0] mk_sync;
    (* ASYNC_REG = "TRUE" *) reg [1:0] cl_sync;
    reg         mk_r, mk_r1, cl_r1;
    reg [255:0] mk_sr;
    reg [15:0]  mk_sel;
    reg         mk_d, mk_d1;
    wire        mk_rise   = mk_r & ~mk_r1;
    wire        cl_rise   = cl_sync[1] & ~cl_r1;
    wire        win_open  = mk_d & ~mk_d1;
    wire        win_close = ~mk_d & mk_d1;
    // mk_v is mk_sr's next value, so the two-stage tap gives mk_d(n) = mk_sr(n-1)[delay]
    wire [255:0] mk_v     = {mk_sr[254:0], mk_r};

    // ---- decision state
    reg [63:0] dec_word;
    reg        dec_fresh;
    // shot binding
    reg [28:0] last_f;                         // shot field of the latest decision received
    reg [3:0]  pend;                           // classify triggers since then (saturating)
    reg [28:0] exp_f, w_exp;                   // field this shot's decision must carry
    reg [31:0] n_mis;
    reg        mis_seen, w_mis;
    wire [28:0] dec_f   = dec_axis_tdata[60:32];
    wire        bind_ok = (SHOT_BIND == 0) || (dec_f == exp_f);
    wire [3:0]  pend_up = (pend == 4'hF) ? 4'hF : pend + 4'd1;
    reg        opened;                         // this shot's window has opened
    reg [31:0] t_dec, n_dec, n_late, late_by, n_cls, t_open;
    reg [31:0] t_cls, w_tdec, w_tmk, w_topen;  // absolute timestamps (words 15-18)

    // ---- window state
    reg        in_win, pass;
    (* max_fanout = 16 *) reg blank;
    reg        blank_e;                        // early blank, the window's first two cycles
    reg [31:0] cfg_raw_q;                      // CFG_SAFE: previous read of word 0
    reg        cfg_eq, quiet;
    reg        w_stale, w_bit, w_pass, w_en;
    reg [31:0] w_margin, t_mk, w_nz;
    reg        nz_seen;
    reg [63:0] w_dec;
    reg [31:0] n_win, n_pass, n_blank, n_stale;

    // ---- |sample| > NZ_THR at the gate input (3-cycle detector)
    reg [15:0]   absv [0:NS-1];
    reg [NS-1:0] nz_bits;
    reg          nz_any;
    integer k;
    always @(posedge aclk) begin
        for (k = 0; k < NS; k = k + 1) begin
            absv[k]    <= s_axis_tdata[16*k+15] ? (~s_axis_tdata[16*k +: 16] + 16'd1)
                                                : s_axis_tdata[16*k +: 16];
            nz_bits[k] <= absv[k] > nz_thr;
        end
        nz_any <= |nz_bits;
    end

    // ---- stream: one register stage; zero inside a blanked window
    always @(posedge aclk) begin
        m_axis_tdata <= blank ? {16*NS{1'b0}} : s_axis_tdata;
        if (!aresetn) m_axis_tvalid <= 1'b0;
        else          m_axis_tvalid <= s_axis_tvalid;
    end

    // ---- BRAM writer (b_* registers, then one more register stage to the port)
    reg        wr_busy;
    reg [4:0]  wr_k;
    reg [31:0] rec [0:N_REC-1];                // words 1..N_REC
    reg [2:0]  rd_cnt;
    reg        b_en;
    reg [3:0]  b_we;
    reg [31:0] b_addr, b_din;

    // ---- early blank. *_nx = next-cycle values, from the D inputs of the registers they shadow
    wire        win_open_nx = mk_sel[delay_r[7:4]] & ~mk_d;           // win_open in the next cycle
    wire        cl_rise_nx  = cl_sync[0] & ~cl_sync[1];               // cl_rise in the next cycle
    wire        cfg_ld      = (CFG_SAFE == 0) ? 1'b1 : (cfg_eq && quiet);
    wire        cfg_take    = !wr_busy && (rd_cnt == RD_OK) && cfg_ld; // cfg <= cfg_raw at this edge
    wire [3:0]  pol_nx      = cfg_take ? cfg_raw[3:0] : cfg[3:0];     // {PASS_STALE, PASS0, PASS1, ENABLE} next cycle
    // Policy of the window opening next cycle, from registered state only. A decision arriving now is still
    // fresh at the window but is not used for its first two words: they blank.
    wire        fresh_nx    = dec_fresh && !cl_rise && !cl_rise_nx && !dec_axis_tvalid;
    wire        pass_nx     = fresh_nx ? (dec_word[63] ? pol_nx[1] : pol_nx[2])
                                       : (pol_nx[3] && !dec_axis_tvalid);
    wire        blank_nx    = (FAIL_SAFE != 0) ? !(pol_nx[0] && pass_nx) : (pol_nx[0] && !pass_nx);
    // blank for the window's first cycle (decided in the cycle before) and its second (held)
    wire        blank_e_d   = (EARLY_BLANK != 0) && ((win_open_nx && blank_nx) || (win_open && blank_e));
    // blank from the window's third cycle to one cycle after it closes
    wire        blank_in_win = in_win && ((FAIL_SAFE != 0) ? !(en && pass) : (en && !pass));

    always @(posedge aclk) begin
        if (!aresetn) begin
            cfg <= 32'd0; cfg_raw <= 32'd0; delay_r <= 8'd0; t <= 0;
            mk_sync <= 0; cl_sync <= 0; mk_r <= 0; mk_r1 <= 0; cl_r1 <= 0;
            mk_sr <= 0; mk_sel <= 0; mk_d <= 0; mk_d1 <= 0;
            dec_word <= 0; dec_fresh <= 0; opened <= 0;
            last_f <= {29{1'b1}}; pend <= 0; exp_f <= 0; w_exp <= 0; n_mis <= 0;
            mis_seen <= 0; w_mis <= 0; blank_e <= 0; cfg_raw_q <= 0; cfg_eq <= 0; quiet <= 0;
            t_dec <= 0; n_dec <= 0; n_late <= 0; late_by <= 0; n_cls <= 0; t_open <= 0;
            t_cls <= 0; w_tdec <= NONE; w_tmk <= NONE; w_topen <= NONE;
            in_win <= 0; pass <= 1'b1; blank <= 1'b0;
            w_stale <= 0; w_bit <= 0; w_pass <= 0; w_en <= 0;
            w_margin <= 0; t_mk <= 0; w_nz <= NONE; nz_seen <= 0; w_dec <= 0;
            n_win <= 0; n_pass <= 0; n_blank <= 0; n_stale <= 0;
            wr_busy <= 0; wr_k <= 0; rd_cnt <= 0;
            b_en <= 0; b_we <= 0; b_addr <= 0; b_din <= 0;
            bram_en <= 0; bram_we <= 0; bram_addr <= 0; bram_din <= 0;
            for (k = 0; k < N_REC; k = k + 1) rec[k] <= 0;
        end else begin
            t <= t + 32'd1;
            delay_r <= cfg[15:8];

            mk_sync <= {mk_sync[0], marker};
            cl_sync <= {cl_sync[0], cls_trig};
            mk_r  <= mk_sync[1];
            mk_r1 <= mk_r;
            cl_r1 <= cl_sync[1];
            mk_sr <= mk_v;
            for (k = 0; k < 16; k = k + 1) mk_sel[k] <= mk_v[16*k + delay_r[3:0]];
            mk_d  <= mk_sel[delay_r[7:4]];
            mk_d1 <= mk_d;

            blank   <= blank_in_win || blank_e_d;
            blank_e <= blank_e_d;

            // ---- shot binding bookkeeping (every result counts, used or not)
            if (dec_axis_tvalid) last_f <= dec_f;
            if (cl_rise) begin
                pend  <= dec_axis_tvalid ? 4'd1 : pend_up;
                exp_f <= dec_axis_tvalid ? dec_f + 29'd1 : last_f + {25'd0, pend_up};
            end else if (dec_axis_tvalid) begin
                pend  <= 4'd0;
            end

            // ---- decisions. A new shot clears everything older; a decision after the window opened is late.
            if (dec_axis_tvalid) n_dec <= n_dec + 32'd1;
            if (cl_rise) begin
                n_cls     <= n_cls + 32'd1;
                t_cls     <= t;
                dec_fresh <= 1'b0;
                opened    <= 1'b0;
                mis_seen  <= 1'b0;
            end else if (dec_axis_tvalid && !win_open) begin
                if (opened) begin
                    n_late  <= n_late + 32'd1;
                    late_by <= t - t_open;
                end else if (!bind_ok) begin
                    n_mis    <= n_mis + 32'd1;
                    mis_seen <= 1'b1;
                end else begin
                    dec_word  <= dec_axis_tdata;
                    dec_fresh <= 1'b1;
                    t_dec     <= t;
                end
            end

            // marker rise: start timing the pulse at the gate input
            if (mk_rise) begin
                t_mk    <= t;
                w_tmk   <= t;
                w_nz    <= NONE;
                nz_seen <= 1'b0;
            end else if (!nz_seen && nz_any && (mk_r || in_win)) begin
                w_nz    <= t - 32'd3 - t_mk;
                nz_seen <= 1'b1;
            end

            if (win_open) begin
                in_win <= 1'b1;
                w_en   <= en;
                n_win  <= n_win + 32'd1;
                t_open <= t;
                w_topen <= t;
                w_exp  <= exp_f;
                w_mis  <= mis_seen;
                // a decision arriving in this cycle is late (the branch above skips it on win_open)
                if (!cl_rise) opened <= 1'b1;
                if (dec_axis_tvalid && !cl_rise) begin
                    n_late  <= n_late + 32'd1;
                    late_by <= 32'd0;
                end
                if (dec_fresh && !cl_rise) begin
                    pass      <= dec_word[63] ? pass1 : pass0;
                    w_pass    <= dec_word[63] ? pass1 : pass0;
                    w_stale   <= 1'b0;
                    w_bit     <= dec_word[63];
                    w_dec     <= dec_word;
                    w_margin  <= t - t_dec;
                    w_tdec    <= t_dec;
                    dec_fresh <= 1'b0;
                end else begin
                    pass      <= pass_stale;
                    w_pass    <= pass_stale;
                    w_stale   <= 1'b1;
                    w_bit     <= 1'b0;
                    w_dec     <= dec_word;
                    w_margin  <= 32'h7FFFFFFF;
                    w_tdec    <= NONE;
                    n_stale   <= n_stale + 32'd1;
                end
            end else if (win_close) begin
                in_win <= 1'b0;
                pass   <= 1'b1;
                if (w_pass) n_pass <= n_pass + 32'd1; else n_blank <= n_blank + 32'd1;
                if (!wr_busy) begin
                    rec[0]  <= n_win;
                    rec[1]  <= n_pass + (w_pass ? 32'd1 : 32'd0);
                    rec[2]  <= n_blank + (w_pass ? 32'd0 : 32'd1);
                    rec[3]  <= n_stale;
                    rec[4]  <= n_dec;
                    rec[5]  <= w_margin;
                    rec[6]  <= w_nz;
                    rec[7]  <= w_dec[31:0];
                    rec[8]  <= w_dec[63:32];
                    rec[9]  <= {20'd0, PARAM_FLAGS, 3'd0, w_mis, w_en, w_bit, w_stale, w_pass};
                    rec[10] <= MAGIC;
                    rec[11] <= n_late;
                    rec[12] <= late_by;
                    rec[13] <= n_cls;
                    rec[14] <= w_tdec;
                    rec[15] <= w_tmk;
                    rec[16] <= w_topen;
                    rec[17] <= t_cls;
                    rec[18] <= n_mis;
                    rec[19] <= {3'd0, w_exp};
                    wr_busy <= 1'b1;
                    wr_k    <= 5'd0;
                end
            end

            // ---- BRAM port: write words 1..N_REC after a window; otherwise read word 0.
            // b_* -> bram_* -> BRAM (READ_LATENCY 1) -> cfg_raw: a read reaches cfg_raw four cycles after it
            // is issued, so word 0 is valid from the fifth consecutive read cycle (rd_cnt == RD_OK).
            b_en <= 1'b1;
            if (wr_busy) begin
                b_we   <= 4'hF;
                rd_cnt <= 3'd0;
                b_addr <= {25'd0, wr_k + 5'd1, 2'b00};
                b_din  <= rec[wr_k];
                if (wr_k == N_REC - 1) wr_busy <= 1'b0;
                wr_k <= wr_k + 5'd1;
            end else begin
                b_we   <= 4'h0;
                b_addr <= 32'd0;
                if (rd_cnt == RD_OK) begin
                    if (cfg_ld) cfg <= cfg_raw;
                end else rd_cnt <= rd_cnt + 3'd1;
            end
            bram_en   <= b_en;
            bram_we   <= b_we;
            bram_addr <= b_addr;
            bram_din  <= b_din;
            cfg_raw   <= bram_dout;
            // CFG_SAFE: cfg_eq = two consecutive reads agree; quiet = no marker in flight, no window open
            cfg_raw_q <= cfg_raw;
            cfg_eq    <= (bram_dout == cfg_raw);
            quiet     <= !(|mk_v) && !(|mk_sel) && !mk_d && !mk_d1 && !in_win && !(|mk_sync);
        end
    end

endmodule
