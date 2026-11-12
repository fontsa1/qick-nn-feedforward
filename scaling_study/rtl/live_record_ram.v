`timescale 1ns / 1ps
// live_record_ram -- NN_live's host-readable record memory (8 x RAMB36, 8192 x 32 bit), as in the BD
// (axi_bram_ctrl_live_bram), with the write-port register stage planned for 4.1m: NN_live's out_r
// Addr/EN/WEN/Din go through one register before the RAM, which removes the FSM -> 3 LUT -> 8 RAMB36
// path that fails in v1.41 (42 endpoints). Safe because NN_live never reads out_r (out_r_Dout_A is
// tied 0 in NN_live.v): delaying its writes by one cycle changes nothing it can observe.
// Port B = the host read side (AXI BRAM controller clock in the real design).
module live_record_ram (
    input  wire        clka,
    input  wire [31:0] addra,      // byte address (AXI BRAM convention)
    input  wire        ena,
    input  wire [3:0]  wea,
    input  wire [31:0] dina,
    input  wire        clkb,
    input  wire [12:0] addrb,      // word address
    output reg  [31:0] doutb
);
    reg [12:0] a_r;
    reg        en_r;
    reg [3:0]  we_r;
    reg [31:0] d_r;
    always @(posedge clka) begin
        a_r  <= addra[14:2];
        en_r <= ena;
        we_r <= wea;
        d_r  <= dina;
    end

    (* ram_style = "block" *) reg [31:0] mem [0:8191];
    integer i;
    always @(posedge clka) begin
        if (en_r)
            for (i = 0; i < 4; i = i + 1)
                if (we_r[i]) mem[a_r][8*i +: 8] <= d_r[8*i +: 8];
    end
    always @(posedge clkb) doutb <= mem[addrb];
endmodule
