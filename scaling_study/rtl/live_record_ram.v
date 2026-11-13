`timescale 1ns / 1ps
// live_record_ram: NN_live's host-readable record memory (8 x RAMB36, 8192 x 32 bit) with one register stage
// on the write port (Addr/EN/WE/Din), as live_rec_wreg in the deployed design. Port A: NN_live's out_r
// (write only; NN_live never reads it). Port B: the host read side.
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
