/*///////////////////////////////////////////////////////////////////////////////
    Module: CHIP.v
    Creator: Yi-Kai Wu
    Last editor: Yi-Kai Wu
    Last edited date: 2024/06/09
    Discription: RISC-V CPU, Final Project of CA 2024.  
///////////////////////////////////////////////////////////////////////////////*/
//----------------------------- DO NOT MODIFY THE I/O INTERFACE!! ------------------------------//
module CHIP #(                                                                                  //
    parameter BIT_W = 32                                                                        //
)(                                                                                              //
    // clock                                                                                    //
        input               i_clk,                                                              //
        input               i_rst_n,                                                            //
    // instruction memory                                                                       //
        input  [BIT_W-1:0]  i_IMEM_data,                                                        //
        output [BIT_W-1:0]  o_IMEM_addr,                                                        //
        output              o_IMEM_cen,                                                         //
    // data memory                                                                              //
        input               i_DMEM_stall,                                                       //
        input  [BIT_W-1:0]  i_DMEM_rdata,                                                       //
        output              o_DMEM_cen,                                                         //
        output              o_DMEM_wen,                                                         //
        output [BIT_W-1:0]  o_DMEM_addr,                                                        //
        output [BIT_W-1:0]  o_DMEM_wdata,                                                       //
    // finnish procedure                                                                        //
        output              o_finish,                                                           //
    // cache                                                                                    //
        input               i_cache_finish,                                                     //
        output              o_proc_finish                                                       //
);                                                                                              //
//----------------------------- DO NOT MODIFY THE I/O INTERFACE!! ------------------------------//

// ------------------------------------------------------------------------------------------------------------------------------------------------------
// Parameters

// ------------------------------------------------------------------------------------------------------------------------------------------------------

    // TODO: any declaration
    parameter           S_TYPE = 7'b0100011;
    parameter           B_TYPE = 7'b1100011;
    parameter           J_TYPE = 7'b1101111;
    parameter           R_TYPE = 7'b0110011;
    parameter           I_TYPE = 7'b0010011;
    parameter           I_TYPE_LOAD = 7'b0000011;
    parameter           I_TYPE_JALR = 7'b1100111;
    parameter           U_TYPE_AUIPC = 7'b0010111;
    parameter           ECALL = 7'b1110011;

// ------------------------------------------------------------------------------------------------------------------------------------------------------
// Wires and Registers
// ------------------------------------------------------------------------------------------------------------------------------------------------------
    
    // TODO: any declaration
        reg [BIT_W-1:0] PC, next_PC;
        wire mem_cen, mem_wen;
        wire [BIT_W-1:0] mem_addr, o_mem_wdata, mem_rdata;
        wire mem_stall;

         //NOT FF
        wire    [BIT_W-1:0]  instr;
        wire    [6:0]       opcode;
        wire    [BIT_W-1:0]  imm;
        reg     [BIT_W-1:0]  imm_gen_output;
        wire                ctrl_jalr, ctrl_jal, ctrl_branch, ctrl_mem_to_reg, ctrl_mem_write, ctrl_alu_src, ctrl_reg_write, ctrl_shamt;
        wire                ctrl_auipc,ctrl_mem_read, ctrl_ecall;
        wire    [BIT_W-1:0]  write_data, rs1_data, rs2_data;
        wire    [BIT_W-1:0]  alu_in_1, alu_in_2, alu_result;
        wire                br_comp;//Result of the branch instruction;=1 when branch condition holds
        wire                br_equal, br_less;
        reg     [3:0]       ctrl_alu;
        wire    [BIT_W-1:0]  read_or_alu_result;//output of Mux2
        wire    [BIT_W-1:0]  pc_plus_4, pc_plus_imm;
        wire    [BIT_W-1:0]  jal_addr, jalr_addr;
        wire    [6:0]       funct7;
        wire    [2:0]       funct3;
        wire                alu_ready;

// ------------------------------------------------------------------------------------------------------------------------------------------------------
// Continuous Assignment
// ------------------------------------------------------------------------------------------------------------------------------------------------------
    // TODO: any wire assignment
    //Instruction break up
    assign instr = i_IMEM_data;
    assign opcode = instr[6:0];
    assign funct3 = instr[14:12];
    assign funct7 = instr[31:25];

    //Connect pc to the address of instruction memory
    assign o_IMEM_addr = PC;

    //Connect immediate to the output of immediate gernerator
    assign imm = imm_gen_output;

    //Main control
    assign ctrl_jalr = (opcode == I_TYPE_JALR) ? 1'b1 : 1'b0;
    assign ctrl_jal = (opcode == J_TYPE) ? 1'b1 : 1'b0;

    assign ctrl_shamt = ((opcode == I_TYPE) && ((funct3 == 3'b001) || (funct3 == 3'b101))) ? 1'b1 : 1'b0;
        //SLLI: funct3 = 001; SRLI,SRAI: funct3 = 101  
        //When I_type_shift(SLLI, SRAI)
        //alu_input_2 = shamt = instr[24:20];

    assign ctrl_alu_src = ( (opcode == I_TYPE_LOAD)||(opcode == I_TYPE)||(opcode == S_TYPE)||(opcode == U_TYPE_AUIPC) ) ? 1'b1 : 1'b0;
        //When not I_type_shift
        // ctrl_alu_src = 1, alu_input_2 = imm.;
        //else ctrl_alu_src =0,alu_input_2 = rs2

    assign ctrl_auipc = (opcode == U_TYPE_AUIPC) ? 1'b1 : 1'b0;
        //When auipc, write_data = PC + imm.

    assign ctrl_mem_to_reg = ( opcode == I_TYPE_LOAD ) ? 1'b1 : 1'b0;
        //=1 when load, means read_or_alu_result = mem_read_data;
        // else =0, means read_or_alu_result = alu_result

    assign ctrl_reg_write = (( (opcode == S_TYPE) || (opcode == B_TYPE) ) || i_DMEM_stall) ? 1'b0 : 1'b1;
        //=0 when sw,op = instr[6:0] = 0100011; or beq, op = instr[6:0] = 1100011, means reg_file write-enable = 0
        //

    assign ctrl_mem_write = ( opcode == S_TYPE) ? 1'b1 : 1'b0;
        //=1 when sw,op = instr[6:0] = 0100011, write to Dmem; else = 0, read from Dmem

    assign ctrl_mem_read = ( opcode == I_TYPE_LOAD) ? 1'b1 : 1'b0;
        //=1 when lw

    assign ctrl_branch = ( opcode == B_TYPE) ? 1'b1 : 1'b0;
        //=1 when branch, op = instr[6:0] = 1100011

    //Connect alu inputs to the rs1 and output of MUX6, MUX1
    assign alu_in_1 = rs1_data;
    assign alu_in_2 = (ctrl_shamt) ? {27'b0, instr[24:20]}
                    :(ctrl_alu_src) ? imm : rs2_data;

    //Connect D_cashe_data and alu output to MUX2
    assign read_or_alu_result = (ctrl_mem_to_reg) ? i_DMEM_rdata: alu_result;

    //Connect (pc +4) and read_or_alu_result to MUX3, connect output to MUX7
    //, connect output of MUX7 to register file to rd
    assign write_data = (ctrl_auipc) ? pc_plus_imm
                        : (ctrl_jal ||ctrl_jalr) ? pc_plus_4 : read_or_alu_result;

    //Connect to Dmem
    //(cen, wen): Hold:(0,x), Read:(1,0), Write:(1,1)
    assign o_DMEM_cen = (ctrl_mem_write || ctrl_mem_read) ; 
    //assign o_DMEM_cen = 1'b1;
    assign o_DMEM_wen = ctrl_mem_write;
    assign o_DMEM_addr = alu_result;
    assign o_DMEM_wdata = rs2_data;


    /*Following cope with PC*/
    assign pc_plus_4 = PC +4;
    assign pc_plus_imm = PC + imm;

    //: branch controller (br_less, br_equal, opcode) -> br_comp
    //BEQ: 000, BNE: 001, BLT: 100, BGE: 101
    assign br_comp = ( 
        (funct3 == 3'b000 && br_equal) || 
        (funct3 == 3'b001 && ~br_equal) || 
        (funct3 == 3'b100 && br_less) || 
        (funct3 == 3'b101 && ~br_less) 
    );
    
    //MUX4
    assign jal_addr = (ctrl_jal||(ctrl_branch && br_comp)) ? pc_plus_imm : pc_plus_4;
    assign jalr_addr = rs1_data + imm;
    //MUX5 
    always @(*) begin
        next_PC = (i_DMEM_stall || !alu_ready) ? PC 
                : (ctrl_jalr) ? jalr_addr : jal_addr;
    end

    assign o_IMEM_cen = 1'b1;

    //Deal with ecall
    assign ctrl_ecall = ({funct3, opcode}== {3'b000, ECALL}) ? 1'b1 : 1'b0;
    //To-do:
    //Ask cache to store all values back to memory. When received cache store_finish, pull up o_finish
    assign o_proc_finish = ctrl_ecall;
    //Assume no cache:
    assign o_finish = (ctrl_ecall && i_cache_finish) ? 1'b1 : 1'b0;
    

// ------------------------------------------------------------------------------------------------------------------------------------------------------
// Submoddules
// ------------------------------------------------------------------------------------------------------------------------------------------------------

    // TODO: Reg_file wire connection
    Reg_file reg0(               
        .i_clk  (i_clk),             
        .i_rst_n(i_rst_n),         
        .wen    (ctrl_reg_write),          
        .rs1    (instr[19:15]),                
        .rs2    (instr[24:20]),                
        .rd     (instr[11:7]),                 
        .wdata  (write_data),             
        .rdata1 (rs1_data),           
        .rdata2 (rs2_data)
    );
    alu #(.BIT_W(BIT_W)) alu_U0(
        .ctrl(ctrl_alu),
        .x(alu_in_1),
        .y(alu_in_2),
        .i_clk(i_clk),
        .br_equal(br_equal),
        .br_less(br_less),
        .out(alu_result),
        .o_ready(alu_ready)
    );


// ------------------------------------------------------------------------------------------------------------------------------------------------------
// Always Blocks
// ------------------------------------------------------------------------------------------------------------------------------------------------------
    
    // Todo: any combinational/sequential circuit
    /* ====================Combinational Part================== */
    //Immediate Generator
    always @(*) begin
        case(opcode)
            I_TYPE, I_TYPE_JALR, I_TYPE_LOAD : begin
                imm_gen_output = {{21{instr[31]}}, instr[30:25], instr[24:21], instr[20]};
            end
            S_TYPE : begin
                imm_gen_output = {{21{instr[31]}}, instr[30:25], instr[11:8], instr[7]};
            end
            B_TYPE : begin
                imm_gen_output = {{20{instr[31]}}, instr[7], instr[30:25], instr[11:8], 1'b0};
            end
            J_TYPE : begin
                imm_gen_output = {{12{instr[31]}}, instr[19:12], instr[20],instr[30:25], instr[24:21], 1'b0};
            end
            U_TYPE_AUIPC : begin
                imm_gen_output = {instr[31:12], 12'b0};
            end
            default : imm_gen_output = 0;  
        endcase
    end
   
    //Alu control: Tofix
    always @(*) begin
        case(opcode)
            R_TYPE : begin
                case ({funct7, funct3})
                    {7'b0000000, 3'b111} : ctrl_alu = 4'b0000;//AND
                    {7'b0000000, 3'b110} : ctrl_alu = 4'b0001;//OR
                    {7'b0000000, 3'b000} : ctrl_alu = 4'b0010;//ADD
                    {7'b0100000, 3'b000} : ctrl_alu = 4'b0110;//SUB
                    {7'b0000000, 3'b100} : ctrl_alu = 4'b0100;//XOR
                    {7'b0000001, 3'b000} : ctrl_alu = 4'b0111;//MUL
                    default : ctrl_alu = 4'b0010;//ADD
                endcase
            end
            B_TYPE : ctrl_alu = 4'b0110;//SUB
            I_TYPE : begin
                case (funct3)
                    {3'b000} : ctrl_alu = 4'b0010;//ADD
                    {3'b001} : ctrl_alu = 4'b0101;//SLL
                    {3'b101} : ctrl_alu = 4'b0011;//SRA
                    {3'b010} : ctrl_alu = 4'b1000;//SLT
                    default : ctrl_alu = 4'b0010;//ADD
                endcase
            end
            S_TYPE, I_TYPE_LOAD : ctrl_alu = 4'b0010;//ADD
            default : ctrl_alu = 4'b0010;//ADD
        endcase
    end
    // assign ctrl_alu[0] = (opcode[4] & instr[14] & instr[13] & ~instr[12]);
    // assign ctrl_alu[1] = !(opcode[4] & ~opcode[3] & instr[13]);
    // assign ctrl_alu[2] = (~opcode[4] & ~instr[13] | (instr[30] & opcode[4]));
    // assign ctrl_alu[3] = (opcode[4] & ~instr[14] & instr[13] );
    //next-state logic
			  
    // output logic
    // The output MUX

    /* ====================Sequential Part=================== */
    //Current State Register or Memory Element 

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            PC <= 32'h00010000; // Do not modify this value!!!
        end
        else begin
            PC <= next_PC;
        end
    end
endmodule

module Reg_file(i_clk, i_rst_n, wen, rs1, rs2, rd, wdata, rdata1, rdata2);
   
    parameter BITS = 32;
    parameter word_depth = 32;
    parameter addr_width = 5; // 2^addr_width >= word_depth
    
    input i_clk, i_rst_n, wen; // wen: 0:read | 1:write
    input [BITS-1:0] wdata;
    input [addr_width-1:0] rs1, rs2, rd;

    output [BITS-1:0] rdata1, rdata2;

    reg [BITS-1:0] mem [0:word_depth-1];
    reg [BITS-1:0] mem_nxt [0:word_depth-1];

    integer i;

    assign rdata1 = mem[rs1];
    assign rdata2 = mem[rs2];

    always @(*) begin
        for (i=0; i<word_depth; i=i+1)
            mem_nxt[i] = (wen && (rd == i)) ? wdata : mem[i];
    end

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            mem[0] <= 0;
            for (i=1; i<word_depth; i=i+1) begin
                case(i)
                    32'd2: mem[i] <= 32'hbffffff0;
                    32'd3: mem[i] <= 32'h10008000;
                    default: mem[i] <= 32'h0;
                endcase
            end
        end
        else begin
            mem[0] <= 0;
            for (i=1; i<word_depth; i=i+1)
                mem[i] <= mem_nxt[i];
        end       
    end
endmodule

// module Cache#(
//         parameter BIT_W = 32,
//         parameter ADDR_W = 32
//     )(
//         input i_clk,
//         input i_rst_n,
//         // processor interface
//             input i_proc_cen,
//             input i_proc_wen,
//             input [ADDR_W-1:0] i_proc_addr,
//             input [BIT_W-1:0]  i_proc_wdata,
//             output [BIT_W-1:0] o_proc_rdata,
//             output o_proc_stall,
//             input i_proc_finish,
//             output o_cache_finish,
//         // memory interface
//             output o_mem_cen,
//             output o_mem_wen,
//             output [ADDR_W-1:0] o_mem_addr,
//             output [BIT_W*4-1:0]  o_mem_wdata,
//             input [BIT_W*4-1:0] i_mem_rdata,
//             input i_mem_stall,
//             output o_cache_available,
//         // others
//         input  [ADDR_W-1: 0] i_offset
//     );

//     assign o_cache_available = 0; // change this value to 1 if the cache is implemented

//     //------------------------------------------//
//     //          default connection              //
//     assign o_mem_cen = i_proc_cen;              //
//     assign o_mem_wen = i_proc_wen;              //
//     assign o_mem_addr = i_proc_addr;            //
//     assign o_mem_wdata = i_proc_wdata;          //
//     assign o_proc_rdata = i_mem_rdata[0+:BIT_W];//
//     assign o_proc_stall = i_mem_stall;          //
//     //------------------------------------------//

//     // Todo: BONUS
// endmodule

module Cache#(
        parameter BIT_W = 32,
        parameter ADDR_W = 32
    )(
        input i_clk,
        input i_rst_n,
        // processor interface
            input i_proc_cen,
            input i_proc_wen,
            input [ADDR_W-1:0] i_proc_addr,
            input [BIT_W-1:0]  i_proc_wdata,
            output reg [BIT_W-1:0] o_proc_rdata,
            output reg o_proc_stall,
            input i_proc_finish,
            output reg o_cache_finish,
        // memory interface
            output o_mem_cen,
            output o_mem_wen,
            output [ADDR_W-1:0] o_mem_addr,
            output reg [BIT_W*4-1:0]  o_mem_wdata,
            input [BIT_W*4-1:0] i_mem_rdata,
            input i_mem_stall,
            output o_cache_available,
        // others
        input  [ADDR_W-1: 0] i_offset
    );

    assign o_cache_available = 1; // change this value to 1 if the cache is implemented

    // ------------------------------------------//
    //          default connection              //
    // assign o_mem_cen = i_proc_cen;              //
    // assign o_mem_wen = i_proc_wen;              //
    // assign o_mem_addr = i_proc_addr;            //
    // assign o_mem_wdata = i_proc_wdata;          //
    // assign o_proc_rdata = i_mem_rdata[0+:BIT_W]; //
    // assign o_proc_stall = i_mem_stall;           //
    // ------------------------------------------//
        
    // //==== wire/reg definition ================================
    //     // internal storage
        reg  [  2:0] state_r, state_w;
        reg  [ 31:0] cache_data [0:31], cache_data_w [0:31];
        reg  [ 25:0] cache_tags [0:7 ], cache_tags_w [0:7 ];
        reg          cache_valid[0:7 ], cache_valid_w[0:7 ];
        reg          cache_dirty[0:7 ], cache_dirty_w[0:7 ];
        reg          cache_lru  [0:3 ], cache_lru_w  [0:3 ];
        // processor interface
        wire         miss;
        wire [ 25:0] tag;
        wire [  1:0] set_i;
        wire [  1:0] offset;
        wire [  3:0] word_i;
        wire [29:0] proc_addr;
        //reg          o_proc_stall;
        //reg  [ 31:0] o_proc_rdata;
        wire         ele_i, lru_i;
        // memory interface
        reg  [ 27:0] mem_addr;
        //reg  [127:0] o_mem_wdata;

        wire mem_ready;
        assign mem_ready = ~i_mem_stall;
        wire   proc_read, proc_write;
        assign proc_read = (i_proc_cen && (!i_proc_wen));
        assign proc_write = (i_proc_cen && i_proc_wen);
        reg mem_read, mem_write;
        assign o_mem_cen = (mem_read || mem_write);
        assign o_mem_wen = mem_write;

        integer i;

    //==== combinational circuit ==============================

        localparam S_IDLE    = 3'd0;
        localparam S_READ    = 3'd1;
        localparam S_WRITE   = 3'd2;
        localparam S_FLUSH   = 3'd3;
        localparam S_FINISH  = 3'd4;
        
        wire [31:0] correct_address;
        assign correct_address = i_proc_addr - i_offset;
        assign proc_addr = correct_address[31:2];
        assign {tag, word_i}   = proc_addr;
        assign {set_i, offset} = word_i;
        //assign proc_cen        = proc_write || proc_read;
        assign miss            = (!cache_valid[{1'b0, set_i}] || (cache_tags[{1'b0, set_i}] != tag))
                            && (!cache_valid[{1'b1, set_i}] || (cache_tags[{1'b1, set_i}] != tag));
        assign ele_i           = (cache_valid[{1'b1, set_i}] && cache_tags[{1'b1, set_i}] == tag);
        assign lru_i           = cache_lru[{set_i}];

        reg [2:0] flush_index;
        wire flush_done;
        assign flush_done = (flush_index == 3'd7) && mem_ready;

        always @(*) begin
            case (state_r) // synopsys full_case
                S_IDLE : state_w = (i_proc_cen && miss) ? (cache_dirty[{lru_i, set_i}] ? S_WRITE : S_READ) :
                                   (i_proc_finish ? S_FLUSH : S_IDLE);
                S_READ : state_w = mem_ready ? S_IDLE : S_READ;
                S_WRITE: state_w = mem_ready ? S_READ : S_WRITE;
                S_FLUSH: state_w = (mem_ready && flush_done) ? S_FINISH : (mem_ready ? S_FLUSH : S_FLUSH);
                S_FINISH: state_w = S_IDLE;
            endcase
        end

        always @(*) begin
            for (i = 0; i < 32; i = i + 1)
                cache_data_w[i] = cache_data[i];
            if (state_r == S_IDLE && proc_write && !miss)
                cache_data_w[{ele_i, word_i}] = i_proc_wdata;
            if (state_r == S_READ) begin
                {cache_data_w[{lru_i, set_i, 2'b11}], cache_data_w[{lru_i, set_i, 2'b10}],
                cache_data_w[{lru_i, set_i, 2'b01}], cache_data_w[{lru_i, set_i, 2'b00}]} = i_mem_rdata;
                if (proc_write) cache_data_w[{lru_i, word_i}] = i_proc_wdata;
            end
        end

        always @(*) begin
            for (i = 0; i < 8; i = i + 1) begin
                cache_dirty_w[i] = cache_dirty[i];
                cache_tags_w [i] = cache_tags [i];
            end
            if (state_r == S_IDLE && proc_write) begin
                if (miss) cache_dirty_w[{lru_i, set_i}] = 1'b1;
                else      cache_dirty_w[{ele_i, set_i}] = 1'b1;
            end
            if (state_r == S_READ) begin
                cache_dirty_w[{lru_i, set_i}] = proc_write;
                cache_tags_w [{lru_i, set_i}] = tag;
            end
        end

        always @(*) begin
            for (i = 0; i < 8; i = i + 1)
                cache_valid_w[i] = cache_valid[i] || (state_r == S_READ && i == {lru_i, set_i});
        end

        always @(*) begin
            for (i = 0; i < 4; i = i + 1)
                cache_lru_w[i] = cache_lru[i];
            if (state_r == S_IDLE && i_proc_cen && !miss)
                cache_lru_w[set_i] = !ele_i;
            if (state_r == S_READ && mem_ready)
                cache_lru_w[set_i] = !lru_i;
        end

        always @(*) begin
            case (state_r) // synopsys full_case
                S_IDLE : o_proc_stall = i_proc_cen && miss;
                S_READ : o_proc_stall = i_mem_stall;
                S_WRITE: o_proc_stall = 1'b1;
                S_FLUSH: o_proc_stall = 1'b1;
                S_FINISH: o_proc_stall = 1'b0;
            endcase
        end

        always @(*) begin
            case (state_r) // synopsys full_case
                S_IDLE: o_proc_rdata = cache_data[{ele_i, set_i, offset}];
                S_READ: o_proc_rdata = i_mem_rdata[32 * offset +: 32];
            endcase
        end

        always @(*) begin
            case (state_r) // synopsys full_case
                S_IDLE: begin
                    mem_read  = proc_read  && miss;
                    mem_write = proc_write && miss && cache_dirty[{lru_i, set_i}];
                end
                S_READ: begin
                    mem_read  = 1'b1;
                    mem_write = 1'b0;
                end
                S_WRITE: begin
                    mem_read  = 1'b0;
                    mem_write = 1'b1;
                end
                S_FLUSH: begin
                    mem_read  = 1'b0;
                    mem_write = cache_dirty[flush_index];
                end
                S_FINISH: begin
                    mem_read  = 1'b0;
                    mem_write = 1'b0;
                end
            endcase
        end

        always @(*) begin
            case (state_r) // synopsys full_case
                S_IDLE : mem_addr = cache_dirty[{lru_i, set_i}] ? {cache_tags[{lru_i, set_i}], set_i} : {tag, set_i};
                S_READ : mem_addr = {tag, set_i};
                S_WRITE: mem_addr = {cache_tags[{lru_i, set_i}], set_i};
                S_FLUSH: mem_addr = {cache_tags[flush_index], flush_index[1:0]};
            endcase
        end
        assign o_mem_addr = {mem_addr, 4'b0}+ i_offset;

        always @(*) begin
            o_mem_wdata = {cache_data[{lru_i, set_i, 2'b11}], cache_data[{lru_i, set_i, 2'b10}],
                        cache_data[{lru_i, set_i, 2'b01}], cache_data[{lru_i, set_i, 2'b00}]};
            if (state_r == S_FLUSH)
                o_mem_wdata = {cache_data[{flush_index, 2'b11}], cache_data[{flush_index, 2'b10}],
                             cache_data[{flush_index, 2'b01}], cache_data[{flush_index, 2'b00}]};
        end


    //==== sequential circuit =================================
    always@( posedge i_clk ) begin
        if( !i_rst_n ) begin
            for (i = 0; i < 32; i = i + 1) begin
                cache_data [i] <= 32'b0;
            end
            for (i = 0; i < 8; i = i + 1) begin
                cache_tags [i] <= 26'b0;
                cache_valid[i] <= 1'b0;
                cache_dirty[i] <= 1'b0;
            end
            for (i = 0; i < 4; i = i + 1) begin
                cache_lru  [i] <= 1'b0;
            end
            state_r <= S_IDLE;
            flush_index <= 3'd0;
            o_cache_finish <= 1'b0;
        end
        else begin
            for (i = 0; i < 32; i = i + 1) begin
                cache_data [i] <= cache_data_w [i];
            end
            for (i = 0; i < 8; i = i + 1) begin
                cache_tags [i] <= cache_tags_w [i];
                cache_valid[i] <= cache_valid_w[i];
                cache_dirty[i] <= cache_dirty_w[i];
            end
            for (i = 0; i < 4; i = i + 1) begin
                cache_lru  [i] <= cache_lru_w  [i];
            end
            state_r <= state_w;
            if (state_r == S_FLUSH && mem_ready) begin
                    flush_index <= flush_index + 1;
                end

            if (state_r == S_FINISH) begin
                o_cache_finish <= 1'b1;
            end else if (i_proc_finish) begin
                o_cache_finish <= 1'b0;
            end
        end
    end
endmodule



module alu #(
    parameter BIT_W = 32
)(
    ctrl,
    x,
    y,
    i_clk,
    br_equal,
    br_less,
    out,
    o_ready
);

//Inout port:   
    input  [3:0] ctrl;
    input  i_clk;
    input  [BIT_W-1:0] x;//rs1
    input  [BIT_W-1:0] y;//rs2
    // output reg        br_comp;
    output reg      br_equal, br_less;
    output [BIT_W-1:0] out;
    output o_ready;

    //Parameter
    parameter   BITWISE_AND = 4'b0000, BITWISE_OR = 4'b0001, ADD = 4'b0010, SUB = 4'b0110, SLT = 4'b1000,
                SRA = 4'b0011, XOR = 4'b0100, SLL = 4'b0101, MUL = 4'b0111; 
    
    //wire & reg declaration
    reg            carry;
    reg [BIT_W-1:0] out;
    reg start_calc = 0;
    wire o_ready;
    wire [63:0] out_64;

    MUL_unit muldiv_unit(
        .A(x),
        .B(y),
        .i_clk(i_clk),
        .i_start_calc(start_calc),
        .Y(out_64),
        .o_ready(o_ready)
    );


//CL:
    always@(*)begin
        start_calc = 0;
        case(ctrl)
            ADD: begin
                {carry, out} = {x[BIT_W-1], x} + {y[BIT_W-1],y};
                br_equal = 1'b0;
                br_less = 1'b0;
            end
            SUB: begin
                {carry, out} = {x[BIT_W-1], x} - {y[BIT_W-1],y};
                br_equal = ~(|out);//=1 when all bits are 0; Test is this is smaller than br_comp = (out == 0) ? 1'b1 : 1'b0;
                br_less = carry;//=1 when rs1<rs2, carry(sign bit) ==1
            end
            BITWISE_AND:  begin
                {carry, out} = {1'b0, x & y};
                br_equal = 1'b0;
                br_less = 1'b0;
            end  
            BITWISE_OR:begin
                {carry, out} = {1'b0,  x | y};
                br_equal = 1'b0;
                br_less = 1'b0;
            end     

            SLT:  begin
                carry = 1'b0;
                out = ($signed(x) < $signed(y)) ? {31'b0, 1'b1} : {31'b0, 1'b0};
                br_equal = 1'b0;
                br_less = 1'b0;
            end 
            SRA:  begin
                carry = 1'b0;
                out = $signed(x) >>> y[4:0];
                br_equal = 1'b0;
                br_less = 1'b0;
            end 
            SLL:  begin
                carry = 1'b0;
                out = x << y[4:0];
                br_equal = 1'b0;
                br_less = 1'b0;
            end 
            XOR:  begin
                {carry, out} = {1'b0, x ^ y};
                br_equal = 1'b0;
                br_less = 1'b0;
            end 
            MUL : begin
                {carry, out} = {1'b0, out_64[31:0]};
                br_equal = 1'b0;
                br_less = 1'b0;
                start_calc = 1;
            end

            default:    {start_calc, br_equal, br_less, carry, out} = {1'b0, 1'b0, 1'b0, 1'b0, 32'b0};
        endcase
    end   
endmodule

module MUL_unit(
    input [31:0] A, 
    input [31:0] B, 
    input i_clk,
    input i_start_calc, // reset and start calculation
    output o_ready,
    output [63:0] Y
);

    // assign o_ready = 1'b1;
    // assign Y = A * B;
    reg  [32:0] adder_out;
    reg  [4: 0] counter, counter_nxt;
    reg  [64:0] shreg, shreg_nxt;
    reg state, state_nxt;
    reg [31:0] a, b, a_nxt, b_nxt;
    reg finished, finished_nxt;

    assign Y = shreg;
    // assign o_ready = (state_nxt && counter != 31) || finished;
    assign o_ready = state_nxt;
    // state definition
    localparam S_IDLE = 1'b1, S_CALC = 1'b0;
    // state machine
    always @(*) begin
        case (state)
            S_IDLE: state_nxt = i_start_calc ? S_CALC : S_IDLE;
            S_CALC: state_nxt = (counter == 5'b11111) ? S_IDLE : S_CALC;
            default: state_nxt = S_IDLE;
        endcase
    end
    always @(*) begin
        case (state)
            S_IDLE: begin
                a_nxt = A;
                b_nxt = B;
            end
            default : begin
                a_nxt = a;
                b_nxt = b;
            end
        endcase
    end
    always @(*) begin
        case (state)
            S_IDLE: counter_nxt = 5'b0;
            S_CALC: counter_nxt = (counter == 5'b11111) ? 5'b0 : counter + 5'b1;
        endcase
    end 
    always @(*) begin
        case (counter)
            5'b11111: finished_nxt = 1'b1;
            default: finished_nxt = 1'b0;
        endcase
    end
    // adder logic
    always @(*) begin
        adder_out = shreg[63:32] + b;
    end
    // shift register logic 
    always @(*) begin
        case (state)
            S_IDLE: 
                case (state_nxt)
                    S_IDLE: shreg_nxt = 0;
                    S_CALC: 
                        if (A[0]) shreg_nxt = {B, A[31:1]};
                        else shreg_nxt = {33'b0, A[31:1]};
                endcase
            S_CALC:    
                if (shreg[0]) shreg_nxt = {adder_out, shreg[31:1]};
                else shreg_nxt = {1'b0, shreg[64:1]};
        endcase
    end
    // sequential logic
    always @(posedge i_clk) begin
        begin
            state <= state_nxt;
            counter <= counter_nxt;
            shreg <= shreg_nxt;
            a <= a_nxt;
            b <= b_nxt;
            finished <= finished_nxt;
        end
    end
endmodule



