/*///////////////////////////////////////////////////////////////////////////////
    Module: CHIP.v
    Creator: Yi-Kai Wu
    Last editor: Yi-Kai Wu
    Last edited date: 2024/05/17
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
        wire [BIT_W-1:0] mem_addr, mem_wdata, mem_rdata;
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
        reg     [3:0]       ctrl_alu;
        wire    [BIT_W-1:0]  read_or_alu_result;//output of Mux2
        wire    [BIT_W-1:0]  pc_plus_4, pc_plus_imm;
        wire    [BIT_W-1:0]  jal_addr, jalr_addr;
        wire    [6:0]       funct7;
        wire    [2:0]       funct3;

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
        //esle ctrl_alu_src =0,alu_input_2 = rs2

    assign ctrl_auipc = (opcode == U_TYPE_AUIPC) 1'b1 : 1'b0;
        //When auipc, write_data = PC + imm.

    assign ctrl_mem_to_reg = ( opcode == I_TYPE_LOAD ) ? 1'b1 : 1'b0;
        //=1 when load, means read_or_alu_result = mem_read_data;
        // else =0, means read_or_alu_result = alu_result

    assign ctrl_reg_write = ( (opcode == S_TYPE) || (opcode == B_TYPE) ) ? 1'b0 : 1'b1;
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
    assign alu_in_2 = (ctrl_shamt) ? instr[24:20]
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
    //MUX4
    assign jal_addr = (ctrl_jal||(ctrl_branch && br_comp)) ? pc_plus_imm : pc_plus_4;
    assign jalr_addr = rs1_data + imm;
    //MUX5 
    always @(*) begin
        next_PC = (ctrl_jalr) ? jalr_addr : jal_addr;
    end

    //Deal with ecall
    assign ctrl_ecall = ({funct3, opcode}== {3'b000, ECALL}) ? 1'b1, 1'b0;

    

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
        .br_comp(br_comp),
        .out(alu_result)  
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
                endcase
            end
            B_TYPE : ctrl_alu = 4'b0110;//SUB
            I_TYPE : begin
                case (funct3)
                    {3'b000} : ctrl_alu = 4'b0010;//ADD
                    {3'b001} : ctrl_alu = 4'b0101;//SLL
                    {3'b101} : ctrl_alu = 4'b0011;//SRA
                    {3'b010} : ctrl_alu = 4'b1000;//SLT
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

module MULDIV_unit(
    // TODO: port declaration
    );
    // Todo: HW2
endmodule

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
            output [BIT_W-1:0] o_proc_rdata,
            output o_proc_stall,
            input i_proc_finish,
            output o_cache_finish,
        // memory interface
            output o_mem_cen,
            output o_mem_wen,
            output [ADDR_W-1:0] o_mem_addr,
            output [BIT_W*4-1:0]  o_mem_wdata,
            input [BIT_W*4-1:0] i_mem_rdata,
            input i_mem_stall,
            output o_cache_available,
        // others
        input  [ADDR_W-1: 0] i_offset
    );

    assign o_cache_available = 0; // change this value to 1 if the cache is implemented

    //------------------------------------------//
    //          default connection              //
    assign o_mem_cen = i_proc_cen;              //
    assign o_mem_wen = i_proc_wen;              //
    assign o_mem_addr = i_proc_addr;            //
    assign o_mem_wdata = i_proc_wdata;          //
    assign o_proc_rdata = i_mem_rdata[0+:BIT_W];//
    assign o_proc_stall = i_mem_stall;          //
    //------------------------------------------//

    // Todo: BONUS

endmodule

module alu #(
    parameter BIT_W = 32
)(
    ctrl,
    x,
    y,
    br_comp,
    out  
);

//Inout port:   
    input  [3:0] ctrl;
    input  [BIT_W-1:0] x;//rs1
    input  [BIT_W-1:0] y;//rs2
    output reg        br_comp;
    output [BIT_W-1:0] out;

    //Parameter
    parameter   BITWISE_AND = 4'b0000, BITWISE_OR = 4'b0001, ADD = 4'b0010, SUB = 4'b0110, SLT = 4'b1000,
                SRA = 4'b0011, XOR = 4'b0100, SLL = 4'b0101, MUL = 4'b0111; 
    
    //wire & reg declaration
    reg            carry;
    reg [BIT_W-1:0] out;

//CL:
    always@(*)begin
        case(ctrl)
            ADD: begin
                {carry,out} = {x[BIT_W-1], x} + {y[BIT_W-1],y};
                br_comp = 1'b0;
            end
            SUB: begin
                {carry,out} = {x[BIT_W-1], x} - {y[BIT_W-1],y};
                br_comp = ~(|out);//=1 when all bits are 0; Test is this is smaller than br_comp = (out == 0) ? 1'b1 : 1'b0;
            end
            BITWISE_AND:  begin
                {carry,out} = {1'b0, x & y};
                br_comp = 1'b0;
            end  
            BITWISE_OR:begin
                {carry,out} = {1'b0,  x | y};
                br_comp = 1'b0;
            end     

            SLT:  begin
                carry = 1'b0;
                out = ($signed(x) < $signed(y)) ? {31'b0, 1'b1} : {31'b0, 1'b0};
                br_comp = 1'b0;
            end 
            SRA:  begin
                carry = 1'b0;
                out = $signed(x) >>> y[4:0];
                br_comp = 1'b0;
            end 
            SLL:  begin
                carry = 1'b0;
                out = x << y[4:0];
                br_comp = 1'b0;
            end 
            XOR:  begin
                {carry,out} = {1'b0, x ^ y};
                br_comp = 1'b0;
            end 

            default:    {br_comp, carry,out} = {1'b0, 1'b0, 32'b0};
        endcase
    end   
endmodule




module cache(
    clk,
    proc_reset,
    proc_read,
    proc_write,
    proc_addr,
    proc_rdata,
    proc_wdata,
    proc_stall,
    mem_read,
    mem_write,
    mem_addr,
    mem_rdata,
    mem_wdata,
    mem_ready
);
    
//==== input/output definition ============================
    input          clk;
    // processor interface
    input          proc_reset;
    input          proc_read, proc_write;
    input   [29:0] proc_addr;
    input   [31:0] proc_wdata;
    output         proc_stall;
    output reg [31:0] proc_rdata;
    // memory interface
    input  [127:0] mem_rdata;
    input          mem_ready;
    output reg        mem_read, mem_write;
    output reg [27:0] mem_addr;
    output reg [127:0] mem_wdata;


// Parameters for cache size and organization
    parameter BLOCK_SIZE = 128; // 4 words of 32 bits each
    parameter NUM_SETS = 4; // Half the number of blocks in a direct-mapped cache
    parameter WORD_SIZE = 32;
    parameter OFFSET_BITS = 2; // 4 words per block
    parameter INDEX_BITS = 2; // log2(NUM_SETS)
    parameter TAG_BITS = 26; // Adjusted for fewer index bits
    parameter WAY_COUNT = 2; // 2-way associative

//==== wire/reg definition ================================
    // Cache block definition
    reg [TAG_BITS-1:0]      tags_r[NUM_SETS-1:0][WAY_COUNT-1:0], tags_w[NUM_SETS-1:0][WAY_COUNT-1:0];
    reg                     valid_r[NUM_SETS-1:0][WAY_COUNT-1:0], valid_w[NUM_SETS-1:0][WAY_COUNT-1:0];
    reg                     dirty_r[NUM_SETS-1:0][WAY_COUNT-1:0], dirty_w[NUM_SETS-1:0][WAY_COUNT-1:0];
    reg [BLOCK_SIZE-1:0]    data_r[NUM_SETS-1:0][WAY_COUNT-1:0], data_w[NUM_SETS-1:0][WAY_COUNT-1:0];
    reg                     lru_r[NUM_SETS-1:0], lru_w[NUM_SETS-1:0]; // 0 for LRU, 1 for MRU
    reg                     way;


    // Address split
    wire [INDEX_BITS-1:0]   index = proc_addr[OFFSET_BITS+INDEX_BITS-1:OFFSET_BITS];
    wire [OFFSET_BITS-1:0]  offset = proc_addr[OFFSET_BITS-1:0];
    wire [TAG_BITS-1:0]     tag = proc_addr[29:OFFSET_BITS+INDEX_BITS];
   
   // FSM states
    localparam IDLE = 2'd0, COMPARE_TAG = 2'd1, WRITE_BACK = 2'd2, ALLOCATE = 2'd3;
    reg [1:0]               state_r, state_w;

    //loop index
    integer i, way_cnt;

    reg                     hit;
//Continuous assignment
    // Assign outputs to output flip-flops
    //This is incorrect
    //assign proc_rdata = (hit && proc_read) ? data_r[index][offset*WORD_SIZE +: WORD_SIZE] : 32'b0;//e.g. offset = 0, [Word_size -1 : 0]
    assign proc_stall = (hit) ? 1'b0 : 1'b1;
    // assign mem_read = mem_read_r;
    // assign mem_write = mem_write_r;
    // assign mem_addr = mem_addr_r;
    // assign mem_wdata = mem_wdata_r;

//==== combinational circuit ==============================
/*========FSM==========================================*/
    // FSM Next State Logic
    always @(*) begin
        hit = 0;
        //state_w = state_r;
        case (state_r)
            //IDLE: state_w = (proc_read || proc_write) ? COMPARE_TAG : IDLE;
            COMPARE_TAG: begin
                hit = (valid_r[index][0] && (tags_r[index][0] == tag)) || (valid_r[index][1] && (tags_r[index][1] == tag));
                if (hit) begin                 
                    state_w = COMPARE_TAG;
                end
                else state_w = (dirty_r[index][lru_r[index]]) ? WRITE_BACK : ALLOCATE; //Cache Miss, check the lru way, if it is dirty, write back
                                                                                //else, allocate 
            end
            WRITE_BACK:
                state_w = (mem_ready) ? ALLOCATE : WRITE_BACK;
            ALLOCATE:
                state_w = (mem_ready) ? COMPARE_TAG : ALLOCATE;
            default: state_w = state_r;
        endcase
    end
    // FSM Output Logic
    always @(*) begin
        //default value
        // proc_rdata_w = proc_rdata_r;  // Default to holding value
        // proc_stall_w = 0;
        mem_read = 0;
        mem_write = 0;
        mem_addr = 0;
        mem_wdata = 128'b0;
        proc_rdata = 32'b0;

        for (i = 0; i < NUM_SETS; i = i + 1) begin
            for (way_cnt = 0; way_cnt < WAY_COUNT; way_cnt = way_cnt + 1) begin
                tags_w[i][way_cnt] = tags_r[i][way_cnt];
                valid_w[i][way_cnt] = valid_r[i][way_cnt];
                dirty_w[i][way_cnt] = dirty_r[i][way_cnt];
                data_w[i][way_cnt] = data_r[i][way_cnt];
            end
            lru_w[i] = lru_r[i];
        end

        case (state_r)

            COMPARE_TAG: begin
                //hit condition:
                if (valid_r[index][0] && (tags_r[index][0] == tag)) begin
                    //which_set = 0;
                    if(proc_read) proc_rdata = data_r[index][0][offset*WORD_SIZE +: WORD_SIZE];//e.g. offset = 0, [Word_size -1 : 0]
                    if (proc_write) begin
                        data_w[index][0][offset*WORD_SIZE +: WORD_SIZE] = proc_wdata;
                        dirty_w[index][0] = 1;
                    end
                    lru_w[index] = 1'b1;
                end
                else if(valid_r[index][1] && (tags_r[index][1] == tag)) begin
                    //which_set = 1;
                    if(proc_read) proc_rdata = data_r[index][1][offset*WORD_SIZE +: WORD_SIZE];//e.g. offset = 0, [Word_size -1 : 0]
                    if (proc_write) begin
                        data_w[index][1][offset*WORD_SIZE +: WORD_SIZE] = proc_wdata;
                        dirty_w[index][1] = 1;
                    end
                    lru_w[index] = 1'b0;
                end              
                else begin
                    //which_set = 1'bz;//proc_stall_w = 1'b1;
                end

                //miss
                // Even if write, no need to set dirty, since after coming back to COMPARE_TAG from ALLOCATE
                // it will compare the tags and find it hit and then set dirty
            end
            WRITE_BACK: begin
                if (mem_ready) begin
                    mem_write = 0;
                    dirty_w[index][lru_r[index]] = 0;
                end 
                else begin
                    mem_addr = {tags_r[index][lru_r[index]], index};//28b, 4-word addressing, with the stored tag
                    mem_wdata = data_r[index][lru_r[index]];
                    mem_write = 1;
                    // proc_stall_w = 1;
                end
            end
            ALLOCATE: begin
                if (mem_ready) begin
                    data_w[index][lru_r[index]] = mem_rdata;
                    tags_w[index][lru_r[index]] = tag;//update tag 
                    valid_w[index][lru_r[index]] = 1;
                end 
                else begin
                    mem_addr = {tag, index};//28b, 4-word addressing, with the new tag
                    mem_read = 1;
                    // proc_stall_w = 1;
                end
            end
            default: begin
                mem_read = 0;
                mem_write = 0;
                mem_addr = 0;
                mem_wdata = 128'b0;
                proc_rdata = 32'b0;

                for (i = 0; i < NUM_SETS; i = i + 1) begin
                    for (way_cnt = 0; way_cnt < WAY_COUNT; way_cnt = way_cnt + 1) begin
                        tags_w[i][way_cnt] = tags_r[i][way_cnt];
                        valid_w[i][way_cnt] = valid_r[i][way_cnt];
                        dirty_w[i][way_cnt] = dirty_r[i][way_cnt];
                        data_w[i][way_cnt] = data_r[i][way_cnt];
                    end
                    lru_w[i] = lru_r[i];
                end
            end
        endcase
    end

//==== sequential circuit =================================
    always@( posedge clk ) begin
        if( proc_reset ) begin
            state_r <= COMPARE_TAG;
            // proc_stall_r <= 0;
            //mem_read_r <= 0;
            // mem_write_r <= 0;
            // // proc_rdata_r <= 32'b0;
            // mem_addr_r <= 28'b0;
            // mem_wdata_r <= {BLOCK_SIZE{1'b0}};
            for (i = 0; i < NUM_SETS; i = i + 1) begin
                for (way_cnt = 0; way_cnt < WAY_COUNT; way_cnt = way_cnt + 1) begin
                    tags_r[i][way_cnt] <= 0;
                    valid_r[i][way_cnt] <= 0;
                    dirty_r[i][way_cnt] <= 0;//0-extension
                    data_r[i][way_cnt] <= 0;
                end
                lru_r[i] <= 0;
            end
        end

        else begin
            state_r <= state_w;           
            // proc_stall_r <= proc_stall_w;
            // mem_read_r <= mem_read_w;
            // mem_write_r <= mem_write_w;
            // // proc_rdata_r <= proc_rdata_w;
            // mem_addr_r <= mem_addr_w;
            // mem_wdata_r <= mem_wdata_w;
            for (i = 0; i < NUM_SETS; i = i + 1) begin
                for (way_cnt = 0; way_cnt < WAY_COUNT; way_cnt = way_cnt + 1) begin
                    tags_r[i][way_cnt] <= tags_w[i][way_cnt];
                    valid_r[i][way_cnt] <= valid_w[i][way_cnt];
                    dirty_r[i][way_cnt] <= dirty_w[i][way_cnt];
                    data_r[i][way_cnt] <= data_w[i][way_cnt];
                end
                lru_r[i] <= lru_w[i];
            end
        end
    end

endmodule
