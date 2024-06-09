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
        //else ctrl_alu_src =0,alu_input_2 = rs2

    assign ctrl_auipc = (opcode == U_TYPE_AUIPC) 1'b1 : 1'b0;
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

    //To-do: branch controller
    //MUX4
    assign jal_addr = (ctrl_jal||(ctrl_branch && br_comp)) ? pc_plus_imm : pc_plus_4;
    assign jalr_addr = rs1_data + imm;
    //MUX5 
    always @(*) begin
        next_PC = (i_DMEM_stall) ? PC 
                : (ctrl_jalr) ? jalr_addr : jal_addr;
    end

    //Deal with ecall
    assign ctrl_ecall = ({funct3, opcode}== {3'b000, ECALL}) ? 1'b1, 1'b0;
    //To-do:
    //Ask cache to store all values back to memory. When received cache store_finish, pull up o_finish
    //Assume no cache:
    assign o_finish = (ctrl_ecall) 1'b1 : 1'b0;
    

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

    //To-do: instantiate MULDIV_unit

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

module MULDIV_unit(
    input [31:0] A, 
    input [31:0] B, 
    output [63:0] Y
);

//outputs of 16*16 dadda.      
    wire [31:0]y11,y12,y21,y22;
    // reg  [31:0]y11_stage_1,y12_stage_1,y21_stage_1,y22_stage_1;
    // reg  [31:0]y11,y12,y21,y22;

//sum and carry of final 2 stages.      
    wire [31:0]s_1,c_1; 
    wire [46:0]c_2;

/*Pipeline Stage1*/
    dadda_16 d1(.A(A[15:0]),.B(B[15:0]),.Y(y11));
   

    dadda_16 d2(.A(A[15:0]),.B(B[31:16]),.Y(y12));
    dadda_16 d3(.A(A[31:16]),.B(B[15:0]),.Y(y21));
    dadda_16 d4(.A(A[31:16]),.B(B[31:16]),.Y(y22));

    // always @( *) begin
    //     y11_stage_1 = y11;
    //     y12_stage_1 = y12;
    //     y21_stage_1 = y21;
    //    // y22_stage_1 = y22;
    // end

	// always @(posedge clk) begin
	// 	if (!rst_n) begin
    //         y11 <= 32'b0;
    //         y12 <= 32'b0;
    //         y21 <= 32'b0;
    //         //y22 <= 32'b0;
    //     end
    //     else begin
    //         y11 <= y11_stage_1;
    //         y12 <= y12_stage_1;
    //         y21 <= y21_stage_1;
    //         //y22 <= y22_stage_1;
    //     end
    // end
    
//Stage 1 - reducing fom 3 to 2
    assign Y[15:0] = y11[15:0];
    csa_dadda c_11(.A(y11[16]),.B(y12[0]),.Cin(y21[0]),.Y(s_1[0]),.Cout(c_1[0]));
    assign Y[16] = s_1[0];
    csa_dadda c_12(.A(y11[17]),.B(y12[1]),.Cin(y21[1]),.Y(s_1[1]),.Cout(c_1[1]));
    csa_dadda c_13(.A(y11[18]),.B(y12[2]),.Cin(y21[2]),.Y(s_1[2]),.Cout(c_1[2]));
    csa_dadda c_14(.A(y11[19]),.B(y12[3]),.Cin(y21[3]),.Y(s_1[3]),.Cout(c_1[3]));
    csa_dadda c_15(.A(y11[20]),.B(y12[4]),.Cin(y21[4]),.Y(s_1[4]),.Cout(c_1[4]));
    csa_dadda c_16(.A(y11[21]),.B(y12[5]),.Cin(y21[5]),.Y(s_1[5]),.Cout(c_1[5]));
    csa_dadda c_17(.A(y11[22]),.B(y12[6]),.Cin(y21[6]),.Y(s_1[6]),.Cout(c_1[6]));
    csa_dadda c_18(.A(y11[23]),.B(y12[7]),.Cin(y21[7]),.Y(s_1[7]),.Cout(c_1[7]));
    csa_dadda c_19(.A(y11[24]),.B(y12[8]),.Cin(y21[8]),.Y(s_1[8]),.Cout(c_1[8]));
    csa_dadda c_110(.A(y11[25]),.B(y12[9]),.Cin(y21[9]),.Y(s_1[9]),.Cout(c_1[9]));
    csa_dadda c_111(.A(y11[26]),.B(y12[10]),.Cin(y21[10]),.Y(s_1[10]),.Cout(c_1[10]));
    csa_dadda c_112(.A(y11[27]),.B(y12[11]),.Cin(y21[11]),.Y(s_1[11]),.Cout(c_1[11]));
    csa_dadda c_113(.A(y11[28]),.B(y12[12]),.Cin(y21[12]),.Y(s_1[12]),.Cout(c_1[12]));
    csa_dadda c_114(.A(y11[29]),.B(y12[13]),.Cin(y21[13]),.Y(s_1[13]),.Cout(c_1[13]));
    csa_dadda c_115(.A(y11[30]),.B(y12[14]),.Cin(y21[14]),.Y(s_1[14]),.Cout(c_1[14]));
    csa_dadda c_116(.A(y11[31]),.B(y12[15]),.Cin(y21[15]),.Y(s_1[15]),.Cout(c_1[15]));
    csa_dadda c_117(.A(y22[0]),.B(y12[16]),.Cin(y21[16]),.Y(s_1[16]),.Cout(c_1[16]));
    csa_dadda c_118(.A(y22[1]),.B(y12[17]),.Cin(y21[17]),.Y(s_1[17]),.Cout(c_1[17]));
    csa_dadda c_119(.A(y22[2]),.B(y12[18]),.Cin(y21[18]),.Y(s_1[18]),.Cout(c_1[18]));
    csa_dadda c_120(.A(y22[3]),.B(y12[19]),.Cin(y21[19]),.Y(s_1[19]),.Cout(c_1[19]));
    csa_dadda c_121(.A(y22[4]),.B(y12[20]),.Cin(y21[20]),.Y(s_1[20]),.Cout(c_1[20]));
    csa_dadda c_122(.A(y22[5]),.B(y12[21]),.Cin(y21[21]),.Y(s_1[21]),.Cout(c_1[21]));
    csa_dadda c_123(.A(y22[6]),.B(y12[22]),.Cin(y21[22]),.Y(s_1[22]),.Cout(c_1[22]));
    csa_dadda c_124(.A(y22[7]),.B(y12[23]),.Cin(y21[23]),.Y(s_1[23]),.Cout(c_1[23]));
    csa_dadda c_125(.A(y22[8]),.B(y12[24]),.Cin(y21[24]),.Y(s_1[24]),.Cout(c_1[24]));
    csa_dadda c_126(.A(y22[9]),.B(y12[25]),.Cin(y21[25]),.Y(s_1[25]),.Cout(c_1[25]));
    csa_dadda c_127(.A(y22[10]),.B(y12[26]),.Cin(y21[26]),.Y(s_1[26]),.Cout(c_1[26]));
    csa_dadda c_128(.A(y22[11]),.B(y12[27]),.Cin(y21[27]),.Y(s_1[27]),.Cout(c_1[27]));
    csa_dadda c_129(.A(y22[12]),.B(y12[28]),.Cin(y21[28]),.Y(s_1[28]),.Cout(c_1[28]));
    csa_dadda c_130(.A(y22[13]),.B(y12[29]),.Cin(y21[29]),.Y(s_1[29]),.Cout(c_1[29]));
    csa_dadda c_131(.A(y22[14]),.B(y12[30]),.Cin(y21[30]),.Y(s_1[30]),.Cout(c_1[30]));
    csa_dadda c_132(.A(y22[15]),.B(y12[31]),.Cin(y21[31]),.Y(s_1[31]),.Cout(c_1[31]));
    
    
//Stage 1 - reducing fom 2 to 1
    // adding total sum and carry to get final output
    HA h1(.a(s_1[1]),.b(c_1[0]),.Sum(Y[17]),.Cout(c_2[0]));
    
    
    csa_dadda c_22(.A(s_1[2]),.B(c_1[1]),.Cin(c_2[0]),.Y(Y[18]),.Cout(c_2[1]));
    csa_dadda c_23(.A(s_1[3]),.B(c_1[2]),.Cin(c_2[1]),.Y(Y[19]),.Cout(c_2[2]));
    csa_dadda c_24(.A(s_1[4]),.B(c_1[3]),.Cin(c_2[2]),.Y(Y[20]),.Cout(c_2[3]));
    csa_dadda c_25(.A(s_1[5]),.B(c_1[4]),.Cin(c_2[3]),.Y(Y[21]),.Cout(c_2[4]));
    csa_dadda c_26(.A(s_1[6]),.B(c_1[5]),.Cin(c_2[4]),.Y(Y[22]),.Cout(c_2[5]));
    csa_dadda c_27(.A(s_1[7]),.B(c_1[6]),.Cin(c_2[5]),.Y(Y[23]),.Cout(c_2[6]));
    csa_dadda c_28(.A(s_1[8]),.B(c_1[7]),.Cin(c_2[6]),.Y(Y[24]),.Cout(c_2[7]));
    csa_dadda c_29(.A(s_1[9]),.B(c_1[8]),.Cin(c_2[7]),.Y(Y[25]),.Cout(c_2[8]));
    csa_dadda c_210(.A(s_1[10]),.B(c_1[9]),.Cin(c_2[8]),.Y(Y[26]),.Cout(c_2[9]));
    csa_dadda c_211(.A(s_1[11]),.B(c_1[10]),.Cin(c_2[9]),.Y(Y[27]),.Cout(c_2[10]));
    csa_dadda c_212(.A(s_1[12]),.B(c_1[11]),.Cin(c_2[10]),.Y(Y[28]),.Cout(c_2[11]));
    csa_dadda c_213(.A(s_1[13]),.B(c_1[12]),.Cin(c_2[11]),.Y(Y[29]),.Cout(c_2[12]));
    csa_dadda c_214(.A(s_1[14]),.B(c_1[13]),.Cin(c_2[12]),.Y(Y[30]),.Cout(c_2[13]));
    csa_dadda c_215(.A(s_1[15]),.B(c_1[14]),.Cin(c_2[13]),.Y(Y[31]),.Cout(c_2[14]));
    
    csa_dadda c_216(.A(s_1[16]),.B(c_1[15]),.Cin(c_2[14]),.Y(Y[32]),.Cout(c_2[15]));
    csa_dadda c_217(.A(s_1[17]),.B(c_1[16]),.Cin(c_2[15]),.Y(Y[33]),.Cout(c_2[16]));
    csa_dadda c_218(.A(s_1[18]),.B(c_1[17]),.Cin(c_2[16]),.Y(Y[34]),.Cout(c_2[17]));
    csa_dadda c_219(.A(s_1[19]),.B(c_1[18]),.Cin(c_2[17]),.Y(Y[35]),.Cout(c_2[18]));
    csa_dadda c_220(.A(s_1[20]),.B(c_1[19]),.Cin(c_2[18]),.Y(Y[36]),.Cout(c_2[19]));
    csa_dadda c_221(.A(s_1[21]),.B(c_1[20]),.Cin(c_2[19]),.Y(Y[37]),.Cout(c_2[20]));
    csa_dadda c_222(.A(s_1[22]),.B(c_1[21]),.Cin(c_2[20]),.Y(Y[38]),.Cout(c_2[21]));
    csa_dadda c_223(.A(s_1[23]),.B(c_1[22]),.Cin(c_2[21]),.Y(Y[39]),.Cout(c_2[22]));
    csa_dadda c_224(.A(s_1[24]),.B(c_1[23]),.Cin(c_2[22]),.Y(Y[40]),.Cout(c_2[23]));
    csa_dadda c_225(.A(s_1[25]),.B(c_1[24]),.Cin(c_2[23]),.Y(Y[41]),.Cout(c_2[24]));
    csa_dadda c_226(.A(s_1[26]),.B(c_1[25]),.Cin(c_2[24]),.Y(Y[42]),.Cout(c_2[25]));
    csa_dadda c_227(.A(s_1[27]),.B(c_1[26]),.Cin(c_2[25]),.Y(Y[43]),.Cout(c_2[26]));
    csa_dadda c_228(.A(s_1[28]),.B(c_1[27]),.Cin(c_2[26]),.Y(Y[44]),.Cout(c_2[27]));
    csa_dadda c_229(.A(s_1[29]),.B(c_1[28]),.Cin(c_2[27]),.Y(Y[45]),.Cout(c_2[28]));
    csa_dadda c_230(.A(s_1[30]),.B(c_1[29]),.Cin(c_2[28]),.Y(Y[46]),.Cout(c_2[29]));
    csa_dadda c_231(.A(s_1[31]),.B(c_1[30]),.Cin(c_2[29]),.Y(Y[47]),.Cout(c_2[30])); 
    csa_dadda c_232(.A(y22[16]),.B(c_1[31]),.Cin(c_2[30]),.Y(Y[48]),.Cout(c_2[31]));


    HA h2(.a(y22[17]),.b(c_2[31]),.Sum(Y[49]),.Cout(c_2[32]));
    HA h3(.a(y22[18]),.b(c_2[32]),.Sum(Y[50]),.Cout(c_2[33]));
    HA h4(.a(y22[19]),.b(c_2[33]),.Sum(Y[51]),.Cout(c_2[34]));
    HA h5(.a(y22[20]),.b(c_2[34]),.Sum(Y[52]),.Cout(c_2[35]));
    HA h6(.a(y22[21]),.b(c_2[35]),.Sum(Y[53]),.Cout(c_2[36]));
    HA h7(.a(y22[22]),.b(c_2[36]),.Sum(Y[54]),.Cout(c_2[37]));
    HA h8(.a(y22[23]),.b(c_2[37]),.Sum(Y[55]),.Cout(c_2[38]));
    HA h9(.a(y22[24]),.b(c_2[38]),.Sum(Y[56]),.Cout(c_2[39]));
    HA h10(.a(y22[25]),.b(c_2[39]),.Sum(Y[57]),.Cout(c_2[40]));
    HA h11(.a(y22[26]),.b(c_2[40]),.Sum(Y[58]),.Cout(c_2[41]));
    HA h12(.a(y22[27]),.b(c_2[41]),.Sum(Y[59]),.Cout(c_2[42]));
    HA h13(.a(y22[28]),.b(c_2[42]),.Sum(Y[60]),.Cout(c_2[43]));
    HA h14(.a(y22[29]),.b(c_2[43]),.Sum(Y[61]),.Cout(c_2[44]));
    HA h15(.a(y22[30]),.b(c_2[44]),.Sum(Y[62]),.Cout(c_2[45]));
    HA h16(.a(y22[31]),.b(c_2[45]),.Sum(Y[63]),.Cout(c_2[46]));
endmodule

module dadda_16(A,B,Y);
    
    input [15:0]A;
    input [15:0]B;
    
    output wire [31:0] Y;
//outputs of 8*8 dadda.    
    wire [15:0]y11,y12,y21,y22;

//sum and carry of final 2 stages.     
    wire [15:0]s_1,c_1;    
    wire [22:0] c_2;
    
    dadda_8 d1(.A(A[7:0]),.B(B[7:0]),.y(y11));
    dadda_8 d2(.A(A[7:0]),.B(B[15:8]),.y(y12));
    dadda_8 d3(.A(A[15:8]),.B(B[7:0]),.y(y21));
    dadda_8 d4(.A(A[15:8]),.B(B[15:8]),.y(y22));
    assign Y[7:0] = y11[7:0];
    
//Stage 1 - reducing fom 3 to 2
    
    csa_dadda c_11(.A(y11[8]),.B(y12[0]),.Cin(y21[0]),.Y(s_1[0]),.Cout(c_1[0]));
    assign Y[8] = s_1[0];
    csa_dadda c_12(.A(y11[9]),.B(y12[1]),.Cin(y21[1]),.Y(s_1[1]),.Cout(c_1[1]));
    csa_dadda c_13(.A(y11[10]),.B(y12[2]),.Cin(y21[2]),.Y(s_1[2]),.Cout(c_1[2]));
    csa_dadda c_14(.A(y11[11]),.B(y12[3]),.Cin(y21[3]),.Y(s_1[3]),.Cout(c_1[3]));
    csa_dadda c_15(.A(y11[12]),.B(y12[4]),.Cin(y21[4]),.Y(s_1[4]),.Cout(c_1[4]));
    csa_dadda c_16(.A(y11[13]),.B(y12[5]),.Cin(y21[5]),.Y(s_1[5]),.Cout(c_1[5]));
    csa_dadda c_17(.A(y11[14]),.B(y12[6]),.Cin(y21[6]),.Y(s_1[6]),.Cout(c_1[6]));
    csa_dadda c_18(.A(y11[15]),.B(y12[7]),.Cin(y21[7]),.Y(s_1[7]),.Cout(c_1[7]));
    csa_dadda c_19(.A(y22[0]),.B(y12[8]),.Cin(y21[8]),.Y(s_1[8]),.Cout(c_1[8]));
    csa_dadda c_110(.A(y22[1]),.B(y12[9]),.Cin(y21[9]),.Y(s_1[9]),.Cout(c_1[9]));
    csa_dadda c_111(.A(y22[2]),.B(y12[10]),.Cin(y21[10]),.Y(s_1[10]),.Cout(c_1[10]));
    csa_dadda c_112(.A(y22[3]),.B(y12[11]),.Cin(y21[11]),.Y(s_1[11]),.Cout(c_1[11]));
    csa_dadda c_113(.A(y22[4]),.B(y12[12]),.Cin(y21[12]),.Y(s_1[12]),.Cout(c_1[12]));
    csa_dadda c_114(.A(y22[5]),.B(y12[13]),.Cin(y21[13]),.Y(s_1[13]),.Cout(c_1[13]));
    csa_dadda c_115(.A(y22[6]),.B(y12[14]),.Cin(y21[14]),.Y(s_1[14]),.Cout(c_1[14]));
    csa_dadda c_116(.A(y22[7]),.B(y12[15]),.Cin(y21[15]),.Y(s_1[15]),.Cout(c_1[15]));
    
//Stage 2 - reducing fom 2 to 1
        // adding total sum and carry to get final output
    HA h1(.a(s_1[1]),.b(c_1[0]),.Sum(Y[9]),.Cout(c_2[0]));


    csa_dadda c_22(.A(s_1[2]),.B(c_1[1]),.Cin(c_2[0]),.Y(Y[10]),.Cout(c_2[1]));
    csa_dadda c_23(.A(s_1[3]),.B(c_1[2]),.Cin(c_2[1]),.Y(Y[11]),.Cout(c_2[2]));
    csa_dadda c_24(.A(s_1[4]),.B(c_1[3]),.Cin(c_2[2]),.Y(Y[12]),.Cout(c_2[3]));
    csa_dadda c_25(.A(s_1[5]),.B(c_1[4]),.Cin(c_2[3]),.Y(Y[13]),.Cout(c_2[4]));
    csa_dadda c_26(.A(s_1[6]),.B(c_1[5]),.Cin(c_2[4]),.Y(Y[14]),.Cout(c_2[5]));
    csa_dadda c_27(.A(s_1[7]),.B(c_1[6]),.Cin(c_2[5]),.Y(Y[15]),.Cout(c_2[6]));
    csa_dadda c_28(.A(s_1[8]),.B(c_1[7]),.Cin(c_2[6]),.Y(Y[16]),.Cout(c_2[7]));
    csa_dadda c_29(.A(s_1[9]),.B(c_1[8]),.Cin(c_2[7]),.Y(Y[17]),.Cout(c_2[8]));
    csa_dadda c_210(.A(s_1[10]),.B(c_1[9]),.Cin(c_2[8]),.Y(Y[18]),.Cout(c_2[9]));
    csa_dadda c_211(.A(s_1[11]),.B(c_1[10]),.Cin(c_2[9]),.Y(Y[19]),.Cout(c_2[10]));
    csa_dadda c_212(.A(s_1[12]),.B(c_1[11]),.Cin(c_2[10]),.Y(Y[20]),.Cout(c_2[11]));
    csa_dadda c_213(.A(s_1[13]),.B(c_1[12]),.Cin(c_2[11]),.Y(Y[21]),.Cout(c_2[12]));
    csa_dadda c_214(.A(s_1[14]),.B(c_1[13]),.Cin(c_2[12]),.Y(Y[22]),.Cout(c_2[13]));
    csa_dadda c_215(.A(s_1[15]),.B(c_1[14]),.Cin(c_2[13]),.Y(Y[23]),.Cout(c_2[14]));
    csa_dadda c_216(.A(y22[8]),.B(c_1[15]),.Cin(c_2[14]),.Y(Y[24]),.Cout(c_2[15]));

    HA h2(.a(y22[9]),.b(c_2[15]),.Sum(Y[25]),.Cout(c_2[16]));
    HA h3(.a(y22[10]),.b(c_2[16]),.Sum(Y[26]),.Cout(c_2[17]));
    HA h4(.a(y22[11]),.b(c_2[17]),.Sum(Y[27]),.Cout(c_2[18]));
    HA h5(.a(y22[12]),.b(c_2[18]),.Sum(Y[28]),.Cout(c_2[19]));
    HA h6(.a(y22[13]),.b(c_2[19]),.Sum(Y[29]),.Cout(c_2[20]));
    HA h7(.a(y22[14]),.b(c_2[20]),.Sum(Y[30]),.Cout(c_2[21]));
    HA h8(.a(y22[15]),.b(c_2[21]),.Sum(Y[31]),.Cout(c_2[22]));
endmodule

module dadda_8(A,B,y);
    
    input [7:0] A;
    input [7:0] B;
    output wire [15:0] y;
    reg  gen_pp [0:7][7:0];
// stage-1 sum and carry
    wire [0:5]s1,c1;
// stage-2 sum and carry
    wire [0:13]s2,c2;   
// stage-3 sum and carry
    wire [0:9]s3,c3;
// stage-4 sum and carry
    wire [0:11]s4,c4;
// stage-5 sum and carry
    wire [0:13]s5,c5;




// generating partial products 
integer i;
integer j;
always @( *) begin
    for(i = 0; i<8; i=i+1)begin
        for(j = 0; j<8;j = j+1)begin
            gen_pp[i][j] = A[j]*B[i];
        end
    end
end


 

//Reduction by stages.
// di_values = 2,3,4,6,8,13...


//Stage 1 - reducing fom 8 to 6  


    HA h1(.a(gen_pp[6][0]),.b(gen_pp[5][1]),.Sum(s1[0]),.Cout(c1[0]));
    HA h2(.a(gen_pp[4][3]),.b(gen_pp[3][4]),.Sum(s1[2]),.Cout(c1[2]));
    HA h3(.a(gen_pp[4][4]),.b(gen_pp[3][5]),.Sum(s1[4]),.Cout(c1[4]));

    csa_dadda c11(.A(gen_pp[7][0]),.B(gen_pp[6][1]),.Cin(gen_pp[5][2]),.Y(s1[1]),.Cout(c1[1]));
    csa_dadda c12(.A(gen_pp[7][1]),.B(gen_pp[6][2]),.Cin(gen_pp[5][3]),.Y(s1[3]),.Cout(c1[3]));     
    csa_dadda c13(.A(gen_pp[7][2]),.B(gen_pp[6][3]),.Cin(gen_pp[5][4]),.Y(s1[5]),.Cout(c1[5]));
    
//Stage 2 - reducing fom 6 to 4

    HA h4(.a(gen_pp[4][0]),.b(gen_pp[3][1]),.Sum(s2[0]),.Cout(c2[0]));
    HA h5(.a(gen_pp[2][3]),.b(gen_pp[1][4]),.Sum(s2[2]),.Cout(c2[2]));


    csa_dadda c21(.A(gen_pp[5][0]),.B(gen_pp[4][1]),.Cin(gen_pp[3][2]),.Y(s2[1]),.Cout(c2[1]));
    csa_dadda c22(.A(s1[0]),.B(gen_pp[4][2]),.Cin(gen_pp[3][3]),.Y(s2[3]),.Cout(c2[3]));
    csa_dadda c23(.A(gen_pp[2][4]),.B(gen_pp[1][5]),.Cin(gen_pp[0][6]),.Y(s2[4]),.Cout(c2[4]));
    csa_dadda c24(.A(s1[1]),.B(s1[2]),.Cin(c1[0]),.Y(s2[5]),.Cout(c2[5]));
    csa_dadda c25(.A(gen_pp[2][5]),.B(gen_pp[1][6]),.Cin(gen_pp[0][7]),.Y(s2[6]),.Cout(c2[6]));
    csa_dadda c26(.A(s1[3]),.B(s1[4]),.Cin(c1[1]),.Y(s2[7]),.Cout(c2[7]));
    csa_dadda c27(.A(c1[2]),.B(gen_pp[2][6]),.Cin(gen_pp[1][7]),.Y(s2[8]),.Cout(c2[8]));
    csa_dadda c28(.A(s1[5]),.B(c1[3]),.Cin(c1[4]),.Y(s2[9]),.Cout(c2[9]));
    csa_dadda c29(.A(gen_pp[4][5]),.B(gen_pp[3][6]),.Cin(gen_pp[2][7]),.Y(s2[10]),.Cout(c2[10]));
    csa_dadda c210(.A(gen_pp[7][3]),.B(c1[5]),.Cin(gen_pp[6][4]),.Y(s2[11]),.Cout(c2[11]));
    csa_dadda c211(.A(gen_pp[5][5]),.B(gen_pp[4][6]),.Cin(gen_pp[3][7]),.Y(s2[12]),.Cout(c2[12]));
    csa_dadda c212(.A(gen_pp[7][4]),.B(gen_pp[6][5]),.Cin(gen_pp[5][6]),.Y(s2[13]),.Cout(c2[13]));
    
//Stage 3 - reducing fom 4 to 3

    HA h6(.a(gen_pp[3][0]),.b(gen_pp[2][1]),.Sum(s3[0]),.Cout(c3[0]));

    csa_dadda c31(.A(s2[0]),.B(gen_pp[2][2]),.Cin(gen_pp[1][3]),.Y(s3[1]),.Cout(c3[1]));
    csa_dadda c32(.A(s2[1]),.B(s2[2]),.Cin(c2[0]),.Y(s3[2]),.Cout(c3[2]));
    csa_dadda c33(.A(c2[1]),.B(c2[2]),.Cin(s2[3]),.Y(s3[3]),.Cout(c3[3]));
    csa_dadda c34(.A(c2[3]),.B(c2[4]),.Cin(s2[5]),.Y(s3[4]),.Cout(c3[4]));
    csa_dadda c35(.A(c2[5]),.B(c2[6]),.Cin(s2[7]),.Y(s3[5]),.Cout(c3[5]));
    csa_dadda c36(.A(c2[7]),.B(c2[8]),.Cin(s2[9]),.Y(s3[6]),.Cout(c3[6]));
    csa_dadda c37(.A(c2[9]),.B(c2[10]),.Cin(s2[11]),.Y(s3[7]),.Cout(c3[7]));
    csa_dadda c38(.A(c2[11]),.B(c2[12]),.Cin(s2[13]),.Y(s3[8]),.Cout(c3[8]));
    csa_dadda c39(.A(gen_pp[7][5]),.B(gen_pp[6][6]),.Cin(gen_pp[5][7]),.Y(s3[9]),.Cout(c3[9]));

//Stage 4 - reducing fom 3 to 2

    HA h7(.a(gen_pp[2][0]),.b(gen_pp[1][1]),.Sum(s4[0]),.Cout(c4[0]));


    csa_dadda c41(.A(s3[0]),.B(gen_pp[1][2]),.Cin(gen_pp[0][3]),.Y(s4[1]),.Cout(c4[1]));
    csa_dadda c42(.A(c3[0]),.B(s3[1]),.Cin(gen_pp[0][4]),.Y(s4[2]),.Cout(c4[2]));
    csa_dadda c43(.A(c3[1]),.B(s3[2]),.Cin(gen_pp[0][5]),.Y(s4[3]),.Cout(c4[3]));
    csa_dadda c44(.A(c3[2]),.B(s3[3]),.Cin(s2[4]),.Y(s4[4]),.Cout(c4[4]));
    csa_dadda c45(.A(c3[3]),.B(s3[4]),.Cin(s2[6]),.Y(s4[5]),.Cout(c4[5]));
    csa_dadda c46(.A(c3[4]),.B(s3[5]),.Cin(s2[8]),.Y(s4[6]),.Cout(c4[6]));
    csa_dadda c47(.A(c3[5]),.B(s3[6]),.Cin(s2[10]),.Y(s4[7]),.Cout(c4[7]));
    csa_dadda c48(.A(c3[6]),.B(s3[7]),.Cin(s2[12]),.Y(s4[8]),.Cout(c4[8]));
    csa_dadda c49(.A(c3[7]),.B(s3[8]),.Cin(gen_pp[4][7]),.Y(s4[9]),.Cout(c4[9]));
    csa_dadda c410(.A(c3[8]),.B(s3[9]),.Cin(c2[13]),.Y(s4[10]),.Cout(c4[10]));
    csa_dadda c411(.A(c3[9]),.B(gen_pp[7][6]),.Cin(gen_pp[6][7]),.Y(s4[11]),.Cout(c4[11]));
    
//Stage 5 - reducing fom 2 to 1
    // adding total sum and carry to get final output

    HA h8(.a(gen_pp[1][0]),.b(gen_pp[0][1]),.Sum(y[1]),.Cout(c5[0]));



    csa_dadda c51(.A(s4[0]),.B(gen_pp[0][2]),.Cin(c5[0]),.Y(y[2]),.Cout(c5[1]));
    csa_dadda c52(.A(c4[0]),.B(s4[1]),.Cin(c5[1]),.Y(y[3]),.Cout(c5[2]));
    csa_dadda c54(.A(c4[1]),.B(s4[2]),.Cin(c5[2]),.Y(y[4]),.Cout(c5[3]));
    csa_dadda c55(.A(c4[2]),.B(s4[3]),.Cin(c5[3]),.Y(y[5]),.Cout(c5[4]));
    csa_dadda c56(.A(c4[3]),.B(s4[4]),.Cin(c5[4]),.Y(y[6]),.Cout(c5[5]));
    csa_dadda c57(.A(c4[4]),.B(s4[5]),.Cin(c5[5]),.Y(y[7]),.Cout(c5[6]));
    csa_dadda c58(.A(c4[5]),.B(s4[6]),.Cin(c5[6]),.Y(y[8]),.Cout(c5[7]));
    csa_dadda c59(.A(c4[6]),.B(s4[7]),.Cin(c5[7]),.Y(y[9]),.Cout(c5[8]));
    csa_dadda c510(.A(c4[7]),.B(s4[8]),.Cin(c5[8]),.Y(y[10]),.Cout(c5[9]));
    csa_dadda c511(.A(c4[8]),.B(s4[9]),.Cin(c5[9]),.Y(y[11]),.Cout(c5[10]));
    csa_dadda c512(.A(c4[9]),.B(s4[10]),.Cin(c5[10]),.Y(y[12]),.Cout(c5[11]));
    csa_dadda c513(.A(c4[10]),.B(s4[11]),.Cin(c5[11]),.Y(y[13]),.Cout(c5[12]));
    csa_dadda c514(.A(c4[11]),.B(gen_pp[7][7]),.Cin(c5[12]),.Y(y[14]),.Cout(c5[13]));

    assign y[0] =  gen_pp[0][0];
    assign y[15] = c5[13];
endmodule 

module csa_dadda(A,B,Cin,Y,Cout);
input A,B,Cin;
output Y,Cout;
    
assign Y = A^B^Cin;
assign Cout = (A&B)|(A&Cin)|(B&Cin);
    
endmodule

module HA(a, b, Sum, Cout);

input a, b; // a and b are inputs with size 1-bit
output Sum, Cout; // Sum and Cout are outputs with size 1-bit

assign Sum = a ^ b; 
assign Cout = a & b; 

endmodule