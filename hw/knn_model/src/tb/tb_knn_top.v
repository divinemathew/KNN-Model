`timescale 1ns / 1ps
//----------------------+-------------------------------------------------------
// Filename             | tb_knn_top.sv
// File created on      | 05 Feb 2026
// Created by           | Divine A Mathew
//                      |
//                      |
//----------------------+-------------------------------------------------------
//
//------------------------------------------------------------------------------
// Linear Testbench for KNN Accelerator Top Module
//------------------------------------------------------------------------------
module tb_knn_top();

    parameter DATA_WIDTH = 8;
    parameter FEATURES   = 8;
    parameter K          = 5;
    parameter NUM_SAMPLES = 614; 
    parameter NUM_TESTS   = 154;

    reg clk, rst_n, data_valid, training_done;
    reg [DATA_WIDTH*FEATURES-1:0] train_data, test_data;
    reg train_label;
    wire out_valid, out_label;

    // Memories
    reg [71:0] mem_array [0:NUM_SAMPLES-1];
    reg [71:0] test_mem_array [0:NUM_TESTS-1]; // Holds test features + expected label

    // Instantiate DUT
    knn_top#(
        DATA_WIDTH, FEATURES, K)
        dut (
        .clk                (clk                ),
        .rst_n              (rst_n              ),
        .train_data         (train_data         ), 
        .train_label        (train_label        ),
        .data_valid         (data_valid         ),
        .training_done      (training_done      ),
        .test_data          (test_data          ),
        .predicted_valid_o  (out_valid          ),
        .predicted_label_o  (out_label          )
    );



    always #5 clk = ~clk;

    integer i, t;
    reg expected_label;

    initial begin
        //Load both files
        $readmemh("D:/Mtech/Hardware For AI/Assignment-1/knn_model/src/tb/train_data.mem", mem_array);
        $readmemh("D:/Mtech/Hardware For AI/Assignment-1/knn_model/src/tb/test_data.mem", test_mem_array);
        
        clk = 0;
        rst_n = 0;
        data_valid = 0;
        training_done = 0;
        #100 rst_n = 1;

        //Outer Loop: Iterate through each test case
        for (t = 0; t < NUM_TESTS; t = t + 1) begin
            
            // Set current test point and extract ground truth (last bit)
            test_data      = test_mem_array[t][71:8];
            expected_label = test_mem_array[t][0];
            
            $display("\n--- Testing Sample %0d ---", t);
            $display("Test Vector: %h | Expected: %b", test_data, expected_label);

            //Inner Loop: Stream ALL training data for THIS test point
            data_valid = 1;
            training_done = 0;
            
            for (i = 0; i < NUM_SAMPLES; i = i + 1) begin
                train_data  = mem_array[i][71:8];
                train_label = mem_array[i][0];
                @(posedge clk);
            end
            
            data_valid = 0;
            training_done = 1; // Signal DUT to finalize calculation

            // 4. Wait for DUT to finish and validate
            wait(out_valid);
            
            if (out_label === expected_label)
                $display("RESULT: PASS | Predicted: %b", out_label);
            else
                $display("RESULT: FAIL | Predicted: %b (Expected: %b)", out_label, expected_label);

            // Reset training_done for next sample
            @(posedge clk);
            training_done = 0; 
            #50; // Small delay between test cases
        end

        $display("\nAll tests completed.");
        $finish;
    end

endmodule