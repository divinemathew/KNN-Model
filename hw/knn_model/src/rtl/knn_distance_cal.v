
//----------------------+-------------------------------------------------------
// Filename             | knn_distance_cal.sv
// File created on      | 05 Feb 2026
// Created by           | Divine A Mathew
//                      |
//                      |
//----------------------+-------------------------------------------------------
//
//------------------------------------------------------------------------------
// KNN Distance Calculator Module
//------------------------------------------------------------------------------


//`define MANHATTAN
// `define EUCLIDEAN
  `define CHEBYSHEV

module knn_distance_cal #(
    parameter DATA_WIDTH = 8,
    parameter FEATURES   = 8,
    parameter K          = 5
    )
    (
    input                           clk,
    input                           rst_n,
    input [DATA_WIDTH*FEATURES-1:0] train_data,
    input                           train_label,
    input                           data_valid,

    input [DATA_WIDTH*FEATURES-1:0] test_data,

    output [DATA_WIDTH-1:0]         distance_o,
    output                          label_o
    );



    reg [DATA_WIDTH-1:0]    train_data_feature  [0:FEATURES-1];
    reg [DATA_WIDTH-1:0]    test_data_feature   [0:FEATURES-1];
    reg [DATA_WIDTH-1:0]    distance            [0:FEATURES-1];
    reg [DATA_WIDTH*2-1:0]  distance_cal                      ;





//------------------------------------------------------------------------------
// Spliting Input Data into Features
//------------------------------------------------------------------------------

    integer i;
    reg [ (2*DATA_WIDTH)-1 : 0] diff;


    always @(*)
        begin
        distance_cal = 0;
        if(data_valid & rst_n)
            begin
            // 1. Calculate Individual Feature Distances
            for (i = 0; i < FEATURES; i = i + 1)
                begin
                train_data_feature[i] = train_data[i * DATA_WIDTH +: DATA_WIDTH];
                test_data_feature [i] = test_data [i * DATA_WIDTH +: DATA_WIDTH];
                // Absolute Difference Calculation
                if (train_data_feature[i] > test_data_feature[i])
                    diff = train_data_feature[i] - test_data_feature[i];
                else
                    diff = test_data_feature[i] - train_data_feature[i];
                // 2. Distance Metric Calculation
                `ifdef MANHATTAN
                    distance[i]     = diff;
                `elsif EUCLIDEAN
                    distance[i]     = diff * diff;
                `elsif CHEBYSHEV
                    distance[i]     = diff;
                `endif
                end

            // Final Aggregation Logic
            `ifdef CHEBYSHEV
                // Chebyshev is the Maximum of the absolute differences
                distance_cal = distance[0];
                for (i = 1; i < FEATURES; i = i + 1)
                    begin
                    if (distance[i] > distance_cal)
                        distance_cal = distance[i];
                    end
            `else
                // Manhattan and Euclidean use the Sum of elements
                distance_cal = distance[0] + distance[1] + distance[2] + distance[3] + 
                               distance[4] + distance[5] + distance[6] + distance[7];
            `endif
            end
        end


//------------------------------------------------------------------------------
// Finding Total Distance
//------------------------------------------------------------------------------

    assign distance_o = distance_cal;
    assign label_o    = train_label;



endmodule