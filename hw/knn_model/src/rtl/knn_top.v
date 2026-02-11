`timescale 1ns / 1ps
//----------------------+-------------------------------------------------------
// Filename             | knn_top.sv
// File created on      | 05 Feb 2026
// Created by           | Divine A Mathew
//                      |
//                      |
//----------------------+-------------------------------------------------------
//
//------------------------------------------------------------------------------
// KNN Accelerator Top Module
//------------------------------------------------------------------------------



//------------------------------------------------------------------------------
// Top Module Declration
//------------------------------------------------------------------------------
module knn_top #(
    parameter DATA_WIDTH = 8,
    parameter FEATURES   = 8,
    parameter K          = 5
    )(
    input                           clk                 ,
    input                           rst_n               ,
    input [DATA_WIDTH*FEATURES-1:0] train_data          ,
    input                           train_label         ,
    input                           data_valid          ,
    input                           training_done       ,
    input [DATA_WIDTH*FEATURES-1:0] test_data           ,
    output                          predicted_valid_o   ,
    output                          predicted_label_o
    );




//------------------------------------------------------------------------------
//  KNN Distance Calculation Module Instantiation
//------------------------------------------------------------------------------

    wire [DATA_WIDTH-1:0] distance_cal;
    wire label;

    knn_distance_cal #(
        .DATA_WIDTH (DATA_WIDTH     ),
        .FEATURES   (FEATURES       ),
        .K          (K              )
    ) knn_distance_cal_inst (
        .clk         (clk           ),
        .rst_n       (rst_n         ),
        .train_data  (train_data    ),
        .train_label (train_label   ),
        .data_valid  (data_valid    ),
        .test_data   (test_data     ),
        .distance_o  (distance_cal  ),
        .label_o     (label         )
        );



//------------------------------------------------------------------------------
//  KNN Distance Comparator Module Instantiation
//------------------------------------------------------------------------------
    knn_distance_comporator #(
        .DATA_WIDTH         (DATA_WIDTH     ),
        .K                  (K              )
    ) knn_distance_comparator_inst (
        .clk                (clk                    ),
        .rst_n              (rst_n                  ),
        .distance_i         (distance_cal           ),
        .label_i            (label                  ),
        .data_valid         (data_valid             ),
        .training_done      (training_done          ),
        .valid_o            (predicted_valid_o      ),
        .label_o            (predicted_label_o      )
        );



endmodule
