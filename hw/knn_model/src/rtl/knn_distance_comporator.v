//----------------------+-------------------------------------------------------
// Filename             | knn_distance_comporator.sv
// File created on      | 05 Feb 2026
// Created by           | Divine A Mathew
//                      |
//                      |
//----------------------+-------------------------------------------------------
//
//------------------------------------------------------------------------------
// KNN Distance Comparator Module
//------------------------------------------------------------------------------



module knn_distance_comporator #(
    parameter DATA_WIDTH = 8,
    parameter K          = 5
    )
    (
    input                           clk             ,
    input                           rst_n           ,
    input [DATA_WIDTH-1:0]          distance_i      ,
    input                           label_i         ,
    input                           training_done   ,
    input                           data_valid      ,

    output                          valid_o         ,
    output                          label_o
    );


// Arrays to store the top K smallest distances and their labels
    reg [DATA_WIDTH-1:0] top_dist [0:K-1] ;
    reg                  top_label [0:K-1];

    reg                  out_valid;
    wire [3:0]           label_sum;
    reg  [3:0]           label_reg;

    wire                 training_posedge;
    reg                  training_done_d0;
    reg                  training_done_d1;
    reg                  training_done_d2;




// Delaying training_done signal to detect posedge
    always @(posedge clk )
        begin
        if (!rst_n) 
            begin
            training_done_d0 <= 1'b0;
            training_done_d1 <= 1'b0;
            training_done_d2 <= 1'b0;
            end
        else
            begin
            training_done_d0 <= training_done;
            training_done_d1 <= training_done_d0;
            training_done_d2 <= training_done_d1;
            end
        end

    assign training_posedge = training_done && ~training_done_d1;




//------------------------------------------------------------------------------
// Parallel Comparison and Insertion Logic
// Shift down and insert new distance if it's smaller than current top K distances
//------------------------------------------------------------------------------
    integer i, j;


    always @(posedge clk)
        begin
        if (!rst_n)
            begin
            out_valid <= 0;
            for (i = 0; i < K; i = i + 1)
                begin
                top_dist[i]  <= {DATA_WIDTH{1'b1}};
                top_label[i] <= 0;
                end
            end
            else if (data_valid && !training_done)
            begin
            // Slot 0 is unique: It only takes the new data or keeps its own
            if (distance_i < top_dist[0]) 
                begin
                top_dist[0]  <= distance_i;
                top_label[0] <= label_i;
                end
            // Slots 1 to K-1 make a 3-way decision
            for (i = 1; i < K; i = i + 1) 
                begin
                if (distance_i < top_dist[i-1])
                    begin
                    // Case A: New distance is better than the slot ABOVE me.
                    // I must take the value that was pushed out of the slot above.
                    top_dist[i]  <= top_dist[i-1];
                    top_label[i] <= top_label[i-1];
                    end
                else if (distance_i < top_dist[i])
                    begin
                    // Case B: New distance is better than ME, but not the guy above.
                    // I take the new data.
                    top_dist[i]  <= distance_i;
                    top_label[i] <= label_i;
                    end
                end
            end
            else if (training_done_d2) 
                begin
                out_valid <= 1;
                end 
            else if (out_valid) 
                begin
                // CRITICAL: Reset the registers ONLY AFTER out_valid is done
                out_valid <= 0;
            for (i = 0; i < K; i = i + 1)
                begin
                top_dist[i]  <= {DATA_WIDTH{1'b1}};
                top_label[i] <= 0;
                end
            end
        end

    assign label_sum = top_label[0] + top_label[1] + top_label[2] + top_label[3] + top_label[4];


// Latching the label sum after training is done
    always @(posedge clk )
        begin
            if (!rst_n)
                begin
                    label_reg <= 0;
                end
            else if (training_posedge)
                begin
                    label_reg <= label_sum;
                end
        end



    assign valid_o = out_valid;
    assign label_o = out_valid ? (label_reg > (K/2)) ? 1'b1 : 1'b0 : 1'b0;

endmodule