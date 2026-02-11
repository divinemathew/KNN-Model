# --- Helper: Load Integers ---
def load_int_data(filename):
    dataset = []
    try:
        with open(filename, 'r') as file:
            for line in file:
                if not line.strip(): continue
                # Read data as integers directly
                row = [int(x) for x in line.strip().split(',')]
                dataset.append(row)
        return dataset
    except FileNotFoundError:
        return None

# --- Distance Calculation (Integer Math) ---
def manhattan_distance(row1, row2):
    distance = 0
    # Iterate features only (stop before the last column)
    for i in range(len(row1) - 1):
        distance += abs(row1[i] - row2[i])
    return distance

# --- Neighbors ---
def get_neighbors(train_data, test_row, k):
    distances = []
    for train_row in train_data:
        dist = manhattan_distance(test_row, train_row)
        distances.append((train_row, dist))
    
    distances.sort(key=lambda tup: tup[1])
    
    neighbors = []
    for i in range(k):
        neighbors.append(distances[i][0])
    return neighbors

# --- Predict ---
def predict(train_data, test_row, k):
    neighbors = get_neighbors(train_data, test_row, k)
    # The last column is now guaranteed to be 0 or 1
    output_values = [row[-1] for row in neighbors]
    return max(set(output_values), key=output_values.count)

# --- Main Verification ---
def main():
    print("--- Verifying FPGA Data ---")
    
    # 1. Load the generated Int8 files
    train_set = load_int_data('fpga_train.txt')
    test_set = load_int_data('fpga_test.txt')
    
    if not train_set or not test_set:
        print("Error: Run Step 1 first to generate .txt files.")
        return

    # 2. Run Tests
    correct = 0
    k = 5
    
    # Prepare to save detailed comparison
    with open('verification_results.csv', 'w') as f:
        f.write("Test_ID,Actual_Label,Predicted_Label,Match\n")
        
        print(f"Testing {len(test_set)} samples...")
        
        for i, row in enumerate(test_set):
            prediction = predict(train_set, row, k)
            actual = row[-1] # This will be 0 or 1
            
            if prediction == actual:
                correct += 1
                match_str = "YES"
            else:
                match_str = "NO"
            
            # Save detail: ID, Actual, Predicted
            f.write(f"{i},{actual},{prediction},{match_str}\n")

    # 3. Final Report
    accuracy = (correct / len(test_set)) * 100
    print(f"---------------------------")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Detailed results saved to 'verification_results.csv'")
    print(f"---------------------------")
    print(f"Example Data Format (First Train Row): {train_set[0]}")

if __name__ == "__main__":
    main()