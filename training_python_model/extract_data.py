import random

# --- 1. Load Data ---
def load_csv(filename):
    dataset = []
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines[1:]:  # Skip header
            if not line.strip(): continue
            row = [float(x) for x in line.strip().split(',')]
            dataset.append(row)
    return dataset

# --- 2. Normalize (0.0 to 1.0) ---
def min_max_normalize(dataset):
    minmax = []
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        minmax.append([min(col_values), max(col_values)])
    
    for row in dataset:
        for i in range(len(row)):
            denom = minmax[i][1] - minmax[i][0]
            if denom == 0: row[i] = 0.0
            else: row[i] = (row[i] - minmax[i][0]) / denom
    return dataset

# --- 3. Quantize (Features -> 255, Label -> 0/1) ---
def quantize_dataset(dataset):
    quantized_data = []
    for row in dataset:
        new_row = []
        for i, val in enumerate(row):
            # CHECK: Is this the last column (the label)?
            if i == len(row) - 1:
                # Keep label as 0 or 1
                new_row.append(int(val)) 
            else:
                # Scale features to 0-255
                new_row.append(int(round(val * 255.0)))
        quantized_data.append(new_row)
    return quantized_data

# --- 4. Save to .txt ---
def save_data_to_file(dataset, filename):
    with open(filename, 'w') as f:
        for row in dataset:
            # Join integers with commas
            line = ",".join([str(x) for x in row])
            f.write(line + "\n")
    print(f"Saved {len(dataset)} rows to {filename}")

# --- Main ---
def main():
    filename = './data/diabetes_int8_all.csv' 
    try:
        dataset = load_csv(filename)
    except FileNotFoundError:
        print(f"Error: Could not find {filename}")
        return

    # 1. Normalize
    dataset = min_max_normalize(dataset)
    
    # 2. Quantize (Apply the fix here)
    dataset = quantize_dataset(dataset)

    # 3. Shuffle
    random.seed(42)
    random.shuffle(dataset)
    
    # 4. Split (80/20)
    split_index = int(len(dataset) * 0.8)
    train_set = dataset[:split_index]
    test_set = dataset[split_index:]
    
    # 5. Save
    save_data_to_file(train_set, 'fpga_train.txt')
    save_data_to_file(test_set, 'fpga_test.txt')
    
    print("\nData extracted successfully.")
    print("Features: Scaled 0-255")
    print("Labels:   Kept 0 or 1")

if __name__ == "__main__":
    main()