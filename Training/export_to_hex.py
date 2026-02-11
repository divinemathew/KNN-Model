def load_int_data(filename):
    dataset = []
    try:
        with open(filename, 'r') as file:
            for line in file:
                if not line.strip(): continue
                row = [int(x) for x in line.strip().split(',')]
                dataset.append(row)
        return dataset
    except FileNotFoundError:
        print(f"Error: Could not find {filename}. Run the previous step first.")
        return []

def save_as_hex_mem(dataset, filename, data_width_bits=8):
    """
    Saves data in a compact hex format (no spaces).
    Each line in the file corresponds to one row of data.
    """
    with open(filename, 'w') as f:
        for row in dataset:
            hex_row = []
            for val in row:
                # Format as 2-digit Hex (e.g., 255 -> FF)
                hex_val = f"{val:02X}" 
                hex_row.append(hex_val)
            
            # Join with an empty string for a compact format: "FF0A1B..."
            line = "".join(hex_row)
            f.write(line + "\n")
            
    print(f"Exported {len(dataset)} rows to {filename}")

def main():
    print("--- Exporting to Hex for Verilog ---")
    
    # 1. Load the Integer Data
    train_data = load_int_data('fpga_train.txt')
    test_data = load_int_data('fpga_test.txt')

    if not train_data: return

    # 2. Export to .mem files
    save_as_hex_mem(train_data, 'train_data.mem')
    save_as_hex_mem(test_data, 'test_data.mem')
    
    print("\nSUCCESS! Files created:")
    print("1. train_data.mem (Load into Train BRAM)")
    print("2. test_data.mem  (Load into Test Bench)")

if __name__ == "__main__":
    main()