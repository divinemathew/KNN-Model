import random
import math
import time

# --- 1. Load & 2. Normalize (Keeping your existing logic) ---
def load_csv(filename):
    dataset = []
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
            for line in lines[1:]:
                if not line.strip(): continue
                row = [float(x) for x in line.strip().split(',')]
                dataset.append(row)
    except FileNotFoundError:
        print("Error: File not found.")
        return []
    return dataset

def min_max_normalize(dataset):
    if not dataset: return []
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

# --- 3. Distance Metrics ---

def manhattan_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1) - 1):
        distance += abs(row1[i] - row2[i])
    return distance

def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1) - 1):
        distance += (row1[i] - row2[i])**2
    return math.sqrt(distance)

def chebyshev_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1) - 1):
        distance = max(distance, abs(row1[i] - row2[i]))
    return distance

# --- 4. Find Neighbors & 5. Predict (Adaptive) ---

def get_neighbors(train_data, test_row, k, distance_fn):
    distances = []
    for train_row in train_data:
        dist = distance_fn(test_row, train_row)
        distances.append((train_row, dist))
    
    distances.sort(key=lambda tup: tup[1])
    return [distances[i][0] for i in range(k)]

def predict(train_data, test_row, k, distance_fn):
    neighbors = get_neighbors(train_data, test_row, k, distance_fn)
    output_values = [row[-1] for row in neighbors]
    return max(set(output_values), key=output_values.count)

# --- Main Execution ---
def main():
    filename = './data/diabetes_int8_all.csv'
    dataset = load_csv(filename)
    if not dataset: return
    
    dataset = min_max_normalize(dataset)
    
    random.seed(42)
    random.shuffle(dataset)
    split_index = int(len(dataset) * 0.8)
    train_set = dataset[:split_index]
    test_set = dataset[split_index:]
    
    k = 5
    metrics = {
        "Manhattan": manhattan_distance,
        "Euclidean": euclidean_distance,
        "Chebyshev": chebyshev_distance
    }
    
    print(f"Evaluating {len(test_set)} samples with k={k}...\n")
    # Updated header to include CPU Time
    print(f"{'Metric':<15} | {'Accuracy':<10} | {'CPU Time (s)':<12}")
    print("-" * 45)

    for name, func in metrics.items():
        # --- Start CPU Timing ---
        start_cpu = time.process_time()
        
        correct = 0
        for row in test_set:
            prediction = predict(train_set, row, k, func)
            if prediction == row[-1]:
                correct += 1
        
        # --- End CPU Timing ---
        end_cpu = time.process_time()
        
        cpu_duration = end_cpu - start_cpu
        accuracy = (correct / len(test_set)) * 100
        
        print(f"{name:<15} | {accuracy:>8.2f}% | {cpu_duration:>12.4f}")

if __name__ == "__main__":
    main()