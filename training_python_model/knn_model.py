# KNN Model for Diabetes Prediction
# Created by: Divine A Mathew


# ============================================================
# k-NN FLOAT vs INT8 Quantized Comparison 
# ============================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# ------------------------------------------------------------
# 1. Load Dataset
# ------------------------------------------------------------
def load_dataset(path):
    df = pd.read_csv(path)
    print("\n===== DATASET LOADED =====")
    print(df.head())
    return df


# ------------------------------------------------------------
# 2. Quantize ALL Features to INT8
# ------------------------------------------------------------
def quantize_to_int8(df):
    df_q = df.copy()

    # Scaling factors chosen to preserve information
    scale_map = {
        "Pregnancies": 1,
        "Glucose": 1,
        "BloodPressure": 1,
        "SkinThickness": 1,
        "Insulin": 1,
        "BMI": 2,                       # float → int
        "DiabetesPedigreeFunction": 100, # float → int
        "Age": 1
    }

    for col, scale in scale_map.items():
        df_q[col] = (df_q[col] * scale).round().astype(np.int8)

    df_q["Outcome"] = df_q["Outcome"].astype(np.int8)

    print("\n===== DATA TYPES AFTER FULL INT8 QUANTIZATION =====")
    print(df_q.dtypes)

    return df_q


# ------------------------------------------------------------
# 3. Train & Evaluate k-NN Model
# ------------------------------------------------------------
def train_knn(df, label="FLOAT"):
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Scaling (can be removed later for pure integer hardware)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    knn = KNeighborsClassifier(
        n_neighbors=5,
        metric="euclidean"
    )

    knn.fit(X_train_s, y_train)
    y_pred = knn.predict(X_test_s)

    acc = accuracy_score(y_test, y_pred)

    print(f"\n===== {label} MODEL RESULT =====")
    print(f"Accuracy : {acc * 100:.2f} %")

    return acc



def extract_fpga_parameters(df_int8, k=5):
    """
    Extract training parameters for FPGA inference
    """

    X = df_int8.drop("Outcome", axis=1).values.astype(np.int8)
    y = df_int8["Outcome"].values.astype(np.int8)

    X_train, _, y_train, _ = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    params = {
        "X_train": X_train,
        "y_train": y_train,
        "num_train_samples": X_train.shape[0],
        "num_features": X_train.shape[1],
        "k": k
    }

    print("\n===== FPGA TRAINING PARAMETERS =====")
    print(f"Training samples : {params['num_train_samples']}")
    print(f"Number of features: {params['num_features']}")
    print(f"k value          : {params['k']}")
    print(f"Feature datatype : INT8")
    print(f"Label datatype   : INT8")

    return params


def extract_inference_data(df_int8, k=5):
    """
    Extract detailed k-NN inference data for FPGA verification
    """

    X = df_int8.drop("Outcome", axis=1).values.astype(np.int8)
    y = df_int8["Outcome"].values.astype(np.int8)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    inference_log = []

    for idx, x in enumerate(X_test):
        # L1 distance (hardware-friendly)
        distances = np.sum(np.abs(X_train - x), axis=1)

        # k nearest neighbors
        knn_idx = np.argsort(distances)[:k]
        knn_dist = distances[knn_idx]
        knn_labels = y_train[knn_idx]

        # Majority vote
        prediction = 1 if np.sum(knn_labels) >= (k // 2 + 1) else 0

        inference_log.append({
            "input_vector": x,
            "true_label": y_test[idx],
            "distances": distances,
            "knn_indices": knn_idx,
            "knn_distances": knn_dist,
            "knn_labels": knn_labels,
            "prediction": prediction
        })

    print("\n===== INFERENCE DATA EXTRACTED =====")
    print(f"Test samples : {len(inference_log)}")
    print(f"k value      : {k}")
    print("Distance     : L1 (Manhattan)")
    print("Datatype     : INT8")

    return inference_log


# ------------------------------------------------------------
# 4. Main Flow
# ------------------------------------------------------------
def main():
    file_path = "./data/diabetes.csv"   # change if needed

    # Load original dataset
    df_float = load_dataset(file_path)

    # FLOAT model
    float_acc = train_knn(df_float, label="FLOAT")

    # INT8 Quantization
    df_int8 = quantize_to_int8(df_float)

    # INT8 model
    int8_acc = train_knn(df_int8, label="INT8")
    # You can save fpga_params to a file if needed for FPGA inference
    fpga_params = extract_fpga_parameters(df_int8, k=5)
    inference_data = extract_inference_data(df_int8, k=5)



    # Comparison
    print("\n================ FINAL COMPARISON ================")
    print(f"Float Accuracy : {float_acc * 100:.2f} %")
    print(f"INT8 Accuracy  : {int8_acc * 100:.2f} %")
    print(f"Accuracy Drop  : {(float_acc - int8_acc) * 100:.2f} %")

    # np.savetxt(
    #     "X_test_int8.hex",
    #     [d["input_vector"] for d in inference_data],
    #     fmt="%d"
    # )
    
    # np.savetxt(
    # "y_pred_golden.hex",
    # [d["prediction"] for d in inference_data],
    # fmt="%d")
    
    # np.savetxt(
    # "knn_distances.hex",
    # np.vstack([d["knn_distances"] for d in inference_data]),
    # fmt="%d"
    # )
    
    # np.savetxt(
    # "knn_labels.hex",
    # np.vstack([d["knn_labels"] for d in inference_data]),
    # fmt="%d"
    # )



    



# ------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------
if __name__ == "__main__":
    main()
