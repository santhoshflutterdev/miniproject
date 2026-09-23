import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)
from sklearn.neural_network import MLPRegressor


def create_sequences(data, sequence_length):
    X = []
    y = []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)


def predict_workload_demands(df, sequence_length=3, epochs=100, batch_size=1, verbose=0):
    """
    Trains a neural network workload prediction model (MLPRegressor / LSTM)
    on the workload dataframe and returns predicted demands matching each task.
    """
    workload = df["gpu_demand"].values.astype(float)
    scaler = MinMaxScaler(feature_range=(0, 1))
    workload_scaled = scaler.fit_transform(workload.reshape(-1, 1))

    if len(workload_scaled) <= sequence_length:
        return workload.tolist()

    X, y = create_sequences(workload_scaled, sequence_length)
    X_flat = X.reshape(X.shape[0], X.shape[1])

    # Default to MLPRegressor on macOS to prevent native C++ OpenMP/LibreSSL mutex lock issues
    use_tf = os.environ.get("USE_TF", "0") == "1" and sys.platform != "darwin"
    preds_actual = None

    if use_tf:
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense

            model = Sequential([
                LSTM(50, activation="tanh", input_shape=(sequence_length, 1)),
                Dense(1)
            ])
            model.compile(optimizer="adam", loss="mse")
            model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=verbose)

            preds_scaled = model.predict(X, verbose=0)
            preds_actual = scaler.inverse_transform(preds_scaled).flatten()
        except Exception:
            preds_actual = None

    if preds_actual is None:
        model = MLPRegressor(hidden_layer_sizes=(50,), max_iter=500, random_state=42)
        model.fit(X_flat, y.ravel())
        preds_scaled = model.predict(X_flat)
        preds_actual = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()

    full_predictions = list(workload[:sequence_length]) + list(preds_actual)
    return full_predictions


def main_run():
    df = pd.read_csv("dataset/workload.csv")
    print("\nOriginal Workload Dataset:")
    print(df)

    predictions = predict_workload_demands(df)
    print("\nPredicted Demands for Tasks:")
    for task_id, orig, pred in zip(df["task_id"], df["gpu_demand"], predictions):
        print(f"Task {task_id}: Actual={orig}% | Predicted={pred:.2f}%")


if __name__ == "__main__":
    main_run()