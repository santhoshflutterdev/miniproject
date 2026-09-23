import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import logging
from typing import Tuple, List, Dict, Any, Optional
from sklearn.preprocessing import MinMaxScaler
from models.lstm_model import LSTMWorkloadNet

logger = logging.getLogger("WorkloadPredictor")

class WorkloadPredictor:
    """
    Sequence-based LSTM Workload Predictor for forecasting future GPU workload demands.
    """
    def __init__(self, sequence_length: int = 10, hidden_dim: int = 64, num_layers: int = 2):
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = LSTMWorkloadNet(input_size=1, hidden_size=hidden_dim, num_layers=num_layers, output_size=1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.is_trained = False
        self.last_mae = 0.0
        self.last_rmse = 0.0

    def prepare_data(self, series: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Converts 1D time-series data into sequence windows (X, y).
        """
        scaled_data = self.scaler.fit_transform(series.reshape(-1, 1))
        X, y = [], []
        for i in range(len(scaled_data) - self.sequence_length):
            X.append(scaled_data[i : i + self.sequence_length])
            y.append(scaled_data[i + self.sequence_length])
        
        if len(X) == 0:
            # Fallback if series is shorter than sequence length
            X = np.zeros((1, self.sequence_length, 1))
            y = np.zeros((1, 1))
            return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

        return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.float32)

    def train(self, df: pd.DataFrame, feature_col: str = "gpu_demand", epochs: int = 30, lr: float = 0.005, batch_size: int = 32) -> Dict[str, float]:
        """
        Trains the PyTorch LSTM workload prediction model on historical trace data.
        """
        if feature_col not in df.columns:
            logger.warning(f"Feature column '{feature_col}' not found. Defaulting to synthetic values.")
            series = np.random.uniform(20, 80, size=100)
        else:
            series = df[feature_col].astype(float).values

        X, y = self.prepare_data(series)
        X, y = X.to(self.device), y.to(self.device)

        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.model.train()
        for epoch in range(epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                predictions = self.model(batch_x)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()

        self.is_trained = True
        
        # Evaluate model error
        eval_metrics = self.evaluate(series)
        logger.info(f"Model trained successfully. MAE: {eval_metrics['mae']:.4f}, RMSE: {eval_metrics['rmse']:.4f}")
        return eval_metrics

    def predict(self, recent_sequence: List[float]) -> float:
        """
        Predicts the GPU demand for the next time window based on recent sequence.
        """
        if not self.is_trained:
            # Heuristic default if model not trained yet
            return float(np.mean(recent_sequence)) if recent_sequence else 50.0

        if len(recent_sequence) < self.sequence_length:
            # Pad sequence if needed
            pad_val = recent_sequence[0] if recent_sequence else 30.0
            recent_sequence = [pad_val] * (self.sequence_length - len(recent_sequence)) + list(recent_sequence)
        
        seq_arr = np.array(recent_sequence[-self.sequence_length:]).reshape(-1, 1)
        scaled_seq = self.scaler.transform(seq_arr)
        input_tensor = torch.tensor(scaled_seq, dtype=torch.float32).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            pred_scaled = self.model(input_tensor).cpu().numpy()

        pred_unscaled = self.scaler.inverse_transform(pred_scaled)[0][0]
        # Clamp prediction to valid GPU % range [0, 100]
        return float(np.clip(pred_unscaled, 0.0, 100.0))

    def evaluate(self, series: np.ndarray) -> Dict[str, float]:
        """
        Computes MAE and RMSE prediction accuracy on input series.
        """
        X, y = self.prepare_data(series)
        X, y = X.to(self.device), y.to(self.device)
        self.model.eval()
        with torch.no_grad():
            preds_scaled = self.model(X).cpu().numpy()
            y_scaled = y.cpu().numpy()

        preds_unscaled = self.scaler.inverse_transform(preds_scaled)
        y_unscaled = self.scaler.inverse_transform(y_scaled)

        mae = float(np.mean(np.abs(preds_unscaled - y_unscaled)))
        rmse = float(np.sqrt(np.mean((preds_unscaled - y_unscaled) ** 2)))
        self.last_mae = mae
        self.last_rmse = rmse
        return {"mae": mae, "rmse": rmse}

    def save_model(self, file_path: str = "models/lstm_workload.pt"):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "scaler": self.scaler,
            "sequence_length": self.sequence_length,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "last_mae": self.last_mae,
            "last_rmse": self.last_rmse
        }, file_path)
        logger.info(f"Saved WorkloadPredictor model to {file_path}")

    def load_model(self, file_path: str = "models/lstm_workload.pt") -> bool:
        if not os.path.exists(file_path):
            return False
        try:
            checkpoint = torch.load(file_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.scaler = checkpoint["scaler"]
            self.sequence_length = checkpoint.get("sequence_length", 10)
            self.last_mae = checkpoint.get("last_mae", 0.0)
            self.last_rmse = checkpoint.get("last_rmse", 0.0)
            self.is_trained = True
            logger.info(f"Loaded WorkloadPredictor model from {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model from {file_path}: {e}")
            return False
