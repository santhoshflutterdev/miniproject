import torch
import torch.nn as nn

class LSTMWorkloadNet(nn.Module):
    """
    PyTorch sequence-to-one LSTM architecture for GPU workload prediction.
    Input shape: (batch_size, sequence_length, input_features)
    Output shape: (batch_size, 1) -> predicted GPU demand / utilization
    """
    def __init__(self, input_size: int = 1, hidden_size: int = 64, num_layers: int = 2, output_size: int = 1, dropout: float = 0.1):
        super(LSTMWorkloadNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        out, _ = self.lstm(x)
        # Take the output of the last time step
        out = out[:, -1, :]
        out = self.fc(out)
        return out
