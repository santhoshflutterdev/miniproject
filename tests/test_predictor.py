import pytest
import numpy as np
import pandas as pd
from models.workload_predictor import WorkloadPredictor

def test_workload_predictor_data_prep():
    predictor = WorkloadPredictor(sequence_length=5)
    data = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0])
    X, y = predictor.prepare_data(data)
    assert X.shape[0] == 2
    assert X.shape[1] == 5

def test_workload_predictor_training():
    predictor = WorkloadPredictor(sequence_length=4)
    df = pd.DataFrame({"gpu_demand": [20, 25, 30, 35, 40, 45, 50, 55, 60, 65]})
    metrics = predictor.train(df, feature_col="gpu_demand", epochs=5, batch_size=2)
    assert "mae" in metrics
    assert "rmse" in metrics
    assert predictor.is_trained is True

    pred = predictor.predict([40.0, 45.0, 50.0, 55.0])
    assert 0.0 <= pred <= 100.0
