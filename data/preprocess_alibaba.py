import os
import random
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("AlibabaPreprocessor")

# Default Column Mapping for Alibaba GPU Trace v2020 / v2023
DEFAULT_COLUMN_MAPPING = {
    "task_name": "task_id",
    "job_name": "job_id",
    "user": "user_id",
    "start_time": "arrival_time",
    "plan_cpu": "cpu_demand",
    "plan_gpu": "gpu_demand",
    "plan_mem": "memory_demand",
    "duration": "duration"
}

def preprocess_alibaba_trace(
    input_file_path: str,
    output_file_path: str = "data/processed/alibaba_processed.csv",
    column_mapping: Optional[Dict[str, str]] = None,
    sample_size: Optional[int] = 5000
) -> pd.DataFrame:
    """
    Preprocesses raw Alibaba GPU cluster trace files (CSV/Parquet).
    Maps column schemas, normalizes timestamps, cleans missing values, and produces a scheduler-ready CSV.
    """
    if not os.path.exists(input_file_path):
        logger.warning(f"Alibaba input dataset file not found at {input_file_path}")
        return pd.DataFrame()

    logger.info(f"Loading raw Alibaba dataset from {input_file_path}...")
    if input_file_path.endswith(".parquet"):
        df = pd.read_parquet(input_file_path)
    else:
        df = pd.read_csv(input_file_path)

    mapping = column_mapping or DEFAULT_COLUMN_MAPPING
    
    # Rename mapped columns
    rename_dict = {k: v for k, v in mapping.items() if k in df.columns}
    df = df.rename(columns=rename_dict)

    logger.info(f"Mapped Alibaba columns: {list(rename_dict.keys())} -> {list(rename_dict.values())}")

    # Clean & Impute Missing Values
    if "gpu_demand" not in df.columns:
        df["gpu_demand"] = 25.0
    else:
        # Convert fractional Alibaba GPU demand (e.g. 0.5, 1.0, 2.0) to percentage 0-100%
        max_val = df["gpu_demand"].max()
        if max_val <= 8.0:
            df["gpu_demand"] = (df["gpu_demand"] / 8.0) * 100.0
        df["gpu_demand"] = df["gpu_demand"].fillna(25.0).clip(5.0, 100.0)

    if "memory_demand" not in df.columns:
        df["memory_demand"] = 8192.0
    else:
        df["memory_demand"] = df["memory_demand"].fillna(8192.0).astype(float)

    if "duration" not in df.columns:
        df["duration"] = 15.0
    else:
        # Convert seconds to minutes if values are large
        if df["duration"].median() > 100:
            df["duration"] = df["duration"] / 60.0
        df["duration"] = df["duration"].fillna(15.0).clip(1.0, 300.0)

    if "arrival_time" not in df.columns:
        df["arrival_time"] = np.arange(len(df)) * 0.5
    else:
        df["arrival_time"] = pd.to_numeric(df["arrival_time"], errors="coerce").fillna(0)
        min_time = df["arrival_time"].min()
        # Normalize relative arrival time in minutes
        df["arrival_time"] = ((df["arrival_time"] - min_time) / 60.0).round(1)

    if "user_id" not in df.columns:
        df["user_id"] = [f"U{random.randint(1, 10):03d}" for _ in range(len(df))]

    if "task_id" not in df.columns:
        df["task_id"] = [f"AliTask_{i:04d}" for i in range(1, len(df) + 1)]

    # Subsample if dataset exceeds specified sample_size
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42).sort_values("arrival_time")

    # Add Simulation Attributes (Required for carbon & isolation experiments)
    # Note: These simulated values supplement the trace data for research evaluation
    random.seed(42)
    priorities = ["low", "medium", "high"]
    isolation_levels = ["soft", "strong"]
    carbon_prefs = ["low", "standard", "high"]

    df["priority"] = [random.choice(priorities) for _ in range(len(df))]
    df["isolation_level"] = [random.choice(isolation_levels) for _ in range(len(df))]
    df["carbon_preference"] = [random.choice(carbon_prefs) for _ in range(len(df))]
    df["deadline"] = df["arrival_time"] + df["duration"] + [random.randint(10, 60) for _ in range(len(df))]

    output_cols = [
        "task_id", "user_id", "arrival_time", "gpu_demand", "memory_demand",
        "duration", "priority", "deadline", "isolation_level", "carbon_preference"
    ]
    processed_df = df[output_cols]

    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    processed_df.to_csv(output_file_path, index=False)
    logger.info(f"Successfully processed {len(processed_df)} Alibaba trace records to {output_file_path}")
    return processed_df

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys
    input_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw_alibaba.csv"
    preprocess_alibaba_trace(input_path)
