import os
import random
import pandas as pd
import numpy as np

def generate_gpu_config(output_path: str = "data/gpu_config.csv") -> pd.DataFrame:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    gpus_data = [
        {
            "gpu_id": "GPU1",
            "name": "NVIDIA A100-80GB (US-East)",
            "total_gpu_capacity": 100.0,
            "total_memory": 81920.0,
            "power_watts": 350.0,
            "carbon_intensity": 600.0,  # High carbon regional grid (gCO2/kWh)
            "renewable_ratio": 0.15,
            "energy_price": 0.14,
            "location": "us-east-1"
        },
        {
            "gpu_id": "GPU2",
            "name": "NVIDIA V100-32GB (US-West)",
            "total_gpu_capacity": 100.0,
            "total_memory": 32768.0,
            "power_watts": 300.0,
            "carbon_intensity": 300.0,  # Moderate carbon intensity
            "renewable_ratio": 0.45,
            "energy_price": 0.11,
            "location": "us-west-2"
        },
        {
            "gpu_id": "GPU3",
            "name": "NVIDIA A10G-24GB (EU-Nordic)",
            "total_gpu_capacity": 100.0,
            "total_memory": 24576.0,
            "power_watts": 250.0,
            "carbon_intensity": 100.0,  # Low carbon hydro/wind grid
            "renewable_ratio": 0.85,
            "energy_price": 0.08,
            "location": "eu-north-1"
        },
        {
            "gpu_id": "GPU4",
            "name": "NVIDIA RTX4090-24GB (Asia-East)",
            "total_gpu_capacity": 100.0,
            "total_memory": 24576.0,
            "power_watts": 450.0,
            "carbon_intensity": 500.0,
            "renewable_ratio": 0.25,
            "energy_price": 0.13,
            "location": "ap-east-1"
        }
    ]
    df = pd.DataFrame(gpus_data)
    df.to_csv(output_path, index=False)
    print(f"Generated GPU configuration at {output_path}")
    return df

def generate_sample_tasks(num_tasks: int = 100, output_path: str = "data/sample_tasks.csv") -> pd.DataFrame:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    random.seed(42)
    np.random.seed(42)

    users = [f"U{i:03d}" for i in range(1, 11)]
    priorities = ["low", "medium", "high"]
    isolation_levels = ["soft", "strong"]
    carbon_prefs = ["low", "standard", "high"]

    tasks = []
    current_arrival = 0.0

    for i in range(1, num_tasks + 1):
        # Inter-arrival time exponential distribution
        inter_arrival = round(float(np.random.exponential(scale=1.5)), 1)
        current_arrival += inter_arrival
        
        gpu_demand = float(random.choice([10.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 75.0, 100.0]))
        memory_demand = float(random.choice([2048.0, 4096.0, 8192.0, 12288.0, 16384.0, 24576.0, 32768.0]))
        duration = float(random.randint(5, 45))
        deadline = round(current_arrival + duration + float(random.randint(10, 60)), 1)

        tasks.append({
            "task_id": f"T{i:03d}",
            "user_id": random.choice(users),
            "arrival_time": round(current_arrival, 1),
            "gpu_demand": gpu_demand,
            "memory_demand": memory_demand,
            "duration": duration,
            "priority": random.choice(priorities),
            "deadline": deadline,
            "isolation_level": random.choice(isolation_levels),
            "carbon_preference": random.choice(carbon_prefs)
        })

    df = pd.DataFrame(tasks)
    df.to_csv(output_path, index=False)
    print(f"Generated {num_tasks} synthetic sample tasks at {output_path}")
    return df

if __name__ == "__main__":
    generate_gpu_config()
    generate_sample_tasks()
