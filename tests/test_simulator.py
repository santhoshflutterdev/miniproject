import pytest
import pandas as pd
from backend.models import Task
from simulation.simulator import GPUSimulator

def test_simulator_execution():
    sim = GPUSimulator(scheduler_type="proposed")
    sim.reset()
    
    tasks_df = pd.DataFrame([
        {"task_id": "T001", "user_id": "U1", "arrival_time": 0.0, "gpu_demand": 30.0, "memory_demand": 4096.0, "duration": 5.0, "priority": "high", "deadline": 20.0, "isolation_level": "soft", "carbon_preference": "low"},
        {"task_id": "T002", "user_id": "U2", "arrival_time": 1.0, "gpu_demand": 40.0, "memory_demand": 8192.0, "duration": 10.0, "priority": "medium", "deadline": 30.0, "isolation_level": "strong", "carbon_preference": "standard"}
    ])
    
    sim.load_tasks_from_dataframe(tasks_df)
    assert len(sim.queued_tasks) == 2

    # Run simulation steps
    for _ in range(15):
        sim.step()

    metrics = sim.get_summary_metrics()
    assert metrics.total_tasks == 2
    assert metrics.completed_tasks > 0
    assert metrics.total_energy_kwh >= 0.0
    assert metrics.total_carbon_gco2 >= 0.0
