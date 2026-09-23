from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class Task(BaseModel):
    task_id: str
    user_id: str
    arrival_time: float = 0.0
    gpu_demand: float = Field(..., ge=0, le=100, description="GPU compute requirement (%)")
    memory_demand: float = Field(..., ge=0, description="GPU memory requirement in MB")
    duration: float = Field(..., gt=0, description="Estimated duration in minutes")
    priority: str = "medium"  # low, medium, high
    deadline: float = 1000.0
    isolation_level: str = "soft"  # strong, soft
    carbon_preference: str = "standard"  # low, standard, high
    
    # Dynamic runtime status fields
    status: str = "queued"  # queued, running, completed, failed
    assigned_gpu: Optional[str] = None
    allocated_gpu: float = 0.0
    allocated_memory: float = 0.0
    start_time: Optional[float] = None
    finish_time: Optional[float] = None
    waiting_time: float = 0.0
    turnaround_time: float = 0.0
    energy_consumption: float = 0.0  # in kWh
    carbon_emission: float = 0.0      # in gCO2
    isolation_violations: int = 0

class TaskCreate(BaseModel):
    user_id: str
    gpu_demand: float = Field(..., ge=0, le=100)
    memory_demand: float = Field(..., ge=0)
    duration: float = Field(..., gt=0)
    priority: str = "medium"
    deadline: float = 100.0
    isolation_level: str = "soft"
    carbon_preference: str = "standard"

class GPU(BaseModel):
    gpu_id: str
    name: str = "NVIDIA A100-Sim"
    total_gpu_capacity: float = 100.0
    available_gpu_capacity: float = 100.0
    total_memory: float = 16384.0  # MB
    available_memory: float = 16384.0  # MB
    current_utilization: float = 0.0
    power_watts: float = 300.0
    carbon_intensity: float = 400.0  # gCO2/kWh
    renewable_ratio: float = 0.3
    energy_price: float = 0.12  # $/kWh
    location: str = "us-east-1"
    running_tasks: List[str] = []

class IsolationViolation(BaseModel):
    violation_id: str
    task_id: str
    user_id: str
    gpu_id: str
    timestamp: float
    requested_gpu: float
    allocated_gpu: float
    actual_usage: float
    violation_type: str  # compute_quota_exceeded, memory_quota_exceeded, interference
    description: str

class SchedulingResult(BaseModel):
    task_id: str
    assigned_gpu: Optional[str]
    scheduler_type: str  # baseline, predictive, proposed
    allocated_gpu: float
    allocated_memory: float
    expected_start: float
    expected_finish: float
    estimated_energy: float
    estimated_carbon: float
    isolation_status: str
    score: float = 0.0
    reason: str = ""

class PredictionResult(BaseModel):
    timestamp: float
    actual_workload: Optional[float] = None
    predicted_workload: float
    horizon: int = 1

class MetricsSummary(BaseModel):
    scheduler_type: str
    total_tasks: int
    completed_tasks: int
    avg_gpu_utilization: float
    avg_waiting_time: float
    avg_turnaround_time: float
    total_energy_kwh: float
    total_carbon_gco2: float
    avg_carbon_per_task: float
    deadline_violations: int
    isolation_violations: int
    jains_fairness_index: float
    prediction_mae: Optional[float] = None
    prediction_rmse: Optional[float] = None

class SimulationControl(BaseModel):
    action: str  # start, pause, reset, step
    scheduler_type: str = "proposed"
