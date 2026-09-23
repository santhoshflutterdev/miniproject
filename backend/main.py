import os
import pandas as pd
import logging
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from backend.config import settings
from backend.models import (
    Task, TaskCreate, GPU, SchedulingResult, IsolationViolation,
    MetricsSummary, SimulationControl
)
from backend.firebase_service import (
    get_tasks, get_task, create_task, get_gpus, get_results,
    get_violations, reset_simulation_store, is_firebase_active
)
from simulation.simulator import GPUSimulator
from models.workload_predictor import WorkloadPredictor
from data.generate_sample_data import generate_sample_tasks, generate_gpu_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FastAPIBackend")

app = FastAPI(
    title="Predictive & Carbon-Aware GPU Scheduler API",
    description="Backend API for Dynamic GPU Scheduling, Workload Prediction, Carbon Optimization, and Software Isolation",
    version="1.0.0"
)

# Enable CORS for Streamlit UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Simulation Engine Instance
simulator = GPUSimulator(scheduler_type="proposed")

@app.on_event("startup")
def startup_event():
    logger.info("Initializing GPU Scheduler System...")
    # Load sample task dataset if empty
    tasks = get_tasks()
    if not tasks:
        logger.info("No tasks found in DB. Loading default sample task dataset...")
        if not os.path.exists("data/sample_tasks.csv"):
            generate_gpu_config()
            generate_sample_tasks()
        df = pd.read_csv("data/sample_tasks.csv")
        simulator.load_tasks_from_dataframe(df)

@app.get("/health")
def health_check():
    return {
        "status": "online",
        "simulation_mode": settings.SIMULATION_MODE,
        "firebase_active": is_firebase_active(),
        "trained_model": simulator.predictor.is_trained,
        "active_scheduler": simulator.scheduler_type
    }

@app.get("/tasks", response_model=List[Dict[str, Any]])
def list_tasks():
    return get_tasks()

@app.get("/tasks/{task_id}")
def get_task_by_id(task_id: str):
    t = get_task(task_id)
    if not t:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return t

@app.post("/tasks", response_model=Dict[str, Any])
def submit_task(task_in: TaskCreate):
    task_count = len(get_tasks()) + 1
    new_task = Task(
        task_id=f"T{task_count:03d}",
        user_id=task_in.user_id,
        gpu_demand=task_in.gpu_demand,
        memory_demand=task_in.memory_demand,
        duration=task_in.duration,
        priority=task_in.priority,
        deadline=task_in.deadline,
        isolation_level=task_in.isolation_level,
        carbon_preference=task_in.carbon_preference
    )
    simulator.add_task(new_task)
    return new_task.dict()

@app.get("/gpus", response_model=List[Dict[str, Any]])
def list_gpus():
    return [g.dict() for g in simulator.gpus]

@app.post("/simulate")
def control_simulation(control: SimulationControl):
    simulator.set_scheduler_type(control.scheduler_type)
    action = control.action.lower()
    
    if action == "start":
        simulator.run_all()
        return {"status": "started", "message": "Simulation executed successfully"}
    elif action == "step":
        res = simulator.step()
        return {"status": "step_executed", "state": res}
    elif action == "pause":
        simulator.is_running = False
        return {"status": "paused"}
    elif action == "reset":
        simulator.reset()
        reset_simulation_store()
        if os.path.exists("data/sample_tasks.csv"):
            df = pd.read_csv("data/sample_tasks.csv")
            simulator.load_tasks_from_dataframe(df)
        return {"status": "reset", "message": "Simulation reset with sample tasks"}
    else:
        raise HTTPException(status_code=400, detail=f"Invalid action '{control.action}'")

@app.post("/schedule")
def schedule_pending():
    res = simulator.step()
    return {"status": "scheduled", "result": res}

@app.get("/metrics")
def get_metrics_summary(scheduler_type: Optional[str] = None):
    if scheduler_type:
        orig_type = simulator.scheduler_type
        simulator.set_scheduler_type(scheduler_type)
        summary = simulator.get_summary_metrics()
        simulator.set_scheduler_type(orig_type)
        return summary.dict()
    return simulator.get_summary_metrics().dict()

@app.get("/predictions")
def get_predictions():
    history = simulator.gpu_util_history.get("GPU1", [40.0, 45.0, 50.0, 55.0])
    pred_next = simulator.predictor.predict(history)
    return {
        "sequence_length": simulator.predictor.sequence_length,
        "is_trained": simulator.predictor.is_trained,
        "mae": simulator.predictor.last_mae,
        "rmse": simulator.predictor.last_rmse,
        "recent_history": history,
        "next_prediction": pred_next
    }

@app.get("/violations")
def list_violations():
    return [v.dict() for v in simulator.isolation_manager.violations]

@app.post("/train-model")
def train_workload_model(epochs: int = 20, sample_file: str = "data/sample_tasks.csv"):
    if os.path.exists(sample_file):
        df = pd.read_csv(sample_file)
    else:
        generate_sample_tasks()
        df = pd.read_csv("data/sample_tasks.csv")
    
    metrics = simulator.predictor.train(df, feature_col="gpu_demand", epochs=epochs)
    simulator.predictor.save_model("models/lstm_workload.pt")
    return {"status": "model_trained", "metrics": metrics}

@app.post("/reset-simulation")
def reset_simulation():
    simulator.reset()
    reset_simulation_store()
    if os.path.exists("data/sample_tasks.csv"):
        df = pd.read_csv("data/sample_tasks.csv")
        simulator.load_tasks_from_dataframe(df)
    return {"status": "reset_complete"}

# Mount HTML5/CSS3/JavaScript SPA frontend
from fastapi.staticfiles import StaticFiles
if os.path.exists("web"):
    app.mount("/", StaticFiles(directory="web", html=True), name="web_frontend")

