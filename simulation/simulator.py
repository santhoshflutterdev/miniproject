import random
import logging
import pandas as pd
from typing import List, Dict, Any, Optional
from backend.models import Task, GPU, SchedulingResult, IsolationViolation
from backend.firebase_service import create_task, update_task, save_result, save_gpus, save_violation
from isolation.isolation_manager import SoftwareIsolationManager
from models.workload_predictor import WorkloadPredictor
from schedulers.base_scheduler import BaselineScheduler
from schedulers.predictive_scheduler import PredictiveScheduler
from schedulers.carbon_scheduler import CarbonAwareScheduler
from schedulers.proposed_scheduler import ProposedScheduler
from simulation.metrics import MetricsEngine

logger = logging.getLogger("GPUSimulator")

class GPUSimulator:
    """
    Discrete-Event GPU Task Simulator Engine.
    Simulates GPU task arrivals, scheduling decisions, software isolation enforcement,
    execution progress, and metrics tracking over time.
    """
    def __init__(self, scheduler_type: str = "proposed", gpus: Optional[List[GPU]] = None):
        self.scheduler_type = scheduler_type.lower()
        self.current_time: float = 0.0
        self.is_running: bool = False
        
        # Initialize default candidate GPUs
        self.gpus: List[GPU] = gpus or [
            GPU(gpu_id="GPU1", name="NVIDIA A100 (US-East)", total_gpu_capacity=100.0, total_memory=81920.0, power_watts=350.0, carbon_intensity=600.0, renewable_ratio=0.15, location="us-east-1"),
            GPU(gpu_id="GPU2", name="NVIDIA V100 (US-West)", total_gpu_capacity=100.0, total_memory=32768.0, power_watts=300.0, carbon_intensity=300.0, renewable_ratio=0.45, location="us-west-2"),
            GPU(gpu_id="GPU3", name="NVIDIA A10G (EU-Nordic)", total_gpu_capacity=100.0, total_memory=24576.0, power_watts=250.0, carbon_intensity=100.0, renewable_ratio=0.85, location="eu-north-1"),
            GPU(gpu_id="GPU4", name="NVIDIA RTX4090 (Asia-East)", total_gpu_capacity=100.0, total_memory=24576.0, power_watts=450.0, carbon_intensity=500.0, renewable_ratio=0.25, location="ap-east-1")
        ]

        self.isolation_manager = SoftwareIsolationManager()
        self.predictor = WorkloadPredictor()
        
        # Auto-load pre-trained model if available
        self.predictor.load_model("models/lstm_workload.pt")

        self.queued_tasks: List[Task] = []
        self.running_tasks: List[Task] = []
        self.completed_tasks: List[Task] = []
        self.gpu_util_history: Dict[str, List[float]] = {g.gpu_id: [] for g in self.gpus}
        
        self.metrics_engine = MetricsEngine()
        self.save_gpu_state()

    def set_scheduler_type(self, scheduler_type: str):
        self.scheduler_type = scheduler_type.lower()

    def save_gpu_state(self):
        save_gpus([g.dict() for g in self.gpus])

    def load_tasks_from_dataframe(self, df: pd.DataFrame):
        self.reset()
        records = df.to_dict("records")
        for r in records:
            t = Task(
                task_id=str(r["task_id"]),
                user_id=str(r["user_id"]),
                arrival_time=float(r["arrival_time"]),
                gpu_demand=float(r["gpu_demand"]),
                memory_demand=float(r["memory_demand"]),
                duration=float(r["duration"]),
                priority=str(r.get("priority", "medium")),
                deadline=float(r.get("deadline", r["arrival_time"] + r["duration"] + 30)),
                isolation_level=str(r.get("isolation_level", "soft")),
                carbon_preference=str(r.get("carbon_preference", "standard"))
            )
            self.queued_tasks.append(t)
            create_task(t.dict())
        
        # Sort queue by arrival time
        self.queued_tasks.sort(key=lambda x: x.arrival_time)
        logger.info(f"Loaded {len(self.queued_tasks)} tasks into simulation queue.")

    def add_task(self, task: Task):
        self.queued_tasks.append(task)
        self.queued_tasks.sort(key=lambda x: x.arrival_time)
        create_task(task.dict())

    def reset(self):
        self.current_time = 0.0
        self.is_running = False
        self.queued_tasks.clear()
        self.running_tasks.clear()
        self.completed_tasks.clear()
        self.isolation_manager = SoftwareIsolationManager()
        self.gpu_util_history = {g.gpu_id: [] for g in self.gpus}
        
        for g in self.gpus:
            g.available_gpu_capacity = g.total_gpu_capacity
            g.available_memory = g.total_memory
            g.current_utilization = 0.0
            g.running_tasks.clear()
        
        self.save_gpu_state()
        logger.info("Simulation reset.")

    def _select_gpu_for_task(self, task: Task) -> Optional[GPU]:
        """
        Delegates GPU selection to the active scheduler.
        """
        if self.scheduler_type == "baseline":
            scheduler = BaselineScheduler(self.gpus, self.isolation_manager)
            return scheduler.select_gpu(task)
        elif self.scheduler_type == "predictive":
            scheduler = PredictiveScheduler(self.gpus, self.predictor, self.isolation_manager)
            return scheduler.select_gpu(task, self.gpu_util_history)
        elif self.scheduler_type == "proposed":
            scheduler = ProposedScheduler(self.gpus, self.predictor, self.isolation_manager)
            res = scheduler.select_gpu(task, self.current_time, self.gpu_util_history)
            if res:
                gpu, score, sub_scores = res
                return gpu
            return None
        else:
            scheduler = BaselineScheduler(self.gpus, self.isolation_manager)
            return scheduler.select_gpu(task)

    def step(self) -> Dict[str, Any]:
        """
        Executes a single discrete time step (1 minute increment).
        """
        self.current_time += 1.0

        # 1. Update running tasks execution progress
        finished_now = []
        for task in self.running_tasks:
            if task.start_time is not None and (self.current_time - task.start_time) >= task.duration:
                task.finish_time = self.current_time
                task.status = "completed"
                task.turnaround_time = task.finish_time - task.arrival_time

                # Release GPU resources
                assigned_gpu = next((g for g in self.gpus if g.gpu_id == task.assigned_gpu), None)
                if assigned_gpu:
                    self.isolation_manager.release(task, assigned_gpu)
                    # Compute energy & carbon footprint
                    energy = CarbonAwareScheduler.calculate_energy(task, assigned_gpu)
                    carbon = CarbonAwareScheduler.calculate_carbon(task, assigned_gpu)
                    task.energy_consumption = round(energy, 4)
                    task.carbon_emission = round(carbon, 2)

                update_task(task.task_id, task.dict())
                finished_now.append(task)

        for task in finished_now:
            self.running_tasks.remove(task)
            self.completed_tasks.append(task)

        # 2. Process Arrived Tasks from Queue
        arrived_tasks = [t for t in self.queued_tasks if t.arrival_time <= self.current_time and t.status == "queued"]
        
        for task in arrived_tasks:
            gpu = self._select_gpu_for_task(task)
            if gpu is not None:
                # Attempt allocation via SoftwareIsolationManager
                allocated, msg = self.isolation_manager.allocate(task, gpu)
                if allocated:
                    task.status = "running"
                    task.start_time = self.current_time
                    task.waiting_time = task.start_time - task.arrival_time
                    self.running_tasks.append(task)
                    self.queued_tasks.remove(task)

                    # Record scheduling result
                    res = SchedulingResult(
                        task_id=task.task_id,
                        assigned_gpu=gpu.gpu_id,
                        scheduler_type=self.scheduler_type,
                        allocated_gpu=task.allocated_gpu,
                        allocated_memory=task.allocated_memory,
                        expected_start=task.start_time,
                        expected_finish=task.start_time + task.duration,
                        estimated_energy=CarbonAwareScheduler.calculate_energy(task, gpu),
                        estimated_carbon=CarbonAwareScheduler.calculate_carbon(task, gpu),
                        isolation_status="enforced",
                        reason="Scheduled successfully"
                    )
                    save_result(res.dict())
                    update_task(task.task_id, task.dict())

        # 3. Simulate Resource Spikes & Software Isolation Boundary Enforcement
        for task in self.running_tasks:
            assigned_gpu = next((g for g in self.gpus if g.gpu_id == task.assigned_gpu), None)
            if assigned_gpu:
                # 5% chance of simulated noisy-neighbor usage spike
                if random.random() < 0.05:
                    simulated_actual_usage = task.allocated_gpu * random.uniform(1.1, 1.4)
                    violation = self.isolation_manager.detect_violation(task, simulated_actual_usage, self.current_time)
                    if violation:
                        # Enforce quota ceiling
                        self.isolation_manager.enforce_quota(task)

        # 4. Update GPU Utilization History
        for g in self.gpus:
            self.gpu_util_history[g.gpu_id].append(g.current_utilization)
            if len(self.gpu_util_history[g.gpu_id]) > 50:
                self.gpu_util_history[g.gpu_id].pop(0)

        self.save_gpu_state()

        return {
            "current_time": self.current_time,
            "queued_count": len(self.queued_tasks),
            "running_count": len(self.running_tasks),
            "completed_count": len(self.completed_tasks),
            "total_violations": len(self.isolation_manager.violations)
        }

    def run_all(self, max_steps: int = 500):
        self.is_running = True
        step_count = 0
        while self.is_running and (self.queued_tasks or self.running_tasks) and step_count < max_steps:
            self.step()
            step_count += 1
        self.is_running = False

    def get_summary_metrics(self):
        return self.metrics_engine.compute_summary(
            scheduler_type=self.scheduler_type,
            completed_tasks=self.completed_tasks,
            gpus=self.gpus,
            violations=self.isolation_manager.violations,
            prediction_mae=self.predictor.last_mae,
            prediction_rmse=self.predictor.last_rmse
        )
