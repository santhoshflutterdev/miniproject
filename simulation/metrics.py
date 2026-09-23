import numpy as np
from typing import List, Dict, Any, Optional
from backend.models import Task, GPU, MetricsSummary, IsolationViolation

class MetricsEngine:
    """
    Evaluation Metrics Engine for Cloud GPU Schedulers.
    Calculates empirical performance, energy, carbon footprint, isolation security, and fairness metrics.
    """
    def __init__(self):
        self.results_history: List[Dict[str, Any]] = []

    @staticmethod
    def jains_fairness_index(values: List[float]) -> float:
        """
        Calculates Jain's Fairness Index:
        J(x) = (sum(x)^2) / (n * sum(x^2))
        Result is bounded in [1/n, 1.0]. Higher represents fairer resource allocation.
        """
        if not values or len(values) == 0:
            return 1.0
        arr = np.array(values, dtype=float)
        n = len(arr)
        sum_val = np.sum(arr)
        sum_sq_val = np.sum(arr ** 2)
        if sum_sq_val == 0:
            return 1.0
        return float((sum_val ** 2) / (n * sum_sq_val))

    def compute_summary(
        self,
        scheduler_type: str,
        completed_tasks: List[Task],
        gpus: List[GPU],
        violations: List[IsolationViolation],
        prediction_mae: Optional[float] = None,
        prediction_rmse: Optional[float] = None,
        total_tasks: Optional[int] = None,
        queued_tasks: int = 0,
        running_tasks: int = 0,
        clock_time: float = 0.0
    ) -> MetricsSummary:
        """
        Generates a summary of all metrics for a simulation run.
        """
        completed_count = len(completed_tasks)
        total_count = total_tasks if total_tasks is not None else completed_count

        waiting_times = [t.waiting_time for t in completed_tasks]
        turnaround_times = [t.turnaround_time for t in completed_tasks]
        energy_consumptions = [t.energy_consumption for t in completed_tasks]
        carbon_emissions = [t.carbon_emission for t in completed_tasks]

        deadline_violations_count = sum(
            1 for t in completed_tasks if t.finish_time is not None and t.finish_time > t.deadline
        )
        isolation_violations_count = len(violations)

        avg_gpu_util = float(np.mean([g.current_utilization for g in gpus])) if gpus else 0.0
        avg_wait = float(np.mean(waiting_times)) if waiting_times else 0.0
        avg_turnaround = float(np.mean(turnaround_times)) if turnaround_times else 0.0
        total_energy = float(np.sum(energy_consumptions)) if energy_consumptions else 0.0
        total_carbon = float(np.sum(carbon_emissions)) if carbon_emissions else 0.0
        avg_carbon = total_carbon / completed_count if completed_count > 0 else 0.0

        # Jain's fairness index across user allocated resources
        user_allocations: Dict[str, float] = {}
        for t in completed_tasks:
            user_allocations[t.user_id] = user_allocations.get(t.user_id, 0.0) + t.allocated_gpu
        fairness = self.jains_fairness_index(list(user_allocations.values()))

        summary = MetricsSummary(
            scheduler_type=scheduler_type,
            total_tasks=total_count,
            completed_tasks=completed_count,
            queued_tasks=queued_tasks,
            running_tasks=running_tasks,
            clock_time=clock_time,
            avg_gpu_utilization=round(avg_gpu_util, 2),
            avg_waiting_time=round(avg_wait, 2),
            avg_turnaround_time=round(avg_turnaround, 2),
            total_energy_kwh=round(total_energy, 4),
            total_carbon_gco2=round(total_carbon, 2),
            avg_carbon_per_task=round(avg_carbon, 2),
            deadline_violations=deadline_violations_count,
            isolation_violations=isolation_violations_count,
            jains_fairness_index=round(fairness, 4),
            prediction_mae=prediction_mae,
            prediction_rmse=prediction_rmse
        )
        return summary
