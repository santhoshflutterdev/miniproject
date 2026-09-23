import logging
from typing import List, Optional, Dict, Tuple
from backend.models import Task, GPU
from backend.config import settings
from isolation.isolation_manager import SoftwareIsolationManager
from models.workload_predictor import WorkloadPredictor
from schedulers.carbon_scheduler import CarbonAwareScheduler

logger = logging.getLogger("ProposedScheduler")

class ProposedScheduler:
    """
    Proposed Multi-Objective Predictive, Carbon-Aware, and Isolation-Preserving GPU Scheduler.
    
    Scoring Formula:
    Score = w_res * S_resource + w_pred * S_prediction + w_carb * S_carbon + w_dead * S_deadline + w_iso * S_isolation
    """
    def __init__(
        self,
        gpus: List[GPU],
        predictor: Optional[WorkloadPredictor] = None,
        isolation_manager: Optional[SoftwareIsolationManager] = None,
        weights: Optional[Dict[str, float]] = None
    ):
        self.gpus = gpus
        self.predictor = predictor or WorkloadPredictor()
        self.isolation_manager = isolation_manager or SoftwareIsolationManager()
        
        # Load weights from parameter or system config
        raw_weights = weights or {
            "resource": settings.RESOURCE_WEIGHT,
            "prediction": settings.PREDICTION_WEIGHT,
            "carbon": settings.CARBON_WEIGHT,
            "deadline": settings.DEADLINE_WEIGHT,
            "isolation": settings.ISOLATION_WEIGHT
        }
        self.weights = self._normalize_weights(raw_weights)

    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        total = sum(weights.values())
        if total <= 0:
            return {"resource": 0.3, "prediction": 0.25, "carbon": 0.25, "deadline": 0.1, "isolation": 0.1}
        return {k: v / total for k, v in weights.items()}

    def set_weights(self, new_weights: Dict[str, float]):
        self.weights = self._normalize_weights(new_weights)
        logger.info(f"Updated ProposedScheduler weights: {self.weights}")

    def compute_scores(self, task: Task, gpu: GPU, current_time: float = 0.0, recent_gpu_history: Optional[Dict[str, List[float]]] = None) -> Tuple[float, Dict[str, float]]:
        """
        Computes sub-scores and final weighted multi-objective score for a (task, gpu) pair.
        All sub-scores are strictly normalized in range [0.0, 1.0].
        """
        # 1. Resource Availability Sub-score (S_resource)
        avail_compute_ratio = (gpu.available_gpu_capacity - task.gpu_demand) / gpu.total_gpu_capacity
        avail_memory_ratio = (gpu.available_memory - task.memory_demand) / gpu.total_memory
        s_resource = max(0.0, min(1.0, (avail_compute_ratio + avail_memory_ratio) / 2.0))

        # 2. Workload Prediction Sub-score (S_prediction)
        history = (recent_gpu_history or {}).get(gpu.gpu_id, [gpu.current_utilization])
        predicted_util = self.predictor.predict(history)
        projected_peak = min(100.0, predicted_util + task.gpu_demand)
        s_prediction = max(0.0, min(1.0, (100.0 - projected_peak) / 100.0))

        # 3. Carbon Sub-score (S_carbon)
        s_carbon = CarbonAwareScheduler(self.gpus).calculate_score(task, gpu)

        # 4. Deadline Sub-score (S_deadline)
        est_completion_time = current_time + task.duration
        if task.deadline <= 0:
            s_deadline = 1.0
        else:
            slack = task.deadline - est_completion_time
            s_deadline = max(0.0, min(1.0, slack / (task.deadline + 1e-6))) if slack >= 0 else 0.0

        # 5. Software Isolation Sub-score (S_isolation)
        isolation_ok = self.isolation_manager.check_isolation(task, gpu)
        if not isolation_ok:
            s_isolation = 0.0
        else:
            if task.isolation_level.lower() == "strong":
                s_isolation = 1.0 if (gpu.current_utilization + task.gpu_demand) <= 60.0 else 0.7
            else:
                s_isolation = 0.9

        sub_scores = {
            "resource": s_resource,
            "prediction": s_prediction,
            "carbon": s_carbon,
            "deadline": s_deadline,
            "isolation": s_isolation
        }

        total_score = sum(self.weights[k] * sub_scores[k] for k in self.weights)
        return total_score, sub_scores

    def select_gpu(self, task: Task, current_time: float = 0.0, recent_gpu_history: Optional[Dict[str, List[float]]] = None) -> Optional[Tuple[GPU, float, Dict[str, float]]]:
        """
        Selects the optimal GPU with the highest composite multi-objective score.
        """
        best_gpu = None
        best_score = -1.0
        best_sub_scores = {}

        for gpu in self.gpus:
            # Hard policy constraint check
            if not self.isolation_manager.check_quota(task, gpu) or not self.isolation_manager.check_memory(task, gpu):
                continue

            score, sub_scores = self.compute_scores(task, gpu, current_time, recent_gpu_history)
            if score > best_score:
                best_score = score
                best_gpu = gpu
                best_sub_scores = sub_scores

        if best_gpu is None:
            return None

        return best_gpu, best_score, best_sub_scores
