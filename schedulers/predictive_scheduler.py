from typing import List, Optional, Dict
from backend.models import Task, GPU
from isolation.isolation_manager import SoftwareIsolationManager
from models.workload_predictor import WorkloadPredictor

class PredictiveScheduler:
    """
    Predictive GPU Scheduler.
    Leverages PyTorch LSTM workload predictions to select the GPU with lowest projected future load peak.
    """
    def __init__(self, gpus: List[GPU], predictor: WorkloadPredictor, isolation_manager: Optional[SoftwareIsolationManager] = None):
        self.gpus = gpus
        self.predictor = predictor
        self.isolation_manager = isolation_manager or SoftwareIsolationManager()

    def select_gpu(self, task: Task, recent_gpu_history: Optional[Dict[str, List[float]]] = None) -> Optional[GPU]:
        feasible_gpus = []
        for gpu in self.gpus:
            if self.isolation_manager.check_quota(task, gpu) and self.isolation_manager.check_memory(task, gpu):
                feasible_gpus.append(gpu)
        
        if not feasible_gpus:
            return None

        best_gpu = None
        min_projected_load = float('inf')

        for gpu in feasible_gpus:
            history = (recent_gpu_history or {}).get(gpu.gpu_id, [gpu.current_utilization])
            pred_demand = self.predictor.predict(history)
            projected_load = pred_demand + task.gpu_demand
            
            if projected_load < min_projected_load:
                min_projected_load = projected_load
                best_gpu = gpu

        return best_gpu or feasible_gpus[0]
