from typing import List, Optional
from backend.models import Task, GPU
from isolation.isolation_manager import SoftwareIsolationManager

class BaselineScheduler:
    """
    Baseline First-Fit / Least-Loaded GPU Scheduler.
    Selects the first feasible GPU with sufficient available GPU compute and memory capacity.
    """
    def __init__(self, gpus: List[GPU], isolation_manager: Optional[SoftwareIsolationManager] = None):
        self.gpus = gpus
        self.isolation_manager = isolation_manager or SoftwareIsolationManager()

    def select_gpu(self, task: Task) -> Optional[GPU]:
        """
        Iterates over GPUs and selects the least-loaded candidate GPU that satisfies basic resource constraints.
        """
        feasible_gpus = []
        for gpu in self.gpus:
            if self.isolation_manager.check_quota(task, gpu) and self.isolation_manager.check_memory(task, gpu):
                feasible_gpus.append(gpu)
        
        if not feasible_gpus:
            return None
        
        # Least-loaded selection strategy
        feasible_gpus.sort(key=lambda g: g.current_utilization)
        return feasible_gpus[0]
