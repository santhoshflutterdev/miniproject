from typing import List, Optional
from backend.models import Task, GPU
from isolation.isolation_manager import SoftwareIsolationManager

class CarbonAwareScheduler:
    """
    Carbon-Aware GPU Scheduler.
    Prioritizes GPUs with lower carbon intensity, higher renewable ratio, and lower power consumption.
    """
    def __init__(self, gpus: List[GPU], isolation_manager: Optional[SoftwareIsolationManager] = None):
        self.gpus = gpus
        self.isolation_manager = isolation_manager or SoftwareIsolationManager()

    @staticmethod
    def calculate_energy(task: Task, gpu: GPU) -> float:
        """
        Calculates energy consumption in kWh for a task running on a GPU.
        Power (W) * (duration in hours) * (gpu_demand / 100) / 1000
        """
        duration_hours = task.duration / 60.0
        power_kw = (gpu.power_watts * (task.gpu_demand / 100.0)) / 1000.0
        return float(power_kw * duration_hours)

    @staticmethod
    def calculate_carbon(task: Task, gpu: GPU) -> float:
        """
        Calculates carbon emissions in gCO2.
        Energy (kWh) * Carbon Intensity (gCO2/kWh) * (1 - renewable_ratio)
        """
        energy_kwh = CarbonAwareScheduler.calculate_energy(task, gpu)
        net_carbon_intensity = gpu.carbon_intensity * (1.0 - max(0.0, min(1.0, gpu.renewable_ratio)))
        return float(energy_kwh * net_carbon_intensity)

    def calculate_score(self, task: Task, gpu: GPU) -> float:
        """
        Calculates a carbon optimization score in [0, 1]. Higher is cleaner/better.
        """
        carbon_emitted = self.calculate_carbon(task, gpu)
        # Normalize: lower carbon emissions yield higher score
        max_possible_carbon = (gpu.power_watts / 1000.0) * (task.duration / 60.0) * 800.0
        score = 1.0 - (carbon_emitted / (max_possible_carbon + 1e-6))
        return float(max(0.0, min(1.0, score)))

    def select_gpu(self, task: Task) -> Optional[GPU]:
        feasible_gpus = []
        for gpu in self.gpus:
            if self.isolation_manager.check_quota(task, gpu) and self.isolation_manager.check_memory(task, gpu):
                feasible_gpus.append(gpu)

        if not feasible_gpus:
            return None

        # Sort candidate GPUs by lowest estimated carbon emission
        feasible_gpus.sort(key=lambda g: self.calculate_carbon(task, g))
        return feasible_gpus[0]
