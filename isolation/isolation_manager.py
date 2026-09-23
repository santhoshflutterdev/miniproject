import uuid
import logging
from typing import Dict, List, Tuple, Optional
from backend.models import Task, GPU, IsolationViolation
from backend.firebase_service import save_violation

logger = logging.getLogger("IsolationManager")

class SoftwareIsolationManager:
    """
    Software-Enforced Resource Isolation Manager.
    Enforces logical GPU compute, GPU memory, and time execution quotas.
    
    Academic note: Software partitioning provides software-enforced resource isolation
    using explicit quotas to minimize cross-tenant interference while retaining flexiblity.
    """
    def __init__(self):
        # Map: gpu_id -> list of assigned tasks
        self.gpu_allocations: Dict[str, List[Task]] = {}
        # Recorded violations history
        self.violations: List[IsolationViolation] = []

    def check_quota(self, task: Task, gpu: GPU) -> bool:
        """
        Verifies if the target GPU has sufficient available compute quota for the task.
        """
        current_tasks = self.gpu_allocations.get(gpu.gpu_id, [])
        allocated_gpu_total = sum(t.allocated_gpu for t in current_tasks)
        remaining_capacity = gpu.total_gpu_capacity - allocated_gpu_total
        return remaining_capacity >= task.gpu_demand

    def check_memory(self, task: Task, gpu: GPU) -> bool:
        """
        Verifies if the target GPU has sufficient available memory quota for the task.
        """
        current_tasks = self.gpu_allocations.get(gpu.gpu_id, [])
        allocated_mem_total = sum(t.allocated_memory for t in current_tasks)
        remaining_memory = gpu.total_memory - allocated_mem_total
        return remaining_memory >= task.memory_demand

    def check_isolation(self, task: Task, gpu: GPU) -> bool:
        """
        Evaluates software isolation boundary policy:
        - Check compute & memory quota.
        - If task.isolation_level == 'strong', verify that no other user's tasks are co-located
          if GPU utilization exceeds 70%, or enforce strict single-user or isolated quota partition.
        """
        if not self.check_quota(task, gpu) or not self.check_memory(task, gpu):
            return False

        current_tasks = self.gpu_allocations.get(gpu.gpu_id, [])
        if task.isolation_level.lower() == "strong":
            # Check for multi-tenant interference risk
            other_user_tasks = [t for t in current_tasks if t.user_id != task.user_id]
            if other_user_tasks:
                current_util = sum(t.allocated_gpu for t in current_tasks)
                # Strong isolation rejects placement on multi-tenant GPU if projected utilization > 75%
                if (current_util + task.gpu_demand) > 75.0:
                    return False
        return True

    def allocate(self, task: Task, gpu: GPU) -> Tuple[bool, str]:
        """
        Allocates compute and memory quotas to the task on the target GPU.
        """
        if not self.check_isolation(task, gpu):
            return False, f"Isolation boundary check failed for GPU {gpu.gpu_id}"

        if gpu.gpu_id not in self.gpu_allocations:
            self.gpu_allocations[gpu.gpu_id] = []

        task.assigned_gpu = gpu.gpu_id
        task.allocated_gpu = task.gpu_demand
        task.allocated_memory = task.memory_demand
        
        self.gpu_allocations[gpu.gpu_id].append(task)
        
        # Update GPU state
        gpu.available_gpu_capacity -= task.gpu_demand
        gpu.available_memory -= task.memory_demand
        if task.task_id not in gpu.running_tasks:
            gpu.running_tasks.append(task.task_id)
        
        gpu.current_utilization = sum(t.allocated_gpu for t in self.gpu_allocations[gpu.gpu_id])
        return True, f"Allocated {task.gpu_demand}% GPU compute and {task.memory_demand}MB memory on {gpu.gpu_id}"

    def release(self, task: Task, gpu: GPU):
        """
        Releases allocated resources when a task completes execution.
        """
        if gpu.gpu_id in self.gpu_allocations:
            self.gpu_allocations[gpu.gpu_id] = [t for t in self.gpu_allocations[gpu.gpu_id] if t.task_id != task.task_id]
        
        gpu.available_gpu_capacity = min(gpu.total_gpu_capacity, gpu.available_gpu_capacity + task.allocated_gpu)
        gpu.available_memory = min(gpu.total_memory, gpu.available_memory + task.allocated_memory)
        
        if task.task_id in gpu.running_tasks:
            gpu.running_tasks.remove(task.task_id)
            
        gpu.current_utilization = sum(t.allocated_gpu for t in self.gpu_allocations.get(gpu.gpu_id, []))

    def detect_violation(self, task: Task, actual_usage: float, current_time: float) -> Optional[IsolationViolation]:
        """
        Detects if a running task attempts to exceed its allocated compute/memory quota.
        """
        if actual_usage > task.allocated_gpu:
            violation = IsolationViolation(
                violation_id=f"V-{uuid.uuid4().hex[:8]}",
                task_id=task.task_id,
                user_id=task.user_id,
                gpu_id=task.assigned_gpu or "UNKNOWN",
                timestamp=current_time,
                requested_gpu=task.gpu_demand,
                allocated_gpu=task.allocated_gpu,
                actual_usage=actual_usage,
                violation_type="compute_quota_exceeded",
                description=f"Task {task.task_id} attempted usage ({actual_usage:.1f}%) exceeding quota ({task.allocated_gpu:.1f}%)"
            )
            self.violations.append(violation)
            task.isolation_violations += 1
            save_violation(violation.dict())
            logger.warning(f"ISOLATION VIOLATION DETECTED: {violation.description}")
            return violation
        return None

    def enforce_quota(self, task: Task) -> float:
        """
        Enforces software ceiling: caps excessive compute consumption back to allocated quota.
        """
        logger.info(f"Software quota enforced on Task {task.task_id}: throttled back to quota limit ({task.allocated_gpu}%)")
        return task.allocated_gpu
