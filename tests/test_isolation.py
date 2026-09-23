import pytest
from backend.models import Task, GPU
from isolation.isolation_manager import SoftwareIsolationManager

@pytest.fixture
def manager():
    return SoftwareIsolationManager()

@pytest.fixture
def test_gpu():
    return GPU(gpu_id="GPU_ISO", total_gpu_capacity=100.0, total_memory=16384.0)

def test_quota_check_and_allocation(manager, test_gpu):
    t1 = Task(task_id="T1", user_id="U1", gpu_demand=60.0, memory_demand=8192.0, duration=10.0)
    assert manager.check_quota(t1, test_gpu) is True
    assert manager.check_memory(t1, test_gpu) is True

    allocated, msg = manager.allocate(t1, test_gpu)
    assert allocated is True
    assert test_gpu.available_gpu_capacity == 40.0
    assert test_gpu.available_memory == 8192.0

    # Over-allocation task should fail quota check
    t2 = Task(task_id="T2", user_id="U2", gpu_demand=50.0, memory_demand=4096.0, duration=10.0)
    assert manager.check_quota(t2, test_gpu) is False

def test_violation_detection(manager, test_gpu):
    t = Task(task_id="T_VIOL", user_id="U1", gpu_demand=40.0, memory_demand=4096.0, duration=10.0)
    manager.allocate(t, test_gpu)

    # Actual usage exceeds quota (40%)
    violation = manager.detect_violation(t, actual_usage=55.0, current_time=5.0)
    assert violation is not None
    assert violation.violation_type == "compute_quota_exceeded"
    assert t.isolation_violations == 1
