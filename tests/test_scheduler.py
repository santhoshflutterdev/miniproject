import pytest
from backend.models import Task, GPU
from isolation.isolation_manager import SoftwareIsolationManager
from schedulers.base_scheduler import BaselineScheduler
from schedulers.carbon_scheduler import CarbonAwareScheduler
from schedulers.proposed_scheduler import ProposedScheduler

@pytest.fixture
def sample_gpus():
    return [
        GPU(gpu_id="GPU1", name="GPU-Dirty", power_watts=400.0, carbon_intensity=700.0, renewable_ratio=0.1),
        GPU(gpu_id="GPU2", name="GPU-Clean", power_watts=250.0, carbon_intensity=100.0, renewable_ratio=0.8)
    ]

@pytest.fixture
def sample_task():
    return Task(
        task_id="T_TEST",
        user_id="U1",
        gpu_demand=30.0,
        memory_demand=4096.0,
        duration=15.0,
        isolation_level="soft",
        carbon_preference="low"
    )

def test_baseline_scheduler(sample_gpus, sample_task):
    scheduler = BaselineScheduler(sample_gpus)
    selected = scheduler.select_gpu(sample_task)
    assert selected is not None
    assert selected.gpu_id in ["GPU1", "GPU2"]

def test_carbon_aware_scheduler(sample_gpus, sample_task):
    scheduler = CarbonAwareScheduler(sample_gpus)
    selected = scheduler.select_gpu(sample_task)
    # Cleaner GPU (GPU2) must be selected for lower carbon emission
    assert selected is not None
    assert selected.gpu_id == "GPU2"

def test_proposed_scheduler(sample_gpus, sample_task):
    scheduler = ProposedScheduler(sample_gpus)
    res = scheduler.select_gpu(sample_task)
    assert res is not None
    gpu, score, sub_scores = res
    assert 0.0 <= score <= 1.0
    assert "carbon" in sub_scores
    assert "resource" in sub_scores
