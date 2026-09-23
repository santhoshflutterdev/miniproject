import pandas as pd
import numpy as np
from simulation.gpu_simulator import GPU
from scheduler.carbon_scheduler import CarbonScheduler
from scheduler.load_balancer import FirstFitScheduler, PredictiveLoadBalancer
from prediction.lstm_prediction import predict_workload_demands
from evaluation.metrics import Metrics
from firebase_config import save_simulation_results


def create_gpus():
    gpu1 = GPU(1, idle_power=80, max_power=400, carbon_intensity=200)
    gpu2 = GPU(2, idle_power=80, max_power=350, carbon_intensity=700)
    gpu3 = GPU(3, idle_power=90, max_power=450, carbon_intensity=350)
    gpu4 = GPU(4, idle_power=70, max_power=300, carbon_intensity=150)
    return [gpu1, gpu2, gpu3, gpu4]


def run_simulation(algo_name, scheduler_type, scheduler_obj, gpus, workload_df):
    for gpu in gpus:
        gpu.reset()

    metrics = Metrics()
    tasks_to_schedule = workload_df.to_dict("records")
    scheduled_tasks = []
    current_time = 0.0

    print(f"\n--- Running Simulation: {algo_name} ---")

    max_simulation_time = 200.0

    while len(scheduled_tasks) < len(tasks_to_schedule) and current_time < max_simulation_time:
        for gpu in gpus:
            gpu.update_time(current_time)

        unassigned_tasks = [
            t for t in tasks_to_schedule if t["task_id"] not in scheduled_tasks
        ]

        for task in unassigned_tasks:
            if task["arrival_time"] > current_time:
                continue

            task_id = task["task_id"]
            arrival_time = float(task["arrival_time"])
            gpu_demand = float(task["gpu_demand"])
            predicted_demand = float(task.get("predicted_demand", gpu_demand))
            duration = float(task["duration"])

            selected_gpu = None
            if scheduler_type == "baseline":
                selected_gpu = scheduler_obj.schedule_task(
                    task_id, gpu_demand, duration=duration, current_time=current_time
                )
            elif scheduler_type == "predictive":
                selected_gpu = scheduler_obj.schedule_task(
                    task_id, gpu_demand, predicted_demand=predicted_demand, duration=duration, current_time=current_time
                )
            elif scheduler_type == "proposed":
                selected_gpu = scheduler_obj.select_gpu(
                    task_id, gpu_demand, predicted_demand=predicted_demand, duration_minutes=duration, current_time=current_time
                )

            if selected_gpu is not None:
                start_time = current_time
                completion_time = start_time + duration
                energy = selected_gpu.energy_consumption(duration)
                carbon = selected_gpu.carbon_emission(duration)

                metrics.add_result(
                    task_id,
                    arrival_time,
                    start_time,
                    completion_time,
                    selected_gpu.gpu_id,
                    energy,
                    carbon
                )
                scheduled_tasks.append(task_id)
                print(f"Time {current_time:.1f}m: Task {task_id} -> GPU {selected_gpu.gpu_id} (Wait: {start_time - arrival_time:.1f}m)")

        current_time += 1.0

    max_used = max(gpu.used for gpu in gpus)
    min_used = min(gpu.used for gpu in gpus)
    load_imbalance = max_used - min_used

    summary = metrics.summary()
    print(f"Load Imbalance: {load_imbalance:.2f}%")

    return {
        "Algorithm": algo_name,
        "Waiting_Time": summary["Waiting_Time"],
        "Turnaround_Time": summary["Turnaround_Time"],
        "Energy": summary["Energy"],
        "Carbon": summary["Carbon"],
        "Load_Imbalance": round(load_imbalance, 2)
    }


def main():
    print("========================================")
    print(" PREDICTIVE & CARBON-AWARE GPU SCHEDULER")
    print("========================================")

    workload = pd.read_csv("dataset/workload.csv")
    print("\n[1/3] Training Predictive Workload Neural Network...")
    predicted_demands = predict_workload_demands(workload, sequence_length=3, epochs=100, batch_size=1, verbose=0)
    workload["predicted_demand"] = predicted_demands

    print("\nWorkload Trace with Neural Network Predictions:")
    print(workload[["task_id", "arrival_time", "gpu_demand", "predicted_demand", "duration"]])

    print("\n[2/3] Simulating Schedulers...")
    results = []

    # 1. Baseline First-Fit
    gpus_baseline = create_gpus()
    baseline_scheduler = FirstFitScheduler(gpus_baseline)
    res_baseline = run_simulation("Baseline", "baseline", baseline_scheduler, gpus_baseline, workload)
    results.append(res_baseline)

    # 2. Predictive Load Balancer
    gpus_predictive = create_gpus()
    predictive_scheduler = PredictiveLoadBalancer(gpus_predictive)
    res_predictive = run_simulation("Predictive", "predictive", predictive_scheduler, gpus_predictive, workload)
    results.append(res_predictive)

    # 3. Proposed Predictive & Carbon-Aware Scheduler
    gpus_proposed = create_gpus()
    proposed_scheduler = CarbonScheduler(gpus_proposed)
    res_proposed = run_simulation("Proposed", "proposed", proposed_scheduler, gpus_proposed, workload)
    results.append(res_proposed)

    results_df = pd.DataFrame(results)

    print("\n========================================")
    print(" COMPARATIVE EXPERIMENTAL RESULTS")
    print("========================================")
    print(results_df.to_string(index=False))

    save_simulation_results(results_df)


if __name__ == "__main__":
    main()