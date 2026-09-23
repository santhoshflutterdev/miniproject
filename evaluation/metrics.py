class Metrics:

    def __init__(self):
        self.results = []

    def add_result(
        self,
        task_id,
        arrival_time,
        start_time,
        completion_time,
        gpu_id,
        energy,
        carbon
    ):
        waiting_time = max(0.0, start_time - arrival_time)
        turnaround_time = max(0.0, completion_time - arrival_time)

        self.results.append({
            "task_id": task_id,
            "arrival_time": arrival_time,
            "start_time": start_time,
            "completion_time": completion_time,
            "waiting_time": waiting_time,
            "turnaround_time": turnaround_time,
            "gpu_id": gpu_id,
            "energy": energy,
            "carbon": carbon
        })

    def average_waiting_time(self):
        if not self.results:
            return 0.0
        total = sum(r["waiting_time"] for r in self.results)
        return total / len(self.results)

    def average_turnaround_time(self):
        if not self.results:
            return 0.0
        total = sum(r["turnaround_time"] for r in self.results)
        return total / len(self.results)

    def total_energy(self):
        return sum(r["energy"] for r in self.results)

    def total_carbon(self):
        return sum(r["carbon"] for r in self.results)

    def completed_tasks(self):
        return len(self.results)

    def load_imbalance(self, gpu_utilizations):
        if not gpu_utilizations:
            return 0.0
        return max(gpu_utilizations) - min(gpu_utilizations)

    def summary(self):
        print("\n========== METRICS ==========")
        print(f"Average Waiting Time: {self.average_waiting_time():.2f} min")
        print(f"Average Turnaround Time: {self.average_turnaround_time():.2f} min")
        print(f"Total Energy: {self.total_energy():.4f} kWh")
        print(f"Total Carbon: {self.total_carbon():.2f} gCO2")
        print(f"Completed Tasks: {self.completed_tasks()}")

        return {
            "Waiting_Time": round(self.average_waiting_time(), 2),
            "Turnaround_Time": round(self.average_turnaround_time(), 2),
            "Energy": round(self.total_energy(), 2),
            "Carbon": round(self.total_carbon(), 2),
        }
