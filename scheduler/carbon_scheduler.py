from simulation.gpu_simulator import GPU


class CarbonScheduler:

    def __init__(
        self,
        gpus,
        alpha=0.20,
        beta=0.25,
        gamma=0.35,
        delta=0.20
    ):
        self.gpus = gpus
        self.alpha = alpha   # Execution Time Weight
        self.beta = beta     # Energy Consumption Weight
        self.gamma = gamma   # Carbon Emission Weight
        self.delta = delta   # Projected Load & Queue Weight

    def normalize(self, value, minimum, maximum):
        if maximum == minimum:
            return 0.0
        return (value - minimum) / (maximum - minimum)

    def calculate_score(
        self,
        execution_time,
        energy,
        carbon,
        projected_load,
        min_time,
        max_time,
        min_energy,
        max_energy,
        min_carbon,
        max_carbon,
        min_load,
        max_load
    ):
        normalized_time = self.normalize(execution_time, min_time, max_time)
        normalized_energy = self.normalize(energy, min_energy, max_energy)
        normalized_carbon = self.normalize(carbon, min_carbon, max_carbon)
        normalized_load = self.normalize(projected_load, min_load, max_load)

        score = (
            self.alpha * normalized_time
            + self.beta * normalized_energy
            + self.gamma * normalized_carbon
            + self.delta * normalized_load
        )
        return score

    def select_gpu(
        self,
        task_id,
        gpu_demand,
        predicted_demand,
        duration_minutes,
        current_time=0
    ):
        candidates = []

        for gpu in self.gpus:
            if gpu.available_capacity() >= gpu_demand:
                energy = gpu.energy_consumption(duration_minutes)
                carbon = gpu.carbon_emission(duration_minutes)
                # Projected load factor considering current usage + predicted workload demand
                projected_load = (gpu.used + predicted_demand) / gpu.capacity
                queue_length = len(gpu.active_tasks)

                candidates.append({
                    "gpu": gpu,
                    "time": duration_minutes,
                    "energy": energy,
                    "carbon": carbon,
                    "projected_load": projected_load + (queue_length * 0.1)
                })

        if not candidates:
            return None

        times = [c["time"] for c in candidates]
        energies = [c["energy"] for c in candidates]
        carbons = [c["carbon"] for c in candidates]
        loads = [c["projected_load"] for c in candidates]

        min_time, max_time = min(times), max(times)
        min_energy, max_energy = min(energies), max(energies)
        min_carbon, max_carbon = min(carbons), max(carbons)
        min_load, max_load = min(loads), max(loads)

        for candidate in candidates:
            candidate["score"] = self.calculate_score(
                candidate["time"],
                candidate["energy"],
                candidate["carbon"],
                candidate["projected_load"],
                min_time,
                max_time,
                min_energy,
                max_energy,
                min_carbon,
                max_carbon,
                min_load,
                max_load
            )

        candidates.sort(key=lambda x: x["score"])
        best = candidates[0]
        best_gpu = best["gpu"]

        end_time = current_time + duration_minutes
        best_gpu.allocate_task(task_id, gpu_demand, end_time=end_time)

        return best_gpu


if __name__ == "__main__":
    gpu1 = GPU(1, idle_power=80, max_power=400, carbon_intensity=200)
    gpu2 = GPU(2, idle_power=80, max_power=350, carbon_intensity=700)
    gpu3 = GPU(3, idle_power=90, max_power=450, carbon_intensity=350)
    gpu4 = GPU(4, idle_power=70, max_power=300, carbon_intensity=150)

    gpus = [gpu1, gpu2, gpu3, gpu4]
    scheduler = CarbonScheduler(gpus)

    gpu = scheduler.select_gpu("T1", 30, 32, 10, current_time=0)
    if gpu:
        print(f"Task T1 allocated to GPU {gpu.gpu_id}")