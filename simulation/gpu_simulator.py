class GPU:

    def __init__(
        self,
        gpu_id,
        capacity=100,
        idle_power=80,
        max_power=400,
        carbon_intensity=200
    ):
        self.gpu_id = gpu_id
        self.capacity = capacity
        self.used = 0
        self.tasks = []
        self.active_tasks = []

        self.idle_power = idle_power
        self.max_power = max_power
        self.carbon_intensity = carbon_intensity

    def reset(self):
        """Resets the GPU utilization state and tasks list."""
        self.used = 0
        self.tasks = []
        self.active_tasks = []

    def update_time(self, current_time):
        """Releases capacity for any active tasks whose completion time has passed."""
        finished = [t for t in self.active_tasks if current_time >= t["end_time"]]
        for t in finished:
            self.release_task(t["task_id"], t["gpu_demand"])
            self.active_tasks.remove(t)

    def available_capacity(self):
        return max(0, self.capacity - self.used)

    def power_consumption(self):
        utilization = self.used / self.capacity
        power = (
            self.idle_power
            + utilization * (self.max_power - self.idle_power)
        )
        return power

    def energy_consumption(self, duration_minutes):
        power = self.power_consumption()
        duration_hours = duration_minutes / 60.0
        energy = (power * duration_hours) / 1000.0
        return energy

    def carbon_emission(self, duration_minutes):
        energy = self.energy_consumption(duration_minutes)
        carbon = energy * self.carbon_intensity
        return carbon

    def allocate_task(self, task_id, gpu_demand, end_time=None):
        if gpu_demand <= self.available_capacity():
            self.used += gpu_demand
            self.tasks.append(task_id)
            if end_time is not None:
                self.active_tasks.append({
                    "task_id": task_id,
                    "gpu_demand": gpu_demand,
                    "end_time": end_time
                })
            return True
        return False

    def release_task(self, task_id, gpu_demand):
        if task_id in self.tasks:
            self.used = max(0, self.used - gpu_demand)
            self.tasks.remove(task_id)

    def __str__(self):
        power = self.power_consumption()
        return (
            f"GPU {self.gpu_id}: "
            f"{self.used}% used, "
            f"{self.available_capacity()}% available, "
            f"Power: {power:.2f} W, "
            f"{self.carbon_intensity} gCO2/kWh, "
            f"Tasks: {self.tasks}"
        )


if __name__ == "__main__":
    gpu1 = GPU(1, idle_power=80, max_power=400, carbon_intensity=200)
    gpu2 = GPU(2, idle_power=80, max_power=350, carbon_intensity=700)
    gpu3 = GPU(3, idle_power=90, max_power=450, carbon_intensity=350)
    gpu4 = GPU(4, idle_power=70, max_power=300, carbon_intensity=150)

    gpus = [gpu1, gpu2, gpu3, gpu4]

    gpu1.allocate_task("T1", 50, end_time=15)
    gpu2.allocate_task("T2", 30, end_time=8)
    gpu3.allocate_task("T3", 70, end_time=20)

    print("\nGPU Carbon Status:")
    for gpu in gpus:
        print(gpu)
        energy = gpu.energy_consumption(30)
        carbon = gpu.carbon_emission(30)
        print(f"Energy for 30 minutes: {energy:.4f} kWh")
        print(f"Carbon for 30 minutes: {carbon:.2f} gCO2")