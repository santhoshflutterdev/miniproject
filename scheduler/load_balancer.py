from simulation.gpu_simulator import GPU


class FirstFitScheduler:

    def __init__(self, gpus):
        self.gpus = gpus

    def schedule_task(self, task_id, gpu_demand, duration=10, current_time=0):
        for gpu in self.gpus:
            if gpu.available_capacity() >= gpu_demand:
                end_time = current_time + duration
                if gpu.allocate_task(task_id, gpu_demand, end_time=end_time):
                    return gpu
        return None


class PredictiveLoadBalancer:

    def __init__(self, gpus):
        self.gpus = gpus

    def calculate_score(self, gpu, predicted_demand):
        utilization = (gpu.used + predicted_demand) / gpu.capacity
        queue_length = len(gpu.active_tasks)
        score = 0.7 * utilization + 0.3 * (queue_length / 10.0)
        return score

    def schedule_task(self, task_id, gpu_demand, predicted_demand=None, duration=10, current_time=0):
        if predicted_demand is None:
            predicted_demand = gpu_demand

        candidates = []
        for gpu in self.gpus:
            if gpu.available_capacity() >= gpu_demand:
                score = self.calculate_score(gpu, predicted_demand)
                candidates.append((score, gpu))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0])
        best_gpu = candidates[0][1]
        end_time = current_time + duration
        best_gpu.allocate_task(task_id, gpu_demand, end_time=end_time)
        return best_gpu


if __name__ == "__main__":
    gpus = [GPU(i) for i in range(1, 5)]
    scheduler = PredictiveLoadBalancer(gpus)
    gpu = scheduler.schedule_task("T1", 30, predicted_demand=35, duration=10, current_time=0)
    print("Allocated to:", gpu.gpu_id if gpu else "None")