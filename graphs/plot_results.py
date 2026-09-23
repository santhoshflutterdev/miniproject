import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_all_results():
    results_path = "dataset/results.csv"
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found. Run main.py first.")
        return

    results = pd.read_csv(results_path)
    print("\nExperimental Results for Plotting:")
    print(results)

    os.makedirs("graphs", exist_ok=True)

    # 1. Waiting Time
    plt.figure(figsize=(6, 4))
    plt.bar(results["Algorithm"], results["Waiting_Time"], color=['#e74c3c', '#f39c12', '#2ecc71'])
    plt.xlabel("Scheduling Algorithm")
    plt.ylabel("Average Waiting Time (min)")
    plt.title("Average Waiting Time Comparison")
    plt.tight_layout()
    plt.savefig("graphs/waiting_time.png")
    plt.close()

    # 2. Turnaround Time
    plt.figure(figsize=(6, 4))
    plt.bar(results["Algorithm"], results["Turnaround_Time"], color=['#e74c3c', '#f39c12', '#2ecc71'])
    plt.xlabel("Scheduling Algorithm")
    plt.ylabel("Average Turnaround Time (min)")
    plt.title("Average Turnaround Time Comparison")
    plt.tight_layout()
    plt.savefig("graphs/turnaround_time.png")
    plt.close()

    # 3. Energy
    plt.figure(figsize=(6, 4))
    plt.bar(results["Algorithm"], results["Energy"], color=['#e74c3c', '#f39c12', '#2ecc71'])
    plt.xlabel("Scheduling Algorithm")
    plt.ylabel("Energy (kWh)")
    plt.title("Energy Consumption Comparison")
    plt.tight_layout()
    plt.savefig("graphs/energy.png")
    plt.close()

    # 4. Carbon
    plt.figure(figsize=(6, 4))
    plt.bar(results["Algorithm"], results["Carbon"], color=['#e74c3c', '#f39c12', '#2ecc71'])
    plt.xlabel("Scheduling Algorithm")
    plt.ylabel("Carbon Emission (gCO2)")
    plt.title("Carbon Emission Comparison")
    plt.tight_layout()
    plt.savefig("graphs/carbon.png")
    plt.close()

    # 5. Load Imbalance
    plt.figure(figsize=(6, 4))
    plt.bar(results["Algorithm"], results["Load_Imbalance"], color=['#e74c3c', '#f39c12', '#2ecc71'])
    plt.xlabel("Scheduling Algorithm")
    plt.ylabel("Load Imbalance (%)")
    plt.title("GPU Load Imbalance Comparison")
    plt.tight_layout()
    plt.savefig("graphs/load_imbalance.png")
    plt.close()

    print("\nSuccessfully updated all comparison graphs in graphs/ directory!")


if __name__ == "__main__":
    plot_all_results()