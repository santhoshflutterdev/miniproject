import os
import sys
import pandas as pd
from flask import Flask, jsonify, send_from_directory, request

from main import main as run_main_simulation
from firebase_config import fetch_simulation_results

app = Flask(__name__, static_folder="web", static_url_path="")


@app.route("/")
def serve_index():
    return send_from_directory("web", "index.html")


@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory("web", path)


@app.route("/api/results", methods=["GET"])
def get_results():
    try:
        results_df = fetch_simulation_results()
        data = results_df.to_dict(orient="records")
        return jsonify({"status": "success", "data": data})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/run-simulation", methods=["POST"])
def trigger_simulation():
    try:
        print("[API] Triggering live simulation pipeline...")
        run_main_simulation()
        results_df = fetch_simulation_results()
        data = results_df.to_dict(orient="records")
        return jsonify({
            "status": "success",
            "message": "Simulation completed and synced with Firebase successfully!",
            "data": data
        })
    except Exception as e:
        print(f"[API Error] Simulation failed: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/gpu-status", methods=["GET"])
def get_gpu_status():
    gpu_data = [
        {
            "id": 1,
            "name": "GPU 1",
            "region": "Region A (US-East)",
            "carbon_intensity": 200,
            "simulated_load": 65,
            "idle_power": 80,
            "max_power": 400,
            "status": "Optimal"
        },
        {
            "id": 2,
            "name": "GPU 2",
            "region": "Region B (US-Central)",
            "carbon_intensity": 700,
            "simulated_load": 45,
            "idle_power": 80,
            "max_power": 350,
            "status": "High Carbon"
        },
        {
            "id": 3,
            "name": "GPU 3",
            "region": "Region C (EU-West)",
            "carbon_intensity": 350,
            "simulated_load": 55,
            "idle_power": 90,
            "max_power": 450,
            "status": "Moderate"
        },
        {
            "id": 4,
            "name": "GPU 4",
            "region": "Region D (AP-South)",
            "carbon_intensity": 150,
            "simulated_load": 75,
            "idle_power": 70,
            "max_power": 300,
            "status": "Clean Grid"
        }
    ]
    return jsonify({"status": "success", "data": gpu_data})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    print(f"\n=======================================================")
    print(f" ⚡ GPU SCHEDULER WEB SERVER RUNNING")
    print(f" 🌐 Access Web Dashboard: http://localhost:{port}")
    print(f"=======================================================\n")
    app.run(host="0.0.0.0", port=port, debug=True)

