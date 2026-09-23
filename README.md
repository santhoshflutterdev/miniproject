# Predictive & Carbon-Aware GPU Scheduling with Software-Enforced Resource Isolation in Cloud Environments

> **Final-Year Academic Research Project & System Prototype**  
> *Extension of Base Paper: "Dynamic Task Scheduling and Adaptive GPU Resource Allocation in the Cloud"*

---

## 📌 Executive Summary & Academic Background

Modern cloud infrastructure relies heavily on shared Multi-GPU nodes to execute machine learning (DL/ML) workloads, high-performance computing (HPC) simulations, and data processing tasks. Traditional cloud GPU schedulers prioritize either high throughput or simple load balancing, leading to key inefficiencies:
1. **Unchecked Inter-Tenant Interference**: Co-locating multiple tasks on shared GPUs without strict resource boundaries causes "noisy-neighbor" compute and memory contention.
2. **Reactive Resource Allocation**: Allocating GPUs based solely on instantaneous utilization leads to resource fragmentation during workload bursts.
3. **Carbon & Energy Inefficiency**: Cloud data centers span heterogeneous power grids with vastly different regional carbon intensities ($gCO_2/kWh$) and renewable energy availability.

This project implements a complete, working simulation platform and web application introducing **three major research contributions**:
1. **Software-Enforced Resource Isolation**: Logical GPU compute, memory, and time quotas managed via `SoftwareIsolationManager` with quota checking, ceiling limits, and violation detection.
2. **Predictive GPU Scheduling**: Deep learning workload forecasting using a **PyTorch Sequence-based LSTM** neural network (`WorkloadPredictor`).
3. **Carbon-Aware Scheduling**: Multi-objective scheduler (`ProposedScheduler`) balancing energy consumption, regional grid carbon intensity, deadline constraints, and software isolation safety.

---

## 🔬 Important Research Distinction

> [!IMPORTANT]
> **Software-Enforced Resource Isolation vs. Hardware Partitioning**  
> The proposed system implements **Software-Enforced Resource Isolation** using explicit GPU compute, memory, and execution quotas rather than hardware-level partitioning (such as NVIDIA MIG). This approach minimizes cross-tenant interference while retaining high resource utilization and flexible multi-tenant GPU sharing without requiring dedicated hardware slicing support.

---

## 🏛️ System Architecture

```
                          USER
                           |
                           v
                  STREAMLIT DASHBOARD (10 Pages)
                           |
                           v
                    FASTAPI BACKEND
                           |
           +---------------+---------------+
           |                               |
           v                               v
    Firebase Firestore            Local Task Queue / State
     (with Fallback)                       |
                                           v
                                   Predictive Model (PyTorch LSTM)
                                           |
                                           v
                                   Future GPU Demand Forecast
                                           |
                                           v
                                Software Isolation Layer
                                           |
                                           v
                                   Candidate GPU List
                                           |
                                           v
                                 Carbon-Aware Scheduler
                                           |
                                           v
                                   Selected GPU Node
                                           |
                                           v
                                   Simulation Engine
                                           |
                                           v
                                    Metrics Engine
                                           |
                                           v
                               Firebase / Local Storage
```

---

## 📐 Multi-Objective Scoring Formula

For every arriving GPU task, the **Proposed Scheduler** evaluates all candidate GPUs meeting basic capacity and isolation constraints, calculating a composite score $S \in [0, 1]$:

$$\text{Score} = w_{\text{res}} \cdot S_{\text{resource}} + w_{\text{pred}} \cdot S_{\text{prediction}} + w_{\text{carb}} \cdot S_{\text{carbon}} + w_{\text{dead}} \cdot S_{\text{deadline}} + w_{\text{iso}} \cdot S_{\text{isolation}}$$

Where:
- $S_{\text{resource}}$: Normalized remaining compute and memory headroom on candidate GPU.
- $S_{\text{prediction}}$: Inverse of projected peak GPU utilization predicted by the PyTorch LSTM model.
- $S_{\text{carbon}}$: Carbon efficiency score based on power draw ($W$), regional carbon intensity ($gCO_2/kWh$), and renewable energy ratio.
- $S_{\text{deadline}}$: Time slack margin before deadline expiry.
- $S_{\text{isolation}}$: Isolation boundary rating derived from `SoftwareIsolationManager` policy checks.

*All weights ($w_{\text{res}}, w_{\text{pred}}, w_{\text{carb}}, w_{\text{dead}}, w_{\text{iso}}$) sum to $1.0$ and are dynamically adjustable from the dashboard.*

---

## 🚀 Quick Start & Installation

### 1. Prerequisites
- Python 3.10 or higher
- `pip` package manager

### 2. Setup Virtual Environment
```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
# On Mac/Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Generate Sample Datasets & Pretrain Model
```bash
# Generate synthetic GPU configs & 100 sample tasks
python3 data/generate_sample_data.py
```

### 4. Start the Application

**Option A: Start FastAPI Backend Service**
```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
*API Swagger Documentation available at `http://127.0.0.1:8000/docs`.*

**Option B: Start Streamlit Dashboard**
```bash
streamlit run frontend/dashboard.py
```
*Open `http://localhost:8501` in your browser.*

---

## 📊 Streamlit Dashboard Pages

The application includes **10 comprehensive pages**:
1. **📊 Overview**: High-level KPI summary, active GPU utilization, regional carbon grid map.
2. **📥 Submit Task**: Interactive form to submit new GPU tasks with custom isolation levels and carbon preferences.
3. **🖥️ GPU Cluster**: Real-time telemetry, power draw, and active workload breakdown per node.
4. **📋 Task Queue**: Filterable data grid of queued, executing, and completed tasks.
5. **🔮 Predictive Scheduling**: PyTorch LSTM workload forecast visualizations, sequence windowing, and MAE/RMSE metrics.
6. **🌱 Carbon Analytics**: Total kWh energy consumption, carbon emissions ($gCO_2$) by GPU and task.
7. **🛡️ Software Isolation**: Logical quota allocation boundaries, quota breach detection log, and enforcement actions.
8. **⚖️ Scheduler Comparison**: Side-by-side benchmark comparison (**Baseline vs. Predictive vs. Proposed**) across 8 key performance metrics with Plotly charts and CSV export.
9. **🎮 Simulation Engine**: Step-by-step discrete clock controls (`START`, `PAUSE`, `RESET`, `STEP`).
10. **⚙️ Settings**: Interactive sliders to tweak scheduler weight coefficients in real time.

---

## 🧪 Running Unit Tests

Run the automated test suite using `pytest`:
```bash
pytest -v tests/
```

Tests cover:
- Baseline, Predictive, Carbon, and Proposed Schedulers
- Software Isolation Manager quota checks and violation detection
- PyTorch LSTM model preparation, training, and prediction
- Discrete-event simulator execution and metric engine calculations

---

## 📁 Repository Structure

```
├── backend/
│   ├── config.py              # System settings & environment variables
│   ├── database.py            # Local DB / SQLite fallback manager
│   ├── firebase_service.py    # Firebase Firestore integration & fallback
│   ├── main.py                # FastAPI REST API application
│   └── models.py              # Pydantic data schemas
├── data/
│   ├── generate_sample_data.py # Synthetic task and GPU dataset generator
│   ├── preprocess_alibaba.py  # Alibaba GPU Cluster Trace importer
│   ├── sample_tasks.csv       # 100 sample workload tasks
│   └── gpu_config.csv         # 4-GPU cloud cluster configuration
├── frontend/
│   └── dashboard.py           # Multi-page Streamlit Dashboard
├── isolation/
│   └── isolation_manager.py   # SoftwareIsolationManager class
├── models/
│   ├── lstm_model.py          # PyTorch LSTM neural network architecture
│   └── workload_predictor.py  # WorkloadPredictor training & inference wrapper
├── schedulers/
│   ├── base_scheduler.py      # Baseline First-Fit / Least-Loaded Scheduler
│   ├── carbon_scheduler.py    # Carbon-Aware Energy & Footprint Scheduler
│   ├── predictive_scheduler.py# Predictive LSTM Forecast Scheduler
│   └── proposed_scheduler.py  # Multi-Objective Proposed Scheduler
├── simulation/
│   ├── metrics.py             # MetricsEngine & Jain's Fairness Index
│   └── simulator.py           # GPUSimulator discrete-event engine
├── tests/
│   ├── test_isolation.py      # Quota & violation tests
│   ├── test_predictor.py      # LSTM model tests
│   ├── test_scheduler.py      # Multi-scheduler tests
│   └── test_simulator.py      # Simulation engine tests
├── .env.example               # Environment variables template
├── requirements.txt           # Python library dependencies
└── README.md                  # Project documentation & Viva guide
```

---

## 🎓 Academic Viva & Presentation Guide

When presenting this project for an academic defense or viva:
1. **Problem Statement**: Explain how uncoordinated GPU sharing in cloud environments causes inter-tenant interference and excessive carbon emissions.
2. **Base Paper Context**: Note that the base paper focused on dynamic scheduling with Alibaba traces, while your project adds **software resource isolation**, **LSTM workload prediction**, and **carbon grid awareness**.
3. **Software Isolation**: Emphasize that `SoftwareIsolationManager` enforces compute/memory quota boundaries in software without requiring expensive hardware MIG slicing.
4. **Predictive Model**: Highlight that the PyTorch LSTM neural network predicts upcoming workload peaks to prevent placement on GPUs about to experience congestion.
5. **Experimental Results**: Use the **Scheduler Comparison** page to demonstrate empirical reductions in waiting time, carbon emissions ($gCO_2$), and isolation violations when using the **Proposed Scheduler**.
