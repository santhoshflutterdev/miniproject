import os
import sys

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as gg
import streamlit as st

# Application Imports
from backend.models import Task
from simulation.simulator import GPUSimulator
from schedulers.proposed_scheduler import ProposedScheduler
from data.generate_sample_data import generate_sample_tasks

# Configure Streamlit Page
st.set_page_config(
    page_title="Carbon-Aware GPU Scheduler",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Backend API Configuration
API_BASE_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# Custom CSS Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00E5FF, #7C4DFF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #B0BEC5;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1E2430;
        border-radius: 10px;
        padding: 15px;
        border-left: 4px solid #00E5FF;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .metric-title {
        font-size: 0.85rem;
        color: #90A4AE;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #FFFFFF;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Helper API Fetchers
def fetch_api(endpoint: str, default_val=None, method: str = "GET", payload=None):
    try:
        url = f"{API_BASE_URL}{endpoint}"
        if method == "POST":
            res = requests.post(url, json=payload, timeout=5)
        else:
            res = requests.get(url, timeout=5)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return default_val

# Direct Local Fallback Engine if API Server is initializing
if "local_simulator" not in st.session_state:
    st.session_state.local_simulator = GPUSimulator(scheduler_type="proposed")
    if os.path.exists("data/sample_tasks.csv"):
        df_sample = pd.read_csv("data/sample_tasks.csv")
        st.session_state.local_simulator.load_tasks_from_dataframe(df_sample)

sim = st.session_state.local_simulator

# Sidebar Navigation
st.sidebar.markdown("## ⚡ GPU Scheduler")
st.sidebar.markdown("Predictive, Carbon-Aware & Isolated Cloud Scheduling")

page = st.sidebar.radio("Navigation", [
    "📊 Overview",
    "📥 Submit Task",
    "🖥️ GPU Cluster",
    "📋 Task Queue",
    "🔮 Predictive Scheduling",
    "🌱 Carbon Analytics",
    "🛡️ Software Isolation",
    "⚖️ Scheduler Comparison",
    "🎮 Simulation Engine",
    "⚙️ Settings"
])

# System Status Banner
api_health = fetch_api("/health", {"status": "offline"})
is_api_online = api_health.get("status") == "online"

if not is_api_online:
    st.sidebar.info("💡 Running in Direct Simulation Engine mode")
else:
    st.sidebar.success("🟢 FastAPI Backend Connected")

# ==========================================
# 1. OVERVIEW PAGE
# ==========================================
if page == "📊 Overview":
    st.markdown('<div class="main-header">Predictive & Carbon-Aware GPU Scheduler</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Cloud GPU Workload Optimization with Software-Enforced Resource Isolation</div>', unsafe_allow_html=True)

    metrics = sim.get_summary_metrics().dict()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Completed Tasks</div>
            <div class="metric-value">{metrics['completed_tasks']} / {metrics['total_tasks']}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #7C4DFF;">
            <div class="metric-title">Avg GPU Utilization</div>
            <div class="metric-value">{metrics['avg_gpu_utilization']}%</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #00E676;">
            <div class="metric-title">Carbon Footprint</div>
            <div class="metric-value">{metrics['total_carbon_gco2']} <span style="font-size:0.9rem">gCO₂</span></div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #FF5252;">
            <div class="metric-title">Isolation Violations</div>
            <div class="metric-value">{metrics['isolation_violations']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### ")
    
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("#### 🖥️ Active GPU Utilization")
        gpus = [g.dict() for g in sim.gpus]
        gpu_df = pd.DataFrame(gpus)
        fig_gpu = px.bar(
            gpu_df, x="gpu_id", y="current_utilization", color="location",
            labels={"current_utilization": "Utilization (%)", "gpu_id": "GPU Node"},
            range_y=[0, 100], text="current_utilization",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_gpu.update_layout(template="plotly_dark", height=320)
        st.plotly_chart(fig_gpu, use_container_width=True)

    with col_right:
        st.markdown("#### 🌍 Grid Carbon Intensity by Region")
        fig_carbon = px.bar(
            gpu_df, x="gpu_id", y="carbon_intensity", color="renewable_ratio",
            labels={"carbon_intensity": "Carbon Intensity (gCO₂/kWh)", "renewable_ratio": "Renewable %"},
            text="carbon_intensity", color_continuous_scale="Viridis"
        )
        fig_carbon.update_layout(template="plotly_dark", height=320)
        st.plotly_chart(fig_carbon, use_container_width=True)

# ==========================================
# 2. SUBMIT TASK PAGE
# ==========================================
elif page == "📥 Submit Task":
    st.markdown('<div class="main-header">Submit New GPU Task</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Configure task requirements for intelligent scheduling</div>', unsafe_allow_html=True)

    with st.form("task_form"):
        col1, col2 = st.columns(2)
        with col1:
            user_id = st.text_input("User ID", value="U101")
            gpu_demand = st.slider("GPU Compute Demand (%)", 5, 100, 40, step=5)
            memory_demand = st.select_slider("GPU Memory Demand (MB)", options=[2048, 4096, 8192, 16384, 24576, 32768, 81920], value=8192)
            duration = st.number_input("Estimated Duration (Minutes)", min_value=1, max_value=300, value=20)
        with col2:
            priority = st.selectbox("Priority Level", ["low", "medium", "high"], index=1)
            deadline = st.number_input("Deadline Relative Time (Minutes)", min_value=5, max_value=500, value=60)
            isolation_level = st.selectbox("Software Isolation Level", ["soft", "strong"], index=1)
            carbon_preference = st.selectbox("Carbon Preference", ["low", "standard", "high"], index=0)

        submitted = st.form_submit_button("🚀 Submit & Schedule Task")

    if submitted:
        task_count = len(sim.queued_tasks) + len(sim.running_tasks) + len(sim.completed_tasks) + 1
        new_task = Task(
            task_id=f"T{task_count:03d}",
            user_id=user_id,
            gpu_demand=float(gpu_demand),
            memory_demand=float(memory_demand),
            duration=float(duration),
            priority=priority,
            deadline=float(deadline),
            isolation_level=isolation_level,
            carbon_preference=carbon_preference,
            arrival_time=sim.current_time
        )
        sim.add_task(new_task)
        st.success(f"Task **{new_task.task_id}** submitted successfully!")
        
        # Immediate placement preview
        gpu = sim._select_gpu_for_task(new_task)
        if gpu:
            st.info(f"📍 **Assigned Target GPU**: `{gpu.gpu_id}` ({gpu.name}) | Grid Carbon Intensity: `{gpu.carbon_intensity} gCO₂/kWh`")
        else:
            st.warning("⚠️ No immediately feasible GPU available with requested isolation boundary. Queued for next window.")

# ==========================================
# 3. GPU CLUSTER PAGE
# ==========================================
elif page == "🖥️ GPU Cluster":
    st.markdown('<div class="main-header">GPU Cluster Node Monitors</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Real-time status, power draw, and tenant workload breakdown</div>', unsafe_allow_html=True)

    cols = st.columns(2)
    for idx, gpu in enumerate(sim.gpus):
        with cols[idx % 2]:
            st.markdown(f"### {gpu.gpu_id} — {gpu.name}")
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Utilization", f"{gpu.current_utilization:.1f}%")
            col_b.metric("Power Draw", f"{gpu.power_watts} W")
            col_c.metric("Carbon Intensity", f"{gpu.carbon_intensity} gCO₂")

            st.progress(gpu.current_utilization / 100.0)
            st.caption(f"📍 Location: `{gpu.location}` | Memory: `{gpu.available_memory:.0f} / {gpu.total_memory:.0f} MB` | Active Tasks: `{len(gpu.running_tasks)}`")
            st.markdown("---")

# ==========================================
# 4. TASK QUEUE PAGE
# ==========================================
elif page == "📋 Task Queue":
    st.markdown('<div class="main-header">Cloud Task Queue & Execution Table</div>', unsafe_allow_html=True)

    all_tasks = [t.dict() for t in (sim.queued_tasks + sim.running_tasks + sim.completed_tasks)]
    if all_tasks:
        df_tasks = pd.DataFrame(all_tasks)
        st.dataframe(
            df_tasks[["task_id", "user_id", "status", "gpu_demand", "memory_demand", "duration", "assigned_gpu", "isolation_level", "carbon_emission"]],
            use_container_width=True
        )
    else:
        st.info("No tasks currently loaded in queue.")

# ==========================================
# 5. PREDICTIVE SCHEDULING PAGE
# ==========================================
elif page == "🔮 Predictive Scheduling":
    st.markdown('<div class="main-header">PyTorch LSTM Workload Forecasting</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Deep Learning prediction of future cluster GPU demand spikes</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("#### Model Controls")
        epochs = st.slider("Training Epochs", 10, 100, 30, step=10)
        if st.button("🧠 Train PyTorch LSTM Model"):
            with st.spinner("Training sequence-based LSTM workload model..."):
                if os.path.exists("data/sample_tasks.csv"):
                    df = pd.read_csv("data/sample_tasks.csv")
                else:
                    generate_sample_tasks()
                    df = pd.read_csv("data/sample_tasks.csv")
                eval_res = sim.predictor.train(df, feature_col="gpu_demand", epochs=epochs)
                sim.predictor.save_model("models/lstm_workload.pt")
                st.success(f"Model trained! MAE: {eval_res['mae']:.4f}, RMSE: {eval_res['rmse']:.4f}")

        st.markdown("---")
        st.metric("Model Status", "Trained 🟢" if sim.predictor.is_trained else "Untrained 🔴")
        st.metric("Mean Absolute Error (MAE)", f"{sim.predictor.last_mae:.4f}")
        st.metric("Root Mean Sq Error (RMSE)", f"{sim.predictor.last_rmse:.4f}")

    with col2:
        st.markdown("#### Actual vs Predicted GPU Workload Demand")
        history = sim.gpu_util_history.get("GPU1", [30.0, 35.0, 42.0, 50.0, 48.0, 55.0, 60.0])
        future_pred = sim.predictor.predict(history)
        
        time_steps = list(range(1, len(history) + 1))
        fig = gg.Figure()
        fig.add_trace(gg.Scatter(x=time_steps, y=history, mode='lines+markers', name='Actual GPU Utilization (%)', line=dict(color='#00E5FF', width=3)))
        fig.add_trace(gg.Scatter(x=[len(history) + 1], y=[future_pred], mode='markers', name='LSTM Next Window Forecast', marker=dict(color='#FF5252', size=14, symbol='star')))
        
        fig.update_layout(template="plotly_dark", height=380, xaxis_title="Time Step Window", yaxis_title="GPU Demand (%)")
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 6. CARBON ANALYTICS PAGE
# ==========================================
elif page == "🌱 Carbon Analytics":
    st.markdown('<div class="main-header">Carbon & Energy Footprint Tracker</div>', unsafe_allow_html=True)
    
    metrics = sim.get_summary_metrics().dict()
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Energy Consumed", f"{metrics['total_energy_kwh']} kWh")
    col2.metric("Total Carbon Emissions", f"{metrics['total_carbon_gco2']} gCO₂")
    col3.metric("Avg Carbon per Task", f"{metrics['avg_carbon_per_task']} gCO₂")

    completed = [t.dict() for t in sim.completed_tasks]
    if completed:
        df_comp = pd.DataFrame(completed)
        fig_carbon_task = px.histogram(
            df_comp, x="assigned_gpu", y="carbon_emission", color="carbon_preference",
            title="Carbon Footprint by GPU Node and Task Preference",
            barmode="group", template="plotly_dark"
        )
        st.plotly_chart(fig_carbon_task, use_container_width=True)
    else:
        st.info("Run simulation steps to view completed task carbon emissions breakdown.")

# ==========================================
# 7. SOFTWARE ISOLATION PAGE
# ==========================================
elif page == "🛡️ Software Isolation":
    st.markdown('<div class="main-header">Software-Enforced Resource Isolation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Logical resource boundaries, compute/memory quotas, and interference tracking</div>', unsafe_allow_html=True)

    violations = [v.dict() for v in sim.isolation_manager.violations]
    st.metric("Total Recorded Isolation Violations", len(violations))

    if violations:
        st.markdown("#### Recorded Quota Breach & Isolation Events")
        df_v = pd.DataFrame(violations)
        st.dataframe(df_v[["violation_id", "task_id", "user_id", "gpu_id", "violation_type", "description"]], use_container_width=True)
    else:
        st.success("🛡️ No software isolation quota breaches detected in current run.")

# ==========================================
# 8. SCHEDULER COMPARISON PAGE
# ==========================================
elif page == "⚖️ Scheduler Comparison":
    st.markdown('<div class="main-header">Comparative Experimental Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Baseline (Least-Loaded) vs Predictive vs Proposed Multi-Objective Scheduler</div>', unsafe_allow_html=True)

    if st.button("⚡ Run Comparative Experiment Benchmark"):
        with st.spinner("Executing comparative experiments across all 3 schedulers..."):
            results = []
            df_sample = pd.read_csv("data/sample_tasks.csv") if os.path.exists("data/sample_tasks.csv") else generate_sample_tasks()
            
            for alg in ["baseline", "predictive", "proposed"]:
                test_sim = GPUSimulator(scheduler_type=alg)
                test_sim.load_tasks_from_dataframe(df_sample.copy())
                test_sim.run_all(max_steps=200)
                m = test_sim.get_summary_metrics().dict()
                results.append(m)
            
            st.session_state.comp_df = pd.DataFrame(results)

    if "comp_df" in st.session_state:
        df_res = st.session_state.comp_df
        st.dataframe(df_res, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            fig_wait = px.bar(df_res, x="scheduler_type", y="avg_waiting_time", color="scheduler_type", title="Average Waiting Time (min)", template="plotly_dark")
            st.plotly_chart(fig_wait, use_container_width=True)
        with col_b:
            fig_carbon = px.bar(df_res, x="scheduler_type", y="total_carbon_gco2", color="scheduler_type", title="Total Carbon Emissions (gCO₂)", template="plotly_dark")
            st.plotly_chart(fig_carbon, use_container_width=True)

        csv_bytes = df_res.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Export Comparative Results CSV", csv_bytes, "scheduler_comparison_results.csv", "text/csv")

# ==========================================
# 9. SIMULATION ENGINE PAGE
# ==========================================
elif page == "🎮 Simulation Engine":
    st.markdown('<div class="main-header">Interactive Simulation Controls</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("▶️ Step Next Minute"):
            res = sim.step()
            st.success(f"Time {res['current_time']:.0f}m step executed.")
    with col2:
        if st.button("⏩ Run All Tasks"):
            with st.spinner("Simulating task execution..."):
                sim.run_all()
            st.success("All tasks executed.")
    with col3:
        if st.button("🔄 Reset Simulation"):
            sim.reset()
            if os.path.exists("data/sample_tasks.csv"):
                df_s = pd.read_csv("data/sample_tasks.csv")
                sim.load_tasks_from_dataframe(df_s)
            st.info("Simulation reset.")
    with col4:
        scheduler_choice = st.selectbox("Active Scheduler", ["proposed", "predictive", "baseline"], index=0)
        sim.set_scheduler_type(scheduler_choice)

    st.markdown(f"### Current Simulation Clock: `{sim.current_time:.0f} Minutes`")
    st.markdown(f"**Queued**: `{len(sim.queued_tasks)}` | **Running**: `{len(sim.running_tasks)}` | **Completed**: `{len(sim.completed_tasks)}`")

# ==========================================
# 10. SETTINGS PAGE
# ==========================================
elif page == "⚙️ Settings":
    st.markdown('<div class="main-header">Algorithm Weights & System Settings</div>', unsafe_allow_html=True)

    st.markdown("#### Multi-Objective Weight Coefficients")
    w_res = st.slider("Resource Availability Weight (w_res)", 0.0, 1.0, 0.30, 0.05)
    w_pred = st.slider("Prediction Forecast Weight (w_pred)", 0.0, 1.0, 0.25, 0.05)
    w_carb = st.slider("Carbon Footprint Weight (w_carb)", 0.0, 1.0, 0.25, 0.05)
    w_dead = st.slider("Deadline Urgency Weight (w_dead)", 0.0, 1.0, 0.10, 0.05)
    w_iso = st.slider("Software Isolation Weight (w_iso)", 0.0, 1.0, 0.10, 0.05)

    if st.button("💾 Save Algorithm Weights"):
        weights = {"resource": w_res, "prediction": w_pred, "carbon": w_carb, "deadline": w_dead, "isolation": w_iso}
        # Update proposed scheduler weights
        ps = ProposedScheduler(sim.gpus)
        ps.set_weights(weights)
        st.success("Scheduler weight parameters saved successfully!")
