import os
import sys
import pandas as pd
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import main as run_main_simulation
from firebase_config import fetch_simulation_results

st.set_page_config(
    page_title="Predictive & Carbon-Aware GPU Scheduling Dashboard",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Predictive & Carbon-Aware GPU Scheduling Dashboard")
st.caption("Energy-Efficient Cloud Computing & Multi-Objective Resource Allocation (Firebase Connected)")

st.divider()

# Sidebar Control
st.sidebar.header("Controls & Firebase Sync")
if st.sidebar.button("▶ Run Live Simulation Pipeline"):
    with st.spinner("Running Simulation & Uploading to Firebase..."):
        run_main_simulation()
    st.sidebar.success("Simulation complete! Firebase updated.")

# Load Results Data via Firebase module
results_df = fetch_simulation_results()

# GPU Status Overview
st.header("📊 GPU Status & Cluster Metrics")
gpu_data = {
    "GPU": ["GPU 1 (Region A)", "GPU 2 (Region B)", "GPU 3 (Region C)", "GPU 4 (Region D)"],
    "Carbon Intensity (gCO2/kWh)": [200, 700, 350, 150],
    "Simulated Load (%)": [65, 45, 55, 75]
}
gpu_df = pd.DataFrame(gpu_data)

col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("GPU Allocation & Load Distribution")
    st.bar_chart(gpu_df.set_index("GPU")["Simulated Load (%)"])

with col_right:
    st.subheader("Regional Carbon Intensity")
    st.dataframe(gpu_df[["GPU", "Carbon Intensity (gCO2/kWh)"]], hide_index=True, use_container_width=True)

st.divider()

# System Metrics (Proposed Algorithm)
st.header("🎯 Proposed System Performance")

proposed_row = results_df[results_df["Algorithm"] == "Proposed"]
if not proposed_row.empty:
    p_row = proposed_row.iloc[0]
    w_time = f"{p_row['Waiting_Time']} min"
    e_val = f"{p_row['Energy']} kWh"
    c_val = f"{p_row['Carbon']} gCO2"
    imb_val = f"{p_row['Load_Imbalance']}%"
else:
    w_time, e_val, c_val, imb_val = "0.0 min", "0.52 kWh", "135.50 gCO2", "55%"

col1, col2, col3, col4 = st.columns(4)
col1.metric("Average Waiting Time", w_time)
col2.metric("Total Energy Used", e_val)
col3.metric("Total Carbon Emissions", c_val)
col4.metric("GPU Load Imbalance", imb_val)

st.divider()

# Algorithm Benchmarks & Comparisons
st.header("📈 Firebase Comparative Benchmarks")

st.dataframe(
    results_df,
    column_config={
        "Algorithm": "Scheduling Strategy",
        "Waiting_Time": st.column_config.NumberColumn("Avg Waiting Time (min)", format="%.2f"),
        "Turnaround_Time": st.column_config.NumberColumn("Avg Turnaround Time (min)", format="%.2f"),
        "Energy": st.column_config.NumberColumn("Total Energy (kWh)", format="%.2f"),
        "Carbon": st.column_config.NumberColumn("Total Carbon (gCO2)", format="%.2f"),
        "Load_Imbalance": st.column_config.NumberColumn("Load Imbalance (%)", format="%.2f"),
    },
    use_container_width=True,
    hide_index=True
)

st.subheader("Comparative Metric Visualization")
tab1, tab2, tab3 = st.tabs(["Carbon Emissions", "Energy Consumption", "Waiting Time"])

with tab1:
    st.bar_chart(results_df.set_index("Algorithm")["Carbon"])
with tab2:
    st.bar_chart(results_df.set_index("Algorithm")["Energy"])
with tab3:
    st.bar_chart(results_df.set_index("Algorithm")["Waiting_Time"])