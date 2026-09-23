// Global State
let currentResults = [];
let activeChartType = 'carbon';
let benchmarkChartInstance = null;

// DOM Elements
const runSimBtn = document.getElementById('run-sim-btn');
const refreshDataBtn = document.getElementById('refresh-data-btn');
const tableBody = document.getElementById('table-body');
const gpuClusterGrid = document.getElementById('gpu-cluster-grid');
const tabButtons = document.querySelectorAll('.tab-btn');

// Metric Elements
const metricWaiting = document.getElementById('metric-waiting');
const metricEnergy = document.getElementById('metric-energy');
const metricCarbon = document.getElementById('metric-carbon');
const metricImbalance = document.getElementById('metric-imbalance');

// Initialize Application
document.addEventListener('DOMContentLoaded', () => {
    fetchResults();
    fetchGpuStatus();
    setupEventListeners();
});

// Setup Event Listeners
function setupEventListeners() {
    runSimBtn.addEventListener('click', handleRunSimulation);
    refreshDataBtn.addEventListener('click', () => {
        showToast('Refreshing latest benchmarks...', 'info');
        fetchResults();
        fetchGpuStatus();
    });

    tabButtons.forEach(btn => {
        btn.addEventListener('click', (e) => {
            tabButtons.forEach(b => b.classList.remove('active'));
            e.target.classList.add('active');
            activeChartType = e.target.dataset.chart;
            updateChart();
        });
    });
}

// Fetch Benchmark Results API
async function fetchResults() {
    try {
        const response = await fetch('/api/results');
        const result = await response.json();
        
        if (result.status === 'success' && Array.isArray(result.data)) {
            currentResults = result.data;
            updateMetrics(currentResults);
            updateTable(currentResults);
            updateChart();
        } else {
            showToast('Failed to load results from server', 'error');
        }
    } catch (error) {
        console.error('Error fetching results:', error);
        showToast('Network error while loading results', 'error');
    }
}

// Fetch GPU Cluster Status API
async function fetchGpuStatus() {
    try {
        const response = await fetch('/api/gpu-status');
        const result = await response.json();
        
        if (result.status === 'success' && Array.isArray(result.data)) {
            renderGpuCards(result.data);
        }
    } catch (error) {
        console.error('Error fetching GPU status:', error);
    }
}

// Handle Run Simulation Event
async function handleRunSimulation() {
    runSimBtn.disabled = true;
    runSimBtn.innerHTML = `<span class="btn-icon">⏳</span><span class="btn-text">Running Pipeline...</span>`;
    showToast('Simulation started. Training neural network & testing schedulers...', 'info');

    try {
        const response = await fetch('/api/run-simulation', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        const result = await response.json();

        if (result.status === 'success') {
            currentResults = result.data;
            updateMetrics(currentResults);
            updateTable(currentResults);
            updateChart();
            showToast('Simulation complete! Firebase & CSV updated successfully. 🎉', 'success');
        } else {
            showToast(`Simulation failed: ${result.message}`, 'error');
        }
    } catch (error) {
        console.error('Simulation execution error:', error);
        showToast('Server error during simulation run.', 'error');
    } finally {
        runSimBtn.disabled = false;
        runSimBtn.innerHTML = `<span class="btn-icon">▶</span><span class="btn-text">Run Live Simulation</span>`;
    }
}

// Update Top KPI Metrics
function updateMetrics(results) {
    const proposed = results.find(r => r.Algorithm === 'Proposed');
    if (proposed) {
        metricWaiting.textContent = Number(proposed.Waiting_Time).toFixed(2);
        metricEnergy.textContent = Number(proposed.Energy).toFixed(2);
        metricCarbon.textContent = Number(proposed.Carbon).toFixed(2);
        metricImbalance.textContent = Number(proposed.Load_Imbalance).toFixed(1);
    } else {
        metricWaiting.textContent = '0.00';
        metricEnergy.textContent = '0.52';
        metricCarbon.textContent = '135.50';
        metricImbalance.textContent = '55.0';
    }
}

// Render GPU Cards
function renderGpuCards(gpus) {
    gpuClusterGrid.innerHTML = '';
    
    gpus.forEach(gpu => {
        const card = document.createElement('div');
        card.className = 'gpu-card';

        let badgeClass = 'badge-success';
        if (gpu.carbon_intensity > 500) badgeClass = 'badge-rose';
        else if (gpu.carbon_intensity > 300) badgeClass = 'badge-amber';

        let barColor = '#10b981';
        if (gpu.simulated_load > 70) barColor = '#06b6d4';

        card.innerHTML = `
            <div class="gpu-card-header">
                <span class="gpu-title">${gpu.name}</span>
                <span class="badge ${badgeClass}">${gpu.status}</span>
            </div>
            <div class="gpu-region">${gpu.region}</div>
            
            <div class="gpu-stat-row">
                <span class="gpu-stat-label">Carbon Intensity</span>
                <span class="gpu-stat-val">${gpu.carbon_intensity} gCO₂/kWh</span>
            </div>
            <div class="gpu-stat-row">
                <span class="gpu-stat-label">Power Specs</span>
                <span class="gpu-stat-val">${gpu.idle_power}W / ${gpu.max_power}W</span>
            </div>

            <div class="gpu-stat-row" style="margin-top: 10px;">
                <span class="gpu-stat-label">Simulated Cluster Load</span>
                <span class="gpu-stat-val">${gpu.simulated_load}%</span>
            </div>
            <div class="progress-bar-bg">
                <div class="progress-bar-fill" style="width: ${gpu.simulated_load}%; background-color: ${barColor};"></div>
            </div>
        `;

        gpuClusterGrid.appendChild(card);
    });
}

// Update Benchmark Data Table
function updateTable(results) {
    tableBody.innerHTML = '';

    results.forEach(row => {
        const tr = document.createElement('tr');
        const isProposed = row.Algorithm === 'Proposed';
        if (isProposed) tr.className = 'proposed-row';

        let icon = '⚙️';
        if (row.Algorithm === 'Baseline') icon = '🔴';
        else if (row.Algorithm === 'Predictive') icon = '🟡';
        else if (row.Algorithm === 'Proposed') icon = '🟢';

        tr.innerHTML = `
            <td>
                <div class="strategy-cell">
                    <span>${icon}</span>
                    <span>${row.Algorithm} ${isProposed ? '<span class="badge badge-success" style="margin-left:8px;">Proposed</span>' : ''}</span>
                </div>
            </td>
            <td><strong>${Number(row.Waiting_Time).toFixed(2)}</strong></td>
            <td>${Number(row.Turnaround_Time).toFixed(2)}</td>
            <td>${Number(row.Energy).toFixed(2)}</td>
            <td><strong>${Number(row.Carbon).toFixed(2)}</strong></td>
            <td>${Number(row.Load_Imbalance).toFixed(2)}%</td>
        `;

        tableBody.appendChild(tr);
    });
}

// Update Chart.js Instance
function updateChart() {
    const ctx = document.getElementById('benchmarkChart').getContext('2d');
    
    if (currentResults.length === 0) return;

    const labels = currentResults.map(r => r.Algorithm);
    let data = [];
    let labelText = '';
    let barColors = ['#f43f5e', '#f59e0b', '#10b981'];

    if (activeChartType === 'carbon') {
        data = currentResults.map(r => r.Carbon);
        labelText = 'Carbon Emission (gCO₂)';
    } else if (activeChartType === 'energy') {
        data = currentResults.map(r => r.Energy);
        labelText = 'Energy Consumption (kWh)';
    } else if (activeChartType === 'waiting') {
        data = currentResults.map(r => r.Waiting_Time);
        labelText = 'Average Waiting Time (min)';
    }

    if (benchmarkChartInstance) {
        benchmarkChartInstance.destroy();
    }

    benchmarkChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: labelText,
                data: data,
                backgroundColor: barColors,
                borderRadius: 8,
                borderSkipped: false
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    display: true,
                    labels: { color: '#94a3b8', font: { family: 'Inter', size: 12 } }
                },
                tooltip: {
                    backgroundColor: '#1e293b',
                    titleColor: '#ffffff',
                    bodyColor: '#cbd5e1',
                    borderColor: 'rgba(255, 255, 255, 0.1)',
                    borderWidth: 1,
                    padding: 12,
                    displayColors: false
                }
            },
            scales: {
                x: {
                    ticks: { color: '#94a3b8', font: { family: 'Inter', size: 13 } },
                    grid: { display: false }
                },
                y: {
                    ticks: { color: '#94a3b8', font: { family: 'Inter', size: 12 } },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' }
                }
            }
        }
    });
}

// Toast Notifications Helper
function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    
    let icon = 'ℹ️';
    if (type === 'success') icon = '✅';
    if (type === 'error') icon = '❌';

    toast.innerHTML = `<span>${icon}</span><span>${message}</span>`;
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(100%)';
        toast.style.transition = 'all 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}
