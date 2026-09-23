document.addEventListener("DOMContentLoaded", () => {
    // API Endpoint Base URL (Same origin or localhost)
    const API_BASE = "";

    // DOM Elements
    const navItems = document.querySelectorAll(".nav-item");
    const tabPanes = document.querySelectorAll(".tab-pane");

    // Navigation Tab Switching
    navItems.forEach(item => {
        item.addEventListener("click", () => {
            const targetTab = item.getAttribute("data-tab");

            navItems.forEach(i => i.classList.remove("active"));
            tabPanes.forEach(p => p.classList.remove("active"));

            item.classList.add("active");
            const activePane = document.getElementById(targetTab);
            if (activePane) activePane.classList.add("active");

            // Refresh charts when tab switches
            refreshDashboardData();
        });
    });

    // Toast Notification Helper
    function showToast(message, type = "info") {
        const toast = document.getElementById("toast");
        toast.textContent = message;
        toast.className = `toast toast-${type}`;
        toast.classList.remove("hidden");
        setTimeout(() => toast.classList.add("hidden"), 3500);
    }

    // Generic Fetch Wrapper
    async function apiFetch(endpoint, method = "GET", body = null) {
        try {
            const options = {
                method,
                headers: { "Content-Type": "application/json" }
            };
            if (body) options.body = JSON.stringify(body);
            const res = await fetch(`${API_BASE}${endpoint}`, options);
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            return await res.json();
        } catch (e) {
            console.warn(`API fetch error on ${endpoint}:`, e);
            return null;
        }
    }

    // Chart.js Instances Reference Store
    const charts = {};

    // Initialize Chart.js Charts
    function initCharts() {
        // 1. GPU Utilization Chart
        const ctxUtil = document.getElementById("chart-gpu-utilization").getContext("2d");
        charts.gpuUtil = new Chart(ctxUtil, {
            type: "bar",
            data: {
                labels: ["GPU1", "GPU2", "GPU3", "GPU4"],
                datasets: [{
                    label: "GPU Utilization (%)",
                    data: [0, 0, 0, 0],
                    backgroundColor: ["#00E5FF", "#7C4DFF", "#00E676", "#FF5252"],
                    borderRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: { y: { min: 0, max: 100, grid: { color: "rgba(255,255,255,0.08)" } } },
                plugins: { legend: { display: false } }
            }
        });

        // 2. Carbon Intensity Chart
        const ctxCarbon = document.getElementById("chart-carbon-intensity").getContext("2d");
        charts.carbonIntensity = new Chart(ctxCarbon, {
            type: "bar",
            data: {
                labels: ["GPU1", "GPU2", "GPU3", "GPU4"],
                datasets: [{
                    label: "Carbon Intensity (gCO₂/kWh)",
                    data: [600, 300, 100, 500],
                    backgroundColor: "rgba(0, 230, 118, 0.7)",
                    borderColor: "#00E676",
                    borderWidth: 1,
                    borderRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: { y: { grid: { color: "rgba(255,255,255,0.08)" } } },
                plugins: { legend: { display: false } }
            }
        });

        // 3. Predictive Workload Chart
        const ctxPred = document.getElementById("chart-predictive-workload").getContext("2d");
        charts.predictive = new Chart(ctxPred, {
            type: "line",
            data: {
                labels: ["T-6", "T-5", "T-4", "T-3", "T-2", "T-1", "Next Forecast"],
                datasets: [
                    {
                        label: "Actual GPU Demand (%)",
                        data: [30, 35, 42, 50, 48, 55, null],
                        borderColor: "#00E5FF",
                        backgroundColor: "rgba(0, 229, 255, 0.15)",
                        fill: true,
                        tension: 0.3
                    },
                    {
                        label: "PyTorch LSTM Forecast",
                        data: [null, null, null, null, null, 55, 62],
                        borderColor: "#FF5252",
                        borderDash: [5, 5],
                        pointStyle: "star",
                        pointRadius: 8
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: { y: { min: 0, max: 100, grid: { color: "rgba(255,255,255,0.08)" } } }
            }
        });

        // 4. Carbon Footprint Per GPU Chart
        const ctxCarbonGpu = document.getElementById("chart-carbon-per-gpu").getContext("2d");
        charts.carbonGpu = new Chart(ctxCarbonGpu, {
            type: "bar",
            data: {
                labels: ["GPU1 (US-East)", "GPU2 (US-West)", "GPU3 (EU-Nordic)", "GPU4 (Asia-East)"],
                datasets: [{
                    label: "Carbon Footprint (gCO₂)",
                    data: [120, 65, 15, 90],
                    backgroundColor: "#00E676",
                    borderRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: { y: { grid: { color: "rgba(255,255,255,0.08)" } } },
                plugins: { legend: { display: false } }
            }
        });

        // 5. Comparative Waiting Time Chart
        const ctxCompWait = document.getElementById("chart-comp-waiting").getContext("2d");
        charts.compWait = new Chart(ctxCompWait, {
            type: "bar",
            data: {
                labels: ["Baseline", "Predictive", "Proposed"],
                datasets: [{
                    label: "Avg Waiting Time (min)",
                    data: [12.4, 8.2, 4.1],
                    backgroundColor: ["#FF5252", "#FF9100", "#00E5FF"],
                    borderRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: { y: { grid: { color: "rgba(255,255,255,0.08)" } } },
                plugins: { legend: { display: false } }
            }
        });

        // 6. Comparative Carbon Emissions Chart
        const ctxCompCarbon = document.getElementById("chart-comp-carbon").getContext("2d");
        charts.compCarbon = new Chart(ctxCompCarbon, {
            type: "bar",
            data: {
                labels: ["Baseline", "Predictive", "Proposed"],
                datasets: [{
                    label: "Total Carbon (gCO₂)",
                    data: [4500, 3200, 1850],
                    backgroundColor: ["#FF5252", "#FF9100", "#00E676"],
                    borderRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: { y: { grid: { color: "rgba(255,255,255,0.08)" } } },
                plugins: { legend: { display: false } }
            }
        });
    }

    // Cache store for tasks to support live search & status filtering
    let allTasksCache = [];
    let activeTaskFilter = "all";

    // Refresh Dashboard Data from FastAPI Backend
    async function refreshDashboardData() {
        // 1. Fetch Metrics Summary
        const metrics = await apiFetch("/metrics");
        if (metrics) {
            document.getElementById("kpi-completed-tasks").textContent = `${metrics.completed_tasks} / ${metrics.total_tasks}`;
            const queuedEl = document.getElementById("kpi-queued-count");
            if (queuedEl) queuedEl.textContent = metrics.queued_tasks;

            document.getElementById("kpi-avg-util").textContent = `${metrics.avg_gpu_utilization}%`;
            document.getElementById("kpi-total-carbon").innerHTML = `${metrics.total_carbon_gco2} <span class="unit">gCO₂</span>`;
            document.getElementById("kpi-total-energy").textContent = metrics.total_energy_kwh;
            document.getElementById("kpi-violations").textContent = metrics.isolation_violations;

            document.getElementById("carbon-energy-val").textContent = `${metrics.total_energy_kwh} kWh`;
            document.getElementById("carbon-total-val").textContent = `${metrics.total_carbon_gco2} gCO₂`;
            document.getElementById("carbon-avg-val").textContent = `${metrics.avg_carbon_per_task} gCO₂`;

            // Sync Simulation Controls UI
            const clockValEl = document.getElementById("sim-clock-val");
            if (clockValEl) clockValEl.textContent = metrics.clock_time || 0;
            const simQueuedEl = document.getElementById("sim-queued-val");
            if (simQueuedEl) simQueuedEl.textContent = metrics.queued_tasks;
            const simRunningEl = document.getElementById("sim-running-val");
            if (simRunningEl) simRunningEl.textContent = metrics.running_tasks;
            const simCompletedEl = document.getElementById("sim-completed-val");
            if (simCompletedEl) simCompletedEl.textContent = metrics.completed_tasks;
        }

        // 2. Fetch GPU Nodes Telemetry
        const gpus = await apiFetch("/gpus");
        if (gpus && Array.isArray(gpus)) {
            renderGpuCards(gpus);
            if (charts.gpuUtil) {
                charts.gpuUtil.data.labels = gpus.map(g => g.gpu_id);
                charts.gpuUtil.data.datasets[0].data = gpus.map(g => g.current_utilization);
                charts.gpuUtil.update();
            }
            if (charts.carbonIntensity) {
                charts.carbonIntensity.data.labels = gpus.map(g => g.gpu_id);
                charts.carbonIntensity.data.datasets[0].data = gpus.map(g => g.carbon_intensity);
                charts.carbonIntensity.update();
            }
            if (charts.carbonGpu) {
                charts.carbonGpu.data.labels = gpus.map(g => `${g.gpu_id} (${g.location})`);
                charts.carbonGpu.data.datasets[0].data = gpus.map(g => (g.power_watts * 0.15 * (g.carbon_intensity / 1000.0)).toFixed(2));
                charts.carbonGpu.update();
            }
        }

        // 3. Fetch Tasks Queue
        const tasks = await apiFetch("/tasks");
        if (tasks && Array.isArray(tasks)) {
            allTasksCache = tasks;
            applyTaskFilters();
        }

        // 4. Fetch Predictions
        const predData = await apiFetch("/predictions");
        if (predData) {
            document.getElementById("pred-model-status").textContent = predData.is_trained ? "Trained 🟢" : "Untrained 🔴";
            document.getElementById("pred-mae").textContent = (predData.mae || 0).toFixed(4);
            document.getElementById("pred-rmse").textContent = (predData.rmse || 0).toFixed(4);

            if (charts.predictive && predData.recent_history) {
                const history = predData.recent_history;
                const labels = history.map((_, idx) => `T-${history.length - idx}`);
                labels.push("Next Forecast");
                const actualSeries = [...history, null];
                const forecastSeries = new Array(history.length - 1).fill(null);
                forecastSeries.push(history[history.length - 1]);
                forecastSeries.push(predData.next_prediction);

                charts.predictive.data.labels = labels;
                charts.predictive.data.datasets[0].data = actualSeries;
                charts.predictive.data.datasets[1].data = forecastSeries;
                charts.predictive.update();
            }
        }

        // 5. Fetch Isolation Violations
        const violations = await apiFetch("/violations");
        if (violations && Array.isArray(violations)) {
            document.getElementById("badge-isolation-count").textContent = `${violations.length} Events`;
            renderIsolationTable(violations);
        }
    }

    // Render GPU Cards Component with Interactive Action Buttons
    function renderGpuCards(gpus) {
        const container = document.getElementById("gpu-nodes-container");
        container.innerHTML = gpus.map(gpu => `
            <div class="gpu-card">
                <div class="gpu-card-header">
                    <span class="gpu-name">${gpu.gpu_id} — ${gpu.name}</span>
                    <span class="badge ${gpu.current_utilization > 80 ? 'badge-danger' : 'badge-info'}">
                        ${gpu.current_utilization.toFixed(1)}% Util
                    </span>
                </div>
                <div class="progress-bar-bg">
                    <div class="progress-bar-fill" style="width: ${Math.min(100, gpu.current_utilization)}%"></div>
                </div>
                <div class="kpi-subtext mt-2 mb-3">
                    📍 Location: <strong>${gpu.location}</strong><br>
                    ⚡ Power Draw: <strong>${gpu.power_watts} W</strong> | Carbon: <strong>${gpu.carbon_intensity} gCO₂/kWh</strong><br>
                    💾 Memory: <strong>${gpu.available_memory.toFixed(0)} / ${gpu.total_memory.toFixed(0)} MB</strong><br>
                    📋 Active Tasks: <strong>${gpu.running_tasks ? gpu.running_tasks.length : 0}</strong>
                </div>
                <div class="controls-grid" style="gap: 8px;">
                    <button class="btn btn-secondary btn-sm gpu-spike-btn" data-gpu="${gpu.gpu_id}">⚡ Spike +25%</button>
                    <button class="btn btn-secondary btn-sm gpu-maint-btn" data-gpu="${gpu.gpu_id}">🛠️ Maintenance</button>
                </div>
            </div>
        `).join("");

        // Attach GPU Action Event Handlers
        document.querySelectorAll(".gpu-spike-btn").forEach(btn => {
            btn.addEventListener("click", async () => {
                const gpuId = btn.getAttribute("data-gpu");
                const res = await apiFetch(`/gpus/${gpuId}/spike?spike_amount=25.0`, "POST");
                if (res) {
                    showToast(`Simulated +25% load spike on ${gpuId}!`, "warning");
                    refreshDashboardData();
                }
            });
        });

        document.querySelectorAll(".gpu-maint-btn").forEach(btn => {
            btn.addEventListener("click", async () => {
                const gpuId = btn.getAttribute("data-gpu");
                const res = await apiFetch(`/gpus/${gpuId}/toggle-maintenance`, "POST");
                if (res) {
                    showToast(`Toggled ${gpuId} status to ${res.status}!`, "info");
                    refreshDashboardData();
                }
            });
        });
    }

    // Filter Tasks in Queue Table
    function applyTaskFilters() {
        const searchInput = document.getElementById("input-queue-search");
        const query = searchInput ? searchInput.value.trim().toLowerCase() : "";

        let filtered = allTasksCache;
        if (activeTaskFilter !== "all") {
            filtered = filtered.filter(t => t.status === activeTaskFilter);
        }
        if (query) {
            filtered = filtered.filter(t => t.task_id.toLowerCase().includes(query) || t.user_id.toLowerCase().includes(query));
        }
        renderTasksTable(filtered);
    }

    // Task Queue Search Input Listener
    const searchInput = document.getElementById("input-queue-search");
    if (searchInput) {
        searchInput.addEventListener("input", applyTaskFilters);
    }

    // Task Queue Status Filter Buttons
    document.querySelectorAll(".filter-task-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            document.querySelectorAll(".filter-task-btn").forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            activeTaskFilter = btn.getAttribute("data-filter");
            applyTaskFilters();
        });
    });

    // Render Tasks Table Component
    function renderTasksTable(tasks) {
        const tbody = document.getElementById("tasks-table-body");
        if (tasks.length === 0) {
            tbody.innerHTML = `<tr><td colspan="9" class="text-center">No matching tasks found</td></tr>`;
            return;
        }

        tbody.innerHTML = tasks.map(t => {
            let statusBadge = `<span class="badge badge-info">Queued</span>`;
            if (t.status === "running") statusBadge = `<span class="badge badge-warning">Running</span>`;
            if (t.status === "completed") statusBadge = `<span class="badge badge-success">Completed</span>`;

            return `
                <tr>
                    <td><strong>${t.task_id}</strong></td>
                    <td>${t.user_id}</td>
                    <td>${statusBadge}</td>
                    <td>${t.gpu_demand}%</td>
                    <td>${t.memory_demand} MB</td>
                    <td>${t.duration} min</td>
                    <td><code>${t.assigned_gpu || 'Unassigned'}</code></td>
                    <td><span class="badge ${t.isolation_level === 'strong' ? 'badge-danger' : 'badge-info'}">${t.isolation_level}</span></td>
                    <td>${t.carbon_emission ? t.carbon_emission.toFixed(2) : '0.00'}</td>
                </tr>
            `;
        }).join("");
    }

    // Render Isolation Violations Table
    function renderIsolationTable(violations) {
        const tbody = document.getElementById("isolation-table-body");
        if (violations.length === 0) {
            tbody.innerHTML = `<tr><td colspan="7" class="text-center">🛡️ No isolation quota breaches detected</td></tr>`;
            return;
        }

        tbody.innerHTML = violations.map(v => `
            <tr>
                <td><code>${v.violation_id}</code></td>
                <td><strong>${v.task_id}</strong></td>
                <td>${v.user_id}</td>
                <td><code>${v.gpu_id}</code></td>
                <td><span class="badge badge-danger">${v.violation_type}</span></td>
                <td>Min ${v.timestamp}</td>
                <td>${v.description}</td>
            </tr>
        `).join("");
    }

    // Form Range Dynamic Label Update
    const gpuDemandInput = document.getElementById("form-gpu-demand");
    const gpuDemandLabel = document.getElementById("val-gpu-demand");
    if (gpuDemandInput) {
        gpuDemandInput.addEventListener("input", (e) => {
            gpuDemandLabel.textContent = e.target.value;
        });
    }

    // Submit Task Presets Handlers
    document.getElementById("btn-preset-ai")?.addEventListener("click", () => {
        document.getElementById("form-user-id").value = "U_LLM_TRAIN";
        document.getElementById("form-gpu-demand").value = 90;
        document.getElementById("val-gpu-demand").textContent = "90";
        document.getElementById("form-memory-demand").value = "24576";
        document.getElementById("form-duration").value = 60;
        document.getElementById("form-priority").value = "high";
        document.getElementById("form-deadline").value = 120;
        document.getElementById("form-isolation").value = "strong";
        document.getElementById("form-carbon-pref").value = "low";
        showToast("Loaded AI Model Fine-Tuning Preset", "info");
    });

    document.getElementById("btn-preset-inference")?.addEventListener("click", () => {
        document.getElementById("form-user-id").value = "U_INFERENCE_API";
        document.getElementById("form-gpu-demand").value = 25;
        document.getElementById("val-gpu-demand").textContent = "25";
        document.getElementById("form-memory-demand").value = "4096";
        document.getElementById("form-duration").value = 10;
        document.getElementById("form-priority").value = "high";
        document.getElementById("form-deadline").value = 15;
        document.getElementById("form-isolation").value = "strong";
        document.getElementById("form-carbon-pref").value = "standard";
        showToast("Loaded Fast Inference API Preset", "info");
    });

    document.getElementById("btn-preset-batch")?.addEventListener("click", () => {
        document.getElementById("form-user-id").value = "U_BATCH_ETL";
        document.getElementById("form-gpu-demand").value = 60;
        document.getElementById("val-gpu-demand").textContent = "60";
        document.getElementById("form-memory-demand").value = "16384";
        document.getElementById("form-duration").value = 120;
        document.getElementById("form-priority").value = "low";
        document.getElementById("form-deadline").value = 300;
        document.getElementById("form-isolation").value = "soft";
        document.getElementById("form-carbon-pref").value = "low";
        showToast("Loaded Big Data Batch ETL Preset", "info");
    });

    // Submit Task Form Handler
    const taskForm = document.getElementById("task-submission-form");
    if (taskForm) {
        taskForm.addEventListener("submit", async (e) => {
            e.preventDefault();
            const payload = {
                user_id: document.getElementById("form-user-id").value,
                gpu_demand: parseFloat(document.getElementById("form-gpu-demand").value),
                memory_demand: parseFloat(document.getElementById("form-memory-demand").value),
                duration: parseFloat(document.getElementById("form-duration").value),
                priority: document.getElementById("form-priority").value,
                deadline: parseFloat(document.getElementById("form-deadline").value),
                isolation_level: document.getElementById("form-isolation").value,
                carbon_preference: document.getElementById("form-carbon-pref").value
            };

            const createdTask = await apiFetch("/tasks", "POST", payload);
            if (createdTask) {
                showToast(`Task ${createdTask.task_id} submitted successfully!`, "success");
                
                const resCard = document.getElementById("submit-result-card");
                const resDetails = document.getElementById("submit-result-details");
                resCard.classList.remove("hidden");
                resDetails.innerHTML = `
                    <p>✅ <strong>Task ID</strong>: <code>${createdTask.task_id}</code></p>
                    <p>👤 <strong>User</strong>: <code>${createdTask.user_id}</code> | Compute: <code>${createdTask.gpu_demand}%</code> | Memory: <code>${createdTask.memory_demand} MB</code></p>
                    <p>🛡️ <strong>Isolation Policy</strong>: <code>${createdTask.isolation_level}</code> | Carbon Preference: <code>${createdTask.carbon_preference}</code></p>
                    <p>💡 Task queued for immediate simulation scheduling step.</p>
                `;

                refreshDashboardData();
            }
        });
    }

    // Train LSTM Model Button Handler
    const btnTrain = document.getElementById("btn-train-model");
    if (btnTrain) {
        btnTrain.addEventListener("click", async () => {
            const epochs = parseInt(document.getElementById("input-epochs").value) || 30;
            showToast("Training PyTorch LSTM workload predictor model...", "info");
            btnTrain.disabled = true;

            const res = await apiFetch(`/train-model?epochs=${epochs}`, "POST");
            btnTrain.disabled = false;
            if (res) {
                showToast(`Model trained! MAE: ${res.metrics.mae.toFixed(4)}, RMSE: ${res.metrics.rmse.toFixed(4)}`, "success");
                refreshDashboardData();
            }
        });
    }

    // Interactive Inference Test Simulator
    const sliderHistLoad = document.getElementById("slider-hist-load");
    const lblHistLoad = document.getElementById("lbl-hist-load");
    if (sliderHistLoad && lblHistLoad) {
        sliderHistLoad.addEventListener("input", (e) => {
            lblHistLoad.textContent = e.target.value;
        });
    }

    const btnRunInference = document.getElementById("btn-run-inference-test");
    if (btnRunInference) {
        btnRunInference.addEventListener("click", async () => {
            const baseVal = parseFloat(sliderHistLoad.value);
            const history = [
                Math.max(5, baseVal - 15),
                Math.max(5, baseVal - 10),
                Math.max(5, baseVal - 5),
                baseVal
            ];

            const res = await apiFetch("/predict/inference-test", "POST", { history });
            const resultBox = document.getElementById("inference-test-result");
            if (res && resultBox) {
                resultBox.classList.remove("hidden");
                resultBox.innerHTML = `
                    <p>🔮 <strong>Input Historical Demand Sequence</strong>: <code>[${history.join("%, ")}%]</code></p>
                    <p>📊 <strong>PyTorch LSTM Next-Step Forecast</strong>: <strong style="color: var(--accent-cyan); font-size: 1.1rem;">${res.predicted_utilization}%</strong> (Confidence Delta: ±${res.confidence_delta}%)</p>
                `;
                showToast(`Forecast complete: ${res.predicted_utilization}% GPU demand predicted`, "success");
            }
        });
    }

    // Green Energy Grid Shift Handler
    const btnGreenGrid = document.getElementById("btn-green-grid-shift");
    if (btnGreenGrid) {
        btnGreenGrid.addEventListener("click", async () => {
            const res = await apiFetch("/carbon/grid-shift", "POST");
            if (res) {
                showToast(res.message, "success");
                refreshDashboardData();
            }
        });
    }

    // Noisy-Neighbor Violation Trigger Handler
    const sliderSpikeFactor = document.getElementById("slider-spike-factor");
    const lblSpikeFactor = document.getElementById("lbl-spike-factor");
    if (sliderSpikeFactor && lblSpikeFactor) {
        sliderSpikeFactor.addEventListener("input", (e) => {
            lblSpikeFactor.textContent = parseFloat(e.target.value).toFixed(2);
        });
    }

    const btnTriggerViolation = document.getElementById("btn-trigger-violation");
    if (btnTriggerViolation) {
        btnTriggerViolation.addEventListener("click", async () => {
            const gpuId = document.getElementById("select-trigger-gpu").value;
            const spikeFactor = parseFloat(sliderSpikeFactor.value);

            const res = await apiFetch("/isolation/trigger-violation", "POST", { gpu_id: gpuId, spike_factor: spikeFactor });
            if (res) {
                if (res.status === "violation_detected") {
                    showToast(`Software isolation quota breach detected on ${gpuId}! Cgroup ceiling enforced.`, "danger");
                } else {
                    showToast(`Workload spike within quota limits on ${gpuId}.`, "info");
                }
                refreshDashboardData();
            }
        });
    }

    // Benchmark Execution Handler
    let benchmarkResultsData = null;
    const btnBenchmark = document.getElementById("btn-run-benchmark");
    if (btnBenchmark) {
        btnBenchmark.addEventListener("click", async () => {
            showToast("Running comparative experiments across all 3 schedulers...", "info");
            btnBenchmark.disabled = true;

            const algs = ["baseline", "predictive", "proposed"];
            const results = [];

            for (const alg of algs) {
                const res = await apiFetch(`/metrics?scheduler_type=${alg}`);
                if (res) results.push(res);
            }

            btnBenchmark.disabled = false;
            if (results.length > 0) {
                benchmarkResultsData = results;
                renderBenchmarkResults(results);
                document.getElementById("btn-export-csv").disabled = false;
                showToast("Comparative experiment benchmark completed!", "success");
            }
        });
    }

    // Render Benchmark Results
    function renderBenchmarkResults(results) {
        const labels = results.map(r => r.scheduler_type.toUpperCase());
        const waitData = results.map(r => r.avg_waiting_time);
        const carbonData = results.map(r => r.total_carbon_gco2);

        if (charts.compWait) {
            charts.compWait.data.labels = labels;
            charts.compWait.data.datasets[0].data = waitData;
            charts.compWait.update();
        }

        if (charts.compCarbon) {
            charts.compCarbon.data.labels = labels;
            charts.compCarbon.data.datasets[0].data = carbonData;
            charts.compCarbon.update();
        }

        const tbody = document.getElementById("comp-table-body");
        tbody.innerHTML = results.map(r => `
            <tr>
                <td><strong>${r.scheduler_type.toUpperCase()}</strong></td>
                <td>${r.completed_tasks} / ${r.total_tasks}</td>
                <td>${r.avg_waiting_time.toFixed(2)} min</td>
                <td>${r.avg_turnaround_time.toFixed(2)} min</td>
                <td>${r.total_energy_kwh.toFixed(4)} kWh</td>
                <td>${r.total_carbon_gco2.toFixed(2)} gCO₂</td>
                <td>${r.deadline_violations}</td>
                <td>${r.isolation_violations}</td>
            </tr>
        `).join("");
    }

    // Export Benchmark CSV Handler
    const btnExportCsv = document.getElementById("btn-export-csv");
    if (btnExportCsv) {
        btnExportCsv.addEventListener("click", () => {
            if (!benchmarkResultsData) return;
            const headers = ["scheduler_type", "total_tasks", "completed_tasks", "avg_waiting_time", "avg_turnaround_time", "total_energy_kwh", "total_carbon_gco2", "deadline_violations", "isolation_violations"];
            let csv = headers.join(",") + "\n";
            benchmarkResultsData.forEach(r => {
                csv += headers.map(h => r[h]).join(",") + "\n";
            });

            const blob = new Blob([csv], { type: "text/csv" });
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = "scheduler_comparison_results.csv";
            a.click();
            window.URL.revokeObjectURL(url);
        });
    }

    // Simulation Controls Handler
    const btnSimStep = document.getElementById("sim-btn-step");
    const btnSimRun = document.getElementById("sim-btn-run");
    const btnSimReset = document.getElementById("sim-btn-reset");
    const selectScheduler = document.getElementById("sim-select-scheduler");

    if (btnSimStep) {
        btnSimStep.addEventListener("click", async () => {
            const scheduler = selectScheduler.value;
            const res = await apiFetch("/simulate", "POST", { action: "step", scheduler_type: scheduler });
            if (res && res.state) {
                showToast(`Executed step to minute ${res.state.current_time}`, "info");
                refreshDashboardData();
            }
        });
    }

    if (btnSimRun) {
        btnSimRun.addEventListener("click", async () => {
            const scheduler = selectScheduler.value;
            showToast("Executing all queued simulation tasks...", "info");
            const res = await apiFetch("/simulate", "POST", { action: "start", scheduler_type: scheduler });
            if (res) {
                showToast("Simulation completed!", "success");
                refreshDashboardData();
            }
        });
    }

    if (btnSimReset) {
        btnSimReset.addEventListener("click", async () => {
            const res = await apiFetch("/simulate", "POST", { action: "reset" });
            if (res) {
                showToast("Simulation engine reset!", "info");
                refreshDashboardData();
            }
        });
    }

    // Sliders Label Synchronizer for Multi-Objective Weights
    ["res", "pred", "carb", "dead", "iso"].forEach(key => {
        const slider = document.getElementById(`slider-w-${key}`);
        const label = document.getElementById(`lbl-w-${key}`);
        if (slider && label) {
            slider.addEventListener("input", (e) => {
                label.textContent = parseFloat(e.target.value).toFixed(2);
            });
        }
    });

    // Save Weights Handler
    const btnSaveWeights = document.getElementById("btn-save-weights");
    if (btnSaveWeights) {
        btnSaveWeights.addEventListener("click", async () => {
            const weights = {
                resource: parseFloat(document.getElementById("slider-w-res").value),
                prediction: parseFloat(document.getElementById("slider-w-pred").value),
                carbon: parseFloat(document.getElementById("slider-w-carb").value),
                deadline: parseFloat(document.getElementById("slider-w-dead").value),
                isolation: parseFloat(document.getElementById("slider-w-iso").value)
            };

            const res = await apiFetch("/settings/weights", "POST", weights);
            if (res) {
                showToast("Algorithm multi-objective weights saved and applied!", "success");
            }
        });
    }

    // Initialize App & Auto Refresh Loop
    initCharts();
    refreshDashboardData();
    setInterval(refreshDashboardData, 4000);
});
