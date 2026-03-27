/* ═══════════════════════════════════════════════════════════════════════════
   Smart City Traffic Control Center — Frontend Logic
   ═══════════════════════════════════════════════════════════════════════════ */

// ── DOM References ──────────────────────────────────────────────────────────
const tabFleet  = document.getElementById('tab-fleet');
const tabLab    = document.getElementById('tab-lab');
const viewFleet = document.getElementById('view-fleet');
const viewLab   = document.getElementById('view-lab');
const detailPanel = document.getElementById('detail-panel');
const panelClose  = document.getElementById('panel-close');

// ═══════════════════════════════════════════════════════════════════════════
//  1) TAB SWITCHING (SPA)
// ═══════════════════════════════════════════════════════════════════════════
function switchView(tabEl, viewId) {
    document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.view').forEach(v => v.classList.add('hidden'));
    tabEl.classList.add('active');
    document.getElementById(viewId).classList.remove('hidden');
}

tabFleet.addEventListener('click', () => switchView(tabFleet, 'view-fleet'));
tabLab.addEventListener('click',   () => switchView(tabLab,   'view-lab'));

// ═══════════════════════════════════════════════════════════════════════════
//  2) LIVE FLEET — LEAFLET MAP
// ═══════════════════════════════════════════════════════════════════════════
const OFFLINE_IDS = [4, 11, 19];

// 25 intersections spread across a real city block grid (centered on Delhi, India)
const BASE_LAT = 28.6139, BASE_LNG = 77.2090;
const nodeCoords = [
    [28.6280, 77.1975], [28.6265, 77.2080], [28.6250, 77.2180], [28.6235, 77.2280], [28.6220, 77.2380],
    [28.6170, 77.1975], [28.6155, 77.2075], [28.6140, 77.2175], [28.6125, 77.2275], [28.6110, 77.2375],
    [28.6060, 77.1975], [28.6045, 77.2075], [28.6030, 77.2175], [28.6015, 77.2275], [28.6000, 77.2375],
    [28.5950, 77.1975], [28.5935, 77.2075], [28.5920, 77.2175], [28.5905, 77.2275], [28.5890, 77.2375],
    [28.5840, 77.1975], [28.5825, 77.2075], [28.5810, 77.2175], [28.5795, 77.2275], [28.5780, 77.2375],
];

const nodes = nodeCoords.map((coord, i) => {
    const id = i + 1;
    const status = OFFLINE_IDS.includes(id) ? 'offline' : 'online';
    return {
        id, label: `J${id}`, status,
        lat: coord[0], lng: coord[1],
        error: status === 'offline' ? '502' : null,
        uptime: status === 'offline'
            ? `${Math.floor(Math.random()*30)}d ${Math.floor(Math.random()*24)}h ${Math.floor(Math.random()*60)}m`
            : `${30 + Math.floor(Math.random()*60)}d`,
        marker: null,
    };
});

// Init Leaflet map
const map = L.map('fleet-map', { zoomControl: true, scrollWheelZoom: true })
    .setView([BASE_LAT, BASE_LNG], 13);

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap contributors',
    maxZoom: 19,
}).addTo(map);

function makeIcon(status) {
    return L.divIcon({
        className: '',
        html: `<div class="map-node ${status}">J</div>`,
        iconSize: [32, 32],
        iconAnchor: [16, 16],
        popupAnchor: [0, -18],
    });
}

let selectedNodeIdx = null;

function updateStats() {
    let onlineC = 0, offlineC = 0, pendingC = 0;
    nodes.forEach(n => {
        if (n.status === 'online')  onlineC++;
        else if (n.status === 'offline') offlineC++;
        else pendingC++;
    });
    document.getElementById('stat-online').textContent  = onlineC;
    document.getElementById('stat-offline').textContent = offlineC;
    document.getElementById('stat-pending').textContent = pendingC;
}

// Add markers to map
nodes.forEach((n, idx) => {
    const marker = L.marker([n.lat, n.lng], { icon: makeIcon(n.status) }).addTo(map);
    n.marker = marker;

    marker.bindPopup(() => {
        const statusLabel = { online: 'Online', offline: 'Camera Signal Lost', pending: 'Pending Repair' }[n.status];
        return `<div class="map-popup">
            <h4>${n.label} — Intersection ${n.id}</h4>
            <small>Uptime: ${n.uptime}</small>
            <div class="status-line"><div class="pop-dot ${n.status}"></div>${statusLabel}</div>
        </div>`;
    }, { closeButton: false });

    marker.on('click', () => openPanel(idx));
});

function openPanel(idx) {
    const n = nodes[idx];
    selectedNodeIdx = idx;

    document.getElementById('panel-title').textContent   = `Node ${n.label} — Details`;
    document.getElementById('panel-node-id').textContent  = n.label;
    document.getElementById('panel-uptime').textContent   = n.uptime;

    const alertEl     = document.getElementById('panel-alert');
    const errorEl     = document.getElementById('panel-error');
    const statusEl    = document.getElementById('panel-status');
    const dispatchBtn = document.getElementById('btn-dispatch');

    if (n.status === 'offline') {
        alertEl.className = 'panel-alert';
        alertEl.querySelector('span').textContent = 'Camera Signal Lost';
        errorEl.textContent = n.error || '502';
        statusEl.textContent = 'Offline';
        statusEl.style.color = 'var(--red)';
        dispatchBtn.className = 'btn-dispatch';
        dispatchBtn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg> Dispatch Technician`;
        dispatchBtn.style.display = 'flex';
    } else if (n.status === 'pending') {
        alertEl.className = 'panel-alert pending-alert';
        alertEl.querySelector('span').textContent = 'Maintenance Ticket Dispatched';
        errorEl.textContent = '—';
        statusEl.textContent = 'Pending Repair';
        statusEl.style.color = 'var(--yellow)';
        dispatchBtn.style.display = 'none';
    } else {
        alertEl.className = 'panel-alert';
        alertEl.style.background = 'rgba(22,163,74,.08)';
        alertEl.style.borderColor = 'rgba(22,163,74,.25)';
        alertEl.style.color = 'var(--green)';
        alertEl.querySelector('span').textContent = 'Camera Online — No Issues';
        errorEl.textContent = '—';
        statusEl.textContent = 'Online';
        statusEl.style.color = 'var(--green)';
        dispatchBtn.style.display = 'none';
    }

    detailPanel.classList.add('open');
    // Pan map to selected node
    map.panTo([n.lat, n.lng], { animate: true, duration: 0.5 });
    n.marker.openPopup();
}

panelClose.addEventListener('click', () => {
    detailPanel.classList.remove('open');
    selectedNodeIdx = null;
    map.closePopup();
});

// ── Dispatch Button ─────────────────────────────────────────────────────
document.getElementById('btn-dispatch').addEventListener('click', () => {
    if (selectedNodeIdx === null) return;
    const n = nodes[selectedNodeIdx];
    if (n.status !== 'offline') return;

    const ticketNum = 4000 + Math.floor(Math.random() * 999);
    n.status = 'pending';
    n.error = null;

    // Update marker icon on map
    n.marker.setIcon(makeIcon('pending'));
    updateStats();
    openPanel(selectedNodeIdx);

    alert(`Maintenance Ticket #${ticketNum} Dispatched to Vendor`);
});

updateStats();

// ═══════════════════════════════════════════════════════════════════════════
//  3) ALGORITHM LAB — UPLOAD & TEST
// ═══════════════════════════════════════════════════════════════════════════
const labUploadArea = document.getElementById('lab-upload-area');
const labFileInput  = document.getElementById('lab-file-input');
const labFileName   = document.getElementById('lab-file-name');
const btnLabTest    = document.getElementById('btn-lab-test');

let labFile = null;

labUploadArea.addEventListener('click', () => labFileInput.click());

labUploadArea.addEventListener('dragover', (e) => { e.preventDefault(); labUploadArea.classList.add('active'); });
labUploadArea.addEventListener('dragleave', () => labUploadArea.classList.remove('active'));
labUploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    labUploadArea.classList.remove('active');
    if (e.dataTransfer.files.length) handleLabFile(e.dataTransfer.files[0]);
});
labFileInput.addEventListener('change', () => {
    if (labFileInput.files.length) handleLabFile(labFileInput.files[0]);
});

function handleLabFile(file) {
    if (!file.name.toLowerCase().endsWith('.mp4')) {
        alert('Please select an .mp4 video file.');
        return;
    }
    labFile = file;
    labFileName.textContent = file.name;
    labUploadArea.classList.add('active');
    btnLabTest.disabled = false;
}

btnLabTest.addEventListener('click', async () => {
    if (!labFile) return;

    const testCard    = document.getElementById('test-card');
    const testLoading = document.getElementById('test-loading');
    const testResults = document.getElementById('test-results');

    testCard.classList.add('hidden');
    testLoading.classList.remove('hidden');
    testResults.classList.add('hidden');

    try {
        const formData = new FormData();
        formData.append('video', labFile);

        const resp = await fetch('/api/run', { method: 'POST', body: formData });
        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.error || 'Server error');
        }

        const data = await resp.json();
        populateResults(data);

    } catch (err) {
        alert('Error: ' + err.message);
        testCard.classList.remove('hidden');
    } finally {
        testLoading.classList.add('hidden');
    }
});

function populateResults(data) {
    const a = data.instance_a;
    const b = data.instance_b;

    document.getElementById('r-a-all').textContent = a.avg_wait_all.toFixed(1);
    document.getElementById('r-a-amb').textContent = a.avg_wait_ambulance.toFixed(1);
    document.getElementById('r-b-all').textContent = b.avg_wait_all.toFixed(1);
    document.getElementById('r-b-amb').textContent = b.avg_wait_ambulance.toFixed(1);

    // Positive = AI (A) is better than Baseline (B)
    const overallPct   = b.avg_wait_all > 0
        ? ((b.avg_wait_all - a.avg_wait_all) / b.avg_wait_all * 100)
        : 0;
    const ambulancePct = b.avg_wait_ambulance > 0
        ? ((b.avg_wait_ambulance - a.avg_wait_ambulance) / b.avg_wait_ambulance * 100)
        : 0;

    setReduction('overall-reduction',   overallPct);
    setReduction('ambulance-reduction', ambulancePct);

    document.getElementById('test-card').classList.remove('hidden');
    document.getElementById('test-results').classList.remove('hidden');

    // ── Append result to chart & version table ──────────────────────────
    appendRunToChart(a.avg_wait_all, a.avg_wait_ambulance);
    appendRunToTable(a.avg_wait_all, a.avg_wait_ambulance, overallPct);
}

// Track how many live test runs have been added
let runCount = 0;

function appendRunToChart(waitAll, waitAmb) {
    if (!perfChart) return;
    runCount++;
    // Generate a version label like V2.1, V2.2 … continuing from V2.0
    const label = `V2.${runCount}`;
    perfChart.data.labels.push(label);
    perfChart.data.datasets[0].data.push(parseFloat(waitAll.toFixed(1)));
    perfChart.data.datasets[1].data.push(parseFloat(waitAmb.toFixed(1)));
    perfChart.update('active');   // animated update
}

function appendRunToTable(waitAll, waitAmb, overallPct) {
    const tbody = document.querySelector('#version-table tbody');
    if (!tbody) return;
    runCount; // already incremented in appendRunToChart
    const label = `V2.${runCount}`;
    const today = new Date().toISOString().split('T')[0];
    const isImprovement = overallPct > 0;
    const badge = isImprovement
        ? '<span class="badge badge-ready">Ready</span>'
        : '<span class="badge badge-archived">Tested</span>';
    const deployBtn = isImprovement
        ? `<button class="btn-deploy run-deploy-btn">Deploy to Edge Fleet</button>`
        : '—';

    const tr = document.createElement('tr');
    tr.className = 'best-row';
    tr.innerHTML = `
        <td><span class="version-tag best">${label}</span></td>
        <td>${today}</td>
        <td>${waitAll.toFixed(1)} s</td>
        <td>${waitAmb.toFixed(1)} s</td>
        <td>${badge}</td>
        <td>${deployBtn}</td>
    `;
    tbody.appendChild(tr);

    // Wire up deploy button if present
    const btn = tr.querySelector('.run-deploy-btn');
    if (btn) btn.addEventListener('click', triggerDeploy);

    // Scroll table into view so user notices
    tr.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

function setReduction(id, pct) {
    const el = document.getElementById(id);
    const absVal = Math.abs(pct).toFixed(1);
    el.textContent = (pct >= 0 ? '' : '-') + absVal + '%';
    el.classList.remove('positive', 'negative', 'neutral');
    if (Math.abs(pct) < 0.5) {
        el.classList.add('neutral');
    } else if (pct > 0) {
        el.classList.add('positive');
    } else {
        el.classList.add('negative');
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  4) HISTORICAL ANALYTICS — LINE CHART
// ═══════════════════════════════════════════════════════════════════════════
const chartCtx = document.getElementById('perf-chart');
let perfChart = null;

if (chartCtx) {
    perfChart = new Chart(chartCtx, {
        type: 'line',
        data: {
            labels: ['V1.0', 'V1.1', 'V2.0'],
            datasets: [
                {
                    label: 'Overall Traffic Wait (s)',
                    data: [50, 38.7, 28.3],
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59,130,246,.12)',
                    borderWidth: 3,
                    tension: 0.3,
                    fill: true,
                    pointRadius: 6,
                    pointBackgroundColor: '#3b82f6',
                    pointBorderColor: '#0a0e17',
                    pointBorderWidth: 2,
                },
                {
                    label: 'Ambulance Wait (s)',
                    data: [80, 45.1, 12.6],
                    borderColor: '#06b6d4',
                    backgroundColor: 'rgba(6,182,212,.08)',
                    borderWidth: 3,
                    tension: 0.3,
                    fill: true,
                    pointRadius: 6,
                    pointBackgroundColor: '#06b6d4',
                    pointBorderColor: '#0a0e17',
                    pointBorderWidth: 2,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: {
                    labels: { color: '#475569', font: { family: "'Inter', sans-serif", size: 12 }, padding: 20 },
                },
                tooltip: {
                    backgroundColor: '#ffffff',
                    borderColor: '#e2e8f0',
                    borderWidth: 1,
                    titleColor: '#0f172a',
                    bodyColor: '#475569',
                    padding: 12,
                    cornerRadius: 8,
                },
            },
            scales: {
                x: {
                    title: { display: true, text: 'Algorithm Version', color: '#94a3b8', font: { size: 12 } },
                    ticks: { color: '#475569' },
                    grid:  { color: '#e2e8f0' },
                },
                y: {
                    title: { display: true, text: 'Wait Time (Seconds)', color: '#94a3b8', font: { size: 12 } },
                    ticks: { color: '#475569' },
                    grid:  { color: '#e2e8f0' },
                    beginAtZero: true,
                },
            },
        },
    });
}

// ═══════════════════════════════════════════════════════════════════════════
//  5) DEPLOY TO EDGE FLEET
// ═══════════════════════════════════════════════════════════════════════════
const btnDeploy      = document.getElementById('btn-deploy');
const deployOverlay  = document.getElementById('deploy-overlay');
const deployProgress = document.getElementById('deploy-progress');
const deployPct      = document.getElementById('deploy-pct');
const deployTitle    = document.getElementById('deploy-title');
const deploySub      = document.getElementById('deploy-sub');
const deployIcon     = document.getElementById('deploy-icon');

function triggerDeploy(versionLabel) {
    const label = versionLabel || 'V2.0';
    deployOverlay.classList.remove('hidden');
    deployProgress.style.width = '0%';
    deployPct.textContent = '0%';
    deployPct.style.color = '';
    deployTitle.textContent = 'Pushing OTA Update to Edge Containers…';
    deploySub.textContent = `Deploying ${label} across 25 intersection nodes`;
    deployIcon.classList.remove('success');

    let progress = 0;
    const interval = setInterval(() => {
        progress += Math.random() * 8 + 2;
        if (progress >= 100) progress = 100;
        deployProgress.style.width = progress + '%';
        deployPct.textContent = Math.round(progress) + '%';

        if (progress >= 100) {
            clearInterval(interval);
            setTimeout(() => {
                deployTitle.textContent = '✓ OTA Update Complete!';
                deploySub.textContent = `${label} is now live across all 25 edge nodes.`;
                deployIcon.classList.add('success');
                deployPct.textContent = '100%';
                deployPct.style.color = '#16a34a';

                // Mark all "Ready" badges as deployed
                document.querySelectorAll('.badge-ready').forEach(b => {
                    b.className = 'badge badge-deployed';
                    b.textContent = 'Deployed';
                });

                setTimeout(() => {
                    deployOverlay.classList.add('hidden');
                    // Disable all deploy buttons
                    document.querySelectorAll('.btn-deploy, .run-deploy-btn').forEach(b => {
                        b.textContent = 'Deployed ✓';
                        b.disabled = true;
                        b.style.opacity = '.5';
                        b.style.cursor = 'default';
                    });
                }, 2000);
            }, 400);
        }
    }, 100);
}

btnDeploy.addEventListener('click', () => triggerDeploy('V2.0'));

