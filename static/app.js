/* ═══════════════════════════════════════════════════════════════════════════
   Digital Twin Dashboard - Frontend Logic
   ═══════════════════════════════════════════════════════════════════════════ */

const fileInput    = document.getElementById('file-input');
const uploadArea   = document.getElementById('upload-area');
const fileNameEl   = document.getElementById('file-name');
const btnTest      = document.getElementById('btn-test');
const uploadSec    = document.getElementById('upload-section');
const loadingSec   = document.getElementById('loading-section');
const dashboard    = document.getElementById('dashboard');

let selectedFile = null;

// ── File selection ──────────────────────────────────────────────────────────
uploadArea.addEventListener('click', () => fileInput.click());

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('active');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('active');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('active');
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});

fileInput.addEventListener('change', () => {
    if (fileInput.files.length) handleFile(fileInput.files[0]);
});

function handleFile(file) {
    if (!file.name.toLowerCase().endsWith('.mp4')) {
        alert('Please select an .mp4 video file.');
        return;
    }
    selectedFile = file;
    fileNameEl.textContent = file.name;
    uploadArea.classList.add('active');
    btnTest.disabled = false;
}

// ── Run simulation ──────────────────────────────────────────────────────────
btnTest.addEventListener('click', async () => {
    if (!selectedFile) return;

    // Show loading, hide upload + dashboard
    uploadSec.style.display = 'none';
    loadingSec.classList.add('visible');
    dashboard.classList.remove('visible');

    try {
        const formData = new FormData();
        formData.append('video', selectedFile);

        const resp = await fetch('/api/run', {
            method: 'POST',
            body: formData,
        });

        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.error || 'Server error');
        }

        const data = await resp.json();
        populateDashboard(data);

    } catch (err) {
        alert('Error: ' + err.message);
        // Show upload again
        uploadSec.style.display = 'block';
    } finally {
        loadingSec.classList.remove('visible');
    }
});

// ── Populate dashboard ──────────────────────────────────────────────────────
function populateDashboard(data) {
    const a = data.instance_a;
    const b = data.instance_b;

    // Instance A stats
    document.getElementById('a-wait-all').textContent       = a.avg_wait_all.toFixed(1);
    document.getElementById('a-wait-ambulance').textContent  = a.avg_wait_ambulance.toFixed(1);

    // Instance B stats
    document.getElementById('b-wait-all').textContent       = b.avg_wait_all.toFixed(1);
    document.getElementById('b-wait-ambulance').textContent  = b.avg_wait_ambulance.toFixed(1);

    // Calculate reductions: (A - B) / A * 100
    // Positive = B is better (lower wait)
    const overallReduction   = a.avg_wait_all > 0
        ? ((a.avg_wait_all - b.avg_wait_all) / a.avg_wait_all * 100)
        : 0;
    const ambulanceReduction = a.avg_wait_ambulance > 0
        ? ((a.avg_wait_ambulance - b.avg_wait_ambulance) / a.avg_wait_ambulance * 100)
        : 0;

    setReduction('overall-reduction',   overallReduction);
    setReduction('ambulance-reduction', ambulanceReduction);

    // Show dashboard
    uploadSec.style.display = 'block';
    dashboard.classList.add('visible');
}

function setReduction(id, pct) {
    const el = document.getElementById(id);
    const sign   = pct >= 0 ? '' : '+';
    const absVal = Math.abs(pct).toFixed(1);
    el.textContent = sign + absVal + '%';

    // Color: positive reduction = green, negative = red, zero = neutral
    el.classList.remove('positive', 'negative', 'neutral');
    if (Math.abs(pct) < 0.5) {
        el.classList.add('neutral');
    } else if (pct > 0) {
        el.classList.add('positive');
    } else {
        el.classList.add('negative');
    }
}
