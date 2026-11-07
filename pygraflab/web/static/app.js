/* ==========================================================================
   pyGrafLab — web console logic
   ========================================================================== */
'use strict';

const API = '';  // same origin

let currentResults = null;
let functionalMap = {};
let pollTimer = null;
let modelLoaded = false;
let loadedModelPath = '';
let isComputing = false;
let plot = null;
let outputMenuEl = null;

/* ==================================================================
   Theme & view chrome
   ================================================================== */
function applyTheme(name) {
  const theme = name === 'dark' ? 'dark' : 'light';
  document.documentElement.dataset.theme = theme;
  try { localStorage.setItem('pygraflab-theme', theme); } catch (e) {}
  if (window.FieldPlotUtil && window.FieldPlotUtil.setTheme) {
    window.FieldPlotUtil.setTheme(theme);
  }
  const btn = document.getElementById('btn-theme');
  if (btn) {
    btn.innerHTML = theme === 'dark'
      ? '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><circle cx="12" cy="12" r="4.2"/><path d="M12 2v2.5M12 19.5V22M2 12h2.5M19.5 12H22M4.9 4.9l1.8 1.8M17.3 17.3l1.8 1.8M19.1 4.9l-1.8 1.8M6.7 17.3l-1.8 1.8"/></svg>'
      : '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><path d="M20 14.5A8.5 8.5 0 0 1 9.5 4a8.5 8.5 0 1 0 10.5 10.5z"/></svg>';
  }
  // Repaint the canvas: fill and grid colours live in JS, not in CSS.
  if (currentResults) updateVisualization();
  else if (plot) plot.clear();
}

function toggleTheme() {
  applyTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark');
}

function setView(view) {
  const name = view === 'results' ? 'results' : 'plot';
  const body = document.getElementById('editor-body');
  if (!body) return;
  body.dataset.view = name;
  document.getElementById('tab-plot').classList.toggle('is-active', name === 'plot');
  document.getElementById('tab-results').classList.toggle('is-active', name === 'results');
  document.getElementById('act-plot').classList.toggle('is-active', name === 'plot');
  document.getElementById('act-results').classList.toggle('is-active', name === 'results');
  if (name === 'plot' && plot) {
    // The canvas was display:none while the table was up; resize then redraw.
    requestAnimationFrame(() => plot.resize());
  }
}

function toggleSidebar() {
  const sidebar = document.getElementById('sidebar');
  if (!sidebar) return;
  sidebar.classList.toggle('is-collapsed');
  document.getElementById('act-controls')
    .classList.toggle('is-active', !sidebar.classList.contains('is-collapsed'));
  if (plot) requestAnimationFrame(() => plot.resize());
}


const FUNCTIONAL_OPTIONS = [
  { code: 0,  name: '— None —' },
  { code: 2,  name: 'Deflection η (arcsec)' },
  { code: 3,  name: 'Deflection ξ (arcsec)' },
  { code: 4,  name: 'Deflection Θ (arcsec)' },
  { code: 5,  name: 'Disturbing potential T (m²/s²)' },
  { code: 6,  name: 'Dist. tensor Trr Tpp Tll (E)' },
  { code: 7,  name: 'Dist. tensor Trf Trl Tfl (E)' },
  { code: 8,  name: 'Dist. tensor LNOF Txx Tyy Tzz (E)' },
  { code: 9,  name: 'Dist. tensor LNOF Txy Txz Tyz (E)' },
  { code: 10, name: 'Geoid undulation N (m)' },
  { code: 11, name: 'Gravitational potential V (m²/s²)' },
  { code: 12, name: 'Grav. tensor Vrr Vpp Vll (E)' },
  { code: 13, name: 'Grav. tensor Vrf Vrl Vfl (E)' },
  { code: 14, name: 'Grav. tensor LNOF Vxx Vyy Vzz (E)' },
  { code: 15, name: 'Grav. tensor LNOF Vxy Vxz Vyz (E)' },
  { code: 16, name: 'Gravity vector gX gY gZ (mGal)' },
  { code: 17, name: 'Gravity (mGal)' },
  { code: 18, name: 'Gravity potential W (m²/s²)' },
  { code: 19, name: 'Gravity anomaly (mGal)' },
  { code: 20, name: 'Gravity disturbance (mGal)' },
  { code: 21, name: 'Gravity disturbance sa (mGal)' },
  { code: 22, name: 'Height anomaly ell (m)' },
  { code: 23, name: 'Height anomaly (m)' },
  { code: 24, name: 'T_rr (Eotvos)' },
  { code: 25, name: 'W_rr (Eotvos)' },
];

const UNIT_BY_CODE = {
  2: 'arcsec', 3: 'arcsec', 4: 'arcsec',
  5: 'm²/s²', 6: 'Eotvos', 7: 'Eotvos', 8: 'Eotvos', 9: 'Eotvos',
  10: 'm', 11: 'm²/s²', 12: 'Eotvos', 13: 'Eotvos', 14: 'Eotvos',
  15: 'Eotvos', 16: 'mGal', 17: 'mGal', 18: 'm²/s²',
  19: 'mGal', 20: 'mGal', 21: 'mGal', 22: 'm', 23: 'm',
  24: 'Eotvos', 25: 'Eotvos',
};

/* Validity ranges of the three fnALF recursions, as stated in GrafLab. */
const METHOD_HINTS = {
  1: 'Recommended up to degree 1800 at all latitudes, and up to 2190 '
    + 'between 0–56° and 80–90°.',
  2: 'Recommended from degree 1801 to 2700 at all latitudes; '
    + 'higher degrees overflow.',
  3: 'Valid at any latitude and any degree.',
};
const COMMISSION_METHOD_HINT =
  'Commission error is always evaluated with the standard forward column method.';

/* ==================================================================
   Boot
   ================================================================== */
document.addEventListener('DOMContentLoaded', () => {
  populateFunctionalDropdowns();

  const canvas = document.getElementById('plot-canvas');
  const tooltip = document.getElementById('plot-tooltip');
  if (canvas && window.FieldPlot) plot = new window.FieldPlot(canvas, tooltip);
  applyTheme(document.documentElement.dataset.theme || 'light');

  // The chosen recursion is remembered across sessions; first visit starts on
  // the standard forward column method, as GrafLab does.
  const methodSel = document.getElementById('inp-method');
  let methodBeforeCommission = null;
  try {
    const saved = parseInt(localStorage.getItem('pygraflab-method'), 10);
    if (saved === 1 || saved === 2 || saved === 3) methodSel.value = String(saved);
  } catch (e) { /* storage unavailable — keep the default */ }
  methodSel.addEventListener('change', () => {
    try { localStorage.setItem('pygraflab-method', methodSel.value); } catch (e) {}
    updateMethodHint();
  });

  document.getElementById('chk-commission').addEventListener('change', (e) => {
    document.getElementById('commission-settings').style.display =
      e.target.checked ? 'flex' : 'none';
    // Commission errors are propagated with the standard forward column
    // method only, so the recursion selector is pinned while it is enabled.
    if (e.target.checked) {
      methodBeforeCommission = methodSel.value;
      methodSel.value = '1';
      methodSel.disabled = true;
    } else {
      methodSel.disabled = false;
      if (methodBeforeCommission) methodSel.value = methodBeforeCommission;
    }
    updateMethodHint();
    updateRunSummary();
  });

  document.getElementById('chk-use-nmax').addEventListener('change', (e) => {
    document.getElementById('inp-nmax').disabled = e.target.checked;
    if (e.target.checked) {
      const info = document.getElementById('inp-nmax');
      info.value = '';
    }
  });
  document.getElementById('inp-nmax').disabled = true;

  ['func-1', 'func-2', 'func-3', 'func-4', 'inp-method', 'mode-grid', 'mode-pointwise']
    .forEach((id) => {
      const el = document.getElementById(id);
      if (el) el.addEventListener('change', updateRunSummary);
    });

  ['lon-min', 'lon-step', 'lon-max', 'lat-min', 'lat-step', 'lat-max']
    .forEach((id) => document.getElementById(id).addEventListener('input', updateGridHint));

  ['inp-gm', 'inp-r', 'inp-nmin', 'inp-nmax', 'inp-ellipsoid', 'inp-h']
    .forEach((id) => {
      const el = document.getElementById(id);
      if (el) el.addEventListener('change', updateRunSummary);
    });

  updateGridHint();
  updateMethodHint();
  updateRunSummary();
  fetchModelInfo();
});

/* ==================================================================
   UI helpers
   ================================================================== */
function populateFunctionalDropdowns() {
  for (let i = 1; i <= 4; i++) {
    const sel = document.getElementById(`func-${i}`);
    FUNCTIONAL_OPTIONS.forEach((opt) => {
      const o = document.createElement('option');
      o.value = opt.code;
      o.textContent = opt.name;
      sel.appendChild(o);
    });
  }
  document.getElementById('func-1').value = 11;
  document.getElementById('func-2').value = 19;
  document.getElementById('func-3').value = 10;
  document.getElementById('func-4').value = 0;
}

function togglePanel(id) {
  document.getElementById(id).classList.toggle('collapsed');
}

function switchMode(mode) {
  document.getElementById('grid-inputs').style.display = mode === 'grid' ? 'block' : 'none';
  document.getElementById('pointwise-inputs').style.display = mode === 'pointwise' ? 'block' : 'none';
  updateRunSummary();
}

function showToast(msg, type = 'info', ttl = 4200) {
  const container = document.getElementById('toast-container');
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.textContent = msg;
  container.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateX(12px)';
    toast.style.transition = '0.25s ease';
    setTimeout(() => toast.remove(), 250);
  }, ttl);
}

function showChecklistToast(errors) {
  const container = document.getElementById('toast-container');
  const toast = document.createElement('div');
  toast.className = 'toast error';
  const title = document.createElement('strong');
  title.textContent = 'Please fix the following:';
  toast.appendChild(title);
  const ul = document.createElement('ul');
  errors.forEach((msg) => {
    const li = document.createElement('li');
    li.textContent = msg;
    ul.appendChild(li);
  });
  toast.appendChild(ul);
  container.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transition = '0.25s ease';
    setTimeout(() => toast.remove(), 250);
  }, 6500);
}

function setStatus(text, type = '') {
  const chip = document.getElementById('status-chip');
  chip.className = 'status-chip' + (type ? ` ${type}` : '');
  document.getElementById('status-text').textContent = text;
  document.getElementById('footer-msg').textContent = `pyGrafLab — ${text}`;
}

function showProgress(show) {
  document.getElementById('progress-overlay').classList.toggle('active', show);
  const tp = document.getElementById('title-progress');
  if (tp) tp.style.display = show ? 'flex' : 'none';
}

function toggleResultsTable() {
  const container = document.getElementById('results-table-container');
  container.style.display = container.style.display === 'none' ? 'block' : 'none';
}

function markFieldError(id) {
  const el = document.getElementById(id);
  if (!el) return;
  el.classList.add('field-error');
  setTimeout(() => el.classList.remove('field-error'), 3000);
}

function clearFieldErrors() {
  document.querySelectorAll('.field-error').forEach((el) => el.classList.remove('field-error'));
}

function updateGridHint() {
  const el = document.getElementById('grid-size-hint');
  if (!el) return;
  const lonMin = parseFloat(document.getElementById('lon-min').value);
  const lonMax = parseFloat(document.getElementById('lon-max').value);
  const lonStep = parseFloat(document.getElementById('lon-step').value);
  const latMin = parseFloat(document.getElementById('lat-min').value);
  const latMax = parseFloat(document.getElementById('lat-max').value);
  const latStep = parseFloat(document.getElementById('lat-step').value);
  if ([lonMin, lonMax, lonStep, latMin, latMax, latStep].some((v) => Number.isNaN(v)) || lonStep <= 0 || latStep <= 0) {
    el.textContent = '—';
    return;
  }
  const nLon = Math.floor((lonMax - lonMin) / lonStep + 1e-9) + 1;
  const nLat = Math.floor((latMax - latMin) / latStep + 1e-9) + 1;
  el.textContent = `${nLon} × ${nLat} = ${(nLon * nLat).toLocaleString()} nodes`;
}

function updateRunSummary() {
  const el = document.getElementById('run-summary');
  if (!el) return;
  const mode = document.querySelector('input[name="mode"]:checked').value;
  const codes = [];
  for (let i = 1; i <= 4; i++) {
    const v = parseInt(document.getElementById(`func-${i}`).value, 10);
    if (v > 0) codes.push(v);
  }
  const parts = [
    modelLoaded ? (loadedModelPath || 'model loaded') : 'no model',
    mode,
    codes.length ? `${codes.length} functional${codes.length > 1 ? 's' : ''}` : 'no functional',
  ];
  el.textContent = parts.join(' · ');
}

/* Explain the selected recursion's validity range, or why it is pinned. */
function updateMethodHint() {
  const hint = document.getElementById('method-hint');
  if (!hint) return;
  const sel = document.getElementById('inp-method');
  const box = document.getElementById('chk-commission');
  hint.textContent = (box && box.checked)
    ? COMMISSION_METHOD_HINT
    : (METHOD_HINTS[parseInt(sel.value, 10)] || '');
}

/* ==================================================================
   Files
   ================================================================== */
async function onFileSelected(input) {
  if (!input.files.length) return;
  const file = input.files[0];
  const sizeMB = (file.size / 1024 / 1024).toFixed(1);
  const pathInput = document.getElementById('ggm-path');

  pathInput.value = file.name;
  pathInput.dataset.uploadedFile = file.name;

  const btn = document.getElementById('btn-load');
  btn.disabled = true;
  btn.textContent = 'Uploading 0%';
  showToast(`Uploading ${file.name} (${sizeMB} MB)…`, 'info');

  try {
    const formData = new FormData();
    formData.append('file', file);
    const nmax = document.getElementById('chk-use-nmax').checked
      ? '' : (document.getElementById('inp-nmax').value || '');
    if (nmax) formData.append('nmax', nmax);

    const data = await new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open('POST', `${API}/api/upload-model`);
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) {
          btn.textContent = `Uploading ${Math.round((e.loaded / e.total) * 100)}%`;
        }
      };
      xhr.onload = () => {
        try { resolve(JSON.parse(xhr.responseText)); }
        catch (err) { reject(new Error('Invalid server response')); }
      };
      xhr.onerror = () => reject(new Error('Upload failed — server unreachable'));
      xhr.ontimeout = () => reject(new Error('Upload timed out'));
      xhr.timeout = 600000;
      xhr.send(formData);
    });

    if (data.error) {
      showToast(`Upload failed: ${data.error}`, 'error');
      markFieldError('ggm-path');
      return;
    }

    btn.textContent = 'Parsing…';
    applyModelInfo(data);
    const serverPath = data.path || data.filename || file.name;
    pathInput.dataset.serverPath = serverPath;
    loadedModelPath = serverPath;
    setStatus('Model loaded', 'success');
    showToast(`Loaded ${data.filename} (nmax = ${data.nmax})`, 'success');
  } catch (err) {
    showToast(`Upload error: ${err.message}`, 'error');
    markFieldError('ggm-path');
  } finally {
    btn.disabled = false;
    btn.textContent = 'Load';
    updateRunSummary();
  }
}

async function onCovFileSelected(input) {
  if (!input.files.length) return;
  const file = input.files[0];
  showToast(`Uploading ${file.name}…`, 'info');
  try {
    const formData = new FormData();
    formData.append('file', file);
    const resp = await fetch(`${API}/api/upload-covariance`, { method: 'POST', body: formData });
    const data = await resp.json();
    if (!resp.ok || data.error) throw new Error(data.error || 'Upload failed');

    const pathInput = document.getElementById('cov-path');
    pathInput.value = data.path;
    pathInput.dataset.uploadedPath = data.path;
    const dim = data.dimension ? ` (${data.dimension}×${data.dimension})` : '';
    showToast(`Covariance uploaded: ${data.filename}${dim}`, 'success');
  } catch (err) {
    showToast(`Upload error: ${err.message}`, 'error');
    input.value = '';
  }
}

/* ==================================================================
   Model loading
   ================================================================== */
function applyModelInfo(data) {
  document.getElementById('inp-gm').value = Number(data.GM).toExponential(10);
  document.getElementById('inp-r').value = data.R;
  document.getElementById('inp-nmin').value = data.nmin;
  if (document.getElementById('chk-use-nmax').checked) {
    document.getElementById('inp-nmax').value = data.nmax;
  }
  showModelBadges(data);
  modelLoaded = true;
  updateRunSummary();
}

function showModelBadges(info) {
  const container = document.getElementById('model-info');
  container.innerHTML = '';
  [
    info.filename,
    `nmax = ${info.nmax}`,
    `GM = ${Number(info.GM).toExponential(4)}`,
    `R = ${info.R}`,
  ].forEach((text) => {
    if (!text) return;
    const span = document.createElement('span');
    span.className = 'info-badge';
    span.textContent = text;
    container.appendChild(span);
  });
}

async function fetchModelInfo() {
  try {
    const resp = await fetch(`${API}/api/model-info`);
    const data = await resp.json();
    if (data.loaded) {
      showModelBadges(data);
      modelLoaded = true;
      loadedModelPath = data.path || data.filename || '';
      updateRunSummary();
    }
  } catch (err) { /* server not ready yet */ }
}

async function loadModel() {
  const pathInput = document.getElementById('ggm-path');
  const path = pathInput.value.trim();

  const serverPath = pathInput.dataset.serverPath;
  const uploadedFile = pathInput.dataset.uploadedFile;
  const effectivePath = (uploadedFile && path === uploadedFile && serverPath) ? serverPath : path;

  if (modelLoaded && effectivePath && effectivePath === loadedModelPath) {
    showToast('This model is already loaded. Change the path to load a different model.', 'info');
    return;
  }
  if (!path) {
    showToast('Enter a GGM file path, or click Browse to select a file', 'error');
    markFieldError('ggm-path');
    return;
  }

  const nmax = document.getElementById('chk-use-nmax').checked
    ? null
    : (parseInt(document.getElementById('inp-nmax').value, 10) || null);

  const btn = document.getElementById('btn-load');
  btn.disabled = true;
  btn.textContent = 'Loading…';

  try {
    const resp = await fetch(`${API}/api/load-model`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path: effectivePath, nmax }),
    });
    const data = await resp.json();
    if (!resp.ok || data.error) throw new Error(data.error || `Server error ${resp.status}`);

    applyModelInfo(data);
    loadedModelPath = data.path || effectivePath;
    setStatus('Model loaded', 'success');
    showToast(`Loaded ${data.filename} (nmax = ${data.nmax})`, 'success');
  } catch (err) {
    showToast(err.message, 'error');
    markFieldError('ggm-path');
  } finally {
    btn.disabled = false;
    btn.textContent = 'Load';
    updateRunSummary();
  }
}

/* ==================================================================
   Compute
   ================================================================== */
function parseCSV(str) {
  if (!str || !str.trim()) return [];
  return str.split(',').map((s) => parseFloat(s.trim())).filter((n) => !Number.isNaN(n));
}

async function startCompute() {
  if (isComputing) {
    showToast('A computation is already running, please wait…', 'info');
    return;
  }

  clearFieldErrors();
  const errors = [];

  if (!modelLoaded) {
    errors.push('GGM model not loaded — click "Load" in the model panel first');
    markFieldError('ggm-path');
  }

  const codes = [];
  for (let i = 1; i <= 4; i++) {
    const v = parseInt(document.getElementById(`func-${i}`).value, 10);
    if (v > 0) codes.push(v);
  }
  if (codes.length === 0) {
    errors.push('Select at least one functional (Functional 1–4)');
    markFieldError('func-1');
  }

  const mode = document.querySelector('input[name="mode"]:checked').value;
  const coordType = parseInt(document.querySelector('input[name="coord-type"]:checked').value, 10);

  if (mode === 'grid') {
    const ids = ['lon-min', 'lon-step', 'lon-max', 'lat-min', 'lat-step', 'lat-max'];
    const labels = {
      'lon-min': 'Longitude min', 'lon-step': 'Longitude step', 'lon-max': 'Longitude max',
      'lat-min': 'Latitude min', 'lat-step': 'Latitude step', 'lat-max': 'Latitude max',
    };
    ids.forEach((id) => {
      const raw = document.getElementById(id).value;
      if (raw === '') {
        errors.push(`${labels[id]} is empty`);
        markFieldError(id);
      } else if (Number.isNaN(parseFloat(raw))) {
        errors.push(`${labels[id]} is not a number`);
        markFieldError(id);
      }
    });

    const lonStep = parseFloat(document.getElementById('lon-step').value);
    const latStep = parseFloat(document.getElementById('lat-step').value);
    if (!Number.isNaN(lonStep) && lonStep <= 0) {
      errors.push('Longitude step must be greater than 0');
      markFieldError('lon-step');
    }
    if (!Number.isNaN(latStep) && latStep <= 0) {
      errors.push('Latitude step must be greater than 0');
      markFieldError('lat-step');
    }

    const lonMin = parseFloat(document.getElementById('lon-min').value);
    const lonMax = parseFloat(document.getElementById('lon-max').value);
    const latMin = parseFloat(document.getElementById('lat-min').value);
    const latMax = parseFloat(document.getElementById('lat-max').value);
    if (!Number.isNaN(lonMin) && !Number.isNaN(lonMax) && lonMin >= lonMax) {
      errors.push('Longitude min must be smaller than max');
      markFieldError('lon-min');
      markFieldError('lon-max');
    }
    if (!Number.isNaN(latMin) && !Number.isNaN(latMax) && latMin >= latMax) {
      errors.push('Latitude min must be smaller than max');
      markFieldError('lat-min');
      markFieldError('lat-max');
    }
    if (latMin < -90 || latMax > 90) {
      errors.push('Latitude must stay within −90 … 90°');
      markFieldError('lat-min');
      markFieldError('lat-max');
    }
  }

  if (mode === 'pointwise') {
    const lonVal = document.getElementById('pw-lon').value.trim();
    const latVal = document.getElementById('pw-lat').value.trim();
    const hVal = document.getElementById('pw-h').value.trim();
    if (!lonVal) { errors.push('Point-wise: longitudes are empty'); markFieldError('pw-lon'); }
    if (!latVal) { errors.push('Point-wise: latitudes are empty'); markFieldError('pw-lat'); }
    if (!hVal) { errors.push('Point-wise: heights/radii are empty'); markFieldError('pw-h'); }

    const lonArr = parseCSV(lonVal);
    const latArr = parseCSV(latVal);
    const hArr = parseCSV(hVal);
    if (lonArr.length && latArr.length && lonArr.length !== latArr.length) {
      errors.push(`Point count mismatch: ${lonArr.length} longitudes vs ${latArr.length} latitudes`);
      markFieldError('pw-lon'); markFieldError('pw-lat');
    }
    if (lonArr.length && hArr.length && lonArr.length !== hArr.length) {
      errors.push(`Point count mismatch: ${lonArr.length} longitudes vs ${hArr.length} heights`);
      markFieldError('pw-h');
    }
  }

  if (document.getElementById('chk-commission').checked) {
    const covPath = document.getElementById('cov-path').value.trim();
    if (!covPath) {
      errors.push('Commission error is enabled but no covariance file is set');
      markFieldError('cov-path');
    }
  }

  if (errors.length > 0) {
    showChecklistToast(errors);
    const first = document.querySelector('.field-error');
    if (first) first.scrollIntoView({ behavior: 'smooth', block: 'center' });
    return;
  }

  const params = {
    functional_codes: codes,
    mode,
    ellipsoid: parseInt(document.getElementById('inp-ellipsoid').value, 10),
    method: parseInt(document.getElementById('inp-method').value, 10),
    coord_type: coordType,
    commission: document.getElementById('chk-commission').checked,
  };

  if (mode === 'grid') {
    params.lon_min = parseFloat(document.getElementById('lon-min').value);
    params.lon_max = parseFloat(document.getElementById('lon-max').value);
    params.lon_step = parseFloat(document.getElementById('lon-step').value);
    params.lat_min = parseFloat(document.getElementById('lat-min').value);
    params.lat_max = parseFloat(document.getElementById('lat-max').value);
    params.lat_step = parseFloat(document.getElementById('lat-step').value);
    params.h = parseFloat(document.getElementById('inp-h').value) || 0;
  } else {
    params.lon_deg = parseCSV(document.getElementById('pw-lon').value);
    params.lat_deg = parseCSV(document.getElementById('pw-lat').value);
    params.h_or_r = parseCSV(document.getElementById('pw-h').value);
  }

  const nmin = document.getElementById('inp-nmin').value;
  if (nmin !== '') params.nmin = parseInt(nmin, 10);

  if (params.commission) {
    params.covmat_path = document.getElementById('cov-path').value.trim();
    const nmaxCov = document.getElementById('inp-nmax-cov').value;
    if (nmaxCov) params.nmax_cov = parseInt(nmaxCov, 10);
  }

  isComputing = true;
  document.getElementById('btn-compute').disabled = true;
  document.getElementById('btn-compute-top').disabled = true;
  showProgress(true);
  setStatus('Computing…', 'computing');

  try {
    const resp = await fetch(`${API}/api/compute`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    const data = await resp.json();
    if (!resp.ok || data.error) throw new Error(data.error || `Server error ${resp.status}`);
    if (data.notice) showToast(data.notice, 'info', 6000);
    pollForResults();
  } catch (err) {
    showToast(err.message, 'error');
    showProgress(false);
    setStatus('Error', 'error');
    resetComputeButtons();
  }
}

function resetComputeButtons() {
  isComputing = false;
  document.getElementById('btn-compute').disabled = false;
  document.getElementById('btn-compute-top').disabled = false;
}

function pollForResults() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(async () => {
    try {
      const resp = await fetch(`${API}/api/status`);
      const status = await resp.json();
      document.getElementById('progress-msg').textContent = status.status;

      if (!status.computing) {
        clearInterval(pollTimer);
        pollTimer = null;
        showProgress(false);
        resetComputeButtons();

        if (status.error) {
          showToast(status.error, 'error', 8000);
          setStatus('Error', 'error');
        } else {
          setStatus('Complete', 'success');
          document.getElementById('footer-time').textContent =
            `Elapsed ${Number(status.elapsed || 0).toFixed(2)} s`;
          await fetchAndDisplayResults();
        }
      }
    } catch (err) {
      /* keep polling */
    }
  }, 500);
}

/* ==================================================================
   Results
   ================================================================== */
async function fetchAndDisplayResults() {
  try {
    const resp = await fetch(`${API}/api/results`);
    const data = await resp.json();
    if (!resp.ok || data.error) throw new Error(data.error || 'No results');

    currentResults = data;

    const vizSel = document.getElementById('viz-functional');
    vizSel.innerHTML = '';
    for (const [code, info] of Object.entries(data.functionals)) {
      const opt = document.createElement('option');
      opt.value = code;
      opt.textContent = `${info.name} [${code}]`;
      vizSel.appendChild(opt);
    }
    if (vizSel.options.length > 0) vizSel.selectedIndex = 0;

    updateVizInfo(data);
    updateVisualization();
    renderResultsTable(data);
    document.getElementById('btn-download').style.display = 'inline-flex';
    document.getElementById('footer-grid-info').textContent =
      `${data.lon.length} × ${data.lat.length} grid`;
    showToast('Computation complete', 'success');
  } catch (err) {
    showToast(`Failed to fetch results: ${err.message}`, 'error');
  }
}

function updateVizInfo(data) {
  const info = document.getElementById('viz-info');
  const nFuncs = Object.keys(data.functionals).length;
  info.innerHTML = `
    <span>functionals <span class="val">${nFuncs}</span></span>
    <span>lon <span class="val">${data.lon.length}</span></span>
    <span>lat <span class="val">${data.lat.length}</span></span>
    <span>time <span class="val">${Number(data.elapsed || 0).toFixed(2)}s</span></span>
  `;
}

function extractSeries(info) {
  const shape = info.shape || [];
  let values = info.data;

  // Multi-column tensor: keep the first column and say so in the title
  let columnNote = '';
  if (shape.length === 3) {
    values = values.map((row) => row.map((cell) => cell[0]));
    columnNote = ' (component 1)';
  } else if (shape.length === 2 && currentResults && currentResults.lon.length > 1
             && shape[0] === values.length && shape[1] > 1
             && shape[0] !== shape[1]) {
    // pointwise multi-column
    values = values.map((row) => row[0]);
    columnNote = ' (component 1)';
  }
  return { values, columnNote };
}

function updateVisualization() {
  const sel = document.getElementById('viz-functional');
  const code = sel.value;
  const placeholder = document.getElementById('plot-placeholder');

  if (!currentResults || !code || !currentResults.functionals[code]) {
    placeholder.style.display = 'flex';
    if (plot) plot.clear();
    return;
  }
  placeholder.style.display = 'none';

  const info = currentResults.functionals[code];
  const { values, columnNote } = extractSeries(info);
  const unit = UNIT_BY_CODE[parseInt(code, 10)] || '';
  const isGrid = currentResults.lon.length > 1 && currentResults.lat.length > 1
    && Array.isArray(values[0]);

  if (plot) {
    plot.setData({
      kind: isGrid ? 'grid' : 'points',
      lon: currentResults.lon,
      lat: currentResults.lat,
      values,
      title: info.name + columnNote,
      unit,
    });
  }
}

function renderResultsTable(data) {
  const panel = document.getElementById('results-panel');
  const container = document.getElementById('results-table-container');
  panel.classList.remove('hidden');
  container.style.display = 'block';

  const codes = Object.keys(data.functionals);
  const lon = data.lon;
  const isGrid = lon.length > 1 && data.lat.length > 1;
  const maxRows = 100;

  let html = '<table class="data-table"><thead><tr><th>#</th>';
  if (isGrid) html += '<th>lon</th><th>lat</th>';
  else html += '<th>lon</th><th>lat</th>';
  codes.forEach((c) => { html += `<th>${data.functionals[c].name}</th>`; });
  html += '</tr></thead><tbody>';

  const total = isGrid ? lon.length * data.lat.length : lon.length;
  const rows = Math.min(total, maxRows);

  for (let i = 0; i < rows; i++) {
    html += `<tr><td>${i + 1}</td>`;
    if (isGrid) {
      const nl = lon.length;
      const li = Math.floor(i / nl);
      const lj = i % nl;
      html += `<td>${lon[lj].toFixed(4)}</td><td>${data.lat[li].toFixed(4)}</td>`;
    } else {
      html += `<td>${lon[i].toFixed(4)}</td><td>${data.lat[i].toFixed(4)}</td>`;
    }
    codes.forEach((c) => {
      const series = data.functionals[c].data;
      let val;
      const shape = data.functionals[c].shape || [];
      if (isGrid) {
        const nl = lon.length;
        const li = Math.floor(i / nl);
        const lj = i % nl;
        const cell = series[li] ? series[li][lj] : null;
        val = Array.isArray(cell) ? cell[0] : cell;
      } else {
        val = Array.isArray(series[i]) ? series[i][0] : series[i];
      }
      if (typeof val === 'number' && Number.isFinite(val)) {
        html += `<td>${Math.abs(val) >= 1e4 || (Math.abs(val) < 1e-3 && val !== 0)
          ? val.toExponential(4) : val.toFixed(6)}</td>`;
      } else {
        html += '<td>—</td>';
      }
    });
    html += '</tr>';
  }
  html += '</tbody></table>';
  if (total > maxRows) {
    html += `<div style="padding:8px 2px;font-size:11px;color:var(--ink-3)">
      Showing ${maxRows} of ${total.toLocaleString()} rows — export the .txt for the full table.</div>`;
  }
  container.innerHTML = html;
}

/* ==================================================================
   Download
   ================================================================== */
function closeOutputMenu() {
  if (outputMenuEl) {
    outputMenuEl.remove();
    outputMenuEl = null;
  }
}

async function downloadResults() {
  closeOutputMenu();
  try {
    const resp = await fetch(`${API}/api/outputs`);
    const files = await resp.json();
    if (!Array.isArray(files) || files.length === 0) {
      showToast('No output files available', 'error');
      return;
    }
    if (files.length === 1) {
      triggerDownload(files[0].name);
      return;
    }

    const btn = document.getElementById('btn-download');
    const rect = btn.getBoundingClientRect();
    const menu = document.createElement('div');
    menu.style.cssText = `position:fixed;z-index:90;top:${rect.bottom + 4}px;right:16px;
      background:var(--bg-side);border:1px solid var(--border-strong);border-radius:3px;
      box-shadow:var(--shadow-pop);padding:4px;min-width:220px;max-height:280px;overflow:auto;`;
    files.forEach((f) => {
      const item = document.createElement('div');
      item.textContent = `${f.name}  (${(f.size / 1024).toFixed(0)} kB)`;
      item.style.cssText = 'padding:6px 9px;font-family:var(--mono);font-size:11px;' +
        'cursor:pointer;border-radius:4px;color:var(--ink);';
      item.onmouseenter = () => { item.style.background = '#F4F8FC'; };
      item.onmouseleave = () => { item.style.background = 'transparent'; };
      item.onclick = () => { triggerDownload(f.name); closeOutputMenu(); };
      menu.appendChild(item);
    });
    document.body.appendChild(menu);
    outputMenuEl = menu;
    setTimeout(() => document.addEventListener('click', closeOutputMenu, { once: true }), 0);
  } catch (err) {
    showToast(`Download failed: ${err.message}`, 'error');
  }
}

function triggerDownload(name) {
  const a = document.createElement('a');
  a.href = `${API}/api/download/${encodeURIComponent(name)}`;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  showToast(`Downloading ${name}`, 'success');
}
