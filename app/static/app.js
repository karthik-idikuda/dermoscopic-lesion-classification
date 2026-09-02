/* ==========================================================================
   Dermoscopic Lesion Analysis — frontend
   Vanilla ES2020, no build step. Consumes the FastAPI service under /api.
   ========================================================================== */
'use strict';

/* ---------------------------------------------------------------- utilities */

const $ = (sel, scope = document) => scope.querySelector(sel);
const $$ = (sel, scope = document) => Array.from(scope.querySelectorAll(sel));

/** Escape untrusted values before they reach innerHTML. */
function esc(value) {
  if (value === null || value === undefined) return '';
  return String(value)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

const isNum = (v) => v !== null && v !== undefined && v !== '' && Number.isFinite(Number(v));
const fx = (v, d = 2) => (isNum(v) ? Number(v).toFixed(d) : '—');
const pc = (v, d = 1) => (isNum(v) ? `${(Number(v) * 100).toFixed(d)}%` : '—');
const tc = (s) => String(s || '').replace(/[_-]+/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
/**
 * Inline sprite icon. Defaults to the `.ic` class, which pins the icon to 1em.
 * An <svg><use/></svg> with no sizing resolves to 100% of its container and
 * renders enormous, so a default is required rather than optional.
 */
const icon = (id, cls = '') =>
  `<svg class="${cls || 'ic'}" aria-hidden="true"><use href="#${id}"/></svg>`;

function kb(size) {
  if (!size) return '0 B';
  const u = ['B', 'KB', 'MB', 'GB'];
  const i = Math.min(u.length - 1, Math.floor(Math.log(size) / Math.log(1024)));
  return `${(size / 1024 ** i).toFixed(i ? 1 : 0)} ${u[i]}`;
}

function when(iso) {
  if (!iso) return '—';
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? String(iso) : d.toLocaleString();
}

/* ------------------------------------------------------------------ motion */

const REDUCED = window.matchMedia?.('(prefers-reduced-motion: reduce)');
const reduced = () => Boolean(REDUCED?.matches);

/**
 * Assign an incrementing --i to each match so CSS can stagger their entrance.
 * Cheaper and smoother than scheduling per-element timers in JS.
 */
function stagger(root, selector, limit = 40) {
  $$(selector, root || document).forEach((node, i) => {
    node.style.setProperty('--i', String(Math.min(i, limit)));
  });
}

/** Ease-out cubic, close enough to the CSS --ease-out feel. */
const easeOut = (t) => 1 - (1 - t) ** 3;

/**
 * Count a number up from zero. The target lives in data-count and the label is
 * rebuilt each frame so prefixes and suffixes ("/100", "%") stay intact.
 */
function countUp(root, duration = 900) {
  $$('[data-count]', root || document).forEach((node) => {
    const target = Number(node.dataset.count);
    if (!Number.isFinite(target)) return;
    const decimals = Number(node.dataset.decimals || 0);
    const prefix = node.dataset.prefix || '';
    const suffix = node.dataset.suffix || '';
    const render = (v) => { node.textContent = `${prefix}${v.toFixed(decimals)}${suffix}`; };

    if (reduced() || duration <= 0) { render(target); return; }

    const start = performance.now();
    const tick = (now) => {
      const t = Math.min(1, (now - start) / duration);
      render(target * easeOut(t));
      if (t < 1) requestAnimationFrame(tick);
      else render(target);
    };
    render(0);
    requestAnimationFrame(tick);
  });
}

/**
 * Animate SVG ring gauges from empty to their value.
 *
 * Rings render with stroke-dashoffset at full circumference and carry the target
 * in data-dash. Applying the target on a later frame gives the CSS transition two
 * distinct values to interpolate; writing the final value during render would
 * paint it already full with no animation at all.
 */
function drawGauges(root) {
  const rings = $$('[data-dash]', root || document);
  if (!rings.length) return;
  const apply = () => rings.forEach((r) => { r.style.strokeDashoffset = r.dataset.dash; });
  if (reduced()) apply();
  else requestAnimationFrame(() => requestAnimationFrame(apply));
}

/** Slide the ABCD total-dermoscopy-score marker from 0 to its position. */
function slideMarkers(root) {
  const marks = $$('[data-left]', root || document);
  if (!marks.length) return;
  const apply = () => marks.forEach((m) => { m.style.left = `${m.dataset.left}%`; });
  if (reduced()) apply();
  else requestAnimationFrame(() => requestAnimationFrame(apply));
}

/** Run every reveal effect over a freshly rendered subtree. */
function animate(root) {
  [
    '.prob', '.ab', '.kpi', '.mcell', '.kv__r', '.issue', '.driver',
    '.nar__sec', '.case', 'figure.fig', '.pipe__s',
  ].forEach((sel) => stagger(root, sel));
  stagger(root, 'table.tbl tbody tr', 60);
  drawGauges(root);
  slideMarkers(root);
  countUp(root);
}

/* ------------------------------------------------------------------- toasts */

function toast(message, kind = '') {
  const host = $('#toasts');
  const node = document.createElement('div');
  node.className = `toast${kind ? ` toast--${kind}` : ''}`;
  node.innerHTML =
    `${icon(kind === 'error' ? 'i-alert' : kind === 'ok' ? 'i-check' : 'i-info')}<span>${esc(message)}</span>`;
  host.appendChild(node);
  setTimeout(() => {
    node.classList.add('is-out');
    setTimeout(() => node.remove(), 220);
  }, 4200);
}

/* --------------------------------------------------------------------- state */

const S = {
  meta: null,
  classes: [],
  file: null,
  result: null,
  resultSelection: null,
  views: [],
  view: null,
  mode: 'single',
  splitAt: 50,
  selectionVersion: 0,
  requestVersion: 0,
  paintVersion: 0,
  activeAnalysis: null,
  previewUrl: null,
  cmp: { a: null, b: null },
  batch: [],
  key: localStorage.getItem('derm.key') || '',
};

function cancelActiveAnalysis() {
  const job = S.activeAnalysis;
  if (!job) return;
  clearInterval(job.tick);
  job.controller.abort();
  S.activeAnalysis = null;
}

function isCurrentAnalysis(job) {
  return S.activeAnalysis === job
    && S.selectionVersion === job.selection
    && S.file === job.file;
}

function revokePreviewUrl() {
  if (!S.previewUrl) return;
  URL.revokeObjectURL(S.previewUrl);
  S.previewUrl = null;
}

/** Remove every value/render derived from the previously analysed image. */
function clearAnalysisOutput() {
  S.result = null;
  S.resultSelection = null;
  S.views = [];
  S.view = null;
  S.paintVersion += 1;

  $('#result-top').innerHTML = '';
  $('#result-top').classList.add('hidden');
  $('#analyze-out').innerHTML = '';
  $('#analyze-out').classList.add('hidden');
  $('#analyze-busy').classList.add('hidden');
  $('#analyze-idle').classList.remove('hidden');
  $('#stage').classList.add('hidden');
  $('#stage-views').innerHTML = '';
  $('#stage-note').textContent = '';
  $('#stage-pre').textContent = '';
  $('#stage-frame').classList.remove('is-swapping');
  ['#img-under', '#img-over'].forEach((selector) => {
    const image = $(selector);
    image.removeAttribute('src');
    image.alt = '';
  });
  $('#card-attention').classList.add('hidden');
  $('#attention-metrics').innerHTML = '';
  $('#attention-note').textContent = '';
  $('#card-narrative').classList.add('hidden');
  $('#narrative').innerHTML = '';
}

async function sha256File(file) {
  if (!globalThis.crypto?.subtle) return null;
  const digest = await crypto.subtle.digest('SHA-256', await file.arrayBuffer());
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, '0')).join('');
}

function currentResult() {
  return S.result && S.resultSelection === S.selectionVersion ? S.result : null;
}

/* ----------------------------------------------------------------- transport */

async function api(path, options = {}) {
  const headers = { ...(options.headers || {}) };
  if (S.key) headers['X-API-Key'] = S.key;

  const res = await fetch(path, { ...options, headers });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) {
        detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail);
      }
    } catch { /* non-JSON error body — keep the status line */ }
    if (res.status === 401) askKey();
    throw new Error(detail);
  }
  const type = res.headers.get('content-type') || '';
  if (type.includes('application/json')) return res.json();
  if (type.includes('application/pdf')) return res.blob();
  return res.text();
}

function askKey() {
  const k = window.prompt('This server requires an API key (DERM_API_KEY):', S.key);
  if (k !== null) {
    S.key = k.trim();
    localStorage.setItem('derm.key', S.key);
  }
}

function save(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/* ------------------------------------------------------------------ dropzone */

const looksLikeImage = (f) =>
  f && (f.type.startsWith('image/') || /\.(jpe?g|png|webp|bmp|tiff?)$/i.test(f.name));

function dropzone(zone, input, onFiles, { multiple = false, paste = false } = {}) {
  const open = () => {
    // Clearing first lets selecting the same path fire `change` again.
    input.value = '';
    input.click();
  };

  zone.addEventListener('click', open);
  zone.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(); }
  });

  input.addEventListener('change', () => {
    const files = Array.from(input.files || []).filter(looksLikeImage);
    if (files.length) onFiles(multiple ? files : [files[0]]);
  });

  let depth = 0;
  ['dragenter', 'dragover'].forEach((t) =>
    zone.addEventListener(t, (e) => { e.preventDefault(); depth++; zone.classList.add('is-drag'); }));
  ['dragleave', 'dragend'].forEach((t) =>
    zone.addEventListener(t, () => { if (--depth <= 0) { depth = 0; zone.classList.remove('is-drag'); } }));
  zone.addEventListener('drop', (e) => {
    e.preventDefault();
    depth = 0;
    zone.classList.remove('is-drag');
    const files = Array.from(e.dataTransfer?.files || []).filter(looksLikeImage);
    if (!files.length) { toast('That does not look like an image file.', 'error'); return; }
    onFiles(multiple ? files : [files[0]]);
  });

  if (paste) {
    document.addEventListener('paste', (e) => {
      if ($('#view-analyze').classList.contains('hidden')) return;
      const item = Array.from(e.clipboardData?.items || [])
        .find((i) => i.type.startsWith('image/'));
      const file = item?.getAsFile();
      if (file) { onFiles([file]); toast('Image pasted from clipboard.'); }
    });
  }
}

/* ----------------------------------------------------------------- bootstrap */

document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  initNav();
  initModal();
  initAnalyze();
  initBatch();
  initCompare();
  initHistory();
  initShortcuts();
  loadMeta();
});

function initTheme() {
  const stored = localStorage.getItem('derm.theme');
  if (stored === 'dark' || stored === 'light') {
    document.documentElement.dataset.theme = stored;
  }
  $('#theme-toggle').addEventListener('click', () => {
    const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = next;
    localStorage.setItem('derm.theme', next);
  });
}

const VIEW_META = {
  analyze: ['Analyse', 'Single dermoscopic image — classification, explanation and severity grade'],
  batch: ['Batch triage', 'Score a set of images and review them in priority order'],
  compare: ['Track change', 'Measure how a lesion has evolved between two captures'],
  history: ['Case history', 'Every analysis stored locally, thumbnails only'],
  metrics: ['Model metrics', 'Evaluation artefacts written by the CLI scripts'],
  about: ['Method', 'How the pipeline works, and where it should not be trusted'],
};

function initNav() {
  $$('.navitem').forEach((btn) =>
    btn.addEventListener('click', () => showView(btn.dataset.view)));
}

function showView(name) {
  $$('.navitem').forEach((b) =>
    b.setAttribute('aria-selected', String(b.dataset.view === name)));
  $$('.view').forEach((v) => v.classList.toggle('hidden', v.id !== `view-${name}`));

  const [title, sub] = VIEW_META[name] || ['', ''];
  $('#view-title').textContent = title;
  $('#view-sub').textContent = sub;

  if (name === 'history') loadHistory();
  if (name === 'metrics') loadMetrics();
}

function initShortcuts() {
  document.addEventListener('keydown', (e) => {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement?.tagName || '');
    if (typing) return;

    if (e.key === 'Enter' && !$('#view-analyze').classList.contains('hidden')
        && S.file && !$('#btn-run').disabled) {
      e.preventDefault();
      runAnalysis();
    }
    if (e.key === 'Escape') $('#modal').classList.add('hidden');
  });
}

/* ---------------------------------------------------------------- meta load */

async function loadMeta() {
  try {
    const meta = await api('/api/meta');
    S.meta = meta;
    S.classes = meta.classes || [];

    $('#disclaimer-text').textContent = meta.disclaimer;
    $('#batch-limit').textContent = meta.limits?.max_batch_size ?? 24;
    if (meta.auth_required && !S.key) askKey();

    renderStatus(meta.model);
    fillClassSelect();
    renderClassTable();
    refreshCaseCount();
  } catch (err) {
    setStatus('error', 'API unreachable', err.message);
    toast(`Could not reach the API: ${err.message}`, 'error');
  }
}

function setStatus(kind, label, meta = '') {
  $('#statusblock').className = `statusblock statusblock--${kind}`;
  $('#status-label').textContent = label;
  $('#status-meta').textContent = meta;
  $('#statusblock').title = meta || label;
}

function renderStatus(model) {
  if (!model || model.error) {
    setStatus('error', 'Model unavailable', model?.error || 'unknown error');
    $('#notice-weights').classList.remove('hidden');
    return;
  }
  const trained = Boolean(model.is_trained);
  const acc = model.metrics?.test_accuracy;
  setStatus(
    trained ? 'ok' : 'warn',
    trained ? 'Model ready' : 'Untrained weights',
    `${model.architecture} · ${model.device}${acc ? ` · ${acc}% test acc` : ''}`,
  );
  $('#notice-weights').classList.toggle('hidden', trained);
}

$('#reload-model')?.addEventListener('click', async (e) => {
  const btn = e.currentTarget;
  btn.disabled = true;
  try {
    const res = await api('/api/model/reload', { method: 'POST' });
    renderStatus(res.model);
    toast(
      res.model.is_trained ? 'Trained checkpoint loaded.' : 'Reloaded — still no trained checkpoint found.',
      res.model.is_trained ? 'ok' : 'error',
    );
  } catch (err) {
    toast(err.message, 'error');
  } finally {
    btn.disabled = false;
  }
});

function fillClassSelect() {
  const sel = $('#h-code');
  sel.innerHTML = '<option value="">All classes</option>';
  S.classes.forEach((c) => {
    const o = document.createElement('option');
    o.value = c.code;
    o.textContent = c.short_name;
    sel.appendChild(o);
  });
}

function renderClassTable() {
  $('#about-classes').innerHTML = `<table class="tbl">
    <thead><tr><th>Code</th><th>Diagnosis</th><th>Class</th><th class="r">Images</th></tr></thead>
    <tbody>${S.classes.map((c) => `<tr>
      <td><code>${esc(c.code)}</code></td>
      <td><strong>${esc(c.name)}</strong><div class="t-small" style="margin-top:3px">${esc(c.description)}</div></td>
      <td><span class="tag tag--${esc(c.malignancy)}">${esc(c.malignancy)}</span></td>
      <td class="r">${c.ham10000_count.toLocaleString()}</td>
    </tr>`).join('')}</tbody></table>`;
}

async function refreshCaseCount() {
  try {
    const stats = await api('/api/cases/stats');
    const badge = $('#nav-case-count');
    badge.textContent = stats.total;
    badge.classList.toggle('hidden', !stats.total);
  } catch { /* non-critical */ }
}

/* ═════════════════════════════════ ANALYSE ════════════════════════════════ */

function initAnalyze() {
  dropzone($('#dropzone'), $('#file-input'), ([f]) => pickFile(f), { paste: true });
  $('#btn-run').addEventListener('click', runAnalysis);
  $('#btn-reset').addEventListener('click', resetAnalyze);
  $('#filebar-change').addEventListener('click', () => {
    showDropzone();
    const input = $('#file-input');
    input.value = '';
    input.click();
  });
  $('#btn-pdf').addEventListener('click', downloadPdf);
  $('#btn-copy').addEventListener('click', copyReport);
  $('#btn-json').addEventListener('click', downloadJson);
  initStage();
}

function pickFile(file) {
  const limit = S.meta?.limits?.max_upload_bytes ?? 12 * 1024 * 1024;
  if (file.size > limit) {
    toast(`That file is ${kb(file.size)}; the limit is ${kb(limit)}.`, 'error');
    return;
  }

  // A file selection is a new analysis identity. Cancel/ignore any older
  // request and remove every result derived from the previous image.
  S.selectionVersion += 1;
  cancelActiveAnalysis();
  clearAnalysisOutput();
  revokePreviewUrl();
  S.file = file;
  S.previewUrl = URL.createObjectURL(file);

  const prev = $('#preview');
  prev.src = S.previewUrl;
  prev.classList.remove('hidden');

  $('#filebar-thumb').src = S.previewUrl;
  $('#filebar-name').textContent = file.name;
  $('#filebar-meta').textContent = kb(file.size);

  $('#btn-run').disabled = false;
  $('#btn-reset').classList.remove('hidden');
  $('#run-error').classList.add('hidden');
  $('#dropzone').setAttribute('aria-label', `${file.name} selected. Press Enter to replace.`);
  showDropzone();
}

/** Swap the hero slot between the dropzone and the compact source bar. */
function showDropzone() {
  $('#dropzone').classList.remove('hidden');
  $('#filebar').classList.add('hidden');
}

function showFilebar() {
  $('#dropzone').classList.add('hidden');
  $('#filebar').classList.remove('hidden');
}

function resetAnalyze() {
  S.selectionVersion += 1;
  cancelActiveAnalysis();
  S.file = null;
  $('#file-input').value = '';
  revokePreviewUrl();

  const prev = $('#preview');
  prev.classList.add('hidden');
  prev.removeAttribute('src');
  $('#filebar-thumb').removeAttribute('src');
  $('#filebar-name').textContent = '';
  $('#filebar-meta').textContent = '';

  clearAnalysisOutput();
  $('#btn-run').disabled = true;
  $('#btn-reset').classList.add('hidden');
  $('#run-error').classList.add('hidden');
  showDropzone();
}

function options() {
  return {
    hair_removal: $('#o-hair').checked,
    color_constancy: $('#o-color').checked,
    vignette_crop: $('#o-vig').checked,
    segmentation: $('#o-seg').checked,
    morphometry: $('#o-morph').checked,
    gradcam: $('#o-cam').checked,
    gradcam_method: $('#o-method').value,
    colormap: $('#o-cmap').value,
    tta: $('#o-tta').checked,
    mc_dropout: $('#o-mc').checked,
    include_images: true,
    narrative: true,
    persist: $('#o-persist').checked,
  };
}

const STAGES = [
  'Decoding and quality assessment',
  'Hair removal and colour constancy',
  'Lesion segmentation',
  'ABCD morphometry',
  'Classification with augmentation',
  'Uncertainty estimation',
  'Grad-CAM explanation',
  'Severity grading and report',
];

/**
 * Render the pipeline checklist.
 *
 * Rows are mutated in place rather than re-created, so the completed rows keep
 * their existing DOM nodes and only the newly finished step plays its check
 * animation. Rebuilding the list each tick would replay every animation at once.
 */
function renderPipeline(active) {
  const host = $('#pipeline');

  if (host.children.length !== STAGES.length) {
    host.innerHTML = STAGES.map((label, i) => `<div class="pipe__s" style="--i:${i}">
      <span class="pipe__ic"></span><span>${esc(label)}</span></div>`).join('');
  }

  Array.from(host.children).forEach((row, i) => {
    const done = i < active;
    const now = i === active;
    if (done && !row.classList.contains('is-done')) {
      row.querySelector('.pipe__ic').innerHTML = icon('i-check');
    }
    row.classList.toggle('is-done', done);
    row.classList.toggle('is-now', now);
    if (!done) row.querySelector('.pipe__ic').innerHTML = '';
  });
}

async function runAnalysis() {
  if (!S.file) return;

  cancelActiveAnalysis();
  const job = {
    id: ++S.requestVersion,
    selection: S.selectionVersion,
    file: S.file,
    controller: new AbortController(),
    tick: null,
  };
  S.activeAnalysis = job;

  $('#analyze-idle').classList.add('hidden');
  $('#analyze-out').classList.add('hidden');
  $('#run-error').classList.add('hidden');
  $('#analyze-busy').classList.remove('hidden');
  $('#btn-run').disabled = true;

  let step = 0;
  renderPipeline(0);
  job.tick = setInterval(() => {
    if (!isCurrentAnalysis(job)) return;
    step = Math.min(step + 1, STAGES.length - 1);
    renderPipeline(step);
  }, 520);

  try {
    // Hash the exact bytes selected in the browser. The backend computes the
    // same value independently and the result is rendered only when they match.
    const fingerprint = await sha256File(job.file);
    if (!isCurrentAnalysis(job)) return;

    const form = new FormData();
    form.append('file', job.file);
    form.append('options', JSON.stringify(options()));

    const result = await api('/api/analyze', {
      method: 'POST',
      body: form,
      signal: job.controller.signal,
      cache: 'no-store',
    });
    if (!isCurrentAnalysis(job)) return;

    const source = result.source || {};
    const identityMatches = source.filename === job.file.name
      && Number(source.bytes) === job.file.size
      && (!fingerprint || source.sha256 === fingerprint);
    if (!identityMatches) {
      throw new Error('Upload identity mismatch. The result was not displayed; please analyse the image again.');
    }

    S.result = result;
    S.resultSelection = job.selection;
    clearInterval(job.tick);
    renderPipeline(STAGES.length);
    $('#analyze-busy').classList.add('hidden');
    renderResult(result);
    refreshCaseCount();
  } catch (err) {
    if (err.name === 'AbortError' || !isCurrentAnalysis(job)) return;
    clearInterval(job.tick);
    $('#analyze-busy').classList.add('hidden');
    $('#analyze-idle').classList.remove('hidden');
    const box = $('#run-error');
    box.innerHTML = `${icon('i-alert', 'issue__i')}<span>${esc(err.message)}</span>`;
    box.classList.remove('hidden');
    toast(err.message, 'error');
  } finally {
    if (isCurrentAnalysis(job)) {
      clearInterval(job.tick);
      S.activeAnalysis = null;
      $('#btn-run').disabled = !S.file;
    }
  }
}

/* ------------------------------------------------------------ result render */

function renderResult(r) {
  // Verdict spans the full workspace width; evidence cards go in the right rail.
  const top = $('#result-top');
  top.innerHTML = verdictHtml(r.severity, true) + reviewHtml(r.severity);
  top.classList.remove('hidden');

  const out = $('#analyze-out');
  const neuralUsable = r.severity?.neural_usable !== false;
  const diagnosisCap = neuralUsable
    ? ''
    : (r.quality?.is_skin_like === false ? 'Not a skin image — no diagnosis shown' : 'Untrained model — ranking is random');
  out.innerHTML = [
    cardHtml('Diagnosis', diagnosisHtml(r), diagnosisCap),
    cardHtml('ABCD morphometry', abcdHtml(r.morphology, r.segmentation), 'Stolz total dermoscopy score'),
    cardHtml('Confidence &amp; uncertainty', uncertaintyHtml(r), ''),
    cardHtml('Image quality', qualityHtml(r.quality), ''),
    cardHtml('What drove this grade', driversHtml(r.severity), ''),
    `<p class="t-small faint">Current upload <strong>${esc(r.source?.filename || r.filename || 'image')}</strong>
      · ${r.source?.width || '—'}×${r.source?.height || '—'} px · ${kb(r.source?.bytes || 0)}
      · fingerprint <code>${esc((r.source?.sha256 || '').slice(0, 12))}</code><br>
      Case <code>${esc(r.case_id)}</code> · ${Math.round(r.timings_ms?.total || 0)} ms total
      (classification ${Math.round(r.timings_ms?.classification || 0)} ms, Grad-CAM
      ${Math.round(r.timings_ms?.gradcam || 0)} ms)${r.persisted ? ' · saved to history' : ''}</p>`,
  ].join('');
  out.classList.remove('hidden');
  $('#analyze-idle').classList.add('hidden');
  showFilebar();

  animate(top);
  animate(out);

  $('#verdict-pdf')?.addEventListener('click', downloadPdf);
  $('#verdict-new')?.addEventListener('click', () => {
    resetAnalyze();
    $('#dropzone').focus();
  });

  renderStage(r);
  renderAttention(r.explanation);
  renderNarrative(r.narrative);
}

function cardHtml(title, body, cap) {
  return `<div class="card">
    <div class="card__head"><h3>${title}</h3>${cap ? `<span class="t-cap">${esc(cap)}</span>` : ''}</div>
    <div class="card__body">${body}</div>
  </div>`;
}

function verdictHtml(sev, withActions) {
  const C = 2 * Math.PI * 32;
  const frac = Math.max(0, Math.min(1, (sev.score || 0) / 100));
  // Rendered empty (dashoffset = C) with the target in data-dash; drawGauges()
  // applies it on a later frame so the ring sweeps into place.
  return `<div class="verdict tier-${esc(sev.tier)}">
    <div class="verdict__gauge">
      <svg viewBox="0 0 72 72"><circle class="gauge__bg" cx="36" cy="36" r="32"/>
        <circle class="gauge__fg" cx="36" cy="36" r="32" data-dash="${C * (1 - frac)}"
          style="stroke-dasharray:${C};stroke-dashoffset:${C}"/></svg>
      <div class="verdict__gv">
        <b data-count="${Math.round(sev.score || 0)}">0</b><span>score</span></div>
    </div>
    <div class="verdict__main">
      <span class="verdict__tier">${esc(sev.tier)}</span>
      <div class="verdict__h">${esc(sev.headline)}</div>
      <div class="verdict__r">${esc(sev.recommendation)}</div>
      <div class="verdict__when">${icon('i-clock')}${esc(sev.timeframe)}</div>
    </div>
    ${withActions ? `<div class="verdict__acts">
      <button class="btn btn--primary btn--sm" id="verdict-pdf" type="button">${icon('i-pdf')}Report</button>
      <button class="btn btn--ghost btn--sm" id="verdict-new" type="button">${icon('i-plus')}New</button>
    </div>` : '<div></div>'}
  </div>`;
}

function reviewHtml(sev) {
  if (!sev.requires_human_review) return '';
  const reasons = sev.review_reasons?.length
    ? `<ul style="margin-top:6px">${sev.review_reasons.map((x) => `<li>${esc(x)}</li>`).join('')}</ul>`
    : `<p>Severity tier ${esc(sev.tier)} always requires clinician review.</p>`;
  return `<div class="notice notice--info">
    ${icon('i-info', 'notice__icon')}
    <div class="notice__body"><b>Flagged for human review.</b>${reasons}</div>
  </div>`;
}

function diagnosisHtml(r) {
  const neuralUsable = r.severity?.neural_usable !== false;
  if (!neuralUsable) {
    const nonSkin = r.quality && r.quality.is_skin_like === false;
    return `<div class="issue issue--critical">
      ${icon('i-alert', 'issue__i')}
      <span>${nonSkin
        ? 'This does not look like a dermoscopic photograph of skin. The '
          + 'classifier was not run on it, and no diagnosis is shown — a named '
          + 'result here would just be noise from a network trained on skin '
          + 'close-ups, applied to something else entirely.'
        : 'No trained model weights are loaded, so there is nothing real to '
          + 'rank here. Class probabilities from an untrained network are '
          + 'random and are not shown as a diagnosis.'}</span>
    </div>
    <p class="t-small faint" style="margin-top:12px">Upload a clear, in-focus,
    close-up colour photograph of the lesion itself to get a real analysis.</p>`;
  }

  const p = r.prediction;
  const bars = (r.probabilities || []).map((e, i) => `<div class="prob${i === 0 ? ' is-top' : ''}">
      <span class="prob__dot" style="background:${esc(e.color)}"></span>
      <span class="prob__mid">
        <span class="prob__nm">${esc(e.short_name)}</span>
        <span class="prob__track"><span class="prob__fill"
          style="width:${Math.max(0.8, e.percentage)}%;background:${esc(e.color)}"></span></span>
      </span>
      <span class="prob__v">${fx(e.percentage, 1)}%</span>
    </div>`).join('');

  return `<div class="dx">
      <div class="dx__top">
        <span class="dx__name">${esc(p.short_name)}</span>
        <span class="tag tag--${esc(p.malignancy)}">${esc(p.malignancy)}</span>
        <span class="dx__pct" data-count="${p.percentage}" data-decimals="1" data-suffix="%">0.0%</span>
      </div>
      <p class="dx__desc">${esc(p.description)}</p>
      <p class="dx__desc"><strong style="color:var(--fg)">Usual management.</strong> ${esc(p.management)}</p>
    </div>
    <div class="probs" style="margin-top:14px">${bars}</div>`;
}

function abcdHtml(m, seg) {
  if (!m) return '<p class="t-small">Morphometry was disabled for this analysis.</p>';
  const a = m.abcd;
  const sh = m.shape;
  const co = m.color;

  const ring = (letter, score, max, name) => {
    const C = 2 * Math.PI * 19;
    const frac = max ? Math.max(0, Math.min(1, score / max)) : 0;
    return `<div class="ab">
      <div class="ab__ring">
        <svg viewBox="0 0 44 44"><circle class="bg" cx="22" cy="22" r="19"/>
          <circle class="fg" cx="22" cy="22" r="19" data-dash="${C * (1 - frac)}"
            style="stroke-dasharray:${C};stroke-dashoffset:${C}"/></svg>
        <span class="ab__l">${letter}</span>
      </div>
      <div class="ab__s"><span data-count="${score}">0</span><span class="faint">/${max}</span></div>
      <div class="ab__n">${name}</div>
    </div>`;
  };

  const chips = (list) => list?.length
    ? list.map((v) => `<span class="chip chip--static">${esc(tc(v))}</span>`).join(' ')
    : '<span class="t-small faint">none detected</span>';

  const band = a.interpretation === 'benign' ? 'LOW'
    : a.interpretation === 'suspicious' ? 'MODERATE' : 'HIGH';

  return `<div class="abcd">
      ${ring('A', a.asymmetry, 2, 'Asymmetry')}
      ${ring('B', a.border, 8, 'Border')}
      ${ring('C', a.colors, 6, 'Colour')}
      ${ring('D', a.structures, 5, 'Structures')}
    </div>
    <div class="tds">
      <div class="tds__top">
        <div><span class="t-cap">Total dermoscopy score</span>
          <div class="tds__v"><span data-count="${a.tds}" data-decimals="2">0.00</span><span
            class="faint" style="font-size:13px"> / ${fx(a.tds_max, 1)}</span></div>
        </div>
        <span class="badge tier-${band}">${esc(tc(a.interpretation))}</span>
      </div>
      <div class="tds__bar"><span class="tds__mk" data-left="${Math.max(0, Math.min(100, (a.tds / a.tds_max) * 100))}"
        style="left:0%"></span></div>
      <div class="tds__sc"><span>0 benign</span><span>4.76 suspicious</span><span>5.45+ high</span></div>
    </div>
    <div class="row" style="margin-top:14px;gap:6px"><span class="t-cap" style="width:100%">Colours present</span>${chips(a.colors_present)}</div>
    <div class="row" style="margin-top:10px;gap:6px"><span class="t-cap" style="width:100%">Structures</span>${chips(a.structures_present)}</div>
    <div class="kv" style="margin-top:14px">
      ${kvRow('Asymmetry index', fx(sh.asymmetry_index, 3))}
      ${kvRow('Border irregularity', fx(sh.border_irregularity, 3))}
      ${kvRow('Circularity', fx(sh.circularity, 3))}
      ${kvRow('Solidity', fx(sh.solidity, 3))}
      ${kvRow('Eccentricity', fx(sh.eccentricity, 3))}
      ${kvRow('Diameter', `${fx(sh.diameter_px, 0)} px <small>(${pc(sh.diameter_fraction, 0)} of frame)</small>`)}
      ${kvRow('Lesion/skin contrast', fx(co.lesion_skin_contrast, 3))}
      ${kvRow('Blue-white veil', `${pc(co.blue_white_veil, 1)} <small>of lesion</small>`)}
      ${kvRow('Segmentation', `${esc(tc(seg?.method))} <small>conf ${fx(seg?.confidence, 2)}</small>`)}
    </div>
    ${m.reliable ? '' : `<div class="issue issue--warning" style="margin-top:12px">
      ${icon('i-alert', 'issue__i')}<span>Segmentation was unreliable, so these geometric
      measurements are approximate.</span></div>`}`;
}

const kvRow = (k, v) => `<div class="kv__r"><span class="kv__k">${k}</span><span class="kv__v">${v}</span></div>`;

function uncertaintyHtml(r) {
  const u = r.uncertainty;
  if (!u) return '<p class="t-small">Uncertainty estimation was disabled.</p>';
  const neuralUsable = r.severity?.neural_usable !== false;
  if (!neuralUsable) {
    return '<p class="t-small">The classifier did not run on this image, so there is no '
      + 'prediction confidence to report.</p>';
  }
  const color = u.verdict === 'confident' ? 'low' : u.verdict === 'borderline' ? 'mod' : 'high';

  return `<div class="row" style="gap:12px;margin-bottom:12px">
      <div class="kpi" style="flex:1;min-width:120px">
        <div class="kpi__v" style="color:var(--${color})">${esc(tc(u.verdict))}</div>
        <div class="kpi__l">Stability</div>
      </div>
      <div class="kpi" style="flex:1;min-width:120px">
        <div class="kpi__v">${fx(r.prediction.percentage, 1)}%</div>
        <div class="kpi__l">Confidence</div>
      </div>
    </div>
    <div class="kv">
      ${kvRow('Normalised entropy', `${fx(u.entropy, 3)} <small>0 = certain, 1 = uniform</small>`)}
      ${kvRow('Top-two margin', fx(u.margin, 3))}
      ${kvRow('TTA agreement', `${pc(u.tta_agreement, 0)} <small>of ${u.n_tta} views</small>`)}
      ${kvRow('TTA spread', `±${fx(u.tta_std, 4)}`)}
      ${kvRow('MC dropout spread', u.n_mc ? `±${fx(u.mc_std, 4)} <small>over ${u.n_mc} passes</small>` : '<small>not available</small>')}
      ${kvRow('Mutual information', `${fx(u.mutual_information, 4)} <small>BALD</small>`)}
      ${kvRow('Temperature', `${fx(r.model?.temperature, 3)} <small>calibration</small>`)}
    </div>
    <p class="t-small faint" style="margin-top:12px">Dermoscopic images have no canonical
    orientation, so a prediction that changes when the image is flipped is not a stable one.</p>`;
}

function qualityHtml(q) {
  if (!q) return '<p class="t-small">No quality report.</p>';
  const color = q.verdict === 'good' ? 'low' : q.verdict === 'acceptable' ? 'mod' : 'high';
  const m = q.metrics || {};
  const issues = q.issues?.length
    ? q.issues.map((i) => `<div class="issue issue--${esc(i.severity)}">
        ${icon(i.severity === 'critical' ? 'i-alert' : 'i-info', 'issue__i')}
        <span>${esc(i.message)}</span></div>`).join('')
    : `<div class="issue issue--info">${icon('i-check', 'issue__i')}<span>No quality problems detected.</span></div>`;

  return `<div class="row" style="gap:12px;margin-bottom:12px">
      <div class="kpi" style="flex:1;min-width:120px">
        <div class="kpi__v" style="color:var(--${color})">${fx(q.score, 0)}<span class="faint" style="font-size:14px">/100</span></div>
        <div class="kpi__l">${esc(tc(q.verdict))}</div>
      </div>
      <div class="kpi" style="flex:1;min-width:120px">
        <div class="kpi__v" style="color:var(--${q.is_skin_like ? 'low' : 'high'})">${q.is_skin_like ? 'Yes' : 'No'}</div>
        <div class="kpi__l">Looks like skin</div>
        <div class="kpi__n">${pc(m.skin_fraction, 0)} skin-toned</div>
      </div>
    </div>
    ${issues}
    <div class="kv" style="margin-top:12px">
      ${kvRow('Resolution', esc(m.resolution))}
      ${kvRow('Focus measure', `${fx(m.sharpness, 0)} <small>variance of Laplacian</small>`)}
      ${kvRow('Brightness', `${fx(m.brightness, 0)} <small>/ 255</small>`)}
      ${kvRow('Contrast', fx(m.contrast, 0))}
      ${kvRow('Colourfulness', fx(m.colorfulness, 1))}
      ${kvRow('Specular glare', pc(m.glare_fraction, 1))}
    </div>`;
}

function driversHtml(sev) {
  const neuralUsable = sev.neural_usable !== false;
  const comps = Object.entries(sev.components || {})
    // The neural component is deliberately zeroed (not "low risk") whenever
    // the classifier's output was excluded from grading; showing a 0-width
    // bar next to real ones would read as a finding instead of an omission.
    .filter(([k]) => neuralUsable || k !== 'neural')
    .map(([k, v]) => `<div class="prob">
      <span class="prob__dot" style="background:var(--a)"></span>
      <span class="prob__mid"><span class="prob__nm">${esc(tc(k))} risk</span>
        <span class="prob__track"><span class="prob__fill"
          style="width:${Math.max(0, Math.min(100, v))}%;background:var(--a)"></span></span></span>
      <span class="prob__v">${fx(v, 0)}</span>
    </div>`).join('');

  const list = (sev.drivers || []).map((d) => `<div class="driver driver--${esc(d.direction)}">
      <div class="driver__t"><span class="driver__n">${esc(d.label)}</span>
        <span class="driver__p">${d.contribution ? `+${fx(d.contribution, 1)} pts` : ''}</span></div>
      <div class="driver__d">${esc(d.detail)}</div>
    </div>`).join('');

  const over = sev.overrides_applied?.length
    ? `<span class="t-cap" style="display:block;margin:14px 0 7px">Safety overrides applied</span>
       <ul style="padding-left:16px;display:flex;flex-direction:column;gap:5px">
       ${sev.overrides_applied.map((o) => `<li class="t-small">${esc(o)}</li>`).join('')}</ul>`
    : '';

  return `<div class="probs" style="margin-bottom:14px">${comps}</div>${list}${over}
    <p class="t-small faint" style="margin-top:12px">Overrides can raise the tier but never lower it.
    The cost of a missed melanoma is not symmetric with the cost of an unnecessary referral.</p>`;
}

/* ------------------------------------------------------------------- stage */

const VIEWS = {
  overlay: ['Grad-CAM', 'Warm regions carry the evidence for the predicted class.'],
  original: ['Original', 'The submitted image after EXIF rotation.'],
  heatmap: ['Heatmap', 'The raw activation map without the underlying image.'],
  cam_contour: ['Evidence outline', 'Iso-contour at 60% of peak activation.'],
  segmentation: ['Lesion mask', 'Green tint marks the segmented lesion.'],
  contour: ['Outline', 'Detected lesion boundary and bounding box.'],
  restored: ['Restored', 'After hair inpainting and colour constancy — geometry only.'],
  hair_mask: ['Hair mask', 'Pixels identified as hair and inpainted before measurement.'],
};
const VIEW_ORDER = ['overlay', 'original', 'heatmap', 'cam_contour', 'segmentation', 'contour', 'restored', 'hair_mask'];

function initStage() {
  $$('#stage-mode .seg__btn').forEach((btn) =>
    btn.addEventListener('click', () => setMode(btn.dataset.mode)));
  wireSplit();
}

function renderStage(r) {
  const images = r.images || {};
  S.views = VIEW_ORDER.filter((k) => images[k]);
  if (!S.views.length) { $('#stage').classList.add('hidden'); return; }

  S.view = S.views.includes('overlay') ? 'overlay' : S.views[0];

  $('#stage-views').innerHTML = S.views.map((k) => {
    const [label] = VIEWS[k] || [tc(k)];
    return `<button class="chip" role="tab" data-view="${k}"
      aria-selected="${k === S.view}">${esc(label)}</button>`;
  }).join('');

  $$('#stage-views .chip').forEach((chip) =>
    chip.addEventListener('click', () => {
      S.view = chip.dataset.view;
      $$('#stage-views .chip').forEach((c) =>
        c.setAttribute('aria-selected', String(c === chip)));
      paintStage();
    }));

  const canSplit = Boolean(images.original && images.overlay);
  $('#stage-mode').classList.toggle('hidden', !canSplit);
  if (!canSplit && S.mode === 'split') S.mode = 'single';

  $('#stage').classList.remove('hidden');
  setMode(S.mode);

  const steps = r.preprocessing?.length ? r.preprocessing.join(' · ') : 'no restoration required';
  $('#stage-pre').textContent = `Preprocessing: ${steps}`;
}

function setMode(mode) {
  S.mode = mode;
  $$('#stage-mode .seg__btn').forEach((b) =>
    b.setAttribute('aria-selected', String(b.dataset.mode === mode)));
  const split = mode === 'split';
  $('#split').classList.toggle('hidden', !split);
  $('#tag-l').classList.toggle('hidden', !split);
  $('#tag-r').classList.toggle('hidden', !split);
  $('#stage-views').parentElement.classList.toggle('hidden', split);
  paintStage();
}

function paintStage() {
  const result = S.result;
  const images = result?.images || {};
  const under = $('#img-under');
  const over = $('#img-over');
  const frame = $('#stage-frame');
  const paint = ++S.paintVersion;
  frame.classList.remove('is-swapping');

  if (!result) {
    under.removeAttribute('src');
    over.removeAttribute('src');
    return;
  }

  if (S.mode === 'split') {
    under.src = images.original || '';
    over.src = images.overlay || '';
    over.style.clipPath = `inset(0 0 0 ${S.splitAt}%)`;
    over.alt = 'Grad-CAM overlay, right of the comparison divider';
    $('#split-line').style.left = `${S.splitAt}%`;
    $('#stage-note').textContent =
      'Drag the divider to compare the original against the Grad-CAM overlay.';
    return;
  }

  over.style.clipPath = 'none';
  under.removeAttribute('src');
  const selectedView = S.view;
  const [label, caption] = VIEWS[selectedView] || [tc(selectedView), ''];
  const next = images[selectedView] || '';

  // A decode from an older tab/result must never overwrite the current image.
  const stillCurrent = () => S.paintVersion === paint
    && S.result === result
    && S.view === selectedView
    && S.mode === 'single';

  if (over.getAttribute('src') && over.getAttribute('src') !== next && !reduced()) {
    frame.classList.add('is-swapping');
    const probe = new Image();
    probe.src = next;
    const show = () => {
      if (!stillCurrent()) return;
      over.src = next;
      over.alt = `${label} view`;
      requestAnimationFrame(() => {
        if (stillCurrent()) frame.classList.remove('is-swapping');
      });
    };
    (probe.decode ? probe.decode().catch(() => {}) : Promise.resolve()).then(show);
  } else if (stillCurrent()) {
    if (next) over.src = next;
    else over.removeAttribute('src');
    over.alt = `${label} view`;
  }

  if (stillCurrent()) $('#stage-note').textContent = caption;
}

function wireSplit() {
  const split = $('#split');
  let dragging = false;

  const move = (clientX) => {
    const rect = $('#stage-frame').getBoundingClientRect();
    const pct = ((clientX - rect.left) / rect.width) * 100;
    S.splitAt = Math.max(0, Math.min(100, pct));
    split.setAttribute('aria-valuenow', String(Math.round(S.splitAt)));
    paintStage();
  };

  split.addEventListener('pointerdown', (e) => {
    dragging = true;
    split.setPointerCapture(e.pointerId);
    move(e.clientX);
  });
  split.addEventListener('pointermove', (e) => { if (dragging) move(e.clientX); });
  split.addEventListener('pointerup', (e) => {
    dragging = false;
    split.releasePointerCapture(e.pointerId);
  });
  split.addEventListener('keydown', (e) => {
    const delta = e.key === 'ArrowLeft' ? -4 : e.key === 'ArrowRight' ? 4 : 0;
    if (!delta) return;
    e.preventDefault();
    S.splitAt = Math.max(0, Math.min(100, S.splitAt + delta));
    split.setAttribute('aria-valuenow', String(Math.round(S.splitAt)));
    paintStage();
  });
}

function renderAttention(explanation) {
  const card = $('#card-attention');
  const a = explanation?.attention;
  if (!a) { card.classList.add('hidden'); return; }
  card.classList.remove('hidden');

  const good = (a.verdict ?? 0) >= 0.5;
  $('#attention-metrics').innerHTML = `
    <div class="mcell"><div class="mcell__l">Inside lesion</div>
      <div class="mcell__v">${pc(a.inside_ratio, 0)}</div>
      <div class="mcell__n">of total activation</div></div>
    <div class="mcell"><div class="mcell__l">Attention lift</div>
      <div class="mcell__v">${a.lift === null ? '∞' : `${fx(a.lift, 1)}×`}</div>
      <div class="mcell__n">inside vs outside</div></div>
    <div class="mcell"><div class="mcell__l">Focus</div>
      <div class="mcell__v">${pc(explanation.concentration, 0)}</div>
      <div class="mcell__n">in hottest 10%</div></div>
    <div class="mcell"><div class="mcell__l">Verdict</div>
      <div class="mcell__v" style="color:var(--${good ? 'low' : 'high'})">${good ? 'Anchored' : 'Poor'}</div>
      <div class="mcell__n">${good ? 'on the lesion' : 'localisation'}</div></div>`;

  $('#attention-note').textContent = good
    ? 'The explanation overlaps the segmented lesion, so the prediction is anchored to it.'
    : 'Evidence sits largely outside the lesion. The prediction may be responding to background skin, hair or frame artefacts — weight it accordingly.';

  animate(card);
}

function renderNarrative(n) {
  const card = $('#card-narrative');
  if (!n) { card.classList.add('hidden'); return; }
  card.classList.remove('hidden');

  const sec = (title, items) => items?.length
    ? `<div class="nar__sec"><h4 class="t-cap">${esc(title)}</h4>
       <ul>${items.map((i) => `<li>${esc(i)}</li>`).join('')}</ul></div>`
    : '';

  $('#narrative').innerHTML = `
    <div class="nar__imp">${esc(n.impression)}</div>
    <div class="nar__sec"><h4 class="t-cap">Summary</h4><p>${esc(n.summary)}</p></div>
    ${sec('Findings', n.findings)}
    ${sec('Differential', n.differential)}
    ${sec('Basis for the prediction', n.explanation)}
    ${sec('Recommendation', n.recommendation)}
    ${sec('Limitations', n.limitations)}
    <p class="t-small faint"><strong>Disclaimer.</strong> ${esc(n.disclaimer)}</p>`;

  animate(card);
}

/* --------------------------------------------------------------- exports */

async function downloadPdf() {
  const result = currentResult();
  if (!result) return;
  const selection = S.selectionVersion;
  const btn = $('#btn-pdf');
  const original = btn.innerHTML;
  btn.disabled = true;
  btn.textContent = 'Rendering…';
  try {
    const blob = await api('/api/report/pdf', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(result),
      cache: 'no-store',
    });
    if (S.selectionVersion !== selection || currentResult() !== result) return;
    save(blob, `lesion-report-${result.case_id}.pdf`);
    toast('PDF report downloaded.', 'ok');
  } catch (err) {
    if (S.selectionVersion === selection) toast(err.message, 'error');
  } finally {
    if (S.selectionVersion === selection) {
      btn.disabled = false;
      btn.innerHTML = original;
    }
  }
}

function reportText(r) {
  const n = r.narrative || {};
  const lines = [
    'DERMOSCOPIC LESION ANALYSIS',
    `Case ${r.case_id} · ${r.created_at}`,
    `Source: ${r.filename || 'uploaded image'}`,
    '',
    `SEVERITY: ${r.severity.tier} (${Math.round(r.severity.score)}/100)`,
    r.severity.headline,
    `${r.severity.recommendation} (${r.severity.timeframe})`,
    '',
    'IMPRESSION', n.impression || '',
    '', 'SUMMARY', n.summary || '',
    '', 'CLASS PROBABILITIES',
    ...(r.probabilities || []).map((e) =>
      `  ${e.code.padEnd(6)} ${e.percentage.toFixed(2).padStart(6)}%  ${e.name}`),
  ];
  [['FINDINGS', n.findings], ['DIFFERENTIAL', n.differential],
   ['BASIS FOR THE PREDICTION', n.explanation], ['RECOMMENDATION', n.recommendation],
   ['LIMITATIONS', n.limitations]].forEach(([t, items]) => {
    if (items?.length) lines.push('', t, ...items.map((i) => `  - ${i}`));
  });
  lines.push('', 'DISCLAIMER', r.disclaimer || '');
  return lines.join('\n');
}

async function copyReport() {
  const result = currentResult();
  if (!result) return;
  const text = reportText(result);
  try {
    await navigator.clipboard.writeText(text);
    toast('Report copied to the clipboard.', 'ok');
  } catch {
    save(new Blob([text], { type: 'text/plain' }), `lesion-report-${result.case_id}.txt`);
    toast('Clipboard unavailable — downloaded as text instead.');
  }
}

function downloadJson() {
  const result = currentResult();
  if (!result) return;
  const copy = { ...result, images: '<stripped>' };
  save(new Blob([JSON.stringify(copy, null, 2)], { type: 'application/json' }),
    `lesion-analysis-${result.case_id}.json`);
  toast('JSON downloaded — base64 images stripped.', 'ok');
}

/* ═══════════════════════════════════ BATCH ════════════════════════════════ */

function initBatch() {
  dropzone($('#batch-drop'), $('#batch-input'), (files) => {
    const limit = S.meta?.limits?.max_batch_size ?? 24;
    S.batch = files.slice(0, limit);
    if (files.length > limit) toast(`Only the first ${limit} files will be analysed.`);
    $('#batch-label').textContent = `${S.batch.length} image${S.batch.length === 1 ? '' : 's'} selected`;
    $('#batch-count-cap').textContent = `${S.batch.length} queued`;
    $('#batch-run').disabled = !S.batch.length;
  }, { multiple: true });

  $('#batch-run').addEventListener('click', runBatch);
}

async function runBatch() {
  if (!S.batch.length) return;
  const btn = $('#batch-run');
  btn.disabled = true;
  $('#batch-error').classList.add('hidden');
  $('#batch-bar').classList.remove('hidden');
  $('#batch-out').classList.add('hidden');

  const form = new FormData();
  S.batch.forEach((f) => form.append('files', f));
  form.append('options', JSON.stringify({
    ...options(), include_images: false, persist: $('#batch-persist').checked,
  }));

  try {
    const res = await api('/api/analyze/batch', { method: 'POST', body: form });
    renderBatch(res);
    $('#batch-out').classList.remove('hidden');
    animate($('#batch-out'));
    toast(`Analysed ${res.succeeded} of ${res.count} images.`, 'ok');
    refreshCaseCount();
  } catch (err) {
    const box = $('#batch-error');
    box.innerHTML = `${icon('i-alert', 'issue__i')}<span>${esc(err.message)}</span>`;
    box.classList.remove('hidden');
    toast(err.message, 'error');
  } finally {
    $('#batch-bar').classList.add('hidden');
    btn.disabled = false;
  }
}

/**
 * KPI tile. A plain numeric value is animated with a count-up; anything with
 * mixed content (e.g. "3/17") is rendered as-is.
 */
const kpi = (v, l, color, note) => {
  const raw = String(v);
  const plain = /^-?\d+(\.\d+)?$/.test(raw);
  const pctish = /^-?\d+(\.\d+)?%$/.test(raw);
  let inner = raw;
  if (plain || pctish) {
    const n = parseFloat(raw);
    const decimals = (raw.split('.')[1] || '').replace('%', '').length;
    inner = `<span data-count="${n}" data-decimals="${decimals}"${
      pctish ? ' data-suffix="%"' : ''}>0</span>`;
  }
  return `<div class="kpi">
    <div class="kpi__v"${color ? ` style="color:var(--${color})"` : ''}>${inner}</div>
    <div class="kpi__l">${esc(l)}</div>
    ${note ? `<div class="kpi__n">${esc(note)}</div>` : ''}</div>`;
};

function renderBatch(res) {
  const tiers = res.tier_distribution || {};
  const urgent = (tiers.CRITICAL || 0) + (tiers.HIGH || 0);

  const queue = res.priority_queue || [];
  const queueHtml = queue.length
    ? `<div class="tw"><table class="tbl">
        <thead><tr><th>#</th><th>File</th><th>Tier</th><th>Prediction</th><th class="r">Confidence</th></tr></thead>
        <tbody>${queue.map((row, i) => `<tr>
          <td class="r">${i + 1}</td><td>${esc(row.filename)}</td>
          <td><span class="badge tier-${esc(row.tier)}">${esc(row.tier)}</span></td>
          <td>${esc(row.prediction)}</td><td class="r">${fx(row.confidence, 1)}%</td>
        </tr>`).join('')}</tbody></table></div>`
    : '<div class="card__body"><p class="t-small">No images were graded HIGH or CRITICAL in this batch.</p></div>';

  const rows = (res.items || []).map((it) => {
    if (!it.ok) {
      return `<tr><td>${esc(it.filename)}</td>
        <td colspan="6" style="color:var(--high)">${esc(it.error)}</td></tr>`;
    }
    const r = it.result;
    const tds = r.morphology?.abcd?.tds;
    const neuralUsable = r.severity?.neural_usable !== false;
    const predictionCell = neuralUsable
      ? `<td>${esc(r.prediction.short_name)}</td><td class="r">${fx(r.prediction.percentage, 1)}%</td>`
      : `<td colspan="2" class="t-small faint">${r.quality.is_skin_like === false ? 'Not a skin image' : 'Untrained model'}</td>`;
    return `<tr>
      <td>${esc(it.filename)}</td>
      <td><span class="badge tier-${esc(r.severity.tier)}">${esc(r.severity.tier)}</span></td>
      ${predictionCell}
      <td class="r">${fx(r.severity.score, 0)}</td>
      <td class="r">${isNum(tds) ? fx(tds, 2) : '—'}</td>
      <td class="r">${fx(r.quality.score, 0)}</td>
    </tr>`;
  }).join('');

  $('#batch-out').innerHTML = `
    <div class="grid g-4">
      ${kpi(res.succeeded, 'Analysed')}
      ${kpi(res.failed, 'Failed', res.failed ? 'high' : '')}
      ${kpi(urgent, 'High or critical', urgent ? 'high' : '')}
      ${kpi(queue.length, 'Review first', queue.length ? 'mod' : '')}
    </div>
    <div class="card">
      <div class="card__head"><h3>Priority queue</h3><span class="t-cap">highest tier first</span></div>
      <div class="card__body card__body--flush">${queueHtml}</div>
    </div>
    <div class="card">
      <div class="card__head"><h3>All results</h3></div>
      <div class="card__body card__body--flush"><div class="tw"><table class="tbl">
        <thead><tr><th>File</th><th>Tier</th><th>Prediction</th><th class="r">Conf.</th>
        <th class="r">Severity</th><th class="r">TDS</th><th class="r">Quality</th></tr></thead>
        <tbody>${rows}</tbody></table></div></div>
    </div>`;
}

/* ══════════════════════════════════ COMPARE ═══════════════════════════════ */

function initCompare() {
  const wire = (zoneSel, inputSel, prevSel, slot) =>
    dropzone($(zoneSel), $(inputSel), ([f]) => {
      S.cmp[slot] = f;
      const img = $(prevSel);
      img.src = URL.createObjectURL(f);
      img.classList.remove('hidden');
      $('#cmp-run').disabled = !(S.cmp.a && S.cmp.b);
    });

  wire('#cmp-drop-a', '#cmp-in-a', '#cmp-prev-a', 'a');
  wire('#cmp-drop-b', '#cmp-in-b', '#cmp-prev-b', 'b');
  $('#cmp-run').addEventListener('click', runCompare);
}

async function runCompare() {
  if (!(S.cmp.a && S.cmp.b)) return;
  const btn = $('#cmp-run');
  btn.disabled = true;
  btn.textContent = 'Comparing…';
  $('#cmp-error').classList.add('hidden');

  const form = new FormData();
  form.append('baseline', S.cmp.a);
  form.append('followup', S.cmp.b);
  const da = $('#cmp-date-a').value;
  const db = $('#cmp-date-b').value;
  if (da) form.append('baseline_date', da);
  if (db) form.append('followup_date', db);
  const fov = $('#cmp-fov').value;
  if (fov) form.append('frame_width_mm', fov);
  form.append('include_images', 'true');

  try {
    const res = await api('/api/compare', { method: 'POST', body: form });
    renderCompare(res);
    $('#cmp-out').classList.remove('hidden');
    animate($('#cmp-out'));
  } catch (err) {
    const box = $('#cmp-error');
    box.innerHTML = `${icon('i-alert', 'issue__i')}<span>${esc(err.message)}</span>`;
    box.classList.remove('hidden');
    toast(err.message, 'error');
  } finally {
    btn.disabled = false;
    btn.textContent = 'Compare captures';
  }
}

function renderCompare(res) {
  const tier = { stable: 'LOW', minor_change: 'MODERATE', significant_change: 'HIGH' }[res.verdict]
    || 'INDETERMINATE';
  const C = 2 * Math.PI * 32;
  const frac = Math.max(0, Math.min(1, res.change_score / 100));

  const panels = [
    ['baseline', 'Baseline lesion'], ['followup', 'Follow-up lesion'],
    ['difference', 'Difference map'], ['baseline_contour', 'Baseline outline'],
    ['followup_contour', 'Follow-up outline'],
  ].filter(([k]) => res.images?.[k]);

  const rows = (res.metrics || []).map((m) => {
    const dir = Math.abs(m.percent_change) < 1 ? 'flat' : m.percent_change > 0 ? 'up' : 'down';
    const arrow = dir === 'flat' ? '→' : dir === 'up' ? '↑' : '↓';
    return `<tr>
      <td>${esc(m.name)}${m.note ? `<div class="t-small faint" style="margin-top:3px">${esc(m.note)}</div>` : ''}</td>
      <td class="r">${fx(m.baseline, 3)}</td>
      <td class="r">${fx(m.followup, 3)}</td>
      <td class="r delta-${dir}">${arrow} ${fx(Math.abs(m.percent_change), 1)}%</td>
      <td>${m.significant ? '<span class="badge tier-HIGH">significant</span>' : '<span class="faint">—</span>'}</td>
    </tr>`;
  }).join('');

  const list = (title, items) => items?.length
    ? `<span class="t-cap" style="display:block;margin:12px 0 6px">${esc(title)}</span>
       <div class="row" style="gap:6px">${items.map((i) => `<span class="chip chip--static">${esc(tc(i))}</span>`).join('')}</div>`
    : '';

  $('#cmp-out').innerHTML = `
    <div class="verdict tier-${tier}">
      <div class="verdict__gauge">
        <svg viewBox="0 0 72 72"><circle class="gauge__bg" cx="36" cy="36" r="32"/>
          <circle class="gauge__fg" cx="36" cy="36" r="32" data-dash="${C * (1 - frac)}"
            style="stroke-dasharray:${C};stroke-dashoffset:${C}"/></svg>
        <div class="verdict__gv">
          <b data-count="${Math.round(res.change_score)}">0</b><span>change</span></div>
      </div>
      <div class="verdict__main">
        <span class="verdict__tier">${esc(tc(res.verdict))}</span>
        <div class="verdict__h">${esc(res.headline)}</div>
        <div class="verdict__r">${esc(res.recommendation)}</div>
        <div class="verdict__when">${icon('i-clock')}${
          res.days_between !== null ? `${res.days_between} days apart` : 'capture dates not supplied'}</div>
      </div>
      <div></div>
    </div>
    ${panels.length ? `<div class="grid g-3">${panels.map(([k, cap]) => `<figure class="fig">
      <img src="${esc(res.images[k])}" alt="${esc(cap)}"><figcaption>${esc(cap)}</figcaption>
    </figure>`).join('')}</div>` : ''}
    <div class="card">
      <div class="card__head"><h3>Measured differences</h3></div>
      <div class="card__body card__body--flush"><div class="tw"><table class="tbl">
        <thead><tr><th>Measurement</th><th class="r">Baseline</th><th class="r">Follow-up</th>
        <th class="r">Change</th><th>Flag</th></tr></thead><tbody>${rows}</tbody></table></div></div>
    </div>
    <div class="card">
      <div class="card__head"><h3>Interpretation and caveats</h3></div>
      <div class="card__body">
        <div class="kv">
          ${kvRow('Structural similarity', `${fx(res.structural_similarity, 3)} <small>1 = identical</small>`)}
          ${kvRow('Growth per month', res.growth_per_month === null ? '<small>not computable</small>' : pc(res.growth_per_month, 1))}
        </div>
        ${list('New colours appearing', res.new_colors)}
        ${list('Colours no longer present', res.lost_colors)}
        ${list('New structures', res.new_structures)}
        <span class="t-cap" style="display:block;margin:14px 0 6px">Caveats</span>
        <ul style="padding-left:16px;display:flex;flex-direction:column;gap:5px">
          ${(res.caveats || []).map((c) => `<li class="t-small">${esc(c)}</li>`).join('')}</ul>
      </div>
    </div>`;
}

/* ══════════════════════════════════ HISTORY ═══════════════════════════════ */

function initHistory() {
  $('#h-refresh').addEventListener('click', loadHistory);
  ['#h-tier', '#h-code'].forEach((s) => $(s).addEventListener('change', loadHistory));
  $('#h-review').addEventListener('change', loadHistory);
}

async function loadHistory() {
  const params = new URLSearchParams({ limit: '60' });
  const tier = $('#h-tier').value;
  const code = $('#h-code').value;
  if (tier) params.set('tier', tier);
  if (code) params.set('code', code);
  if ($('#h-review').checked) params.set('review_only', 'true');

  const list = $('#h-list');
  list.innerHTML = `<div class="cases">${'<div class="skel" style="height:250px"></div>'.repeat(4)}</div>`;

  try {
    const [page, stats] = await Promise.all([api(`/api/cases?${params}`), api('/api/cases/stats')]);

    const urgent = (stats.by_tier || [])
      .filter((r) => r.tier === 'HIGH' || r.tier === 'CRITICAL')
      .reduce((t, r) => t + r.count, 0);

    $('#h-stats').innerHTML = [
      kpi(stats.total, 'Cases stored'),
      kpi(urgent, 'High or critical', urgent ? 'high' : ''),
      kpi(stats.flagged_for_review, 'Flagged for review', stats.flagged_for_review ? 'mod' : ''),
      kpi(`${fx(stats.mean_confidence * 100, 1)}%`, 'Mean confidence'),
    ].join('');

    if (!page.items.length) {
      list.innerHTML = `<div class="empty">
        ${icon('i-archive', 'empty__art')}
        <h3>No cases match</h3>
        <p>Analyses are stored here automatically while "Save to history" is enabled.</p></div>`;
      return;
    }

    list.innerHTML = `<p class="t-small faint" style="margin-bottom:12px">Showing
      ${page.items.length} of ${page.total} cases.</p>
      <div class="cases">${page.items.map(caseCard).join('')}</div>`;
    animate($('#h-stats'));
    animate(list);

    $$('.case', list).forEach((card) =>
      card.addEventListener('click', (e) => {
        if (e.target.closest('[data-del]')) return;
        openCase(card.dataset.id);
      }));

    $$('[data-del]', list).forEach((btn) =>
      btn.addEventListener('click', async (e) => {
        e.stopPropagation();
        if (!window.confirm('Delete this case from the local history?')) return;
        try {
          await api(`/api/cases/${encodeURIComponent(btn.dataset.del)}`, { method: 'DELETE' });
          toast('Case deleted.', 'ok');
          loadHistory();
          refreshCaseCount();
        } catch (err) { toast(err.message, 'error'); }
      }));
  } catch (err) {
    list.innerHTML = `<div class="issue issue--critical">${icon('i-alert', 'issue__i')}
      <span>${esc(err.message)}</span></div>`;
  }
}

function caseCard(it) {
  // The list endpoint only exposes flat summary columns, not the full severity
  // object, but INDETERMINATE always means the classifier output was excluded
  // from grading (untrained weights, or non-skin input) - so it is a reliable
  // signal here for not printing a name+confidence that was never meaningful.
  const neuralUsable = it.tier !== 'INDETERMINATE';
  const nameLine = neuralUsable
    ? `<span class="case__n">${esc(it.top_name)}</span>
       <span class="case__m">${fx(it.confidence * 100, 1)}% · TDS ${isNum(it.tds) ? fx(it.tds, 2) : '—'}</span>`
    : `<span class="case__n">Indeterminate</span>
       <span class="case__m">No diagnosis — see case for details</span>`;
  return `<article class="case" data-id="${esc(it.id)}" tabindex="0" role="button">
    <div class="case__img">
      ${it.thumbnail ? `<img src="${esc(it.thumbnail)}" alt="Thumbnail for case ${esc(it.id)}" loading="lazy">` : ''}
      <span class="badge case__badge tier-${esc(it.tier)}">${esc(it.tier)}</span>
    </div>
    <div class="case__b">
      ${nameLine}
      <span class="case__m">${esc(when(it.created_at))}</span>
      <div class="case__f">
        <span class="t-small faint">Q ${fx(it.quality_score, 0)}</span>
        <button class="btn btn--ghost btn--sm btn--danger" data-del="${esc(it.id)}"
          type="button" aria-label="Delete case">${icon('i-trash')}</button>
      </div>
    </div>
  </article>`;
}

async function openCase(id) {
  try {
    const r = await api(`/api/cases/${encodeURIComponent(id)}`);
    const sev = r.severity || {};
    const p = r.prediction || {};
    const abcd = r.morphology?.abcd;

    $('#modal-title').textContent = `Case ${id}`;
    $('#modal-body').innerHTML = `
      ${verdictHtml(sev, false)}
      ${r.images?.original ? `<img src="${esc(r.images.original)}" alt="Case thumbnail"
        style="width:100%;max-height:300px;object-fit:contain;background:#05070a;
        border-radius:var(--r-md);margin:16px 0">` : '<div style="height:16px"></div>'}
      <div class="kv">
        ${kvRow('Prediction', sev.neural_usable === false
          ? `<span class="faint">Not shown — ${r.quality?.is_skin_like === false ? 'not a skin image' : 'untrained model'}</span>`
          : `${esc(p.name || '—')} <small>${fx(p.percentage, 1)}%</small>`)}
        ${kvRow('Severity score', `${fx(sev.score, 0)}/100`)}
        ${kvRow('Malignant probability', sev.neural_usable === false ? '<span class="faint">—</span>' : pc(sev.malignancy_probability, 1))}
        ${kvRow('ABCD TDS', abcd ? `${fx(abcd.tds, 2)} <small>${esc(tc(abcd.interpretation))}</small>` : '—')}
        ${kvRow('Image quality', `${fx(r.quality?.score, 0)}/100`)}
        ${kvRow('Analysed', esc(when(r.created_at)))}
        ${kvRow('Source file', esc(r.filename || '—'))}
      </div>
      <h4 class="t-cap" style="margin:18px 0 7px">Impression</h4>
      <p class="t-small">${esc(r.narrative?.impression || 'No narrative stored.')}</p>
      <h4 class="t-cap" style="margin:18px 0 7px">Clinician notes</h4>
      <textarea id="case-notes" rows="3" placeholder="Add a note…">${esc(r.notes || '')}</textarea>
      <div class="row row--end" style="margin-top:12px">
        <button class="btn btn--ghost btn--sm" id="case-pdf" type="button">${icon('i-pdf')}PDF</button>
        <button class="btn btn--primary btn--sm" id="case-save" type="button">Save note</button>
      </div>`;
    $('#modal').classList.remove('hidden');
    animate($('#modal-body'));

    $('#case-save').addEventListener('click', async () => {
      try {
        await api(`/api/cases/${encodeURIComponent(id)}`, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ notes: $('#case-notes').value }),
        });
        toast('Note saved.', 'ok');
      } catch (err) { toast(err.message, 'error'); }
    });

    $('#case-pdf').addEventListener('click', async () => {
      try {
        const blob = await api('/api/report/pdf', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ case_id: id }),
        });
        save(blob, `lesion-report-${id}.pdf`);
      } catch (err) { toast(err.message, 'error'); }
    });
  } catch (err) {
    toast(err.message, 'error');
  }
}

function initModal() {
  $$('[data-close]').forEach((n) =>
    n.addEventListener('click', () => $('#modal').classList.add('hidden')));
}

/* ══════════════════════════════════ METRICS ═══════════════════════════════ */

const FIG_CAPTIONS = {
  'class_distribution.png': 'HAM10000 class distribution. 67% of images are melanocytic nevi, which is why accuracy alone is misleading.',
  'sample_images_per_class.png': 'Representative images from each of the seven diagnostic classes.',
  'training_curves.png': 'Training and validation curves plus validation macro-F1.',
  'efficientnet_confusion_matrix.png': 'Test-set confusion matrix, raw counts and row-normalised.',
  'svm_confusion_matrix.png': 'SVM baseline confusion matrix for comparison.',
  'gradcam_results.png': 'Grad-CAM on correct and incorrect predictions.',
  'severity_grading_analysis.png': 'Severity tier distribution and the melanoma safety-net catch rate.',
  'calibration.png': 'Reliability diagram after temperature scaling.',
};

async function loadMetrics() {
  const out = $('#metrics-out');
  out.innerHTML = `<div class="grid g-2">${'<div class="skel" style="height:200px"></div>'.repeat(2)}</div>`;
  try {
    const d = await api('/api/metrics');
    out.innerHTML = [
      comparisonHtml(d.comparison),
      auditHtml(d.split_audit),
      evaluationHtml(d.evaluation),
      figuresHtml(d.figures),
    ].filter(Boolean).join('');
    animate(out);
  } catch (err) {
    out.innerHTML = `<div class="issue issue--critical">${icon('i-alert', 'issue__i')}
      <span>${esc(err.message)}</span></div>`;
  }
}

function comparisonHtml(cmp) {
  if (!cmp || !Object.keys(cmp).length) return '';
  // Underscore-prefixed keys are provenance metadata, not models.
  const models = Object.keys(cmp).filter((k) => !k.startsWith('_'));
  if (!models.length) return '';

  const unverified = models.filter((m) => cmp[m].verified === 'self_reported');
  const keys = [
    ['verified', 'Provenance'],
    ['accuracy', 'Accuracy %'], ['balanced_accuracy', 'Balanced accuracy %'],
    ['macro_f1', 'Macro F1'], ['melanoma_recall', 'Melanoma recall %'],
    ['vasc_recall', 'Vascular recall %'], ['df_recall', 'Dermatofibroma recall %'],
    ['roc_auc_macro', 'ROC-AUC macro'], ['ece', 'Calibration error'],
    ['melanoma_safety_net_catch_rate', 'Safety-net catch %'],
    ['grouped_by_lesion', 'Leak-free split'],
  ];

  const rows = keys.filter(([k]) => models.some((m) => cmp[m][k] !== undefined))
    .map(([k, label]) => {
      const cells = models.map((m) => {
        const v = cmp[m][k];
        if (v === undefined || v === null) return '<td class="r faint">—</td>';
        if (k === 'verified') {
          const measured = v === 'measured';
          return `<td class="r"><span class="badge tier-${measured ? 'LOW' : 'MODERATE'}">${
            measured ? 'measured' : 'self-reported'}</span></td>`;
        }
        if (typeof v === 'boolean') {
          return `<td class="r">${v
            ? '<span style="color:var(--low)">yes</span>'
            : '<span style="color:var(--mod)">no</span>'}</td>`;
        }
        return `<td class="r">${typeof v === 'number' ? fx(v, v < 5 ? 4 : 2) : esc(v)}</td>`;
      }).join('');
      return `<tr><td>${esc(label)}</td>${cells}</tr>`;
    }).join('');

  const warn = unverified.length
    ? `<div class="notice notice--warn" style="margin:0 16px 16px">
        ${icon('i-alert', 'notice__icon')}
        <div class="notice__body"><b>${unverified.length} figure set${
          unverified.length === 1 ? ' is' : 's are'} self-reported.</b>
        <p>${esc(unverified.map(tc).join(' and '))} ${unverified.length === 1 ? 'was' : 'were'}
        transcribed from a notebook run and cannot be reproduced from this repository. They were
        measured on an image-wise split — see the leakage audit below. Regenerate verified numbers
        with <code>python -m derm.evaluate</code>.</p></div>
      </div>`
    : '';

  return `<div class="card" style="margin-bottom:16px">
    <div class="card__head"><h3>Model comparison</h3>
      <span class="t-cap">docs/model_comparison.json</span></div>
    <div class="card__body card__body--flush"><div class="tw"><table class="tbl">
      <thead><tr><th>Metric</th>${models.map((m) => `<th class="r">${esc(tc(m))}</th>`).join('')}</tr></thead>
      <tbody>${rows}</tbody></table></div></div>
    ${warn}
  </div>`;
}

/** Reproducible leakage audit from scripts/audit_leakage.py. */
function auditHtml(audit) {
  if (!audit) return '';
  const ds = audit.dataset || {};
  const iw = audit.image_wise_split?.test || {};
  const gr = audit.lesion_grouped_split?.test || {};

  const byClass = Object.entries(iw.by_class || {}).map(([code, r]) => {
    const cls = S.classes.find((c) => c.code === code);
    const severe = r.leaked_pct >= 50;
    return `<tr>
      <td><code>${esc(code)}</code> ${esc(cls?.short_name || '')}</td>
      <td class="r">${r.images}</td>
      <td class="r">${r.leaked}</td>
      <td class="r"${severe ? ' style="color:var(--high);font-weight:650"' : ''}>${fx(r.leaked_pct, 1)}%</td>
    </tr>`;
  }).join('');

  return `<div class="card" style="margin-bottom:16px">
    <div class="card__head"><h3>Data leakage audit</h3>
      <span class="t-cap">reproducible · metadata only</span></div>
    <div class="card__body">
      <p class="t-body" style="margin-bottom:14px">HAM10000 holds
      <strong>${ds.images?.toLocaleString()}</strong> images of only
      <strong>${ds.lesions?.toLocaleString()}</strong> distinct lesions —
      ${ds.repeat_images?.toLocaleString()} images (${fx(ds.repeat_images_pct, 1)}%) are repeat
      photographs of a lesion that appears elsewhere. Splitting on images therefore puts
      near-duplicate photographs of the same physical lesion in both train and test.</p>

      <div class="grid g-3" style="margin-bottom:16px">
        ${kpi(`${fx(iw.leaked_pct, 1)}%`, 'Image-wise split leakage', 'high',
              `${iw.leaked_images}/${iw.images} test images`)}
        ${kpi(`${fx(gr.leaked_pct, 1)}%`, 'Lesion-grouped leakage', 'low',
              `${gr.leaked_images}/${gr.images} test images`)}
        ${kpi(ds.max_images_per_lesion ?? '—', 'Max images of one lesion')}
      </div>

      <span class="t-cap" style="display:block;margin-bottom:8px">
        Per-class leakage under the image-wise split</span>
      <div class="tw"><table class="tbl">
        <thead><tr><th>Class</th><th class="r">Test images</th>
        <th class="r">Leaked</th><th class="r">Leaked %</th></tr></thead>
        <tbody>${byClass}</tbody></table></div>
    </div>
    <div class="card__foot">Leakage is worst exactly where it matters most: the clinically
      critical classes are the ones most often re-photographed. Any metric measured on this split
      overstates real-world performance.</div>
  </div>`;
}

function evaluationHtml(ev) {
  if (!ev) {
    return `<div class="card" style="margin-bottom:16px">
      <div class="card__head"><h3>Detailed evaluation</h3></div>
      <div class="card__body"><p class="t-body">Not available yet. Run
        <code>python -m derm.evaluate</code> against a trained checkpoint and the HAM10000 dataset to
        generate per-class metrics, calibration analysis and the melanoma safety-net audit.</p></div>
    </div>`;
  }

  const perClass = (ev.per_class || []).map((r) => `<tr>
    <td><code>${esc(r.code)}</code> ${esc(r.name)}</td>
    <td class="r">${fx(r.precision, 3)}</td><td class="r">${fx(r.recall, 3)}</td>
    <td class="r">${fx(r.specificity, 3)}</td><td class="r">${fx(r.f1, 3)}</td>
    <td class="r">${r.support}</td></tr>`).join('');

  const net = ev.safety_net || {};

  return `<div class="grid g-4" style="margin-bottom:16px">
      ${kpi(pc(ev.accuracy, 2), 'Test accuracy')}
      ${kpi(pc(ev.balanced_accuracy, 2), 'Balanced accuracy')}
      ${kpi(fx(ev.macro_f1, 4), 'Macro F1')}
      ${kpi(fx(ev.calibration?.ece_calibrated, 4), 'Calibration error')}
    </div>
    <div class="card" style="margin-bottom:16px">
      <div class="card__head"><h3>Per-class performance</h3>
        <span class="t-cap">${ev.test_images} images · ${ev.test_lesions} lesions ·
        ${ev.grouped_by_lesion ? 'lesion-grouped' : 'image-wise'} split</span></div>
      <div class="card__body card__body--flush"><div class="tw"><table class="tbl">
        <thead><tr><th>Class</th><th class="r">Precision</th><th class="r">Recall</th>
        <th class="r">Specificity</th><th class="r">F1</th><th class="r">n</th></tr></thead>
        <tbody>${perClass}</tbody></table></div></div>
    </div>
    <div class="card" style="margin-bottom:16px">
      <div class="card__head"><h3>Melanoma safety net</h3>
        <span class="t-cap">does grading escalate true melanomas?</span></div>
      <div class="card__body">
        <div class="grid g-3">
          ${kpi(`${net.melanoma_escalated ?? 0}/${net.true_melanoma ?? 0}`, 'Melanomas escalated', 'low')}
          ${kpi(pc(net.melanoma_catch_rate, 1), 'Catch rate', 'low')}
          ${kpi(pc(net.benign_escalation_rate, 1), 'Benign over-referral', 'mod')}
        </div>
      </div>
      <div class="card__foot">The catch rate is the metric that matters for triage. Argmax
        classification under-calls melanoma on a dataset that is 67% nevi; the grading overrides
        recover those cases, at the cost of some benign over-referral.</div>
    </div>`;
}

function figuresHtml(figures) {
  if (!figures?.length) return '';
  return `<div class="card">
    <div class="card__head"><h3>Figures</h3><span class="t-cap">docs/</span></div>
    <div class="card__body"><div class="grid g-2">${figures.map((n) => `<figure class="fig">
      <img src="/api/figures/${encodeURIComponent(n)}" alt="${esc(n)}" loading="lazy">
      <figcaption><b>${esc(n)}</b><div style="margin-top:4px">${esc(FIG_CAPTIONS[n] || '')}</div></figcaption>
    </figure>`).join('')}</div></div>
  </div>`;
}
