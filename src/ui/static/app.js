"use strict";
const $ = (id) => document.getElementById(id);
let currentJob = null, ws = null, timerHandle = null, startedAt = 0, lastTotal = 0;
// Resilient watch state: a single WS never survives 20+ min OCR gaps, so
// reconnect on close (server replays missed events) + poll REST as fallback.
let watchSeq = 0, pollHandle = null, reconnectHandle = null, jobDone = false;
const seenEvents = new Set();
const TERMINAL_STATUSES = ["completed", "failed", "cancelled", "paused"];

function show(name) {
  for (const v of ["upload", "progress", "done", "history", "health"]) {
    $("view-" + v).hidden = v !== name;
  }
  if (name === "history") loadHistory();
  if (name === "health") loadHealth();
}
document.querySelectorAll("nav button").forEach((b) => b.addEventListener("click", () => show(b.dataset.view)));

// --- status line: the heartbeat you stare at ---
function setActivity(text, state) {
  $("activity").textContent = text;
  $("statusline").className = state || "";
}
function tickElapsed() {
  const secs = Math.floor((Date.now() - startedAt) / 1000);
  $("elapsed").textContent = `elapsed ${Math.floor(secs / 60)}:${String(secs % 60).padStart(2, "0")}`;
}
function startClock() {
  startedAt = Date.now();
  tickElapsed();
  if (timerHandle) clearInterval(timerHandle);
  timerHandle = setInterval(tickElapsed, 1000);
}
function stopClock() {
  if (timerHandle) clearInterval(timerHandle);
  timerHandle = null;
  tickElapsed();
}

function logLine(text, cls) {
  const el = $("log");
  const div = document.createElement("div");
  if (cls) div.className = cls;
  div.textContent = text;
  el.appendChild(div);
  el.scrollTop = el.scrollHeight;
}

// Minimal Markdown preview (headings, bold, code, tables skipped as text, images).
function renderPreview(md) {
  const esc = md.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const html = esc
    .split("\n")
    .map((line) => {
      const m = line.match(/^(#{1,6})\s+(.*)$/);
      if (m) return `<h${m[1].length}>${m[2]}</h${m[1].length}>`;
      const img = line.match(/^!\[([^\]]*)\]\(([^)]+)\)/);
      if (img) return `<p><em>[image: ${img[1]} — ${img[2]}]</em></p>`;
      if (/^<!--FIG:/.test(line)) return `<p><em>[figure pending: ${line}]</em></p>`;
      if (/^\|/.test(line)) return `<p><code>${line}</code></p>`;
      if (/^\s*$/.test(line)) return "";
      return `<p>${line}</p>`;
    })
    .join("\n");
  $("preview").innerHTML = html;
}

// --- output dir: remembered, never blank twice ---
const DIR_KEY = "pdf2md.output_dir";
async function defaultOutputDir() {
  const saved = localStorage.getItem(DIR_KEY);
  if (saved) return saved;
  try {
    const jobs = await (await fetch("/api/v1/jobs")).json();
    if (jobs.length && jobs[0].output_dir) return jobs[0].output_dir;
  } catch (e) { /* offline on first paint; ignore */ }
  return "";
}
defaultOutputDir().then((d) => { if (d && !$("output_dir").value) $("output_dir").value = d; });

// --- upload (7.3) ---
const drop = $("dropzone");
["dragover", "dragenter"].forEach((e) => drop.addEventListener(e, (ev) => { ev.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((e) => drop.addEventListener(e, (ev) => { ev.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (ev) => { if (ev.dataTransfer.files.length) $("file").files = ev.dataTransfer.files; });

$("output_dir").addEventListener("input", () => {
  const v = $("output_dir").value.trim();
  const ok = /^[A-Za-z]:\\/.test(v) || v.startsWith("\\\\") || v.startsWith("/");
  $("output_hint").textContent = v === "" ? "" : ok ? "Absolute path — OK (remembered for next run)" : "Must be an absolute path";
});

// --- server-side folder picker (browsers expose no native dir picker) ---
let pickerPath = "";
async function pickerLoad(path) {
  const resp = await fetch("/api/v1/browse?path=" + encodeURIComponent(path || ""));
  const data = await resp.json();
  pickerPath = data.path || "";
  $("picker-path").textContent = data.path || "Drives";
  const ul = $("picker-list");
  ul.innerHTML = "";
  $("picker-up").disabled = !data.parent;
  $("picker-up").onclick = () => pickerLoad(data.parent);
  for (const entry of data.entries) {
    const li = document.createElement("li");
    li.textContent = "📁 " + entry.name;
    li.addEventListener("click", () => pickerLoad(entry.path));
    ul.appendChild(li);
  }
}
$("browse").addEventListener("click", () => {
  pickerLoad($("output_dir").value.trim());
  $("picker").showModal();
});
$("picker-cancel").addEventListener("click", () => $("picker").close());
$("picker-select").addEventListener("click", () => {
  if (pickerPath) {
    $("output_dir").value = pickerPath;
    localStorage.setItem(DIR_KEY, pickerPath);
    $("output_hint").textContent = "Absolute path — OK (remembered for next run)";
  }
  $("picker").close();
});

$("start").addEventListener("click", async () => {
  $("upload_error").textContent = "";
  const file = $("file").files[0];
  if (!file) { $("upload_error").textContent = "Choose a PDF first."; return; }
  const outDir = $("output_dir").value.trim();
  if (!outDir) { $("upload_error").textContent = "Set an output directory first."; return; }
  localStorage.setItem(DIR_KEY, outDir);
  const form = new FormData();
  form.append("file", file);
  form.append("output_dir", outDir);
  form.append("options", JSON.stringify({
    render_dpi: parseInt($("opt_dpi").value, 10),
    coverage_threshold: parseInt($("opt_cov").value, 10),
    toc_enabled: $("opt_toc").checked,
    fig_details: $("opt_fig").checked,
    diagram_to_mermaid: $("opt_diagram").checked,
    native_text_first: $("opt_native").checked,
    ocr_enabled: $("opt_ocr").checked,
  }));
  const resp = await fetch("/api/v1/jobs", { method: "POST", body: form });
  if (resp.status === 409) { $("upload_error").textContent = "A job is already running — wait for it to finish."; return; }
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }));
    $("upload_error").textContent = err.detail || err.title || "Upload failed.";
    return;
  }
  const { job_id } = await resp.json();
  watchJob(job_id);
});

// --- progress (7.4) ---
function ensureGrid(total) {
  const grid = $("grid");
  if (grid.children.length === total && total > 0) return;
  grid.innerHTML = "";
  for (let p = 1; p <= total; p++) {
    const div = document.createElement("div");
    div.className = "cell pending";
    div.id = "cell-" + p;
    div.textContent = "#" + p + " pending";
    grid.appendChild(div);
  }
}

function paintCell(page, status, coverage) {
  const div = $("cell-" + page);
  if (!div) return;
  div.className = "cell " + status;
  div.textContent = `#${page} ${status}` + (coverage != null ? ` ${coverage}` : "");
}

const STAGE_HINTS = {
  preflight: "validating the PDF…",
  rendering: "rendering pages…",
  ocr: "local OCR starting (slow pages are normal — model runs partly on CPU)…",
  transcribing: "agent transcribing + verifying, page by page…",
  assembling: "resolving figures and assembling the document…",
  qa: "final QA pass…",
  completed: "done.",
};

function updateControls(live, status) {
  const running = !!live && !["completed", "failed", "cancelled"].includes(status);
  $("btn-pause").hidden = !running;
  $("btn-resume").hidden = !running;
  $("btn-cancel").hidden = !running;
  $("btn-restart").hidden = running || ["completed"].includes(status);
}

async function watchJob(job_id) {
  currentJob = job_id;
  lastTotal = 0;
  const seq = ++watchSeq;
  jobDone = false;
  seenEvents.clear();
  if (pollHandle) { clearInterval(pollHandle); pollHandle = null; }
  if (reconnectHandle) { clearTimeout(reconnectHandle); reconnectHandle = null; }
  show("progress");
  $("grid").innerHTML = "";
  $("log").innerHTML = "";
  $("preview").innerHTML = "";
  $("counter").textContent = "";
  $("pct").textContent = "";
  let hinted = false;
  try {
    const detail = await (await fetch(`/api/v1/jobs/${job_id}`)).json();
    lastTotal = detail.page_count || 0; // Bug-4 fix: denominator is the job total, never the rows seen so far
    updateControls(detail.live, detail.status);
    if (!detail.live && detail.status === "paused") {
      setActivity(`paused at page ${detail.pages_done}/${detail.page_count} — press Restart to resume from checkpoint.`, "paused");
      hinted = true;
    } else if (!detail.live && ["failed", "cancelled"].includes(detail.status)) {
      setActivity(`${detail.status} — press Restart to retry from checkpoint.`, "paused");
      hinted = true;
    } else if (!detail.live && detail.status !== "completed") {
      setActivity(`${detail.status}, not currently running — press Restart to resume from checkpoint.`, "paused");
      hinted = true;
    }
  } catch (e) { /* progress still streams; bar fills in from events */ }
  if (!hinted) setActivity("connecting…", "");
  startClock();
  if (ws) { const old = ws; ws = null; try { old.close(); } catch (e) { /* already gone */ } }

  function stopWatchTimers() {
    if (pollHandle) { clearInterval(pollHandle); pollHandle = null; }
    if (reconnectHandle) { clearTimeout(reconnectHandle); reconnectHandle = null; }
  }

  async function handleEvent(e) {
    if (!e || e.event === "ping") return; // server keepalive for long OCR gaps
    // Reconnects replay the full buffer: skip duplicates so the log does
    // not double every history entry (page_update repaint is idempotent).
    let key = null;
    try { key = JSON.stringify(e); } catch (err) { key = null; }
    if (key !== null) {
      if (seenEvents.has(key)) return;
      if (seenEvents.size > 2000) seenEvents.clear();
      seenEvents.add(key);
    }
    if (e.event === "stage_changed") {
      setActivity(`stage: ${e.stage} — ${STAGE_HINTS[e.stage] || ""}`, "");
      logLine(`[stage] ${e.stage}`);
    } else if (e.event === "page_update") {
      lastTotal = e.total;
      ensureGrid(e.total);
      paintCell(e.page, e.status, e.coverage);
      $("counter").textContent = `page ${e.page}/${e.total}`;
      if (e.status === "processing") setActivity(`transcribing page ${e.page}/${e.total} (render → OCR → agent → verify)…`, "");
      else if (e.status === "verified") setActivity(`page ${e.page}/${e.total} verified (coverage ${e.coverage})`, "");
      else if (e.status === "needs_review") setActivity(`page ${e.page}/${e.total} needs review (coverage ${e.coverage})`, "paused");
      await refreshGrid();
    } else if (e.event === "page_markdown") {
      logLine(`page ${e.page} checkpointed`);
    } else if (e.event === "log") {
      logLine(e.message, e.level === "warning" ? "warn" : "");
    } else if (e.event === "job_finished") {
      finishJob(job_id, e);
    }
  }

  // REST fallback: if the stream dies mid-job (idle timeout, laptop sleep,
  // server restart), polling still advances the grid and finishes the UI
  // from the DB state once the driver completes.
  async function pollOnce() {
    if (seq !== watchSeq || jobDone || currentJob !== job_id) return;
    let detail = null;
    try {
      detail = await (await fetch(`/api/v1/jobs/${job_id}`)).json();
    } catch (e) { return; }
    if (seq !== watchSeq || jobDone) return;
    if (detail && detail.page_count) {
      lastTotal = detail.page_count;
      ensureGrid(detail.page_count);
    }
    try { await refreshGrid(); } catch (e) { /* next tick retries */ }
    updateControls(detail.live, detail.status);
    if (detail && TERMINAL_STATUSES.includes(detail.status) && !detail.live) {
      finishJob(job_id, {
        event: "job_finished",
        job_id: job_id,
        status: detail.status,
        error: (detail.error && (detail.error.error || JSON.stringify(detail.error))) || detail.status,
        output_path: "",
      });
    }
  }

  function connect() {
    if (seq !== watchSeq || jobDone) return;
    let socket = null;
    try {
      socket = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws/jobs/" + job_id);
    } catch (e) {
      scheduleReconnect();
      return;
    }
    ws = socket;
    socket.onmessage = async (msg) => {
      if (seq !== watchSeq) return;
      let e = null;
      try { e = JSON.parse(msg.data); } catch (err) { return; }
      await handleEvent(e);
    };
    socket.onclose = () => {
      if (seq !== watchSeq || jobDone || currentJob !== job_id) return;
      if (ws === socket) ws = null;
      logLine("(stream closed — reconnecting…)");
      scheduleReconnect();
    };
    socket.onerror = () => {
      try { socket.close(); } catch (e) { /* onclose reschedules */ }
    };
  }

  function scheduleReconnect() {
    if (seq !== watchSeq || jobDone || currentJob !== job_id) return;
    if (reconnectHandle) return; // one pending attempt at a time
    reconnectHandle = setTimeout(() => {
      reconnectHandle = null;
      // Poll immediately so a completed-while-away job finishes even if
      // the socket is still down; then re-open the stream for replay.
      pollOnce();
      connect();
    }, 3000);
  }

  connect();
  pollHandle = setInterval(pollOnce, 10000);
}

async function refreshGrid() {
  if (!currentJob) return;
  const pages = await (await fetch(`/api/v1/jobs/${currentJob}/pages`)).json();
  if (pages.length && !lastTotal) { lastTotal = pages.length; ensureGrid(pages.length); }
  for (const p of pages) paintCell(p.page, p.status, p.coverage);
  const preview = await (await fetch(`/api/v1/jobs/${currentJob}/preview`)).json();
  renderPreview(preview.markdown || "");
  const done = pages.filter((p) => p.status === "verified" || p.status === "needs_review").length;
  const total = lastTotal || pages.length;
  const pct = total ? Math.round((done / total) * 100) : 0;
  $("bar").value = pct;
  $("pct").textContent = total ? `${pct}% (${done}/${total} pages)` : "";
}

async function control(action) {
  if (!currentJob) return;
  const resp = await fetch(`/api/v1/jobs/${currentJob}/${action}`, { method: "POST" });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({}));
    logLine(`control ${action} rejected: ${err.detail || err.title || resp.status}`, "err");
    return;
  }
  if (action === "pause") setActivity("pausing after the current page…", "paused");
  if (action === "resume") setActivity("resuming…", "");
  if (action === "cancel") setActivity("cancelling after the current page…", "paused");
}
$("btn-pause").addEventListener("click", () => control("pause"));
$("btn-resume").addEventListener("click", () => control("resume"));
$("btn-cancel").addEventListener("click", () => control("cancel"));
$("btn-restart").addEventListener("click", async () => {
  if (!currentJob) return;
  $("btn-restart").hidden = true;
  const resp = await fetch(`/api/v1/jobs/${currentJob}/restart`, { method: "POST" });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({}));
    logLine(`restart rejected: ${err.detail || err.title || resp.status}`, "err");
    return;
  }
  logLine("restarted from last checkpoint");
  watchJob(currentJob);
});

// --- completion (7.5) ---
async function finishJob(job_id, event) {
  if (currentJob !== job_id) return;
  jobDone = true;
  if (pollHandle) { clearInterval(pollHandle); pollHandle = null; }
  if (reconnectHandle) { clearTimeout(reconnectHandle); reconnectHandle = null; }
  if (ws) { const s = ws; ws = null; try { s.close(); } catch (e) { /* already gone */ } }
  stopClock();
  if (event.status === "paused") {
    setActivity("paused — press Resume (live) or Restart (from checkpoint).", "paused");
    $("btn-restart").hidden = false;
    return;
  }
  if (event.status === "cancelled") { setActivity("cancelled — partial work is resumable.", "paused"); return; }
  if (event.status !== "completed") {
    setActivity(`FAILED: ${event.error || event.status}`, "failed");
    logLine(`job failed: ${event.error || event.status}`, "err");
    return;
  }
  setActivity("completed.", "");
  show("done");
  $("done-path").textContent = "Output: " + (event.output_path || "(see report)");
  $("dl-error").textContent = "";
  wireDownload($("dl-md"), `/api/v1/jobs/${job_id}/artifact/document`, `job-${job_id}.md`);
  wireDownload($("dl-report"), `/api/v1/jobs/${job_id}/artifact/report`, `job-${job_id}-report.md`);
  const report = await (await fetch(`/api/v1/jobs/${job_id}/report`)).json();
  $("done-summary").textContent = (report.report || "").slice(0, 4000);
  const pages = await (await fetch(`/api/v1/jobs/${job_id}/pages`)).json();
  const list = $("review-list");
  list.innerHTML = "";
  for (const p of pages.filter((p) => p.needs_review)) {
    const div = document.createElement("div");
    div.textContent = `Page ${p.page} — coverage ${p.coverage}`;
    list.appendChild(div);
  }
  if (!list.children.length) list.textContent = "No pages need review.";
}

// Downloads stay in-page: failures surface inline, never strand the user
// on a raw JSON error page (bugs 2/3 class).
async function wireDownload(anchor, url, filename) {
  anchor.onclick = async (ev) => {
    ev.preventDefault();
    $("dl-error").textContent = "";
    const resp = await fetch(url);
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      $("dl-error").textContent = `Download failed: ${err.detail || err.title || resp.status}`;
      return;
    }
    const blob = await resp.blob();
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = filename;
    link.click();
    setTimeout(() => URL.revokeObjectURL(link.href), 5000);
  };
}

// --- history + health (7.6) ---
async function loadHistory() {
  const jobs = await (await fetch("/api/v1/jobs")).json();
  const ul = $("history-list");
  ul.innerHTML = "";
  for (const j of jobs) {
    const li = document.createElement("li");
    const a = document.createElement("a");
    a.href = "#";
    a.textContent = `${j.filename} — ${j.status}`;
    a.addEventListener("click", (ev) => { ev.preventDefault(); watchJob(j.job_id); });
    li.appendChild(a);
    if (["paused", "failed", "cancelled"].includes(j.status)) {
      const r = document.createElement("a");
      r.href = "#";
      r.textContent = " — restart from checkpoint";
      r.addEventListener("click", async (ev) => {
        ev.preventDefault();
        const resp = await fetch(`/api/v1/jobs/${j.job_id}/restart`, { method: "POST" });
        if (!resp.ok) {
          const err = await resp.json().catch(() => ({}));
          alert(`Restart rejected: ${err.detail || err.title || resp.status}`);
          return;
        }
        watchJob(j.job_id);
      });
      li.appendChild(r);
    }
    ul.appendChild(li);
  }
}

async function loadHealth() {
  const health = await (await fetch("/api/v1/health")).json();
  const ul = $("health-list");
  ul.innerHTML = "";
  const head = document.createElement("li");
  head.textContent = "Overall: " + health.status;
  ul.appendChild(head);
  for (const c of health.components) {
    const li = document.createElement("li");
    li.textContent = `${c.ok ? "OK" : "FAIL"} ${c.name} — ${c.message}`;
    ul.appendChild(li);
  }
}
