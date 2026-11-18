"use strict";

/* DocuMind frontend – plain JavaScript, no build step, no external requests. */

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

/* ---------------------------------------------------------------- state */
function getWorkspaceId() {
  let id = localStorage.getItem("documind_ws");
  if (!id || !/^[A-Za-z0-9_-]{8,64}$/.test(id)) {
    const raw = window.crypto && crypto.randomUUID ? crypto.randomUUID() : Date.now().toString(16) + Math.random().toString(16).slice(2);
    id = raw.replace(/-/g, "");
    localStorage.setItem("documind_ws", id);
  }
  return id;
}

const state = { ws: getWorkspaceId(), docs: [], history: [], busy: false, abort: null, config: null, evalTimer: null };
const els = {
  messages: $("#messages"), empty: $("#emptyState"), form: $("#composer"), question: $("#question"),
  send: $("#sendBtn"), stop: $("#stopBtn"), newChat: $("#newChatBtn"),
  docList: $("#docList"), docEmpty: $("#docEmpty"), fileInput: $("#fileInput"), dropzone: $("#dropzone"),
  uploadStatus: $("#uploadStatus"), mode: $("#modeSelect"), topK: $("#topK"), topKOut: $("#topKOut"),
  sidebar: $("#sidebar"), scrim: $("#scrim"), menuBtn: $("#menuBtn"),
};

/* ---------------------------------------------------------------- helpers */
async function api(path, opts = {}) {
  const res = await fetch(path, { ...opts, headers: { "X-Workspace-Id": state.ws, ...(opts.headers || {}) } });
  if (!res.ok) {
    let msg = res.statusText || "Request failed";
    try {
      const body = await res.json();
      msg = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch (_) { /* ignore */ }
    throw new Error(msg);
  }
  return res.json();
}

function toast(message, kind = "info") {
  const el = document.createElement("div");
  el.className = "toast" + (kind === "error" ? " error" : "");
  el.textContent = message;
  $("#toasts").appendChild(el);
  setTimeout(() => el.remove(), 5000);
}

/* ---------------------------------------------------------------- tiny markdown renderer (XSS-safe) */
function inline(text) {
  let s = esc(text);
  s = s.replace(/`([^`]+)`/g, "<code>$1</code>");
  s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  s = s.replace(/(^|[^*\w])_([^_]+)_(?!\w)/g, "$1<em>$2</em>");
  s = s.replace(/\[(\d{1,2})\]/g, '<button type="button" class="cite" data-n="$1" aria-label="Open source $1">$1</button>');
  return s;
}

function renderMarkdown(src) {
  const blocks = [];
  const text = src.replace(/```[\w-]*\n?([\s\S]*?)```/g, (_, code) => {
    blocks.push(`<pre><code>${esc(code.replace(/\n$/, ""))}</code></pre>`);
    return `\u0000${blocks.length - 1}\u0000`;
  });
  let html = "", list = null, para = [];
  const closeList = () => { if (list) { html += `</${list}>`; list = null; } };
  const flush = () => { if (para.length) { html += `<p>${inline(para.join(" "))}</p>`; para = []; } };
  const openList = (tag) => { flush(); if (list !== tag) { closeList(); html += `<${tag}>`; list = tag; } };

  for (const raw of text.split("\n")) {
    const line = raw.trimEnd();
    let m;
    if (!line.trim()) { flush(); closeList(); }
    else if ((m = line.match(/^\u0000(\d+)\u0000$/))) { flush(); closeList(); html += blocks[+m[1]]; }
    else if ((m = line.match(/^\s*[-*•]\s+(.*)$/))) { openList("ul"); html += `<li>${inline(m[1])}</li>`; }
    else if ((m = line.match(/^\s*\d+[.)]\s+(.*)$/))) { openList("ol"); html += `<li>${inline(m[1])}</li>`; }
    else if ((m = line.match(/^#{1,6}\s+(.*)$/))) { flush(); closeList(); html += `<h4>${inline(m[1])}</h4>`; }
    else { closeList(); para.push(line.trim()); }
  }
  flush(); closeList();
  return html;
}

/* ---------------------------------------------------------------- config / theme */
async function loadConfig() {
  try {
    state.config = await api("/api/config");
    const { llm, reranker, limits, defaults } = state.config;
    const llmChip = $("#llmChip");
    llmChip.textContent = llm.available ? `LLM: ${llm.model}` : "LLM: not set (extractive mode)";
    llmChip.className = "chip " + (llm.available ? "on" : "off");
    const rc = $("#rerankChip");
    rc.textContent = reranker ? "Reranker: on" : "Reranker: off";
    rc.className = "chip " + (reranker ? "on" : "off");
    $("#maxMb").textContent = limits.max_upload_mb;
    els.topK.value = defaults.top_k; els.topKOut.textContent = defaults.top_k;
    if (!reranker) { $('option[value="hybrid_rerank"]', els.mode).remove(); els.mode.value = "hybrid"; }
  } catch (e) { toast("Could not reach the server: " + e.message, "error"); }
}

$("#themeBtn").addEventListener("click", () => {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  localStorage.setItem("documind_theme", next);
});

/* ---------------------------------------------------------------- sidebar / tabs */
function setSidebar(open) {
  els.sidebar.classList.toggle("open", open);
  els.scrim.hidden = !open;
  els.menuBtn.setAttribute("aria-expanded", String(open));
}
els.menuBtn.addEventListener("click", () => setSidebar(!els.sidebar.classList.contains("open")));
els.scrim.addEventListener("click", () => setSidebar(false));

$$(".tab").forEach((btn) => btn.addEventListener("click", () => {
  $$(".tab").forEach((b) => { const on = b === btn; b.classList.toggle("active", on); b.setAttribute("aria-selected", String(on)); });
  $$(".tabpanel").forEach((p) => { const on = p.id === "tab-" + btn.dataset.tab; p.classList.toggle("active", on); p.hidden = !on; });
}));

els.topK.addEventListener("input", () => { els.topKOut.textContent = els.topK.value; });

/* ---------------------------------------------------------------- documents */
function renderDocs() {
  els.docList.innerHTML = state.docs.map((d) => `
    <li class="doc">
      <span class="doc-name" title="${esc(d.name)}">${esc(d.name)}</span>
      <button class="icon-btn" data-del="${esc(d.id)}" aria-label="Delete ${esc(d.name)}" title="Delete">
        <svg width="16" height="16" viewBox="0 0 20 20" aria-hidden="true"><path d="M5 5l10 10M15 5L5 15" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>
      </button>
      <span class="doc-meta">${d.pages} ${d.pages === 1 ? "page" : "pages"} · ${d.chunks} passages</span>
    </li>`).join("");
  els.docEmpty.hidden = state.docs.length > 0;
}

async function loadDocs() {
  try { state.docs = (await api("/api/documents")).documents; renderDocs(); }
  catch (e) { toast(e.message, "error"); }
}

async function uploadFiles(fileList) {
  const files = [...fileList];
  if (!files.length) return;
  const form = new FormData();
  files.forEach((f) => form.append("files", f));
  els.uploadStatus.textContent = `Reading and indexing ${files.length} file${files.length > 1 ? "s" : ""}… the first upload can take a little longer while models load.`;
  try {
    const data = await api("/api/documents", { method: "POST", body: form });
    state.docs = data.documents; renderDocs();
    const ok = data.results.filter((r) => r.status === "indexed").length;
    data.results.filter((r) => r.status !== "indexed").forEach((r) => toast(`${r.name}: ${r.message}`, "error"));
    els.uploadStatus.textContent = ok ? `Added ${ok} document${ok > 1 ? "s" : ""}.` : "";
  } catch (e) { els.uploadStatus.textContent = ""; toast(e.message, "error"); }
  els.fileInput.value = "";
}

els.fileInput.addEventListener("change", () => uploadFiles(els.fileInput.files));
["dragenter", "dragover"].forEach((ev) => els.dropzone.addEventListener(ev, (e) => { e.preventDefault(); els.dropzone.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => els.dropzone.addEventListener(ev, (e) => { e.preventDefault(); els.dropzone.classList.remove("over"); }));
els.dropzone.addEventListener("drop", (e) => uploadFiles(e.dataTransfer.files));

els.docList.addEventListener("click", async (e) => {
  const btn = e.target.closest("[data-del]");
  if (!btn) return;
  try { state.docs = (await api(`/api/documents/${btn.dataset.del}`, { method: "DELETE" })).documents; renderDocs(); }
  catch (err) { toast(err.message, "error"); }
});

$("#clearBtn").addEventListener("click", async () => {
  if (!state.docs.length || !confirm("Delete all documents in this workspace?")) return;
  try { await api("/api/documents", { method: "DELETE" }); state.docs = []; renderDocs(); newChat(); toast("All documents deleted."); }
  catch (err) { toast(err.message, "error"); }
});

/* ---------------------------------------------------------------- chat */
let msgCounter = 0;

function addMessage(role) {
  els.empty.hidden = true;
  const el = document.createElement("article");
  el.className = `msg ${role}`;
  el.dataset.id = ++msgCounter;
  el.innerHTML = `<div class="msg-role">${role === "user" ? "You" : "DocuMind"}</div><div class="content"></div>`;
  els.messages.appendChild(el);
  els.messages.scrollTop = els.messages.scrollHeight;
  return el;
}

function renderSources(msgEl, sources, cited) {
  $(".sources", msgEl)?.remove();
  if (!sources.length) return;
  const det = document.createElement("details");
  det.className = "sources";
  const items = sources.map((s) => {
    const sc = s.scores;
    const score = sc.rerank != null ? `relevance ${(sc.rerank * 100).toFixed(0)}%` : sc.dense != null ? `similarity ${sc.dense.toFixed(2)}` : sc.bm25 != null ? `keyword score ${sc.bm25.toFixed(1)}` : "";
    return `<li class="src ${cited.includes(s.n) ? "cited" : ""}" data-src="${s.n}">
      <div class="src-head"><span class="n">${s.n}</span><span class="file">${esc(s.doc_name)}</span><span>page ${s.page}</span><span class="score">${score}</span></div>
      <p class="src-text">${esc(s.text)}</p></li>`;
  }).join("");
  det.innerHTML = `<summary>${sources.length} source passage${sources.length > 1 ? "s" : ""}</summary><ol class="src-list">${items}</ol>`;
  msgEl.appendChild(det);
}

function renderMeta(msgEl, done) {
  const t = done.timings_ms || {};
  const bits = [];
  if (done.model) bits.push(done.model);
  if (done.mode) bits.push({ hybrid_rerank: "hybrid + rerank", hybrid: "hybrid", dense: "vectors", bm25: "BM25" }[done.mode] || done.mode);
  if (t.total) bits.push(`${(t.total / 1000).toFixed(1)} s`);
  const meta = document.createElement("div");
  meta.className = "meta";
  meta.textContent = bits.join(" · ");
  msgEl.appendChild(meta);
  if (!done.grounded) {
    const b = document.createElement("span");
    b.className = "badge warn";
    b.textContent = "Not found in your documents";
    $(".content", msgEl).after(b);
  }
}

async function readSSE(res, onEvent) {
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf("\n\n")) !== -1) {
      const raw = buf.slice(0, idx); buf = buf.slice(idx + 2);
      let event = "message", data = "";
      for (const line of raw.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data += line.slice(5).trim();
      }
      if (data) onEvent(event, JSON.parse(data));
    }
  }
}

function setBusy(busy) {
  state.busy = busy;
  els.send.hidden = busy; els.stop.hidden = !busy;
  els.question.disabled = busy;
  if (!busy) els.question.focus();
}

async function ask(question) {
  if (state.busy) return;
  if (!state.docs.length) { toast("Add at least one document first.", "error"); return; }
  const userEl = addMessage("user");
  $(".content", userEl).textContent = question;
  const msgEl = addMessage("assistant");
  const content = $(".content", msgEl);
  content.classList.add("typing");
  setBusy(true);

  const body = { question, history: state.history.slice(-6), mode: els.mode.value, top_k: +els.topK.value };
  state.abort = new AbortController();
  let answer = "", sources = [], done = null, errored = false, pending = false;
  const paint = () => { pending = false; content.innerHTML = renderMarkdown(answer); els.messages.scrollTop = els.messages.scrollHeight; };

  try {
    const res = await fetch("/api/chat", {
      method: "POST", signal: state.abort.signal,
      headers: { "Content-Type": "application/json", "X-Workspace-Id": state.ws },
      body: JSON.stringify(body),
    });
    if (!res.ok) { const j = await res.json().catch(() => ({})); throw new Error(j.detail || res.statusText); }
    await readSSE(res, (event, data) => {
      if (event === "sources") { sources = data.sources; }
      else if (event === "token") { answer += data.text; if (!pending) { pending = true; requestAnimationFrame(paint); } }
      else if (event === "done") { done = data; }
      else if (event === "error") { errored = true; content.innerHTML = `<div class="error-box">${esc(data.message)}</div>`; }
    });
  } catch (e) {
    if (e.name === "AbortError") answer += answer ? "\n\n_(stopped)_" : "_(stopped)_";
    else { errored = true; content.innerHTML = `<div class="error-box">${esc(e.message)}</div>`; }
  } finally {
    content.classList.remove("typing");
    if (!errored) paint();
    if (done) { renderSources(msgEl, sources, done.cited); renderMeta(msgEl, done); }
    if (!errored && answer) state.history.push({ role: "user", content: question }, { role: "assistant", content: answer.slice(0, 1500) });
    state.abort = null;
    setBusy(false);
  }
}

els.messages.addEventListener("click", (e) => {
  const cite = e.target.closest(".cite");
  if (!cite) return;
  const msg = cite.closest(".msg");
  const det = $("details.sources", msg);
  if (!det) return;
  det.open = true;
  const item = $(`[data-src="${cite.dataset.n}"]`, det);
  if (item) { item.scrollIntoView({ block: "nearest", behavior: "smooth" }); item.classList.remove("flash"); void item.offsetWidth; item.classList.add("flash"); }
});

function submitQuestion() {
  const q = els.question.value.trim();
  if (!q) return;
  els.question.value = ""; autosize();
  ask(q);
}
els.form.addEventListener("submit", (e) => { e.preventDefault(); submitQuestion(); });
els.question.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); submitQuestion(); } });
function autosize() { els.question.style.height = "auto"; els.question.style.height = Math.min(els.question.scrollHeight, 160) + "px"; }
els.question.addEventListener("input", autosize);
els.stop.addEventListener("click", () => state.abort && state.abort.abort());
$$(".starter").forEach((b) => b.addEventListener("click", () => ask(b.textContent)));

function newChat() {
  if (state.abort) state.abort.abort();
  state.history = [];
  $$(".msg", els.messages).forEach((m) => m.remove());
  els.empty.hidden = false;
}
els.newChat.addEventListener("click", newChat);

/* ---------------------------------------------------------------- evaluation */
const MODE_LABEL = { dense: "Meaning only (vectors)", bm25: "Keywords only (BM25)", hybrid: "Hybrid (RRF)", hybrid_rerank: "Hybrid + rerank" };
const pct = (v) => (v == null ? "–" : Math.round(v * 100) + "%");
const meter = (v) => `<div class="meter"><i style="width:${Math.round((v || 0) * 100)}px"></i><span>${pct(v)}</span></div>`;

function renderEval(r) {
  const best = Object.entries(r.retrieval).sort((a, b) => b[1].mrr - a[1].mrr)[0][0];
  const rows = Object.entries(r.retrieval).map(([m, v]) => `
    <tr class="${m === best ? "best" : ""}"><td>${MODE_LABEL[m] || m}</td><td>${meter(v.hit_at_k)}</td><td class="num">${v.mrr.toFixed(2)}</td><td class="num">${Math.round(v.avg_latency_ms)} ms</td></tr>`).join("");
  const g = r.generation;
  const qrows = r.per_question.map((q) => {
    const ranks = Object.entries(q.ranks).map(([m, k]) => `<span class="${k ? "rank-hit" : "rank-miss"}" title="${MODE_LABEL[m] || m}">${k ? "#" + k : "miss"}</span>`).join(" / ");
    return `<tr><td>${esc(q.question)}<br><span class="muted small">Source: ${esc(q.gold)}</span></td><td>${ranks}</td><td class="num">${pct(q.faithfulness)}</td><td class="num">${pct(q.relevancy)}</td><td class="num">${pct(q.correctness)}</td></tr>`;
  }).join("");
  const order = Object.keys(r.retrieval).map((m) => MODE_LABEL[m] || m).join(" / ");
  $("#evalResults").innerHTML = `
    <div class="card">
      <h2>Which search method finds the right passage?</h2>
      <p class="hint">Hit rate = how often the passage a question was written from appears in the top ${r.top_k} results. MRR rewards ranking it higher (1.0 is always first). Highlighted row: best MRR.</p>
      <div class="table-wrap"><table><thead><tr><th>Method</th><th>Hit rate @${r.top_k}</th><th class="num">MRR</th><th class="num">Search time</th></tr></thead><tbody>${rows}</tbody></table></div>
    </div>
    <div class="card">
      <h2>How good are the final answers? <span class="muted small">(${MODE_LABEL[g.mode] || g.mode})</span></h2>
      <div class="metrics">
        <div class="metric"><b>${pct(g.faithfulness)}</b><span>Faithfulness: claims backed by sources</span></div>
        <div class="metric"><b>${pct(g.relevancy)}</b><span>Relevancy: answers the question</span></div>
        <div class="metric"><b>${pct(g.correctness)}</b><span>Correctness: matches reference</span></div>
        <div class="metric"><b>${pct(g.refusal_rate)}</b><span>Refused to answer</span></div>
      </div>
      <p class="hint">Scores come from a language-model judge, so treat small differences as noise. ${r.n_questions} questions · ${esc(r.config.llm)} · ${esc(r.config.embedding_model)}</p>
    </div>
    <div class="card">
      <h2>Question by question</h2>
      <p class="hint">Rank of the correct passage for each method (${esc(order)}), then answer scores.</p>
      <div class="table-wrap"><table><thead><tr><th>Question</th><th>Rank</th><th class="num">Faithful</th><th class="num">Relevant</th><th class="num">Correct</th></tr></thead><tbody>${qrows}</tbody></table></div>
    </div>`;
}

async function runEval() {
  const btn = $("#runEval"), box = $("#evalProgress"), bar = $("#evalBar"), stage = $("#evalStage"), out = $("#evalResults");
  out.innerHTML = "";
  if (!state.docs.length) { toast("Add at least one document first.", "error"); return; }
  btn.disabled = true; box.hidden = false; bar.style.width = "2%"; stage.textContent = "Starting…";
  try {
    const { job_id } = await api("/api/evaluate", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ num_questions: +$("#evalN").value || 8, top_k: +els.topK.value }),
    });
    await new Promise((resolve, reject) => {
      state.evalTimer = setInterval(async () => {
        try {
          const job = await api(`/api/evaluate/${job_id}`);
          bar.style.width = Math.max(2, job.progress * 100) + "%"; stage.textContent = job.stage;
          if (job.status === "done") { clearInterval(state.evalTimer); renderEval(job.result); resolve(); }
          if (job.status === "error") { clearInterval(state.evalTimer); reject(new Error(job.error)); }
        } catch (e) { clearInterval(state.evalTimer); reject(e); }
      }, 1500);
    });
  } catch (e) { out.innerHTML = `<div class="error-box">${esc(e.message)}</div>`; }
  finally { btn.disabled = false; box.hidden = true; }
}
$("#runEval").addEventListener("click", runEval);

/* ---------------------------------------------------------------- start */
loadConfig();
loadDocs();
