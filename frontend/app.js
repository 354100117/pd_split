const modeBadge = document.getElementById("modeBadge");
const runBtn = document.getElementById("runBtn");
const promptInput = document.getElementById("promptInput");
const batchMode = document.getElementById("batchMode");
const maxNewTokens = document.getElementById("maxNewTokens");
const batchSizeInput = document.getElementById("batchSizeInput");
const decodeBatchSizeInput = document.getElementById("decodeBatchSizeInput");
const decodeBatchTimeoutInput = document.getElementById("decodeBatchTimeoutInput");
const decodeWorkersInput = document.getElementById("decodeWorkersInput");
const batchApply = document.getElementById("batchApply");
const batchHint = document.getElementById("batchHint");

const inflightEl = document.getElementById("inflight");
const completedRequestsEl = document.getElementById("completedRequests");
const prefillQueueEl = document.getElementById("prefillQueue");
const decodeQueueEl = document.getElementById("decodeQueue");
const prefillActiveEl = document.getElementById("prefillActive");
const decodeActiveEl = document.getElementById("decodeActive");

const p50El = document.getElementById("p50");
const p95El = document.getElementById("p95");
const p99El = document.getElementById("p99");
const avgTpsEl = document.getElementById("avgTps");
const avgComputeTpsEl = document.getElementById("avgComputeTps");
const summaryCompletedEl = document.getElementById("summaryCompleted");
const summaryAvgLatencyEl = document.getElementById("summaryAvgLatency");
const summaryAvgTpsEl = document.getElementById("summaryAvgTps");
const summaryAvgComputeTpsEl = document.getElementById("summaryAvgComputeTps");

const timelineTable = document.getElementById("timelineTable");
const resultList = document.getElementById("resultList");
const gpuList = document.getElementById("gpuList");

const logRequest = document.getElementById("logRequest");
const logStage = document.getElementById("logStage");
const logPipeline = document.getElementById("logPipeline");
const logSystem = document.getElementById("logSystem");
const singleNodeSelect = document.getElementById("singleNodeSelect");
const singleNodeHint = document.getElementById("singleNodeHint");
const modelNameDisplay = document.getElementById("modelNameDisplay");
const fullPipelineStrategy = document.getElementById("fullPipelineStrategy");
const fullPipelineNodes = document.getElementById("fullPipelineNodes");
const fullPipelineApply = document.getElementById("fullPipelineApply");
const fullPipelineHint = document.getElementById("fullPipelineHint");
const fullPipelineLayout = document.getElementById("fullPipelineLayout");
const fullBatchSizeInput = document.getElementById("fullBatchSizeInput");
const fullBatchTimeoutInput = document.getElementById("fullBatchTimeoutInput");
const fullDecodeBatchSizeInput = document.getElementById("fullDecodeBatchSizeInput");
const fullDecodeBatchTimeoutInput = document.getElementById("fullDecodeBatchTimeoutInput");
const fullDecodeWorkersInput = document.getElementById("fullDecodeWorkersInput");
const fullBatchApply = document.getElementById("fullBatchApply");
const fullBatchHint = document.getElementById("fullBatchHint");
const pdPrefillLayout = document.getElementById("pdPrefillLayout");
const pdDecodeLayout = document.getElementById("pdDecodeLayout");
const singleNodeLayout = document.getElementById("singleNodeLayout");

const tabButtons = document.querySelectorAll(".tab");
const panels = {
  pd_split: document.getElementById("panel-pd_split"),
  single_node: document.getElementById("panel-single_node"),
  full_pipeline: document.getElementById("panel-full_pipeline"),
};
const panelBodies = {
  pd_split: document.getElementById("panelBody-pd_split"),
  single_node: document.getElementById("panelBody-single_node"),
  full_pipeline: document.getElementById("panelBody-full_pipeline"),
};

const requestMap = new Map();
let experimentMode = "pd_split";
let nodeSnapshot = null;
let summaryStats = { avgTotal: 0, avgTps: 0, avgComputeTps: 0, p95: 0 };
let lastResultOpen = null;

function renderTabs(activeMode) {
  tabButtons.forEach((btn) => {
    const mode = btn.dataset.mode;
    btn.classList.toggle("active", mode === activeMode);
    panels[mode].style.display = "block";
    if (mode === activeMode) {
      if (mode !== "single_node" && mode !== "full_pipeline" && mode !== "pd_split") {
        panelBodies[mode].textContent = "当前后端正在运行该实验模式。";
      }
    } else {
      panelBodies[mode].textContent = "该模式需要重启服务切换。";
    }
  });
}

function formatMs(value) {
  if (value === undefined || value === null) return "-";
  return value.toFixed(2);
}

function computeComputeTps(item) {
  const tokens = Number(item.generated_tokens || 0);
  const prefillMs = Number(item.prefill_latency_ms || 0);
  const decodeMs = Number(item.decode_latency_ms || 0);
  const computeMs = prefillMs + decodeMs;
  if (!tokens || computeMs <= 0) return null;
  return tokens / (computeMs / 1000);
}

function computePercentile(values, p) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const idx = Math.min(sorted.length - 1, Math.floor((p / 100) * sorted.length));
  return sorted[idx];
}

function refreshSummary() {
  const totals = [];
  let totalTokens = 0;
  let totalLatencyMs = 0;
  let computeTokens = 0;
  let computeTimeMs = 0;
  requestMap.forEach((item) => {
    if (item.total_latency_ms !== undefined) {
      const totalMs = Number(item.total_latency_ms || 0);
      totals.push(totalMs);
      totalLatencyMs += totalMs;
    }
    if (item.generated_tokens !== undefined) {
      totalTokens += Number(item.generated_tokens || 0);
    }
    if (item.generated_tokens !== undefined) {
      const prefillMs = Number(item.prefill_latency_ms || 0);
      const decodeMs = Number(item.decode_latency_ms || 0);
      const computeMs = prefillMs + decodeMs;
      if (computeMs > 0) {
        computeTokens += Number(item.generated_tokens || 0);
        computeTimeMs += computeMs;
      }
    }
  });
  p50El.textContent = formatMs(computePercentile(totals, 50));
  p95El.textContent = formatMs(computePercentile(totals, 95));
  p99El.textContent = formatMs(computePercentile(totals, 99));
  const avgTotal = totals.length ? totals.reduce((a, b) => a + b, 0) / totals.length : 0;
  let avgTps = 0;
  if (totalTokens > 0 && totalLatencyMs > 0) {
    avgTps = totalTokens / (totalLatencyMs / 1000);
    avgTpsEl.textContent = avgTps.toFixed(2);
  } else {
    avgTpsEl.textContent = "-";
  }
  let avgComputeTps = 0;
  if (avgComputeTpsEl) {
    if (computeTimeMs > 0) {
      avgComputeTps = computeTokens / (computeTimeMs / 1000);
      avgComputeTpsEl.textContent = avgComputeTps.toFixed(2);
    } else {
      avgComputeTpsEl.textContent = "-";
    }
  }
  summaryStats = {
    avgTotal,
    avgTps,
    avgComputeTps,
    p95: computePercentile(totals, 95) || 0,
  };
  if (summaryCompletedEl) summaryCompletedEl.textContent = String(totals.length);
  if (summaryAvgLatencyEl) summaryAvgLatencyEl.textContent = avgTotal ? `${formatMs(avgTotal)} ms` : "-";
  if (summaryAvgTpsEl) summaryAvgTpsEl.textContent = summaryStats.avgTps ? summaryStats.avgTps.toFixed(2) : "-";
  if (summaryAvgComputeTpsEl) {
    summaryAvgComputeTpsEl.textContent = summaryStats.avgComputeTps ? summaryStats.avgComputeTps.toFixed(2) : "-";
  }
}

function renderTimeline() {
  const header = document.createElement("div");
  header.className = "table-row header timeline-row";
  header.innerHTML =
    "<div>请求</div><div>总延迟</div><div>Decode</div><div>Prefill</div><div>序号(S/F)</div><div>TPS</div>";
  timelineTable.innerHTML = "";
  timelineTable.appendChild(header);
  const items = Array.from(requestMap.values());
  const bySubmit = [...items].sort((a, b) => (a.arrival_time_ns || 0) - (b.arrival_time_ns || 0));
  const byFinish = [...items].sort((a, b) => (a.finish_time_ns || 0) - (b.finish_time_ns || 0));
  const submitRank = new Map();
  const finishRank = new Map();
  bySubmit.forEach((item, idx) => submitRank.set(item.request_id, idx + 1));
  byFinish.forEach((item, idx) => finishRank.set(item.request_id, idx + 1));
  const recent = bySubmit.slice(-20);
  const maxTotal = Math.max(1, ...recent.map((x) => Number(x.total_latency_ms || 0)));
  const maxTps = Math.max(1, ...recent.map((x) => Number(x.throughput_tps || 0)));
  const maxComputeTps = Math.max(1, ...recent.map((x) => Number(computeComputeTps(x) || 0)));
  const slowThreshold = summaryStats.p95 || summaryStats.avgTotal * 2 || 0;

  recent.forEach((item) => {
    const row = document.createElement("div");
    row.className = "table-row timeline-row";
    const totalMs = Number(item.total_latency_ms || 0);
    const decodeMs = Number(item.decode_latency_ms || 0);
    const prefillMs = Number(item.prefill_latency_ms || 0);
    const decodePct = totalMs > 0 ? Math.round((decodeMs / totalMs) * 100) : 0;
    const computeTps = computeComputeTps(item);
    const submitIdx = submitRank.get(item.request_id) || "-";
    const finishIdx = finishRank.get(item.request_id) || "-";
    const outOfOrder = submitIdx !== finishIdx;
    const finishDiff =
      typeof submitIdx === "number" && typeof finishIdx === "number" ? finishIdx - submitIdx : 0;
    if (slowThreshold && totalMs >= slowThreshold) {
      row.classList.add("slow");
    }
    if (outOfOrder) {
      row.classList.add("out-of-order");
    }
    let decodeClass = "decode-low";
    if (decodePct >= 70) decodeClass = "decode-high";
    else if (decodePct >= 40) decodeClass = "decode-mid";
    const totalWidth = Math.min(100, Math.round((totalMs / maxTotal) * 100));
    const tpsWidth = Math.min(100, Math.round((Number(item.throughput_tps || 0) / maxTps) * 100));
    const computeWidth = Math.min(100, Math.round((Number(computeTps || 0) / maxComputeTps) * 100));
    row.innerHTML = `
      <div class="req-id">${item.request_id}</div>
      <div class="total-cell">
        <div class="latency-main">${formatMs(totalMs)} ms</div>
        <div class="latency-bar"><div class="latency-fill" style="width:${totalWidth}%"></div></div>
      </div>
      <div class="decode-cell ${decodeClass}">
        <div class="latency-main">${formatMs(decodeMs)} ms</div>
        <div class="latency-sub">${decodePct}%</div>
        <div class="latency-bar"><div class="latency-fill" style="width:${decodePct}%"></div></div>
      </div>
      <div class="prefill-cell">
        <div class="latency-main">${formatMs(prefillMs)} ms</div>
      </div>
      <div class="order-cell">
        <div class="order-main">S${submitIdx} / F${finishIdx}</div>
        ${outOfOrder ? `<div class="order-note">${finishDiff < 0 ? "提前" : "滞后"} ${Math.abs(finishDiff)}</div>` : ""}
      </div>
      <div class="tps-cell">
        <div class="tps-row">
          <span class="tps-label">端</span>
          <div class="tps-bar"><div class="tps-fill" style="width:${tpsWidth}%"></div></div>
          <span class="tps-value">${item.throughput_tps ? item.throughput_tps.toFixed(2) : "-"}</span>
        </div>
        <div class="tps-row compute">
          <span class="tps-label">算</span>
          <div class="tps-bar"><div class="tps-fill" style="width:${computeWidth}%"></div></div>
          <span class="tps-value">${computeTps ? computeTps.toFixed(2) : "-"}</span>
        </div>
      </div>
    `;
    timelineTable.appendChild(row);
  });
}

function addResult(payload) {
  const summary = requestMap.get(payload.request_id) || {};
  const totalMs = Number(summary.total_latency_ms || 0);
  const decodeMs = Number(summary.decode_latency_ms || 0);
  const decodePct = totalMs > 0 ? Math.round((decodeMs / totalMs) * 100) : 0;
  const endTps = summary.throughput_tps;
  const computeTps = computeComputeTps(summary);
  const slowThreshold = summaryStats.p95 || summaryStats.avgTotal * 2 || 0;
  const isSlow = slowThreshold && totalMs >= slowThreshold;

  const details = document.createElement("details");
  details.className = "result-item";
  if (isSlow) details.classList.add("slow");
  details.open = true;

  const summaryEl = document.createElement("summary");
  summaryEl.className = "result-header";
  summaryEl.innerHTML = `
    <div class="result-id">${payload.request_id}</div>
    <div class="result-meta">
      <span>${payload.generated_tokens} tok</span>
      <span>${totalMs ? formatMs(totalMs) + " ms" : "-"}</span>
      <span>${decodePct ? `Decode ${decodePct}%` : "Decode -"}</span>
      <span>${endTps ? `TPS ${endTps.toFixed(2)}` : "TPS -"}</span>
      <span>${computeTps ? `算 ${computeTps.toFixed(2)}` : "算 -"}</span>
    </div>
    <div class="result-toggle">展开</div>
  `;

  const body = document.createElement("div");
  body.className = "result-body";
  body.textContent = payload.text || "";

  details.appendChild(summaryEl);
  details.appendChild(body);
  resultList.prepend(details);
  if (lastResultOpen && lastResultOpen !== details) {
    lastResultOpen.open = false;
  }
  lastResultOpen = details;
  if (resultList.children.length > 30) {
    resultList.removeChild(resultList.lastChild);
  }
}

function updateGpuList(gpuUsed, gpuTotal) {
  gpuList.innerHTML = "";
  const usedMap = gpuUsed || {};
  const totalMap = gpuTotal || {};
  const ips = new Set([...Object.keys(usedMap), ...Object.keys(totalMap)]);
  Array.from(ips).forEach((ip) => {
    const div = document.createElement("div");
    div.className = "gpu-item";
    const used = Number(usedMap[ip] || 0);
    const total = Number(totalMap[ip] || 0);
    const pct = total > 0 ? (used / total) * 100 : 0;
    div.innerHTML = `
      <div class="gpu-header">${ip}</div>
      <div class="gpu-mem">${total > 0 ? `${used}/${total} MB (${pct.toFixed(1)}%)` : `${used} MB`}</div>
      <div class="gpu-bar"><div class="gpu-fill" style="width:${Math.min(100, pct).toFixed(1)}%"></div></div>
    `;
    gpuList.appendChild(div);
  });
}

function parsePrompts() {
  const text = promptInput.value || "";
  const mode = batchMode.value;
  if (mode === "single") {
    return [text.trim()].filter((t) => t.length > 0);
  }
  if (mode === "line") {
    return text.split("\n").map((t) => t.trim()).filter((t) => t.length > 0);
  }
  if (mode === "json") {
    try {
      const arr = JSON.parse(text);
      if (Array.isArray(arr)) {
        return arr.map((t) => String(t)).filter((t) => t.length > 0);
      }
    } catch (e) {
      alert("JSON 解析失败，请输入数组格式");
    }
  }
  return [];
}

runBtn.addEventListener("click", async () => {
  const prompts = parsePrompts();
  if (!prompts.length) {
    alert("请输入 prompt");
    return;
  }
  const payload = {
    prompts,
    max_new_tokens: Number(maxNewTokens.value || 32),
  };
  if (experimentMode === "single_node" && singleNodeSelect) {
    if (singleNodeSelect.disabled || !singleNodeSelect.value) {
      alert("没有可用节点运行当前模型");
      return;
    }
    payload.target_node = singleNodeSelect.value;
  }
  const resp = await fetch("/api/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!resp.ok) {
    const detail = await resp.text();
    alert(`请求失败: ${detail}`);
  }
});

function renderNodeSelect() {
  if (!singleNodeSelect) return;
  singleNodeSelect.innerHTML = "";
  if (!nodeSnapshot || !Array.isArray(nodeSnapshot.nodes)) {
    singleNodeSelect.disabled = true;
    if (singleNodeHint) singleNodeHint.textContent = "节点信息未加载。";
    return;
  }
  const nodes = nodeSnapshot.nodes;
  let firstEnabled = null;
  nodes.forEach((node) => {
    const opt = document.createElement("option");
    opt.value = node.name;
    const mem = node.gpu_mem_gb !== undefined ? `${node.gpu_mem_gb.toFixed(2)}GB` : "未知";
    const suffix = node.can_run ? "" : " - 不可用";
    opt.textContent = `${node.name} (${mem})${suffix}`;
    if (!node.can_run) {
      opt.disabled = true;
    } else if (!firstEnabled) {
      firstEnabled = node.name;
    }
    singleNodeSelect.appendChild(opt);
  });
  const selected = nodeSnapshot.selected_node;
  if (selected && nodes.find((n) => n.name === selected && n.can_run)) {
    singleNodeSelect.value = selected;
  } else if (firstEnabled) {
    singleNodeSelect.value = firstEnabled;
  }
  singleNodeSelect.disabled = !firstEnabled;
  if (singleNodeHint) {
    const required = nodeSnapshot.required_mem_gb || 0;
    const base = required > 0 ? `模型估算显存需求约 ${required.toFixed(2)} GB` : "模型显存需求未知";
    const note = firstEnabled ? "不可用节点已灰置" : "无可用节点";
    singleNodeHint.textContent = `${base} - ${note}`;
  }
}

async function loadNodes() {
  try {
    nodeSnapshot = await fetch("/api/nodes").then((r) => r.json());
  } catch (e) {
    nodeSnapshot = null;
  }
  renderNodeSelect();
}

function renderFullPipelineConfig(data) {
  if (!fullPipelineNodes || !fullPipelineStrategy) return;
  const nodes = Array.isArray(data.nodes) ? data.nodes : [];
  const selected = new Set(data.selected_nodes || []);
  fullPipelineNodes.innerHTML = "";
  let belowCount = 0;
  nodes.forEach((node) => {
    const opt = document.createElement("option");
    opt.value = node.name;
    const mem = node.gpu_mem_gb !== undefined ? `${node.gpu_mem_gb.toFixed(2)}GB` : "未知";
    const suffix = node.can_run ? "" : " - 低于估算";
    opt.textContent = `${node.name} (${mem})${suffix}`;
    if (!node.can_run) {
      belowCount += 1;
    }
    if (selected.has(node.name)) {
      opt.selected = true;
    }
    fullPipelineNodes.appendChild(opt);
  });
  if (fullPipelineStrategy) {
    fullPipelineStrategy.value = data.strategy || "mem";
  }
  if (fullPipelineHint) {
    const required = data.required_mem_gb || 0;
    const base = required > 0 ? `模型估算显存需求约 ${required.toFixed(2)} GB` : "模型显存需求未知";
    const note = belowCount > 0 ? `已允许选择低于估算的节点（${belowCount}）` : "所有节点满足估算";
    fullPipelineHint.textContent = `${base} - ${note}`;
  }
  renderFullPipelineLayout(data.layout || {});
}

function renderFullPipelineLayout(layout) {
  if (!fullPipelineLayout) return;
  const nodes = Array.isArray(layout.nodes) ? layout.nodes : [];
  const header = document.createElement("div");
  header.className = "table-row header";
  header.innerHTML = "<div>节点</div><div>层范围</div><div>层数</div>";
  fullPipelineLayout.innerHTML = "";
  fullPipelineLayout.appendChild(header);
  nodes.forEach((item) => {
    const row = document.createElement("div");
    row.className = "table-row";
    const range = item.layer_range || "-";
    let count = "-";
    if (typeof range === "string" && range.includes("-")) {
      const parts = range.split("-");
      const start = parseInt(parts[0], 10);
      const end = parseInt(parts[1], 10);
      if (!Number.isNaN(start) && !Number.isNaN(end)) {
        count = Math.max(0, end - start);
      }
    }
    row.innerHTML = `<div>${item.node || "-"}</div><div>${range}</div><div>${count}</div>`;
    fullPipelineLayout.appendChild(row);
  });
  if (!nodes.length) {
    const row = document.createElement("div");
    row.className = "table-row";
    row.innerHTML = "<div>-</div><div>暂无层划分</div><div>-</div>";
    fullPipelineLayout.appendChild(row);
  }
}

function renderStageTable(container, items, emptyText) {
  if (!container) return;
  const rows = Array.isArray(items) ? items : [];
  const header = document.createElement("div");
  header.className = "table-row header";
  header.innerHTML = "<div>节点</div><div>层范围</div><div>层数</div>";
  container.innerHTML = "";
  container.appendChild(header);
  rows.forEach((item) => {
    const row = document.createElement("div");
    row.className = "table-row";
    const range = item.layer_range || "-";
    let count = "-";
    if (typeof range === "string" && range.includes("-")) {
      const parts = range.split("-");
      const start = parseInt(parts[0], 10);
      const end = parseInt(parts[1], 10);
      if (!Number.isNaN(start) && !Number.isNaN(end)) {
        count = Math.max(0, end - start);
      }
    }
    row.innerHTML = `<div>${item.node || "-"}</div><div>${range}</div><div>${count}</div>`;
    container.appendChild(row);
  });
  if (!rows.length) {
    const row = document.createElement("div");
    row.className = "table-row";
    row.innerHTML = `<div>-</div><div>${emptyText}</div><div>-</div>`;
    container.appendChild(row);
  }
}

function renderPdSplitLayout(layout) {
  renderStageTable(pdPrefillLayout, layout.prefill || [], "暂无 Prefill 划分");
  renderStageTable(pdDecodeLayout, layout.decode || [], "暂无 Decode 划分");
}

function renderSingleNodeLayout(layout) {
  renderStageTable(singleNodeLayout, layout.nodes || [], "暂无层划分");
}

async function loadFullPipelineConfig() {
  try {
    const data = await fetch("/api/full_pipeline/config").then((r) => r.json());
    renderFullPipelineConfig(data);
  } catch (e) {
    if (fullPipelineHint) fullPipelineHint.textContent = "加载全节点配置失败。";
  }
}

async function loadLayout() {
  try {
    const data = await fetch("/api/layout").then((r) => r.json());
    const layout = data.layout || {};
    if (data.experiment_mode === "pd_split") {
      renderPdSplitLayout(layout);
    } else if (data.experiment_mode === "single_node") {
      renderSingleNodeLayout(layout);
    }
  } catch (e) {
    renderPdSplitLayout({});
    renderSingleNodeLayout({});
  }
}

async function applyFullPipelineConfig() {
  if (!fullPipelineNodes || !fullPipelineStrategy) return;
  const selected = Array.from(fullPipelineNodes.selectedOptions)
    .filter((opt) => !opt.disabled)
    .map((opt) => opt.value);
  if (!selected.length) {
    alert("请选择至少一个可用节点");
    return;
  }
  if (fullPipelineApply) fullPipelineApply.disabled = true;
  try {
    const resp = await fetch("/api/full_pipeline/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ strategy: fullPipelineStrategy.value, nodes: selected }),
    });
    if (!resp.ok) {
      const detail = await resp.text();
      alert(`应用失败: ${detail}`);
      return;
    }
    const payload = await resp.json().catch(() => ({}));
    requestMap.clear();
    renderTimeline();
    refreshSummary();
    if (resultList) resultList.innerHTML = "";
    if (payload.layout) {
      renderFullPipelineLayout(payload.layout);
    } else {
      await loadFullPipelineConfig();
    }
  } finally {
    if (fullPipelineApply) fullPipelineApply.disabled = false;
  }
}

async function init() {
  const config = await fetch("/api/config").then((r) => r.json());
  experimentMode = config.experiment_mode;
  modeBadge.textContent = `当前模式：${experimentMode}`;
  renderTabs(experimentMode);
  if (modelNameDisplay) {
    modelNameDisplay.textContent = config.model_name || "-";
  }
  if (batchSizeInput) {
    batchSizeInput.value = config.batch_size || 4;
    batchSizeInput.disabled = experimentMode !== "pd_split";
  }
  if (decodeBatchSizeInput) {
    decodeBatchSizeInput.value = config.decode_batch_size || 1;
    decodeBatchSizeInput.disabled = experimentMode !== "pd_split";
  }
  if (decodeBatchTimeoutInput) {
    decodeBatchTimeoutInput.value = config.decode_batch_timeout_ms || 10;
    decodeBatchTimeoutInput.disabled = experimentMode !== "pd_split";
  }
  if (decodeWorkersInput) {
    decodeWorkersInput.value = config.decode_workers || 1;
    decodeWorkersInput.disabled = experimentMode !== "pd_split";
  }
  if (batchApply) {
    batchApply.disabled = experimentMode !== "pd_split";
  }
  if (fullBatchSizeInput) {
    fullBatchSizeInput.value = config.batch_size || 4;
    fullBatchSizeInput.disabled = experimentMode !== "full_pipeline";
  }
  if (fullBatchTimeoutInput) {
    fullBatchTimeoutInput.value = config.batch_timeout_ms || 20;
    fullBatchTimeoutInput.disabled = experimentMode !== "full_pipeline";
  }
  if (fullDecodeBatchSizeInput) {
    fullDecodeBatchSizeInput.value = config.decode_batch_size || 1;
    fullDecodeBatchSizeInput.disabled = experimentMode !== "full_pipeline";
  }
  if (fullDecodeBatchTimeoutInput) {
    fullDecodeBatchTimeoutInput.value = config.decode_batch_timeout_ms || 10;
    fullDecodeBatchTimeoutInput.disabled = experimentMode !== "full_pipeline";
  }
  if (fullDecodeWorkersInput) {
    fullDecodeWorkersInput.value = config.decode_workers || 1;
    fullDecodeWorkersInput.disabled = experimentMode !== "full_pipeline";
  }
  if (fullBatchApply) {
    fullBatchApply.disabled = experimentMode !== "full_pipeline";
  }
  loadNodes();
  loadLayout();
  if (fullPipelineApply) {
    fullPipelineApply.addEventListener("click", applyFullPipelineConfig);
  }
  if (batchApply) {
    batchApply.addEventListener("click", async () => {
      if (experimentMode !== "pd_split") return;
      const size = Number(batchSizeInput?.value || 0);
      const decodeSize = Number(decodeBatchSizeInput?.value || 0);
      const decodeTimeout = Number(decodeBatchTimeoutInput?.value || 0);
      const decodeWorkers = Number(decodeWorkersInput?.value || 0);
      if (!size || size < 1) {
        alert("batch_size 必须 >= 1");
        return;
      }
      if (!decodeSize || decodeSize < 1) {
        alert("decode_batch_size 必须 >= 1");
        return;
      }
      if (!decodeTimeout || decodeTimeout < 1) {
        alert("decode_batch_timeout_ms 必须 >= 1");
        return;
      }
      if (!decodeWorkers || decodeWorkers < 1) {
        alert("decode_workers 必须 >= 1");
        return;
      }
      try {
        const resp = await fetch("/api/pd_split/batching", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            batch_size: size,
            decode_batch_size: decodeSize,
            decode_batch_timeout_ms: decodeTimeout,
            decode_workers: decodeWorkers,
          }),
        });
        if (!resp.ok) {
          const detail = await resp.text();
          if (batchHint) batchHint.textContent = `更新失败: ${detail}`;
          return;
        }
        const data = await resp.json();
        if (batchHint) {
          batchHint.textContent =
            `已更新 batch_size=${data.batch_size}, decode_batch_size=${data.decode_batch_size}, decode_workers=${data.decode_workers}`;
        }
      } catch (e) {
        if (batchHint) batchHint.textContent = "更新失败，请检查服务状态。";
      }
    });
  }
  if (fullBatchApply) {
    fullBatchApply.addEventListener("click", async () => {
      if (experimentMode !== "full_pipeline") return;
      const size = Number(fullBatchSizeInput?.value || 0);
      const timeout = Number(fullBatchTimeoutInput?.value || 0);
      const decodeSize = Number(fullDecodeBatchSizeInput?.value || 0);
      const decodeTimeout = Number(fullDecodeBatchTimeoutInput?.value || 0);
      const decodeWorkers = Number(fullDecodeWorkersInput?.value || 0);
      if (!size || size < 1) {
        alert("batch_size 必须 >= 1");
        return;
      }
      if (!timeout || timeout < 1) {
        alert("batch_timeout_ms 必须 >= 1");
        return;
      }
      if (!decodeSize || decodeSize < 1) {
        alert("decode_batch_size 必须 >= 1");
        return;
      }
      if (!decodeTimeout || decodeTimeout < 1) {
        alert("decode_batch_timeout_ms 必须 >= 1");
        return;
      }
      if (!decodeWorkers || decodeWorkers < 1) {
        alert("decode_workers 必须 >= 1");
        return;
      }
      try {
        const resp = await fetch("/api/full_pipeline/batching", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            batch_size: size,
            batch_timeout_ms: timeout,
            decode_batch_size: decodeSize,
            decode_batch_timeout_ms: decodeTimeout,
            decode_workers: decodeWorkers,
          }),
        });
        if (!resp.ok) {
          const detail = await resp.text();
          if (fullBatchHint) fullBatchHint.textContent = `更新失败: ${detail}`;
          return;
        }
        const data = await resp.json();
        if (fullBatchHint) {
          fullBatchHint.textContent =
            `已更新 batch_size=${data.batch_size}, decode_batch_size=${data.decode_batch_size}, decode_workers=${data.decode_workers}`;
        }
      } catch (e) {
        if (fullBatchHint) fullBatchHint.textContent = "更新失败，请检查服务状态。";
      }
    });
  }
  if (experimentMode === "full_pipeline") {
    loadFullPipelineConfig();
  }

  logRequest.href = "/api/logs/file?name=request";
  logStage.href = "/api/logs/file?name=stage";
  logPipeline.href = "/api/logs/file?name=pipeline";
  logSystem.href = "/api/logs/file?name=system";

  setInterval(async () => {
    const status = await fetch("/api/status").then((r) => r.json());
    inflightEl.textContent = status.inflight_requests;
    if (completedRequestsEl) {
      completedRequestsEl.textContent = status.completed_requests_total ?? 0;
    }
    prefillQueueEl.textContent = status.prefill_queue_len;
    decodeQueueEl.textContent = status.decode_queue_len;
    prefillActiveEl.textContent = status.prefill_active;
    decodeActiveEl.textContent = status.decode_active;
  }, 1000);

  const evt = new EventSource("/api/stream");
  evt.onmessage = (msg) => {
    if (!msg.data) return;
    let payload = null;
    try {
      payload = JSON.parse(msg.data);
    } catch (e) {
      return;
    }
    if (!payload || !payload.type) return;

    if (payload.type === "system_tick") {
      updateGpuList(payload.gpu_mem_used_mb, payload.gpu_mem_total_mb);
    }

    if (payload.type === "request_summary") {
      requestMap.set(payload.request_id, payload);
      refreshSummary();
      renderTimeline();
    }

    if (payload.type === "request_result") {
      addResult(payload);
    }
  };
}

init();
