// ---------------------------------------------------------------------------
//  Triality Agentic Simulator — Frontend
//  Chat-based UI with SSE streaming, tool call visualization, field charts
// ---------------------------------------------------------------------------

const state = {
    catalog: null,
    messages: [],
    isRunning: false,
    currentEventSource: null,
};

const els = {
    chatMessages: document.getElementById("chat-messages"),
    chatContainer: document.getElementById("chat-container"),
    promptInput: document.getElementById("prompt-input"),
    sendButton: document.getElementById("send-button"),
    scenarioList: document.getElementById("scenario-list"),
    modulePills: document.getElementById("module-pills"),
    toolPills: document.getElementById("tool-pills"),
    capabilitiesList: document.getElementById("capabilities-list"),
    tokenInput: document.getElementById("token-input"),
    modelSelect: document.getElementById("model-select"),
    agentStatus: document.getElementById("agent-status"),
    agentStatusDot: document.getElementById("agent-status-dot"),
    footerStatus: document.getElementById("footer-status"),
};

// ---------------------------------------------------------------------------
//  Utilities
// ---------------------------------------------------------------------------
function esc(value) {
    return String(value).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function autoResize(el) {
    el.style.height = "auto";
    el.style.height = `${el.scrollHeight}px`;
}

function scrollToBottom() {
    requestAnimationFrame(() => {
        els.chatContainer.scrollTop = els.chatContainer.scrollHeight;
    });
}

function setAgentStatus(text, active = false) {
    els.agentStatus.textContent = text;
    els.agentStatusDot.className = `w-1.5 h-1.5 rounded-full ${active ? "bg-accent animate-pulse" : "bg-accent/60"}`;
}

function formatNumber(n) {
    if (typeof n !== "number") return String(n);
    if (Math.abs(n) < 0.001 || Math.abs(n) > 99999) return n.toExponential(3);
    return n.toFixed(4).replace(/\.?0+$/, "");
}

// ---------------------------------------------------------------------------
//  Simple Markdown → HTML (headers, bold, tables, lists, code)
// ---------------------------------------------------------------------------
function renderMarkdown(md) {
    if (!md) return "";
    let html = esc(md);
    // Code blocks
    html = html.replace(/```(\w*)\n([\s\S]*?)```/g, '<pre class="text-[11px] font-mono bg-surface-900/60 rounded-lg p-3 my-2 overflow-auto text-white/50">$2</pre>');
    // Inline code
    html = html.replace(/`([^`]+)`/g, '<code class="text-[11px] font-mono bg-surface-800/60 px-1.5 py-0.5 rounded text-accent/70">$1</code>');
    // Headers
    html = html.replace(/^### (.+)$/gm, '<h3 class="text-sm font-display font-semibold text-white/80 mt-3 mb-1">$1</h3>');
    html = html.replace(/^## (.+)$/gm, '<h2 class="text-base font-display font-bold text-white/85 mt-4 mb-2">$1</h2>');
    // Bold
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong class="text-white/80 font-semibold">$1</strong>');
    // Italic
    html = html.replace(/\*(.+?)\*/g, '<em class="text-white/60">$1</em>');
    // Tables
    html = html.replace(/^\|(.+)\|$/gm, (match) => {
        const cells = match.split("|").filter(c => c.trim());
        if (cells.every(c => /^[\s-:]+$/.test(c))) {
            return ""; // separator row
        }
        const isHeader = false; // simplified
        const tds = cells.map(c => `<td class="px-2 py-1 border border-white/[0.06] text-[11px] font-mono text-white/55">${c.trim()}</td>`).join("");
        return `<tr>${tds}</tr>`;
    });
    html = html.replace(/((<tr>.*<\/tr>\s*)+)/g, '<table class="w-full border-collapse my-2">$1</table>');
    // Lists
    html = html.replace(/^- (.+)$/gm, '<li class="text-xs text-white/50 ml-3 list-disc list-inside leading-relaxed">$1</li>');
    html = html.replace(/((<li[^>]*>.*<\/li>\s*)+)/g, '<ul class="my-1">$1</ul>');
    // Line breaks
    html = html.replace(/\n\n/g, "<br>");
    html = html.replace(/\n/g, "<br>");
    return html;
}

// ---------------------------------------------------------------------------
//  Quickstart Cards — rendered in chat area
// ---------------------------------------------------------------------------
function renderQuickstartCards(scenarios) {
    if (!scenarios || !scenarios.length) return "";
    let html = `<div class="grid grid-cols-1 md:grid-cols-2 gap-3 mt-4">`;
    for (const s of scenarios) {
        const title = esc(s.business_title || s.title);
        const problem = esc(s.industry_problem || s.description);
        const focus = s.decision_focus ? `<p class="text-[10px] font-mono text-accent/30 mt-1.5 line-clamp-2">${esc(s.decision_focus)}</p>` : "";
        const icon = esc(s.icon || "flask-conical");
        html += `
        <button class="quickstart-card text-left rounded-xl bg-surface-850/60 border border-white/[0.06] p-4 hover:border-accent/30 hover:bg-surface-800/60 transition-all group cursor-pointer" data-id="${esc(s.id)}" data-prompt="${esc(s.prompt)}">
            <div class="flex items-start gap-3">
                <div class="w-8 h-8 rounded-lg bg-accent/10 border border-accent/15 flex items-center justify-center flex-shrink-0 mt-0.5">
                    <i data-lucide="${icon}" class="w-4 h-4 text-accent/60"></i>
                </div>
                <div class="flex-1 min-w-0">
                    <p class="text-[12px] font-semibold text-white/70 group-hover:text-white/90 transition-colors">${title}</p>
                    <p class="text-[11px] text-white/40 mt-1 line-clamp-2">${problem}</p>
                    ${focus}
                </div>
            </div>
        </button>`;
    }
    html += `</div>`;
    return html;
}

function renderWelcomeWithCards(scenarios) {
    const welcomeDiv = els.chatMessages.querySelector(".agent-message");
    if (!welcomeDiv) return;

    const contentDiv = welcomeDiv.querySelector(".flex-1");
    if (!contentDiv) return;

    contentDiv.innerHTML = `
        <p class="text-[11px] font-mono text-accent/50 mb-2">Triality Agent</p>
        <div class="text-sm text-white/60 leading-relaxed space-y-2">
            <p>I'm your physics analysis agent. I can <strong class="text-white/80">plan</strong>, <strong class="text-white/80">execute</strong>, and <strong class="text-white/80">analyse</strong> engineering problems autonomously.</p>
            <p class="text-white/40 text-xs">Choose a scenario below, or describe what you'd like to analyze. You can also ask me what I can do.</p>
        </div>
        ${renderQuickstartCards(scenarios)}
    `;

    // Wire click handlers on the cards
    contentDiv.querySelectorAll(".quickstart-card").forEach(btn => {
        btn.addEventListener("click", () => {
            const prompt = btn.dataset.prompt;
            const id = btn.dataset.id;
            els.promptInput.value = prompt;
            autoResize(els.promptInput);
            runAgent(prompt, id);
        });
    });

    lucide.createIcons();
}

// ---------------------------------------------------------------------------
//  Message Rendering
// ---------------------------------------------------------------------------
function addUserMessage(text) {
    const div = document.createElement("div");
    div.className = "flex justify-end";
    div.innerHTML = `
        <div class="max-w-[80%] rounded-2xl rounded-br-md bg-accent/10 border border-accent/15 px-4 py-3">
            <p class="text-sm text-white/75 leading-relaxed">${esc(text)}</p>
        </div>
    `;
    els.chatMessages.appendChild(div);
    scrollToBottom();
}

function createAgentMessage() {
    const wrapper = document.createElement("div");
    wrapper.className = "agent-message";
    wrapper.innerHTML = `
        <div class="flex items-start gap-3">
            <div class="w-7 h-7 rounded-lg bg-gradient-to-br from-accent/20 to-cyan-dim/20 border border-accent/15 flex items-center justify-center flex-shrink-0 mt-0.5">
                <i data-lucide="microscope" class="w-3.5 h-3.5 text-accent"></i>
            </div>
            <div class="flex-1 min-w-0 space-y-3" id="agent-content-${Date.now()}">
                <p class="text-[11px] font-mono text-accent/50">Triality Agent</p>
            </div>
        </div>
    `;
    els.chatMessages.appendChild(wrapper);
    lucide.createIcons();
    scrollToBottom();
    return wrapper.querySelector("[id^='agent-content-']");
}

function appendToAgent(container, html) {
    const div = document.createElement("div");
    div.innerHTML = html;
    container.appendChild(div);
    lucide.createIcons();
    scrollToBottom();
}

// ---------------------------------------------------------------------------
//  Tool Result Visualizations
// ---------------------------------------------------------------------------
function renderFieldHeatmap(fieldName, data, unit) {
    if (!Array.isArray(data) || !Array.isArray(data[0])) return "";
    const rows = data.length;
    const cols = data[0].length;
    const canvasId = `heatmap-${fieldName}-${Date.now()}`;

    // We'll render heatmaps after DOM insertion
    setTimeout(() => {
        const canvas = document.getElementById(canvasId);
        if (!canvas) return;
        const ctx = canvas.getContext("2d");
        const w = canvas.width;
        const h = canvas.height;

        // Find min/max
        let min = Infinity, max = -Infinity;
        for (let r = 0; r < rows; r++) {
            for (let c = 0; c < cols; c++) {
                const v = data[r][c];
                if (v < min) min = v;
                if (v > max) max = v;
            }
        }
        const range = max - min || 1;

        const cellW = w / cols;
        const cellH = h / rows;

        for (let r = 0; r < rows; r++) {
            for (let c = 0; c < cols; c++) {
                const t = (data[r][c] - min) / range;
                // Viridis-like: dark purple → teal → yellow
                const rr = Math.floor(255 * Math.min(1, t * 1.5));
                const g = Math.floor(255 * (t < 0.5 ? t * 1.5 : 0.75 + t * 0.25));
                const b = Math.floor(255 * (1 - t) * 0.8);
                ctx.fillStyle = `rgb(${rr}, ${g}, ${b})`;
                ctx.fillRect(c * cellW, r * cellH, cellW + 1, cellH + 1);
            }
        }

        // Colorbar labels
        ctx.fillStyle = "rgba(255,255,255,0.6)";
        ctx.font = "10px JetBrains Mono";
        ctx.fillText(formatNumber(min), 2, h - 3);
        ctx.textAlign = "right";
        ctx.fillText(formatNumber(max), w - 2, 12);
    }, 50);

    return `
        <div class="mt-2">
            <p class="text-[10px] font-mono text-white/30 mb-1">${esc(fieldName)} ${unit ? `(${esc(unit)})` : ""} — ${rows}x${cols}</p>
            <canvas id="${canvasId}" width="${Math.min(cols * 8, 400)}" height="${Math.min(rows * 8, 300)}" class="rounded-lg border border-white/[0.06] w-full max-w-[400px]" style="image-rendering: pixelated;"></canvas>
        </div>
    `;
}

function renderFieldLine(fieldName, data, unit) {
    if (!Array.isArray(data)) return "";
    const flat = Array.isArray(data[0]) ? data.flat() : data;
    if (flat.length < 2) return "";
    const canvasId = `line-${fieldName}-${Date.now()}`;

    setTimeout(() => {
        const canvas = document.getElementById(canvasId);
        if (!canvas) return;
        const ctx = canvas.getContext("2d");
        const w = canvas.width;
        const h = canvas.height;
        const pad = 20;

        let min = Infinity, max = -Infinity;
        for (const v of flat) { if (v < min) min = v; if (v > max) max = v; }
        const range = max - min || 1;

        ctx.strokeStyle = "#6ee7b7";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        for (let i = 0; i < flat.length; i++) {
            const x = pad + (i / (flat.length - 1)) * (w - 2 * pad);
            const y = h - pad - ((flat[i] - min) / range) * (h - 2 * pad);
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();

        // Axes labels
        ctx.fillStyle = "rgba(255,255,255,0.4)";
        ctx.font = "10px JetBrains Mono";
        ctx.fillText(formatNumber(min), 2, h - 5);
        ctx.fillText(formatNumber(max), 2, 12);
    }, 50);

    return `
        <div class="mt-2">
            <p class="text-[10px] font-mono text-white/30 mb-1">${esc(fieldName)} ${unit ? `(${esc(unit)})` : ""} — ${flat.length} pts</p>
            <canvas id="${canvasId}" width="400" height="160" class="rounded-lg border border-white/[0.06] w-full max-w-[400px]"></canvas>
        </div>
    `;
}

function renderSweepChart(sweepResult) {
    const results = sweepResult.results || [];
    if (results.length < 2) return "";

    const paramPath = sweepResult.param_path || "param";
    const fieldNames = Object.keys(results[0].fields || {});
    if (!fieldNames.length) return "";

    let chartsHtml = "";
    for (const fname of fieldNames.slice(0, 3)) {
        const canvasId = `sweep-${fname}-${Date.now()}`;
        const xVals = [];
        const yVals = [];
        for (const r of results) {
            if (r.success && r.fields && r.fields[fname]) {
                xVals.push(r.param_value);
                yVals.push(r.fields[fname].mean);
            }
        }
        if (xVals.length < 2) continue;

        setTimeout(() => {
            const canvas = document.getElementById(canvasId);
            if (!canvas) return;
            const ctx = canvas.getContext("2d");
            const w = canvas.width, h = canvas.height;
            const pad = 30;

            const xMin = Math.min(...xVals), xMax = Math.max(...xVals);
            const yMin = Math.min(...yVals), yMax = Math.max(...yVals);
            const xRange = xMax - xMin || 1, yRange = yMax - yMin || 1;

            // Grid
            ctx.strokeStyle = "rgba(255,255,255,0.05)";
            ctx.lineWidth = 0.5;
            for (let i = 0; i <= 4; i++) {
                const y = pad + (i / 4) * (h - 2 * pad);
                ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(w - pad, y); ctx.stroke();
            }

            // Line
            ctx.strokeStyle = "#22d3ee";
            ctx.lineWidth = 2;
            ctx.beginPath();
            for (let i = 0; i < xVals.length; i++) {
                const x = pad + ((xVals[i] - xMin) / xRange) * (w - 2 * pad);
                const y = h - pad - ((yVals[i] - yMin) / yRange) * (h - 2 * pad);
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.stroke();

            // Points
            ctx.fillStyle = "#22d3ee";
            for (let i = 0; i < xVals.length; i++) {
                const x = pad + ((xVals[i] - xMin) / xRange) * (w - 2 * pad);
                const y = h - pad - ((yVals[i] - yMin) / yRange) * (h - 2 * pad);
                ctx.beginPath(); ctx.arc(x, y, 3, 0, 2 * Math.PI); ctx.fill();
            }

            // Labels
            ctx.fillStyle = "rgba(255,255,255,0.4)";
            ctx.font = "10px JetBrains Mono";
            ctx.fillText(formatNumber(xMin), pad, h - 5);
            ctx.textAlign = "right";
            ctx.fillText(formatNumber(xMax), w - pad, h - 5);
            ctx.textAlign = "left";
            ctx.fillText(formatNumber(yMax), 2, pad + 4);
            ctx.fillText(formatNumber(yMin), 2, h - pad);
        }, 50);

        chartsHtml += `
            <div class="mt-2">
                <p class="text-[10px] font-mono text-white/30 mb-1">${esc(paramPath)} vs ${esc(fname)} (mean)</p>
                <canvas id="${canvasId}" width="400" height="180" class="rounded-lg border border-white/[0.06] w-full max-w-[400px]"></canvas>
            </div>
        `;
    }
    return chartsHtml;
}

function renderOptimizationChart(result) {
    const evals = result.evaluations || [];
    if (evals.length < 2) return "";
    const canvasId = `opt-${Date.now()}`;

    setTimeout(() => {
        const canvas = document.getElementById(canvasId);
        if (!canvas) return;
        const ctx = canvas.getContext("2d");
        const w = canvas.width, h = canvas.height;
        const pad = 30;

        const xVals = evals.map(e => e.param_value);
        const yVals = evals.map(e => e.metric);
        const xMin = Math.min(...xVals), xMax = Math.max(...xVals);
        const yMin = Math.min(...yVals), yMax = Math.max(...yVals);
        const xRange = xMax - xMin || 1, yRange = yMax - yMin || 1;

        // Points
        ctx.fillStyle = "rgba(110, 231, 183, 0.5)";
        for (let i = 0; i < xVals.length; i++) {
            const x = pad + ((xVals[i] - xMin) / xRange) * (w - 2 * pad);
            const y = h - pad - ((yVals[i] - yMin) / yRange) * (h - 2 * pad);
            ctx.beginPath(); ctx.arc(x, y, 3, 0, 2 * Math.PI); ctx.fill();
        }

        // Optimal point
        const optX = pad + ((result.optimal_param_value - xMin) / xRange) * (w - 2 * pad);
        const optY = h - pad - ((result.optimal_metric - yMin) / yRange) * (h - 2 * pad);
        ctx.fillStyle = "#6ee7b7";
        ctx.beginPath(); ctx.arc(optX, optY, 6, 0, 2 * Math.PI); ctx.fill();
        ctx.strokeStyle = "#6ee7b7";
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.arc(optX, optY, 10, 0, 2 * Math.PI); ctx.stroke();

        // Labels
        ctx.fillStyle = "rgba(255,255,255,0.4)";
        ctx.font = "10px JetBrains Mono";
        ctx.fillText(formatNumber(xMin), pad, h - 5);
        ctx.textAlign = "right";
        ctx.fillText(formatNumber(xMax), w - pad, h - 5);
    }, 50);

    return `
        <div class="mt-2">
            <p class="text-[10px] font-mono text-white/30 mb-1">Optimization: ${esc(result.param_path)} → ${esc(result.objective_field)}</p>
            <canvas id="${canvasId}" width="400" height="180" class="rounded-lg border border-white/[0.06] w-full max-w-[400px]"></canvas>
        </div>
    `;
}

function renderToolResult(toolName, result) {
    let content = "";

    if (toolName === "run_module") {
        const success = result.success;
        const module = result.module_name || "";
        content += `<div class="flex items-center gap-2 mb-2">
            <span class="cap-badge ${success ? "cap-full" : "cap-fail"}">${success ? "Success" : "Failed"}</span>
            <span class="text-[11px] font-mono text-white/40">${esc(module)} — ${formatNumber(result.elapsed_time_s || result._elapsed_s || 0)}s</span>
        </div>`;

        if (result.error) {
            content += `<p class="text-xs text-red-300/70">${esc(result.error)}</p>`;
        }

        // Field stats table
        const fields = result.fields_stats || {};
        const fieldEntries = Object.entries(fields);
        if (fieldEntries.length) {
            content += `<div class="grid grid-cols-2 md:grid-cols-3 gap-2 mt-2">`;
            for (const [fname, stats] of fieldEntries) {
                content += `<div class="metric-card rounded-lg p-2">
                    <p class="text-[9px] font-mono text-white/25 uppercase">${esc(fname)}</p>
                    <p class="text-[11px] text-white/60 mt-0.5">[${formatNumber(stats.min)}, ${formatNumber(stats.max)}]</p>
                    <p class="text-[10px] text-white/35">mean: ${formatNumber(stats.mean)}</p>
                </div>`;
            }
            content += `</div>`;
        }

        // Field visualizations
        const stateFields = result.state?.fields || {};
        for (const [fname, fdata] of Object.entries(stateFields)) {
            if (fdata.data) {
                if (Array.isArray(fdata.data[0])) {
                    content += renderFieldHeatmap(fname, fdata.data, fdata.unit);
                } else {
                    content += renderFieldLine(fname, fdata.data, fdata.unit);
                }
            }
        }

    } else if (toolName === "sweep_parameter") {
        const n = (result.results || []).length;
        content += `<div class="flex items-center gap-2 mb-2">
            <span class="cap-badge cap-full">Sweep</span>
            <span class="text-[11px] font-mono text-white/40">${esc(result.param_path)} — ${n} points</span>
        </div>`;
        content += renderSweepChart(result);

    } else if (toolName === "optimize_parameter") {
        content += `<div class="flex items-center gap-2 mb-2">
            <span class="cap-badge cap-full">Optimized</span>
            <span class="text-[11px] font-mono text-white/40">${esc(result.param_path)}</span>
        </div>`;
        content += `<div class="grid grid-cols-2 gap-2 mt-2">
            <div class="metric-card rounded-lg p-2">
                <p class="text-[9px] font-mono text-white/25 uppercase">Optimal Value</p>
                <p class="text-sm text-accent/80 mt-0.5">${formatNumber(result.optimal_param_value)}</p>
            </div>
            <div class="metric-card rounded-lg p-2">
                <p class="text-[9px] font-mono text-white/25 uppercase">Metric</p>
                <p class="text-sm text-cyan/80 mt-0.5">${formatNumber(result.optimal_metric)}</p>
            </div>
        </div>`;
        content += renderOptimizationChart(result);

    } else if (toolName === "compare_scenarios") {
        const comp = result.comparison || [];
        content += `<div class="flex items-center gap-2 mb-2">
            <span class="cap-badge cap-full">Compared</span>
            <span class="text-[11px] font-mono text-white/40">${comp.length} scenarios</span>
        </div>`;
        if (comp.length) {
            const allFields = new Set();
            comp.forEach(c => Object.keys(c.fields || {}).forEach(f => allFields.add(f)));
            for (const fname of [...allFields].slice(0, 3)) {
                content += `<p class="text-[10px] font-mono text-white/30 mt-2 mb-1">${esc(fname)}</p>`;
                content += `<div class="space-y-1">`;
                for (const c of comp) {
                    const fs = (c.fields || {})[fname];
                    if (!fs) continue;
                    const barWidth = Math.max(5, Math.min(100, Math.abs(fs.mean) / Math.max(...comp.map(x => Math.abs((x.fields?.[fname]?.mean) || 1))) * 100));
                    content += `<div class="flex items-center gap-2">
                        <span class="text-[10px] font-mono text-white/50 w-16 flex-shrink-0">${esc(c.label)}</span>
                        <div class="flex-1 h-3 bg-surface-800 rounded-full overflow-hidden">
                            <div class="h-full bg-gradient-to-r from-accent/40 to-cyan-dim/40 rounded-full" style="width: ${barWidth}%"></div>
                        </div>
                        <span class="text-[10px] font-mono text-white/40 w-20 text-right">${formatNumber(fs.mean)}</span>
                    </div>`;
                }
                content += `</div>`;
            }
        }

    } else if (toolName === "run_uncertainty_quantification") {
        const stats = result.statistics || {};
        content += `<div class="flex items-center gap-2 mb-2">
            <span class="cap-badge cap-full">UQ</span>
            <span class="text-[11px] font-mono text-white/40">${result.n_samples} samples</span>
        </div>`;
        for (const [key, s] of Object.entries(stats)) {
            content += `<div class="metric-card rounded-lg p-2 mt-1.5">
                <p class="text-[9px] font-mono text-white/25 uppercase">${esc(key)}</p>
                <p class="text-[11px] text-white/60 mt-0.5">${formatNumber(s.mean)} &plusmn; ${formatNumber(s.std)} <span class="text-white/30">(CV=${s.cv_percent.toFixed(1)}%)</span></p>
                <p class="text-[10px] text-white/30">90% CI: [${formatNumber(s.p5)}, ${formatNumber(s.p95)}]</p>
            </div>`;
        }

    } else if (toolName === "list_modules" || toolName === "describe_module") {
        content += `<pre class="text-[11px] font-mono text-white/45 bg-surface-900/40 rounded-lg p-3 overflow-auto max-h-60">${esc(JSON.stringify(result, null, 2))}</pre>`;

    } else if (toolName === "chain_modules") {
        const chain = result.chain_results || [];
        content += `<div class="flex items-center gap-2 mb-2">
            <span class="cap-badge cap-full">Chain</span>
            <span class="text-[11px] font-mono text-white/40">${chain.length} steps</span>
        </div>`;
        for (const step of chain) {
            const success = step.success !== false && !step.error;
            content += `<div class="mt-2 pl-3 border-l-2 ${success ? "border-accent/20" : "border-red-400/20"}">
                <p class="text-[10px] font-mono text-white/40">${esc(step.step_label || step.module_name || `step ${step.step}`)}</p>
                ${step.error ? `<p class="text-xs text-red-300/60">${esc(step.error)}</p>` : ""}
            </div>`;
        }

    } else {
        content += `<pre class="text-[11px] font-mono text-white/45 bg-surface-900/40 rounded-lg p-3 overflow-auto max-h-40">${esc(JSON.stringify(result, null, 2).slice(0, 2000))}</pre>`;
    }

    return content;
}

// ---------------------------------------------------------------------------
//  SSE Agent Communication
// ---------------------------------------------------------------------------
async function runAgent(prompt, scenarioId = null) {
    if (state.isRunning) return;

    const apiToken = els.tokenInput.value.trim();
    if (!apiToken) {
        els.tokenInput.focus();
        els.tokenInput.classList.add("ring", "ring-red-400/50");
        setTimeout(() => els.tokenInput.classList.remove("ring", "ring-red-400/50"), 2000);
        return;
    }

    state.isRunning = true;
    els.sendButton.disabled = true;
    setAgentStatus("Thinking...", true);

    addUserMessage(prompt);
    const container = createAgentMessage();

    const body = JSON.stringify({
        prompt,
        scenario_id: scenarioId,
        replicate_api_token: apiToken,
        llm_model: els.modelSelect.value,
    });

    try {
        const response = await fetch("/api/agent", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body,
        });

        if (!response.ok) {
            const err = await response.text();
            appendToAgent(container, `<p class="text-xs text-red-300/70">Error: ${esc(err)}</p>`);
            return;
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let currentToolDiv = null;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n");
            buffer = lines.pop() || "";

            let eventType = null;
            let eventData = null;

            for (const line of lines) {
                if (line.startsWith("event: ")) {
                    eventType = line.slice(7).trim();
                } else if (line.startsWith("data: ")) {
                    try {
                        eventData = JSON.parse(line.slice(6));
                    } catch { eventData = null; }

                    if (eventType && eventData) {
                        handleSSEEvent(eventType, eventData, container);
                        eventType = null;
                        eventData = null;
                    }
                }
            }
        }
    } catch (err) {
        appendToAgent(container, `<p class="text-xs text-red-300/70">Connection error: ${esc(err.message)}</p>`);
    } finally {
        state.isRunning = false;
        els.sendButton.disabled = false;
        setAgentStatus("Ready");
    }
}

function handleSSEEvent(type, data, container) {
    switch (type) {
        case "phase":
            setAgentStatus(data.message || data.phase, true);
            break;

        case "thinking": {
            const source = data.source === "llm" ? "LLM" : "Heuristic";
            let html = `<div class="thinking-block rounded-xl bg-surface-850/60 border border-white/[0.04] p-3">
                <div class="flex items-center gap-2 mb-2">
                    <i data-lucide="brain" class="w-3 h-3 text-accent/50"></i>
                    <span class="text-[10px] font-mono text-accent/40 uppercase">${source} Planning</span>
                </div>`;
            if (data.thinking) {
                html += `<p class="text-xs text-white/40 italic mb-2">${esc(data.thinking)}</p>`;
            }
            if (data.plan && data.plan.length) {
                html += `<div class="space-y-1">`;
                data.plan.forEach((step, i) => {
                    html += `<div class="flex items-center gap-2">
                        <span class="text-[10px] font-mono text-accent/30">${i + 1}.</span>
                        <span class="text-[11px] text-white/45">${esc(step)}</span>
                    </div>`;
                });
                html += `</div>`;
            }
            html += `</div>`;
            appendToAgent(container, html);
            break;
        }

        case "tool_start": {
            setAgentStatus(`Running ${data.tool}...`, true);
            const toolId = `tool-${data.index}-${Date.now()}`;
            const html = `<div id="${toolId}" class="tool-block rounded-xl bg-surface-900/50 border border-white/[0.04] overflow-hidden">
                <button class="tool-header w-full flex items-center justify-between px-3 py-2 hover:bg-white/[0.02] transition-colors" onclick="this.parentElement.querySelector('.tool-body').classList.toggle('open')">
                    <div class="flex items-center gap-2">
                        <div class="w-4 h-4 rounded bg-accent/10 flex items-center justify-center">
                            <i data-lucide="play" class="w-2.5 h-2.5 text-accent/60"></i>
                        </div>
                        <span class="text-[11px] font-mono text-white/50">${esc(data.tool)}</span>
                        <span class="tool-status text-[10px] font-mono text-yellow-400/50">running...</span>
                    </div>
                    <i data-lucide="chevron-down" class="w-3 h-3 text-white/20 transition-transform"></i>
                </button>
                <div class="tool-body px-3 pb-3">
                    <div class="tool-content">
                        <p class="text-[10px] font-mono text-white/20 mb-1">Args:</p>
                        <pre class="text-[10px] font-mono text-white/30 bg-surface-950/50 rounded-lg p-2 overflow-auto max-h-24">${esc(JSON.stringify(data.args, null, 2))}</pre>
                    </div>
                </div>
            </div>`;
            appendToAgent(container, html);
            break;
        }

        case "tool_result": {
            setAgentStatus(`Completed ${data.tool}`, true);
            // Find the tool block and update it
            const toolBlocks = container.querySelectorAll(".tool-block");
            const block = toolBlocks[toolBlocks.length - 1];
            if (block) {
                const statusEl = block.querySelector(".tool-status");
                if (statusEl) {
                    statusEl.textContent = data.success ? `${data.elapsed_s}s` : "failed";
                    statusEl.className = `tool-status text-[10px] font-mono ${data.success ? "text-accent/50" : "text-red-400/50"}`;
                }
                const icon = block.querySelector(".tool-header i[data-lucide='play']");
                if (icon) {
                    icon.setAttribute("data-lucide", data.success ? "check" : "x");
                    lucide.createIcons();
                }
                // Add result visualization
                const contentDiv = block.querySelector(".tool-content");
                if (contentDiv) {
                    const resultHtml = renderToolResult(data.tool, data.result);
                    const resultDiv = document.createElement("div");
                    resultDiv.className = "mt-2 pt-2 border-t border-white/[0.04]";
                    resultDiv.innerHTML = resultHtml;
                    contentDiv.appendChild(resultDiv);
                    lucide.createIcons();
                }
                // Auto-open if it has visualizations
                const body = block.querySelector(".tool-body");
                if (body) body.classList.add("open");
            }
            scrollToBottom();
            break;
        }

        case "summary": {
            const html = `<div class="summary-block mt-2">
                <div class="text-sm text-white/60 leading-relaxed prose-sm">
                    ${renderMarkdown(data.summary)}
                </div>
            </div>`;
            appendToAgent(container, html);
            break;
        }

        case "reflection": {
            const html = `<div class="flex items-center gap-2 mt-2">
                <i data-lucide="sparkles" class="w-3 h-3 text-cyan/50"></i>
                <span class="text-[10px] font-mono text-cyan/40">Reflecting: ${esc(data.reason)}</span>
            </div>`;
            appendToAgent(container, html);
            break;
        }

        case "reflection_addendum": {
            if (data.addendum) {
                const html = `<div class="text-xs text-white/40 italic mt-1 pl-5">${esc(data.addendum)}</div>`;
                appendToAgent(container, html);
            }
            break;
        }

        case "turn_complete": {
            const badge = data.all_succeeded ? "cap-full" : "cap-fail";
            const label = data.all_succeeded ? "Complete" : "Partial";
            const html = `<div class="flex items-center gap-2 mt-3 pt-2 border-t border-white/[0.04]">
                <span class="cap-badge ${badge}">${label}</span>
                <span class="text-[10px] font-mono text-white/25">${data.total_tools} tool(s) executed</span>
            </div>`;
            appendToAgent(container, html);
            break;
        }

        case "chitchat": {
            const html = `<div class="summary-block">
                <div class="text-sm text-white/60 leading-relaxed prose-sm">
                    ${renderMarkdown(data.response)}
                </div>
            </div>`;
            appendToAgent(container, html);
            break;
        }

        case "error": {
            appendToAgent(container, `<p class="text-xs text-red-300/70">Agent error: ${esc(data.error)}</p>`);
            break;
        }
    }
    scrollToBottom();
}

// ---------------------------------------------------------------------------
//  Catalog & Sidebar
// ---------------------------------------------------------------------------
async function fetchCatalog() {
    try {
        const resp = await fetch("/api/catalog");
        const catalog = await resp.json();
        state.catalog = catalog;
        renderSidebar(catalog);
        renderWelcomeWithCards(catalog.scenarios || []);
        els.footerStatus.textContent = `${Object.keys(catalog.modules || {}).length} modules · ${(catalog.tools || []).length} tools`;
    } catch (err) {
        els.footerStatus.textContent = "Catalog failed";
    }
}

function renderSidebar(catalog) {
    // Scenarios
    els.scenarioList.innerHTML = (catalog.scenarios || []).map(s => `
        <button class="scenario-card w-full text-left rounded-lg px-3 py-2.5 cursor-pointer group" data-id="${esc(s.id)}" data-prompt="${esc(s.prompt)}">
            <div class="flex items-center justify-between">
                <div class="flex-1 min-w-0">
                    <p class="text-[12px] font-medium text-white/60 group-hover:text-white/85 transition-colors truncate">${esc(s.title)}</p>
                    <p class="text-[10px] font-mono text-accent/30 mt-0.5">${esc(s.subtitle)}</p>
                </div>
                <i data-lucide="chevron-right" class="w-3 h-3 text-white/15 group-hover:text-accent/50 transition-colors flex-shrink-0"></i>
            </div>
        </button>
    `).join("");

    els.scenarioList.querySelectorAll(".scenario-card").forEach(btn => {
        btn.addEventListener("click", () => {
            const prompt = btn.dataset.prompt;
            const id = btn.dataset.id;
            els.promptInput.value = prompt;
            autoResize(els.promptInput);
            // Auto-run
            runAgent(prompt, id);
        });
    });

    // Module pills
    els.modulePills.innerHTML = Object.keys(catalog.modules || {}).map(m => `
        <span class="pill rounded-full px-2 py-0.5 text-[10px] font-mono text-accent/50">${esc(m)}</span>
    `).join("");

    // Tool pills
    els.toolPills.innerHTML = (catalog.tools || []).map(t => `
        <span class="pill rounded-full px-2 py-0.5 text-[10px] font-mono text-cyan/50">${esc(t)}</span>
    `).join("");

    // Capabilities
    els.capabilitiesList.innerHTML = (catalog.capabilities || []).map(c => `
        <div class="flex items-center gap-2 py-1">
            <i data-lucide="${esc(c.icon)}" class="w-3 h-3 text-accent/40"></i>
            <span class="text-[10px] text-white/40">${esc(c.name)}</span>
            <span class="cap-badge ${c.status === 'Full' ? 'cap-full' : 'cap-partial'} text-[8px] ml-auto">${esc(c.status)}</span>
        </div>
    `).join("");

    lucide.createIcons();
}

// ---------------------------------------------------------------------------
//  Event Wiring
// ---------------------------------------------------------------------------
function wireEvents() {
    els.sendButton.addEventListener("click", () => {
        const prompt = els.promptInput.value.trim();
        if (prompt && !state.isRunning) {
            els.promptInput.value = "";
            autoResize(els.promptInput);
            runAgent(prompt);
        }
    });

    els.promptInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            const prompt = els.promptInput.value.trim();
            if (prompt && !state.isRunning) {
                els.promptInput.value = "";
                autoResize(els.promptInput);
                runAgent(prompt);
            }
        }
    });

    els.promptInput.addEventListener("input", () => autoResize(els.promptInput));

    document.getElementById("toggle-password").addEventListener("click", () => {
        const hidden = els.tokenInput.type === "password";
        els.tokenInput.type = hidden ? "text" : "password";
    });

    document.getElementById("clear-chat").addEventListener("click", () => {
        els.chatMessages.innerHTML = `
            <div class="agent-message">
                <div class="flex items-start gap-3">
                    <div class="w-7 h-7 rounded-lg bg-gradient-to-br from-accent/20 to-cyan-dim/20 border border-accent/15 flex items-center justify-center flex-shrink-0 mt-0.5">
                        <i data-lucide="microscope" class="w-3.5 h-3.5 text-accent"></i>
                    </div>
                    <div class="flex-1 min-w-0">
                        <p class="text-[11px] font-mono text-accent/50 mb-2">Triality Agent</p>
                        <p class="text-sm text-white/50">Chat cleared. Ready for new analyses.</p>
                    </div>
                </div>
            </div>
        `;
        lucide.createIcons();
        if (state.catalog && state.catalog.scenarios) {
            renderWelcomeWithCards(state.catalog.scenarios);
        }
    });
}

// ---------------------------------------------------------------------------
//  Boot
// ---------------------------------------------------------------------------
async function boot() {
    try {
        wireEvents();
        autoResize(els.promptInput);
        await fetchCatalog();
        setAgentStatus("Ready");
    } catch (err) {
        console.error("Boot failed:", err);
        els.footerStatus.textContent = "Boot failed — check console";
    }
    lucide.createIcons();
}

boot();
