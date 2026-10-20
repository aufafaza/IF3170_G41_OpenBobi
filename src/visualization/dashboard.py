from __future__ import annotations

import html
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

from .charts import group_colors, overlay_figure, overview_figures, run_figures
from .data import ALGORITHM_ORDER, ALGORITHMS, Dataset, Run, aggregate, load_dataset
from .scene import build_state_figure

CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

:root {
  --font-sans: "IBM Plex Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  --font-display: "IBM Plex Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  --font-mono: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;

  --text-2xs: 0.6875rem;
  --text-xs: 0.75rem;
  --text-sm: 0.8125rem;
  --text-base: 0.9375rem;
  --text-md: 1rem;
  --text-lg: 1.125rem;
  --text-xl: 1.375rem;
  --text-2xl: 1.75rem;
  --text-3xl: 2.125rem;

  --ink: #0f172a;
  --muted: #475569;
  --base: #f1f5f9;
  --panel: #ffffff;
  --line: #cbd5e1;
  --ok: #0d7644;
  --bad: #b91c1c;
  --focus: #2563eb;
  color-scheme: light;
}

*, *::before, *::after {
  box-sizing: border-box;
}

::selection {
  background: #cbd5e1;
  color: var(--ink);
}

::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}
::-webkit-scrollbar-track {
  background: var(--base);
}
::-webkit-scrollbar-thumb {
  background: #cbd5e1;
  border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
  background: #94a3b8;
}

body {
  margin: 0;
  background: var(--base);
  color: var(--ink);
  font-family: var(--font-sans);
  font-size: var(--text-base);
  line-height: 1.6;
  text-rendering: optimizeLegibility;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

.wrap {
  max-width: 1360px;
  margin: 0 auto;
  padding: 0 28px;
}

header.top {
  border-bottom: 2px solid var(--line);
  background: var(--panel);
  padding: 32px 0 24px;
}
header.top h1 {
  margin: 0 0 6px;
  font-family: var(--font-display);
  font-size: var(--text-3xl);
  font-weight: 800;
  line-height: 1.15;
  letter-spacing: -0.03em;
  color: var(--ink);
}
header.top p {
  margin: 0;
  font-size: var(--text-base);
  line-height: 1.5;
  color: var(--muted);
  max-width: 68ch;
}

.spec {
  display: flex;
  flex-wrap: wrap;
  gap: 0;
  margin-top: 20px;
  border: 1px solid var(--line);
  background: #f8fafc;
}
.spec div {
  padding: 10px 18px;
  border-right: 1px solid var(--line);
  min-width: 150px;
}
.spec div:last-child {
  border-right: 0;
}
.spec dt {
  font-size: var(--text-xs);
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--muted);
  margin-bottom: 2px;
}
.spec dd {
  margin: 0;
  font-family: var(--font-sans);
  font-size: var(--text-base);
  font-weight: 700;
  color: var(--ink);
  font-variant-numeric: tabular-nums;
}

.warn {
  margin: 16px 0 0;
  padding: 12px 16px;
  border: 1px solid #fecaca;
  border-radius: 6px;
  background: #fef2f2;
  font-size: var(--text-sm);
  color: var(--bad);
  line-height: 1.5;
}
.warn strong {
  font-weight: 700;
}
.warn ul {
  margin: 6px 0 0;
  padding-left: 20px;
}

nav.tabs {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 20px;
  border-bottom: 1px solid var(--line);
}
.tab {
  appearance: none;
  background: none;
  border: 0;
  border-bottom: 3px solid transparent;
  margin-bottom: -1px;
  padding: 12px 18px;
  font-family: var(--font-sans);
  font-size: var(--text-sm);
  font-weight: 600;
  letter-spacing: -0.01em;
  color: var(--muted);
  cursor: pointer;
  transition: color 0.15s ease, border-color 0.15s ease;
}
.tab:hover {
  color: var(--ink);
}
.tab[aria-selected="true"] {
  color: var(--ink);
  border-bottom-color: var(--accent, var(--ink));
}
.tab:focus-visible, .pick select:focus-visible, tbody tr:focus-visible {
  outline: 2px solid var(--focus);
  outline-offset: 2px;
}

main {
  padding: 24px 0 64px;
}
.panel[hidden] {
  display: none;
}

.bar {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 16px;
}
.bar h2 {
  margin: 0;
  font-family: var(--font-display);
  font-size: var(--text-xl);
  font-weight: 700;
  letter-spacing: -0.02em;
  line-height: 1.25;
  color: var(--ink);
}

.pick {
  display: flex;
  flex-direction: column;
  font-size: var(--text-xs);
  font-weight: 500;
  color: var(--muted);
  gap: 4px;
}
.pick select {
  font-family: var(--font-sans);
  font-size: var(--text-sm);
  font-weight: 500;
  color: var(--ink);
  padding: 8px 12px;
  min-width: 320px;
  background: #fff;
  border: 1px solid var(--line);
  border-radius: 6px;
  cursor: pointer;
}

.block {
  background: var(--panel);
  border: 1px solid var(--line);
  padding: 18px 20px 14px;
  margin-bottom: 20px;
}
.block h3 {
  margin: 0 0 6px;
  font-family: var(--font-display);
  font-size: var(--text-md);
  font-weight: 700;
  letter-spacing: -0.01em;
  color: var(--ink);
}
.hint {
  margin: 0 0 10px;
  color: var(--muted);
  font-size: var(--text-sm);
  line-height: 1.5;
  max-width: 68ch;
}

.badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 14px;
  font-weight: 600;
  font-size: var(--text-xs);
  letter-spacing: 0.02em;
  margin-bottom: 8px;
}
.badge.bad {
  background: #fef2f2;
  color: var(--bad);
  border: 1px solid #fecaca;
}
.run-note {
  margin: 0 0 12px;
  font-size: var(--text-sm);
  color: var(--muted);
  line-height: 1.5;
}
.issues {
  margin: 4px 0 16px;
  padding-left: 22px;
  color: var(--bad);
  font-size: var(--text-sm);
  line-height: 1.5;
}

.metrics {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(190px, 1fr));
  gap: 0;
  margin: 10px 0 20px;
  border: 1px solid var(--line);
  background: var(--panel);
}
.metrics div {
  padding: 10px 16px;
  border-right: 1px solid var(--line);
  border-bottom: 1px solid var(--line);
}
.metrics dt {
  font-size: var(--text-xs);
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--muted);
  margin-bottom: 2px;
}
.metrics dd {
  margin: 0;
  font-family: var(--font-sans);
  font-weight: 700;
  font-size: var(--text-md);
  color: var(--ink);
  font-variant-numeric: tabular-nums;
  overflow-wrap: anywhere;
}

.plot {
  width: 100%;
}
.plot.tall {
  min-height: 620px;
}
.empty {
  padding: 40px 16px;
  color: var(--muted);
  font-style: italic;
  text-align: center;
}
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(520px, 1fr));
  gap: 20px;
}

.tablewrap {
  overflow-x: auto;
}
table {
  border-collapse: collapse;
  width: 100%;
  font-variant-numeric: tabular-nums;
  font-size: var(--text-sm);
  border: 1px solid var(--line);
}
th, td {
  text-align: left;
  padding: 9px 14px;
  border-bottom: 1px solid var(--line);
  white-space: nowrap;
}
th {
  font-size: var(--text-xs);
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--muted);
  background: #f8fafc;
}
td.num, th.num {
  text-align: right;
  font-variant-numeric: tabular-nums;
}
tbody tr[data-index] {
  cursor: pointer;
  transition: background-color 0.1s ease;
}
tbody tr[data-index]:hover {
  background: #f1f5f9;
}
tbody tr.selected {
  background: #e2e8f0;
  background: color-mix(in srgb, var(--accent, var(--ink)) 12%, #ffffff);
  font-weight: 600;
}
tbody tr.selected:hover {
  background: color-mix(in srgb, var(--accent, var(--ink)) 18%, #ffffff);
}

.cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 20px;
  margin-bottom: 20px;
}
.cards .block h3 {
  font-family: var(--font-display);
  font-size: var(--text-md);
  font-weight: 700;
  letter-spacing: -0.01em;
  padding-left: 0;
  margin-bottom: 8px;
  display: flex;
  align-items: center;
  gap: 8px;
}
.cards .block h3::before {
  content: "";
  display: inline-block;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--accent, var(--ink));
}
.cards dl {
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 4px 14px;
  margin: 10px 0 4px;
}
.cards dt {
  color: var(--muted);
  font-size: var(--text-xs);
  font-weight: 500;
}
.cards dd {
  margin: 0;
  text-align: right;
  font-weight: 700;
  font-size: var(--text-sm);
  color: var(--ink);
  font-variant-numeric: tabular-nums;
}

@media (max-width: 720px) {
  .wrap { padding: 0 16px; }
  .grid { grid-template-columns: 1fr; }
  .pick select { min-width: 0; width: 100%; }
  header.top h1 { font-size: var(--text-2xl); }
}
"""

JS = """
(function () {
  "use strict";
  var DATA = JSON.parse(document.getElementById("dashboard-data").textContent);
  var CONFIG = { responsive: true, displaylogo: false };
  var tabs = Array.prototype.slice.call(document.querySelectorAll(".tab"));

  function byId(id) { return document.getElementById(id); }
  function purge(node) {
    if (node && node.classList && node.classList.contains("js-plotly-plot")) { Plotly.purge(node); }
  }
  function clear(node) { purge(node); node.replaceChildren(); }
  function draw(node, fig, emptyText) {
    clear(node);
    if (!fig) {
      var note = document.createElement("div");
      note.className = "empty";
      note.textContent = emptyText;
      node.appendChild(note);
      return;
    }
    Plotly.react(node, fig.data, fig.layout, CONFIG);
  }

  function freeAlgorithm(key) {
    clear(byId(key + "-state"));
    clear(byId(key + "-charts"));
    clear(byId(key + "-overlay"));
  }

  function renderRun(key, index) {
    var alg = DATA.algorithms[key];
    var run = alg.runs[index];
    byId(key + "-select").value = String(index);

    var metrics = byId(key + "-metrics");
    metrics.replaceChildren();
    run.metrics.forEach(function (pair) {
      var cell = document.createElement("div");
      var dt = document.createElement("dt");
      var dd = document.createElement("dd");
      dt.textContent = pair[0];
      dd.textContent = pair[1];
      cell.appendChild(dt);
      cell.appendChild(dd);
      metrics.appendChild(cell);
    });

    var badge = byId(key + "-badge");
    badge.replaceChildren();
    if (run.issues.length) {
      var tag = document.createElement("span");
      tag.className = "badge bad";
      tag.textContent = "Ada " + run.issues.length + " masalah validasi";
      badge.appendChild(tag);
      var list = document.createElement("ul");
      list.className = "issues";
      run.issues.forEach(function (text) {
        var item = document.createElement("li");
        item.textContent = text;
        list.appendChild(item);
      });
      badge.appendChild(list);
    } else {
      var note = document.createElement("p");
      note.className = "run-note";
      note.textContent = "Catatan: Semua constraint terpenuhi dan catatan run konsisten (diverifikasi ulang dari file solusi)";
      badge.appendChild(note);
    }

    draw(byId(key + "-state"), run.state, "File state awal/akhir untuk run ini tidak ditemukan.");

    var charts = byId(key + "-charts");
    clear(charts);
    run.charts.forEach(function (fig) {
      var holder = document.createElement("div");
      holder.className = "block";
      var target = document.createElement("div");
      target.className = "plot";
      holder.appendChild(target);
      charts.appendChild(holder);
      draw(target, fig, "");
    });

    var rows = byId(key + "-table").querySelectorAll("tbody tr");
    Array.prototype.forEach.call(rows, function (row) {
      row.classList.toggle("selected", row.getAttribute("data-index") === String(index));
    });
  }

  function activate(id) {
    tabs.forEach(function (tab) {
      var selected = tab.getAttribute("data-tab") === id;
      tab.setAttribute("aria-selected", selected ? "true" : "false");
      tab.tabIndex = selected ? 0 : -1;
    });
    Array.prototype.forEach.call(document.querySelectorAll(".panel"), function (panel) {
      panel.hidden = panel.id !== "panel-" + id;
    });
    DATA.order.forEach(function (key) { if (key !== id) { freeAlgorithm(key); } });

    if (id === "overview") {
      draw(byId("overview-final"), DATA.overview.final, "Data tidak tersedia.");
      draw(byId("overview-duration"), DATA.overview.duration, "Data durasi tidak tersedia.");
      return;
    }
    var alg = DATA.algorithms[id];
    renderRun(id, alg.selected);
    draw(byId(id + "-overlay"), alg.overlay, "Riwayat objective tidak tersedia.");
  }

  DATA.order.forEach(function (key) {
    var alg = DATA.algorithms[key];
    alg.selected = 0;
    byId(key + "-select").addEventListener("change", function (event) {
      alg.selected = Number(event.target.value);
      renderRun(key, alg.selected);
    });
    Array.prototype.forEach.call(byId(key + "-table").querySelectorAll("tbody tr"), function (row) {
      function choose() {
        alg.selected = Number(row.getAttribute("data-index"));
        renderRun(key, alg.selected);
        var panel = byId("panel-" + key);
        if (panel.scrollIntoView) { panel.scrollIntoView({ block: "start" }); }
      }
      row.addEventListener("click", choose);
      row.addEventListener("keydown", function (event) {
        if (event.key === "Enter" || event.key === " ") { event.preventDefault(); choose(); }
      });
    });
  });

  tabs.forEach(function (tab, position) {
    tab.addEventListener("click", function () { activate(tab.getAttribute("data-tab")); });
    tab.addEventListener("keydown", function (event) {
      var target = null;
      if (event.key === "ArrowRight") { target = tabs[(position + 1) % tabs.length]; }
      else if (event.key === "ArrowLeft") { target = tabs[(position - 1 + tabs.length) % tabs.length]; }
      else if (event.key === "Home") { target = tabs[0]; }
      else if (event.key === "End") { target = tabs[tabs.length - 1]; }
      if (target) {
        event.preventDefault();
        target.focus();
        activate(target.getAttribute("data-tab"));
      }
    });
  });
  activate("overview");
})();
"""

def _fig_dict(fig: Optional[go.Figure]) -> Optional[Dict[str, Any]]:
    return json.loads(fig.to_json()) if fig is not None else None

def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)

def _num(value: Optional[float], spec: str = "g", suffix: str = "") -> str:
    return "-" if value is None else f"{value:{spec}}{suffix}"

def _json_for_script(payload: Dict[str, Any]) -> str:
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return text.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")

def _gap(upper: Optional[float], value: float) -> str:
    if upper is None or upper <= 0:
        return "-"
    return f"{(upper - value) / upper * 100:.2f}%"

def _upper_bound(dataset: Dataset) -> Optional[float]:
    from ..experiments.upper_bound import bound_from_items

    if any(name != "default" for name in dataset.objectives):
        return None
    reference = next((r.final or r.initial for r in dataset.runs if r.final or r.initial), None)
    if reference is None:
        return None
    items = [
        (p.value, p.weight, p.dimensions[0] * p.dimensions[1] * p.dimensions[2])
        for p in reference.placed + reference.outside
    ]
    width, depth, height = reference.truck_dims
    return bound_from_items(items, reference.max_capacity, width * depth * height)

def _run_payload(run: Run, color: str) -> Dict[str, Any]:
    state = build_state_figure([("State awal", run.initial), ("State akhir", run.final)])
    return {
        "label": run.label,
        "group": run.group,
        "metrics": [list(pair) for pair in run.metrics],
        "issues": run.issues,
        "state": _fig_dict(state),
        "charts": [_fig_dict(fig) for fig in run_figures(run, color)],
    }

def _runs_table(key: str, runs: List[Run]) -> str:
    head = (
        "<thead><tr><th>Run</th><th class=\"num\">Value awal</th><th class=\"num\">Value akhir</th>"
        "<th class=\"num\">Durasi (detik)</th><th class=\"num\">Iterasi</th><th>Validasi</th></tr></thead>"
    )
    body = []
    for index, run in enumerate(runs):
        status = "Valid" if not run.issues else f"{len(run.issues)} masalah"
        body.append(
            f'<tr data-index="{index}" tabindex="0"><td>{_esc(run.label)}</td>'
            f'<td class="num">{_num(run.initial_value)}</td><td class="num">{_num(run.final_value)}</td>'
            f'<td class="num">{_num(run.duration, ".4f")}</td>'
            f'<td class="num">{_num(run.iterations)}</td><td>{status}</td></tr>'
        )
    return f'<table id="{key}-table">{head}<tbody>{"".join(body)}</tbody></table>'

def _select_options(runs: List[Run]) -> str:
    parts: List[str] = []
    current: Optional[str] = None
    for index, run in enumerate(runs):
        if run.group != current:
            if current is not None:
                parts.append("</optgroup>")
            parts.append(f'<optgroup label="{_esc(run.group)}">')
            current = run.group
        parts.append(f'<option value="{index}">{_esc(run.label)}</option>')
    parts.append("</optgroup>")
    return "".join(parts)

def _algorithm_panel(key: str, runs: List[Run], color: str) -> str:
    name = ALGORITHMS[key]["name"]
    return f"""
<section class="panel" role="tabpanel" id="panel-{key}" aria-labelledby="tab-{key}" style="--accent:{color}" hidden>
  <div class="bar">
    <h2>{_esc(name)}</h2>
    <label class="pick">Pilih run<select id="{key}-select">{_select_options(runs)}</select></label>
  </div>
  <div id="{key}-badge"></div>
  <dl class="metrics" id="{key}-metrics"></dl>
  <div class="block">
    <h3>State awal dan state akhir</h3>
    <p class="hint">Seret untuk memutar, scroll untuk zoom, arahkan kursor ke paket untuk detailnya. Paket di luar truk diletakkan di area sebelah kanan peti kemas.</p>
    <div class="plot tall" id="{key}-state"></div>
  </div>
  <div id="{key}-charts"></div>
  <div class="block">
    <h3>Perbandingan semua run</h3>
    <div class="plot" id="{key}-overlay"></div>
  </div>
  <div class="block">
    <h3>Daftar run</h3>
    <p class="hint">Klik baris untuk menampilkan run tersebut.</p>
    <div class="tablewrap">{_runs_table(key, runs)}</div>
  </div>
</section>"""

def _overview_panel(dataset: Dataset, upper: Optional[float]) -> str:
    cards = []
    for key in ALGORITHM_ORDER:
        runs = [r for r in dataset.runs if r.key == key]
        if not runs:
            continue
        finals = [r.final_value for r in runs]
        mean_final = sum(finals) / len(finals)
        durations = [r.duration for r in runs if r.duration is not None]
        mean_duration = sum(durations) / len(durations) if durations else None
        gap_row = (
            f"<dt>Gap rata-rata ke batas atas</dt><dd>{_gap(upper, mean_final)}</dd>"
            if upper is not None else ""
        )
        cards.append(
            f'<div class="block" style="--accent:{dataset.colors[key]}"><h3>{_esc(ALGORITHMS[key]["name"])}</h3>'
            f"<dl><dt>Jumlah run</dt><dd>{len(runs)}</dd>"
            f"<dt>Value akhir terbaik</dt><dd>{max(finals):g}</dd>"
            f"<dt>Rata-rata value akhir</dt><dd>{mean_final:.2f}</dd>"
            f"{gap_row}"
            f"<dt>Rata-rata durasi</dt><dd>{_num(mean_duration, '.4f', ' detik')}</dd></dl></div>"
        )

    rows = []
    for stat in aggregate(dataset.runs):
        rows.append(
            f"<tr><td>{_esc(ALGORITHMS[stat.key]['name'])}</td><td>{_esc(stat.group)}</td>"
            f'<td class="num">{stat.count}</td><td class="num">{stat.mean:.2f}</td>'
            f'<td class="num">{stat.std:.2f}</td><td class="num">{stat.minimum:g}</td>'
            f'<td class="num">{stat.maximum:g}</td>'
            f'<td class="num">{_gap(upper, stat.mean)}</td>'
            f'<td class="num">{_num(stat.mean_duration, ".4f")}</td>'
            f'<td class="num">{_num(stat.mean_iterations, ".1f")}</td></tr>'
        )
    table = (
        "<table><thead><tr><th>Algoritma</th><th>Konfigurasi</th><th class=\"num\">Run</th>"
        "<th class=\"num\">Rata-rata value</th><th class=\"num\">Simpangan baku</th>"
        "<th class=\"num\">Minimum</th><th class=\"num\">Maksimum</th>"
        "<th class=\"num\">Gap ke batas atas</th>"
        "<th class=\"num\">Rata-rata durasi (detik)</th><th class=\"num\">Rata-rata iterasi</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )
    bound_hint = (
        " Gap = (batas atas - rata-rata value) / batas atas. Batas atas berasal dari relaksasi knapsack "
        "(berat dan volume total), jadi bukan jaminan optimum geometri."
        if upper is not None else ""
    )
    return f"""
<section class="panel" role="tabpanel" id="panel-overview" aria-labelledby="tab-overview">
  <div class="bar">
    <h2>Ringkasan Hasil Eksperimen</h2>
  </div>
  <div class="cards">{''.join(cards)}</div>
  <div class="block"><div class="plot" id="overview-final"></div></div>
  <div class="block"><div class="plot" id="overview-duration"></div></div>
  <div class="block">
    <h3>Ringkasan per konfigurasi</h3>
    <p class="hint">Simpangan baku memakai n-1. Nilai kecil berarti hasil konsisten antar run.{bound_hint}</p>
    <div class="tablewrap">{table}</div>
  </div>
</section>"""

def _header(dataset: Dataset, upper: Optional[float]) -> str:
    truck = next((r.final or r.initial for r in dataset.runs if r.final or r.initial), None)
    spec = ""
    if truck is not None:
        w, d, h = truck.truck_dims
        total = len(truck.placed) + len(truck.outside)
        bound = f"<div><dt>Batas atas value</dt><dd>{upper:g}</dd></div>" if upper is not None else ""
        spec = (
            '<dl class="spec">'
            f"<div><dt>Truk</dt><dd>{_esc(truck.truck_label)}</dd></div>"
            f"<div><dt>Dimensi peti kemas (w x l x h)</dt><dd>{w:g} x {d:g} x {h:g}</dd></div>"
            f"<div><dt>Kapasitas maksimum</dt><dd>{truck.max_capacity:g}</dd></div>"
            f"<div><dt>Jumlah paket</dt><dd>{total}</dd></div>{bound}</dl>"
        )
    warn = ""
    if dataset.warnings:
        items = "".join(f"<li>{_esc(w)}</li>" for w in dataset.warnings)
        warn = f'<div class="warn"><strong>Beberapa data dilewati</strong><ul>{items}</ul></div>'
    return (
        '<header class="top"><div class="wrap"><h1>Visualisasi hasil local search</h1>'
        f"<p>Pengisian peti kemas truk: Hill-climbing, Simulated Annealing, dan Genetic Algorithm.</p>{spec}{warn}</div></header>"
    )

def build_dashboard(output_dir: str, out_path: Optional[str] = None) -> Tuple[str, List[str]]:
    dataset = load_dataset(output_dir)
    keys = [k for k in ALGORITHM_ORDER if any(r.key == k for r in dataset.runs)]
    upper = _upper_bound(dataset)

    algorithms: Dict[str, Any] = {}
    panels: List[str] = []
    for key in keys:
        runs = [r for r in dataset.runs if r.key == key]
        color = dataset.colors[key]
        algorithms[key] = {
            "runs": [_run_payload(run, color) for run in runs],
            "overlay": _fig_dict(overlay_figure(runs, group_colors(runs))),
        }
        panels.append(_algorithm_panel(key, runs, color))

    overview = overview_figures(aggregate(dataset.runs), dataset.colors, upper)
    payload = {
        "order": keys,
        "algorithms": algorithms,
        "overview": {name: _fig_dict(fig) for name, fig in overview.items()},
    }

    tab_buttons = [
        '<button class="tab" role="tab" id="tab-overview" aria-controls="panel-overview" '
        'data-tab="overview" aria-selected="true">Ringkasan</button>'
    ]
    for key in keys:
        tab_buttons.append(
            f'<button class="tab" role="tab" id="tab-{key}" aria-controls="panel-{key}" '
            f'data-tab="{key}" aria-selected="false" '
            f'style="--accent:{dataset.colors[key]}">{_esc(ALGORITHMS[key]["name"])}</button>'
        )

    document = (
        '<!DOCTYPE html>\n<html lang="id">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        "<title>Visualisasi hasil local search</title>\n"
        f"<style>{CSS}</style>\n</head>\n<body>\n"
        f"{_header(dataset, upper)}\n"
        f'<div class="wrap"><nav class="tabs" role="tablist" aria-label="Algoritma">{"".join(tab_buttons)}</nav>\n'
        f"<main>{_overview_panel(dataset, upper)}{''.join(panels)}</main></div>\n"
        f'<script id="dashboard-data" type="application/json">{_json_for_script(payload)}</script>\n'
        f"<script>{get_plotlyjs()}</script>\n"
        f"<script>{JS}</script>\n</body>\n</html>\n"
    )

    target = out_path or os.path.join(output_dir, "dashboard.html")
    os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        handle.write(document)
    return target, dataset.warnings