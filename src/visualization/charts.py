from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .data import ALGORITHMS, GroupStats, Run

FONT = dict(family="Segoe UI, system-ui, sans-serif", size=12, color="#17212b")
GRID = "#e3e7e8"
MUTED = "#8d99ae"
BOUND_COLOR = "#9b2226"
GA_COLOR = "#386641"
GROUP_COLORS = (
    "#1d6fa5", "#bc6c25", "#9b2226", "#6a4c93",
    "#e09f3e", "#5c677d", "#d6336c", "#0b7285",
)
DASHES = ("solid", "dash", "dot", "dashdot")
GA_DASHES = ("solid", "dash", "dot", "dashdot", "longdash", "longdashdot")
MAIN_SERIES = {"hc": "history_fitness", "sa": "history_best", "ga": "history_max"}

def _style(fig: go.Figure, title: str, height: int = 380) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0.01, xanchor="left", font=dict(size=15)),
        height=height,
        margin=dict(l=64, r=20, t=56, b=56),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=FONT,
        hovermode="closest",
        legend=dict(orientation="h", x=0, y=-0.22, yanchor="top"),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor="#9aa5aa")
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor="#9aa5aa")
    return fig

def _x(values: Sequence, start: int = 0) -> List[int]:
    return list(range(start, start + len(values)))

def group_colors(runs: Sequence[Run]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for run in runs:
        if run.group in mapping:
            continue
        if run.key == "ga":
            mapping[run.group] = GA_COLOR
        else:
            mapping[run.group] = GROUP_COLORS[len(mapping) % len(GROUP_COLORS)]
    return mapping

def objective_figure(run: Run, color: str) -> Optional[go.Figure]:
    fig = go.Figure()
    if run.key == "ga":
        color = GA_COLOR
        best = run.series.get("history_max")
        mean = run.series.get("history_avg")
        if not best:
            return None
        fig.add_trace(go.Scatter(
            x=_x(best), y=best, mode="lines", name="Maksimum populasi",
            line=dict(color=color, width=3),
        ))
        if mean:
            fig.add_trace(go.Scatter(
                x=_x(mean), y=mean, mode="lines", name="Rata-rata populasi",
                line=dict(color=color, width=2, dash="dash"), opacity=0.75,
            ))
    elif run.key == "sa":
        current = run.series.get("history_current")
        best = run.series.get("history_best")
        if not current and not best:
            return None
        if current:
            fig.add_trace(go.Scatter(
                x=_x(current), y=current, mode="lines", name="Objective state saat ini",
                line=dict(color=color, width=1.5), opacity=0.45,
            ))
        if best:
            fig.add_trace(go.Scatter(
                x=_x(best), y=best, mode="lines", name="Objective terbaik",
                line=dict(color=color, width=3),
            ))
    else:
        values = run.series.get("history_fitness")
        if not values:
            return None
        raw = run.series.get("history_raw")
        if raw:
            fig.add_trace(go.Scatter(
                x=_x(raw), y=raw, mode="lines", name="Objective tiap restart",
                line=dict(color=color, width=2), opacity=0.55,
            ))
            fig.add_trace(go.Scatter(
                x=_x(values), y=values, mode="lines", name="Objective terbaik sejauh ini",
                line=dict(color=color, width=3, dash="dash"),
            ))
        else:
            fig.add_trace(go.Scatter(
                x=_x(values), y=values,
                mode="lines+markers" if len(values) <= 80 else "lines",
                name="Objective", line=dict(color=color, width=3), marker=dict(size=6),
            ))
        position = 0
        restart_number = 1
        for count in (run.series.get("iterations_per_restart") or [])[:-1]:
            if count is None:
                break
            position += int(count) + 1
            restart_number += 1
            fig.add_vline(
                x=position, line_dash="dot", line_color=MUTED,
                annotation_text=f"Restart {restart_number}", annotation_position="top",
            )
    _style(fig, "Nilai objective function vs iterasi")
    fig.update_xaxes(title_text="Iterasi")
    fig.update_yaxes(title_text="Nilai objective")
    return fig

def sa_probability_figure(run: Run, color: str) -> Optional[go.Figure]:
    delta = run.series.get("history_delta")
    prob = run.series.get("history_acceptance_probability")
    temp = run.series.get("history_temperature")
    if not delta or not prob:
        return None

    worse_x, worse_y, worse_text = [], [], []
    keep_x, keep_y = [], []
    for step, (d, p) in enumerate(zip(delta, prob), start=1):
        if d is None or p is None:
            continue
        if d < 0:
            worse_x.append(step)
            worse_y.append(p)
            worse_text.append(f"Iterasi {step}<br>ΔE = {d:g}<br>e^(ΔE/T) = {p:.4g}")
        else:
            keep_x.append(step)
            keep_y.append(p)

    has_temp = bool(temp)
    fig = make_subplots(
        rows=2 if has_temp else 1, cols=1, shared_xaxes=True,
        row_heights=[0.68, 0.32] if has_temp else [1.0], vertical_spacing=0.08,
    )
    fig.add_trace(go.Scatter(
        x=keep_x, y=keep_y, mode="markers", name="ΔE ≥ 0 (selalu diterima, e^(ΔE/T) ≥ 1 dipotong ke 1)",
        marker=dict(size=4, color=MUTED), opacity=0.5, hoverinfo="skip",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=worse_x, y=worse_y, mode="markers", name="ΔE < 0 (peluang e^(ΔE/T))",
        marker=dict(size=5, color=color), opacity=0.7, text=worse_text, hoverinfo="text",
    ), row=1, col=1)
    if has_temp:
        steps = _x(delta, 1)
        fig.add_trace(go.Scatter(
            x=steps, y=temp[:len(delta)], mode="lines", name="Suhu T",
            line=dict(color=color, width=2),
        ), row=2, col=1)
        fig.update_yaxes(title_text="Suhu T", row=2, col=1)
        fig.update_xaxes(title_text="Iterasi", row=2, col=1)
    else:
        fig.update_xaxes(title_text="Iterasi", row=1, col=1)

    _style(fig, "Peluang penerimaan e^(ΔE/T) vs iterasi", height=470)
    fig.update_yaxes(title_text="e^(ΔE/T)", range=[-0.03, 1.05], row=1, col=1)
    fig.update_layout(legend=dict(orientation="h", x=0, y=-0.18, yanchor="top"))
    return fig

def run_figures(run: Run, color: str) -> List[go.Figure]:
    figures = [objective_figure(run, color)]
    if run.key == "sa":
        figures.append(sa_probability_figure(run, color))
    return [fig for fig in figures if fig is not None]

def overlay_figure(runs: Sequence[Run], colors: Dict[str, str]) -> Optional[go.Figure]:
    fig = go.Figure()
    announced = set()
    group_index: Dict[str, int] = {}
    for run in runs:
        group_index.setdefault(run.group, len(group_index))
    has_ga = any(run.key == "ga" for run in runs)

    for run in runs:
        values = run.series.get(MAIN_SERIES[run.key])
        if not values:
            continue
        first = run.group not in announced
        announced.add(run.group)
        if run.key == "ga":
            color = GA_COLOR
            dash = GA_DASHES[group_index[run.group] % len(GA_DASHES)]
            width = 2.5
            opacity = 0.85
        else:
            color = colors[run.group]
            dash = DASHES[(run.repetition - 1) % len(DASHES)]
            width = 2
            opacity = 1.0
        fig.add_trace(go.Scatter(
            x=_x(values), y=values, mode="lines",
            name=run.group, legendgroup=run.group, showlegend=first,
            line=dict(color=color, width=width, dash=dash),
            opacity=opacity,
            meta=run.label, hovertemplate="%{meta}<br>Iterasi %{x}<br>Objective %{y:g}<extra></extra>",
        ))
    if not fig.data:
        return None
    title = (
        "Perbandingan kurva maksimum populasi semua run (gaya garis membedakan konfigurasi)"
        if has_ga
        else "Perbandingan kurva semua run (garis putus-putus membedakan run)"
    )
    _style(fig, title, height=420)
    fig.update_xaxes(title_text="Iterasi")
    fig.update_yaxes(title_text="Nilai objective")
    fig.update_layout(legend=dict(orientation="h", x=0, y=-0.22, yanchor="top"))
    return fig

def _bar_figure(
    stats: Sequence[GroupStats],
    colors: Dict[str, str],
    title: str,
    y_title: str,
    pick_values,
    reference: Optional[float] = None,
) -> Optional[go.Figure]:
    rows = []
    for entry in stats:
        values = pick_values(entry)
        if values:
            rows.append((entry, values))
    if not rows:
        return None

    labels = []
    for entry, _ in rows:
        name = ALGORITHMS[entry.key]["name"]
        labels.append(name if entry.group == name else f"{name}<br>{entry.group}")

    fig = go.Figure()
    for key in ALGORITHMS:
        subset = [(label, e, v) for label, (e, v) in zip(labels, rows) if e.key == key]
        if not subset:
            continue
        means = [sum(v) / len(v) for _, _, v in subset]
        stds = [
            (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5 if len(v) > 1 else 0.0
            for (_, _, v), m in zip(subset, means)
        ]
        bar_color = GA_COLOR if key == "ga" else colors[key]
        fig.add_trace(go.Bar(
            x=[label for label, _, _ in subset], y=means, name=ALGORITHMS[key]["name"],
            marker=dict(color=bar_color),
            error_y=dict(type="data", array=stds, visible=True, color="#17212b", thickness=1.5, width=5),
            hovertemplate="%{x}<br>Rata-rata %{y:.4g}<extra></extra>",
        ))
    point_x = [label for label, (_, v) in zip(labels, rows) for _ in v]
    point_y = [x for _, v in rows for x in v]
    fig.add_trace(go.Scatter(
        x=point_x, y=point_y, mode="markers", name="Tiap run",
        marker=dict(size=8, color="#ffffff", line=dict(color="#17212b", width=1.5)),
        hovertemplate="%{y:.4g}<extra></extra>",
    ))
    _style(fig, title, height=430)
    fig.update_layout(barmode="overlay", legend=dict(orientation="h", x=0, y=-0.45, yanchor="top"))
    fig.update_xaxes(categoryorder="array", categoryarray=labels, tickangle=-25)
    fig.update_yaxes(title_text=y_title, rangemode="tozero")
    if reference is not None and reference > 0:
        fig.add_hline(
            y=reference,
            line_dash="dash",
            line_color=BOUND_COLOR,
            annotation_text=f"Batas atas {reference:g}",
            annotation_position="top left",
        )
    return fig

def overview_figures(
    stats: Sequence[GroupStats],
    colors: Dict[str, str],
    upper_bound: Optional[float] = None,
) -> Dict[str, Optional[go.Figure]]:
    return {
        "final": _bar_figure(
            stats, colors, "Value akhir per konfigurasi (rata-rata dan simpangan baku)",
            "Value akhir", lambda e: e.values, upper_bound,
        ),
        "duration": _bar_figure(
            stats, colors, "Durasi pencarian per konfigurasi (rata-rata dan simpangan baku)",
            "Detik", lambda e: e.durations,
        ),
    }