from __future__ import annotations

import html
from dataclasses import dataclass
from typing import List, Optional, Sequence, Set, Tuple

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .data import PackageView, StateView, Vec

PALETTE = (
    "#4c78a8", "#f58518", "#54a24b", "#b279a2", "#9d755d",
    "#72b7b2", "#eeca3b", "#79706e", "#d67195", "#439894",
    "#9ecae9", "#ffbf79", "#b6992d", "#86bcb6", "#d4a6c8",
    "#8cd17d", "#6b8ba4", "#c6a15b", "#a5a5a5", "#5f9ea0",
)
EDGE_COLOR = "#1f1f1f"
FRAGILE_EDGE_COLOR = "#c1121f"
TRUCK_COLOR = "#2b2b2b"
FLOOR_COLOR = "#cfd6d9"
OUTSIDE_GAP = 2
OUTSIDE_SPACING = 1

_TRIANGLES = (
    (0, 1, 2), (0, 2, 3), (4, 5, 6), (4, 6, 7),
    (0, 1, 5), (0, 5, 4), (3, 2, 6), (3, 6, 7),
    (0, 3, 7), (0, 7, 4), (1, 2, 6), (1, 6, 5),
)
_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)
_LIGHTING = dict(ambient=0.55, diffuse=0.85, specular=0.12, roughness=0.9, fresnel=0.1)
_LIGHT_POSITION = dict(x=200, y=-300, z=400)
_AXES = ("x", "y", "z")


@dataclass
class OutsideLayout:
    items: List[Tuple[PackageView, Vec]]
    extent_x: float
    extent_y: float


def _corners(pos: Vec, dims: Vec) -> List[Vec]:
    x, y, z = pos
    w, d, h = dims
    return [
        (x, y, z), (x + w, y, z), (x + w, y + d, z), (x, y + d, z),
        (x, y, z + h), (x + w, y, z + h), (x + w, y + d, z + h), (x, y + d, z + h),
    ]


def _fmt(value: float) -> str:
    return f"{value:g}"


def _orientation_text(pkg: PackageView) -> Optional[str]:
    code = pkg.orientation
    if not code or len(code) != 3:
        return None
    return ", ".join(f"{axis}={side}" for axis, side in zip(_AXES, code))


def _hover(pkg: PackageView, pos: Optional[Vec]) -> str:
    lines = [f"<b>{html.escape(pkg.id)}</b>"]
    if pos is None:
        lines.append("Di luar truk")
    else:
        lines.append(f"Posisi (x, y, z): {', '.join(_fmt(v) for v in pos)}")
    w, d, h = pkg.dimensions
    lines.append(f"Dimensi (w x l x h): {_fmt(w)} x {_fmt(d)} x {_fmt(h)}")
    orientation = _orientation_text(pkg)
    if orientation is not None:
        lines.append(f"Orientasi (sisi paket di sumbu): {html.escape(orientation)}")
    lines.append(f"Value: {_fmt(pkg.value)}")
    lines.append(f"Berat: {_fmt(pkg.weight)}")
    lines.append(f"Pecah-belah: {'Ya' if pkg.is_fragile else 'Tidak'}")
    if pkg.eta is not None:
        lines.append(f"ETA: {_fmt(pkg.eta)} hari")
    return "<br>".join(lines)


def _mesh(pkg: PackageView, pos: Vec, color: str, inside: bool) -> go.Mesh3d:
    corners = _corners(pos, pkg.dimensions)
    i, j, k = zip(*_TRIANGLES)
    return go.Mesh3d(
        x=[c[0] for c in corners],
        y=[c[1] for c in corners],
        z=[c[2] for c in corners],
        i=list(i),
        j=list(j),
        k=list(k),
        color=color,
        opacity=1.0 if inside else 0.85,
        flatshading=True,
        lighting=_LIGHTING,
        lightposition=_LIGHT_POSITION,
        hoverinfo="text",
        hovertext=_hover(pkg, pos if inside else None),
        showlegend=False,
    )


def _edges(
    items: Sequence[Tuple[Vec, Vec]],
    color: str,
    width: int,
    name: str,
    show_legend: bool,
) -> go.Scatter3d:
    xs: List[Optional[float]] = []
    ys: List[Optional[float]] = []
    zs: List[Optional[float]] = []
    for pos, dims in items:
        corners = _corners(pos, dims)
        for a, b in _EDGES:
            xs += [corners[a][0], corners[b][0], None]
            ys += [corners[a][1], corners[b][1], None]
            zs += [corners[a][2], corners[b][2], None]
    return go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="lines",
        line=dict(color=color, width=width),
        hoverinfo="skip",
        name=name,
        legendgroup=name,
        showlegend=show_legend,
    )


def _floor(x0: float, y0: float, x1: float, y1: float, name: str, show_legend: bool, opacity: float) -> go.Mesh3d:
    return go.Mesh3d(
        x=[x0, x1, x1, x0],
        y=[y0, y0, y1, y1],
        z=[0, 0, 0, 0],
        i=[0, 0],
        j=[1, 2],
        k=[2, 3],
        color=FLOOR_COLOR,
        opacity=opacity,
        hoverinfo="skip",
        name=name,
        legendgroup=name,
        showlegend=show_legend,
    )


def layout_outside(packages: Sequence[PackageView], truck_w: float, row_limit: float) -> OutsideLayout:
    if not packages:
        return OutsideLayout([], truck_w, 0)
    origin_x = truck_w + OUTSIDE_GAP
    items: List[Tuple[PackageView, Vec]] = []
    cursor_x = 0.0
    cursor_y = 0.0
    row_depth = 0.0
    for pkg in packages:
        w, d, _ = pkg.dimensions
        if cursor_x > 0 and cursor_x + w > row_limit:
            cursor_x = 0.0
            cursor_y += row_depth + OUTSIDE_SPACING
            row_depth = 0.0
        items.append((pkg, (origin_x + cursor_x, cursor_y, 0)))
        cursor_x += w + OUTSIDE_SPACING
        row_depth = max(row_depth, d)
    return OutsideLayout(items, origin_x + row_limit, cursor_y + row_depth)


def _title(label: str, state: StateView) -> str:
    capacity = f"{state.max_capacity:g}"
    return (
        f"{label}<br><sup>Value {state.total_value:g}, berat {state.total_weight:g}/{capacity}, "
        f"volume {state.volume_utilization:.1f}%, {len(state.placed)} di dalam, "
        f"{len(state.outside)} di luar</sup>"
    )


def build_state_figure(views: Sequence[Tuple[str, Optional[StateView]]]) -> Optional[go.Figure]:
    views = [(label, state) for label, state in views if state is not None]
    if not views:
        return None

    truck_w, truck_d, truck_h = views[0][1].truck_dims
    outside_all = [p for _, s in views for p in s.outside]
    row_limit = max([truck_w] + [p.dimensions[0] for p in outside_all])
    ids = sorted({p.id for _, s in views for p in s.placed + s.outside})
    colors = {pid: PALETTE[n % len(PALETTE)] for n, pid in enumerate(ids)}
    layouts = [layout_outside(s.outside, truck_w, row_limit) for _, s in views]

    x_max = max(lay.extent_x + (OUTSIDE_SPACING if lay.items else 0) for lay in layouts)
    y_max = max([truck_d] + [lay.extent_y + OUTSIDE_SPACING for lay in layouts if lay.items])
    z_max = max([truck_h] + [p.dimensions[2] for p in outside_all])

    count = len(views)
    fig = make_subplots(
        rows=1,
        cols=count,
        specs=[[{"type": "scene"}] * count],
        subplot_titles=[_title(label, state) for label, state in views],
        horizontal_spacing=0.01,
    )

    announced: Set[str] = set()

    def legend(name: str) -> bool:
        if name in announced:
            return False
        announced.add(name)
        return True

    for col, ((_, state), lay) in enumerate(zip(views, layouts), start=1):
        fig.add_trace(_floor(0, 0, truck_w, truck_d, "Lantai peti kemas", False, 0.35), row=1, col=col)
        fig.add_trace(
            _edges([((0, 0, 0), (truck_w, truck_d, truck_h))], TRUCK_COLOR, 5, "Peti kemas truk", legend("Peti kemas truk")),
            row=1, col=col,
        )
        if lay.items:
            fig.add_trace(
                _floor(truck_w + OUTSIDE_GAP - OUTSIDE_SPACING, 0, x_max, lay.extent_y + OUTSIDE_SPACING,
                       "Area di luar truk", legend("Area di luar truk"), 0.3),
                row=1, col=col,
            )

        drawables: List[Tuple[PackageView, Vec, bool]] = [(p, p.position, True) for p in state.placed]
        drawables += [(p, pos, False) for p, pos in lay.items]
        for pkg, pos, inside in drawables:
            fig.add_trace(_mesh(pkg, pos, colors[pkg.id], inside), row=1, col=col)

        normal = [(pos, p.dimensions) for p, pos, _ in drawables if not p.is_fragile]
        fragile = [(pos, p.dimensions) for p, pos, _ in drawables if p.is_fragile]
        if normal:
            fig.add_trace(_edges(normal, EDGE_COLOR, 3, "Paket", legend("Paket")), row=1, col=col)
        if fragile:
            fig.add_trace(
                _edges(fragile, FRAGILE_EDGE_COLOR, 6, "Pecah-belah (tepi merah)", legend("Pecah-belah (tepi merah)")),
                row=1, col=col,
            )

    fig.update_scenes(
        dict(
            xaxis=dict(title=dict(text="Lebar (x)"), range=[0, x_max]),
            yaxis=dict(title=dict(text="Panjang (y)"), range=[0, y_max]),
            zaxis=dict(title=dict(text="Tinggi (z)"), range=[0, z_max]),
            aspectmode="data",
            camera=dict(eye=dict(x=1.7, y=-1.9, z=1.1)),
        )
    )
    fig.update_layout(
        height=620,
        margin=dict(l=0, r=0, t=70, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", x=0.5, xanchor="center", y=0, yanchor="top"),
        font=dict(family="Segoe UI, system-ui, sans-serif", size=12, color="#17212b"),
    )
    return fig