from __future__ import annotations

import glob
import json
import math
import os
import re
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

Vec = Tuple[float, float, float]

ALGORITHMS: Dict[str, Dict[str, str]] = {
    "hc": {"name": "Hill-climbing", "color": "#bc6c25"},
    "sa": {"name": "Simulated Annealing", "color": "#1d6fa5"},
    "ga": {"name": "Genetic Algorithm", "color": "#386641"},
}
ALGORITHM_ORDER = ("hc", "sa", "ga")

SERIES_KEYS = {
    "hc": ("history_fitness", "history_raw", "iterations_per_restart"),
    "sa": (
        "history_current",
        "history_best",
        "history_temperature",
        "history_delta",
        "history_acceptance_probability",
    ),
    "ga": ("history_max", "history_avg", "history_min"),
}
FIRST_VALUE_SERIES = {"hc": "history_fitness", "sa": "history_current", "ga": "history_max"}

ITERATION_LABEL = {
    "hc": "Iterasi hingga berhenti",
    "sa": "Iterasi",
    "ga": "Banyak iterasi",
}

METRIC_FIELDS = {
    "hc": (
        ("max_iterations", "Maksimum iterasi"),
        ("max_stagnation", "Maksimum iterasi tanpa perbaikan"),
        ("neighbors_evaluated", "Neighbor dievaluasi"),
        ("stopped_at_local_optimum", "Berhenti di local optimum"),
        ("sideways_moves", "Jumlah sideways move"),
        ("max_sideways_moves", "Maksimum sideways move"),
        ("restarts", "Jumlah restart"),
        ("max_restarts", "Maksimum restart"),
        ("iterations_per_restart", "Iterasi per restart"),
        ("restart_final_values", "Value akhir tiap restart"),
        ("restarts_at_local_optimum", "Restart yang berhenti di local optimum"),
    ),
    "sa": (
        ("max_iterations", "Maksimum iterasi"),
        ("initial_temperature", "Suhu awal"),
        ("cooling_rate", "Cooling rate"),
        ("min_temperature", "Suhu minimum"),
        ("final_temperature", "Suhu akhir"),
        ("stop_reason", "Penyebab berhenti"),
        ("accepted_improving", "Diterima (membaik)"),
        ("accepted_sideways", "Diterima (sideways)"),
        ("accepted_worsening", "Diterima (memburuk)"),
        ("rejected", "Ditolak"),
        ("no_neighbor", "Iterasi tanpa neighbor"),
        ("stuck_window", "Jendela stuck (iterasi)"),
        ("stuck_episodes", "Episode stuck (best tidak membaik)"),
        ("stuck_iterations", "Iterasi stuck (best tidak membaik)"),
        ("stuck_frequency", "Frekuensi stuck (best tidak membaik)"),
        ("trapped_episodes", "Episode terperangkap (state tidak berubah)"),
        ("trapped_iterations", "Iterasi terperangkap (state tidak berubah)"),
        ("trapped_frequency", "Frekuensi terperangkap (state tidak berubah)"),
    ),
    "ga": (
        ("pop_size", "Jumlah populasi"),
        ("crossover_rate", "Crossover rate"),
        ("mutation_rate", "Mutation rate"),
        ("elitism", "Jumlah elitisme"),
        ("selection", "Metode seleksi"),
        ("seed", "Seed"),
        ("evaluations", "Evaluasi fitness"),
    ),
}

AXIS_NAMES = "wlh"
TOLERANCE = 1e-6

_HEX_COLOR = re.compile(r"^#[0-9a-fA-F]{6}$")

@dataclass(frozen=True)
class PackageView:
    id: str
    dimensions: Vec
    value: float
    weight: float
    is_fragile: bool
    eta: Optional[float]
    position: Optional[Vec] = None
    original_dimensions: Optional[Vec] = None
    orientation: Optional[str] = None

@dataclass
class StateView:
    truck_label: str
    truck_dims: Vec
    max_capacity: float
    placed: List[PackageView]
    outside: List[PackageView]

    @property
    def total_value(self) -> float:
        return sum(p.value for p in self.placed)

    @property
    def total_weight(self) -> float:
        return sum(p.weight for p in self.placed)

    @property
    def volume_utilization(self) -> float:
        w, d, h = self.truck_dims
        volume = w * d * h
        used = sum(p.dimensions[0] * p.dimensions[1] * p.dimensions[2] for p in self.placed)
        return 100.0 * used / volume if volume > 0 else 0.0

    @property
    def weight_utilization(self) -> float:
        if not math.isfinite(self.max_capacity) or self.max_capacity <= 0:
            return 0.0
        return 100.0 * self.total_weight / self.max_capacity

@dataclass
class Run:
    key: str
    group: str
    label: str
    repetition: int
    initial_value: Optional[float]
    final_value: float
    duration: Optional[float]
    iterations: Optional[int]
    initial: Optional[StateView]
    final: Optional[StateView]
    series: Dict[str, List[Optional[float]]]
    metrics: List[Tuple[str, str]]
    issues: List[str]

@dataclass
class GroupStats:
    key: str
    group: str
    values: List[float]
    durations: List[float]
    iterations: List[float]

    @property
    def count(self) -> int:
        return len(self.values)

    @property
    def mean(self) -> float:
        return statistics.fmean(self.values)

    @property
    def std(self) -> float:
        return statistics.stdev(self.values) if len(self.values) > 1 else 0.0

    @property
    def minimum(self) -> float:
        return min(self.values)

    @property
    def maximum(self) -> float:
        return max(self.values)

    @property
    def mean_duration(self) -> Optional[float]:
        return statistics.fmean(self.durations) if self.durations else None

    @property
    def mean_iterations(self) -> Optional[float]:
        return statistics.fmean(self.iterations) if self.iterations else None

@dataclass
class Dataset:
    runs: List[Run]
    colors: Dict[str, str]
    warnings: List[str] = field(default_factory=list)
    objectives: List[str] = field(default_factory=list)

def algorithm_key(name: Any) -> Optional[str]:
    text = str(name).lower()
    if "genetic" in text:
        return "ga"
    if "anneal" in text:
        return "sa"
    if "hill" in text:
        return "hc"
    return None

def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

def _number(value: Any, default: float = 0.0) -> float:
    return value if _is_num(value) else default

def _close(a: float, b: float) -> bool:
    return abs(a - b) <= TOLERANCE * max(1.0, abs(a), abs(b))

def _vec3(raw: Any, what: str) -> Vec:
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise ValueError(f"{what} harus berisi tepat 3 angka")
    for value in raw:
        if not _is_num(value):
            raise ValueError(f"{what} berisi nilai tidak valid: {value!r}")
    return (raw[0], raw[1], raw[2])

def _package(raw: Any, with_position: bool) -> PackageView:
    if not isinstance(raw, dict):
        raise ValueError("data paket harus berupa objek")
    pid = str(raw["id"])
    position = _vec3(raw["position"], f"posisi {pid}") if with_position else None
    original = raw.get("original_dimensions")
    orientation = raw.get("orientation")
    eta = raw.get("eta")
    return PackageView(
        id=pid,
        dimensions=_vec3(raw["dimensions"], f"dimensi {pid}"),
        value=_number(raw.get("value")),
        weight=_number(raw.get("weight")),
        is_fragile=bool(raw.get("is_fragile", False)),
        eta=eta if _is_num(eta) else None,
        position=position,
        original_dimensions=_vec3(original, f"dimensi asli {pid}") if original is not None else None,
        orientation=orientation if isinstance(orientation, str) else None,
    )

def state_from_dict(data: Any) -> StateView:
    if not isinstance(data, dict):
        raise ValueError("file solusi harus berupa objek JSON")
    try:
        truck = data["truck"]
        placed = [_package(p, True) for p in data.get("placed_packages", [])]
        outside = [_package(p, False) for p in data.get("outside_packages", [])]
        return StateView(
            truck_label=str(truck.get("label", "Truck")),
            truck_dims=_vec3(truck["dimensions"], "dimensi truk"),
            max_capacity=_number(truck.get("max_capacity"), math.inf),
            placed=placed,
            outside=outside,
        )
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"struktur file solusi tidak lengkap ({exc!r})") from exc

def recompute_objective(state: StateView, objective: Optional[str]) -> Optional[float]:
    name = objective or "default"
    if name == "default":
        return float(state.total_value)
    if name == "density":
        w, d, h = state.truck_dims
        volume = w * d * h
        used = sum(p.dimensions[0] * p.dimensions[1] * p.dimensions[2] for p in state.placed)
        ratio = used / volume if volume > 0 else 0.0
        return float(state.total_value) * (1.0 + ratio)
    if name == "eta_urgency":
        return math.fsum(
            p.value * (1.0 + 1.0 / (max(0, p.eta or 0) + 1.0)) for p in state.placed
        )
    return None

def _orientation_matches(pkg: PackageView) -> bool:
    if pkg.orientation is None or pkg.original_dimensions is None:
        return True
    code = pkg.orientation
    if len(code) != 3 or sorted(code) != sorted(AXIS_NAMES):
        return False
    rebuilt = tuple(pkg.original_dimensions[AXIS_NAMES.index(c)] for c in code)
    return rebuilt == tuple(pkg.dimensions)

def validate_state(state: StateView) -> List[str]:
    issues: List[str] = []
    width, depth, height = state.truck_dims
    placed = state.placed

    counts: Dict[str, int] = {}
    for pkg in placed + state.outside:
        counts[pkg.id] = counts.get(pkg.id, 0) + 1
    for pid, count in counts.items():
        if count > 1:
            issues.append(f"{pid}: muncul lebih dari sekali pada state (di dalam dan/atau di luar truk)")

    for pkg in placed + state.outside:
        if pkg.original_dimensions is not None and sorted(pkg.dimensions) != sorted(pkg.original_dimensions):
            issues.append(f"{pkg.id}: dimensi bukan hasil rotasi dari dimensi asli")
        elif not _orientation_matches(pkg):
            issues.append(f"{pkg.id}: kode orientasi tidak cocok dengan dimensinya")

    for pkg in placed:
        x, y, z = pkg.position
        w, d, h = pkg.dimensions
        if any(float(v) != int(v) for v in (x, y, z, w, d, h)):
            issues.append(f"{pkg.id}: koordinat atau dimensi bukan bilangan bulat")
        if min(w, d, h) <= 0:
            issues.append(f"{pkg.id}: dimensi harus positif")
        if x < 0 or y < 0 or z < 0 or x + w > width or y + d > depth or z + h > height:
            issues.append(f"{pkg.id}: menembus dinding peti kemas")

    for i, a in enumerate(placed):
        for b in placed[i + 1:]:
            if _overlap_3d(a, b):
                issues.append(f"{a.id} dan {b.id}: saling tumpang tindih")

    for pkg in placed:
        z = pkg.position[2]
        supporters = []
        for q in placed:
            if q is pkg or not _overlap_xy(pkg, q):
                continue
            top = q.position[2] + q.dimensions[2]
            if top <= z:
                if q.is_fragile:
                    issues.append(f"{q.id}: barang pecah-belah berada di bawah {pkg.id}")
                if top == z:
                    supporters.append(q)
        if z != 0 and not supporters:
            issues.append(f"{pkg.id}: melayang (tidak ada penyangga)")

    if state.total_weight > state.max_capacity:
        issues.append(
            f"total berat {state.total_weight:g} melebihi kapasitas {state.max_capacity:g}"
        )
    return list(dict.fromkeys(issues))

def _overlap_xy(a: PackageView, b: PackageView) -> bool:
    return (
        a.position[0] < b.position[0] + b.dimensions[0]
        and b.position[0] < a.position[0] + a.dimensions[0]
        and a.position[1] < b.position[1] + b.dimensions[1]
        and b.position[1] < a.position[1] + a.dimensions[1]
    )

def _overlap_3d(a: PackageView, b: PackageView) -> bool:
    return (
        _overlap_xy(a, b)
        and a.position[2] < b.position[2] + b.dimensions[2]
        and b.position[2] < a.position[2] + a.dimensions[2]
    )

def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)

def _resolve(base: str, relative: Any) -> Optional[str]:
    if not isinstance(relative, str) or not relative:
        return None
    target = os.path.abspath(os.path.join(base, relative.replace("\\", "/")))
    try:
        if os.path.commonpath([base, target]) != base:
            return None
    except ValueError:
        return None
    return target

def _load_state(base: str, relative: Any, what: str, warnings: List[str]) -> Optional[StateView]:
    if relative is None:
        return None
    path = _resolve(base, relative)
    if path is None:
        warnings.append(f"{what}: path file tidak valid atau di luar folder output ({relative!r})")
        return None
    try:
        return state_from_dict(_read_json(path))
    except (OSError, ValueError) as exc:
        warnings.append(f"{what}: gagal membaca {relative} ({exc})")
        return None

def _series(raw: Any) -> List[Optional[float]]:
    return [float(v) if _is_num(v) else None for v in raw]

def _group_label(key: str, record: Dict[str, Any], meta: Dict[str, Any]) -> str:
    if record.get("group"):
        return str(record["group"])
    if key == "ga" and (record.get("pop_size") is not None or record.get("iterations") is not None):
        config = f"Pop {record.get('pop_size', '?')}, Iter {record.get('iterations', '?')}"
        kind = record.get("experiment_set")
        return f"{kind}: {config}" if kind else config
    return str(meta.get("group") or "Semua run")

def _format_metric(name: str, value: Any) -> str:
    if isinstance(value, bool):
        return "Ya" if value else "Tidak"
    if name.endswith("frequency") and _is_num(value):
        return f"{value * 100:.2f}%"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, list):
        def piece(item: Any) -> str:
            if _is_num(item):
                return f"{item:g}"
            return str(item)
        head = ", ".join(piece(v) for v in value[:20])
        return head + (", ..." if len(value) > 20 else "")
    return str(value)

def _build_metrics(
    key: str,
    record: Dict[str, Any],
    initial_value: Optional[float],
    final_value: float,
    duration: Optional[float],
    iterations: Optional[int],
    final: Optional[StateView],
) -> List[Tuple[str, str]]:
    metrics: List[Tuple[str, str]] = []
    if initial_value is not None:
        metrics.append(("Value awal", f"{initial_value:g}"))
    metrics.append(("Value akhir", f"{final_value:g}"))
    if initial_value is not None:
        metrics.append(("Peningkatan", f"{final_value - initial_value:+g}"))
    total_value = record.get("total_value")
    if _is_num(total_value) and total_value != final_value:
        metrics.append(("Total value paket", f"{total_value:g}"))
    if duration is not None:
        metrics.append(("Durasi pencarian", f"{duration:.4f} detik"))
    if iterations is not None:
        metrics.append((ITERATION_LABEL[key], str(iterations)))
    if final is not None:
        metrics.append(("Paket di dalam / di luar", f"{len(final.placed)} / {len(final.outside)}"))
        capacity = f"{final.max_capacity:g}" if math.isfinite(final.max_capacity) else "-"
        metrics.append(("Total berat", f"{final.total_weight:g} / {capacity}"))
        metrics.append(("Utilisasi volume", f"{final.volume_utilization:.1f}%"))
    for name, label in METRIC_FIELDS[key]:
        value = record.get(name)
        if value is not None:
            metrics.append((label, _format_metric(name, value)))
    return metrics

def _record_issues(
    record: Dict[str, Any],
    objective: Optional[str],
    initial: Optional[StateView],
    final: Optional[StateView],
) -> List[str]:
    issues: List[str] = []
    checks = (
        ("final_value", "state akhir", final),
        ("initial_value", "state awal", initial),
    )
    for field_name, label, state in checks:
        stored = record.get(field_name)
        if state is None or not _is_num(stored):
            continue
        expected = recompute_objective(state, objective)
        if expected is not None and not _close(expected, float(stored)):
            issues.append(
                f"Catatan run: {field_name} = {stored:g} tidak cocok dengan hitung ulang dari {label} ({expected:g})"
            )
    total = record.get("total_value")
    if final is not None and _is_num(total) and not _close(float(total), float(final.total_value)):
        issues.append(
            f"Catatan run: total_value = {total:g} tidak cocok dengan state akhir ({final.total_value:g})"
        )
    return issues

def _parse_run(
    key: str,
    record: Any,
    meta: Dict[str, Any],
    base: str,
    index: int,
    warnings: List[str],
) -> Run:
    if not isinstance(record, dict):
        raise ValueError("catatan run harus berupa objek")

    group = _group_label(key, record, meta)
    repetition = int(record["repetition"]) if _is_num(record.get("repetition")) else index
    label = f"{group} (run {repetition})"

    series = {
        name: _series(record[name])
        for name in SERIES_KEYS[key]
        if isinstance(record.get(name), list)
    }

    initial = _load_state(base, record.get("initial_state_file"), f"{label}, state awal", warnings)
    final = _load_state(base, record.get("final_state_file"), f"{label}, state akhir", warnings)

    final_value = record.get("final_value")
    if not _is_num(final_value):
        if final is None:
            raise ValueError("final_value tidak ada dan state akhir tidak bisa dibaca")
        final_value = final.total_value

    initial_value = record.get("initial_value")
    if not _is_num(initial_value):
        first_series = series.get(FIRST_VALUE_SERIES[key]) or []
        if first_series and first_series[0] is not None:
            initial_value = first_series[0]
        elif initial is not None:
            initial_value = initial.total_value
        else:
            initial_value = None

    duration = record.get("duration_seconds", record.get("duration"))
    duration = float(duration) if _is_num(duration) else None

    iterations = record.get("iterations")
    iterations = int(iterations) if _is_num(iterations) else None

    issues: List[str] = []
    if final is not None:
        issues.extend(f"State akhir: {msg}" for msg in validate_state(final))
    if initial is not None:
        issues.extend(f"State awal: {msg}" for msg in validate_state(initial))
    issues.extend(_record_issues(record, meta.get("objective"), initial, final))

    return Run(
        key=key,
        group=group,
        label=label,
        repetition=repetition,
        initial_value=initial_value,
        final_value=final_value,
        duration=duration,
        iterations=iterations,
        initial=initial,
        final=final,
        series=series,
        metrics=_build_metrics(key, record, initial_value, final_value, duration, iterations, final),
        issues=issues,
    )

def _cross_check(runs: List[Run]) -> None:
    reference: Optional[StateView] = None
    for run in runs:
        reference = run.final or run.initial
        if reference is not None:
            break
    if reference is None:
        return
    ids = {p.id for p in reference.placed + reference.outside}
    for run in runs:
        for name, state in (("State awal", run.initial), ("State akhir", run.final)):
            if state is None:
                continue
            if {p.id for p in state.placed + state.outside} != ids:
                run.issues.append(f"{name}: himpunan paket berbeda dari run lain")
            if state.truck_dims != reference.truck_dims or state.max_capacity != reference.max_capacity:
                run.issues.append(f"{name}: dimensi atau kapasitas truk berbeda dari run lain")

def load_dataset(output_dir: str) -> Dataset:
    base = os.path.abspath(output_dir)
    folder = os.path.join(base, "experiments")
    files = sorted(glob.glob(os.path.join(folder, "*summary*.json")))
    if not files:
        raise FileNotFoundError(
            f"Tidak ada file ringkasan eksperimen (*summary*.json) di {folder}"
        )

    runs: List[Run] = []
    warnings: List[str] = []
    objectives = set()
    colors = {key: info["color"] for key, info in ALGORITHMS.items()}

    for path in files:
        name = os.path.basename(path)
        try:
            summary = _read_json(path)
        except (OSError, ValueError) as exc:
            warnings.append(f"{name}: gagal dibaca ({exc})")
            continue
        if not isinstance(summary, dict):
            warnings.append(f"{name}: isi file bukan objek JSON")
            continue

        meta = summary.get("metadata") if isinstance(summary.get("metadata"), dict) else {}
        key = algorithm_key(meta.get("algorithm", ""))
        if key is None:
            warnings.append(f"{name}: metadata.algorithm tidak dikenali ({meta.get('algorithm')!r})")
            continue
        records = next(
            (summary[k] for k in ("runs", "all_runs", "results") if isinstance(summary.get(k), list)),
            None,
        )
        if records is None:
            warnings.append(f"{name}: tidak ada daftar run (runs / all_runs / results)")
            continue

        objective = meta.get("objective")
        if objective is not None:
            objectives.add(str(objective))

        plot_color = meta.get("plot_color")
        if key == "ga" and isinstance(plot_color, str) and _HEX_COLOR.match(plot_color):
            colors["ga"] = plot_color

        for index, record in enumerate(records, start=1):
            try:
                runs.append(_parse_run(key, record, meta, base, index, warnings))
            except (KeyError, TypeError, ValueError) as exc:
                warnings.append(f"{name}: run ke-{index} dilewati ({exc})")

    if not runs:
        raise ValueError("Tidak ada run yang dapat dibaca dari folder eksperimen")

    group_order: Dict[Tuple[str, str], int] = {}
    for run in runs:
        group_order.setdefault((run.key, run.group), len(group_order))
    runs.sort(
        key=lambda r: (ALGORITHM_ORDER.index(r.key), group_order[(r.key, r.group)], r.repetition)
    )
    _cross_check(runs)
    return Dataset(runs=runs, colors=colors, warnings=warnings, objectives=sorted(objectives))

def aggregate(runs: List[Run]) -> List[GroupStats]:
    stats: Dict[Tuple[str, str], GroupStats] = {}
    for run in runs:
        entry = stats.setdefault((run.key, run.group), GroupStats(run.key, run.group, [], [], []))
        entry.values.append(float(run.final_value))
        if run.duration is not None:
            entry.durations.append(run.duration)
        if run.iterations is not None:
            entry.iterations.append(float(run.iterations))
    return list(stats.values())