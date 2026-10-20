from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional

HC_OPTIONAL_ATTRS = (
    "max_iterations",
    "max_stagnation",
    "sideways_moves",
    "max_sideways_moves",
    "restarts",
    "max_restarts",
    "iterations_per_restart",
    "restart_final_values",
    "restarts_at_local_optimum",
    "history_raw",
)
SA_PARAM_ATTRS = (
    "max_iterations",
    "initial_temperature",
    "cooling_rate",
    "min_temperature",
    "final_temperature",
    "stop_reason",
    "accepted_improving",
    "accepted_sideways",
    "accepted_worsening",
    "rejected",
    "no_neighbor",
    "stuck_window",
    "stuck_episodes",
    "stuck_iterations",
    "stuck_frequency",
    "trapped_episodes",
    "trapped_iterations",
    "trapped_frequency",
)

def _plain(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value

def _relative(path: str, base: str) -> str:
    return os.path.relpath(path, base).replace(os.sep, "/")

def _state_summary(state: Any) -> Dict[str, Any]:
    if state is None:
        return {}
    return {
        "total_value": state.total_value,
        "packages_placed": len(state.placed),
        "volume_utilization": round(state.volume_utilization_ratio * 100, 2),
        "weight_utilization": round(state.weight_utilization_ratio * 100, 2),
    }

def _save_states(output_dir: str, name: str, initial: Any, final: Any, algorithm: str) -> Dict[str, Optional[str]]:
    from ..utils.data_loader import save_solution

    folder = os.path.join(output_dir, "solutions")
    files: Dict[str, Optional[str]] = {}
    for tag, state in (("initial", initial), ("final", final)):
        if state is None:
            files[f"{tag}_state_file"] = None
            continue
        path = os.path.join(folder, f"{name}_{tag}.json")
        save_solution(path, state, metadata={"algorithm": algorithm, "state": tag, "run": name})
        files[f"{tag}_state_file"] = _relative(path, output_dir)
    return files

def hill_climbing_record(
    hc: Any,
    output_dir: str,
    name: str,
    run_id: int,
    repetition: int,
    seed: Optional[int] = None,
    group: Optional[str] = None,
) -> Dict[str, Any]:
    history = [float(v) for v in hc.history_fitness]
    record: Dict[str, Any] = {
        "run_id": run_id,
        "repetition": repetition,
        "seed": seed,
        "group": group,
        "initial_value": history[0] if history else None,
        "final_value": history[-1] if history else None,
        "duration_seconds": round(float(hc.search_duration), 4),
        "iterations": hc.iterations,
        "neighbors_evaluated": hc.neighbors_evaluated,
        "stopped_at_local_optimum": hc.stopped_at_local_optimum,
        "history_fitness": history,
    }
    for attr in HC_OPTIONAL_ATTRS:
        if hasattr(hc, attr):
            record[attr] = _plain(getattr(hc, attr))
    record.update(_state_summary(hc.final_state))
    record.update(_save_states(output_dir, name, hc.initial_state, hc.final_state, "Hill-climbing"))
    return record

def simulated_annealing_record(
    sa: Any,
    output_dir: str,
    name: str,
    run_id: int,
    repetition: int,
    seed: Optional[int] = None,
    group: Optional[str] = None,
) -> Dict[str, Any]:
    current = [float(v) for v in sa.history_current]
    best = [float(v) for v in sa.history_best]
    record: Dict[str, Any] = {
        "run_id": run_id,
        "repetition": repetition,
        "seed": seed,
        "group": group,
        "initial_value": current[0] if current else None,
        "final_value": best[-1] if best else None,
        "duration_seconds": round(float(sa.search_duration), 4),
        "iterations": sa.iterations,
        "history_current": current,
        "history_best": best,
        "history_temperature": [float(v) for v in sa.history_temperature],
        "history_delta": [None if v is None else float(v) for v in sa.history_delta],
        "history_acceptance_probability": [
            None if v is None else float(v) for v in sa.history_acceptance_probability
        ],
    }
    for attr in SA_PARAM_ATTRS:
        if hasattr(sa, attr):
            record[attr] = _plain(getattr(sa, attr))
    record.update(_state_summary(sa.final_state))
    record.update(_save_states(output_dir, name, sa.initial_state, sa.final_state, "Simulated Annealing"))
    return record

def write_summary(
    output_dir: str,
    algorithm: str,
    container: Any,
    total_packages: int,
    runs: Iterable[Dict[str, Any]],
    filename: Optional[str] = None,
    **metadata: Any,
) -> str:
    run_list: List[Dict[str, Any]] = list(runs)
    folder = os.path.join(output_dir, "experiments")
    os.makedirs(folder, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "_", algorithm.lower()).strip("_") or "run"
    path = os.path.join(folder, filename or f"{slug}_summary.json")
    summary = {
        "metadata": {
            "algorithm": algorithm,
            "container": {
                "label": container.label,
                "dimensions": list(container.dimensions),
                "max_capacity": container.max_capacity,
            },
            "total_packages": total_packages,
            "total_runs": len(run_list),
            **metadata,
        },
        "runs": run_list,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return path