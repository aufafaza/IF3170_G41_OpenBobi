import json
import os
from typing import Any, Dict, List, Optional, Tuple

from ..models.box import Box
from ..models.container import Container
from ..models.state import State

AXIS_NAMES = "wlh"


def orientation_code(original: Tuple[int, int, int], placed: Tuple[int, int, int]) -> Optional[str]:
    remaining = [0, 1, 2]
    code: List[str] = []
    for value in placed:
        chosen = None
        for index in remaining:
            if original[index] == value:
                chosen = index
                break
        if chosen is None:
            return None
        remaining.remove(chosen)
        code.append(AXIS_NAMES[chosen])
    return "".join(code)


def _to_int(raw: Any, what: str, minimum: int) -> int:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f"{what} harus berupa bilangan bulat, ditemukan {raw!r}")
    if isinstance(raw, float) and not raw.is_integer():
        raise ValueError(f"{what} harus berupa bilangan bulat, ditemukan {raw!r}")
    value = int(raw)
    if value < minimum:
        raise ValueError(f"{what} harus bernilai minimal {minimum}, ditemukan {value}")
    return value


def _dimensions(raw: Any, what: str) -> Tuple[int, int, int]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise ValueError(f"{what} harus berisi tepat 3 angka (w, l, h)")
    return tuple(_to_int(v, f"{what}[{i}]", 1) for i, v in enumerate(raw))


def load_instance(filepath: str) -> Tuple[Container, Dict[str, Box]]:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    try:
        truck_data = data["truck"]
        container = Container(
            label=str(truck_data.get("label", "Truck")),
            dimensions=_dimensions(truck_data["dimensions"], "dimensi truk"),
            max_capacity=_to_int(truck_data["max_capacity"], "max_capacity", 1),
        )

        raw_packages = data["packages"]
        if not isinstance(raw_packages, list) or not raw_packages:
            raise ValueError("daftar packages harus berupa list yang tidak kosong")

        boxes: Dict[str, Box] = {}
        for pkg in raw_packages:
            pkg_id = str(pkg["id"])
            if pkg_id in boxes:
                raise ValueError(f"id paket duplikat: {pkg_id}")
            fragile = pkg.get("is_fragile", False)
            if not isinstance(fragile, bool):
                raise ValueError(f"{pkg_id}: is_fragile harus true atau false, ditemukan {fragile!r}")
            boxes[pkg_id] = Box(
                label=pkg_id,
                dimensions=_dimensions(pkg["dimensions"], f"dimensi {pkg_id}"),
                value=_to_int(pkg["value"], f"value {pkg_id}", 0),
                weight=_to_int(pkg["weight"], f"weight {pkg_id}", 0),
                is_fragile=fragile,
                eta=_to_int(pkg.get("eta", 0), f"eta {pkg_id}", 0),
            )
    except KeyError as exc:
        raise ValueError(f"field wajib tidak ditemukan pada instance: {exc}") from exc
    except (TypeError, AttributeError) as exc:
        raise ValueError(f"struktur instance tidak valid ({exc})") from exc

    return container, boxes


def save_instance(filepath: str, container: Container, boxes: Dict[str, Box]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    data = {
        "truck": {
            "label": container.label,
            "dimensions": list(container.dimensions),
            "max_capacity": container.max_capacity
        },
        "packages": [
            {
                "id": b.label,
                "dimensions": list(b.dimensions),
                "value": b.value,
                "weight": b.weight,
                "is_fragile": b.is_fragile,
                "eta": b.eta
            }
            for b in boxes.values()
        ]
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_solution(filepath: str, state: State, metadata: Optional[Dict[str, Any]] = None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    solution_data = {
        "metadata": metadata or {},
        "truck": {
            "label": state.container.label,
            "dimensions": list(state.container.dimensions),
            "max_capacity": state.container.max_capacity,
            "total_weight": state.total_weight,
            "total_value": state.total_value,
            "volume_utilized": state.volume_utilized,
            "volume_utilization_ratio": round(state.volume_utilization_ratio * 100, 2),
            "weight_utilization_ratio": round(state.weight_utilization_ratio * 100, 2),
        },
        "placed_packages": [
            {
                "id": label,
                "position": list(placement.position),
                "dimensions": list(placement.dimensions),
                "original_dimensions": list(state.boxes[label].dimensions),
                "orientation": orientation_code(state.boxes[label].dimensions, placement.dimensions),
                "value": state.boxes[label].value,
                "weight": state.boxes[label].weight,
                "is_fragile": state.boxes[label].is_fragile,
                "eta": state.boxes[label].eta,
            }
            for label, placement in state.placed.items()
        ],
        "outside_packages": [
            {
                "id": label,
                "dimensions": list(state.boxes[label].dimensions),
                "original_dimensions": list(state.boxes[label].dimensions),
                "orientation": AXIS_NAMES,
                "value": state.boxes[label].value,
                "weight": state.boxes[label].weight,
                "is_fragile": state.boxes[label].is_fragile,
                "eta": state.boxes[label].eta,
            }
            for label in state.outside
        ]
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(solution_data, f, indent=2)