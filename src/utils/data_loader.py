"""
Data loading and persistence utilities for Container Loading Problem.
Supports JSON format for problem instances and solution outputs.
"""

import json
import os
from typing import Dict, Tuple, Any, Optional
from ..models.box import Box
from ..models.container import Container
from ..models.state import State, Placement


def load_instance(filepath: str) -> Tuple[Container, Dict[str, Box]]:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    truck_data = data["truck"]
    container = Container(
        label=truck_data.get("label", "Truck"),
        dimensions=tuple(truck_data["dimensions"]),
        max_capacity=int(truck_data["max_capacity"])
    )

    boxes: Dict[str, Box] = {}
    for pkg in data["packages"]:
        pkg_id = pkg["id"]
        box = Box(
            label=pkg_id,
            dimensions=tuple(pkg["dimensions"]),
            value=int(pkg["value"]),
            weight=int(pkg["weight"]),
            is_fragile=bool(pkg.get("is_fragile", False)),
            eta=int(pkg.get("eta", 0))
        )
        boxes[pkg_id] = box

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
