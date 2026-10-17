"""
Instance generator for 3D Container Loading Problem.
Generates randomized problem instances with custom truck dimensions,
weights, values, and fragility.
"""

import random
from typing import Dict, Tuple
from ..models.box import Box
from ..models.container import Container


def generate_instance(
    num_boxes: int = 25,
    container_dims: Tuple[int, int, int] = (10, 20, 5),
    max_capacity: int = 500,
    fragile_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[Container, Dict[str, Box]]:
    rng = random.Random(seed)
    container = Container(
        label="MejiroTruck",
        dimensions=container_dims,
        max_capacity=max_capacity
    )

    boxes: Dict[str, Box] = {}

    boxes["RusdiPackage"] = Box(
        label="RusdiPackage",
        dimensions=(3, 4, 1),
        value=60,
        weight=5,
        is_fragile=False,
        eta=3
    )

    c_w, c_d, c_h = container_dims
    for i in range(1, num_boxes):
        label = f"PKG_{i:02d}"
        w = rng.randint(1, max(1, c_w // 2))
        d = rng.randint(1, max(1, c_d // 2))
        h = rng.randint(1, max(1, c_h // 2))
        
        vol = w * d * h
        weight = max(1, int(vol * rng.uniform(0.5, 2.0)))
        value = int(vol * rng.uniform(2.0, 8.0) + weight * rng.uniform(1.0, 3.0))
        is_fragile = rng.random() < fragile_ratio
        eta = rng.randint(1, 7)

        boxes[label] = Box(
            label=label,
            dimensions=(w, d, h),
            value=value,
            weight=weight,
            is_fragile=is_fragile,
            eta=eta
        )

    return container, boxes
