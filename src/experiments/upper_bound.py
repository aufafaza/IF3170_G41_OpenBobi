import math
import sys
from typing import Iterable, List, Tuple

from ..utils.data_loader import load_instance

Item = Tuple[float, float, float]

class _NodeLimit(Exception):
    pass

def _fractional_bound(items: List[Item], index: int, capacity: float) -> float:
    bound = 0.0
    remaining = capacity
    for value, weight, _ in items[index:]:
        if weight <= remaining:
            bound += value
            remaining -= weight
        else:
            bound += value * remaining / weight
            break
    return bound

def bound_from_items(
    items: Iterable[Item],
    capacity_weight: float,
    capacity_volume: float,
    node_limit: int = 2_000_000,
) -> float:
    ordered: List[Item] = sorted(
        items,
        key=lambda item: item[0] / max(item[1], 1),
        reverse=True,
    )
    best = 0
    nodes = 0

    def search(index: int, value: float, weight: float, volume: float) -> None:
        nonlocal best, nodes
        nodes += 1
        if nodes > node_limit:
            raise _NodeLimit
        if value > best:
            best = value
        if index == len(ordered):
            return
        if value + _fractional_bound(ordered, index, capacity_weight - weight) <= best:
            return
        item_value, item_weight, item_volume = ordered[index]
        if weight + item_weight <= capacity_weight and volume + item_volume <= capacity_volume:
            search(index + 1, value + item_value, weight + item_weight, volume + item_volume)
        search(index + 1, value, weight, volume)

    try:
        search(0, 0, 0, 0)
    except _NodeLimit:
        root = _fractional_bound(ordered, 0, capacity_weight)
        return math.floor(root + 1e-9) if math.isfinite(root) else root
    return best

def upper_bound(path: str) -> float:
    containers, boxes = load_instance(path)
    items = [(b.value, b.weight, b.volume()) for b in boxes.values()]
    total_capacity = sum(c.max_capacity for c in containers)
    total_volume = sum(c.volume() for c in containers)
    return bound_from_items(items, total_capacity, total_volume)

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "data/mejiro_express.json"
    print(f"Batas atas value untuk {target}: {upper_bound(target)}")