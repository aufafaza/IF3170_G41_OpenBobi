from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from .box import Box, PlacedBox 
from .container import Container 

@dataclass 
class Placement: 
    position: Tuple[int, int, int] 
    dimensions: Tuple[int, int, int] 

@dataclass
class State: 
    containers: List[Container]
    boxes: Dict[str, Box]
    # placed: {truck_label: {box_label: Placement}}
    placed: Dict[str, Dict[str, Placement]] = field(default_factory=dict) 

    def __post_init__(self):
        if not self.placed:
            for c in self.containers:
                self.placed[c.label] = {}

    @property 
    def outside(self) -> List[str]: 
        placed_labels = set()
        for truck_placed in self.placed.values():
            placed_labels.update(truck_placed.keys())
        return [i for i in self.boxes if i not in placed_labels] 

    @property 
    def total_value(self) -> int: 
        total = 0
        for truck_placed in self.placed.values():
            for label in truck_placed:
                total += self.boxes[label].value
        return total

    @property 
    def total_weight(self) -> int: 
        total = 0
        for truck_placed in self.placed.values():
            for label in truck_placed:
                total += self.boxes[label].weight
        return total

    @property
    def volume_utilized(self) -> int:
        total = 0
        for truck_placed in self.placed.values():
            for p in truck_placed.values():
                total += p.dimensions[0] * p.dimensions[1] * p.dimensions[2]
        return total

    @property
    def volume_utilization_ratio(self) -> float:
        total_vol = sum(c.volume() for c in self.containers)
        return self.volume_utilized / total_vol if total_vol > 0 else 0.0

    @property
    def weight_utilization_ratio(self) -> float:
        total_cap = sum(c.max_capacity for c in self.containers)
        return self.total_weight / total_cap if total_cap > 0 else 0.0

    def copy(self) -> "State": 
        new_placed = {truck: dict(pkgs) for truck, pkgs in self.placed.items()}
        return State(self.containers, self.boxes, new_placed)