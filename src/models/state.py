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
    container: Container 
    boxes: Dict[str, Box]
    placed: Dict[str, Placement] = field(default_factory=dict) 

    @property 
    def outside(self) -> List[str]: 
        return [i for i in self.boxes if i not in self.placed] 

    @property 
    def total_value(self) -> int: 
        return sum(self.boxes[i].value for i in self.placed)

    @property 
    def total_weight(self) -> int: 
        return sum(self.boxes[i].weight for i in self.placed) 

    @property
    def volume_utilized(self) -> int:
        return sum(
            p.dimensions[0] * p.dimensions[1] * p.dimensions[2] 
            for p in self.placed.values()
        )

    @property
    def volume_utilization_ratio(self) -> float:
        c_vol = self.container.volume()
        return self.volume_utilized / c_vol if c_vol > 0 else 0.0

    @property
    def weight_utilization_ratio(self) -> float:
        c_max = self.container.max_capacity
        return self.total_weight / c_max if c_max > 0 else 0.0

    def copy(self) -> "State": 
        return State(self.container, self.boxes, dict(self.placed))
