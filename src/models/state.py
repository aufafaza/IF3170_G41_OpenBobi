from dataclasses import dataclass, field
from typing import List, Tuple, Dict
from .box import Box, PlacedBox 
from .container import Container, Space 

@dataclass 
class Placement: 
    position: Tuple[int, int, int] 
    dimensions: Tuple[int, int, int] 

@dataclass
class State: 
    container: Container 
    boxes: Dict[str, Box]
    placed: Dict[str, Placement] = field(default_factory=dict) 

    spaces: List[Space] = field(default_factory=list) 

    @property 
    def outside(self) -> List[str]: 
        return [i for i in self.boxes if i not in self.placed] 

    @property 
    def total_value(self) -> int: 
        return sum(self.boxes[i].value for i in self.placed)

    @property 
    def total_weight(self) -> int: 
        return sum(self.boxes[i].weight for i in self.placed) 

    def copy(self) -> "State": 
        return State(self.container, self.boxes, dict(self.placed), list(self.spaces)) 



