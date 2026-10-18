from dataclasses import dataclass 
from typing import List, Tuple 
from .box import Box, PlacedBox 

@dataclass(frozen=True) 
class Container: 
    label: str 
    dimensions: Tuple[int, int, int] 
    max_capacity: int 

    def volume(self) -> int: 
        return self.dimensions[0] * self.dimensions[1] * self.dimensions[2]     


