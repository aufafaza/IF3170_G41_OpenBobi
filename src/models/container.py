from dataclasses import dataclass 
from typing import List, Tuple 
from .box import Box, PlacedBox 

@dataclass 
class Space: 
    position: Tuple[int, int, int] 
    dimensions: Tuple[int, int, int] 
    
    def volume(self) -> int: 
        return self.dimensions[0] * self.dimensions[1] * self.dimensions[2] 

    def can_fit(self, box_dimensions: Tuple[int, int, int])->bool: 
        return (box_dimensions[0] <= self.dimensions[0] and 
                box_dimensions[1] <= self.dimensions[1] and 
                box_dimensions[2] <= self.dimensions[2]) 


@dataclass(frozen=True) 
class Container: 
    label: str 
    dimensions: Tuple[int, int, int] 
    max_capacity: int 

    def volume(self) -> int: 
        return self.dimensions[0] * self.dimensions[1] * self.dimensions[2]     


