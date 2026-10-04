from dataclasses import dataclass 
from typing import List, Tuple 

@dataclass 
class Box: 
    label: str
    dimensions: Tuple[int, int, int] 
    value: int 
    weight: int 
    is_fragile: bool = False 
    

    def volume(self) -> int: 
        return self.dimensions[0] * self.dimensions[1] * self.dimensions[2] 

    def get_rotations(self) -> List[Tuple[int, int, int]]: 
        w, d, h = self.dimensions 

        rotations = [ 
            (w, d, h), (w, h, d), 
            (d, w, h), (d, h, w), 
            (h, w, d), (h, d, w) 
        ] 

        seen = set() 
        unique = [] 

        for r in rotations : 
            if r not in seen: 
                seen.add(r) 
                unique.append(r)
        return unique 
 

@dataclass 
class PlacedBox: 
    box: Box 
    position: Tuple[int, int, int] 
    dimensions: Tuple[int, int, int] 

    def get_bounds(self) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]: 
        '''
            Returns: 
                tuple of (xmin, xmax), (ymin, ymax), (zmin, zmax) 
        '''
        x, y, z = self.position 
        w, d, h = self.dimensions
        return ((x, x + w), (y, y + d), (z, z + h)) 

    def check_overlap(self, other: 'PlacedBox') -> bool: 
        b1 = self.get_bounds() 
        b2 = other.get_bounds() 

        x_overlap = b1[0][0] < b2[0][1] and b1[0][1] > b2[0][0]
        y_overlap = b1[1][0] < b2[1][1] and b1[1][1] > b2[1][0] 
        z_overlap = b1[2][0] < b2[2][1] and b1[2][1] > b2[2][0] 
        return x_overlap and y_overlap and z_overlap
    

