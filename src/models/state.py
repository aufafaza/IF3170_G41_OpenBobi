from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from .box import Box, PlacedBox 
from .container import Container, Space 

state_score_trig: float = 0.0


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
        return State(self.container, self.boxes, dict(self.placed), list(self.spaces)) 


def state_resolution(state: State) -> State:
    W, D, H = state.container.dimensions
    new_placed: Dict[str, Placement] = {}

    sorted_items = sorted(
        state.placed.items(),
        key=lambda item: (item[1].position[2], -state.boxes[item[0]].value)
    )

    for label, placement in sorted_items:
        box = state.boxes[label]
        w, d, h = placement.dimensions
        x, y, z = placement.position

        if x < 0 or y < 0 or z < 0 or x + w > W or y + d > D or z + h > H:
            continue

        b1 = PlacedBox(box, (x, y, z), (w, d, h))
        has_overlap = False
        for acc_label, acc_p in new_placed.items():
            b2 = PlacedBox(state.boxes[acc_label], acc_p.position, acc_p.dimensions)
            if b1.check_overlap(b2):
                has_overlap = True
                break
        if has_overlap:
            continue

        if z > 0:
            supported_area = 0
            rests_on_fragile = False

            for acc_label, acc_p in new_placed.items():
                acc_box = state.boxes[acc_label]
                xj, yj, zj = acc_p.position
                wj, dj, hj = acc_p.dimensions

                if zj + hj == z:
                    x_start = max(x, xj)
                    x_end = min(x + w, xj + wj)
                    y_start = max(y, yj)
                    y_end = min(y + d, yj + dj)

                    if x_start < x_end and y_start < y_end:
                        if acc_box.is_fragile:
                            rests_on_fragile = True
                            break
                        supported_area += (x_end - x_start) * (y_end - y_start)

            if rests_on_fragile or supported_area <= 0:
                candidate_z = None
                possible_z_levels = [0]
                for acc_label, acc_p in new_placed.items():
                    top = acc_p.position[2] + acc_p.dimensions[2]
                    if top < z:
                        possible_z_levels.append(top)
                
                possible_z_levels.sort(reverse=True)
                for test_z in possible_z_levels:
                    test_b = PlacedBox(box, (x, y, test_z), (w, d, h))
                    collides = any(
                        test_b.check_overlap(PlacedBox(state.boxes[al], ap.position, ap.dimensions))
                        for al, ap in new_placed.items()
                    )
                    if collides:
                        continue
                    if test_z == 0:
                        candidate_z = 0
                        break
                    else:
                        area = 0
                        on_frag = False
                        for al, ap in new_placed.items():
                            ab = state.boxes[al]
                            if ap.position[2] + ap.dimensions[2] == test_z:
                                xs = max(x, ap.position[0])
                                xe = min(x + w, ap.position[0] + ap.dimensions[0])
                                ys = max(y, ap.position[1])
                                ye = min(y + d, ap.position[1] + ap.dimensions[1])
                                if xs < xe and ys < ye:
                                    if ab.is_fragile:
                                        on_frag = True
                                        break
                                    area += (xe - xs) * (ye - ys)
                        if area > 0 and not on_frag:
                            candidate_z = test_z
                            break
                
                if candidate_z is not None:
                    z = candidate_z
                    placement = Placement((x, y, z), (w, d, h))
                else:
                    continue

        new_placed[label] = placement

    max_cap = state.container.max_capacity
    total_w = sum(state.boxes[lbl].weight for lbl in new_placed)

    if total_w > max_cap:
        density_sorted = sorted(
            new_placed.keys(),
            key=lambda lbl: state.boxes[lbl].value / max(1, state.boxes[lbl].weight)
        )
        for remove_lbl in density_sorted:
            if total_w <= max_cap:
                break
            del new_placed[remove_lbl]
            total_w -= state.boxes[remove_lbl].weight

    resolved_state = State(
        container=state.container,
        boxes=state.boxes,
        placed=new_placed,
        spaces=list(state.spaces)
    )
    return resolved_state
