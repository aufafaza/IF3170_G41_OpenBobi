import random
from typing import Dict, List, Optional, Set, Tuple
from ..models.box import Box
from ..models.container import Container
from ..models.state import State, Placement
from .objective import evaluate_fitness

Vec = Tuple[int, int, int]
state_score_trig: float = 0.0

def _xy_overlap(p1: Vec, d1: Vec, p2: Vec, d2: Vec) -> bool:
    return (p1[0] < p2[0] + d2[0] and p2[0] < p1[0] + d1[0]
            and p1[1] < p2[1] + d2[1] and p2[1] < p1[1] + d1[1])

def _overlap(p1: Vec, d1: Vec, p2: Vec, d2: Vec) -> bool:
    return _xy_overlap(p1, d1, p2, d2) and p1[2] < p2[2] + d2[2] and p2[2] < p1[2] + d1[2]

def _point_order(point: Vec) -> Tuple[int, int, int]:
    return (point[2], point[1], point[0])

def quarter_turns(dims: Vec) -> List[Vec]:
    w, d, h = dims
    options: List[Vec] = []
    for candidate in ((d, w, h), (w, h, d), (h, d, w)):
        if candidate != dims and candidate not in options:
            options.append(candidate)
    return options

class StateSpace:
    def __init__(self, containers: List[Container], boxes: Dict[str, Box], objective_type: str = "default", seed: Optional[int] = None):
        self.containers = containers
        self.boxes = boxes
        self.objective_type = objective_type
        self.rng = random.Random(seed)
        self.rotations: Dict[str, List[Vec]] = {l: b.get_rotations() for l, b in boxes.items()}

    def f(self, state: State) -> float:
        return evaluate_fitness(state, self.objective_type)

    def _can_add(self, placed_in_truck: Dict[str, Placement], weight: int, truck: Container, label: str, pos: Vec, dims: Vec) -> bool:
        W, D, H = truck.dimensions
        x, y, z = pos
        w, d, h = dims
        box = self.boxes[label]
        if x < 0 or y < 0 or z < 0 or w <= 0 or d <= 0 or h <= 0:
            return False
        if x + w > W or y + d > D or z + h > H or weight + box.weight > truck.max_capacity:
            return False

        supported = z == 0
        for lbl, q in placed_in_truck.items():
            if _overlap(pos, dims, q.position, q.dimensions):
                return False
            if not _xy_overlap(pos, dims, q.position, q.dimensions):
                continue
            top = q.position[2] + q.dimensions[2]
            if top <= z:
                if self.boxes[lbl].is_fragile:
                    return False
                if top == z:
                    supported = True
            elif box.is_fragile:
                return False
        return supported

    def _all_supported(self, placed_in_truck: Dict[str, Placement]) -> bool:
        for lbl, p in placed_in_truck.items():
            z = p.position[2]
            ok = z == 0
            for other, q in placed_in_truck.items():
                if other == lbl or not _xy_overlap(p.position, p.dimensions, q.position, q.dimensions):
                    continue
                top = q.position[2] + q.dimensions[2]
                if top <= z:
                    if self.boxes[other].is_fragile:
                        return False
                    if top == z:
                        ok = True
            if not ok:
                return False
        return True

    def _is_valid(self, placed_all: Dict[str, Dict[str, Placement]]) -> bool:
        for truck in self.containers:
            placed_in_truck = placed_all.get(truck.label, {})
            W, D, H = truck.dimensions
            items = list(placed_in_truck.items())
            total_weight = 0
            for lbl, p in items:
                x, y, z = p.position
                w, d, h = p.dimensions
                if x < 0 or y < 0 or z < 0 or x + w > W or y + d > D or z + h > H:
                    return False
                total_weight += self.boxes[lbl].weight
            if total_weight > truck.max_capacity:
                return False
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    a, b = items[i][1], items[j][1]
                    if _overlap(a.position, a.dimensions, b.position, b.dimensions):
                        return False
            if not self._all_supported(placed_in_truck):
                return False
        return True

    def _points(self, placed_in_truck: Dict[str, Placement], truck: Container) -> List[Vec]:
        W, D, H = truck.dimensions
        pts: Set[Vec] = {(0, 0, 0)}
        for p in placed_in_truck.values():
            (x, y, z), (w, d, h) = p.position, p.dimensions
            for pt in ((x + w, y, z), (x, y + d, z), (x, y, z + h)):
                if pt[0] < W and pt[1] < D and pt[2] < H:
                    pts.add(pt)
        return sorted(pts, key=_point_order)

    def generate_initial_state(self) -> State:
        state = State(self.containers, self.boxes)
        labels = list(self.boxes)
        self.rng.shuffle(labels)
        for lbl in labels:
            dims = self.rng.choice(self.rotations[lbl])
            trucks = list(self.containers)
            self.rng.shuffle(trucks)
            for truck in trucks:
                placed = state.placed[truck.label]
                weight = sum(self.boxes[l].weight for l in placed)
                points = self._points(placed, truck)
                self.rng.shuffle(points)
                placed_success = False
                for pt in points:
                    if self._can_add(placed, weight, truck, lbl, pt, dims):
                        placed[lbl] = Placement(pt, dims)
                        placed_success = True
                        break
                if placed_success:
                    break
        return state

    def _rotated_states(self, placed_all: Dict[str, Dict[str, Placement]]) -> List[State]:
        states: List[State] = []
        for truck in self.containers:
            placed = placed_all[truck.label]
            for lbl, p in placed.items():
                for dims in quarter_turns(p.dimensions):
                    new_placed = {t: dict(pl) for t, pl in placed_all.items()}
                    new_placed[truck.label][lbl] = Placement(p.position, dims)
                    if self._is_valid(new_placed):
                        states.append(State(self.containers, self.boxes, new_placed))
        return states

    def _exchanged_states(self, placed_all: Dict[str, Dict[str, Placement]]) -> List[State]:
        states: List[State] = []
        for i, t1 in enumerate(self.containers):
            for j, t2 in enumerate(self.containers):
                if i > j: continue
                p1 = placed_all[t1.label]
                p2 = placed_all[t2.label]
                for a in p1:
                    for b in p2:
                        if t1.label == t2.label and a >= b: continue
                        pa, pb = p1[a], p2[b]
                        if t1.label == t2.label and pa.position == pb.position: continue
                        new_placed = {t: dict(pl) for t, pl in placed_all.items()}
                        new_placed[t1.label][a] = Placement(pb.position, pa.dimensions)
                        new_placed[t2.label][b] = Placement(pa.position, pb.dimensions)
                        if self._is_valid(new_placed):
                            states.append(State(self.containers, self.boxes, new_placed))
        return states

    def neighbors(self, state: State) -> List[State]:
        result: List[State] = []
        placed = state.placed

        for t1 in self.containers:
            for lbl in list(placed[t1.label].keys()):
                rest_placed = {t: dict(p) for t, p in placed.items()}
                del rest_placed[t1.label][lbl]
                if not self._all_supported(rest_placed[t1.label]):
                    continue
                for t2 in self.containers:
                    weight = sum(self.boxes[l].weight for l in rest_placed[t2.label])
                    points = self._points(rest_placed[t2.label], t2)
                    for dims in self.rotations[lbl]:
                        for pt in points:
                            if self._can_add(rest_placed[t2.label], weight, t2, lbl, pt, dims):
                                new_placed = {t: dict(p) for t, p in rest_placed.items()}
                                new_placed[t2.label][lbl] = Placement(pt, dims)
                                result.append(State(self.containers, self.boxes, new_placed))

        outside = state.outside
        for lbl in outside:
            for t in self.containers:
                weight = sum(self.boxes[l].weight for l in placed[t.label])
                points = self._points(placed[t.label], t)
                for dims in self.rotations[lbl]:
                    for pt in points:
                        if self._can_add(placed[t.label], weight, t, lbl, pt, dims):
                            new_placed = {t_dict: dict(p) for t_dict, p in placed.items()}
                            new_placed[t.label][lbl] = Placement(pt, dims)
                            result.append(State(self.containers, self.boxes, new_placed))

        result.extend(self._rotated_states(placed))
        result.extend(self._exchanged_states(placed))
        return result

    def random_neighbor(self, state: State, max_tries: int = 50) -> Optional[State]:
        outside = state.outside
        for _ in range(max_tries):
            move = self.rng.choice(["add", "remove", "move", "swap", "rotate", "exchange"])
            new_placed = {t: dict(p) for t, p in state.placed.items()}
            all_placed = {lbl for t_placed in new_placed.values() for lbl in t_placed}

            if move == "rotate":
                if not all_placed: continue
                lbl = self.rng.choice(list(all_placed))
                t = next(t for t, p in new_placed.items() if lbl in p)
                options = quarter_turns(new_placed[t][lbl].dimensions)
                if not options: continue
                new_placed[t][lbl] = Placement(new_placed[t][lbl].position, self.rng.choice(options))
                if self._is_valid(new_placed): return State(self.containers, self.boxes, new_placed)
                continue

            if move == "exchange":
                if len(all_placed) < 2: continue
                a, b = self.rng.sample(list(all_placed), 2)
                ta = next(t for t, p in new_placed.items() if a in p)
                tb = next(t for t, p in new_placed.items() if b in p)
                pa, pb = new_placed[ta][a], new_placed[tb][b]
                new_placed[ta][a] = Placement(pb.position, pa.dimensions)
                new_placed[tb][b] = Placement(pa.position, pb.dimensions)
                if self._is_valid(new_placed): return State(self.containers, self.boxes, new_placed)
                continue

            if move == "add":
                if not outside: continue
                lbl = self.rng.choice(outside)
                truck = self.rng.choice(self.containers)
                rest = new_placed[truck.label]
                dims = self.rng.choice(self.rotations[lbl])
                pt = self.rng.choice(self._points(rest, truck))
                if self._can_add(rest, sum(self.boxes[l].weight for l in rest), truck, lbl, pt, dims):
                    rest[lbl] = Placement(pt, dims)
                    return State(self.containers, self.boxes, new_placed)
                continue

            if not all_placed: continue
            lbl = self.rng.choice(list(all_placed))
            t_old = next(t for t, p in new_placed.items() if lbl in p)
            rest_old = {k: v for k, v in new_placed[t_old].items() if k != lbl}
            
            if move == "remove":
                new_placed[t_old] = rest_old
                if self._is_valid(new_placed): return State(self.containers, self.boxes, new_placed)
                continue

            truck = self.rng.choice(self.containers)
            rest = new_placed[truck.label].copy()
            if lbl in rest: del rest[lbl]
            
            if truck.label != t_old and not self._all_supported(rest_old): continue
            
            dims = self.rng.choice(self.rotations[lbl])
            pt = self.rng.choice(self._points(rest, truck))
            
            if self._can_add(rest, sum(self.boxes[l].weight for l in rest), truck, lbl, pt, dims):
                new_placed[t_old] = rest_old
                new_placed[truck.label] = rest
                new_placed[truck.label][lbl] = Placement(pt, dims)
                if self._is_valid(new_placed): return State(self.containers, self.boxes, new_placed)
        return None

def state_resolution(state: State, objective_type: str = "default") -> Tuple[bool, float]:
    global state_score_trig
    space = StateSpace(state.containers, state.boxes, objective_type)
    valid = space._is_valid(state.placed)
    score = space.f(state) if valid else 0.0
    state_score_trig = score
    return valid, score