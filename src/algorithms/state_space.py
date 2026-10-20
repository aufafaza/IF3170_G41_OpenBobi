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

    def __init__(
        self,
        container: Container,
        boxes: Dict[str, Box],
        objective_type: str = "default",
        seed: Optional[int] = None,
    ):
        self.container = container
        self.boxes = boxes
        self.objective_type = objective_type
        self.rng = random.Random(seed)
        self.rotations: Dict[str, List[Vec]] = {l: b.get_rotations() for l, b in boxes.items()}

    def f(self, state: State) -> float:
        return evaluate_fitness(state, self.objective_type)

    def _can_add(self, placed: Dict[str, Placement], weight: int,
                 label: str, pos: Vec, dims: Vec) -> bool:
        W, D, H = self.container.dimensions
        x, y, z = pos
        w, d, h = dims
        box = self.boxes[label]
        if x < 0 or y < 0 or z < 0:
            return False
        if w <= 0 or d <= 0 or h <= 0:
            return False
        if x + w > W or y + d > D or z + h > H:
            return False
        if weight + box.weight > self.container.max_capacity:
            return False

        supported = z == 0
        for lbl, q in placed.items():
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

    def _all_supported(self, placed: Dict[str, Placement]) -> bool:
        for lbl, p in placed.items():
            z = p.position[2]
            ok = z == 0
            for other, q in placed.items():
                if other == lbl:
                    continue
                if not _xy_overlap(p.position, p.dimensions, q.position, q.dimensions):
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

    def _is_valid(self, placed: Dict[str, Placement]) -> bool:
        W, D, H = self.container.dimensions
        items = list(placed.items())
        total_weight = 0
        for lbl, p in items:
            x, y, z = p.position
            w, d, h = p.dimensions
            if x < 0 or y < 0 or z < 0 or x + w > W or y + d > D or z + h > H:
                return False
            total_weight += self.boxes[lbl].weight
        if total_weight > self.container.max_capacity:
            return False
        for i in range(len(items)):
            a = items[i][1]
            for j in range(i + 1, len(items)):
                b = items[j][1]
                if _overlap(a.position, a.dimensions, b.position, b.dimensions):
                    return False
        return self._all_supported(placed)

    def _rotated_states(self, placed: Dict[str, Placement]) -> List[State]:
        states: List[State] = []
        for lbl, p in placed.items():
            for dims in quarter_turns(p.dimensions):
                new = dict(placed)
                new[lbl] = Placement(p.position, dims)
                if self._is_valid(new):
                    states.append(State(self.container, self.boxes, new))
        return states

    def _exchanged_states(self, placed: Dict[str, Placement]) -> List[State]:
        states: List[State] = []
        labels = list(placed)
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                a, b = labels[i], labels[j]
                pa, pb = placed[a], placed[b]
                if pa.position == pb.position:
                    continue
                new = dict(placed)
                new[a] = Placement(pb.position, pa.dimensions)
                new[b] = Placement(pa.position, pb.dimensions)
                if self._is_valid(new):
                    states.append(State(self.container, self.boxes, new))
        return states

    def _points(self, placed: Dict[str, Placement]) -> List[Vec]:
        W, D, H = self.container.dimensions
        pts: Set[Vec] = {(0, 0, 0)}
        for p in placed.values():
            (x, y, z), (w, d, h) = p.position, p.dimensions
            for pt in ((x + w, y, z), (x, y + d, z), (x, y, z + h)):
                if pt[0] < W and pt[1] < D and pt[2] < H:
                    pts.add(pt)
        return sorted(pts, key=_point_order)

    def generate_initial_state(self) -> State:
        placed: Dict[str, Placement] = {}
        weight = 0
        labels = list(self.boxes)
        self.rng.shuffle(labels)
        for lbl in labels:
            dims = self.rng.choice(self.rotations[lbl])
            points = self._points(placed)
            self.rng.shuffle(points)
            for pt in points:
                if self._can_add(placed, weight, lbl, pt, dims):
                    placed[lbl] = Placement(pt, dims)
                    weight += self.boxes[lbl].weight
                    break
        return State(self.container, self.boxes, placed)

    def neighbors(self, state: State) -> List[State]:
        result: List[State] = []
        placed = state.placed

        bases: List[Tuple[Optional[str], Dict[str, Placement]]] = [(None, placed)]
        for lbl in placed:
            rest = {k: v for k, v in placed.items() if k != lbl}
            if self._all_supported(rest):
                bases.append((lbl, rest))

        for removed, base in bases:
            if removed is not None:
                result.append(State(self.container, self.boxes, base))

            weight = sum(self.boxes[l].weight for l in base)
            points = self._points(base)
            old = placed.get(removed) if removed is not None else None
            for lbl in self.boxes:
                if lbl in base:
                    continue
                for dims in self.rotations[lbl]:
                    for pt in points:
                        if old is not None and lbl == removed and pt == old.position and dims == old.dimensions:
                            continue
                        if self._can_add(base, weight, lbl, pt, dims):
                            new = dict(base)
                            new[lbl] = Placement(pt, dims)
                            result.append(State(self.container, self.boxes, new))

        result.extend(self._rotated_states(placed))
        result.extend(self._exchanged_states(placed))
        return result

    def random_neighbor(self, state: State, max_tries: int = 50) -> Optional[State]:
        placed = state.placed
        outside = state.outside

        for _ in range(max_tries):
            move = self.rng.choice(["add", "remove", "move", "swap", "rotate", "exchange"])

            if move == "rotate":
                if not placed:
                    continue
                lbl = self.rng.choice(list(placed))
                options = quarter_turns(placed[lbl].dimensions)
                if not options:
                    continue
                new = dict(placed)
                new[lbl] = Placement(placed[lbl].position, self.rng.choice(options))
                if self._is_valid(new):
                    return State(self.container, self.boxes, new)
                continue

            if move == "exchange":
                if len(placed) < 2:
                    continue
                a, b = self.rng.sample(list(placed), 2)
                pa, pb = placed[a], placed[b]
                if pa.position == pb.position:
                    continue
                new = dict(placed)
                new[a] = Placement(pb.position, pa.dimensions)
                new[b] = Placement(pa.position, pb.dimensions)
                if self._is_valid(new):
                    return State(self.container, self.boxes, new)
                continue

            if move == "add":
                if not outside:
                    continue
                rest = placed
                lbl = self.rng.choice(outside)
            elif move == "remove":
                if not placed:
                    continue
                removed = self.rng.choice(list(placed))
                rest = {k: v for k, v in placed.items() if k != removed}
                if self._all_supported(rest):
                    return State(self.container, self.boxes, rest)
                continue
            elif move == "move":
                if not placed:
                    continue
                lbl = self.rng.choice(list(placed))
                rest = {k: v for k, v in placed.items() if k != lbl}
            else:
                if not placed or not outside:
                    continue
                removed = self.rng.choice(list(placed))
                rest = {k: v for k, v in placed.items() if k != removed}
                lbl = self.rng.choice(outside)

            if move != "add" and not self._all_supported(rest):
                continue

            dims = self.rng.choice(self.rotations[lbl])
            pt = self.rng.choice(self._points(rest))
            if move == "move" and placed[lbl].position == pt and placed[lbl].dimensions == dims:
                continue

            weight = sum(self.boxes[l].weight for l in rest)
            if self._can_add(rest, weight, lbl, pt, dims):
                new = dict(rest)
                new[lbl] = Placement(pt, dims)
                return State(self.container, self.boxes, new)

        return None

def state_resolution(state: State, objective_type: str = "default") -> Tuple[bool, float]:
    global state_score_trig
    space = StateSpace(state.container, state.boxes, objective_type)
    valid = space._is_valid(state.placed)
    score = space.f(state) if valid else 0.0
    state_score_trig = score
    return valid, score