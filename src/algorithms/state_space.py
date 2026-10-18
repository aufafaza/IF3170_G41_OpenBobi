import random
from typing import Dict, List, Optional, Set, Tuple

from ..models.box import Box
from ..models.container import Container
from ..models.state import State, Placement
from .objective import evaluate_fitness

Vec = Tuple[int, int, int]


def _xy_overlap(p1: Vec, d1: Vec, p2: Vec, d2: Vec) -> bool:
    return (p1[0] < p2[0] + d2[0] and p2[0] < p1[0] + d1[0]
            and p1[1] < p2[1] + d2[1] and p2[1] < p1[1] + d1[1])


def _overlap(p1: Vec, d1: Vec, p2: Vec, d2: Vec) -> bool:
    return _xy_overlap(p1, d1, p2, d2) and p1[2] < p2[2] + d2[2] and p2[2] < p1[2] + d1[2]


def _point_order(point: Vec) -> Tuple[int, int, int]:
    return (point[2], point[1], point[0])


class StateSpace:
    """
    shared search space for the local search algorithms.

    state: a State (which boxes are placed, where, and in which orientation).
    every state generated here is feasible.

    moves: ADD an outside box, REMOVE a placed box, MOVE a placed box,
    SWAP a placed box for an outside box.
    """

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
        if x + w > W or y + d > D or z + h > H:
            return False
        if weight + box.weight > self.container.max_capacity:
            return False

        supported = z == 0
        for lbl, q in placed.items():
            if _overlap(pos, dims, q.position, q.dimensions):
                return False
            qz, qh = q.position[2], q.dimensions[2]
            if qz + qh == z and _xy_overlap(pos, dims, q.position, q.dimensions):
                if self.boxes[lbl].is_fragile:
                    return False
                supported = True
            if box.is_fragile and qz == z + h and _xy_overlap(pos, dims, q.position, q.dimensions):
                return False
        return supported

    def _all_supported(self, placed: Dict[str, Placement]) -> bool:
        for p in placed.values():
            z = p.position[2]
            if z == 0:
                continue
            ok = False
            for lbl, q in placed.items():
                if q.position[2] + q.dimensions[2] == z and _xy_overlap(
                        p.position, p.dimensions, q.position, q.dimensions):
                    if self.boxes[lbl].is_fragile:
                        return False
                    ok = True
            if not ok:
                return False
        return True

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

        return result

    def random_neighbor(self, state: State, max_tries: int = 50) -> Optional[State]:
        placed = state.placed
        outside = state.outside

        for _ in range(max_tries):
            move = self.rng.choice(["add", "remove", "move", "swap"])

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
