import random
import time
from typing import Callable, Dict, List, Optional, Set, Tuple

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
    # lowest z first, then y, then x
    return (point[2], point[1], point[0])


class SteepestAscentHillClimbing:
    """
    Initial state: random feasible loading.

    neighborhood of a state (every neighbour is feasible):
      - ADD     an outside box (any rotation) at a candidate point
      - REMOVE  a placed box
      - MOVE    a placed box to another candidate point / rotation
      - SWAP    a placed box out, an outside box in
    each iteration evaluates the whole neighbourhood, computes
    delta_f = f(neighbour) - f(current), and moves to the neighbour with the
    largest delta_f. stops when the best delta_f <= 0 or max_iterations.
    """

    def __init__(
        self,
        container: Container,
        boxes: Dict[str, Box],
        max_iterations: int = 100,
        objective_type: str = "default",
        seed: Optional[int] = None,
    ):
        self.container = container
        self.boxes = boxes
        self.max_iterations = max(1, max_iterations)
        self.objective_type = objective_type
        self.rng = random.Random(seed)
        self.rotations: Dict[str, List[Vec]] = {l: b.get_rotations() for l, b in boxes.items()}

        self.iterations: int = 0
        self.neighbors_evaluated: int = 0
        self.stopped_at_local_optimum: bool = False
        self.history_fitness: List[float] = []
        self.initial_state: Optional[State] = None
        self.final_state: Optional[State] = None
        self.search_duration: float = 0.0

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


    def run(self, progress_callback: Optional[Callable[[int, float], None]] = None) -> State:
        start_time = time.perf_counter()

        current = self.generate_initial_state()
        current_f = self.f(current)
        self.initial_state = current

        self.iterations = 0
        self.neighbors_evaluated = 0
        self.stopped_at_local_optimum = False
        self.history_fitness = [current_f]
        if progress_callback:
            progress_callback(0, current_f)

        while self.iterations < self.max_iterations:
            best_state: Optional[State] = None
            best_delta = 0.0
            for neighbor in self.neighbors(current):
                self.neighbors_evaluated += 1
                delta = self.f(neighbor) - current_f
                if delta > best_delta:
                    best_delta, best_state = delta, neighbor

            if best_state is None:
                self.stopped_at_local_optimum = True
                break

            current, current_f = best_state, current_f + best_delta
            self.iterations += 1
            self.history_fitness.append(current_f)
            if progress_callback:
                progress_callback(self.iterations, current_f)

        self.final_state = current
        self.search_duration = time.perf_counter() - start_time
        return current
