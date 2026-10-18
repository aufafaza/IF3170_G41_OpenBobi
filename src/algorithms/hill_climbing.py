import time
from typing import Callable, Dict, List, Optional

from ..models.box import Box
from ..models.container import Container
from ..models.state import State
from .state_space import StateSpace


class SteepestAscentHillClimbing(StateSpace):

    def __init__(
        self,
        container: Container,
        boxes: Dict[str, Box],
        max_iterations: int = 100,
        objective_type: str = "default",
        seed: Optional[int] = None,
    ):
        super().__init__(container, boxes, objective_type, seed)
        self.max_iterations = max(1, max_iterations)

        self.iterations: int = 0
        self.neighbors_evaluated: int = 0
        self.stopped_at_local_optimum: bool = False
        self.history_fitness: List[float] = []
        self.initial_state: Optional[State] = None
        self.final_state: Optional[State] = None
        self.search_duration: float = 0.0

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
