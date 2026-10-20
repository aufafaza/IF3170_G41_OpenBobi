import time
from typing import Callable, Dict, List, Optional, Tuple

from ..models.box import Box
from ..models.container import Container
from ..models.state import State
from .state_space import StateSpace

EPS = 1e-9

# (delta_f, neighbour state)
Scored = List[Tuple[float, State]]


class HillClimbing(StateSpace):

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
        self.sideways_moves: int = 0
        self.stopped_at_local_optimum: bool = False
        self.history_fitness: List[float] = []
        self.initial_state: Optional[State] = None
        self.final_state: Optional[State] = None
        self.search_duration: float = 0.0

    def choose_move(self, scored: Scored) -> Optional[Tuple[float, State]]:
        raise NotImplementedError

    def run(self, progress_callback: Optional[Callable[[int, float], None]] = None) -> State:
        start_time = time.perf_counter()

        current = self.generate_initial_state()
        current_f = self.f(current)
        self.initial_state = current

        self.iterations = 0
        self.neighbors_evaluated = 0
        self.sideways_moves = 0
        self.stopped_at_local_optimum = False
        self.history_fitness = [current_f]
        if progress_callback:
            progress_callback(0, current_f)

        while self.iterations < self.max_iterations:
            scored: Scored = []
            for neighbor in self.neighbors(current):
                self.neighbors_evaluated += 1
                scored.append((self.f(neighbor) - current_f, neighbor))

            move = self.choose_move(scored)
            if move is None:
                self.stopped_at_local_optimum = True
                break

            delta, current = move
            current_f += delta
            if abs(delta) <= EPS:
                self.sideways_moves += 1
            self.iterations += 1
            self.history_fitness.append(current_f)
            if progress_callback:
                progress_callback(self.iterations, current_f)

        self.final_state = current
        self.search_duration = time.perf_counter() - start_time
        return current


class SteepestAscentHillClimbing(HillClimbing):

    def choose_move(self, scored: Scored) -> Optional[Tuple[float, State]]:
        best: Optional[Tuple[float, State]] = None
        for delta, neighbor in scored:
            if delta > EPS and (best is None or delta > best[0]):
                best = (delta, neighbor)
        return best


class StochasticHillClimbing(HillClimbing):

    def choose_move(self, scored: Scored) -> Optional[Tuple[float, State]]:
        uphill = [item for item in scored if item[0] > EPS]
        if not uphill:
            return None
        return self.rng.choice(uphill)


class SidewaysHillClimbing(HillClimbing):

    def __init__(
        self,
        container: Container,
        boxes: Dict[str, Box],
        max_iterations: int = 100,
        max_sideways: int = 10,
        objective_type: str = "default",
        seed: Optional[int] = None,
    ):
        super().__init__(container, boxes, max_iterations, objective_type, seed)
        self.max_sideways = max_sideways
        self.consecutive_sideways = 0

    def run(self, progress_callback: Optional[Callable[[int, float], None]] = None) -> State:
        self.consecutive_sideways = 0
        return super().run(progress_callback)

    def choose_move(self, scored: Scored) -> Optional[Tuple[float, State]]:
        best: Optional[Tuple[float, State]] = None
        for delta, neighbor in scored:
            if delta > EPS and (best is None or delta > best[0]):
                best = (delta, neighbor)
        if best is not None:
            self.consecutive_sideways = 0
            return best

        if self.consecutive_sideways >= self.max_sideways:
            return None
        flat = [item for item in scored if abs(item[0]) <= EPS]
        if not flat:
            return None
        self.consecutive_sideways += 1
        return self.rng.choice(flat)


class RandomRestartHillClimbing(StateSpace):

    def __init__(
        self,
        container: Container,
        boxes: Dict[str, Box],
        max_restarts: int = 10,
        max_iterations: int = 100,
        objective_type: str = "default",
        seed: Optional[int] = None,
    ):
        super().__init__(container, boxes, objective_type, seed)
        self.max_restarts = max(1, max_restarts)
        self.max_iterations = max(1, max_iterations)

        self.restarts: int = 0
        self.iterations: int = 0
        self.neighbors_evaluated: int = 0
        self.iterations_per_restart: List[int] = []
        self.fitness_per_restart: List[float] = []
        self.history_best: List[float] = []
        self.initial_state: Optional[State] = None
        self.final_state: Optional[State] = None
        self.search_duration: float = 0.0

    def run(self, progress_callback: Optional[Callable[[int, float], None]] = None) -> State:
        start_time = time.perf_counter()

        self.restarts = 0
        self.iterations = 0
        self.neighbors_evaluated = 0
        self.iterations_per_restart = []
        self.fitness_per_restart = []
        self.history_best = []
        best_state = State(self.container, self.boxes)
        best_f = self.f(best_state)

        for _ in range(self.max_restarts):
            climber = SteepestAscentHillClimbing(
                self.container, self.boxes,
                max_iterations=self.max_iterations,
                objective_type=self.objective_type,
                seed=self.rng.randint(0, 2 ** 31 - 1),
            )
            state = climber.run()
            final_f = self.f(state)

            self.restarts += 1
            self.iterations += climber.iterations
            self.neighbors_evaluated += climber.neighbors_evaluated
            self.iterations_per_restart.append(climber.iterations)
            self.fitness_per_restart.append(final_f)

            if self.restarts == 1 or final_f > best_f:
                best_state, best_f = state, final_f
                self.initial_state = climber.initial_state
            self.history_best.append(best_f)
            if progress_callback:
                progress_callback(self.restarts, best_f)

        self.final_state = best_state
        self.search_duration = time.perf_counter() - start_time
        return best_state
