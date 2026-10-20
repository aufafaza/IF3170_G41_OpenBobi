import time
from typing import Callable, Dict, List, Optional, Tuple

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
            best_state: Optional[State] = None
            best_f = current_f
            for neighbor in self.neighbors(current):
                self.neighbors_evaluated += 1
                neighbor_f = self.f(neighbor)
                if neighbor_f > best_f:
                    best_f, best_state = neighbor_f, neighbor

            move = self.choose_move(scored)
            if move is None:
                self.stopped_at_local_optimum = True
                break

            current, current_f = best_state, best_f
            self.iterations += 1
            self.history_fitness.append(current_f)
            if progress_callback:
                progress_callback(self.iterations, current_f)

        self.final_state = current
        self.search_duration = time.perf_counter() - start_time
        return current

class SidewaysMoveHillClimbing(SteepestAscentHillClimbing):

    def __init__(
        self,
        container: Container,
        boxes: Dict[str, Box],
        max_iterations: int = 200,
        max_sideways_moves: int = 25,
        objective_type: str = "default",
        seed: Optional[int] = None,
    ):
        super().__init__(container, boxes, max_iterations, objective_type, seed)
        self.max_sideways_moves = max(0, max_sideways_moves)
        self.sideways_moves: int = 0

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
            best_delta: Optional[float] = None
            best: List[Tuple[State, float]] = []
            for neighbor in self.neighbors(current):
                self.neighbors_evaluated += 1
                neighbor_f = self.f(neighbor)
                delta = neighbor_f - current_f
                if delta < 0:
                    continue
                if best_delta is None or delta > best_delta:
                    best_delta, best = delta, [(neighbor, neighbor_f)]
                elif delta == best_delta:
                    best.append((neighbor, neighbor_f))

            if not best:
                self.stopped_at_local_optimum = True
                break

            if best_delta == 0:
                if self.sideways_moves >= self.max_sideways_moves:
                    self.stopped_at_local_optimum = True
                    break
                self.sideways_moves += 1

            current, current_f = self.rng.choice(best)
            self.iterations += 1
            self.history_fitness.append(current_f)
            if progress_callback:
                progress_callback(self.iterations, current_f)

        self.final_state = current
        self.search_duration = time.perf_counter() - start_time
        return current

class StochasticHillClimbing(StateSpace):

    def __init__(
        self,
        container: Container,
        boxes: Dict[str, Box],
        max_iterations: int = 500,
        max_stagnation: int = 200,
        objective_type: str = "default",
        seed: Optional[int] = None,
    ):
        super().__init__(container, boxes, objective_type, seed)
        self.max_iterations = max(1, max_iterations)
        self.max_stagnation = max(1, max_stagnation)

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
        stagnation = 0
        if progress_callback:
            progress_callback(0, current_f)

        while self.iterations < self.max_iterations:
            self.iterations += 1
            neighbor = self.random_neighbor(current)
            if neighbor is not None:
                self.neighbors_evaluated += 1
                neighbor_f = self.f(neighbor)
                if neighbor_f > current_f:
                    current, current_f = neighbor, neighbor_f
                    stagnation = 0
                else:
                    stagnation += 1
            else:
                stagnation += 1

            self.history_fitness.append(current_f)
            if progress_callback:
                progress_callback(self.iterations, current_f)
            if stagnation >= self.max_stagnation:
                self.stopped_at_local_optimum = True
                break

        self.final_state = current
        self.search_duration = time.perf_counter() - start_time
        return current

class RandomRestartHillClimbing(StateSpace):

    def __init__(
        self,
        container: Container,
        boxes: Dict[str, Box],
        max_restarts: int = 5,
        max_iterations: int = 200,
        objective_type: str = "default",
        seed: Optional[int] = None,
    ):
        super().__init__(container, boxes, objective_type, seed)
        self.max_restarts = max(1, max_restarts)
        self.max_iterations = max(1, max_iterations)

        self.restarts: int = 0
        self.iterations_per_restart: List[int] = []
        self.restart_final_values: List[float] = []
        self.restarts_at_local_optimum: int = 0
        self.iterations: int = 0
        self.neighbors_evaluated: int = 0
        self.stopped_at_local_optimum: bool = False
        self.history_fitness: List[float] = []
        self.history_raw: List[float] = []
        self.initial_state: Optional[State] = None
        self.final_state: Optional[State] = None
        self.search_duration: float = 0.0

    def run(self, progress_callback: Optional[Callable[[int, float], None]] = None) -> State:
        start_time = time.perf_counter()

        self.restarts = 0
        self.iterations_per_restart = []
        self.restart_final_values = []
        self.restarts_at_local_optimum = 0
        self.iterations = 0
        self.neighbors_evaluated = 0
        self.stopped_at_local_optimum = False
        self.history_fitness = []
        self.history_raw = []
        self.initial_state = None

        best_state: Optional[State] = None
        best_f: Optional[float] = None
        running: Optional[float] = None

        while self.restarts < self.max_restarts:
            climber = SteepestAscentHillClimbing(
                self.container,
                self.boxes,
                self.max_iterations,
                self.objective_type,
                self.rng.randrange(10 ** 9),
            )
            climber.run()

            self.restarts += 1
            self.iterations_per_restart.append(climber.iterations)
            self.iterations += climber.iterations
            self.neighbors_evaluated += climber.neighbors_evaluated
            if climber.stopped_at_local_optimum:
                self.restarts_at_local_optimum += 1
            if self.initial_state is None:
                self.initial_state = climber.initial_state

            self.history_raw.extend(climber.history_fitness)
            for value in climber.history_fitness:
                if running is None or value > running:
                    running = value
                self.history_fitness.append(running)

            final_f = climber.history_fitness[-1]
            self.restart_final_values.append(final_f)
            if best_f is None or final_f > best_f:
                best_f, best_state = final_f, climber.final_state

            if progress_callback:
                progress_callback(self.restarts, best_f)

        self.stopped_at_local_optimum = self.restarts_at_local_optimum == self.restarts
        self.final_state = best_state
        self.search_duration = time.perf_counter() - start_time
        return best_state
