import math
import time
from typing import Callable, Dict, List, Optional

from ..models.box import Box
from ..models.container import Container
from ..models.state import State
from .state_space import StateSpace


class SimulatedAnnealing(StateSpace):
    """
    each iteration picks ONE random feasible neighbour and computes
    delta_f = f(neighbour) - f(current).
      delta_f >= 0 -> always accept (delta_f == 0 is a sideways move)
      delta_f < 0  -> accept with probability exp(delta_f / T)
    """

    def __init__(
        self,
        container: Container,
        boxes: Dict[str, Box],
        max_iterations: int = 1000,
        initial_temperature: float = 100.0,
        cooling_rate: float = 0.995,
        min_temperature: float = 0.001,
        objective_type: str = "default",
        seed: Optional[int] = None,
    ):
        super().__init__(container, boxes, objective_type, seed)
        self.max_iterations = max(1, max_iterations)
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature

        self.iterations: int = 0
        self.accepted_improving: int = 0
        self.accepted_sideways: int = 0
        self.accepted_worsening: int = 0
        self.rejected: int = 0
        self.no_neighbor: int = 0
        self.final_temperature: float = initial_temperature
        self.history_current: List[float] = []
        self.history_best: List[float] = []
        self.history_temperature: List[float] = []
        self.history_delta: List[Optional[float]] = []
        self.history_acceptance_probability: List[Optional[float]] = []
        self.initial_state: Optional[State] = None
        self.final_state: Optional[State] = None
        self.search_duration: float = 0.0

    def run(self, progress_callback: Optional[Callable[[int, float], None]] = None) -> State:
        start_time = time.perf_counter()

        current = self.generate_initial_state()
        current_f = self.f(current)
        best, best_f = current, current_f
        self.initial_state = current

        temperature = self.initial_temperature
        self.iterations = 0
        self.accepted_improving = 0
        self.accepted_sideways = 0
        self.accepted_worsening = 0
        self.rejected = 0
        self.no_neighbor = 0
        self.history_current = [current_f]
        self.history_best = [best_f]
        self.history_temperature = [temperature]
        self.history_delta = []
        self.history_acceptance_probability = []
        if progress_callback:
            progress_callback(0, best_f)

        while self.iterations < self.max_iterations and temperature >= self.min_temperature:
            self.iterations += 1
            neighbor = self.random_neighbor(current)

            if neighbor is None:
                self.no_neighbor += 1
                self.history_delta.append(None)
                self.history_acceptance_probability.append(None)
            else:
                delta = self.f(neighbor) - current_f
                probability = 1.0 if delta >= 0 else math.exp(delta / temperature)
                self.history_delta.append(delta)
                self.history_acceptance_probability.append(probability)
                if delta > 0:
                    self.accepted_improving += 1
                    current, current_f = neighbor, current_f + delta
                elif delta == 0:
                    self.accepted_sideways += 1
                    current = neighbor
                elif self.rng.random() < probability:
                    # eq: e^{DeltaE div Temp} 
                    self.accepted_worsening += 1
                    current, current_f = neighbor, current_f + delta
                else:
                    self.rejected += 1

                if current_f > best_f:
                    best, best_f = current, current_f

            temperature *= self.cooling_rate
            self.history_current.append(current_f)
            self.history_best.append(best_f)
            self.history_temperature.append(temperature)
            if progress_callback:
                progress_callback(self.iterations, best_f)

        self.final_temperature = temperature
        self.final_state = best
        self.search_duration = time.perf_counter() - start_time
        return best

    @property
    def stuck_count(self) -> int:
        return self.rejected + self.no_neighbor

    @property
    def stuck_frequency(self) -> float:
        return self.stuck_count / self.iterations if self.iterations else 0.0

    def plot_data(self) -> Dict[str, list]:
                return {
            "iteration": list(range(1, self.iterations + 1)),
            "temperature": self.history_temperature[:-1],
            "delta_f": self.history_delta,
            "acceptance_probability": self.history_acceptance_probability,
            "current_fitness": self.history_current[1:],
            "best_fitness": self.history_best[1:],
        }
