import math
from typing import Callable, Dict

from ..models.state import State


def default_objective_function(state: State) -> float:
    return float(state.total_value)


def density_objective_function(state: State) -> float:
    base_val = float(state.total_value)
    ratio = state.volume_utilization_ratio
    return base_val * (1.0 + ratio)


def eta_urgency_objective_function(state: State) -> float:
    return math.fsum(
        state.boxes[label].value * (1.0 + 1.0 / (max(0, state.boxes[label].eta) + 1.0))
        for label in state.placed
    )


OBJECTIVE_FUNCTIONS: Dict[str, Callable[[State], float]] = {
    "default": default_objective_function,
    "density": density_objective_function,
    "eta_urgency": eta_urgency_objective_function,
}


def evaluate_fitness(
    state: State,
    objective_type: str = "default",
) -> float:
    obj_fn = OBJECTIVE_FUNCTIONS.get(objective_type, default_objective_function)
    return obj_fn(state)