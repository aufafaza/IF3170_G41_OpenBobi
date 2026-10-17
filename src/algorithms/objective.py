"""
Objective functions and fitness evaluation for the 3D Container Loading Problem.

Supported objective functions:
1. 'default': Pure total value of placed packages (standard specification).
2. 'density': Value balanced with container volume utilization (Bonus specification).
3. 'eta_urgency': Value weighted by delivery urgency / ETA (Bonus specification).
"""

from typing import Callable, Dict
from ..models.state import State


def default_objective_function(state: State) -> float:
    return float(state.total_value)


def density_objective_function(state: State) -> float:
    base_val = float(state.total_value)
    ratio = state.volume_utilization_ratio
    return base_val * (1.0 + ratio)


def eta_urgency_objective_function(state: State) -> float:
    score = 0.0
    for label in state.placed:
        box = state.boxes[label]
        urgency_multiplier = 1.0 + (1.0 / (max(0, box.eta) + 1.0))
        score += box.value * urgency_multiplier
    return score


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
