from typing import Dict, List

from ..models.box import Box
from ..models.container import Container
from ..algorithms.hill_climbing import (
    SteepestAscentHillClimbing,
    SidewaysMoveHillClimbing,
    StochasticHillClimbing,
    RandomRestartHillClimbing,
)
from ..algorithms.simulated_annealing import SimulatedAnnealing
from ..visualization.records import (
    hill_climbing_record,
    simulated_annealing_record,
    write_summary,
)

def run_local_search_experiments(
    containers: List[Container],
    boxes: Dict[str, Box],
    output_dir: str = "output",
    repetitions: int = 3,
    hc_max_iterations: int = 200,
    sa_max_iterations: int = 3000,
    max_sideways_moves: int = 25,
    max_restarts: int = 5,
    objective_type: str = "default",
    verbose: bool = True,
) -> Dict[str, str]:
    variants = {
        "steepest": ("Steepest Ascent", lambda s: SteepestAscentHillClimbing(containers, boxes, hc_max_iterations, objective_type, s)),
        "sideways": ("Sideways Move", lambda s: SidewaysMoveHillClimbing(containers, boxes, hc_max_iterations, max_sideways_moves, objective_type, s)),
        "stochastic": ("Stochastic", lambda s: StochasticHillClimbing(containers, boxes, hc_max_iterations * 3, 200, objective_type, s)),
        "restart": ("Random Restart", lambda s: RandomRestartHillClimbing(containers, boxes, max_restarts, hc_max_iterations, objective_type, s)),
    }

    paths: Dict[str, str] = {}
    for key, (group, factory) in variants.items():
        runs = []
        for rep in range(1, repetitions + 1):
            seed = 100 + rep
            hc = factory(seed)
            hc.run()
            runs.append(hill_climbing_record(hc, output_dir, f"hc_{key}_run{rep}", rep, rep, seed, group))
            if verbose:
                print(f"[HC {group}] run {rep}: {hc.final_state.total_value} ({hc.search_duration:.3f}s)")
        paths[f"hc_{key}"] = write_summary(
            output_dir, "Hill-climbing", containers, len(boxes), runs,
            filename=f"hc_{key}_summary.json", objective=objective_type,
        )

    sa_runs = []
    for rep in range(1, repetitions + 1):
        seed = 100 + rep
        sa = SimulatedAnnealing(containers, boxes, max_iterations=sa_max_iterations, objective_type=objective_type, seed=seed)
        sa.run()
        sa_runs.append(simulated_annealing_record(sa, output_dir, f"sa_run{rep}", rep, rep, seed, "Simulated Annealing"))
        if verbose:
            print(f"[SA] run {rep}: {sa.final_state.total_value} ({sa.search_duration:.3f}s)")
    paths["sa"] = write_summary(
        output_dir, "Simulated Annealing", containers, len(boxes), sa_runs,
        filename="sa_summary.json", objective=objective_type,
    )
    return paths