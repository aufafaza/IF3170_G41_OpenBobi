import json
import os
from typing import Any, Dict, List, Optional

from ..models.container import Container
from ..models.box import Box
from ..algorithms.genetic_algorithm import GeneticAlgorithm
from ..algorithms.objective import evaluate_fitness
from ..utils.data_loader import save_solution

def _calculate_stats(subset: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not subset:
        return {}
    vals = [r["final_value"] for r in subset]
    durs = [r["duration_seconds"] for r in subset]
    vol_utils = [r["volume_utilization"] for r in subset]
    wt_utils = [r["weight_utilization"] for r in subset]
    mean_v = sum(vals) / len(vals)
    mean_d = sum(durs) / len(durs)
    mean_vol = sum(vol_utils) / len(vol_utils)
    mean_wt = sum(wt_utils) / len(wt_utils)
    std_v = (sum((v - mean_v) ** 2 for v in vals) / max(1, len(vals) - 1)) ** 0.5
    return {
        "mean_value": round(mean_v, 2),
        "std_value": round(std_v, 2),
        "min_value": min(vals),
        "max_value": max(vals),
        "mean_duration": round(mean_d, 4),
        "mean_volume_utilization": round(mean_vol, 2),
        "mean_weight_utilization": round(mean_wt, 2),
        "run_count": len(subset),
    }

def _run_once(
    container: Container,
    boxes: Dict[str, Box],
    output_dir: str,
    solutions_dir: str,
    run_id: int,
    set_name: str,
    tag: str,
    pop_size: int,
    iterations: int,
    repetition: int,
    seed: int,
    objective_type: str,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.2,
    elitism_count: int = 2,
    selection_method: str = "roulette",
) -> Dict[str, Any]:
    ga = GeneticAlgorithm(
        container=container,
        boxes=boxes,
        population_size=pop_size,
        max_iterations=iterations,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        elitism_count=elitism_count,
        selection_method=selection_method,
        objective_type=objective_type,
        seed=seed,
    )
    final_state = ga.run()

    init_val = evaluate_fitness(ga.initial_state, objective_type) if ga.initial_state else 0
    final_val = evaluate_fitness(final_state, objective_type)

    sol_base = f"{tag}_pop{pop_size}_iter{iterations}_run{repetition}"
    final_sol_path = os.path.join(solutions_dir, f"{sol_base}_final.json")
    save_solution(final_sol_path, final_state)
    init_sol_path = None
    if ga.initial_state:
        init_sol_path = os.path.join(solutions_dir, f"{sol_base}_initial.json")
        save_solution(init_sol_path, ga.initial_state)

    return {
        "run_id": run_id,
        "experiment_set": set_name,
        "pop_size": pop_size,
        "iterations": iterations,
        "repetition": repetition,
        "seed": seed,
        "objective": objective_type,
        "crossover_rate": ga.crossover_rate,
        "mutation_rate": ga.mutation_rate,
        "elitism": ga.elitism_count,
        "selection": ga.selection_method,
        "initial_value": init_val,
        "final_value": final_val,
        "total_value": final_state.total_value,
        "duration_seconds": round(ga.search_duration, 4),
        "evaluations": ga.evaluations,
        "volume_utilization": round(final_state.volume_utilization_ratio * 100, 2),
        "weight_utilization": round(final_state.weight_utilization_ratio * 100, 2),
        "packages_placed": len(final_state.placed),
        "initial_state_file": os.path.relpath(init_sol_path, output_dir).replace(os.sep, "/") if init_sol_path else None,
        "final_state_file": os.path.relpath(final_sol_path, output_dir).replace(os.sep, "/"),
        "history_max": ga.history_max_fitness,
        "history_avg": ga.history_avg_fitness,
        "history_min": ga.history_min_fitness,
    }

def run_full_experiments(
    container: Container,
    boxes: Dict[str, Box],
    output_dir: str = "output",
    fixed_pop_control: int = 50,
    iter_variations: Optional[List[int]] = None,
    fixed_iter_control: int = 100,
    pop_variations: Optional[List[int]] = None,
    repetitions: int = 3,
    objective_type: str = "default",
    verbose: bool = True,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.2,
    elitism_count: int = 2,
    selection_method: str = "roulette",
) -> Dict[str, Any]:
    if iter_variations is None:
        iter_variations = [50, 100, 200]
    if pop_variations is None:
        pop_variations = [20, 50, 100]

    exp_dir = os.path.join(output_dir, "experiments")
    solutions_dir = os.path.join(output_dir, "solutions")
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(solutions_dir, exist_ok=True)

    results: List[Dict[str, Any]] = []
    run_counter = 1

    if verbose:
        print("\n" + "=" * 70)
        print("EXPERIMENT SET 1: POPULATION AS CONTROL")
        print(f"Fixed Population Size = {fixed_pop_control}, Iteration Variations = {iter_variations}")
        print("=" * 70)

    for iters in iter_variations:
        for rep in range(1, repetitions + 1):
            if verbose:
                print(f"  [Exp 1] Pop={fixed_pop_control}, Iters={iters}, Run={rep}/{repetitions}...", end="", flush=True)
            seed = 1000 * fixed_pop_control + 10 * iters + rep
            record = _run_once(
                container, boxes, output_dir, solutions_dir, run_counter,
                "Population Control", "exp1", fixed_pop_control, iters, rep, seed, objective_type,
                crossover_rate, mutation_rate, elitism_count, selection_method,
            )
            results.append(record)
            run_counter += 1
            if verbose:
                print(f" Done! Value: {record['final_value']} | Duration: {record['duration_seconds']:.3f}s")

    if verbose:
        print("\n" + "=" * 70)
        print("EXPERIMENT SET 2: ITERATIONS AS CONTROL")
        print(f"Fixed Iterations = {fixed_iter_control}, Population Variations = {pop_variations}")
        print("=" * 70)

    for pop in pop_variations:
        for rep in range(1, repetitions + 1):
            if verbose:
                print(f"  [Exp 2] Pop={pop}, Iters={fixed_iter_control}, Run={rep}/{repetitions}...", end="", flush=True)
            seed = 2000 * pop + 10 * fixed_iter_control + rep
            record = _run_once(
                container, boxes, output_dir, solutions_dir, run_counter,
                "Iterations Control", "exp2", pop, fixed_iter_control, rep, seed, objective_type,
                crossover_rate, mutation_rate, elitism_count, selection_method,
            )
            results.append(record)
            run_counter += 1
            if verbose:
                print(f" Done! Value: {record['final_value']} | Duration: {record['duration_seconds']:.3f}s")

    exp1_runs = [r for r in results if r["experiment_set"] == "Population Control"]
    exp2_runs = [r for r in results if r["experiment_set"] == "Iterations Control"]

    exp1_aggregates = {
        str(iters): _calculate_stats([r for r in exp1_runs if r["iterations"] == iters])
        for iters in iter_variations
    }
    exp2_aggregates = {
        str(pop): _calculate_stats([r for r in exp2_runs if r["pop_size"] == pop])
        for pop in pop_variations
    }

    summary = {
        "metadata": {
            "algorithm": "Genetic Algorithm",
            "objective": objective_type,
            "container": {
                "label": container.label,
                "dimensions": list(container.dimensions),
                "max_capacity": container.max_capacity,
            },
            "total_packages": len(boxes),
            "plot_color": "#386641",
            "total_runs": len(results),
            "crossover_rate": crossover_rate,
            "mutation_rate": mutation_rate,
            "elitism": elitism_count,
            "selection": selection_method,
        },
        "experiment_1_population_control": {
            "control_type": "population",
            "fixed_population_size": fixed_pop_control,
            "iteration_variations": iter_variations,
            "repetitions": repetitions,
            "aggregates": exp1_aggregates,
            "runs": exp1_runs
        },
        "experiment_2_iterations_control": {
            "control_type": "iterations",
            "fixed_iterations": fixed_iter_control,
            "population_variations": pop_variations,
            "repetitions": repetitions,
            "aggregates": exp2_aggregates,
            "runs": exp2_runs
        },
        "all_runs": results,
    }

    json_path = os.path.join(exp_dir, "experiment_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    summary["json_path"] = json_path

    if verbose:
        print("\n" + "=" * 70)
        print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
        print(f"Summary JSON: {json_path}")
        print("=" * 70 + "\n")

    return summary