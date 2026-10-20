import argparse
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.algorithms import (
    GeneticAlgorithm,
    RandomRestartHillClimbing,
    SidewaysMoveHillClimbing,
    SimulatedAnnealing,
    SteepestAscentHillClimbing,
    StochasticHillClimbing,
)
from src.algorithms.objective import evaluate_fitness
from src.algorithms.state_space import state_resolution
from src.experiments.experiment_runner import run_full_experiments
from src.experiments.local_search_runner import run_local_search_experiments
from src.utils.data_loader import load_instance, save_instance, save_solution
from src.utils.generator import generate_instance

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ALGORITHM_LABELS = {
    "steepest": "Hill-climbing (Steepest Ascent)",
    "sideways": "Hill-climbing (Sideways Move)",
    "stochastic": "Hill-climbing (Stochastic)",
    "restart": "Hill-climbing (Random Restart)",
    "sa": "Simulated Annealing",
    "ga": "Genetic Algorithm",
}

def print_banner():
    banner = r"""
======================================================================
  [TRUCK] MEJIRO EXPRESS - 3D CONTAINER LOADING OPTIMIZATION
  IF3170 Inteligensi Artifisial - Local Search
  Slogan: "Lebih kilat dari Umamusume"
======================================================================
"""
    print(banner)

def get_instance(args):
    if args.generate:
        container, boxes = generate_instance(num_boxes=args.generate, seed=args.generate_seed)
        containers = [container] # Wrap in list for multi-truck compatibility
        save_instance(args.generate_output, containers, boxes)
        args.input = args.generate_output
        print(f"[GENERATE] {len(boxes)} paket disimpan ke: {args.generate_output}")
        return containers, boxes
    return load_instance(args.input)

def build_algorithm(args, containers, boxes):
    name = args.algorithm
    seed = args.seed
    objective = args.objective
    if name == "steepest":
        return SteepestAscentHillClimbing(containers, boxes, args.hc_iterations, objective, seed)
    if name == "sideways":
        return SidewaysMoveHillClimbing(
            containers, boxes, args.hc_iterations, args.max_sideways, objective, seed
        )
    if name == "stochastic":
        return StochasticHillClimbing(containers, boxes, args.hc_iterations * 3, 200, objective, seed)
    if name == "restart":
        return RandomRestartHillClimbing(
            containers, boxes, args.max_restarts, args.hc_iterations, objective, seed
        )
    if name == "sa":
        return SimulatedAnnealing(
            containers, boxes, max_iterations=args.sa_iterations, objective_type=objective, seed=seed
        )
    return GeneticAlgorithm(
        containers=containers,
        boxes=boxes,
        population_size=args.pop_size,
        max_iterations=args.iterations,
        crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate,
        elitism_count=args.elitism,
        selection_method=args.selection,
        objective_type=objective,
        seed=seed,
    )

def algorithm_details(name, algo):
    rows = []
    if name in ("steepest", "sideways", "stochastic", "restart"):
        rows.append(("Iterations", algo.iterations))
        rows.append(("Neighbors Evaluated", algo.neighbors_evaluated))
        rows.append(("Stopped At Local Optimum", algo.stopped_at_local_optimum))
    if name == "sideways":
        rows.append(("Sideways Moves", f"{algo.sideways_moves} / {algo.max_sideways_moves}"))
    if name == "restart":
        rows.append(("Restarts", f"{algo.restarts} / {algo.max_restarts}"))
        rows.append(("Iterations Per Restart", algo.iterations_per_restart))
        rows.append(("Final Value Per Restart", algo.restart_final_values))
    if name == "sa":
        rows.append(("Iterations", algo.iterations))
        rows.append(("Stop Reason", algo.stop_reason))
        rows.append(("Final Temperature", f"{algo.final_temperature:.6g}"))
        rows.append(("Accepted (Improving)", algo.accepted_improving))
        rows.append(("Accepted (Sideways)", algo.accepted_sideways))
        rows.append(("Accepted (Worsening)", algo.accepted_worsening))
        rows.append(("Rejected", algo.rejected))
        rows.append(("Stuck Episodes (Best Not Improving)", algo.stuck_episodes))
        rows.append(("Stuck Frequency (Best Not Improving)", f"{algo.stuck_frequency * 100:.2f}%"))
        rows.append(("Trapped Episodes (State Unchanged)", algo.trapped_episodes))
        rows.append(("Trapped Frequency (State Unchanged)", f"{algo.trapped_frequency * 100:.2f}%"))
    if name == "ga":
        rows.append(("Population Size", algo.population_size))
        rows.append(("Iterations", algo.max_iterations))
        rows.append(("Fitness Evaluations", algo.evaluations))
    return rows

def history_payload(name, algo):
    if name == "sa":
        return {
            "history_current": algo.history_current,
            "history_best": algo.history_best,
            "history_temperature": algo.history_temperature,
        }
    if name == "ga":
        return {
            "history_max": algo.history_max_fitness,
            "history_avg": algo.history_avg_fitness,
        }
    return {"history_fitness": algo.history_fitness}

def run_single(args):
    containers, boxes = get_instance(args)
    name = args.algorithm
    label = ALGORITHM_LABELS[name]
    print(f"\n[LOAD] Problem instance: {args.input}")
    
    total_capacity = sum(c.max_capacity for c in containers)
    print(f"[CONTAINER] {len(containers)} Truck(s) | Total Capacity: {total_capacity} weight units")
    print(f"[PACKAGES]  Total: {len(boxes)} | Objective Type: '{args.objective}'")
    print(f"[ALGORITHM] {label}")
    if name == "ga":
        print(f"[CONFIG]    Pop Size = {args.pop_size}, Iterations = {args.iterations}, Selection = {args.selection}")
    print()

    algo = build_algorithm(args, containers, boxes)

    def on_progress(gen, best_f, avg_f):
        if gen == 0 or gen % max(1, args.iterations // 10) == 0 or gen == args.iterations:
            print(f"  [Gen {gen:4d}/{args.iterations}] Max Fitness: {best_f:8.2f} | Avg Fitness: {avg_f:8.2f}")

    print("[RUNNING] Optimization in progress...")
    final_state = algo.run(progress_callback=on_progress) if name == "ga" else algo.run()

    initial_objective = evaluate_fitness(algo.initial_state, args.objective) if algo.initial_state else 0
    final_objective = evaluate_fitness(final_state, args.objective)
    valid, _ = state_resolution(final_state, args.objective)

    print("\n" + "=" * 70)
    print("[SUCCESS] OPTIMIZATION COMPLETE!")
    print("=" * 70)
    print(f"Algorithm              : {label}")
    print(f"Search Duration        : {algo.search_duration:.4f} seconds")
    print(f"Initial Objective Value: {initial_objective:g}")
    print(f"Final Objective Value  : {final_objective:g}")
    print(f"Final Package Value    : {final_state.total_value}")
    print(f"Total Weight           : {final_state.total_weight} / {total_capacity} ({final_state.weight_utilization_ratio * 100:.1f}%)")
    print(f"Volume Utilization     : {final_state.volume_utilization_ratio * 100:.1f}%")
    print(f"Packages Left Outside  : {len(final_state.outside)} packages")
    print(f"All Constraints Valid  : {'Yes' if valid else 'No'}")
    for key, value in algorithm_details(name, algo):
        print(f"{key:<23}: {value}")
    print("=" * 70)

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    sol_path = os.path.join(out_dir, "latest_solution.json")
    metadata = {
        "algorithm": label,
        "duration_seconds": algo.search_duration,
        "objective": args.objective,
        "initial_value": initial_objective,
        "final_value": final_objective,
    }
    if name == "ga":
        metadata.update({
            "pop_size": args.pop_size,
            "iterations": args.iterations,
            "selection": args.selection,
        })
    metadata.update(history_payload(name, algo))
    save_solution(sol_path, final_state, metadata=metadata)
    print(f"[SAVE] Solution JSON saved to: {sol_path}")

def main():
    print_banner()

    parser = argparse.ArgumentParser(
        description="IF3170 - 3D Container Loading Problem with Local Search",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("-i", "--input", type=str, default="data/mejiro_express.json",
                        help="Path to problem instance JSON file")
    parser.add_argument("-a", "--algorithm", type=str, default="ga",
                        choices=["steepest", "sideways", "stochastic", "restart", "sa", "ga"],
                        help="Algorithm for a single run")
    parser.add_argument("-p", "--pop-size", type=int, default=50,
                        help="Population size for Genetic Algorithm")
    parser.add_argument("-n", "--iterations", type=int, default=100,
                        help="Number of iterations (generations) for Genetic Algorithm")
    parser.add_argument("--crossover-rate", type=float, default=0.8,
                        help="Crossover probability rate")
    parser.add_argument("--mutation-rate", type=float, default=0.2,
                        help="Mutation probability rate")
    parser.add_argument("--elitism", type=int, default=2,
                        help="Number of top individuals preserved per generation")
    parser.add_argument("--selection", type=str, default="roulette", choices=["roulette", "tournament", "rank"],
                        help="Parent selection method")
    parser.add_argument("--objective", type=str, default="default", choices=["default", "density", "eta_urgency"],
                        help="Objective function type")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for deterministic runs")
    parser.add_argument("-o", "--output-dir", type=str, default="output",
                        help="Output directory for results")
    parser.add_argument("--run-experiments", action="store_true",
                        help="Run the Genetic Algorithm experiment matrix (18 runs)")
    parser.add_argument("--run-local-search", action="store_true",
                        help="Run Hill-climbing (4 variants) and Simulated Annealing experiments")
    parser.add_argument("--run-all", action="store_true",
                        help="Run all experiments")
    parser.add_argument("--hc-iterations", type=int, default=200,
                        help="Maximum iterations for Hill-climbing")
    parser.add_argument("--sa-iterations", type=int, default=3000,
                        help="Maximum iterations for Simulated Annealing")
    parser.add_argument("--max-sideways", type=int, default=25,
                        help="Maximum sideways moves for Hill-climbing with Sideways Move")
    parser.add_argument("--max-restarts", type=int, default=5,
                        help="Maximum restarts for Random Restart Hill-climbing")
    parser.add_argument("--generate", type=int, default=0,
                        help="Buat instance acak dengan jumlah paket ini dan pakai untuk run")
    parser.add_argument("--generate-seed", type=int, default=42,
                        help="Seed pembuat instance acak")
    parser.add_argument("--generate-output", type=str, default="data/generated.json",
                        help="Lokasi penyimpanan instance acak")

    args = parser.parse_args()

    run_ls = args.run_local_search or args.run_all
    run_ga = args.run_experiments or args.run_all

    if run_ls or run_ga:
        containers, boxes = get_instance(args)

        if run_ls:
            print("[EXP] RUNNING HILL-CLIMBING AND SIMULATED ANNEALING EXPERIMENTS...")
            run_local_search_experiments(
                containers=containers,
                boxes=boxes,
                output_dir=args.output_dir,
                repetitions=3,
                hc_max_iterations=args.hc_iterations,
                sa_max_iterations=args.sa_iterations,
                max_sideways_moves=args.max_sideways,
                max_restarts=args.max_restarts,
                objective_type=args.objective,
                verbose=True,
            )

        if run_ga:
            print("[EXP] RUNNING GENETIC ALGORITHM EXPERIMENT MATRIX (18 RUNS)...")
            run_full_experiments(
                containers=containers,
                boxes=boxes,
                output_dir=args.output_dir,
                fixed_pop_control=args.pop_size,
                fixed_iter_control=args.iterations,
                repetitions=3,
                objective_type=args.objective,
                verbose=True,
            )
        return

    run_single(args)

if __name__ == "__main__":
    main()