"""
Main Entry Point for 3D Container Loading Problem - Genetic Algorithm
IF3170 Artificial Intelligence 2026/2027

Features:
- Single execution with custom parameters
- Automated experiment suite conforming to course specification
- JSON output generation for visualizer integration
"""

import sys
import os
import argparse
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.models.state import State, Placement
from src.models.box import Box, PlacedBox
from src.models.container import Container
from src.algorithms.genetic_algorithm import GeneticAlgorithm
from src.utils.data_loader import load_instance, save_solution
from src.experiments.experiment_runner import run_full_experiments


if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def print_banner():
    banner = r"""
======================================================================
  [TRUCK] MEJIRO EXPRESS - 3D CONTAINER LOADING OPTIMIZATION
  [GA]    IF3170 Inteligensi Artifisial - Genetic Algorithm
  Slogan: "Lebih kilat dari Umamusume"
======================================================================
"""
    print(banner)


def run_single_ga(args):
    print(f"\n[LOAD] Loading problem instance from: {args.input}")
    container, boxes = load_instance(args.input)
    W, D, H = container.dimensions

    print(f"[CONTAINER] {container.label} | Size: {W} x {D} x {H} | Max Capacity: {container.max_capacity} weight units")
    print(f"[PACKAGES]  Total: {len(boxes)} | Objective Type: '{args.objective}'")
    print(f"[CONFIG]    Pop Size = {args.pop_size}, Iterations = {args.iterations}, Selection = {args.selection}\n")

    ga = GeneticAlgorithm(
        container=container,
        boxes=boxes,
        population_size=args.pop_size,
        max_iterations=args.iterations,
        crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate,
        elitism_count=args.elitism,
        selection_method=args.selection,
        objective_type=args.objective,
        seed=args.seed
    )

    print("[RUNNING] Optimization in progress...")
    last_print_time = time.time()

    def on_progress(gen, best_f, avg_f):
        nonlocal last_print_time
        if gen == 0 or gen % max(1, args.iterations // 10) == 0 or gen == args.iterations:
            print(f"  [Gen {gen:4d}/{args.iterations}] Max Fitness: {best_f:8.2f} | Avg Fitness: {avg_f:8.2f}")

    final_state = ga.run(progress_callback=on_progress)

    print("\n" + "="*70)
    print("[SUCCESS] OPTIMIZATION COMPLETE!")
    print("="*70)
    print(f"Search Duration        : {ga.search_duration:.4f} seconds")
    print(f"Initial Objective Value: {ga.initial_state.total_value if ga.initial_state else 0}")
    print(f"Final Objective Value  : {final_state.total_value}")
    print(f"Total Weight           : {final_state.total_weight} / {container.max_capacity} ({final_state.weight_utilization_ratio*100:.1f}%)")
    print(f"Volume Utilization     : {final_state.volume_utilization_ratio*100:.1f}%")
    print(f"Packages Inside Truck  : {len(final_state.placed)} packages")
    print(f"Packages Left Outside  : {len(final_state.outside)} packages")
    print("="*70)

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    sol_path = os.path.join(out_dir, "latest_solution.json")
    save_solution(sol_path, final_state, metadata={
        "algorithm": "GeneticAlgorithm",
        "duration_seconds": ga.search_duration,
        "pop_size": args.pop_size,
        "iterations": args.iterations,
        "selection": args.selection,
        "objective": args.objective,
        "history_max": ga.history_max_fitness,
        "history_avg": ga.history_avg_fitness,
    })
    print(f"[SAVE] Solution JSON saved to: {sol_path} (Ready for visualizer)")


def main():
    print_banner()

    parser = argparse.ArgumentParser(
        description="IF3170 - 3D Container Loading Problem with Genetic Algorithm",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("-i", "--input", type=str, default="data/mejiro_express.json",
                        help="Path to problem instance JSON file")
    parser.add_argument("-p", "--pop-size", type=int, default=50,
                        help="Population size for Genetic Algorithm")
    parser.add_argument("-n", "--iterations", type=int, default=100,
                        help="Number of iterations (generations)")
    parser.add_argument("--crossover-rate", type=float, default=0.8,
                        help="Crossover probability rate")
    parser.add_argument("--mutation-rate", type=float, default=0.2,
                        help="Mutation probability rate")
    parser.add_argument("--elitism", type=int, default=2,
                        help="Number of top individuals preserved per generation")
    parser.add_argument("--selection", type=str, default="roulette", choices=["roulette", "tournament", "rank"],
                        help="Parent selection method")
    parser.add_argument("--objective", type=str, default="default", choices=["default", "density", "eta_urgency"],
                        help="Objective function type (default, or Bonus: density, eta_urgency)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for deterministic runs")
    parser.add_argument("-o", "--output-dir", type=str, default="output",
                        help="Output directory for plots and results")
    parser.add_argument("--run-experiments", action="store_true",
                        help="Run the complete course experiment matrix (18 runs: population & iteration controls)")

    args = parser.parse_args()

    if args.run_experiments:
        print("[EXP] RUNNING FULL COURSE EXPERIMENTAL MATRIX (18 RUNS)...")
        container, boxes = load_instance(args.input)
        run_full_experiments(
            container=container,
            boxes=boxes,
            output_dir=args.output_dir,
            fixed_pop_control=args.pop_size,
            fixed_iter_control=args.iterations,
            repetitions=3,
            verbose=True
        )
    else:
        run_single_ga(args)


if __name__ == "__main__":
    main()
