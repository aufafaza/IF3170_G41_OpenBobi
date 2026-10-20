import os
import tempfile
import unittest

from src.algorithms import (
    GeneticAlgorithm,
    RandomRestartHillClimbing,
    SidewaysMoveHillClimbing,
    SimulatedAnnealing,
    SteepestAscentHillClimbing,
    StochasticHillClimbing,
)
from src.algorithms.state_space import StateSpace
from src.experiments.local_search_runner import run_local_search_experiments
from src.experiments.upper_bound import bound_from_items
from src.utils.data_loader import load_instance
from src.visualization.dashboard import build_dashboard
from src.visualization.data import PackageView, StateView, validate_state

def to_view(state):
    placed = []
    for truck_label, pkgs in state.placed.items():
        for label, p in pkgs.items():
            b = state.boxes[label]
            placed.append(PackageView(
                label, tuple(p.dimensions), b.value, b.weight, b.is_fragile, b.eta, tuple(p.position)
            ))
    outside = []
    for label in state.outside:
        b = state.boxes[label]
        outside.append(PackageView(
            label, tuple(b.dimensions), b.value, b.weight, b.is_fragile, b.eta
        ))
    c = state.containers[0]
    return StateView(c.label, tuple(c.dimensions), c.max_capacity, placed, outside)

class TestFeasibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.containers, cls.boxes = load_instance("data/small.json")

    def assert_valid(self, state, name):
        self.assertEqual(validate_state(to_view(state)), [], name)

    def test_validator_detects_violations(self):
        fragile = PackageView("A", (2, 2, 1), 1, 1, True, 1, (0, 0, 0))
        on_top = PackageView("B", (2, 2, 1), 1, 1, False, 1, (0, 0, 1))
        self.assertTrue(validate_state(StateView("T", (4, 4, 4), 10, [fragile, on_top], [])))

        floating = PackageView("C", (2, 2, 1), 1, 1, False, 1, (0, 0, 2))
        self.assertTrue(validate_state(StateView("T", (4, 4, 4), 10, [floating], [])))

        wall = PackageView("D", (2, 2, 1), 1, 1, False, 1, (3, 0, 0))
        self.assertTrue(validate_state(StateView("T", (4, 4, 4), 10, [wall], [])))

        heavy = PackageView("E", (1, 1, 1), 1, 20, False, 1, (0, 0, 0))
        self.assertTrue(validate_state(StateView("T", (4, 4, 4), 10, [heavy], [])))

    def test_initial_and_neighbor_states_are_valid(self):
        for seed in range(5):
            space = StateSpace(self.containers, self.boxes, seed=seed)
            state = space.generate_initial_state()
            self.assert_valid(state, "initial")
            for _ in range(50):
                neighbor = space.random_neighbor(state)
                if neighbor is not None:
                    self.assert_valid(neighbor, "random neighbor")
                    state = neighbor

    def test_all_algorithms_return_valid_states(self):
        for seed in range(3):
            algorithms = [
                SteepestAscentHillClimbing(self.containers, self.boxes, 50, seed=seed),
                SidewaysMoveHillClimbing(self.containers, self.boxes, 50, 5, seed=seed),
                StochasticHillClimbing(self.containers, self.boxes, 200, 50, seed=seed),
                RandomRestartHillClimbing(self.containers, self.boxes, 3, 50, seed=seed),
                SimulatedAnnealing(self.containers, self.boxes, max_iterations=500, seed=seed),
                GeneticAlgorithm(self.containers, self.boxes, population_size=10, max_iterations=10, seed=seed),
            ]
            for algo in algorithms:
                final = algo.run()
                name = type(algo).__name__
                self.assert_valid(algo.initial_state, name + " initial")
                self.assert_valid(final, name + " final")

    def test_sideways_and_restart_parameters(self):
        hc = SidewaysMoveHillClimbing(self.containers, self.boxes, 100, 3, seed=1)
        hc.run()
        self.assertLessEqual(hc.sideways_moves, 3)

        rr = RandomRestartHillClimbing(self.containers, self.boxes, 4, 50, seed=1)
        rr.run()
        self.assertEqual(rr.restarts, 4)
        self.assertEqual(len(rr.iterations_per_restart), 4)
        self.assertEqual(rr.history_fitness[-1], rr.f(rr.final_state))

    def test_sa_stuck_metrics(self):
        sa = SimulatedAnnealing(self.containers, self.boxes, max_iterations=300, seed=3, stuck_window=5)
        sa.run()
        self.assertTrue(0.0 <= sa.stuck_frequency <= 1.0)
        self.assertLessEqual(sa.stuck_episodes, sa.stuck_iterations)
        if sa.stuck_episodes:
            self.assertGreaterEqual(sa.stuck_iterations, 5)

    def test_history_matches_objective_for_float_objectives(self):
        for objective in ("density", "eta_urgency"):
            hc = SteepestAscentHillClimbing(self.containers, self.boxes, 50, objective_type=objective, seed=2)
            final = hc.run()
            self.assertAlmostEqual(hc.history_fitness[-1], hc.f(final), places=9)

            sw = SidewaysMoveHillClimbing(self.containers, self.boxes, 50, 5, objective_type=objective, seed=2)
            final = sw.run()
            self.assertAlmostEqual(sw.history_fitness[-1], sw.f(final), places=9)

            sa = SimulatedAnnealing(self.containers, self.boxes, max_iterations=300, objective_type=objective, seed=2)
            final = sa.run()
            self.assertAlmostEqual(sa.history_best[-1], sa.f(final), places=9)

class TestMainInstance(unittest.TestCase):
    def test_algorithms_valid_on_mejiro_express(self):
        containers, boxes = load_instance("data/mejiro_express.json")
        algorithms = [
            SteepestAscentHillClimbing(containers, boxes, 5, seed=1),
            SimulatedAnnealing(containers, boxes, max_iterations=300, seed=1),
            GeneticAlgorithm(containers, boxes, population_size=10, max_iterations=10, seed=1),
        ]
        for algo in algorithms:
            final = algo.run()
            name = type(algo).__name__
            self.assertEqual(validate_state(to_view(algo.initial_state)), [], name + " initial")
            self.assertEqual(validate_state(to_view(final)), [], name + " final")

class TestUpperBound(unittest.TestCase):
    def test_classic_knapsack(self):
        items = [(60, 10, 1), (100, 20, 1), (120, 30, 1)]
        self.assertEqual(bound_from_items(items, 50, 100), 220)

    def test_volume_constraint(self):
        items = [(60, 10, 1), (100, 20, 1), (120, 30, 1)]
        self.assertEqual(bound_from_items(items, 50, 1), 120)

    def test_node_limit_fallback_stays_valid(self):
        items = [(60, 10, 1), (100, 20, 1), (120, 30, 1)]
        self.assertGreaterEqual(bound_from_items(items, 50, 100, node_limit=1), 220)

class TestDashboard(unittest.TestCase):
    def test_dashboard_builds(self):
        containers, boxes = load_instance("data/small.json")
        with tempfile.TemporaryDirectory() as tmp:
            run_local_search_experiments(
                containers, boxes, tmp,
                repetitions=1,
                hc_max_iterations=10,
                sa_max_iterations=50,
                max_restarts=2,
                verbose=False,
            )
            path, warnings = build_dashboard(tmp)
            self.assertTrue(os.path.exists(path))
            self.assertEqual(warnings, [])
            with open(path, "r", encoding="utf-8") as handle:
                self.assertIn("Batas atas", handle.read())

if __name__ == "__main__":
    unittest.main()