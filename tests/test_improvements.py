import json
import os
import tempfile
import unittest

from src.algorithms.hill_climbing import RandomRestartHillClimbing, SteepestAscentHillClimbing
from src.algorithms.simulated_annealing import SimulatedAnnealing
from src.algorithms.state_space import StateSpace, quarter_turns
from src.utils.data_loader import load_instance, orientation_code
from src.visualization.data import PackageView, StateView, recompute_objective, validate_state

class TestQuarterTurns(unittest.TestCase):
    def test_distinct_dimensions(self):
        self.assertEqual(set(quarter_turns((2, 3, 4))), {(3, 2, 4), (2, 4, 3), (4, 3, 2)})

    def test_cube_has_no_turn(self):
        self.assertEqual(quarter_turns((2, 2, 2)), [])

    def test_two_equal_sides(self):
        self.assertEqual(set(quarter_turns((2, 2, 3))), {(2, 3, 2), (3, 2, 2)})

    def test_rotated_states_swap_exactly_two_sides(self):
        containers, boxes = load_instance("data/small.json")
        checked = 0
        for seed in range(10):
            space = StateSpace(containers, boxes, seed=seed)
            state = space.generate_initial_state()
            for neighbor in space._rotated_states(state.placed):
                changed = [
                    label for label in state.placed[containers[0].label]
                    if neighbor.placed[containers[0].label][label].dimensions != state.placed[containers[0].label][label].dimensions
                ]
                self.assertEqual(len(changed), 1)
                label = changed[0]
                self.assertIn(neighbor.placed[containers[0].label][label].dimensions, quarter_turns(state.placed[containers[0].label][label].dimensions))
                self.assertEqual(neighbor.placed[containers[0].label][label].position, state.placed[containers[0].label][label].position)
                checked += 1
        self.assertGreater(checked, 0)

class TestRestartHistory(unittest.TestCase):
    def test_raw_history_covers_every_restart(self):
        containers, boxes = load_instance("data/small.json")
        rr = RandomRestartHillClimbing(containers, boxes, 3, 20, seed=5)
        rr.run()
        self.assertEqual(len(rr.history_raw), sum(n + 1 for n in rr.iterations_per_restart))
        self.assertEqual(len(rr.history_raw), len(rr.history_fitness))
        self.assertEqual(len(rr.restart_final_values), 3)
        self.assertEqual(max(rr.restart_final_values), rr.history_fitness[-1])
        for raw, best in zip(rr.history_raw, rr.history_fitness):
            self.assertLessEqual(raw, best)

    def test_local_optimum_flag_matches_counter(self):
        containers, boxes = load_instance("data/small.json")
        rr = RandomRestartHillClimbing(containers, boxes, 3, 20, seed=5)
        rr.run()
        self.assertEqual(rr.stopped_at_local_optimum, rr.restarts_at_local_optimum == rr.restarts)

    def test_steepest_history_matches_objective(self):
        containers, boxes = load_instance("data/small.json")
        hc = SteepestAscentHillClimbing(containers, boxes, 30, seed=2)
        final = hc.run()
        self.assertEqual(hc.history_fitness[-1], hc.f(final))

class TestAnnealingMetrics(unittest.TestCase):
    def test_trapped_metrics_and_stop_reason(self):
        containers, boxes = load_instance("data/small.json")
        sa = SimulatedAnnealing(containers, boxes, max_iterations=400, seed=4, stuck_window=5)
        sa.run()
        self.assertTrue(0.0 <= sa.trapped_frequency <= 1.0)
        self.assertLessEqual(sa.trapped_episodes, sa.trapped_iterations)
        self.assertIn(sa.stop_reason, ("batas iterasi tercapai", "suhu minimum tercapai"))

    def test_stop_reason_temperature(self):
        containers, boxes = load_instance("data/small.json")
        sa = SimulatedAnnealing(containers, boxes, max_iterations=5000, initial_temperature=1.0,
                                cooling_rate=0.5, min_temperature=0.01, seed=1)
        sa.run()
        self.assertEqual(sa.stop_reason, "suhu minimum tercapai")

class TestLoaderAndOrientation(unittest.TestCase):
    def write(self, payload):
        handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
        json.dump(payload, handle)
        handle.close()
        self.addCleanup(os.remove, handle.name)
        return handle.name

    def base(self):
        return {
            "truck": {"label": "T", "dimensions": [4, 4, 4], "max_capacity": 10},
            "packages": [
                {"id": "A", "dimensions": [1, 1, 1], "value": 1, "weight": 1},
                {"id": "B", "dimensions": [1, 2, 1], "value": 1, "weight": 1},
            ],
        }

    def test_valid_instance(self):
        containers, boxes = load_instance(self.write(self.base()))
        self.assertEqual(len(boxes), 2)

    def test_duplicate_ids_rejected(self):
        data = self.base()
        data["packages"][1]["id"] = "A"
        with self.assertRaises(ValueError):
            load_instance(self.write(data))

    def test_bad_dimensions_rejected(self):
        for bad in ([0, 1, 1], [1, 1], [1.5, 1, 1], [-1, 1, 1]):
            data = self.base()
            data["packages"][0]["dimensions"] = bad
            with self.assertRaises(ValueError):
                load_instance(self.write(data))

    def test_fragile_must_be_boolean(self):
        data = self.base()
        data["packages"][0]["is_fragile"] = "false"
        with self.assertRaises(ValueError):
            load_instance(self.write(data))

    def test_missing_field_rejected(self):
        data = self.base()
        del data["packages"][0]["value"]
        with self.assertRaises(ValueError):
            load_instance(self.write(data))

    def test_orientation_code(self):
        self.assertEqual(orientation_code((2, 3, 4), (2, 3, 4)), "wlh")
        self.assertEqual(orientation_code((2, 3, 4), (3, 2, 4)), "lwh")
        self.assertEqual(orientation_code((2, 3, 4), (4, 3, 2)), "hlw")
        self.assertIsNone(orientation_code((2, 3, 4), (5, 3, 4)))

class TestValidator(unittest.TestCase):
    def test_duplicate_across_inside_and_outside(self):
        inside = PackageView("A", (1, 1, 1), 1, 1, False, 1, (0, 0, 0))
        outside = PackageView("A", (1, 1, 1), 1, 1, False, 1)
        self.assertTrue(validate_state(StateView("T", (4, 4, 4), 10, [inside], [outside])))

    def test_non_rotation_dimensions_flagged(self):
        pkg = PackageView("A", (2, 2, 2), 1, 1, False, 1, (0, 0, 0), original_dimensions=(1, 2, 3), orientation="wlh")
        self.assertTrue(validate_state(StateView("T", (4, 4, 4), 10, [pkg], [])))

    def test_orientation_mismatch_flagged(self):
        pkg = PackageView("A", (2, 1, 3), 1, 1, False, 1, (0, 0, 0), original_dimensions=(1, 2, 3), orientation="wlh")
        self.assertTrue(validate_state(StateView("T", (4, 4, 4), 10, [pkg], [])))

    def test_valid_rotation_passes(self):
        pkg = PackageView("A", (2, 1, 3), 1, 1, False, 1, (0, 0, 0), original_dimensions=(1, 2, 3), orientation="lwh")
        self.assertEqual(validate_state(StateView("T", (4, 4, 4), 10, [pkg], [])), [])

    def test_recompute_objective(self):
        a = PackageView("A", (2, 2, 1), 10, 1, False, 1, (0, 0, 0))
        state = StateView("T", (4, 4, 1), 10, [a], [])
        self.assertEqual(recompute_objective(state, "default"), 10.0)
        self.assertAlmostEqual(recompute_objective(state, "density"), 10.0 * (1 + 4 / 16))
        self.assertAlmostEqual(recompute_objective(state, "eta_urgency"), 10.0 * (1 + 1 / 2))
        self.assertIsNone(recompute_objective(state, "unknown"))

class TestDashboardConsistency(unittest.TestCase):
    def test_dashboard_reports_no_issue_for_clean_runs(self):
        from src.experiments.local_search_runner import run_local_search_experiments
        from src.visualization.data import load_dataset
        from src.visualization.dashboard import build_dashboard

        containers, boxes = load_instance("data/small.json")
        for objective in ("default", "density", "eta_urgency"):
            with tempfile.TemporaryDirectory() as tmp:
                run_local_search_experiments(
                    containers, boxes, tmp, repetitions=1, hc_max_iterations=8,
                    sa_max_iterations=60, max_restarts=2, objective_type=objective, verbose=False,
                )
                dataset = load_dataset(tmp)
                for run in dataset.runs:
                    self.assertEqual(run.issues, [], f"{objective} {run.label}")
                path, warnings = build_dashboard(tmp)
                self.assertTrue(os.path.exists(path))
                self.assertEqual(warnings, [])

if __name__ == "__main__":
    unittest.main()