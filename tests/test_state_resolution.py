import unittest

from src.algorithms import state_space
from src.algorithms.state_space import StateSpace, state_resolution
from src.models.state import Placement, State
from src.utils.data_loader import load_instance
from src.visualization.charts import GA_COLOR, group_colors
from src.visualization.data import Run

def make_run(key, group, repetition=1):
    return Run(key, group, f"{group} (run {repetition})", repetition, 0, 0, None, None, None, None, {}, [], [])

class TestStateResolution(unittest.TestCase):
    def setUp(self):
        self.containers, self.boxes = load_instance("data/small.json")

    def test_valid_state_updates_score(self):
        space = StateSpace(self.containers, self.boxes, seed=1)
        state = space.generate_initial_state()
        valid, score = state_resolution(state)
        self.assertTrue(valid)
        self.assertEqual(score, float(state.total_value))
        self.assertEqual(state_space.state_score_trig, score)

    def test_invalid_state_scores_zero(self):
        label = next(iter(self.boxes))
        dims = self.boxes[label].dimensions
        placed = {self.containers[0].label: {label: Placement((0, 0, 3), dims)}}
        state = State(self.containers, self.boxes, placed)
        valid, score = state_resolution(state)
        self.assertFalse(valid)
        self.assertEqual(score, 0.0)
        self.assertEqual(state_space.state_score_trig, 0.0)

class TestGaColor(unittest.TestCase):
    def test_all_ga_groups_use_required_color(self):
        runs = [make_run("ga", "Pop 50, Iter 50"), make_run("ga", "Pop 50, Iter 100"), make_run("ga", "Pop 20, Iter 100")]
        self.assertEqual(set(group_colors(runs).values()), {"#386641"})
        self.assertEqual(GA_COLOR, "#386641")

    def test_other_algorithms_avoid_ga_color(self):
        runs = [make_run("hc", f"Variant {i}") for i in range(8)]
        self.assertNotIn("#386641", group_colors(runs).values())

if __name__ == "__main__":
    unittest.main()