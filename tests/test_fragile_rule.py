import unittest

from src.algorithms.genetic_algorithm import Chromosome, GeneticAlgorithm
from src.algorithms.state_space import StateSpace
from src.models.box import Box
from src.models.container import Container
from src.models.state import Placement
from src.visualization.data import PackageView, StateView, validate_state

def make_boxes():
    return {
        "T": Box("T", (2, 2, 2), 1, 1),
        "F": Box("F", (2, 2, 1), 1, 1, is_fragile=True),
        "X": Box("X", (4, 2, 1), 1, 1),
    }

class TestFragileOverhang(unittest.TestCase):
    def setUp(self):
        self.container = Container("T", (6, 6, 4), 100)
        self.boxes = make_boxes()
        self.space = StateSpace(self.container, self.boxes)

    def test_fragile_under_overhang_is_rejected(self):
        placed = {
            "T": Placement((0, 0, 0), (2, 2, 2)),
            "F": Placement((2, 0, 0), (2, 2, 1)),
            "X": Placement((0, 0, 2), (4, 2, 1)),
        }
        self.assertFalse(self.space._is_valid(placed))
        self.assertFalse(self.space._all_supported(placed))

    def test_overhang_without_fragile_is_accepted(self):
        placed = {
            "T": Placement((0, 0, 0), (2, 2, 2)),
            "X": Placement((0, 0, 2), (4, 2, 1)),
        }
        self.assertTrue(self.space._is_valid(placed))

    def test_can_add_rejects_package_above_fragile_with_gap(self):
        placed = {
            "T": Placement((0, 0, 0), (2, 2, 2)),
            "F": Placement((2, 0, 0), (2, 2, 1)),
        }
        self.assertFalse(self.space._can_add(placed, 2, "X", (0, 0, 2), (4, 2, 1)))

    def test_can_add_rejects_fragile_below_existing_package(self):
        placed = {
            "T": Placement((0, 0, 0), (2, 2, 2)),
            "X": Placement((0, 0, 2), (4, 2, 1)),
        }
        self.assertFalse(self.space._can_add(placed, 2, "F", (2, 0, 0), (2, 2, 1)))

class TestValidatorFragile(unittest.TestCase):
    def test_validator_flags_fragile_under_overhang(self):
        views = [
            PackageView("T", (2, 2, 2), 1, 1, False, 1, (0, 0, 0)),
            PackageView("F", (2, 2, 1), 1, 1, True, 1, (2, 0, 0)),
            PackageView("X", (4, 2, 1), 1, 1, False, 1, (0, 0, 2)),
        ]
        self.assertTrue(validate_state(StateView("T", (6, 6, 4), 100, views, [])))

    def test_validator_accepts_overhang_without_fragile(self):
        views = [
            PackageView("T", (2, 2, 2), 1, 1, False, 1, (0, 0, 0)),
            PackageView("X", (4, 2, 1), 1, 1, False, 1, (0, 0, 2)),
        ]
        self.assertEqual(validate_state(StateView("T", (6, 6, 4), 100, views, [])), [])

class TestGeneticDecodeFragile(unittest.TestCase):
    def test_decoded_states_respect_fragile_rule(self):
        container = Container("T", (6, 6, 4), 100)
        boxes = make_boxes()
        ga = GeneticAlgorithm(container, boxes, population_size=10, max_iterations=5, seed=3)
        space = StateSpace(container, boxes)
        for order in (["T", "F", "X"], ["T", "X", "F"], ["F", "T", "X"], ["X", "F", "T"]):
            chromosome = Chromosome(
                order=order,
                rotations={label: 0 for label in boxes},
                inclusions={label: True for label in boxes},
            )
            state = ga.decode(chromosome)
            self.assertTrue(space._is_valid(state.placed), order)

if __name__ == "__main__":
    unittest.main()