"""
Unit tests for Genetic Algorithm implementation.
Focuses purely on Genetic Algorithm logic: Chromosome representation,
Selection, Crossover, Mutation (3 neighbor moves), and Objective functions.
"""

import unittest
from src.models.box import Box, PlacedBox
from src.models.container import Container
from src.models.state import State, Placement
from src.algorithms.genetic_algorithm import GeneticAlgorithm, Chromosome
from src.algorithms.objective import (
    default_objective_function,
    density_objective_function,
    eta_urgency_objective_function,
    evaluate_fitness,
)


class TestGeneticAlgorithmLogic(unittest.TestCase):
    def setUp(self):
        self.container = Container("SmallTruck", (8, 8, 4), 100)
        self.boxes = {
            f"P{i}": Box(f"P{i}", (2, 2, 1), 20 + i * 5, 5 + i, is_fragile=(i % 3 == 0), eta=i)
            for i in range(8)
        }
        self.ga = GeneticAlgorithm(
            container=self.container,
            boxes=self.boxes,
            population_size=10,
            max_iterations=10,
            seed=42
        )

    def test_chromosome_initialization(self):
        chrom = self.ga.create_random_chromosome()
        self.assertEqual(len(chrom.order), len(self.boxes))
        self.assertEqual(len(chrom.rotations), len(self.boxes))
        self.assertEqual(len(chrom.inclusions), len(self.boxes))

    def test_population_initialization(self):
        pop = self.ga.initialize_population()
        self.assertEqual(len(pop), 10)
        for ind in pop:
            self.assertGreaterEqual(ind.fitness, 0.0)

    def test_selection_methods(self):
        pop = self.ga.initialize_population()
        for method in ["roulette", "tournament", "rank"]:
            self.ga.selection_method = method
            parent = self.ga.select_parent(pop)
            self.assertIsInstance(parent, Chromosome)

    def test_crossover(self):
        p1 = self.ga.create_random_chromosome()
        p2 = self.ga.create_random_chromosome()
        self.ga.crossover_rate = 1.0
        c1, c2 = self.ga.crossover(p1, p2)

        self.assertEqual(sorted(c1.order), sorted(self.ga.box_keys))
        self.assertEqual(sorted(c2.order), sorted(self.ga.box_keys))

    def test_mutation_operators(self):
        chrom = self.ga.create_random_chromosome()
        self.ga.mutation_rate = 1.0
        for _ in range(20):
            self.ga.mutate(chrom)
            self.assertEqual(sorted(chrom.order), sorted(self.ga.box_keys))

    def test_ga_run_and_evolution(self):
        final_state = self.ga.run()
        self.assertIsInstance(final_state, State)
        self.assertGreater(final_state.total_value, 0)
        self.assertLessEqual(final_state.total_weight, self.container.max_capacity)
        self.assertEqual(len(self.ga.history_max_fitness), 11)
        self.assertGreaterEqual(self.ga.history_max_fitness[-1], self.ga.history_max_fitness[0])

    def test_objective_functions(self):
        placed = {
            "P0": Placement((0, 0, 0), (2, 2, 1)),
            "P1": Placement((2, 0, 0), (2, 2, 1)),
        }
        test_state = State(self.container, self.boxes, placed)
        val_default = default_objective_function(test_state)
        val_density = density_objective_function(test_state)
        val_eta = eta_urgency_objective_function(test_state)

        self.assertGreater(val_default, 0)
        self.assertGreater(val_density, val_default)
        self.assertGreater(val_eta, 0)


if __name__ == "__main__":
    unittest.main()
