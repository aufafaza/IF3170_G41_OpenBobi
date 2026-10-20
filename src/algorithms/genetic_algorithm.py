import time
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Callable

from ..models.box import Box, PlacedBox
from ..models.container import Container
from ..models.state import State, Placement
from .objective import evaluate_fitness

@dataclass
class Chromosome:
    order: List[str]
    rotations: Dict[str, int]
    inclusions: Dict[str, bool]
    fitness: float = -1.0
    decoded_state: Optional[State] = None

    def clone(self) -> "Chromosome":
        return Chromosome(
            order=list(self.order),
            rotations=dict(self.rotations),
            inclusions=dict(self.inclusions),
            fitness=self.fitness,
            decoded_state=self.decoded_state.copy() if self.decoded_state else None
        )

def _xy_overlaps(cx: int, cy: int, w: int, d: int, other: PlacedBox) -> bool:
    ox, oy, _ = other.position
    ow, od, _ = other.dimensions
    return max(cx, ox) < min(cx + w, ox + ow) and max(cy, oy) < min(cy + d, oy + od)

class GeneticAlgorithm:
    def __init__(
        self,
        container: Container,
        boxes: Dict[str, Box],
        population_size: int = 50,
        max_iterations: int = 100,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.2,
        elitism_count: int = 2,
        selection_method: str = "roulette",
        objective_type: str = "default",
        seed: Optional[int] = None
    ):
        self.container = container
        self.boxes = boxes
        self.box_keys = list(boxes.keys())
        self.population_size = max(4, population_size)
        self.max_iterations = max(1, max_iterations)
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_count = min(elitism_count, self.population_size // 2)
        self.selection_method = selection_method.lower()
        self.objective_type = objective_type
        self.rng = random.Random(seed)

        self.box_rotations: Dict[str, List[Tuple[int, int, int]]] = {
            lbl: b.get_rotations() for lbl, b in boxes.items()
        }

        self.history_max_fitness: List[float] = []
        self.history_avg_fitness: List[float] = []
        self.history_min_fitness: List[float] = []
        self.initial_state: Optional[State] = None
        self.final_state: Optional[State] = None
        self.best_chromosome: Optional[Chromosome] = None
        self.search_duration: float = 0.0
        self.evaluations: int = 0

    def decode(self, chromosome: Chromosome) -> State:
        W, D, H = self.container.dimensions
        max_cap = self.container.max_capacity

        placed: Dict[str, Placement] = {}
        placed_boxes_list: List[PlacedBox] = []
        current_weight = 0

        candidates: List[Tuple[int, int, int]] = [(0, 0, 0)]

        for label in chromosome.order:
            if not chromosome.inclusions.get(label, True):
                continue

            box = self.boxes[label]
            if current_weight + box.weight > max_cap:
                continue

            rot_list = self.box_rotations[label]
            rot_idx = chromosome.rotations.get(label, 0) % len(rot_list)
            w, d, h = rot_list[rot_idx]

            best_candidate: Optional[Tuple[int, int, int]] = None

            candidates.sort(key=lambda pt: (pt[2], pt[1], pt[0]))

            for cx, cy, cz in candidates:
                if cx + w > W or cy + d > D or cz + h > H:
                    continue

                candidate_box = PlacedBox(box, (cx, cy, cz), (w, d, h))

                collides = False
                for pb in placed_boxes_list:
                    if candidate_box.check_overlap(pb):
                        collides = True
                        break
                if collides:
                    continue

                blocked = False
                has_support = cz == 0
                for pb in placed_boxes_list:
                    if not _xy_overlaps(cx, cy, w, d, pb):
                        continue
                    top = pb.position[2] + pb.dimensions[2]
                    if top <= cz:
                        if pb.box.is_fragile:
                            blocked = True
                            break
                        if top == cz:
                            has_support = True
                    elif box.is_fragile:
                        blocked = True
                        break
                if blocked or not has_support:
                    continue

                best_candidate = (cx, cy, cz)
                break

            if best_candidate is not None:
                px, py, pz = best_candidate
                placed[label] = Placement((px, py, pz), (w, d, h))
                placed_boxes_list.append(PlacedBox(box, (px, py, pz), (w, d, h)))
                current_weight += box.weight

                new_points = [
                    (px + w, py, pz),
                    (px, py + d, pz),
                ]
                if not box.is_fragile and pz + h < H:
                    new_points.append((px, py, pz + h))

                for pb in placed_boxes_list[:-1]:
                    if pb.position[2] + pb.dimensions[2] == pz:
                        new_points.append((px + w, pb.position[1], pz))
                        new_points.append((pb.position[0], py + d, pz))

                existing_set = set(candidates)
                for nx, ny, nz in new_points:
                    if nx < W and ny < D and nz < H and (nx, ny, nz) not in existing_set:
                        inside_any = any(
                            pb.position[0] <= nx < pb.position[0] + pb.dimensions[0] and
                            pb.position[1] <= ny < pb.position[1] + pb.dimensions[1] and
                            pb.position[2] <= nz < pb.position[2] + pb.dimensions[2]
                            for pb in placed_boxes_list
                        )
                        if not inside_any:
                            candidates.append((nx, ny, nz))
                            existing_set.add((nx, ny, nz))

                candidates = [
                    pt for pt in candidates
                    if not (px <= pt[0] < px + w and py <= pt[1] < py + d and pz <= pt[2] < pz + h)
                ]

        return State(container=self.container, boxes=self.boxes, placed=placed)

    def evaluate(self, chromosome: Chromosome) -> float:
        if chromosome.fitness < 0.0 or chromosome.decoded_state is None:
            state = self.decode(chromosome)
            self.evaluations += 1
            chromosome.decoded_state = state
            chromosome.fitness = evaluate_fitness(
                state=state,
                objective_type=self.objective_type
            )
        return chromosome.fitness

    def create_random_chromosome(self) -> Chromosome:
        order = list(self.box_keys)
        self.rng.shuffle(order)

        rotations = {}
        inclusions = {}
        for lbl in self.box_keys:
            n_rots = len(self.box_rotations[lbl])
            rotations[lbl] = self.rng.randint(0, n_rots - 1)
            inclusions[lbl] = self.rng.random() < 0.85

        return Chromosome(order=order, rotations=rotations, inclusions=inclusions)

    def initialize_population(self) -> List[Chromosome]:
        pop = [self.create_random_chromosome() for _ in range(self.population_size)]
        for ind in pop:
            self.evaluate(ind)
        return pop

    def select_parent(self, population: List[Chromosome]) -> Chromosome:
        if self.selection_method == "tournament":
            k = min(3, len(population))
            sampled = self.rng.sample(population, k)
            return max(sampled, key=lambda ind: ind.fitness)

        if self.selection_method == "rank":
            sorted_pop = sorted(population, key=lambda ind: ind.fitness)
            n = len(sorted_pop)
            total_rank = (n * (n + 1)) // 2
            pick = self.rng.uniform(0, total_rank)
            current = 0
            for rank, ind in enumerate(sorted_pop, start=1):
                current += rank
                if current >= pick:
                    return ind
            return sorted_pop[-1]

        total_fitness = sum(ind.fitness for ind in population)
        if total_fitness <= 0.0:
            return self.rng.choice(population)

        pick = self.rng.uniform(0, total_fitness)
        current = 0.0
        for ind in population:
            current += ind.fitness
            if current >= pick:
                return ind
        return population[-1]

    def crossover(self, parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        if self.rng.random() > self.crossover_rate:
            return parent1.clone(), parent2.clone()

        n = len(parent1.order)
        if n < 2:
            return parent1.clone(), parent2.clone()

        cut1 = self.rng.randint(0, n - 2)
        cut2 = self.rng.randint(cut1 + 1, n - 1)

        def order_crossover_helper(p1_order: List[str], p2_order: List[str]) -> List[str]:
            child_order = [None] * n
            child_order[cut1:cut2 + 1] = p1_order[cut1:cut2 + 1]
            copied_set = set(child_order[cut1:cut2 + 1])

            p2_filtered = [item for item in p2_order if item not in copied_set]
            fill_idx = 0
            for i in range(n):
                if child_order[i] is None:
                    child_order[i] = p2_filtered[fill_idx]
                    fill_idx += 1
            return child_order

        child1_order = order_crossover_helper(parent1.order, parent2.order)
        child2_order = order_crossover_helper(parent2.order, parent1.order)

        child1_rots = {}
        child2_rots = {}
        child1_inc = {}
        child2_inc = {}

        for lbl in self.box_keys:
            if self.rng.random() < 0.5:
                child1_rots[lbl] = parent1.rotations[lbl]
                child1_inc[lbl] = parent1.inclusions[lbl]
                child2_rots[lbl] = parent2.rotations[lbl]
                child2_inc[lbl] = parent2.inclusions[lbl]
            else:
                child1_rots[lbl] = parent2.rotations[lbl]
                child1_inc[lbl] = parent2.inclusions[lbl]
                child2_rots[lbl] = parent1.rotations[lbl]
                child2_inc[lbl] = parent1.inclusions[lbl]

        c1 = Chromosome(order=child1_order, rotations=child1_rots, inclusions=child1_inc)
        c2 = Chromosome(order=child2_order, rotations=child2_rots, inclusions=child2_inc)
        return c1, c2

    def mutate(self, chromosome: Chromosome) -> None:
        if self.rng.random() > self.mutation_rate:
            return

        move_choice = self.rng.choice([1, 2, 3])
        n = len(chromosome.order)
        if n == 0:
            return

        if move_choice == 1 and n >= 2:
            idx1, idx2 = self.rng.sample(range(n), 2)
            chromosome.order[idx1], chromosome.order[idx2] = (
                chromosome.order[idx2],
                chromosome.order[idx1]
            )
            if self.rng.random() < 0.5:
                lbl1, lbl2 = chromosome.order[idx1], chromosome.order[idx2]
                chromosome.inclusions[lbl1], chromosome.inclusions[lbl2] = (
                    chromosome.inclusions[lbl2],
                    chromosome.inclusions[lbl1]
                )

        elif move_choice == 2:
            if self.rng.random() < 0.5 and n >= 2:
                src_idx = self.rng.randint(0, n - 1)
                dst_idx = self.rng.randint(0, n - 1)
                lbl = chromosome.order.pop(src_idx)
                chromosome.order.insert(dst_idx, lbl)
            else:
                target_lbl = self.rng.choice(self.box_keys)
                chromosome.inclusions[target_lbl] = not chromosome.inclusions.get(target_lbl, True)

        elif move_choice == 3:
            target_lbl = self.rng.choice(self.box_keys)
            n_rots = len(self.box_rotations[target_lbl])
            if n_rots > 1:
                cur_rot = chromosome.rotations.get(target_lbl, 0)
                new_rot = (cur_rot + self.rng.randint(1, n_rots - 1)) % n_rots
                chromosome.rotations[target_lbl] = new_rot

        chromosome.fitness = -1.0
        chromosome.decoded_state = None

    def run(self, progress_callback: Optional[Callable[[int, float, float], None]] = None) -> State:
        start_time = time.perf_counter()
        self.evaluations = 0

        population = self.initialize_population()
        population.sort(key=lambda ind: ind.fitness, reverse=True)

        self.best_chromosome = population[0].clone()
        self.initial_state = population[0].decoded_state.copy()

        best_f = population[0].fitness
        avg_f = sum(ind.fitness for ind in population) / len(population)
        min_f = population[-1].fitness

        self.history_max_fitness = [best_f]
        self.history_avg_fitness = [avg_f]
        self.history_min_fitness = [min_f]

        if progress_callback:
            progress_callback(0, best_f, avg_f)

        for gen in range(1, self.max_iterations + 1):
            next_population: List[Chromosome] = []

            for i in range(self.elitism_count):
                next_population.append(population[i].clone())

            while len(next_population) < self.population_size:
                p1 = self.select_parent(population)
                p2 = self.select_parent(population)

                c1, c2 = self.crossover(p1, p2)
                self.mutate(c1)
                self.mutate(c2)

                self.evaluate(c1)
                next_population.append(c1)

                if len(next_population) < self.population_size:
                    self.evaluate(c2)
                    next_population.append(c2)

            next_population.sort(key=lambda ind: ind.fitness, reverse=True)
            population = next_population

            current_best = population[0]
            if current_best.fitness > self.best_chromosome.fitness:
                self.best_chromosome = current_best.clone()

            best_f = population[0].fitness
            avg_f = sum(ind.fitness for ind in population) / len(population)
            min_f = population[-1].fitness

            self.history_max_fitness.append(best_f)
            self.history_avg_fitness.append(avg_f)
            self.history_min_fitness.append(min_f)

            if progress_callback and (gen % 10 == 0 or gen == self.max_iterations):
                progress_callback(gen, best_f, avg_f)

        self.search_duration = time.perf_counter() - start_time
        self.final_state = self.best_chromosome.decoded_state.copy()

        return self.final_state

__all__ = [
    "GeneticAlgorithm",
    "Chromosome",
]