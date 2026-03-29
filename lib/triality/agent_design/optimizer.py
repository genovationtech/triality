"""
Multi-objective optimization with constraint handling.

Implements NSGA-II and Pareto front discovery for design optimization.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Callable, Tuple, Optional
from enum import Enum
import copy


class ObjectiveType(Enum):
    """Optimization objective type"""
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


@dataclass
class Objective:
    """Optimization objective definition"""
    name: str
    function: Callable[[Dict[str, float]], float]
    objective_type: ObjectiveType = ObjectiveType.MINIMIZE
    weight: float = 1.0  # For weighted sum approaches


@dataclass
class Constraint:
    """Optimization constraint"""
    name: str
    function: Callable[[Dict[str, float]], float]
    # Function should return <= 0 for feasible, > 0 for infeasible
    penalty_factor: float = 1000.0  # Penalty for constraint violation


@dataclass
class Individual:
    """Individual in population (candidate design)"""
    parameters: Dict[str, float]
    objectives: Dict[str, float] = field(default_factory=dict)
    constraint_violations: Dict[str, float] = field(default_factory=dict)
    rank: int = 0  # Pareto rank (0 = non-dominated)
    crowding_distance: float = 0.0  # Diversity metric
    is_feasible: bool = True

    def dominates(self, other: 'Individual', objectives: List[Objective]) -> bool:
        """
        Check if this individual dominates another (Pareto dominance).

        Returns True if this is at least as good in all objectives and
        strictly better in at least one.
        """
        at_least_as_good = True
        strictly_better = False

        for obj in objectives:
            val_self = self.objectives[obj.name]
            val_other = other.objectives[obj.name]

            if obj.objective_type == ObjectiveType.MINIMIZE:
                if val_self > val_other:
                    at_least_as_good = False
                if val_self < val_other:
                    strictly_better = True
            else:  # MAXIMIZE
                if val_self < val_other:
                    at_least_as_good = False
                if val_self > val_other:
                    strictly_better = True

        return at_least_as_good and strictly_better


@dataclass
class ParetoFront:
    """Pareto-optimal set of solutions"""
    individuals: List[Individual]

    def get_objectives_array(self, objective_names: List[str]) -> np.ndarray:
        """Return objectives as numpy array for plotting"""
        n_ind = len(self.individuals)
        n_obj = len(objective_names)
        objectives = np.zeros((n_ind, n_obj))

        for i, ind in enumerate(self.individuals):
            for j, obj_name in enumerate(objective_names):
                objectives[i, j] = ind.objectives[obj_name]

        return objectives

    def get_best_for_objective(self, objective_name: str, minimize: bool = True) -> Individual:
        """Return individual best in single objective"""
        if not self.individuals:
            raise ValueError("Empty Pareto front")

        best = self.individuals[0]
        for ind in self.individuals[1:]:
            if minimize:
                if ind.objectives[objective_name] < best.objectives[objective_name]:
                    best = ind
            else:
                if ind.objectives[objective_name] > best.objectives[objective_name]:
                    best = ind

        return best


@dataclass
class OptimizationResult:
    """Results of optimization"""
    pareto_front: ParetoFront
    n_generations: int
    n_evaluations: int
    convergence_history: List[float] = field(default_factory=list)
    # Convergence metric vs generation


class MultiObjectiveOptimizer:
    """
    Multi-objective optimizer using NSGA-II algorithm.

    NSGA-II features:
    1. Fast non-dominated sorting
    2. Crowding distance for diversity
    3. Elite preservation
    4. Tournament selection
    """

    def __init__(self, objectives: List[Objective], constraints: List[Constraint],
                 parameter_space: Dict[str, Tuple[float, float]]):
        """
        Initialize optimizer.

        Args:
            objectives: List of objectives to optimize
            constraints: List of constraints
            parameter_space: Dict of {param_name: (min, max)}
        """
        self.objectives = objectives
        self.constraints = constraints
        self.parameter_space = parameter_space
        self.param_names = list(parameter_space.keys())
        self.n_evaluations = 0

    def evaluate_individual(self, individual: Individual):
        """Evaluate objectives and constraints for an individual"""
        # Evaluate objectives
        for obj in self.objectives:
            individual.objectives[obj.name] = obj.function(individual.parameters)

        # Evaluate constraints
        individual.is_feasible = True
        for constraint in self.constraints:
            violation = constraint.function(individual.parameters)
            individual.constraint_violations[constraint.name] = violation
            if violation > 0:
                individual.is_feasible = False

        self.n_evaluations += 1

    def initialize_population(self, pop_size: int, seed: Optional[int] = None) -> List[Individual]:
        """Initialize random population using Latin Hypercube Sampling"""
        if seed is not None:
            np.random.seed(seed)

        n_params = len(self.param_names)
        population = []

        # LHS for better initial coverage
        lhs = np.zeros((pop_size, n_params))
        for i in range(n_params):
            intervals = np.arange(pop_size)
            shuffled = np.random.permutation(intervals)
            uniform = np.random.uniform(0, 1, pop_size)
            lhs[:, i] = (shuffled + uniform) / pop_size

        # Create individuals
        for i in range(pop_size):
            params = {}
            for j, param_name in enumerate(self.param_names):
                min_val, max_val = self.parameter_space[param_name]
                params[param_name] = min_val + lhs[i, j] * (max_val - min_val)

            individual = Individual(parameters=params)
            self.evaluate_individual(individual)
            population.append(individual)

        return population

    def fast_non_dominated_sort(self, population: List[Individual]) -> List[List[Individual]]:
        """
        Fast non-dominated sorting (NSGA-II).

        Returns list of fronts (Pareto ranks).
        Front 0 = non-dominated solutions.
        """
        n = len(population)

        # For each individual, compute:
        # S[i] = set of individuals dominated by i
        # n[i] = number of individuals dominating i

        S = [[] for _ in range(n)]
        n_dominated = [0] * n

        fronts = [[]]

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                if population[i].dominates(population[j], self.objectives):
                    S[i].append(j)
                elif population[j].dominates(population[i], self.objectives):
                    n_dominated[i] += 1

            if n_dominated[i] == 0:
                population[i].rank = 0
                fronts[0].append(population[i])

        # Build subsequent fronts
        front_idx = 0
        while front_idx < len(fronts) and fronts[front_idx]:
            next_front = []
            for ind in fronts[front_idx]:
                ind_idx = population.index(ind)
                for j in S[ind_idx]:
                    n_dominated[j] -= 1
                    if n_dominated[j] == 0:
                        population[j].rank = front_idx + 1
                        next_front.append(population[j])

            if next_front:
                fronts.append(next_front)
            front_idx += 1

        # Remove empty last front if exists
        if fronts and not fronts[-1]:
            fronts.pop()

        return fronts

    def calculate_crowding_distance(self, front: List[Individual]):
        """
        Calculate crowding distance for individuals in a front.

        Crowding distance measures density of solutions around an individual.
        Higher = more isolated (better for diversity).
        """
        if len(front) <= 2:
            for ind in front:
                ind.crowding_distance = np.inf
            return

        # Initialize
        for ind in front:
            ind.crowding_distance = 0.0

        # For each objective
        for obj in self.objectives:
            # Sort by this objective
            front_sorted = sorted(front, key=lambda x: x.objectives[obj.name])

            # Boundary points get infinite distance
            front_sorted[0].crowding_distance = np.inf
            front_sorted[-1].crowding_distance = np.inf

            # Normalize objective range
            obj_min = front_sorted[0].objectives[obj.name]
            obj_max = front_sorted[-1].objectives[obj.name]
            obj_range = obj_max - obj_min

            if obj_range == 0:
                continue

            # Calculate crowding distance
            for i in range(1, len(front_sorted) - 1):
                distance = (front_sorted[i+1].objectives[obj.name] -
                           front_sorted[i-1].objectives[obj.name]) / obj_range
                front_sorted[i].crowding_distance += distance

    def tournament_selection(self, population: List[Individual], tournament_size: int = 2) -> Individual:
        """
        Tournament selection.

        Select best individual from random tournament based on:
        1. Rank (lower is better)
        2. Crowding distance (higher is better) if same rank
        """
        candidates = np.random.choice(population, size=tournament_size, replace=False)

        best = candidates[0]
        for candidate in candidates[1:]:
            if candidate.rank < best.rank:
                best = candidate
            elif candidate.rank == best.rank:
                if candidate.crowding_distance > best.crowding_distance:
                    best = candidate

        return best

    def crossover(self, parent1: Individual, parent2: Individual, eta: float = 20.0) -> Tuple[Individual, Individual]:
        """
        Simulated Binary Crossover (SBX).

        Args:
            parent1, parent2: Parent individuals
            eta: Distribution index (higher = more similar to parents)

        Returns:
            Two offspring
        """
        child1_params = {}
        child2_params = {}

        for param_name in self.param_names:
            p1 = parent1.parameters[param_name]
            p2 = parent2.parameters[param_name]

            # Random crossover
            u = np.random.random()

            if u <= 0.5:
                beta = (2 * u) ** (1.0 / (eta + 1))
            else:
                beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))

            c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
            c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)

            # Clip to bounds
            min_val, max_val = self.parameter_space[param_name]
            child1_params[param_name] = np.clip(c1, min_val, max_val)
            child2_params[param_name] = np.clip(c2, min_val, max_val)

        child1 = Individual(parameters=child1_params)
        child2 = Individual(parameters=child2_params)

        return child1, child2

    def mutate(self, individual: Individual, mutation_rate: float = 0.1, eta: float = 20.0):
        """
        Polynomial mutation.

        Args:
            individual: Individual to mutate (in-place)
            mutation_rate: Probability of mutating each parameter
            eta: Distribution index
        """
        for param_name in self.param_names:
            if np.random.random() < mutation_rate:
                val = individual.parameters[param_name]
                min_val, max_val = self.parameter_space[param_name]

                delta = max_val - min_val
                u = np.random.random()

                if u < 0.5:
                    delta_q = (2 * u) ** (1.0 / (eta + 1)) - 1.0
                else:
                    delta_q = 1.0 - (2 * (1 - u)) ** (1.0 / (eta + 1))

                val_new = val + delta_q * delta
                individual.parameters[param_name] = np.clip(val_new, min_val, max_val)

    def optimize(self, pop_size: int = 100, n_generations: int = 100,
                crossover_prob: float = 0.9, mutation_prob: float = 0.1,
                seed: Optional[int] = None) -> OptimizationResult:
        """
        Run NSGA-II optimization.

        Args:
            pop_size: Population size
            n_generations: Number of generations
            crossover_prob: Crossover probability
            mutation_prob: Mutation probability per parameter
            seed: Random seed

        Returns:
            OptimizationResult with Pareto front
        """
        if seed is not None:
            np.random.seed(seed)

        # Initialize population
        population = self.initialize_population(pop_size, seed)

        convergence_history = []

        # Evolution loop
        for gen in range(n_generations):
            # Generate offspring
            offspring = []

            while len(offspring) < pop_size:
                # Selection
                parent1 = self.tournament_selection(population)
                parent2 = self.tournament_selection(population)

                # Crossover
                if np.random.random() < crossover_prob:
                    child1, child2 = self.crossover(parent1, parent2)
                else:
                    child1 = copy.deepcopy(parent1)
                    child2 = copy.deepcopy(parent2)

                # Mutation
                self.mutate(child1, mutation_prob)
                self.mutate(child2, mutation_prob)

                # Evaluate
                self.evaluate_individual(child1)
                self.evaluate_individual(child2)

                offspring.extend([child1, child2])

            # Combine parent and offspring
            combined = population + offspring[:pop_size]

            # Non-dominated sorting
            fronts = self.fast_non_dominated_sort(combined)

            # Select next generation
            population = []
            for front in fronts:
                if len(population) + len(front) <= pop_size:
                    # Include entire front
                    self.calculate_crowding_distance(front)
                    population.extend(front)
                else:
                    # Include part of front based on crowding distance
                    self.calculate_crowding_distance(front)
                    front_sorted = sorted(front, key=lambda x: x.crowding_distance, reverse=True)
                    population.extend(front_sorted[:pop_size - len(population)])
                    break

            # Track convergence (size of Pareto front)
            convergence_history.append(len(fronts[0]))

        # Extract final Pareto front
        final_fronts = self.fast_non_dominated_sort(population)
        pareto_front = ParetoFront(individuals=final_fronts[0])

        return OptimizationResult(
            pareto_front=pareto_front,
            n_generations=n_generations,
            n_evaluations=self.n_evaluations,
            convergence_history=convergence_history
        )
