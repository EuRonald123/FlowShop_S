import time
import numpy as np
from typing import Optional, Tuple, List

class BRKGARSCSolver:
    """
    Solver BRKGA-RSC for FlowShop Scheduling
    """
    
    def __init__(
        self,
        engine,
        seed: int = 42,
        objective: str = "Makespan",
        size_population: int = 100,
        pe: float = 0.20,
        pm: float = 0.15,
        rho: float = 0.70,
        max_evals: int = 50000,
        t_param: Optional[float] = None,
        lb: float = 0.0,
    ):
        self.engine = engine
        self.n = engine.n_jobs
        self.m = engine.n_machines
        self.objective = objective
        self.lb = lb
        
        # hyperparameters BRKGA
        self.size_population = size_population
        self.n_elite = max(1, int(size_population * pe))
        self.n_mutants = max(1, int(size_population * pm))
        self.n_children = size_population - self.n_elite - self.n_mutants
        self.rho = rho
        
        # budget control (time or evaluations)
        self.max_evals = max_evals
        self.time_limit = (self.n * self.m * t_param / 2000.0) if t_param is not None else None
        self.n_evals = 0
        self.t0 = 0.0
        
        # random number generator
        self.rng = np.random.default_rng(seed)
        
        # best solution found
        self.best_key: Optional[np.ndarray] = None
        self.best_cost: float = float('inf')
        self.history: List[Tuple[int, float]] = []
        
    def _stop_criterion(self):
        if self.best_cost <= self.lb:
            return True
        if self.time_limit is not None:
            return (time.time() - self.t0) >= self.time_limit
        return self.n_evals >= self.max_evals
    
    def _evaluate_key(self, key):
        self.n_evals += 1
        cost = float(self.engine.evaluate(key, mode="PFS", objective=self.objective))
        if cost < self.best_cost:
            self.best_cost = cost
            self.best_key = key.copy()
        return cost
    
    def _initialize_population(self):
        """Generate and evaluate the initial population"""
        population = self.rng.random((self.size_population, self.n))
        costs = np.array([self._evaluate_key(ind) for ind in population])
        return self._order_population(population, costs)
    
    def _order_population(self, population, costs):
        """Order the population by cost (ascending)"""
        order = np.argsort(costs)
        return population[order], costs[order]
    
    def _cross(self, elite, non_elite):
        idx_elite = self.rng.integers(0, len(elite), size=self.n_children)
        idx_non_elite = self.rng.integers(0, len(non_elite), size=self.n_children)
        mask = self.rng.random((self.n_children, self.n)) < self.rho
        return np.where(mask, elite[idx_elite], non_elite[idx_non_elite])
    
    #LOOP
    def execute(self):
        self.t0 = time.time()
        self.n_evals = 0
        
        population, costs = self._initialize_population()
        self.history.append((self.n_evals, time.time() - self.t0, self.best_cost))
        
        while not self._stop_criterion():
            elite = population[:self.n_elite]
            non_elite = population[self.n_elite :]
            
            # Generate mutants and children
            mutants = self.rng.random((self.n_mutants, self.n))
            children = self._cross(elite, non_elite)
            
            #new population (Elite + Children + Mutants)
            population = np.vstack((elite, children, mutants))
            
            #evaluate new individuals (non-elite)
            for i in range(self.n_elite, self.size_population):
                if self._stop_criterion():
                    break
                costs[i] = self._evaluate_key(population[i])
            
            # Order the new population
            population, costs = self._order_population(population, costs)
            self.history.append((self.n_evals, time.time() - self.t0, self.best_cost))
            
        best_permutation = np.argsort(self.best_key)
        return best_permutation, self.best_cost, self.n_evals, self.history