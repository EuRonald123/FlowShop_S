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
        
        #Individual 0: Exact NEH soluction
        neh_permutation = self._build_neh_permutation()
        population[0] = self._permutation_to_key(neh_permutation)
        
        #Individuals 1-4: Variants of NEH solution
        n_variants = min(4, self.size_population - 1)
        for v in range(1, n_variants + 1):
            perm_var = neh_permutation.copy()
            i, j = self.rng.choice(self.n, size=2, replace=False)
            perm_var[i], perm_var[j] = perm_var[j], perm_var[i]
            population[v] = self._permutation_to_key(perm_var)

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
    
    
    # From here on, there are improvements on top of the pure BRKGA
    
    def _permutation_to_key(self, permutation):
        key = np.empty(self.n, dtype=float)
        key[permutation] = (np.arange(self.n, dtype=float)+ 0.5) / self.n
        return key
    
    
    def _best_neh_insertion(
        self, seq: List[int], job: int, orig_pos: Optional[int] = None
    ) -> List[int]:
        self.n_evals += 1
        k = len(seq)
        m = self.m
        pt = self.engine.proc_times
        
        e = np.zeros((k, m), dtype=float)
        e[0, 0] = pt[seq[0], 0]
        for j in range(1, m):
            e[0, j] = e[0, j - 1] + pt[seq[0], j]
        for i in range(1, k):
            e[i, 0] = e[i - 1, 0] + pt[seq[i], 0]
            for j in range(1, m):
                e[i, j] = max(e[i - 1, j], e[i, j - 1]) + pt[seq[i], j]
        
        q = np.zeros((k + 1, m), dtype=float)
        for i in range(k - 1, -1, -1):
            p_job = pt[seq[i], :]
            q[i, m - 1] = q[i + 1, m - 1] + p_job[m - 1]
            for j in range(m - 2, -1, -1):
                q[i, j] = max(q[i + 1, j], q[i, j + 1]) + p_job[j]
                
        p_new = pt[job]
        f = np.zeros((k + 1, m), dtype=float)
        f[0, 0] = p_new[0]
        f[1:, 0] = e[:, 0] + p_new[0]
        for j in range(1, m):
            f[0, j] = f[0, j - 1] + p_new[j]
            f[1:, j] = np.maximum(e[:, j], f[1:, j - 1]) + p_new[j]
        
        makespan = np.max(f + q, axis=1)
        best_position = int(np.argmin(makespan))

        # Strict improvement rule: keep original position on ties
        if orig_pos is not None and makespan[best_position] >= makespan[orig_pos]:
            best_position = orig_pos
        
        new_seq = list(seq)
        new_seq.insert(best_position, job)
        return new_seq
    
    def _build_neh_permutation(self):
        total_times = np.sum(self.engine.proc_times, axis=1)
        sorted_jobs = np.argsort(-total_times)
        
        seq = [int(sorted_jobs[0])]
        for job in sorted_jobs[1:]:
            seq = self._best_neh_insertion(seq, int(job))
        return np.array(seq, dtype=int)

    def _local_search(self, key, cost, perturb=False, d_perturb=4):
        """
        Fast Discrete Insertion Local Search + IG Perturbation
        with Lamarckian re-encoding into random keys.
        """
        seq = list(np.argsort(key))

        # 1. Perturbation (Destruction-Construction of d jobs) to escape local optima
        if perturb and self.n > d_perturb:
            d = min(d_perturb, self.n - 1)
            removed_idx = sorted(
                self.rng.choice(self.n, size=d, replace=False), reverse=True
            )
            removed_jobs = [seq.pop(idx) for idx in removed_idx]
            for job in removed_jobs:
                seq = self._best_neh_insertion(seq, int(job))

        # 2. Insertion Local Search on a sample of jobs (strict improvement)
        n_checks = min(self.n, 25)
        jobs_to_check = self.rng.choice(self.n, size=n_checks, replace=False)

        for job in jobs_to_check:
            if self._stop_criterion():
                break
            orig_pos = seq.index(int(job))
            seq.pop(orig_pos)
            seq = self._best_neh_insertion(seq, int(job), orig_pos=orig_pos)

        # 3. Re-encode to continuous keys and evaluate
        new_key = self._permutation_to_key(np.array(seq, dtype=int))
        new_cost = self._evaluate_key(new_key)

        return new_key, new_cost

    def _micro_restart(self, population, costs):
        """
        Keep only the top 20% of the Elite intact, diversify the rest of the Elite
        via strong Destruction-Construction, and kill 100% of the Non-Elite (80% of pop)
        replacing them with fresh random individuals.
        """
        n_keep = max(1, int(self.n_elite * 0.20))

        # 1. Diversify the remaining 80% of the Elite with strong perturbations (d=6)
        for i in range(n_keep, self.n_elite):
            if self._stop_criterion():
                break
            base_idx = i % n_keep
            population[i], costs[i] = self._local_search(
                population[base_idx], costs[base_idx], perturb=True, d_perturb=6
            )

        # 2. Kill the entire Non-Elite (80% of the population) and generate fresh individuals
        n_non_elite = self.size_population - self.n_elite
        population[self.n_elite :] = self.rng.random((n_non_elite, self.n))
        for i in range(self.n_elite, self.size_population):
            if self._stop_criterion():
                break
            costs[i] = self._evaluate_key(population[i])

        return self._order_population(population, costs)
    
    #LOOP
    def execute(self):
        self.t0 = time.time()
        self.n_evals = 0
        
        population, costs = self._initialize_population()

        # Refine the best initial (NEH) individual right at the start
        if not self._stop_criterion():
            new_key, new_cost = self._local_search(
                population[0], costs[0], perturb=False
            )
            if new_cost < costs[0]:
                population[0], costs[0] = new_key, new_cost
                population, costs = self._order_population(population, costs)

        self.history.append((self.n_evals, time.time() - self.t0, self.best_cost))
        stagnation_count = 0
        
        while not self._stop_criterion():
            prev_best = self.best_cost
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

            # Passo 3: Lamarckian Local Search on Elite
            if not self._stop_criterion():
                stagnated = (self.best_cost >= prev_best)
                idx_target = 0 if not stagnated else int(self.rng.integers(0, self.n_elite))
                new_key, new_cost = self._local_search(
                    population[idx_target], costs[idx_target], perturb=stagnated
                )
                if new_cost < costs[idx_target]:
                    population[idx_target] = new_key
                    costs[idx_target] = new_cost
                    population, costs = self._order_population(population, costs)
                elif stagnated and new_cost < costs[self.n_elite - 1]:
                    # Replace the worst elite individual to prevent elite cloning
                    population[self.n_elite - 1] = new_key
                    costs[self.n_elite - 1] = new_cost
                    population, costs = self._order_population(population, costs)

            # Passo 4: Stagnation Tracking & Micro-Restart
            if self.best_cost < prev_best:
                stagnation_count = 0
            else:
                stagnation_count += 1

            if stagnation_count >= 20 and not self._stop_criterion():
                population, costs = self._micro_restart(population, costs)
                stagnation_count = 0

            self.history.append((self.n_evals, time.time() - self.t0, self.best_cost))
            
        best_permutation = np.argsort(self.best_key)
        return best_permutation, self.best_cost, self.n_evals, self.history