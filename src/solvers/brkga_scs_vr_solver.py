### VERSÃO REVISADA PARA SINTONIA!

import os
import math
import time
import numpy as np
import random

from typing import Tuple, List, Dict, Any, Callable, Optional
from dataclasses import dataclass

# =====================================================================
# 1. CLASSES DE CONFIGURAÇÃO E GERENCIAMENTO (ARQUITETURA NOVA)
# =====================================================================
@dataclass
class Config:
    tam_pop: int = 100
    pe: float = 0.15
    pm: float = 0.25
    rho: float = 0.85
    raio_cluster: float = 0.5
    z_limiar: float = 1.645
    votos_limiar: int = 5
    step_init: float = 0.1
    step_reduction: float = 0.5
    min_step: float = 0.001
    max_hj_evals: int = 600
    max_generations: int = 129999
    max_evals_total: int = 20000
    t_param: Optional[float] = None
    obj: str = "Makespan"

class BudgetManager:
    def __init__(self, max_evals: int, t_limite_segundos: Optional[float] = None):
        self.max_evals = max_evals
        self.t_limite_segundos = t_limite_segundos
        self.t0_global = time.time()
        self.evals_count = 0

    def increment_eval(self, count: int = 1):
        self.evals_count += count

    def is_exhausted(self, current_best: float, lb: float) -> bool:
        if current_best <= lb:
            return True
        if self.t_limite_segundos is not None:
            return (time.time() - self.t0_global) >= self.t_limite_segundos
        return self.evals_count >= self.max_evals

class ClusterStateful:
    def __init__(self, centro: np.ndarray, fitness: float):
        self.centro = centro.copy()
        self.votos = 1
        self.fitness_centro = fitness

# =====================================================================
# 3. BUSCA LOCAL: COORDINATE DESCENT (Mecânica Antiga + Budget Nova)
# =====================================================================
def local_search_coordinate_descent(
    x_init: np.ndarray, eval_fn: Callable[[np.ndarray], float],
    budget_mgr: BudgetManager, config: Config, lb: float
) -> Tuple[np.ndarray, float]:
    """
    Recupera a lógica vitoriosa do antigo Hooke-Jeeves (apenas movimentos
    +step e -step em cada dimensão), mas agora respeitando o BudgetManager.
    """
    dim = len(x_init)
    x_base = np.clip(np.copy(x_init), 0.0, 1.0)

    f_base = eval_fn(x_base)
    # O eval_fn já incrementa o budget_mgr internamente.

    x_best = np.copy(x_base)
    f_best = f_base

    step = config.step_init
    hj_evals = 1

    while step > config.min_step and hj_evals < config.max_hj_evals and not budget_mgr.is_exhausted(f_best, lb):
        x_novo = np.copy(x_base)
        melhorou = False

        for i in range(dim):
            if hj_evals >= config.max_hj_evals or budget_mgr.is_exhausted(f_best, lb):
                break

            for sinal in [+1.0, -1.0]:
                if hj_evals >= config.max_hj_evals or budget_mgr.is_exhausted(f_best, lb):
                    break

                x_teste = np.copy(x_novo)
                x_teste[i] = np.clip(x_teste[i] + sinal * step, 0.0, 1.0)

                f_teste = eval_fn(x_teste) # Increments budget
                hj_evals += 1

                if f_teste < f_best:
                    f_best = f_teste
                    x_novo = np.copy(x_teste)
                    x_best = np.copy(x_teste)
                    melhorou = True
                    break

        if melhorou:
            x_base = np.copy(x_novo)
        else:
            step *= config.step_reduction

    return x_best, f_best

# =====================================================================
# 4. ALGORITMO PRINCIPAL (2P-BRKGA-SCS HÍBRIDO)
# =====================================================================
def run_brkga_scs_hybrid(
    engine,
    n: int,
    m: int,
    lb: int,
    config: Config,
    seed: int = 42
) -> Dict[str, Any]:

    np.random.seed(seed)
    history_npfs = []

    t_limite_segundos = None
    if config.t_param is not None:
        t_limite_segundos = (n * m * config.t_param / 2.0) / 1000.0

    budget_mgr = BudgetManager(
        max_evals=config.max_evals_total,
        t_limite_segundos=t_limite_segundos
    )

    # Wrappers que incrementam o budget automaticamente
    def avaliar_pfs(sol: np.ndarray) -> float:
        budget_mgr.increment_eval(1)
        return engine.evaluate(sol, mode="PFS", objective=config.obj)

    def avaliar_npfs(sol: np.ndarray) -> float:
        budget_mgr.increment_eval(1)
        return engine.evaluate(sol, mode="NPFS", objective=config.obj)

    best_global_pfs = None
    best_fitness_pfs = float('inf')

    best_global_npfs = None
    best_fitness_npfs = float('inf')

    dim_pfs = n
    n_elite = max(1, int(config.tam_pop * config.pe))
    n_mutants = max(1, int(config.tam_pop * config.pm))
    n_crossover = config.tam_pop - n_elite - n_mutants

    populacao = np.random.uniform(0.0, 1.0, size=(config.tam_pop, dim_pfs))
    fitness_pfs = np.array([avaliar_pfs(ind) for ind in populacao])

    clusters: List[ClusterStateful] = []

    for gen in range(config.max_generations):
        if budget_mgr.is_exhausted(best_fitness_npfs, lb):
            break

        idx_ordenados = np.argsort(fitness_pfs)
        populacao = populacao[idx_ordenados]
        fitness_pfs = fitness_pfs[idx_ordenados]

        # Rastreamento Best PFS
        if fitness_pfs[0] < best_fitness_pfs:
            best_fitness_pfs = fitness_pfs[0]
            best_global_pfs = populacao[0].copy()

            # === SINCRONIZAÇÃO OBRIGATÓRIA (PFS -> NPFS) ===
            # Se o novo best PFS for estritamente melhor que o best NPFS atual,
            # atualizamos o NPFS imediatamente.
            if best_fitness_pfs < best_fitness_npfs:
                best_fitness_npfs = best_fitness_pfs
                best_global_npfs = np.tile(best_global_pfs, m)

        if budget_mgr.is_exhausted(best_fitness_pfs, lb):
            break

        # Módulo de Clustering
        for i in range(n_elite):
            elite_ind = populacao[i]
            elite_fit = fitness_pfs[i]
            min_dist = float('inf')
            nearest_cluster = None

            for c in clusters:
                dist = np.linalg.norm(c.centro - elite_ind)
                if dist < min_dist:
                    min_dist = dist
                    nearest_cluster = c

            if nearest_cluster is not None and min_dist < config.raio_cluster:
                nearest_cluster.votos += 1
                if elite_fit < nearest_cluster.fitness_centro:
                    nearest_cluster.centro = elite_ind.copy()
                    nearest_cluster.fitness_centro = elite_fit
            else:
                clusters.append(ClusterStateful(centro=elite_ind, fitness=elite_fit))

        # Cross-Space Knowledge Transfer & SCS
        media_pop = np.mean(fitness_pfs)
        std_pop = np.std(fitness_pfs) + 1e-9

        for c in clusters:
            if budget_mgr.is_exhausted(best_fitness_npfs, lb):
                break

            z_score = (c.fitness_centro - media_pop) / std_pop

            if (z_score <= -config.z_limiar) or (c.votos >= config.votos_limiar):
                # 1. Expansão (PFS -> NPFS)
                cromossomo_expandido = np.tile(c.centro, m)

                # 2. Busca Local (Coordinate Descent Adaptado)
                sol_refinada, fit_refinado = local_search_coordinate_descent(
                    x_init=cromossomo_expandido,
                    eval_fn=avaliar_npfs,
                    budget_mgr=budget_mgr,
                    config=config,
                    lb=lb
                )

                # 3. Atualização NPFS Best
                if fit_refinado < best_fitness_npfs:
                    best_fitness_npfs = fit_refinado
                    best_global_npfs = sol_refinada.copy()

                c.votos = 0

        if budget_mgr.is_exhausted(best_fitness_npfs, lb):
            break

        # Evolução BRKGA
        nova_pop = np.zeros_like(populacao)
        nova_pop[:n_elite] = populacao[:n_elite]
        nova_pop[n_elite:n_elite + n_mutants] = np.random.uniform(0.0, 1.0, size=(n_mutants, dim_pfs))

        for i in range(n_crossover):
            pai_e = populacao[np.random.randint(0, n_elite)]
            pai_ne = populacao[np.random.randint(n_elite, config.tam_pop)]
            mascara = np.random.rand(dim_pfs) < config.rho
            nova_pop[n_elite + n_mutants + i] = np.where(mascara, pai_e, pai_ne)

        populacao = nova_pop

        # Só avalia se ainda tiver budget
        for i in range(config.tam_pop):
            if budget_mgr.is_exhausted(best_fitness_npfs, lb):
                break
            fitness_pfs[i] = avaliar_pfs(populacao[i])

        tempo_decorrido = time.time() - budget_mgr.t0_global
        history_npfs.append((budget_mgr.evals_count, tempo_decorrido, best_fitness_npfs))

    return {
        "best_global_pfs": best_global_pfs,
        "best_fitness_pfs": best_fitness_pfs,
        "best_global_npfs": best_global_npfs,
        "best_fitness_npfs": best_fitness_npfs,
        "total_evaluations": budget_mgr.evals_count,
        "lower_bound": lb,
        "history": history_npfs,
    }