import os
import time
import numpy as np
import random

from typing import Tuple, List, Dict, Any, Callable, Optional
from dataclasses import dataclass

# =====================================================================
# 1. CLASSES DE CONFIGURAÇÃO E GERENCIAMENTO (ARQUITETURA MANTIDA)
# =====================================================================
@dataclass
class Config:
    tam_pop: int = 125
    pe: float = 0.15
    pm: float = 0.15
    rho: float = 0.6
    raio_cluster: float = 0.15
    z_limiar: float = 1.645
    votos_limiar: int = 5
    max_ls_evals: int = 500  # Renomeado de max_hj_evals para generalizar
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
# 4. BUSCA LOCAL DISCRETA: r-NEH (CORRIGIDO)
# =====================================================================
def local_search_r_neh(
    x_init: np.ndarray,
    engine: Any,
    n: int, m: int,
    budget_mgr: BudgetManager, 
    config: Config, 
    lb: float
) -> Tuple[np.ndarray, float]:
    """
    Busca Local no Espaço de Imagem (r-NEH / Iterated Greedy).
    Decodifica o cromossomo contínuo, opera nas permutações e codifica de volta.
    """
    # 1. Decodificar x_init contínuo para permutações discretas (NPFS)
    chaves_maquinas = x_init.copy().reshape((m, n))
    seq_atual = np.array([np.argsort(chaves_maquinas[maq, :]) for maq in range(m)])

    # Avaliação Base
    def avalia_seq_discreta(seq: np.ndarray) -> float:
        budget_mgr.increment_eval(1)
        engine.cfo_count += 1
        return float(engine._custo(engine._simular(seq), objective=config.obj))

    best_seq = seq_atual.copy()
    best_fit = avalia_seq_discreta(best_seq)

    ls_evals = 1
    melhorou_iteracao = True

    # 2. Loop de Busca (r-NEH / Inserção Gulosa)
    while melhorou_iteracao and ls_evals < config.max_ls_evals and not budget_mgr.is_exhausted(best_fit, lb):
        melhorou_iteracao = False

        # Escolhe aleatoriamente uma máquina e um job para reposicionar
        maq_alvo = random.randint(0, m - 1)
        job_alvo_idx = random.randint(0, n - 1)
        job_id = best_seq[maq_alvo, job_alvo_idx]

        # SOLUÇÃO AQUI: Isola a linha em um array 1D (tamanho n-1), sem tocar na matriz 2D
        linha_reduzida = np.delete(best_seq[maq_alvo], job_alvo_idx)

        melhor_pos = job_alvo_idx
        melhor_fit_local = best_fit

        # Testa inserção do job em todas as posições possíveis daquela máquina
        for pos in range(n):
            if budget_mgr.is_exhausted(melhor_fit_local, lb) or ls_evals >= config.max_ls_evals:
                break

            # Cria a linha candidata reinserindo o job (tamanho volta a ser n)
            linha_candidata = np.insert(linha_reduzida, pos, job_id)

            # Monta a matriz 2D completa e válida para avaliação
            seq_candidata = best_seq.copy()
            seq_candidata[maq_alvo] = linha_candidata

            fit_candidato = avalia_seq_discreta(seq_candidata)
            ls_evals += 1

            if fit_candidato < melhor_fit_local:
                melhor_fit_local = fit_candidato
                melhor_pos = pos
                print("\n Melhorou no r-NEH")
                melhorou_iteracao = True

        # Atualiza o best_seq global se encontrou melhoria
        if melhorou_iteracao:
            best_seq[maq_alvo] = np.insert(linha_reduzida, melhor_pos, job_id)
            best_fit = melhor_fit_local

    # 3. Codificar de volta para Espaço Contínuo
    x_novo = np.zeros((m, n), dtype=float)
    step = 1.0 / n
    for maq in range(m):
        for pos, job in enumerate(best_seq[maq]):
            # Adiciona um pequeno ruído para manter as propriedades contínuas do BRKGA
            x_novo[maq, job] = (pos * step) + random.uniform(0, step * 0.5)

    return x_novo.flatten(), best_fit

# =====================================================================
# 5. ALGORITMO PRINCIPAL (2P-BRKGA-SCS HÍBRIDO COM r-NEH)
# =====================================================================
def run_brkga_scs_img(
    engine,
    n: int,
    m: int,
    lb: int,
    config: Config,
    seed: int = 42
) -> Dict[str, Any]:

    np.random.seed(seed)
    random.seed(seed)
    history_npfs = []
    
    t_limite_segundos = 0
    if config.t_param is not None:
        t_limite_segundos = (n * m * config.t_param / 2.0) / 1000.0
        
    budget_mgr = BudgetManager(
        max_evals=config.max_evals_total,
        t_limite_segundos=t_limite_segundos
        ) 
    

    def avaliar_pfs(sol: np.ndarray) -> float:
        budget_mgr.increment_eval(1)
        return engine.evaluate(sol, mode="PFS", objective=config.obj)

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

        if fitness_pfs[0] < best_fitness_pfs:
            best_fitness_pfs = fitness_pfs[0]
            best_global_pfs = populacao[0].copy()

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

        # Cross-Space Knowledge Transfer & SCS (Agora chamando r-NEH)
        media_pop = np.mean(fitness_pfs)
        std_pop = np.std(fitness_pfs) + 1e-9

        for c in clusters:
            if budget_mgr.is_exhausted(best_fitness_npfs, lb):
                break

            z_score = (c.fitness_centro - media_pop) / std_pop

            if (z_score <= -config.z_limiar) or (c.votos >= config.votos_limiar):
                # 1. Expansão (PFS -> NPFS)
                cromossomo_expandido = np.tile(c.centro, m)

                # 2. Busca Local Discreta (r-NEH)
                sol_refinada, fit_refinado = local_search_r_neh(
                    x_init=cromossomo_expandido,
                    engine=engine,
                    n=n, 
                    m=m,
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

        for i in range(config.tam_pop):
            if budget_mgr.is_exhausted(best_fitness_npfs, lb):
                break
            fitness_pfs[i] = avaliar_pfs(populacao[i])

    return {
        "best_global_pfs": best_global_pfs,
        "best_fitness_pfs": best_fitness_pfs,
        "best_global_npfs": best_global_npfs,
        "best_fitness_npfs": best_fitness_npfs,
        "total_evaluations": budget_mgr.evals_count,
        "lower_bound": lb,
        "history_npfs": history_npfs
    }