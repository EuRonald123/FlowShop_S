"""
Solver 2P-BRKGA-SCS (Clustering Search + Cross-Space Knowledge Transfer).

Referência:
  - 2P-BRKGA-SCS: BRKGA com Cross-Space Knowledge Transfer via Clustering Search
    e Busca Local Hooke & Jeeves no espaço NPFS.
"""

import time
import math
import numpy as np
from typing import Tuple, List, Dict, Any, Callable, Optional


# ============================================================================
# 1. DECODIFICADOR TWO-PHASE (PFS / NPFS)
# ============================================================================

def decodificar_two_phase(
    cromossomo: np.ndarray,
    n: int,
    m: int,
    p_matrix: np.ndarray,
    due_dates: np.ndarray = None,
    modo: str = "PFS",
    obj: str = "SPAM"
) -> float:
    """
    Decodifica o cromossomo contínuo no ambiente Wait Allowed (Flow Shop).

    Parâmetros:
      - cromossomo: Vetor contínuo de Random Keys.
                    Modo PFS  -> Tamanho N (Replicado para as M máquinas).
                    Modo NPFS -> Tamanho N * M (ou matriz M x N).
      - n: Número de tarefas (jobs).
      - m: Número de máquinas.
      - p_matrix: Matriz de tempos de processamento (N x M).
      - due_dates: Vetor de prazos de entrega (Tamanho N). Opcional.
      - modo: 'PFS' (mesma sequência em todas as máquinas) ou 'NPFS'.
      - obj: 'SPAM'/'SPAN'/'CMAX'/'MAKESPAN' ou 'LMAX'/'TMAX' (atraso máximo).

    Retorna:
      - float: Valor da função objetivo calculada.
    """
    modo_upper = modo.upper().strip()
    obj_upper = obj.upper().strip()

    # 1. Obtenção da ordem de processamento por máquina
    if modo_upper == "PFS":
        seq_unica = np.argsort(cromossomo[:n])
        ordens_maquinas = [seq_unica for _ in range(m)]

    elif modo_upper == "NPFS":
        if cromossomo.ndim == 1:
            chaves_maquinas = cromossomo[:n * m].reshape((m, n))
        else:
            chaves_maquinas = cromossomo

        ordens_maquinas = [np.argsort(chaves_maquinas[maq, :]) for maq in range(m)]

    else:
        raise ValueError(f"Modo '{modo}' inválido. Utilize 'PFS' ou 'NPFS'.")

    # 2. Simulação do Chão de Fábrica (Wait Allowed)
    completion_time = np.zeros((n, m), dtype=float)

    # Máquina 0 (Primeiro Estágio)
    tempo_maq = 0.0
    for job in ordens_maquinas[0]:
        tempo_maq += p_matrix[job, 0]
        completion_time[job, 0] = tempo_maq

    # Máquinas Subsequentes (1 até M - 1)
    for maq in range(1, m):
        tempo_maq = 0.0
        for job in ordens_maquinas[maq]:
            pronto_job = completion_time[job, maq - 1]
            inicio = max(tempo_maq, pronto_job)
            completion_time[job, maq] = inicio + p_matrix[job, maq]
            tempo_maq = completion_time[job, maq]

    # 3. Retorno do Objetivo
    if obj_upper in ["SPAM", "SPAN", "CMAX", "MAKESPAN"]:
        return float(np.max(completion_time[:, m - 1]))

    elif obj_upper in ["LMAX", "TMAX"]:
        if due_dates is None:
            raise ValueError("due_dates é obrigatório para obj='LMAX'.")
        atrasos = np.maximum(0.0, completion_time[:, m - 1] - due_dates)
        return float(np.max(atrasos))

    else:
        raise ValueError(
            f"Objetivo '{obj}' inválido. Escolha 'SPAM' (Makespan) ou 'LMAX' (Atraso Máximo)."
        )


# ============================================================================
# 2. BUSCA LOCAL HOOKE & JEEVES ADAPTADA
# ============================================================================

def hooke_jeeves_local_search(
    cromossomo_init: np.ndarray,
    avaliar_func: Callable[[np.ndarray], float],
    step_init: float = 0.1,
    step_reduction: float = 0.5,
    min_step: float = 0.001,
    max_evals: int = 500
) -> Tuple[np.ndarray, float, int]:
    """
    Hooke & Jeeves operando no espaço contínuo [0, 1]^D com clipping.
    """
    dim = len(cromossomo_init)
    best_x = np.clip(cromossomo_init.copy(), 0.0, 1.0)
    best_fit = avaliar_func(best_x)
    evals = 1

    x_base = best_x.copy()
    step = step_init

    while step > min_step and evals < max_evals:
        x_novo = x_base.copy()
        melhorou = False

        # Movimento Exploratório
        for i in range(dim):
            for sinal in [+1, -1]:
                if evals >= max_evals:
                    break
                x_teste = x_novo.copy()
                x_teste[i] = np.clip(x_teste[i] + sinal * step, 0.0, 1.0)
                fit_teste = avaliar_func(x_teste)
                evals += 1

                if fit_teste < best_fit:
                    best_fit = fit_teste
                    x_novo = x_teste.copy()
                    melhorou = True
                    break

        if melhorou:
            x_base = x_novo.copy()
        else:
            step *= step_reduction

    return (best_x if best_fit < avaliar_func(best_x) else x_novo), best_fit, evals


# ============================================================================
# 3. CLASSE DE SUPORTE PARA CLUSTER STATEFUL
# ============================================================================

class ClusterStateful:
    """Cluster com estado: centro, votos acumulados e fitness do centro."""
    def __init__(self, centro: np.ndarray, fitness: float):
        self.centro = centro.copy()
        self.votos = 1
        self.fitness_centro = fitness


# ============================================================================
# 4. ALGORITMO PRINCIPAL — 2P-BRKGA-SCS
# ============================================================================

def run_brkga_scs(
    n: int,
    m: int,
    p_matrix: np.ndarray,
    due_dates: np.ndarray,
    lb: int,
    tam_pop: int = 100,
    pe: float = 0.2,         # fração de elites
    pm: float = 0.15,        # fração de mutantes
    rho: float = 0.7,        # viés do crossover
    raio_cluster: float = 0.5,
    z_limiar: float = 1.5,
    votos_limiar: int = 5,
    step_init: float = 0.1,
    step_reduction: float = 0.5,
    min_step: float = 0.001,
    
    max_evals_total: int = 20000,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Executa o 2P-BRKGA-SCS com Cross-Space Knowledge Transfer e Duplo Best Tracking.

    Parâmetros:
      n, m, p_matrix, due_dates: dados da instância.
      lb: Lower Bound para early stopping.
      tam_pop: tamanho da população.
      pe: fração de elites (0..1).
      pm: fração de mutantes (0..1).
      rho: probabilidade de herdar gene do pai elite no crossover.
      raio_cluster: raio inicial para agrupamento (não usado diretamente,
                    o raio adaptativo é calculado a cada geração).
      z_limiar: threshold Z-Score para ativar SCS.
      votos_limiar: número de votos para ativar SCS.
      step_init, step_reduction, min_step: parâmetros do H&J.
            max_evals_total: limite máximo de avaliações total.
      seed: semente aleatória.

    Retorna:
      dict com:
        - best_global_pfs: melhor solução no espaço PFS (n,)
        - best_fitness_pfs: makespan da melhor PFS
        - best_global_npfs: melhor solução no espaço NPFS (n*m,)
        - best_fitness_npfs: makespan da melhor NPFS
        - total_evaluations: total de avaliações usadas
        - lower_bound: LB da instância
        - history: lista de best_fitness_npfs por geração (para convergência)
    """
    np.random.seed(seed)
    evals_count = 0
    history_npfs = []  # rastreamento de convergência

    # 2. Wrappers de Avaliação
    def avaliar_pfs(sol: np.ndarray) -> float:
        nonlocal evals_count
        evals_count += 1
        return decodificar_two_phase(sol, n, m, p_matrix, due_dates, modo="PFS", obj="SPAM")

    def avaliar_npfs(sol: np.ndarray) -> float:
        nonlocal evals_count
        evals_count += 1
        return decodificar_two_phase(sol, n, m, p_matrix, due_dates, modo="NPFS", obj="SPAM")

    # Duplo Best Rastreamento
    best_global_pfs = None
    best_fitness_pfs = float('inf')

    best_global_npfs = None
    best_fitness_npfs = float('inf')

    # 3. Inicialização BRKGA (Espaço Reduzido D = N)
    dim_pfs = n
    n_elite = max(1, int(tam_pop * pe))
    n_mutants = max(1, int(tam_pop * pm))
    n_crossover = tam_pop - n_elite - n_mutants

    populacao = np.random.uniform(0.0, 1.0, size=(tam_pop, dim_pfs))
    fitness_pfs = np.array([avaliar_pfs(ind) for ind in populacao])

    clusters: List[ClusterStateful] = []

    # Loop Evolutivo
    while evals_count < max_evals_total:
        if evals_count >= max_evals_total or best_fitness_npfs <= lb:
            break

        # Ordenação por Fitness PFS
        idx_ordenados = np.argsort(fitness_pfs)
        populacao = populacao[idx_ordenados]
        fitness_pfs = fitness_pfs[idx_ordenados]

        # Rastreamento Best PFS
        if fitness_pfs[0] < best_fitness_pfs:
            best_fitness_pfs = fitness_pfs[0]
            best_global_pfs = populacao[0].copy()

        # Early Stopping
        if best_fitness_pfs <= lb:
            best_fitness_npfs = min(best_fitness_npfs, best_fitness_pfs)
            break

        # 4. MÓDULO DE CLUSTERING (Atualização com Elite)
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

            # Cálculo do Raio Dinâmico
            if len(clusters) > 0:
                raio_dinamico = 1.0 / (2 * (len(clusters) ** (1.0 / n)))
            else:
                raio_dinamico = raio_cluster  # fallback para o parâmetro

            if nearest_cluster is not None and min_dist < raio_dinamico:
                nearest_cluster.votos += 1
                # Substitui centro se o elite for superior
                if elite_fit < nearest_cluster.fitness_centro:
                    nearest_cluster.centro = elite_ind.copy()
                    nearest_cluster.fitness_centro = elite_fit
            else:
                clusters.append(ClusterStateful(centro=elite_ind, fitness=elite_fit))

        # 5. CROSS-SPACE KNOWLEDGE TRANSFER & SCS
        media_pop = np.mean(fitness_pfs)
        std_pop = np.std(fitness_pfs) + 1e-9

        for c in clusters:
            if evals_count >= max_evals_total or best_fitness_npfs <= lb:
                break

            z_score = (c.fitness_centro - media_pop) / std_pop

            # Gatilho de Ativação do SCS
            if (z_score <= -z_limiar) or (c.votos >= votos_limiar):
                # Passo 1: Projeção PFS -> NPFS via np.tile
                cromossomo_expandido = np.tile(c.centro, m)

                # Passo 2: Explotação via Hooke & Jeeves no Espaço NPFS
                budget_restante = max_evals_total - evals_count
                sol_refinada, fit_refinado, evals_hj = hooke_jeeves_local_search(
                    cromossomo_expandido,
                    avaliar_npfs,
                    step_init=step_init,
                    step_reduction=step_reduction,
                    min_step=min_step,
                    max_evals=min(500, budget_restante)
                )

                # Passo 3: Atualização do NPFS Best
                if fit_refinado < best_fitness_npfs:
                    best_fitness_npfs = fit_refinado
                    best_global_npfs = sol_refinada.copy()

                # Passo 4: Reset do Estado do Cluster
                c.votos = 0

        if best_fitness_npfs <= lb:
            break

        # 6. Evolução da População (BRKGA Crossover & Mutantes)
        nova_pop = np.zeros_like(populacao)
        nova_pop[:n_elite] = populacao[:n_elite]
        nova_pop[n_elite:n_elite + n_mutants] = np.random.uniform(
            0.0, 1.0, size=(n_mutants, dim_pfs)
        )

        for i in range(n_crossover):
            pai_e = populacao[np.random.randint(0, n_elite)]
            pai_ne = populacao[np.random.randint(n_elite, tam_pop)]
            mascara = np.random.rand(dim_pfs) < rho
            nova_pop[n_elite + n_mutants + i] = np.where(mascara, pai_e, pai_ne)

        populacao = nova_pop
        fitness_pfs = np.array([avaliar_pfs(ind) for ind in populacao])

        # Registra histórico de convergência
        history_npfs.append((evals_count, best_fitness_npfs))

    return {
        "best_global_pfs": best_global_pfs,
        "best_fitness_pfs": best_fitness_pfs,
        "best_global_npfs": best_global_npfs,
        "best_fitness_npfs": best_fitness_npfs,
        "total_evaluations": evals_count,
        "lower_bound": lb,
        "history": history_npfs,
    }