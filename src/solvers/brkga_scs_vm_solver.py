### VERSÃO REVISADA PARA SINTONIA!

import os
import math
import time
import numpy as np
import random
from typing import Tuple, List, Dict, Any, Callable
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
    max_generations: int = 200
    max_evals_total: int = 20000
    obj: str = "LMAX"

class BudgetManager:
    def __init__(self, max_evals: int):
        self.max_evals = max_evals
        self.evals_count = 0

    def increment_eval(self, count: int = 1):
        self.evals_count += count

    def is_exhausted(self, current_best: float, lb: float) -> bool:
        return self.evals_count >= self.max_evals or current_best <= lb

class ClusterStateful:
    def __init__(self, centro: np.ndarray, fitness: float):
        self.centro = centro.copy()
        self.votos = 1
        self.fitness_centro = fitness

# =====================================================================
# 2. LEITURA DE INSTÂNCIAS E CÁLCULO DE LOWER BOUND (CÓDIGO ANTIGO)
# =====================================================================
def ler_instancia(caminho_arquivo: str):
    with open(caminho_arquivo, 'r') as f:
        linhas = [linha.strip() for linha in f if linha.strip()]

    dimensoes = list(map(int, linhas[0].split()))
    n, m = dimensoes[0], dimensoes[1]
    p_matrix = np.zeros((n, m), dtype=float)

    idx = 2
    for j in range(n):
        valores = list(map(int, linhas[idx].split()))
        p_matrix[j, :] = valores[1::2][:m]
        idx += 1

    due_dates = None
    duedate_encontrado = False

    while idx < len(linhas):
        if "duedate" in linhas[idx].lower():
            duedate_encontrado = True
            idx += 1
            break
        idx += 1

    if duedate_encontrado:
        valores_dd = []
        while idx < len(linhas) and len(valores_dd) < n:
            nums = list(map(float, linhas[idx].split()))
            valores_dd.extend(nums)
            idx += 1
        due_dates = np.array(valores_dd[:n], dtype=float)

    return n, m, p_matrix, due_dates

def calcular_lower_bound_potts(p_matrix: np.ndarray, due_dates: np.ndarray) -> int:
    n, m = p_matrix.shape
    lb_por_maquina = []

    for maq in range(m):
        r = np.sum(p_matrix[:, :maq], axis=1) if maq > 0 else np.zeros(n)
        q = np.sum(p_matrix[:, maq + 1:], axis=1) if maq < m - 1 else np.zeros(n)
        due_dates_modificados = due_dates - q
        carga_total_maquina = np.sum(p_matrix[:, maq])
        min_r = np.min(r)

        lb_m = max(0, min_r + carga_total_maquina - np.max(due_dates_modificados))
        lb_por_maquina.append(lb_m)

    lb_jobs = np.max(np.maximum(0, np.sum(p_matrix, axis=1) - due_dates))
    return int(max(np.max(lb_por_maquina), lb_jobs))

# =====================================================================
# 3. DECODIFICADOR TWO-PHASE (CÓDIGO ANTIGO + argsort)
# =====================================================================
def decodificar_two_phase(
    cromossomo: np.ndarray, n: int, m: int,
    p_matrix: np.ndarray, due_dates: np.ndarray = None,
    modo: str = "PFS", obj: str = "LMAX"
) -> float:
    modo_upper = modo.upper().strip()
    obj_upper = obj.upper().strip()

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
        raise ValueError("Modo inválido.")

    completion_time = np.zeros((n, m), dtype=float)
    tempo_maq = 0.0
    for job in ordens_maquinas[0]:
        tempo_maq += p_matrix[job, 0]
        completion_time[job, 0] = tempo_maq

    for maq in range(1, m):
        tempo_maq = 0.0
        for job in ordens_maquinas[maq]:
            pronto_job = completion_time[job, maq - 1]
            inicio = max(tempo_maq, pronto_job)
            completion_time[job, maq] = inicio + p_matrix[job, maq]
            tempo_maq = completion_time[job, maq]

    if obj_upper in ["SPAM", "SPAN", "CMAX", "MAKESPAN"]:
        return float(np.max(completion_time[:, m - 1]))
    elif obj_upper in ["LMAX", "TMAX"]:
        atrasos = np.maximum(0.0, completion_time[:, m - 1] - due_dates)
        return float(np.max(atrasos))

# =====================================================================
# 4. BUSCA LOCAL: COORDINATE DESCENT (Mecânica Antiga + Budget Nova)
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
# 5. ALGORITMO PRINCIPAL (2P-BRKGA-SCS HÍBRIDO)
# =====================================================================
def run_brkga_scs_hybrid(
    n: int, m: int, p_matrix: np.ndarray, due_dates: np.ndarray,
    lb: int, config: Config, seed: int = 42
) -> Dict[str, Any]:

    np.random.seed(int(time.time()))
    budget_mgr = BudgetManager(max_evals=config.max_evals_total)

    # Wrappers que incrementam o budget automaticamente
    def avaliar_pfs(sol: np.ndarray) -> float:
        budget_mgr.increment_eval(1)
        return decodificar_two_phase(sol, n, m, p_matrix, due_dates, modo="PFS", obj=config.obj)

    def avaliar_npfs(sol: np.ndarray) -> float:
        budget_mgr.increment_eval(1)
        return decodificar_two_phase(sol, n, m, p_matrix, due_dates, modo="NPFS", obj=config.obj)

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
                    print(f"[Gen {gen}] Novo Best NPFS encontrado via Busca Local: {best_fitness_npfs}")

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

    return {
        "best_global_pfs": best_global_pfs,
        "best_fitness_pfs": best_fitness_pfs,
        "best_global_npfs": best_global_npfs,
        "best_fitness_npfs": best_fitness_npfs,
        "total_evaluations": budget_mgr.evals_count,
        "lower_bound": lb
    }

# =====================================================================
# EXECUÇÃO DE TESTE
# =====================================================================
if __name__ == "__main__":
    caminho = "/content/drive/MyDrive/instances/Large/Ta101_2.txt"
    # caminho = "I_2_16_5_1.txt" # Altere para testar local

    n_test, m_test, p_matrix_test, due_dates_test = ler_instancia(caminho)
    lb_test = calcular_lower_bound_potts(p_matrix_test, due_dates_test)

    print(f"Lower Bound Calculado (McMahon-Florian/Potts): {lb_test}")

    # Inicializa a classe de Configuração herdando os parâmetros do seu código base
    cfg = Config(
        max_evals_total = n_test * m_test * 100,
        max_generations = 129999,
        obj = "SPAM"
    )

    resultado = run_brkga_scs_hybrid(
        n=n_test, m=m_test,
        p_matrix=p_matrix_test, due_dates=due_dates_test,
        lb=lb_test, config=cfg
    )

    print("\n================ RESULTADOS DOS DOIS ESPAÇOS ================")
    print(f"PFS Best Fitness (BRKGA Macro) : {resultado['best_fitness_pfs']}")
    print(f"NPFS Best Fitness (SCS Micro)  : {resultado['best_fitness_npfs']}")
    print(f"Avaliações Totais Utilizadas   : {resultado['total_evaluations']}")
    print("=============================================================")