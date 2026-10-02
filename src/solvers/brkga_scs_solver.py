"""
Solver 2P-BRKGA-SCS (Clustering Search + Cross-Space Knowledge Transfer).

Referência:
  - 2P-BRKGA-SCS: BRKGA com Cross-Space Knowledge Transfer via Clustering Search
    e Busca Local Hooke & Jeeves no espaço NPFS.

Mecânica de Clustering
  - Distância de Chebyshev (L_inf) para delimitação no hipercubo [0,1]^D
  - Raio adaptativo com contração (sucesso) e expansão (falha)
  - Cooldown geométrico para evitar desperdício de CFOs em regiões estagnadas
  - Sincronização PFS/NPFS para nunca retornar solução sub-ótima
  - Eliminação de reavaliação de elites copiados entre gerações
"""

import time
import math
import numpy as np
from enum import Enum
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
            best_x = x_novo.copy()
        else:
            step *= step_reduction

    return best_x, best_fit, evals


# ============================================================================
# 3. STATEFUL CLUSTERING SEARCH
#    - Distância de Chebyshev (L_inf)
#    - Raio adaptativo (contração/expansão)
#    - Cooldown geométrico
# ============================================================================

class ClusterState(Enum):
    """Estado de vida de cada cluster."""
    ACTIVE = 1       # Aceita buscas locais quando elegível
    COOLDOWN = 2     # Bloqueado temporariamente para economizar CFOs
    EXHAUSTED = 3    # Região completamente esgotada


class Cluster:
    """
    Cluster individual com memória de estado completa.

    Restaurado do notebook original (notebooks_alunos/BRKGA_SCS.ipynb),
    classe Cluster (linhas 738-806).
    """
    def __init__(
        self,
        cluster_id: int,
        centroid: np.ndarray,
        fitness: float,
        r_min: float = 0.05,
        r_max: float = 0.25,
        base_cooldown: int = 5,
        gamma: float = 1.5,
        max_cooldown: int = 50
    ):
        self.id = cluster_id
        self.centroid = np.copy(centroid)
        self.best_fitness = fitness

        # Raio adaptativo
        self.r_min = r_min
        self.r_max = r_max
        self.radius = (r_min + r_max) / 2.0

        # Memória de estado e Cooldown
        self.state = ClusterState.ACTIVE
        self.stagnation_count = 0
        self.cooldown_ticks = 0
        self.base_cooldown = base_cooldown
        self.gamma = gamma
        self.max_cooldown = max_cooldown

        self.membros_count = 1

    def chebyshev_distance(self, individual: np.ndarray) -> float:
        """Distância L_infinito (Chebyshev) entre o indivíduo e o centroide."""
        return float(np.max(np.abs(self.centroid - individual)))

    def is_inside(self, individual: np.ndarray) -> bool:
        """Verifica se o indivíduo está dentro da vizinhança hipercúbica."""
        return self.chebyshev_distance(individual) <= self.radius

    def tick_cooldown(self):
        """Decrementa o temporizador de cooldown a cada geração."""
        if self.state == ClusterState.COOLDOWN:
            self.cooldown_ticks -= 1
            if self.cooldown_ticks <= 0:
                self.state = ClusterState.ACTIVE

    def register_local_search_result(
        self, improved: bool, new_centroid: np.ndarray, new_fitness: float
    ):
        """
        Atualiza a memória do cluster com base no resultado da busca local.

        Se melhorou: reseta estagnação, contrai o raio para focar.
        Se falhou: incrementa estagnação, expande raio, entra em cooldown geométrico.
        """
        if improved and new_fitness < self.best_fitness:
            # Sucesso: foco fino na região promissora
            self.centroid = np.copy(new_centroid)
            self.best_fitness = new_fitness
            self.stagnation_count = 0
            self.state = ClusterState.ACTIVE
            self.cooldown_ticks = 0

            # Contração adaptativa do raio
            self.radius = max(self.r_min, self.radius * 0.9)
        else:
            # Falha: região possivelmente esgotada
            self.stagnation_count += 1

            # Expansão adaptativa para tentar vizinhos mais distantes
            self.radius = min(self.r_max, self.radius * 1.1)

            # Cooldown geométrico: base * gamma^estagnacao
            cd = int(self.base_cooldown * (self.gamma ** self.stagnation_count))
            self.cooldown_ticks = min(self.max_cooldown, cd)
            self.state = ClusterState.COOLDOWN


class StatefulClusteringSearch:
    """
    Gerenciador de clusters com memória de estado.

    Restaurado do notebook original (notebooks_alunos/BRKGA_SCS.ipynb),
    classe StatefulClusteringSearch (linhas 809-894).
    """
    def __init__(
        self,
        num_clusters_max: int = 5,
        r_min: float = 0.05,
        r_max: float = 0.25,
        base_cooldown: int = 5,
        gamma: float = 1.5,
        max_stagnation: int = 10
    ):
        self.num_clusters_max = num_clusters_max
        self.r_min = r_min
        self.r_max = r_max
        self.base_cooldown = base_cooldown
        self.gamma = gamma
        self.max_stagnation = max_stagnation
        self.clusters: List[Cluster] = []
        self._next_id = 0

    def update_states(self):
        """Avança o temporizador de cooldown de todos os clusters."""
        for c in self.clusters:
            c.tick_cooldown()

    def process_individual(
        self,
        individual: np.ndarray,
        fitness: float,
        local_search_fn: Callable[[np.ndarray, float], Tuple[np.ndarray, float, int]]
    ) -> Tuple[np.ndarray, float, bool, int]:
        """
        Processa um indivíduo promissor na estrutura de clustering:
        1. Encontra o cluster mais próximo via Distância de Chebyshev.
        2. Se dentro do raio, absorve e decide se faz Busca Local.
        3. Se fora, cria novo cluster se houver vaga.

        Retorna: (solução, fitness, executou_busca_local, evals_gastos)
        """
        if not self.clusters:
            # Cria o primeiro cluster
            new_cluster = Cluster(
                self._next_id, individual, fitness,
                self.r_min, self.r_max, self.base_cooldown, self.gamma
            )
            self._next_id += 1
            self.clusters.append(new_cluster)

            # Dispara busca local no primeiro cluster
            ls_sol, ls_fit, ls_evals = local_search_fn(individual, fitness)
            improved = ls_fit < fitness
            new_cluster.register_local_search_result(improved, ls_sol, ls_fit)
            return ls_sol, ls_fit, True, ls_evals

        # Encontra o cluster mais próximo usando Chebyshev
        distances = [c.chebyshev_distance(individual) for c in self.clusters]
        min_idx = int(np.argmin(distances))
        closest_cluster = self.clusters[min_idx]
        min_dist = distances[min_idx]

        # Verifica se pertence ao cluster mais próximo
        if min_dist <= closest_cluster.radius:
            closest_cluster.membros_count += 1

            # Só executa Busca Local se o cluster estiver ATIVO
            if closest_cluster.state == ClusterState.ACTIVE:
                ls_sol, ls_fit, ls_evals = local_search_fn(individual, fitness)
                improved = ls_fit < fitness or ls_fit < closest_cluster.best_fitness
                closest_cluster.register_local_search_result(improved, ls_sol, ls_fit)
                return ls_sol, ls_fit, True, ls_evals
            else:
                # Cluster em COOLDOWN: economiza avaliações
                return individual, fitness, False, 0
        else:
            # Fora da vizinhança: cria novo cluster se não atingiu o limite
            if len(self.clusters) < self.num_clusters_max:
                new_cluster = Cluster(
                    self._next_id, individual, fitness,
                    self.r_min, self.r_max, self.base_cooldown, self.gamma
                )
                self._next_id += 1
                self.clusters.append(new_cluster)

                # Busca local no centroide da nova região
                ls_sol, ls_fit, ls_evals = local_search_fn(individual, fitness)
                improved = ls_fit < fitness
                new_cluster.register_local_search_result(improved, ls_sol, ls_fit)
                return ls_sol, ls_fit, True, ls_evals

        return individual, fitness, False, 0


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
    # Parâmetros do Clustering (novos, restaurados do original)
    num_clusters_max: int = 5,
    r_min: float = 0.05,
    r_max: float = 0.25,
    base_cooldown: int = 5,
    gamma: float = 1.5,
    z_limiar: float = 1.5,
    votos_limiar: int = 5,
    # Parâmetros do H&J
    step_init: float = 0.1,
    step_reduction: float = 0.5,
    min_step: float = 0.001,
    # Orçamento e semente
    max_evals_total: int = 20000,
    seed: int = 42,
    # Parâmetros legados (mantidos para compatibilidade, mas ignorados)
    raio_cluster: float = 0.5,
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
      num_clusters_max: número máximo de clusters simultâneos.
      r_min, r_max: limites do raio adaptativo.
      base_cooldown, gamma: parâmetros do cooldown geométrico.
      z_limiar: threshold Z-Score para ativar SCS.
      votos_limiar: número de votos para ativar SCS.
      step_init, step_reduction, min_step: parâmetros do H&J.
      max_evals_total: orçamento estrito de avaliações (CFOs).
      seed: semente aleatória.

    Retorna:
      dict com:
        - best_global_pfs: melhor solução no espaço PFS (n,)
        - best_fitness_pfs: makespan da melhor PFS
        - best_global_npfs: melhor solução no espaço NPFS (n*m,)
        - best_fitness_npfs: makespan da melhor NPFS
        - total_evaluations: total de avaliações usadas
        - lower_bound: LB da instância
        - history: lista de (evals, best_fitness_npfs) para convergência
    """
    np.random.seed(seed)
    evals_count = 0
    history_npfs = []  # rastreamento de convergência

    # --- Wrappers de Avaliação (com contagem estrita de CFOs) ---
    def avaliar_pfs(sol: np.ndarray) -> float:
        nonlocal evals_count
        evals_count += 1
        return decodificar_two_phase(sol, n, m, p_matrix, due_dates, modo="PFS", obj="SPAM")

    def avaliar_npfs(sol: np.ndarray) -> float:
        nonlocal evals_count
        evals_count += 1
        return decodificar_two_phase(sol, n, m, p_matrix, due_dates, modo="NPFS", obj="SPAM")

    # --- Duplo Best Rastreamento ---
    best_global_pfs = None
    best_fitness_pfs = float('inf')

    best_global_npfs = None
    best_fitness_npfs = float('inf')

    # --- Inicialização BRKGA (Espaço Reduzido D = N) ---
    dim_pfs = n
    n_elite = max(1, int(tam_pop * pe))
    n_mutants = max(1, int(tam_pop * pm))
    n_crossover = tam_pop - n_elite - n_mutants

    populacao = np.random.uniform(0.0, 1.0, size=(tam_pop, dim_pfs))
    fitness_pfs = np.array([avaliar_pfs(ind) for ind in populacao])

    # --- Inicialização do Stateful Clustering Search (restaurado) ---
    scs = StatefulClusteringSearch(
        num_clusters_max=num_clusters_max,
        r_min=r_min,
        r_max=r_max,
        base_cooldown=base_cooldown,
        gamma=gamma,
    )

    # --- Função de Busca Local para o SCS (Cross-Space: PFS -> NPFS) ---
    def local_search_fn(
        centro_pfs: np.ndarray, fitness_centro: float
    ) -> Tuple[np.ndarray, float, int]:
        """
        Projeta o centro PFS para NPFS via np.tile e refina com Hooke & Jeeves.
        Retorna: (solução_npfs, fitness_npfs, evals_gastos)
        """
        cromossomo_expandido = np.tile(centro_pfs, m)
        budget_restante = max_evals_total - evals_count
        if budget_restante <= 0:
            return cromossomo_expandido, float('inf'), 0

        sol_refinada, fit_refinado, evals_hj = hooke_jeeves_local_search(
            cromossomo_expandido,
            avaliar_npfs,
            step_init=step_init,
            step_reduction=step_reduction,
            min_step=min_step,
            max_evals=min(500, budget_restante)
        )
        return sol_refinada, fit_refinado, evals_hj

    # --- Loop Evolutivo (parada por orçamento de CFOs) ---
    gen = 0
    while evals_count < max_evals_total:
        if best_fitness_npfs <= lb:
            break

        # Ordenação por Fitness PFS
        idx_ordenados = np.argsort(fitness_pfs)
        populacao = populacao[idx_ordenados]
        fitness_pfs = fitness_pfs[idx_ordenados]

        # Rastreamento Best PFS
        if fitness_pfs[0] < best_fitness_pfs:
            best_fitness_pfs = fitness_pfs[0]
            best_global_pfs = populacao[0].copy()

            # *** CORREÇÃO: Sincronização PFS -> NPFS ***
            # Se o PFS encontrou algo melhor que o NPFS atual,
            # projeta e atualiza para nunca retornar solução sub-ótima
            if best_fitness_pfs < best_fitness_npfs:
                best_fitness_npfs = best_fitness_pfs
                best_global_npfs = np.tile(best_global_pfs, m)

        # Early Stopping
        if best_fitness_pfs <= lb:
            break

        # --- Avança o cooldown de todos os clusters ---
        scs.update_states()

        # --- Módulo de Clustering: alimenta o SCS com os elites ---
        media_pop = np.mean(fitness_pfs)
        std_pop = np.std(fitness_pfs) + 1e-9

        for i in range(n_elite):
            if evals_count >= max_evals_total or best_fitness_npfs <= lb:
                break

            elite_ind = populacao[i]
            elite_fit = fitness_pfs[i]

            # Critério de qualidade: z-score do elite vs população
            z_score = (elite_fit - media_pop) / std_pop

            # Só alimenta o clustering se o elite é significativamente bom
            # ou se já acumulou muitas absorções (votos_limiar)
            if z_score <= -z_limiar:
                sol, fit, executou_bl, evals_bl = scs.process_individual(
                    elite_ind, elite_fit, local_search_fn
                )

                # Atualiza o NPFS Best se a busca local melhorou
                if executou_bl and fit < best_fitness_npfs:
                    best_fitness_npfs = fit
                    best_global_npfs = sol.copy()

        if best_fitness_npfs <= lb:
            break

        # --- Evolução da População (BRKGA Crossover & Mutantes) ---
        nova_pop = np.zeros_like(populacao)
        novo_fitness = np.full(tam_pop, float('inf'))

        # *** CORREÇÃO: Elites copiados SEM reavaliação ***
        nova_pop[:n_elite] = populacao[:n_elite]
        novo_fitness[:n_elite] = fitness_pfs[:n_elite]

        # Mutantes (avaliados)
        nova_pop[n_elite:n_elite + n_mutants] = np.random.uniform(
            0.0, 1.0, size=(n_mutants, dim_pfs)
        )
        for i in range(n_mutants):
            if evals_count >= max_evals_total:
                break
            novo_fitness[n_elite + i] = avaliar_pfs(nova_pop[n_elite + i])

        # Crossover (avaliados)
        for i in range(n_crossover):
            if evals_count >= max_evals_total:
                break
            pai_e = populacao[np.random.randint(0, n_elite)]
            pai_ne = populacao[np.random.randint(n_elite, tam_pop)]
            mascara = np.random.rand(dim_pfs) < rho
            filho = np.where(mascara, pai_e, pai_ne)
            nova_pop[n_elite + n_mutants + i] = filho
            novo_fitness[n_elite + n_mutants + i] = avaliar_pfs(filho)

        populacao = nova_pop
        fitness_pfs = novo_fitness

        # Registra histórico de convergência (baseado em CFOs)
        history_npfs.append((evals_count, best_fitness_npfs))
        gen += 1

    return {
        "best_global_pfs": best_global_pfs,
        "best_fitness_pfs": best_fitness_pfs,
        "best_global_npfs": best_global_npfs,
        "best_fitness_npfs": best_fitness_npfs,
        "total_evaluations": evals_count,
        "lower_bound": lb,
        "history": history_npfs,
    }