"""
Adapter layer — interface padronizada entre o orquestrador e os solvers.

"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Callable


# ============================================================================
# 1. SolverResult — estrutura de saída padronizada
# ============================================================================

@dataclass
class SolverResult:
    """Resultado padronizado de uma execução de qualquer solver."""
    solver_name: str                # "ACO" | "Neuro-BOA" | "BRKGA" | "PBIL-Fuzzy"
    version: str                    # "fixo_default" | "fixo_optuna" | "fuzzy" | "adaptativo" | etc.
    mode: str                       # "PFS" | "NPFS"
    best_perm: Optional[np.ndarray] # solução encontrada (PFS: shape (n,); NPFS: shape (m, n))
    best_cost: float                # makespan (calculado pela engine oficial)
    tardiness: float                # tardiness (calculado pela engine oficial sobre a mesma perm)
    history: list            # curva de convergência (custo × iteração/geração)
    time_seconds: float             # tempo total de execução
    n_evaluations: int              # número de chamadas à função objetivo
    params: dict                    # hiperparâmetros usados nesta execução
    lower_bound: float              # lower bound de Potts para referência
    engine_cfo_count: int = 0       # total de chamadas à engine durante avaliação

    def gap_percent(self) -> float:
        """Gap percentual em relação ao lower bound."""
        if self.lower_bound <= 0:
            return float("inf")
        return (self.best_cost - self.lower_bound) / self.lower_bound * 100.0


# ============================================================================
# 2. Funções auxiliares comuns
# ============================================================================

def _avaliar_na_engine(engine, perm, mode="NPFS"):
    """
    Avalia uma permutação na engine oficial e devolve (makespan, tardiness, cfo_count).
    """
    cfo_antes = engine.cfo_count

    if mode == "PFS":
        # PFS: perm é array 1D (n_jobs,)
        makespan = engine.evaluate_with_permutation(perm, objective="Makespan")
        tardiness = engine.evaluate_with_permutation(perm, objective="Tardiness")
    else:
        # NPFS: perm é array 2D (n_machines, n_jobs)
        # Criamos keys artificialmente para usar evaluate com mode NPFS
        n = engine.n_jobs
        m = engine.n_machines
        keys = np.zeros(n * m)
        # Converte permutação para keys: argsort inverso
        for maq in range(m):
            inv = np.argsort(perm[maq])
            keys[maq * n:(maq + 1) * n] = inv.astype(float) / n
        makespan = engine.evaluate(keys, mode="NPFS", objective="Makespan")

        # recalcula tardiness: avaliamos a mesma permutação em PFS para tardiness
        # (tardiness no NPFS é calculado pela última máquina)
        # Usamos evaluate com as mesmas keys para tardiness
        tardiness = engine.evaluate(keys, mode="NPFS", objective="Tardiness")

    cfo_count = engine.cfo_count - cfo_antes
    return makespan, tardiness, cfo_count

# ============================================================================
# 7. Adapter: BRKGA-SCS-VM 
# ============================================================================

def rodar_brkga_scs_vm(
    engine,
    seed=1,
    objective="Makespan",
    max_evals=10000,
    t_param: Optional[float] = None,
    tam_pop=100,
    pe=0.2,
    pm=0.15,
    rho=0.7,
    # Parâmetros do Stateful Clustering Search (restaurados)
    num_clusters_max=5, #pq 5? E se eu testar com mais?
    r_min=0.05,
    r_max=0.25,
    base_cooldown=5,
    gamma=1.5,
    z_limiar=1.5,
    votos_limiar=5,
    # Parâmetros do Hooke & Jeeves
    step_init=0.1,
    step_reduction=0.5,
    min_step=0.001,
    verbose=False,
) -> SolverResult:
    """
    Executa o 2P-BRKGA-SCS (Clustering Search + Cross-Space Knowledge Transfer)
    e retorna SolverResult padronizado.
    """
    from src.solvers.brkga_scs_vm_solver import run_brkga_scs_vm
    from src.flowshop_engine import FlowShopEngine

    n = engine.n_jobs
    m = engine.n_machines

    params = {
        "max_evals": max_evals,
        "t_param": t_param,
        "tam_pop": tam_pop,
        "pe": pe,
        "pm": pm,
        "rho": rho,
        "num_clusters_max": num_clusters_max,
        "r_min": r_min,
        "r_max": r_max,
        "base_cooldown": base_cooldown,
        "gamma": gamma,
        "z_limiar": z_limiar,
        "votos_limiar": votos_limiar,
    }

    # Lower bound
    lb = FlowShopEngine.calcular_lower_bound_potts(
        engine.n_jobs, engine.n_machines,
        engine.proc_times, engine.due_dates
    )

    t0 = time.time()
    result = run_brkga_scs_vm(
        engine=engine,
        n=n, m=m,
        lb=lb,
        objective=objective,
        tam_pop=tam_pop,
        pe=pe,
        pm=pm,
        rho=rho,
        num_clusters_max=num_clusters_max,
        r_min=r_min,
        r_max=r_max,
        base_cooldown=base_cooldown,
        gamma=gamma,
        z_limiar=z_limiar,
        votos_limiar=votos_limiar,
        step_init=step_init,
        step_reduction=step_reduction,
        min_step=min_step,
        max_evals_total=max_evals,
        t_param=t_param,
        seed=seed,
    )
    elapsed = time.time() - t0

    # Converte NPFS keys para permutação
    best_perm = None
    if result["best_global_npfs"] is not None:
        keys = result["best_global_npfs"]
        if keys.ndim == 1:
            keys = keys.reshape((m, n))
        best_perm = np.zeros((m, n), dtype=int)
        for maq in range(m):
            best_perm[maq] = np.argsort(keys[maq])
    elif result["best_global_pfs"] is not None:
        # Fallback: usa solução PFS convertida para NPFS (mesma ordem em todas as máq.)
        keys_pfs = result["best_global_pfs"]
        best_perm = np.zeros((m, n), dtype=int)
        for maq in range(m):
            best_perm[maq] = np.argsort(keys_pfs)

    # Avaliação com engine oficial
    if best_perm is not None:
        makespan, tardiness, cfo = _avaliar_na_engine(engine, best_perm, mode="NPFS")
    else:
        makespan = float("inf")
        tardiness = float("inf")
        cfo = 0

    # Histórico de convergência
    history = result.get("history", [])
    if not history:
        history = [makespan]

    return SolverResult(
        solver_name="BRKGA-SCS",
        version="vm",
        mode="NPFS",
        best_perm=best_perm,
        best_cost=makespan,
        tardiness=tardiness,
        history=history,
        time_seconds=elapsed,
        n_evaluations=result["total_evaluations"],
        params=params,
        lower_bound=lb,
        engine_cfo_count=cfo,
    )


def rodar_brkga_scs_vr(
    engine,
    seed=1,
    objective="Makespan",
    max_evals=10000,
    t_param: Optional[float] = None,
    tam_pop=100,
    pe=0.15,
    pm=0.25,
    rho=0.85,
    raio_cluster=0.5,
    z_limiar=1.645,
    votos_limiar=5,
    step_init=0.1,
    step_reduction=0.5,
    min_step=0.001,
    max_hj_evals=600,
    max_generations=129999,
    verbose=False,
) -> SolverResult:
    """
    Executa o 2P-BRKGA-SCS Versão Revisada (brkga_scs_vr_solver)
    e retorna SolverResult padronizado.
    """
    from src.solvers.brkga_scs_vr_solver import run_brkga_scs_hybrid, Config
    from src.flowshop_engine import FlowShopEngine

    n = engine.n_jobs
    m = engine.n_machines

    cfg = Config(
        tam_pop=tam_pop,
        pe=pe,
        pm=pm,
        rho=rho,
        raio_cluster=raio_cluster,
        z_limiar=z_limiar,
        votos_limiar=votos_limiar,
        step_init=step_init,
        step_reduction=step_reduction,
        min_step=min_step,
        max_hj_evals=max_hj_evals,
        max_generations=max_generations,
        max_evals_total=max_evals,
        t_param=t_param,
        obj=objective,
    )

    params = {
        "max_evals": max_evals,
        "t_param": t_param,
        "tam_pop": tam_pop,
        "pe": pe,
        "pm": pm,
        "rho": rho,
        "raio_cluster": raio_cluster,
        "z_limiar": z_limiar,
        "votos_limiar": votos_limiar,
        "step_init": step_init,
        "step_reduction": step_reduction,
        "min_step": min_step,
        "max_hj_evals": max_hj_evals,
        "max_generations": max_generations,
    }

    lb = FlowShopEngine.calcular_lower_bound_potts(
        engine.n_jobs, engine.n_machines,
        engine.proc_times, engine.due_dates
    )

    t0 = time.time()
    result = run_brkga_scs_hybrid(
        engine=engine,
        n=n, m=m,
        lb=lb,
        config=cfg,
        seed=seed,
    )
    elapsed = time.time() - t0

    best_perm = None
    if result["best_global_npfs"] is not None:
        keys = result["best_global_npfs"]
        if keys.ndim == 1:
            keys = keys.reshape((m, n))
        best_perm = np.zeros((m, n), dtype=int)
        for maq in range(m):
            best_perm[maq] = np.argsort(keys[maq])
    elif result["best_global_pfs"] is not None:
        keys_pfs = result["best_global_pfs"]
        best_perm = np.zeros((m, n), dtype=int)
        for maq in range(m):
            best_perm[maq] = np.argsort(keys_pfs)

    if best_perm is not None:
        makespan, tardiness, cfo = _avaliar_na_engine(engine, best_perm, mode="NPFS")
    else:
        makespan = float("inf")
        tardiness = float("inf")
        cfo = 0

    history = result.get("history", [])
    if not history:
        history = [makespan]

    return SolverResult(
        solver_name="BRKGA-SCS",
        version="vr",
        mode="NPFS",
        best_perm=best_perm,
        best_cost=makespan,
        tardiness=tardiness,
        history=history,
        time_seconds=elapsed,
        n_evaluations=result["total_evaluations"],
        params=params,
        lower_bound=lb,
        engine_cfo_count=cfo,
    )


def rodar_brkga_scs_img(
    engine,
    seed=1,
    objective="Makespan",
    max_evals=10000,
    t_param: Optional[float] = None,
    tam_pop=125,
    pe=0.15,
    pm=0.15,
    rho=0.6,
    raio_cluster=0.15,
    z_limiar=1.645,
    votos_limiar=5,
    max_ls_evals=500,
    max_generations=129999,
    verbose=False,
) -> SolverResult:
    """
    Executa o 2P-BRKGA-SCS com Busca Local no Espaço de Imagem (r-NEH)
    e retorna SolverResult padronizado.
    """
    from src.solvers.brkga_scs_img_solver import run_brkga_scs_img, Config
    from src.flowshop_engine import FlowShopEngine

    n = engine.n_jobs
    m = engine.n_machines

    cfg = Config(
        tam_pop=tam_pop,
        pe=pe,
        pm=pm,
        rho=rho,
        raio_cluster=raio_cluster,
        z_limiar=z_limiar,
        votos_limiar=votos_limiar,
        max_ls_evals=max_ls_evals,
        max_generations=max_generations,
        max_evals_total=max_evals,
        t_param=t_param,
        obj=objective,
    )

    params = {
        "max_evals": max_evals,
        "t_param": t_param,
        "tam_pop": tam_pop,
        "pe": pe,
        "pm": pm,
        "rho": rho,
        "raio_cluster": raio_cluster,
        "z_limiar": z_limiar,
        "votos_limiar": votos_limiar,
        "max_ls_evals": max_ls_evals,
        "max_generations": max_generations,
    }

    lb = FlowShopEngine.calcular_lower_bound_potts(
        engine.n_jobs, engine.n_machines,
        engine.proc_times, engine.due_dates
    )

    t0 = time.time()
    result = run_brkga_scs_img(
        engine=engine,
        n=n, m=m,
        lb=lb,
        config=cfg,
        seed=seed,
    )
    elapsed = time.time() - t0

    best_perm = None
    if result["best_global_npfs"] is not None:
        keys = result["best_global_npfs"]
        if keys.ndim == 1:
            keys = keys.reshape((m, n))
        best_perm = np.zeros((m, n), dtype=int)
        for maq in range(m):
            best_perm[maq] = np.argsort(keys[maq])
    elif result["best_global_pfs"] is not None:
        keys_pfs = result["best_global_pfs"]
        best_perm = np.zeros((m, n), dtype=int)
        for maq in range(m):
            best_perm[maq] = np.argsort(keys_pfs)

    if best_perm is not None:
        makespan, tardiness, cfo = _avaliar_na_engine(engine, best_perm, mode="NPFS")
    else:
        makespan = float("inf")
        tardiness = float("inf")
        cfo = 0

    history = result.get("history", result.get("history_npfs", []))
    if not history:
        history = [makespan]

    return SolverResult(
        solver_name="BRKGA-SCS",
        version="img",
        mode="NPFS",
        best_perm=best_perm,
        best_cost=makespan,
        tardiness=tardiness,
        history=history,
        time_seconds=elapsed,
        n_evaluations=result["total_evaluations"],
        params=params,
        lower_bound=lb,
        engine_cfo_count=cfo,
    )


# ============================================================================
# 8. Função genérica: rodar_experimento
# ============================================================================

SOLVER_REGISTRY = {
    
    "BRKGA_SCS": rodar_brkga_scs_vm,
    "BRKGA_SCS_VR": rodar_brkga_scs_vr,
    "BRKGA_SCS_IMG": rodar_brkga_scs_img,
    
}


def rodar_experimento(
    solver_name: str,
    engine,
    seed: int = 1,
    objective: str = "Makespan",
    params: Optional[dict] = None,
    verbose: bool = False,
) -> SolverResult:
    """
    Função genérica: roda qualquer solver registrado.

    Args:
        solver_name: Nome do solver (chave do SOLVER_REGISTRY)
        engine: FlowShopEngine oficial
        seed: Semente aleatória
        objective: "Makespan" | "Tardiness"
        params: Dict com hiperparâmetros específicos do solver
        verbose: Se True, imprime progresso

    Retorna:
        SolverResult padronizado
    """
    if solver_name not in SOLVER_REGISTRY:
        raise ValueError(
            f"Solver '{solver_name}' não encontrado. "
            f"Disponíveis: {list(SOLVER_REGISTRY.keys())}"
        )

    solver_fn = SOLVER_REGISTRY[solver_name]
    kwargs = params or {}

    return solver_fn(
        engine=engine,
        seed=seed,
        objective=objective,
        verbose=verbose,
        **kwargs,
    )

def listar_solvers_disponiveis():
    """Retorna lista de nomes de solvers disponíveis."""
    return list(SOLVER_REGISTRY.keys())