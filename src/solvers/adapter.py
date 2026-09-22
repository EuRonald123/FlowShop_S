"""
Adapter layer — interface padronizada entre o orquestrador e os solvers.

Cada solver (ACO, BRKGA, Neuro-BOA, PBIL-Fuzzy) é encapsulado por uma
função adapter que:
  1. Converte parâmetros para o formato esperado pelo solver
  2. Executa o solver
  3. Avalia a solução encontrada usando a FlowShopEngine oficial
  4. Mede tempo de execução
  5. Retorna um SolverResult padronizado

Uso no orquestrador (principal.ipynb):
    from src.solvers.adapter import rodar_experimento
    result = rodar_experimento("ACO", engine, seed=42, params={...})
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
    history: List[float]            # curva de convergência (custo × iteração/geração)
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
# 3. Adapter: ACO
# ============================================================================

def rodar_aco(
    engine,
    seed=1,
    objective="Makespan",
    mode="NPFS",
    n_ants=20,
    n_iterations=100,
    alpha=1.0,
    beta=2.0,
    rho=0.1,
    Q=100.0,
    verbose=False,
) -> SolverResult:
    """
    Executa o ACO e retorna SolverResult padronizado.
    """
    from src.solvers.aco_solver import ACOSolver

    params = {
        "n_ants": n_ants,
        "n_iterations": n_iterations,
        "alpha": alpha,
        "beta": beta,
        "rho": rho,
        "Q": Q,
    }

    solver = ACOSolver(
        engine,
        n_ants=n_ants,
        n_iterations=n_iterations,
        alpha=alpha,
        beta=beta,
        rho=rho,
        Q=Q,
        objective=objective,
        seed=seed,
        mode=mode,
    )

    t0 = time.time()
    best_perm, best_cost, history = solver.run(verbose=verbose)
    elapsed = time.time() - t0

    # Avaliação final com engine oficial
    makespan, tardiness, cfo = _avaliar_na_engine(engine, best_perm, mode=mode)

    # Lower bound
    from src.flowshop_engine import FlowShopEngine
    lb = FlowShopEngine.calcular_lower_bound_potts(
        engine.n_jobs, engine.n_machines,
        engine.proc_times, engine.due_dates
    )

    return SolverResult(
        solver_name="ACO",
        version=f"ACO_Optuna" if alpha != 1.0 else "ACO_default",
        mode=mode,
        best_perm=best_perm,
        best_cost=makespan,
        tardiness=tardiness,
        history=history,
        time_seconds=elapsed,
        n_evaluations=solver.n_ants * solver.n_iterations,
        params=params,
        lower_bound=lb,
        engine_cfo_count=cfo,
    )


# ============================================================================
# 4. Adapter: BRKGA
# ============================================================================

def rodar_brkga(
    engine,
    seed=1,
    objective="Makespan",
    version="fixo_default",
    n_gen=100,
    n_elites=20,
    n_offsprings=70,
    n_mutants=10,
    bias=0.7,
    # Parâmetros fuzzy (opcionais)
    frac_elites_min=0.05,
    frac_elites_max=0.40,
    frac_mutantes_min=0.05,
    frac_mutantes_max=0.40,
    frac_offsprings_min=0.20,
    ref_diversidade=None,
    ref_qualidade=None,
    verbose=False,
) -> SolverResult:
    """
    Executa o BRKGA (fixo ou fuzzy) e retorna SolverResult padronizado.
    """
    from src.solvers.brkga_solver import (
        run_brkga_fixo,
        run_brkga_fuzzy,
        analisar_melhor_configuracao,
        construir_referencia_a_partir_da_trajetoria,
    )
    from src.flowshop_engine import FlowShopEngine

    params = {
        "n_gen": n_gen,
        "n_elites": n_elites,
        "n_offsprings": n_offsprings,
        "n_mutants": n_mutants,
        "bias": bias,
    }

    if version == "fuzzy":
        t0 = time.time()
        result, historico = run_brkga_fuzzy(
            engine, n_gen=n_gen, seed=seed,
            n_elites0=n_elites, n_offsprings0=n_offsprings,
            n_mutants0=n_mutants, bias=bias,
            frac_elites_min=frac_elites_min,
            frac_elites_max=frac_elites_max,
            frac_mutantes_min=frac_mutantes_min,
            frac_mutantes_max=frac_mutantes_max,
            frac_offsprings_min=frac_offsprings_min,
            ref_diversidade=ref_diversidade or 0.20,
            ref_qualidade=ref_qualidade or 0.10,
        )
        elapsed = time.time() - t0
        best_X = result["X"]
        best_cost = result["F"]
    else:
        t0 = time.time()
        result, historico = run_brkga_fixo(
            engine, n_gen=n_gen, seed=seed,
            n_elites=n_elites, n_offsprings=n_offsprings,
            n_mutants=n_mutants, bias=bias,
            verbose=verbose,
        )
        elapsed = time.time() - t0
        best_X = result.X
        best_cost = result.F[0]

    # Converte cromossomo (keys) para permutação NPFS
    n = engine.n_jobs
    m = engine.n_machines
    best_perm = np.zeros((m, n), dtype=int)
    for maq in range(m):
        keys_maq = best_X[maq * n:(maq + 1) * n]
        best_perm[maq] = np.argsort(keys_maq)

    # Avaliação com engine oficial
    makespan, tardiness, cfo = _avaliar_na_engine(engine, best_perm, mode="NPFS")

    # Histórico de convergência
    history = [h["best_makespan"] for h in historico]

    # Lower bound
    from src.flowshop_engine import FlowShopEngine
    lb = FlowShopEngine.calcular_lower_bound_potts(
        engine.n_jobs, engine.n_machines,
        engine.proc_times, engine.due_dates
    )

    return SolverResult(
        solver_name="BRKGA",
        version=version,
        mode="NPFS",
        best_perm=best_perm,
        best_cost=makespan,
        tardiness=tardiness,
        history=history,
        time_seconds=elapsed,
        n_evaluations=n_gen * (n_elites + n_offsprings + n_mutants),
        params=params,
        lower_bound=lb,
        engine_cfo_count=cfo,
    )


# ============================================================================
# 5. Adapter: Neuro-BOA
# ============================================================================

def rodar_neuro_boa(
    engine,
    seed=1,
    objective="makespan",
    mode="adaptativo",
    population_size=60,
    generations=80,
    elite_frac=0.15,
    neuro_hidden=128,
    neuro_epochs_per_gen=2,
    neuro_temperature=1.0,
    dropout=0.0,
    bayes_smoothing=1.0,
    quality_fraction=0.75,
    probe_per_generator=4,
    selector_epsilon=0.10,
    selector_hidden=32,
    verbose=False,
) -> SolverResult:
    """
    Executa o Neuro-BOA e retorna SolverResult padronizado.
    Converte a engine oficial para o formato Instance usado pelo solver.
    """
    from src.solvers.neuro_boa_solver import (
        Instance, run_method, ConfigNeuroBOA, decode_npfs,
    )
    from src.flowshop_engine import FlowShopEngine

    # Neuro-BOA espera "makespan" (minusculo), ao contrario da engine que usa "Makespan"
    obj_neuro = objective.lower()

    # Cria uma Instance a partir da engine oficial (formato esperado pelo Neuro-BOA)
    inst = Instance(
        name="orquestrada",
        group="Small",
        n_jobs=engine.n_jobs,
        n_machines=engine.n_machines,
        processing=engine.proc_times.copy(),
        due_dates=engine.due_dates.copy(),
    )

    cfg = ConfigNeuroBOA(
        population_size=population_size,
        generations=generations,
        elite_frac=elite_frac,
        diverse_candidate_frac=0.35,
        diverse_frac=0.20,
        neuro_hidden=neuro_hidden,
        neuro_lr=1e-3,
        neuro_epochs_per_gen=neuro_epochs_per_gen,
        neuro_temperature=neuro_temperature,
        dropout=dropout,
        bayes_smoothing=bayes_smoothing,
        quality_fraction=quality_fraction,
        probe_per_generator=probe_per_generator,
        selector_epsilon=selector_epsilon,
        selector_hidden=selector_hidden,
        selector_lr=1e-3,
    )

    t0 = time.time()
    result = run_method(
        inst, cfg,
        mode=mode,
        seed=seed,
        objective=obj_neuro,
        verbose=verbose,
    )
    elapsed = time.time() - t0

    # Pega o melhor indivíduo e converte para nossa permutação
    best_ind = result["best_individual"]
    if best_ind is not None:
        decoded = decode_npfs(best_ind, inst)
        best_perm = best_ind  # shape (m, n)
    else:
        best_perm = None

    # Avaliação com engine oficial
    if best_perm is not None:
        makespan, tardiness, cfo = _avaliar_na_engine(engine, best_perm, mode="NPFS")
    else:
        makespan = float("inf")
        tardiness = float("inf")
        cfo = 0

    # Histórico de convergência
    df_hist = result.get("history")
    if df_hist is not None and len(df_hist) > 0:
        history = df_hist["best"].tolist()
    else:
        history = []

    # Lower bound
    lb = FlowShopEngine.calcular_lower_bound_potts(
        engine.n_jobs, engine.n_machines,
        engine.proc_times, engine.due_dates
    )

    params = {
        "population_size": population_size,
        "generations": generations,
        "elite_frac": elite_frac,
        "neuro_hidden": neuro_hidden,
        "neuro_epochs_per_gen": neuro_epochs_per_gen,
        "neuro_temperature": neuro_temperature,
        "dropout": dropout,
        "bayes_smoothing": bayes_smoothing,
        "quality_fraction": quality_fraction,
        "selector_epsilon": selector_epsilon,
        "selector_hidden": selector_hidden,
    }

    return SolverResult(
        solver_name="Neuro-BOA",
        version=mode,
        mode="NPFS",
        best_perm=best_perm,
        best_cost=makespan,
        tardiness=tardiness,
        history=history,
        time_seconds=elapsed,
        n_evaluations=generations * population_size,
        params=params,
        lower_bound=lb,
        engine_cfo_count=cfo,
    )


# ============================================================================
# 6. Adapter: PBIL-Fuzzy
# ============================================================================

def rodar_pbil_fuzzy(
    engine,
    seed=1,
    objective="Makespan",
    n_pop=80,
    sigma_amostragem=0.10,
    pct_elite=0.08,
    pct_subelite=0.35,
    delta=0.20,
    alpha_inicial=0.5,
    beta_inicial=0.5,
    max_geracoes=300,
    verbose=False,
) -> SolverResult:
    """
    Executa o PBIL-Fuzzy e retorna SolverResult padronizado.
    """
    from src.solvers.pbil_fuzzy_solver import PBILFuzzySolver
    from src.flowshop_engine import FlowShopEngine

    params = {
        "n_pop": n_pop,
        "sigma_amostragem": sigma_amostragem,
        "pct_elite": pct_elite,
        "pct_subelite": pct_subelite,
        "delta": delta,
        "max_geracoes": max_geracoes,
    }

    solver = PBILFuzzySolver(
        engine,
        seed=seed,
        n_pop=n_pop,
        sigma_amostragem=sigma_amostragem,
        pct_elite=pct_elite,
        pct_subelite=pct_subelite,
        delta=delta,
        alpha_inicial=alpha_inicial,
        beta_inicial=beta_inicial,
        max_geracoes=max_geracoes,
        objective=objective,
    )

    t0 = time.time()
    best_perm, best_cost, history = solver.executar(verbose=verbose)
    elapsed = time.time() - t0

    # Avaliação com engine oficial
    makespan, tardiness, cfo = _avaliar_na_engine(engine, best_perm, mode="NPFS")

    # Lower bound
    from src.flowshop_engine import FlowShopEngine
    lb = FlowShopEngine.calcular_lower_bound_potts(
        engine.n_jobs, engine.n_machines,
        engine.proc_times, engine.due_dates
    )

    return SolverResult(
        solver_name="PBIL-Fuzzy",
        version="default",
        mode="NPFS",
        best_perm=best_perm,
        best_cost=makespan,
        tardiness=tardiness,
        history=history,
        time_seconds=elapsed,
        n_evaluations=max_geracoes * n_pop,
        params=params,
        lower_bound=lb,
        engine_cfo_count=cfo,
    )


# ============================================================================
# 7. Função genérica: rodar_experimento
# ============================================================================

SOLVER_REGISTRY = {
    "ACO": rodar_aco,
    "BRKGA_fixo_default": lambda engine, seed, **kw: rodar_brkga(
        engine, seed, version="fixo_default", **kw
    ),
    "BRKGA_fixo_optuna": lambda engine, seed, **kw: rodar_brkga(
        engine, seed, version="fixo_optuna", **kw
    ),
    "BRKGA_fuzzy": lambda engine, seed, **kw: rodar_brkga(
        engine, seed, version="fuzzy", **kw
    ),
    "Neuro-BOA_adaptativo": lambda engine, seed, **kw: rodar_neuro_boa(
        engine, seed, mode="adaptativo", **kw
    ),
    "Neuro-BOA_boa_puro": lambda engine, seed, **kw: rodar_neuro_boa(
        engine, seed, mode="boa_puro", **kw
    ),
    "Neuro-BOA_neuro_puro": lambda engine, seed, **kw: rodar_neuro_boa(
        engine, seed, mode="neuro_puro", **kw
    ),
    "PBIL-Fuzzy": rodar_pbil_fuzzy,
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