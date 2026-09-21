"""
Solver BRKGA (Biased Random-Key Genetic Algorithm) extraído do aluno_4.ipynb.
Inclui variantes: fixo_default, fixo_optuna, e fuzzy (com controladores Mamdani).
"""

import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.brkga import BRKGA
from pymoo.optimize import minimize
from pymoo.termination import get_termination

# --------------------------------------------------------------------------- #
# 1. Problema NPFS no pymoo
# --------------------------------------------------------------------------- #

class NPFSProblem(ElementwiseProblem):
    """
    Wrapper pymoo para o problema NPFS (Non-Permutation Flow Shop).
    n_var = n_jobs * n_machines (um bloco de n_jobs chaves por máquina).
    """

    def __init__(self, engine: "FlowShopEngine"):
        self.engine = engine
        n_var = engine.n_jobs * engine.n_machines
        super().__init__(n_var=n_var, n_obj=1, n_constr=0, xl=0.0, xu=1.0)

    def _evaluate(self, x, out, *args, **kwargs):
        makespan = self.engine.evaluate(x, mode="NPFS", objective="Makespan")
        out["F"] = np.array([makespan])


# --------------------------------------------------------------------------- #
# 2. BRKGA fixo
# --------------------------------------------------------------------------- #

def run_brkga_fixo(engine, n_gen=200, seed=1,
                   n_elites=20, n_offsprings=70, n_mutants=10,
                   bias=0.7, verbose=False):
    """
    Executa o BRKGA com parâmetros fixos no modo NPFS.

    Retorna:
        result       -> objeto Result do pymoo (result.X, result.F)
        historico    -> lista de dicts por geração
    """
    problem = NPFSProblem(engine)

    algorithm = BRKGA(
        n_elites=n_elites,
        n_offsprings=n_offsprings,
        n_mutants=n_mutants,
        bias=bias,
        eliminate_duplicates=True,
    )

    termination = get_termination("n_gen", n_gen)

    result = minimize(
        problem,
        algorithm,
        termination,
        seed=seed,
        save_history=True,
        verbose=verbose,
    )

    historico = []
    for g, algo_estado in enumerate(result.history):
        pop = algo_estado.pop
        F = pop.get("F").flatten()
        historico.append({
            "gen": g,
            "best_makespan": float(F.min()),
            "pop_X": pop.get("X").copy(),
        })

    return result, historico


# --------------------------------------------------------------------------- #
# 3. Métricas: diversidade, qualidade
# --------------------------------------------------------------------------- #

from scipy.spatial.distance import pdist


def calcular_diversidade(pop_X):
    """
    pop_X: array (pop_size, n_var)
    Retorna diversidade média entre todos os pares.
    """
    if pop_X.shape[0] < 2:
        return 0.0
    distancias_soma_abs = pdist(pop_X, metric='cityblock')
    n_var = pop_X.shape[1]
    distancias_media_abs = distancias_soma_abs / n_var
    return float(distancias_media_abs.mean())


def calcular_qualidade(best_makespan_gen, best_makespan_gen0):
    """qualidade(g) = (Cmax_best(0) - Cmax_best(g)) / Cmax_best(0)"""
    if best_makespan_gen0 == 0:
        return 0.0
    return float((best_makespan_gen0 - best_makespan_gen) / best_makespan_gen0)


def _resolver_referencia(referencia, gen):
    """Aceita um valor fixo (int/float) ou uma função gen -> valor."""
    if callable(referencia):
        return float(referencia(gen))
    return float(referencia)


def computar_metricas(historico, ref_diversidade=0.20, ref_qualidade=0.10):
    """
    Recebe o histórico de run_brkga_fixo e devolve lista de dicts com métricas.
    """
    metricas = []
    best_makespan_gen0 = historico[0]["best_makespan"]

    erro_diversidade_anterior = None
    erro_qualidade_anterior = None

    for entrada in historico:
        gen = entrada["gen"]
        best_makespan = entrada["best_makespan"]
        pop_X = entrada["pop_X"]

        diversidade = calcular_diversidade(pop_X)
        qualidade = calcular_qualidade(best_makespan, best_makespan_gen0)

        ref_div_g = _resolver_referencia(ref_diversidade, gen)
        ref_qual_g = _resolver_referencia(ref_qualidade, gen)

        erro_diversidade = ref_div_g - diversidade
        erro_qualidade = ref_qual_g - qualidade

        variacao_erro_diversidade = (
            0.0 if erro_diversidade_anterior is None
            else erro_diversidade - erro_diversidade_anterior
        )
        variacao_erro_qualidade = (
            0.0 if erro_qualidade_anterior is None
            else erro_qualidade - erro_qualidade_anterior
        )

        metricas.append({
            "gen": gen,
            "best_makespan": best_makespan,
            "diversidade": diversidade,
            "qualidade": qualidade,
            "erro_diversidade": erro_diversidade,
            "variacao_erro_diversidade": variacao_erro_diversidade,
            "erro_qualidade": erro_qualidade,
            "variacao_erro_qualidade": variacao_erro_qualidade,
            "ref_diversidade": ref_div_g,
            "ref_qualidade": ref_qual_g,
        })

        erro_diversidade_anterior = erro_diversidade
        erro_qualidade_anterior = erro_qualidade

    return metricas


# --------------------------------------------------------------------------- #
# 4. Controladores Fuzzy (Mamdani, erro + Δerro)
# --------------------------------------------------------------------------- #

import skfuzzy as fuzz
from skfuzzy import control as ctrl

_CATEGORIAS = ['NB', 'NS', 'Z', 'PS', 'PB']
_NUM = {'NB': -2, 'NS': -1, 'Z': 0, 'PS': 1, 'PB': 2}
_CAT = {v: k for k, v in _NUM.items()}
_PESO_DELTA = 0.4


def _saida_categoria(e_cat, de_cat):
    score = _NUM[e_cat] + _PESO_DELTA * _NUM[de_cat]
    score = max(-2, min(2, round(score)))
    return _CAT[score]


_TABELA_REGRAS = {
    (e_cat, de_cat): _saida_categoria(e_cat, de_cat)
    for e_cat in _CATEGORIAS
    for de_cat in _CATEGORIAS
}


def _adicionar_termos_5niveis(variavel):
    lo, hi = variavel.universe[0], variavel.universe[-1]
    meio = (lo + hi) / 2
    q1 = lo + (hi - lo) * 0.25
    q3 = lo + (hi - lo) * 0.75
    variavel['NB'] = fuzz.trimf(variavel.universe, [lo, lo, meio])
    variavel['NS'] = fuzz.trimf(variavel.universe, [lo, q1, meio])
    variavel['Z'] = fuzz.trimf(variavel.universe, [q1, meio, q3])
    variavel['PS'] = fuzz.trimf(variavel.universe, [meio, q3, hi])
    variavel['PB'] = fuzz.trimf(variavel.universe, [meio, hi, hi])


class ControladorFuzzyPD:
    """
    Controlador fuzzy genérico erro + variação do erro -> saída.
    """

    def __init__(self, nome, erro_range, delta_erro_range, saida_range, n_pontos=201):
        self.nome = nome
        self._erro_range = erro_range
        self._delta_erro_range = delta_erro_range
        self._saida_range = saida_range

        e_universe = np.linspace(erro_range[0], erro_range[1], n_pontos)
        de_universe = np.linspace(delta_erro_range[0], delta_erro_range[1], n_pontos)
        s_universe = np.linspace(saida_range[0], saida_range[1], n_pontos)

        self.erro = ctrl.Antecedent(e_universe, 'erro')
        self.delta_erro = ctrl.Antecedent(de_universe, 'delta_erro')
        self.saida = ctrl.Consequent(s_universe, 'saida')

        for variavel in (self.erro, self.delta_erro, self.saida):
            _adicionar_termos_5niveis(variavel)

        regras = [
            ctrl.Rule(self.erro[e_cat] & self.delta_erro[de_cat], self.saida[s_cat])
            for (e_cat, de_cat), s_cat in _TABELA_REGRAS.items()
        ]

        self.sistema = ctrl.ControlSystem(regras)
        self.sim = ctrl.ControlSystemSimulation(self.sistema)

    def calcular(self, erro_valor, delta_erro_valor):
        e = float(np.clip(erro_valor, self._erro_range[0], self._erro_range[1]))
        de = float(np.clip(delta_erro_valor, self._delta_erro_range[0], self._delta_erro_range[1]))
        self.sim.input['erro'] = e
        self.sim.input['delta_erro'] = de
        self.sim.compute()
        return float(self.sim.output['saida'])


def criar_controlador_diversidade(
    erro_range=(-0.3, 0.3),
    delta_erro_range=(-0.15, 0.15),
    saida_range=(-0.05, 0.05),
):
    """Saída: ajuste na proporção de mutantes."""
    return ControladorFuzzyPD("diversidade", erro_range, delta_erro_range, saida_range)


def criar_controlador_qualidade(
    erro_range=(-0.3, 0.3),
    delta_erro_range=(-0.15, 0.15),
    saida_range=(-0.05, 0.05),
):
    """Saída: ajuste na proporção de elites."""
    return ControladorFuzzyPD("qualidade", erro_range, delta_erro_range, saida_range)


# --------------------------------------------------------------------------- #
# 5. BRKGA adaptativo (fuzzy)
# --------------------------------------------------------------------------- #

def run_brkga_fuzzy(
    engine,
    n_gen=200,
    seed=1,
    n_elites0=20,
    n_offsprings0=70,
    n_mutants0=10,
    bias=0.7,
    frac_elites_min=0.05,
    frac_elites_max=0.40,
    frac_mutantes_min=0.05,
    frac_mutantes_max=0.40,
    frac_offsprings_min=0.20,
    ref_diversidade=0.20,
    ref_qualidade=0.10,
    controlador_diversidade=None,
    controlador_qualidade=None,
):
    """
    Executa o BRKGA com dois controladores fuzzy ativos.

    Retorna:
        result_final -> dict com 'X' (melhor cromossomo) e 'F' (melhor makespan)
        historico    -> lista de dicts por geração
    """
    pop_size = n_elites0 + n_offsprings0 + n_mutants0

    if controlador_diversidade is None:
        controlador_diversidade = criar_controlador_diversidade()
    if controlador_qualidade is None:
        controlador_qualidade = criar_controlador_qualidade()

    problem = NPFSProblem(engine)
    algorithm = BRKGA(
        n_elites=n_elites0,
        n_offsprings=n_offsprings0,
        n_mutants=n_mutants0,
        bias=bias,
        eliminate_duplicates=True,
    )
    termination = get_termination("n_gen", n_gen)
    algorithm.setup(problem, termination=termination, seed=seed, verbose=False)

    historico = []
    best_makespan_gen0 = None
    erro_diversidade_anterior = None
    erro_qualidade_anterior = None

    frac_elites = n_elites0 / pop_size
    frac_mutantes = n_mutants0 / pop_size

    gen = 0
    while algorithm.has_next():
        algorithm.next()

        pop_X = algorithm.pop.get("X")
        pop_F = algorithm.pop.get("F").flatten()
        best_makespan = float(pop_F.min())

        if best_makespan_gen0 is None:
            best_makespan_gen0 = best_makespan

        diversidade = calcular_diversidade(pop_X)
        qualidade = calcular_qualidade(best_makespan, best_makespan_gen0)

        ref_div_g = _resolver_referencia(ref_diversidade, gen)
        ref_qual_g = _resolver_referencia(ref_qualidade, gen)

        erro_diversidade = ref_div_g - diversidade
        erro_qualidade = ref_qual_g - qualidade

        delta_erro_diversidade = (
            0.0 if erro_diversidade_anterior is None
            else erro_diversidade - erro_diversidade_anterior
        )
        delta_erro_qualidade = (
            0.0 if erro_qualidade_anterior is None
            else erro_qualidade - erro_qualidade_anterior
        )

        ajuste_mutantes = controlador_diversidade.calcular(erro_diversidade, delta_erro_diversidade)
        ajuste_elites = controlador_qualidade.calcular(erro_qualidade, delta_erro_qualidade)

        frac_mutantes = float(np.clip(frac_mutantes + ajuste_mutantes, frac_mutantes_min, frac_mutantes_max))
        frac_elites = float(np.clip(frac_elites + ajuste_elites, frac_elites_min, frac_elites_max))

        if frac_elites + frac_mutantes > (1 - frac_offsprings_min):
            excesso = (frac_elites + frac_mutantes) - (1 - frac_offsprings_min)
            total = frac_elites + frac_mutantes
            frac_elites -= excesso * (frac_elites / total)
            frac_mutantes -= excesso * (frac_mutantes / total)

        novo_n_elites = max(1, int(round(frac_elites * pop_size)))
        novo_n_mutantes = max(1, int(round(frac_mutantes * pop_size)))
        novo_n_offsprings = pop_size - novo_n_elites - novo_n_mutantes

        algorithm.survival.n_elites = novo_n_elites
        algorithm.n_elites = novo_n_elites
        algorithm.n_mutants = novo_n_mutantes
        algorithm.n_offsprings = novo_n_offsprings

        historico.append({
            "gen": gen,
            "best_makespan": best_makespan,
            "diversidade": diversidade,
            "qualidade": qualidade,
            "erro_diversidade": erro_diversidade,
            "variacao_erro_diversidade": delta_erro_diversidade,
            "erro_qualidade": erro_qualidade,
            "variacao_erro_qualidade": delta_erro_qualidade,
            "ref_diversidade": ref_div_g,
            "ref_qualidade": ref_qual_g,
            "ajuste_mutantes": ajuste_mutantes,
            "ajuste_elites": ajuste_elites,
            "n_elites": novo_n_elites,
            "n_mutants": novo_n_mutantes,
            "n_offsprings": novo_n_offsprings,
            "pop_X": pop_X.copy(),
        })

        erro_diversidade_anterior = erro_diversidade
        erro_qualidade_anterior = erro_qualidade
        gen += 1

    result_final = algorithm.result()
    melhor = {"X": result_final.X, "F": float(result_final.F[0])}

    return melhor, historico


# --------------------------------------------------------------------------- #
# 6. Optuna para BRKGA fixo
# --------------------------------------------------------------------------- #

def _fracoes_para_contagens(frac_elites, frac_mutantes, pop_size):
    n_elites = max(1, int(round(frac_elites * pop_size)))
    n_mutantes = max(1, int(round(frac_mutantes * pop_size)))
    n_offsprings = max(1, pop_size - n_elites - n_mutantes)
    return n_elites, n_offsprings, n_mutantes


def construir_objetivo(engine, pop_size=100, n_gen=100, seeds=(1, 2, 3)):
    """Cria a função-objetivo do Optuna para BRKGA fixo."""

    def objetivo(trial):
        import optuna
        frac_elites = trial.suggest_float("frac_elites", 0.05, 0.40)
        frac_mutantes = trial.suggest_float("frac_mutantes", 0.05, 0.40)
        bias = trial.suggest_float("bias", 0.40, 0.85)

        if frac_elites + frac_mutantes > 0.80:
            raise optuna.TrialPruned()

        n_elites, n_offsprings, n_mutantes = _fracoes_para_contagens(
            frac_elites, frac_mutantes, pop_size
        )

        makespans = []
        for seed in seeds:
            result, _ = run_brkga_fixo(
                engine, n_gen=n_gen, seed=seed,
                n_elites=n_elites, n_offsprings=n_offsprings, n_mutants=n_mutantes,
                bias=bias,
            )
            makespans.append(result.F[0])

        return float(np.mean(makespans))

    return objetivo


def rodar_optuna(engine, pop_size=100, n_gen=100, seeds=(1, 2, 3), n_trials=30, seed_optuna=0):
    """Executa o estudo Optuna e devolve (study, melhor_config)."""
    import optuna
    from optuna.samplers import TPESampler

    sampler = TPESampler(seed=seed_optuna)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(
        construir_objetivo(engine, pop_size, n_gen, seeds),
        n_trials=n_trials, show_progress_bar=False
    )

    melhores = study.best_params
    n_elites, n_offsprings, n_mutantes = _fracoes_para_contagens(
        melhores["frac_elites"], melhores["frac_mutantes"], pop_size
    )
    melhor_config = {
        "n_elites": n_elites,
        "n_offsprings": n_offsprings,
        "n_mutants": n_mutantes,
        "bias": melhores["bias"],
    }
    return study, melhor_config


def analisar_melhor_configuracao(engine, melhor_config, n_gen=100, seeds=(10, 11, 12, 13, 14)):
    """
    Roda a melhor configuração em várias sementes e devolve:
      - todas_metricas, trajetoria_diversidade, trajetoria_qualidade, resumo
    """
    todas_metricas = []
    makespans_finais = []

    for seed in seeds:
        result, historico = run_brkga_fixo(
            engine, n_gen=n_gen, seed=seed,
            n_elites=melhor_config["n_elites"],
            n_offsprings=melhor_config["n_offsprings"],
            n_mutants=melhor_config["n_mutants"],
            bias=melhor_config["bias"],
        )
        metricas = computar_metricas(historico, ref_diversidade=0.0, ref_qualidade=0.0)
        todas_metricas.append(metricas)
        makespans_finais.append(result.F[0])

    n_geracoes = min(len(m) for m in todas_metricas)
    diversidade_matriz = np.array([
        [m[g]["diversidade"] for g in range(n_geracoes)] for m in todas_metricas
    ])
    qualidade_matriz = np.array([
        [m[g]["qualidade"] for g in range(n_geracoes)] for m in todas_metricas
    ])

    trajetoria_diversidade = diversidade_matriz.mean(axis=0)
    trajetoria_qualidade = qualidade_matriz.mean(axis=0)

    resumo = {
        "makespan_medio": float(np.mean(makespans_finais)),
        "makespan_desvio_padrao": float(np.std(makespans_finais)),
        "makespans_finais": makespans_finais,
    }

    return todas_metricas, trajetoria_diversidade, trajetoria_qualidade, resumo


def construir_referencia_a_partir_da_trajetoria(trajetoria):
    """Transforma uma trajetória em referência 'gen -> valor'."""
    trajetoria = np.asarray(trajetoria)

    def referencia(gen):
        idx = min(gen, len(trajetoria) - 1)
        return float(trajetoria[idx])

    return referencia