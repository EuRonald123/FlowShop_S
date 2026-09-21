"""
Solver PBIL-Fuzzy (Population-Based Incremental Learning + Fuzzy) extraído do
repositório github.com/Renef13/trab_sint (aluno_2).
"""

import numpy as np
from scipy.spatial.distance import pdist


# --------------------------------------------------------------------------- #
# 1. Indivíduo e amostragem
# --------------------------------------------------------------------------- #

class IndividuoPBIL:
    """Indivíduo com chaves de prioridade para NPFS."""

    def __init__(self, chaves):
        self.chaves = np.asarray(chaves, dtype=float)  # shape (m, n)
        self.cmax = None

    @property
    def n_maquinas(self):
        return self.chaves.shape[0]

    @property
    def n_jobs(self):
        return self.chaves.shape[1]

    def decodificar(self):
        """Converte chaves em permutações (argsort estável)."""
        ordens = []
        for m in range(self.n_maquinas):
            ordens.append(np.argsort(self.chaves[m, :], kind="stable"))
        return ordens


def amostrar_individuo(matriz_p, sigma, distribuicao="normal_truncada", rng=None):
    """Amostra um indivíduo de P com dispersão sigma."""
    rng = rng or np.random.default_rng()
    if distribuicao == "normal_truncada":
        chaves = rng.normal(loc=matriz_p, scale=sigma)
    elif distribuicao == "uniforme":
        chaves = rng.uniform(low=matriz_p - sigma, high=matriz_p + sigma)
    else:
        raise ValueError(f"Distribuição desconhecida: {distribuicao}")
    chaves = np.clip(chaves, 0.0, 1.0)
    return IndividuoPBIL(chaves)


def amostrar_populacao(matriz_p, n_pop, sigma, distribuicao="normal_truncada", rng=None):
    return [amostrar_individuo(matriz_p, sigma, distribuicao, rng) for _ in range(n_pop)]


# --------------------------------------------------------------------------- #
# 2. Matriz de probabilidades P
# --------------------------------------------------------------------------- #

def inicializar_matriz_p(n_maquinas, n_jobs, valor_inicial=0.5):
    return np.full((n_maquinas, n_jobs), valor_inicial, dtype=float)


def calcular_chave_media_elite(elite):
    if not elite:
        raise ValueError("Elite vazia")
    return np.mean(np.stack([m.chaves for m in elite], axis=0), axis=0)


def atualizar_matriz_p(matriz_p, elite, alpha):
    """Atualização Hebbiana: P = P*(1-alpha) + alpha*media_elite."""
    chave_media = calcular_chave_media_elite(elite)
    return matriz_p * (1.0 - alpha) + alpha * chave_media


# --------------------------------------------------------------------------- #
# 3. Elite e diversidade estrutural
# --------------------------------------------------------------------------- #

def calcular_diversidade_estrutural(populacao, n_maquinas):
    """
    Calcula diversidade estrutural média entre todos os pares de indivíduos.
    Similar a aluno_4, mas usando chaves (não cromossomos pymoo).
    """
    if len(populacao) < 2:
        return 0.0
    chaves_mat = np.array([ind.chaves.flatten() for ind in populacao])
    n_var = chaves_mat.shape[1]
    dists = pdist(chaves_mat, metric='cityblock')
    return float((dists / n_var).mean())


def montar_elite_e_subelite(populacao, cmax_best, pct_elite, pct_subelite, delta):
    """
    Seleciona elite e sub-elite.
    - Elite: melhores indivíduos (menor cmax)
    - Sub-elite: indivíduos com cmax dentro de delta * cmax_best da elite
    """
    populacao.sort(key=lambda ind: ind.cmax)
    n_elite = max(1, int(round(len(populacao) * pct_elite)))
    elite = populacao[:n_elite]

    if delta > 0 and cmax_best > 0:
        limiar = cmax_best * (1.0 + delta)
        subelite = [ind for ind in populacao if ind.cmax <= limiar and ind not in elite]
    else:
        subelite = populacao[n_elite:]

    n_subelite = max(1, int(round(len(populacao) * pct_subelite)))
    subelite = subelite[:n_subelite]

    return {"elite": elite, "subelite": subelite}


# --------------------------------------------------------------------------- #
# 4. Controlador Fuzzy (Mamdani simplificado)
# --------------------------------------------------------------------------- #

def _pertinencia_triangular(valor, pontos):
    """
    Função de pertinência triangular para um conjunto de 3 pontos [a, b, c].
    """
    a, b, c = pontos
    if valor <= a or valor >= c:
        return 0.0
    if a < valor <= b:
        return (valor - a) / (b - a)
    return (c - valor) / (c - b)


def _centroide(regras_ativadas, termos_saida):
    """
    Defuzzificação por centroide simplificada.
    regras_ativadas: [(grau_ativacao, valor_saida), ...]
    """
    numerador = sum(g * v for g, v in regras_ativadas)
    denominador = sum(g for g, _ in regras_ativadas)
    if denominador < 1e-9:
        return 0.5  # valor neutro
    return numerador / denominador


class ControladorFuzzyPBIL:
    """
    Controlador fuzzy Mamdani simplificado.
    2 entradas: progresso_qualidade (q combinado), diversidade
    2 saídas: alpha, beta
    """

    # Termos linguísticos para progresso_qualidade
    _TERMOS_Q = {
        "muito_distante": (0.0, 0.0, 0.05),
        "distante": (0.0, 0.05, 0.15),
        "moderado": (0.05, 0.15, 0.35),
        "proximo": (0.15, 0.35, 0.65),
        "muito_proximo": (0.35, 0.65, 1.0),
    }

    # Termos linguísticos para diversidade
    _TERMOS_DIV = {
        "baixa": (0.0, 0.0, 0.25),
        "media": (0.0, 0.25, 0.60),
        "alta": (0.25, 0.60, 1.0),
    }

    # Termos de saída para alpha e beta
    _TERMOS_SAIDA = {
        "muito_baixo": (-0.5, -0.5, -0.2),
        "baixo": (-0.5, -0.2, 0.0),
        "medio": (-0.2, 0.0, 0.2),
        "alto": (0.0, 0.2, 0.5),
        "muito_alto": (0.2, 0.5, 0.5),
    }

    # FAM (Fuzzy Associative Matrix): 5x3 = 15 regras
    # Cada entrada: (termo_q, termo_div, alpha_saida, beta_saida)
    _FAM = [
        # Progresso muito distante (precisa explorar)
        ("muito_distante", "baixa", "alto", "alto"),
        ("muito_distante", "media", "muito_alto", "alto"),
        ("muito_distante", "alta", "muito_alto", "muito_alto"),
        # Progresso distante
        ("distante", "baixa", "medio", "medio"),
        ("distante", "media", "alto", "alto"),
        ("distante", "alta", "muito_alto", "alto"),
        # Progresso moderado
        ("moderado", "baixa", "baixo", "baixo"),
        ("moderado", "media", "medio", "medio"),
        ("moderado", "alta", "alto", "medio"),
        # Progresso próximo
        ("proximo", "baixa", "muito_baixo", "baixo"),
        ("proximo", "media", "baixo", "baixo"),
        ("proximo", "alta", "medio", "medio"),
        # Progresso muito próximo (quase convergiu)
        ("muito_proximo", "baixa", "muito_baixo", "muito_baixo"),
        ("muito_proximo", "media", "muito_baixo", "baixo"),
        ("muito_proximo", "alta", "baixo", "medio"),
    ]

    def __init__(self, peso_q=0.8, peso_sigma_rel=0.2):
        self.peso_q = peso_q
        self.peso_sigma_rel = peso_sigma_rel

    def _combinar_q_sigma(self, q, sigma_rel):
        q = np.clip(q, 0.0, 1.0)
        sigma_rel = np.clip(sigma_rel, 0.0, 1.0)
        return self.peso_q * q + self.peso_sigma_rel * sigma_rel

    def calcular(self, q, sigma_rel, diversidade_estrutural):
        progresso = self._combinar_q_sigma(q, sigma_rel)
        div = np.clip(diversidade_estrutural, 0.0, 1.0)

        # Avalia cada uma das 15 regras
        ativacoes_alpha = []
        ativacoes_beta = []

        for termo_q, termo_div, termo_alpha, termo_beta in self._FAM:
            # Grau de ativação = min(pertinência_q, pertinência_div)
            g_q = _pertinencia_triangular(progresso, self._TERMOS_Q[termo_q])
            g_div = _pertinencia_triangular(div, self._TERMOS_DIV[termo_div])
            ativacao = min(g_q, g_div)

            if ativacao > 1e-6:
                valor_alpha = _pertinencia_triangular(
                    0.0, self._TERMOS_SAIDA[termo_alpha]
                )  # Pico do termo de saída
                # Usamos o valor de pico do termo triangular
                a, b, c = self._TERMOS_SAIDA[termo_alpha]
                pico_alpha = b

                a, b, c = self._TERMOS_SAIDA[termo_beta]
                pico_beta = b

                ativacoes_alpha.append((ativacao, pico_alpha))
                ativacoes_beta.append((ativacao, pico_beta))

        # Centroide para alpha e beta
        alpha = _centroide(ativacoes_alpha, self._TERMOS_SAIDA)
        beta = _centroide(ativacoes_beta, self._TERMOS_SAIDA)

        # Mapeia de [-0.5, 0.5] para [0.0, 1.0] (faixa útil para alpha/beta)
        alpha = np.clip(alpha + 0.5, 0.0, 1.0)
        beta = np.clip(beta + 0.5, 0.0, 1.0)

        return alpha, beta


# --------------------------------------------------------------------------- #
# 5. Métricas
# --------------------------------------------------------------------------- #

def calcular_gap_relativo(lista_cmax, cmax_best):
    """gap_relativo = (cmax_medio - cmax_best) / cmax_best"""
    if cmax_best <= 0 or not lista_cmax:
        return 0.0
    return (float(np.mean(lista_cmax)) - cmax_best) / cmax_best


def calcular_dispersao_relativa(lista_cmax, cmax_best):
    """dispersao_relativa = std(cmax) / cmax_best"""
    if cmax_best <= 0 or len(lista_cmax) < 2:
        return 0.0
    return float(np.std(lista_cmax)) / cmax_best


# --------------------------------------------------------------------------- #
# 6. Motor principal PBIL-Fuzzy
# --------------------------------------------------------------------------- #

class PBILFuzzySolver:
    """
    PBIL com controle fuzzy adaptativo para alpha/beta.

    Hiperparâmetros:
        n_pop, sigma_amostragem, pct_elite, pct_subelite, delta,
        alpha_inicial, beta_inicial, max_geracoes
    """

    def __init__(self, engine, seed=None,
                 n_pop=80, sigma_amostragem=0.10,
                 pct_elite=0.08, pct_subelite=0.35, delta=0.20,
                 alpha_inicial=0.5, beta_inicial=0.5,
                 max_geracoes=300, objective="Makespan"):
        self.engine = engine
        self.n_jobs = engine.n_jobs
        self.n_maquinas = engine.n_machines
        self.n_pop = n_pop
        self.sigma_amostragem = sigma_amostragem
        self.pct_elite = pct_elite
        self.pct_subelite = pct_subelite
        self.delta = delta
        self.alpha = alpha_inicial
        self.beta = beta_inicial
        self.max_geracoes = max_geracoes
        self.objective = objective
        self.rng = np.random.default_rng(seed)

        self.matriz_p = inicializar_matriz_p(self.n_maquinas, self.n_jobs)
        self.controlador = ControladorFuzzyPBIL()

        self.best_perm = None
        self.best_cost = float("inf")
        self.history_best = []

    def _avaliar_individuo(self, engine, individuo):
        """Avalia um indivíduo usando a FlowShopEngine oficial."""
        ordens = individuo.decodificar()
        # Converte para matriz (m, n) como esperado pela engine
        perm_matrix = np.array([ordens[m] for m in range(self.n_maquinas)])
        # Para NPFS, a engine espera keys com shape (n*m,) ou evaluate com mode
        # Usamos evaluate_with_keys - na verdade, vamos usar evaluate
        # Como a engine espera keys [0,1], e temos permutações, vamos
        # criar keys a partir das permutações
        keys = np.zeros(self.n_jobs * self.n_maquinas)
        for m in range(self.n_maquinas):
            # Inverte argsort: de permutação para keys
            inv = np.argsort(ordens[m])
            keys[m * self.n_jobs:(m + 1) * self.n_jobs] = inv.astype(float) / self.n_jobs

        cost = engine.evaluate(keys, mode="NPFS", objective=self.objective)
        individuo.cmax = cost
        return cost

    def executar(self, verbose=False):
        """Executa o laço principal do PBIL-Fuzzy."""
        # População inicial
        populacao = amostrar_populacao(
            self.matriz_p, self.n_pop, self.sigma_amostragem,
            rng=self.rng
        )
        for ind in populacao:
            self._avaliar_individuo(self.engine, ind)
        populacao.sort(key=lambda ind: ind.cmax)

        cmax_best = float("inf")

        for geracao in range(1, self.max_geracoes + 1):
            lista_cmax = [ind.cmax for ind in populacao]
            cmax_best = min(cmax_best, min(lista_cmax))

            if cmax_best < self.best_cost:
                self.best_cost = cmax_best
                # Pega o melhor indivíduo
                melhor = min(populacao, key=lambda ind: ind.cmax)
                ordens = melhor.decodificar()
                self.best_perm = np.array([ordens[m] for m in range(self.n_maquinas)])

            self.history_best.append(self.best_cost)

            # Meta-features
            gap_rel = calcular_gap_relativo(lista_cmax, cmax_best)
            disp_rel = calcular_dispersao_relativa(lista_cmax, cmax_best)
            diversidade = calcular_diversidade_estrutural(populacao, self.n_maquinas)

            # Fuzzy: calcula alpha e beta
            alpha_novo, beta_novo = self.controlador.calcular(gap_rel, disp_rel, diversidade)

            # Seleção de elite e sub-elite
            selecao = montar_elite_e_subelite(
                populacao, cmax_best, self.pct_elite, self.pct_subelite, self.delta
            )
            elite = selecao["elite"]
            subelite = selecao["subelite"]

            # Atualiza matriz P com alpha
            self.matriz_p = atualizar_matriz_p(self.matriz_p, elite, alpha_novo)

            # Compõe próxima população
            n_elite = len(elite)
            vagas = max(0, self.n_pop - n_elite)
            tamanho_amostrado = int(round(beta_novo * vagas))
            tamanho_subelite = vagas - tamanho_amostrado

            proxima = list(elite)
            proxima.extend(amostrar_populacao(
                self.matriz_p, tamanho_amostrado, self.sigma_amostragem, rng=self.rng
            ))
            copias_subelite = list(subelite[:tamanho_subelite])
            deficit = tamanho_subelite - len(copias_subelite)
            if deficit > 0:
                copias_subelite.extend(amostrar_populacao(
                    self.matriz_p, deficit, self.sigma_amostragem, rng=self.rng
                ))
            proxima.extend(copias_subelite)
            proxima = proxima[:self.n_pop]

            # Avalia nova população
            for ind in proxima:
                self._avaliar_individuo(self.engine, ind)
            proxima.sort(key=lambda ind: ind.cmax)

            populacao = proxima
            self.alpha, self.beta = alpha_novo, beta_novo

            if verbose and (geracao % 50 == 0 or geracao == self.max_geracoes):
                print(f"Geração {geracao:4d}/{self.max_geracoes} | "
                      f"best={self.best_cost:.2f} | "
                      f"alpha={self.alpha:.3f} beta={self.beta:.3f} | "
                      f"div={diversidade:.3f}")

        return self.best_perm, self.best_cost, self.history_best