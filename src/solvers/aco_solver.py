"""
Solver ACO (Ant Colony Optimization) extraído do aluno_1.ipynb.
Mantém a lógica original: construção de formigas com feromônio + heurística EDD,
atualização de feromônio, e suporte a Optuna para hiperparâmetros.
"""

import numpy as np


class ACOSolver:
    """
    ACO para Flow Shop Scheduling com suporte a PFS e NPFS.
    """

    def __init__(self, engine, n_ants=20, n_iterations=100,
                 alpha=1.0, beta=2.0, rho=0.1, Q=100.0,
                 objective="Tardiness", seed=None, mode="PFS"):

        self.engine = engine
        self.n = engine.n_jobs
        self.m = engine.n_machines
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.objective = objective
        self.mode = mode  # "PFS" ou "NPFS"
        self.rng = np.random.default_rng(seed)

        # Feromônio: tau[n][j] = do nó virtual "início" para job j
        #            tau[i][j] = de job i para job j
        if mode == "NPFS":
            # Um feromônio por máquina, pois cada máquina tem sua permutação
            self.tau = [np.ones((self.n + 1, self.n)) for _ in range(self.m)]
            # Heurística local (EDD): mesma para todas as máquinas
            eps = 1e-6
            self.eta = [1.0 / (engine.due_dates.astype(float) + eps)] * self.m
        else:
            self.tau = np.ones((self.n + 1, self.n))
            eps = 1e-6
            self.eta = 1.0 / (engine.due_dates.astype(float) + eps)

        self.best_perm = None
        self.best_cost = float('inf')
        self.history_best = []

    def _construir_formiga_pfs(self):
        """Constrói uma permutação (PFS: mesma ordem em todas as máquinas)."""
        START = self.n
        visitados = np.zeros(self.n, dtype=bool)
        perm = np.empty(self.n, dtype=int)
        atual = START

        for passo in range(self.n):
            candidatos = np.where(~visitados)[0]
            tau_ij = self.tau[atual, candidatos] ** self.alpha
            eta_ij = self.eta[candidatos] ** self.beta
            pesos = tau_ij * eta_ij
            soma = pesos.sum()

            if soma <= 0 or not np.isfinite(soma):
                probs = np.ones(len(candidatos)) / len(candidatos)
            else:
                probs = pesos / soma

            escolhido = self.rng.choice(candidatos, p=probs)
            perm[passo] = escolhido
            visitados[escolhido] = True
            atual = escolhido

        return perm

    def _construir_formiga_npfs(self):
        """
        Constrói uma permutação INDEPENDENTE para cada máquina (NPFS).
        Cada máquina tem seu próprio feromônio.
        """
        perms = np.empty((self.m, self.n), dtype=int)
        for maq in range(self.m):
            START = self.n
            visitados = np.zeros(self.n, dtype=bool)
            atual = START
            tau_m = self.tau[maq]
            eta_m = self.eta[maq]
            alpha = self.alpha
            beta = self.beta

            for passo in range(self.n):
                candidatos = np.where(~visitados)[0]
                tau_ij = tau_m[atual, candidatos] ** alpha
                eta_ij = eta_m[candidatos] ** beta
                pesos = tau_ij * eta_ij
                soma = pesos.sum()

                if soma <= 0 or not np.isfinite(soma):
                    probs = np.ones(len(candidatos)) / len(candidatos)
                else:
                    probs = pesos / soma

                escolhido = self.rng.choice(candidatos, p=probs)
                perms[maq, passo] = escolhido
                visitados[escolhido] = True
                atual = escolhido

        return perms

    def _atualizar_feromonio_pfs(self, permutacoes, custos):
        """Atualiza feromônio para PFS."""
        self.tau *= (1.0 - self.rho)

        for perm, custo in zip(permutacoes, custos):
            if custo <= 0:
                continue
            deposito = self.Q / custo
            no_anterior = self.n
            for job in perm:
                self.tau[no_anterior, job] += deposito
                no_anterior = job

    def _atualizar_feromonio_npfs(self, individuos, custos):
        """Atualiza feromônio para NPFS (um por máquina)."""
        for maq in range(self.m):
            self.tau[maq] *= (1.0 - self.rho)

        for ind, custo in zip(individuos, custos):
            if custo <= 0:
                continue
            deposito = self.Q / custo
            for maq in range(self.m):
                no_anterior = self.n
                for job in ind[maq]:
                    self.tau[maq][no_anterior, job] += deposito
                    no_anterior = job

    def run(self, verbose=False):
        """
        Executa o ACO.

        Retorna:
            best_perm: np.ndarray
                PFS: shape (n,) - permutação única
                NPFS: shape (m, n) - permutação por máquina
            best_cost: float
            history_best: list[float]
        """
        for it in range(self.n_iterations):
            if self.mode == "NPFS":
                individuos = [self._construir_formiga_npfs() for _ in range(self.n_ants)]
                custos = [self.engine.evaluate(ind, mode="NPFS", objective=self.objective)
                          for ind in individuos]
            else:
                permutacoes = [self._construir_formiga_pfs() for _ in range(self.n_ants)]
                custos = [self.engine.evaluate_with_permutation(p, objective=self.objective)
                          for p in permutacoes]

            idx_melhor_iter = int(np.argmin(custos))
            if custos[idx_melhor_iter] < self.best_cost:
                self.best_cost = custos[idx_melhor_iter]
                if self.mode == "NPFS":
                    self.best_perm = individuos[idx_melhor_iter].copy()
                else:
                    self.best_perm = permutacoes[idx_melhor_iter].copy()

            if self.mode == "NPFS":
                self._atualizar_feromonio_npfs(individuos, custos)
            else:
                self._atualizar_feromonio_pfs(permutacoes, custos)

            self.history_best.append(self.best_cost)

            if verbose and (it % 10 == 0 or it == self.n_iterations - 1):
                print(f"Iteração {it:4d} | melhor custo até agora: {self.best_cost:.2f}")

        return self.best_perm, self.best_cost, self.history_best

    @staticmethod
    def optuna_search_space(trial):
        """Espaço de busca para Optuna."""
        return {
            "alpha": trial.suggest_float("alpha", 0.1, 5.0),
            "beta": trial.suggest_float("beta", 0.1, 5.0),
            "rho": trial.suggest_float("rho", 0.01, 0.9),
            "Q": trial.suggest_float("Q", 1.0, 500.0),
            "n_ants": trial.suggest_int("n_ants", 5, 50),
        }