"""
Solver Neuro-BOA (Bayesian Optimization Algorithm + Redes Neurais) extraído do
aluno_3.ipynb.

Inclui:
- BOA (modelo bayesiano autoregressivo para permutações)
- Neuro (MLP para geração de permutações)
- Meta-redes para selecionar BOA vs Neuro (qualidade e diversidade)
- 7 modos experimentais: boa_puro, neuro_puro, boa_duplo, etc.
"""

import copy
import random
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# 1. Instance e decoder (mantidos do original para compatibilidade)
# --------------------------------------------------------------------------- #

@dataclass
class Instance:
    name: str
    group: str
    n_jobs: int
    n_machines: int
    processing: np.ndarray
    due_dates: np.ndarray
    metadata_line: Optional[int] = None


def decode_npfs(ind, inst):
    """
    Decodificador via grafo de precedências (Non-Permutation Flow Shop).
    Nós: (job, máquina). Arestas: precedência tecnológica + ordem na máquina.
    """
    n, M = inst.n_jobs, inst.n_machines
    N = n * M

    succ = [[] for _ in range(N)]
    indeg = np.zeros(N, dtype=np.int32)

    def nid(j, m):
        return j * M + m

    # Precedência tecnológica
    for j in range(n):
        for m in range(1, M):
            u, v = nid(j, m-1), nid(j, m)
            succ[u].append(v)
            indeg[v] += 1

    # Ordem da solução em cada máquina
    for m in range(M):
        seq = ind[m]
        for p in range(1, n):
            u, v = nid(int(seq[p-1]), m), nid(int(seq[p]), m)
            succ[u].append(v)
            indeg[v] += 1

    q = deque([u for u in range(N) if indeg[u] == 0])
    start = np.zeros(N, dtype=np.int64)
    finish = np.zeros(N, dtype=np.int64)
    processed = 0

    while q:
        u = q.popleft()
        processed += 1
        j, m = divmod(u, M)
        finish[u] = start[u] + int(inst.processing[j, m])

        for v in succ[u]:
            start[v] = max(start[v], finish[u])
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)

    if processed != N:
        return {
            "feasible": False,
            "makespan": float("inf"),
            "total_tardiness": float("inf"),
            "completion_jobs": None,
        }

    completion = np.asarray([finish[nid(j, M-1)] for j in range(n)])
    makespan = int(completion.max())
    total_tardiness = int(np.maximum(completion - inst.due_dates, 0).sum())

    return {
        "feasible": True,
        "makespan": makespan,
        "total_tardiness": total_tardiness,
        "completion_jobs": completion,
    }


def evaluate(ind, inst, objective="makespan"):
    """Avalia um indivíduo usando o decoder original do Neuro-BOA."""
    d = decode_npfs(ind, inst)
    if not d["feasible"]:
        return 1e18
    if objective == "makespan":
        return float(d["makespan"])
    if objective == "tardiness":
        return float(d["total_tardiness"])
    raise ValueError("objective deve ser 'makespan' ou 'tardiness'")


# --------------------------------------------------------------------------- #
# 2. Distância Kendall e diversidade
# --------------------------------------------------------------------------- #

def kendall_distance_perm(a, b):
    n = len(a)
    pos = np.empty(n, dtype=np.int32)
    for i, x in enumerate(a):
        pos[int(x)] = i
    arr = [int(pos[int(x)]) for x in b]
    tmp = [0] * len(arr)

    def sort_count(lo, hi):
        if hi - lo <= 1:
            return 0
        mid = (lo + hi) // 2
        c = sort_count(lo, mid) + sort_count(mid, hi)
        i, j, k = lo, mid, lo
        while i < mid and j < hi:
            if arr[i] <= arr[j]:
                tmp[k] = arr[i]
                i += 1
            else:
                tmp[k] = arr[j]
                j += 1
                c += mid - i
            k += 1
        while i < mid:
            tmp[k] = arr[i]
            i += 1
            k += 1
        while j < hi:
            tmp[k] = arr[j]
            j += 1
            k += 1
        arr[lo:hi] = tmp[lo:hi]
        return c

    return sort_count(0, len(arr))


def individual_distance(a, b):
    M, n = a.shape
    denom = n * (n - 1) / 2
    if denom == 0:
        return 0.0
    return float(np.mean([
        kendall_distance_perm(a[m], b[m]) / denom for m in range(M)
    ]))


def distance_to_elite(ind, elite):
    if not elite:
        return 0.0
    return float(np.mean([individual_distance(ind, e) for e in elite]))


def population_diversity(pop, sample_pairs=80, rng=None):
    if len(pop) < 2:
        return 0.0
    rng = rng or np.random.default_rng()
    max_pairs = len(pop) * (len(pop) - 1) // 2
    k = min(sample_pairs, max_pairs)
    pairs = set()
    while len(pairs) < k:
        i, j = rng.integers(0, len(pop), size=2)
        if i == j:
            continue
        i, j = sorted((int(i), int(j)))
        pairs.add((i, j))
    return float(np.mean([
        individual_distance(pop[i], pop[j]) for i, j in pairs
    ]))


def select_diverse_good(pop, fitness, elite_count,
                        candidate_frac=0.35, diverse_count=None):
    order = np.argsort(fitness)
    elite_idx = order[:elite_count]
    elite = [pop[i] for i in elite_idx]

    n_candidates = max(1, int(round(len(pop) * candidate_frac)))
    cand_idx = order[elite_count:min(len(pop), elite_count + n_candidates)]

    if len(cand_idx) == 0:
        return elite, [], []

    scored = [(int(i), distance_to_elite(pop[int(i)], elite)) for i in cand_idx]
    scored.sort(key=lambda z: z[1], reverse=True)

    if diverse_count is None:
        diverse_count = elite_count

    chosen = scored[:min(diverse_count, len(scored))]
    diverse = [pop[i] for i, _ in chosen]
    scores = [s for _, s in chosen]
    return elite, diverse, scores


# --------------------------------------------------------------------------- #
# 3. BOA: modelo bayesiano autoregressivo
# --------------------------------------------------------------------------- #

class BayesianPermutationModel:
    """
    Para cada máquina aprende:
      P(primeiro job)
      P(job atual | job anterior)
    """

    def __init__(self, n_jobs, n_machines, smoothing=1.0):
        self.n = n_jobs
        self.M = n_machines
        self.smoothing = smoothing
        self.first = np.ones((self.M, self.n)) / self.n
        self.trans = np.ones((self.M, self.n, self.n)) / self.n

    def fit(self, individuals):
        if not individuals:
            return
        first_counts = np.full((self.M, self.n), self.smoothing, dtype=np.float64)
        trans_counts = np.full((self.M, self.n, self.n), self.smoothing, dtype=np.float64)

        for ind in individuals:
            for m in range(self.M):
                seq = ind[m]
                first_counts[m, int(seq[0])] += 1
                for p in range(1, self.n):
                    a = int(seq[p - 1])
                    b = int(seq[p])
                    trans_counts[m, a, b] += 1

        self.first = first_counts / first_counts.sum(axis=1, keepdims=True)
        self.trans = trans_counts / trans_counts.sum(axis=2, keepdims=True)

    def sample_machine(self, m, rng):
        used = np.zeros(self.n, dtype=bool)
        seq = []
        p = self.first[m].copy()
        x = int(rng.choice(self.n, p=p / p.sum()))
        seq.append(x)
        used[x] = True

        for _ in range(1, self.n):
            p = self.trans[m, x].copy()
            p[used] = 0.0
            if p.sum() <= 0:
                x = int(rng.choice(np.where(~used)[0]))
            else:
                p /= p.sum()
                x = int(rng.choice(self.n, p=p))
            seq.append(x)
            used[x] = True

        return np.asarray(seq, dtype=np.int32)

    def sample(self, rng):
        return np.vstack([self.sample_machine(m, rng) for m in range(self.M)])


# --------------------------------------------------------------------------- #
# 4. Neuro: MLP autoregressiva para permutações
# --------------------------------------------------------------------------- #

class NeuroPermutationMLP(nn.Module):
    def __init__(self, n_jobs, hidden=128, dropout=0.0):
        super().__init__()
        inp = 2 * n_jobs + 2
        self.net = nn.Sequential(
            nn.Linear(inp, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_jobs),
        )

    def forward(self, x):
        return self.net(x)


def build_neuro_examples(individuals, n_jobs, n_machines):
    X, y = [], []
    for ind in individuals:
        for m in range(n_machines):
            used = np.zeros(n_jobs, dtype=np.float32)
            last = np.zeros(n_jobs, dtype=np.float32)
            for p in range(n_jobs):
                target = int(ind[m, p])
                feat = np.concatenate([
                    used, last,
                    np.array([p / max(1, n_jobs - 1), m / max(1, n_machines - 1)], dtype=np.float32)
                ])
                X.append(feat)
                y.append(target)
                used[target] = 1.0
                last[:] = 0.0
                last[target] = 1.0

    if not X:
        return None, None
    return (
        torch.tensor(np.asarray(X), dtype=torch.float32),
        torch.tensor(np.asarray(y), dtype=torch.long),
    )


def train_neuro(model, individuals, n_jobs, n_machines,
                lr=1e-3, epochs=2, batch_size=256):
    X, y = build_neuro_examples(individuals, n_jobs, n_machines)
    if X is None:
        return

    X, y = X.to(DEVICE), y.to(DEVICE)

    if not hasattr(model, "_optimizer"):
        model._optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        model._lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            model._optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6,
        )

    opt = model._optimizer
    model.train()
    losses = []

    for _ in range(epochs):
        order = torch.randperm(len(X), device=DEVICE)
        Xs, ys = X[order], y[order]
        for s in range(0, len(Xs), batch_size):
            xb = Xs[s:s + batch_size]
            yb = ys[s:s + batch_size]
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))

    if losses:
        model._lr_scheduler.step(float(np.mean(losses)))

    return float(opt.param_groups[0]["lr"])


def neuro_sample(model, n_jobs, n_machines, rng, temperature=1.0):
    model.eval()
    all_machines = []
    with torch.no_grad():
        for m in range(n_machines):
            used = np.zeros(n_jobs, dtype=np.float32)
            last = np.zeros(n_jobs, dtype=np.float32)
            seq = []
            for p in range(n_jobs):
                feat = np.concatenate([
                    used, last,
                    np.array([p / max(1, n_jobs - 1), m / max(1, n_machines - 1)], dtype=np.float32)
                ])
                x = torch.tensor(feat, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                logits = model(x).squeeze(0)
                mask = torch.tensor(used.astype(bool), device=DEVICE)
                logits = logits.masked_fill(mask, -1e9)
                probs = torch.softmax(logits / max(temperature, 1e-4), dim=0).cpu().numpy()
                job = int(rng.choice(n_jobs, p=probs / probs.sum()))
                seq.append(job)
                used[job] = 1.0
                last[:] = 0.0
                last[job] = 1.0
            all_machines.append(seq)
    return np.asarray(all_machines, dtype=np.int32)


# --------------------------------------------------------------------------- #
# 5. Meta-redes: escolhem BOA (0) ou Neuro (1)
# --------------------------------------------------------------------------- #

class MetaSelector(nn.Module):
    def __init__(self, input_dim, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.net(x)


class OnlineSelector:
    def __init__(self, input_dim, hidden=32, lr=1e-3, min_samples=12):
        self.model = MetaSelector(input_dim, hidden).to(DEVICE)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt, mode="min", factor=0.5, patience=3, min_lr=1e-6,
        )
        self.X = []
        self.y = []
        self.min_samples = min_samples

    def add(self, state, label):
        self.X.append(np.asarray(state, dtype=np.float32))
        self.y.append(int(label))

    def train_steps(self, steps=8):
        if len(self.X) < self.min_samples:
            return
        X = torch.tensor(np.asarray(self.X), dtype=torch.float32, device=DEVICE)
        y = torch.tensor(np.asarray(self.y), dtype=torch.long, device=DEVICE)
        self.model.train()
        losses = []
        for _ in range(steps):
            idx = torch.randint(0, len(X), (min(64, len(X)),), device=DEVICE)
            logits = self.model(X[idx])
            loss = F.cross_entropy(logits, y[idx])
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            losses.append(float(loss.detach().cpu()))
        if losses:
            self.lr_scheduler.step(float(np.mean(losses)))
        return float(self.opt.param_groups[0]["lr"])

    def choose(self, state, rng, epsilon=0.10):
        if len(self.X) < self.min_samples or rng.random() < epsilon:
            return int(rng.integers(0, 2))
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            return int(torch.argmax(self.model(x), dim=1).item())


# --------------------------------------------------------------------------- #
# 6. Estado e desempenho dos geradores
# --------------------------------------------------------------------------- #

def safe_improvement(reference, value):
    if not np.isfinite(reference) or reference <= 0:
        return 0.0
    return max(0.0, (reference - value) / reference)


def generator_stats(values, reference):
    vals = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if len(vals) == 0:
        return {"success": 0.0, "mean_improvement": 0.0, "best_improvement": 0.0}
    imps = np.asarray([safe_improvement(reference, v) for v in vals])
    return {
        "success": float(np.mean(vals < reference)),
        "mean_improvement": float(imps.mean()),
        "best_improvement": float(imps.max()),
    }


def quality_generator_score(values, reference):
    s = generator_stats(values, reference)
    return 0.5 * s["success"] + 0.3 * s["mean_improvement"] + 0.2 * s["best_improvement"]


def diversity_generator_score(children, values, elite, reference):
    if not children:
        return 0.0
    quality, diversity = [], []
    for ind, value in zip(children, values):
        if not np.isfinite(value):
            continue
        q = 1.0 / (1.0 + max(0.0, (value - reference) / max(reference, 1e-9)))
        quality.append(q)
        diversity.append(distance_to_elite(ind, elite))
    if not quality:
        return 0.0
    return 0.5 * float(np.mean(quality)) + 0.5 * float(np.mean(diversity))


def _common_state_features(best_hist, pop_fitness, pop, rng):
    best = float(np.min(pop_fitness))
    mean = float(np.mean(pop_fitness))
    var = float(np.var(pop_fitness))

    if len(best_hist) >= 2:
        k = min(5, len(best_hist) - 1)
        prev = best_hist[-1 - k]
        recent_impr = max(0.0, (prev - best_hist[-1]) / max(prev, 1e-9))
    else:
        recent_impr = 0.0

    stagnation = 0
    cur = best_hist[-1]
    for x in reversed(best_hist[:-1]):
        if abs(x - cur) <= 1e-12:
            stagnation += 1
        else:
            break

    div = population_diversity(pop, 60, rng)
    return [mean / max(best, 1e-9), var / max(mean * mean, 1e-9),
            recent_impr, min(stagnation / 20.0, 1.0), div]


def make_quality_state(best_hist, pop_fitness, pop, stats_boa, stats_neuro, rng):
    common = _common_state_features(best_hist, pop_fitness, pop, rng)
    return np.asarray(common + [
        stats_boa["success"], stats_boa["mean_improvement"],
        stats_neuro["success"], stats_neuro["mean_improvement"],
    ], dtype=np.float32)


def diversity_probe_stats(children, values, elite, reference):
    valid = [(ind, value) for ind, value in zip(children, values) if np.isfinite(value)]
    if not valid:
        return {
            "success": 0.0, "mean_improvement": 0.0,
            "mean_distance": 0.0, "best_distance": 0.0, "useful_score": 0.0,
        }
    vals = np.asarray([value for _, value in valid], dtype=np.float64)
    improvements = np.asarray([safe_improvement(reference, value) for value in vals])
    distances = np.asarray([distance_to_elite(ind, elite) for ind, _ in valid], dtype=np.float64)
    useful_score = diversity_generator_score(
        [ind for ind, _ in valid], vals, elite, reference
    )
    return {
        "success": float(np.mean(vals < reference)),
        "mean_improvement": float(improvements.mean()),
        "mean_distance": float(distances.mean()),
        "best_distance": float(distances.max()),
        "useful_score": float(useful_score),
    }


def make_diversity_state(best_hist, pop_fitness, pop,
                         stats_boa_d, stats_neuro_d, rng):
    common = _common_state_features(best_hist, pop_fitness, pop, rng)
    return np.asarray(common + [
        stats_boa_d["success"], stats_boa_d["mean_improvement"],
        stats_boa_d["mean_distance"], stats_boa_d["useful_score"],
        stats_neuro_d["success"], stats_neuro_d["mean_improvement"],
        stats_neuro_d["mean_distance"], stats_neuro_d["useful_score"],
    ], dtype=np.float32)


# --------------------------------------------------------------------------- #
# 7. Configuração
# --------------------------------------------------------------------------- #

@dataclass
class ConfigNeuroBOA:
    population_size: int = 60
    generations: int = 80
    elite_frac: float = 0.15
    diverse_candidate_frac: float = 0.35
    diverse_frac: float = 0.20
    neuro_hidden: int = 128
    neuro_lr: float = 1e-3
    neuro_epochs_per_gen: int = 2
    neuro_temperature: float = 1.0
    dropout: float = 0.0
    bayes_smoothing: float = 1.0
    quality_fraction: float = 0.75
    probe_per_generator: int = 4
    selector_epsilon: float = 0.10
    selector_hidden: int = 32
    selector_lr: float = 1e-3
    max_resample_attempts: int = 40


# --------------------------------------------------------------------------- #
# 8. Funções auxiliares
# --------------------------------------------------------------------------- #

def sample_feasible(generator_name, boa_model, neuro_model, inst, cfg, rng):
    for _ in range(cfg.max_resample_attempts):
        if generator_name == "BOA":
            ind = boa_model.sample(rng)
        elif generator_name == "NEURO":
            ind = neuro_sample(
                neuro_model, inst.n_jobs, inst.n_machines, rng,
                temperature=cfg.neuro_temperature,
            )
        else:
            raise ValueError(generator_name)
        if decode_npfs(ind, inst)["feasible"]:
            return ind
    # fallback
    for _ in range(cfg.max_resample_attempts):
        ind = _random_individual(inst, rng)
        if decode_npfs(ind, inst)["feasible"]:
            return ind
    return None


def _random_individual(inst, rng):
    return np.vstack([
        rng.permutation(inst.n_jobs) for _ in range(inst.n_machines)
    ]).astype(np.int32)


def initialize_population(inst, size, rng):
    pop = []
    attempts = 0
    while len(pop) < size and attempts < size * 200:
        attempts += 1
        ind = _random_individual(inst, rng)
        if decode_npfs(ind, inst)["feasible"]:
            pop.append(ind)
    if len(pop) < size:
        raise RuntimeError("Não foi possível gerar população inicial suficiente.")
    return pop


# --------------------------------------------------------------------------- #
# 9. Loop principal (run_method)
# --------------------------------------------------------------------------- #

def run_method(inst, cfg, mode="adaptativo",
               seed=0, objective="makespan", verbose=False):
    """
    Executa o Neuro-BOA em uma instância.

    Parâmetros:
        inst: Instance
        cfg: ConfigNeuroBOA
        mode: str - "boa_puro", "neuro_puro", "boa_duplo", "boa_q_neuro_d",
                     "neuro_q_boa_d", "neuro_duplo", "adaptativo"
        seed: int
        objective: str - "makespan" ou "tardiness"

    Retorna dict com resultados.
    """
    assert mode in {
        "boa_puro", "neuro_puro", "boa_duplo", "boa_q_neuro_d",
        "neuro_q_boa_d", "neuro_duplo", "adaptativo",
    }

    rng = np.random.default_rng(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    pop = initialize_population(inst, cfg.population_size, rng)
    fit = np.asarray([evaluate(x, inst, objective) for x in pop], dtype=np.float64)

    n_elite = max(2, int(round(cfg.population_size * cfg.elite_frac)))
    n_diverse = max(1, int(round(cfg.population_size * cfg.diverse_frac)))

    # Modelos de qualidade
    boa_q = BayesianPermutationModel(inst.n_jobs, inst.n_machines, cfg.bayes_smoothing)
    neuro_q = NeuroPermutationMLP(inst.n_jobs, cfg.neuro_hidden, cfg.dropout).to(DEVICE)

    # Modelos de diversidade
    boa_d = BayesianPermutationModel(inst.n_jobs, inst.n_machines, cfg.bayes_smoothing)
    neuro_d = NeuroPermutationMLP(inst.n_jobs, cfg.neuro_hidden, cfg.dropout).to(DEVICE)

    # Meta-redes
    quality_state_dim = 9
    diversity_state_dim = 13
    selector_q = OnlineSelector(quality_state_dim, cfg.selector_hidden, cfg.selector_lr)
    selector_d = OnlineSelector(diversity_state_dim, cfg.selector_hidden, cfg.selector_lr)

    best_hist = []
    records = []
    best_ind = None
    best_value = float("inf")

    for gen in range(cfg.generations):
        order = np.argsort(fit)
        pop = [pop[int(i)] for i in order]
        fit = fit[order]

        if fit[0] < best_value:
            best_value = float(fit[0])
            best_ind = pop[0].copy()

        best_hist.append(best_value)

        elite, diverse_train, diverse_scores = select_diverse_good(
            pop, fit, elite_count=n_elite,
            candidate_frac=cfg.diverse_candidate_frac,
            diverse_count=n_diverse,
        )

        # Treina qualidade
        boa_q.fit(elite)
        train_neuro(neuro_q, elite, inst.n_jobs, inst.n_machines,
                    lr=cfg.neuro_lr, epochs=cfg.neuro_epochs_per_gen)

        # Treina diversidade útil
        div_train = diverse_train if diverse_train else elite
        boa_d.fit(div_train)
        train_neuro(neuro_d, div_train, inst.n_jobs, inst.n_machines,
                    lr=cfg.neuro_lr, epochs=cfg.neuro_epochs_per_gen)

        reference = float(fit[0])

        # Probes
        probes_q = {"BOA": [], "NEURO": []}
        probes_q_fit = {"BOA": [], "NEURO": []}
        probes_d = {"BOA": [], "NEURO": []}
        probes_d_fit = {"BOA": [], "NEURO": []}

        for generator_name in ("BOA", "NEURO"):
            for _ in range(cfg.probe_per_generator):
                iq = sample_feasible(generator_name, boa_q, neuro_q, inst, cfg, rng)
                if iq is not None:
                    probes_q[generator_name].append(iq)
                    probes_q_fit[generator_name].append(evaluate(iq, inst, objective))

                idv = sample_feasible(generator_name, boa_d, neuro_d, inst, cfg, rng)
                if idv is not None:
                    probes_d[generator_name].append(idv)
                    probes_d_fit[generator_name].append(evaluate(idv, inst, objective))

        # Estado de qualidade
        stats_q_boa = generator_stats(probes_q_fit["BOA"], reference)
        stats_q_neuro = generator_stats(probes_q_fit["NEURO"], reference)
        state_q = make_quality_state(best_hist, fit, pop, stats_q_boa, stats_q_neuro, rng)

        score_q_boa = quality_generator_score(probes_q_fit["BOA"], reference)
        score_q_neuro = quality_generator_score(probes_q_fit["NEURO"], reference)
        label_q = 0 if score_q_boa >= score_q_neuro else 1
        selector_q.add(state_q, label_q)
        selector_q.train_steps()

        # Estado de diversidade
        stats_d_boa = diversity_probe_stats(probes_d["BOA"], probes_d_fit["BOA"], elite, reference)
        stats_d_neuro = diversity_probe_stats(probes_d["NEURO"], probes_d_fit["NEURO"], elite, reference)
        state_d = make_diversity_state(best_hist, fit, pop, stats_d_boa, stats_d_neuro, rng)

        score_d_boa = stats_d_boa["useful_score"]
        score_d_neuro = stats_d_neuro["useful_score"]
        label_d = 0 if score_d_boa >= score_d_neuro else 1
        selector_d.add(state_d, label_d)
        selector_d.train_steps()

        # Escolhas
        if mode == "boa_puro":
            choice_q, choice_d, diversity_branch = 0, 0, False
        elif mode == "neuro_puro":
            choice_q, choice_d, diversity_branch = 1, 1, False
        elif mode == "boa_duplo":
            choice_q, choice_d, diversity_branch = 0, 0, True
        elif mode == "boa_q_neuro_d":
            choice_q, choice_d, diversity_branch = 0, 1, True
        elif mode == "neuro_q_boa_d":
            choice_q, choice_d, diversity_branch = 1, 0, True
        elif mode == "neuro_duplo":
            choice_q, choice_d, diversity_branch = 1, 1, True
        else:  # adaptativo
            choice_q = selector_q.choose(state_q, rng, cfg.selector_epsilon)
            choice_d = selector_d.choose(state_d, rng, cfg.selector_epsilon)
            diversity_branch = True

        name_q = "BOA" if choice_q == 0 else "NEURO"
        name_d = "BOA" if choice_d == 0 else "NEURO"

        # Nova população
        next_pop = [e.copy() for e in elite]
        remaining = cfg.population_size - len(next_pop)

        if diversity_branch:
            n_quality = int(round(remaining * cfg.quality_fraction))
            n_div = remaining - n_quality
        else:
            n_quality, n_div = remaining, 0

        for _ in range(n_quality):
            child = sample_feasible(name_q, boa_q, neuro_q, inst, cfg, rng)
            if child is not None:
                next_pop.append(child)

        for _ in range(n_div):
            child = sample_feasible(name_d, boa_d, neuro_d, inst, cfg, rng)
            if child is not None:
                next_pop.append(child)

        while len(next_pop) < cfg.population_size:
            child = _random_individual(inst, rng)
            if decode_npfs(child, inst)["feasible"]:
                next_pop.append(child)

        next_pop = next_pop[:cfg.population_size]
        next_fit = np.asarray([evaluate(x, inst, objective) for x in next_pop], dtype=np.float64)

        records.append({
            "generation": gen,
            "best": float(fit[0]),
            "mean": float(np.mean(fit)),
            "variance": float(np.var(fit)),
            "diversity": population_diversity(pop, 60, rng),
            "quality_choice": name_q,
            "diversity_choice": name_d if diversity_branch else "OFF",
            "q_score_boa": score_q_boa,
            "q_score_neuro": score_q_neuro,
            "d_score_boa": score_d_boa,
            "d_score_neuro": score_d_neuro,
            "d_success_boa": stats_d_boa["success"],
            "d_success_neuro": stats_d_neuro["success"],
            "d_mean_improvement_boa": stats_d_boa["mean_improvement"],
            "d_mean_improvement_neuro": stats_d_neuro["mean_improvement"],
            "d_mean_distance_boa": stats_d_boa["mean_distance"],
            "d_mean_distance_neuro": stats_d_neuro["mean_distance"],
        })

        pop, fit = next_pop, next_fit

        if verbose and (gen % 10 == 0 or gen == cfg.generations - 1):
            print(
                f"[{mode}] gen={gen:03d} best={best_value:.1f} "
                f"Q={name_q} D={records[-1]['diversity_choice']}"
            )

    final = decode_npfs(best_ind, inst)

    return {
        "instance": inst.name,
        "group": inst.group,
        "mode": mode,
        "seed": seed,
        "best_objective": best_value,
        "best_makespan": final["makespan"],
        "best_total_tardiness": final["total_tardiness"],
        "best_individual": best_ind,
        "history": pd.DataFrame(records),
    }


# --------------------------------------------------------------------------- #
# 10. Optuna para Neuro-BOA
# --------------------------------------------------------------------------- #

def construir_objetivo_neuroboa(instancias, n_gen=30):
    """Cria função-objetivo para Optuna do Neuro-BOA."""

    def objetivo(trial):
        cfg = ConfigNeuroBOA(
            population_size=trial.suggest_categorical("population_size", [30, 40, 60]),
            generations=n_gen,
            elite_frac=trial.suggest_float("elite_frac", 0.10, 0.25),
            diverse_candidate_frac=trial.suggest_float("diverse_candidate_frac", 0.20, 0.50),
            diverse_frac=trial.suggest_float("diverse_frac", 0.10, 0.30),
            neuro_hidden=trial.suggest_categorical("neuro_hidden", [64, 128, 256]),
            neuro_epochs_per_gen=trial.suggest_int("neuro_epochs_per_gen", 1, 3),
            neuro_temperature=trial.suggest_float("neuro_temperature", 0.5, 1.8),
            dropout=trial.suggest_float("dropout", 0.0, 0.25),
            bayes_smoothing=trial.suggest_float("bayes_smoothing", 0.1, 3.0, log=True),
            quality_fraction=trial.suggest_float("quality_fraction", 0.60, 0.90),
            probe_per_generator=2,
            selector_epsilon=trial.suggest_float("selector_epsilon", 0.05, 0.20),
            selector_hidden=trial.suggest_categorical("selector_hidden", [16, 32, 64]),
        )

        scores = []
        for k, inst in enumerate(instancias):
            r = run_method(inst, cfg, mode="adaptativo", seed=1000 + k, verbose=False)
            scores.append(r["best_makespan"])

        return float(np.mean(scores))

    return objetivo


def rodar_optuna_neuroboa(instancias, n_trials=20, n_gen=30):
    """Executa o Optuna para o Neuro-BOA."""
    import optuna

    study = optuna.create_study(direction="minimize")
    study.optimize(construir_objetivo_neuroboa(instancias, n_gen), n_trials=n_trials)
    return study, study.best_params