import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

import numpy as np
import optuna

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.flowshop_engine import FlowShopEngine
from src.solvers.adapter import rodar_brkga_scs_vm, rodar_brkga_scs_vr


# =====================================================================
# 1. SELEÇÃO DE INSTÂNCIAS DE TREINO (1 POR NÍVEL / DIMENSÃO)
# =====================================================================
def selecionar_instancias_por_nivel(
    data_dir: Union[str, Path],
    instancias_especificas: Optional[List[str]] = None,
    max_jobs: Optional[int] = None,
) -> List[Dict[str, Any]]:

    data_path = Path(data_dir)
    bks_file = data_path / "taillard_bks.json"
    bks_db = {}
    if bks_file.exists():
        with open(bks_file, "r") as f:
            bks_db = json.load(f)

    arquivos = sorted(data_path.glob("ta*.txt"))
    if not arquivos:
        raise FileNotFoundError(f"Nenhuma instância 'ta*.txt' encontrada em {data_path}")

    # Se o usuário passou uma lista específica (ex: ['ta001.txt', 'ta011.txt', 'ta031.txt'])
    if instancias_especificas is not None:
        nomes_alvo = {
            nome if nome.endswith(".txt") else f"{nome}.txt"
            for nome in instancias_especificas
        }
        arquivos = [f for f in arquivos if f.name in nomes_alvo]

    niveis_vistos = set()
    selecionadas = []

    for f in arquivos:
        with open(f, "r") as file:
            linhas = [l.strip() for l in file if l.strip()]
            partes = linhas[1].split()
            n_jobs = int(partes[0])
            n_machines = int(partes[1])
            bks_header = int(partes[3])

        if instancias_especificas is None and max_jobs is not None and n_jobs > max_jobs:
            continue

        nivel = (n_jobs, n_machines)
        # Se não passou lista específica, pega apenas a 1ª instância de cada nível (n_jobs, n_machines)
        if instancias_especificas is None and nivel in niveis_vistos:
            continue

        niveis_vistos.add(nivel)
        bks = bks_db.get(f.stem, bks_header)

        nj, nm, pt, dd = FlowShopEngine.carregar_instancia_taillard(str(f))
        engine = FlowShopEngine(nj, nm, pt, dd)

        selecionadas.append({
            "arquivo": f.name,
            "path": str(f),
            "n_jobs": n_jobs,
            "n_machines": n_machines,
            "bks": bks,
            "engine": engine,
        })

    return selecionadas


#Espaço de busca do Optuna para cada variante do BRKGA-SCS
def sugerir_parametros_optuna(trial: optuna.Trial, solver: str) -> Dict[str, Any]:
    solver_key = solver.lower().strip()

    if solver_key in ["brkga_scs_vr", "vr"]:
        return {
            "tam_pop": trial.suggest_int("tam_pop", 50, 200, step=25),
            "pe": trial.suggest_float("pe", 0.10, 0.30, step=0.05),
            "pm": trial.suggest_float("pm", 0.10, 0.30, step=0.05),
            "rho": trial.suggest_float("rho", 0.60, 0.90, step=0.05),
            "raio_cluster": trial.suggest_float("raio_cluster", 0.15, 0.80, step=0.05),
            "z_limiar": trial.suggest_float("z_limiar", 1.0, 2.5, step=0.15),
            "votos_limiar": trial.suggest_int("votos_limiar", 2, 10),
            "step_init": trial.suggest_float("step_init", 0.05, 0.25, step=0.05),
            "step_reduction": trial.suggest_float("step_reduction", 0.3, 0.7, step=0.1),
            "max_hj_evals": trial.suggest_int("max_hj_evals", 200, 1000, step=100),
        }

    elif solver_key in ["brkga_scs_vm", "vm", "brkga_scs"]:
        return {
            "tam_pop": trial.suggest_int("tam_pop", 50, 200, step=25),
            "pe": trial.suggest_float("pe", 0.10, 0.30, step=0.05),
            "pm": trial.suggest_float("pm", 0.10, 0.30, step=0.05),
            "rho": trial.suggest_float("rho", 0.60, 0.90, step=0.05),
            "num_clusters_max": trial.suggest_int("num_clusters_max", 3, 10),
            "r_min": trial.suggest_float("r_min", 0.02, 0.10, step=0.02),
            "r_max": trial.suggest_float("r_max", 0.15, 0.40, step=0.05),
            "base_cooldown": trial.suggest_int("base_cooldown", 3, 15),
            "gamma": trial.suggest_float("gamma", 1.2, 2.0, step=0.1),
            "z_limiar": trial.suggest_float("z_limiar", 1.0, 2.5, step=0.15),
            "votos_limiar": trial.suggest_int("votos_limiar", 2, 10),
            "step_init": trial.suggest_float("step_init", 0.05, 0.25, step=0.05),
            "step_reduction": trial.suggest_float("step_reduction", 0.3, 0.7, step=0.1),
        }

    else:
        raise ValueError(
            f"Solver '{solver}' inválido para sintonia. Use 'brkga_scs_vr' ou 'brkga_scs_vm'."
        )

#Execucao e salvamento em json
def executar_tuning_optuna(
    solver: str,
    data_dir: Union[str, Path],
    output_json: Union[str, Path],
    instancias_especificas: Optional[List[str]] = None,
    max_jobs_treino: Optional[int] = 100,
    n_trials: int = 30,
    seeds: Optional[List[int]] = None,
    t_param: Optional[float] = 30.0,
    max_evals: int = 20000,
    objective: str = "Makespan",
    seed_optuna: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Executa o Optuna minimizando o RPD médio nas instâncias selecionadas
    e salva o resultado em `output_json`.

    Retorna o dicionário `best_params` pronto para ser desempacotado (`**best_params`)
    nas funções `rodar_brkga_scs_vr` ou `rodar_brkga_scs_vm`.
    """
    if seeds is None:
        seeds = [42]

    instancias = selecionar_instancias_por_nivel(
        data_dir=data_dir,
        instancias_especificas=instancias_especificas,
        max_jobs=max_jobs_treino,
    )

    if not instancias:
        raise ValueError("Nenhuma instância foi selecionada para a sintonia.")

    solver_key = solver.lower().strip()
    fn_solver = (
        rodar_brkga_scs_vr
        if solver_key in ["brkga_scs_vr", "vr"]
        else rodar_brkga_scs_vm
    )

    if verbose:
        print("=" * 72)
        print(f"INICIANDO SINTONIA OPTUNA — Solver: {solver}")
        print(f"Instâncias de treino ({len(instancias)}):")
        for inst in instancias:
            print(f"  - {inst['arquivo']} ({inst['n_jobs']}x{inst['n_machines']}, BKS={inst['bks']})")
        print(f"Trials: {n_trials} | Seeds/instância: {seeds} | t_param: {t_param} | max_evals: {max_evals}")
        print("=" * 72)

    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    best_rpd_so_far = float("inf")

    def _salvar_json(best_rpd: float, best_trial_num: int, best_params_dict: Dict[str, Any], trials_concluidos: int):
        payload = {
            "solver": solver,
            "best_rpd_medio": round(float(best_rpd), 4),
            "best_trial": int(best_trial_num),
            "trials_concluidos": int(trials_concluidos),
            "n_trials_planejados": n_trials,
            "tempo_total_s": round(time.time() - t0, 2),
            "criterio_parada_treino": {
                "t_param": t_param,
                "max_evals": max_evals,
                "seeds": seeds,
            },
            "instancias_treino": [
                f"{i['arquivo']} ({i['n_jobs']}x{i['n_machines']})" for i in instancias
            ],
            "best_params": best_params_dict,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4, ensure_ascii=False)

    def _objective(trial: optuna.Trial) -> float:
        nonlocal best_rpd_so_far
        params_candidatos = sugerir_parametros_optuna(trial, solver_key)
        rpds = []

        for inst in instancias:
            engine = inst["engine"]
            bks = inst["bks"]

            for s in seeds:
                res = fn_solver(
                    engine=engine,
                    seed=s,
                    objective=objective,
                    max_evals=max_evals,
                    t_param=t_param,
                    **params_candidatos,
                )
                rpd = ((res.best_cost - bks) / bks) * 100.0
                rpds.append(rpd)

        rpd_medio = float(np.mean(rpds))
        if verbose:
            print(f"[Trial {trial.number:03d}] RPD Médio: {rpd_medio:.3f}% | Params: {params_candidatos}")

        if rpd_medio < best_rpd_so_far:
            best_rpd_so_far = rpd_medio
            _salvar_json(
                best_rpd=rpd_medio,
                best_trial_num=trial.number,
                best_params_dict=params_candidatos,
                trials_concluidos=trial.number + 1,
            )
            if verbose:
                print(f"  --> [JSON ATUALIZADO] Novo melhor RPD Médio: {rpd_medio:.3f}% salvo em {output_path.name}")

        return rpd_medio

    sampler = optuna.samplers.TPESampler(seed=seed_optuna)
    optuna.logging.set_verbosity(
        optuna.logging.WARNING if verbose else optuna.logging.ERROR
    )

    study = optuna.create_study(direction="minimize", sampler=sampler)
    try:
        study.optimize(_objective, n_trials=n_trials)
    except KeyboardInterrupt:
        if verbose:
            print("\n[AVISO] Sintonia interrompida manualmente pelo usuário! Mantendo o melhor resultado salvo.")

    elapsed = time.time() - t0

    if len(study.trials) > 0 and study.best_trial is not None:
        _salvar_json(
            best_rpd=study.best_value,
            best_trial_num=study.best_trial.number,
            best_params_dict=study.best_params,
            trials_concluidos=len(study.trials),
        )
        if verbose:
            print("=" * 72)
            print(f"SINTONIA CONCLUÍDA em {elapsed:.1f}s ({elapsed/60:.1f} min)")
            print(f"Melhor RPD Médio: {study.best_value:.3f}% (Trial #{study.best_trial.number})")
            print(f"Melhores Parâmetros: {json.dumps(study.best_params, indent=2)}")
            print(f"Salvo em: {output_path}")
            print("=" * 72)
        return study.best_params

    return {}


def carregar_melhores_parametros(caminho_json: Union[str, Path]) -> Dict[str, Any]:
    caminho = Path(caminho_json)
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo de parâmetros não encontrado: {caminho}")

    with open(caminho, "r", encoding="utf-8") as f:
        dados = json.load(f)

    return dados.get("best_params", dados)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sintonia Optuna para BRKGA-SCS (VM e VR)")
    parser.add_argument("--solver", type=str, default="brkga_scs_vr", choices=["brkga_scs_vr", "brkga_scs_vm"])
    parser.add_argument("--data-dir", type=str, default=str(ROOT_DIR / "data" / "taillard"))
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--t-param", type=float, default=30.0, help="Fator de tempo CPU (use 0 para desativar e usar max-evals)")
    parser.add_argument("--max-evals", type=int, default=20000)
    parser.add_argument("--max-jobs", type=int, default=100, help="Limite máximo de jobs das instâncias de treino")
    parser.add_argument("--instancias", nargs="*", default=None, help="Lista opcional de instâncias (ex: ta001.txt ta011.txt)")

    args = parser.parse_args()

    out_file = args.output or str(ROOT_DIR / "resultados" / f"best_params_{args.solver}.json")
    t_val = args.t_param if args.t_param and args.t_param > 0 else None

    executar_tuning_optuna(
        solver=args.solver,
        data_dir=args.data_dir,
        output_json=out_file,
        instancias_especificas=args.instancias,
        max_jobs_treino=args.max_jobs,
        n_trials=args.trials,
        t_param=t_val,
        max_evals=args.max_evals,
    )
