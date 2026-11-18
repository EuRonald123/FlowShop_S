import os

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import argparse
import csv
import json
import signal
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.flowshop_engine import FlowShopEngine
from src.solvers.adapter import (
    rodar_brkga_rsc,
    rodar_brkga_scs_dmn,
    rodar_brkga_scs_img,
    rodar_brkga_scs_vm,
    rodar_brkga_scs_vr,
)

DATA_DIR = ROOT / "data" / "taillard"
RESULTADOS_DIR = ROOT / "resultados" / "experimentos_brkga"
OPTUNA_DIR = ROOT / "resultados" / "optuna"

CAMPOS_CSV = [
    "instancia",
    "grupo",
    "n_jobs",
    "n_machines",
    "metodo",
    "seed",
    "makespan",
    "tardiness",
    "cfo_count",
    "tempo_s",
    "lower_bound",
    "bks",
    "rpd",
]

# 12 instâncias representativas (uma por classe de dimensão do benchmark Taillard).
INSTANCIAS_POUCAS = [
    "ta001.txt", "ta011.txt", "ta021.txt",
    "ta031.txt", "ta041.txt", "ta051.txt",
    "ta061.txt", "ta071.txt", "ta081.txt",
    "ta091.txt", "ta101.txt", "ta111.txt",
]

# Registro declarativo das variantes: facilita adicionar ou ajustar métodos em um único local.
VARIANTES: Dict[str, Dict[str, Any]] = {
    "brkga_rsc": {
        "solver": rodar_brkga_rsc,
        "subdir": "brkga_rsc",
        "params": {},
        "optuna_file": None,
    },
    "brkga_scs_vm": {
        "solver": rodar_brkga_scs_vm,
        "subdir": "brkga_scs_vm",
        "params": {
            "tam_pop": 100, "pe": 0.2, "pm": 0.15, "rho": 0.7,
            "z_limiar": 1.5, "votos_limiar": 5, "num_clusters_max": 5,
            "r_min": 0.05, "r_max": 0.25, "base_cooldown": 5, "gamma": 1.5,
        },
        "optuna_file": None,
    },
    "brkga_scs_vr": {
        "solver": rodar_brkga_scs_vr,
        "subdir": "brkga_scs_vr",
        "params": {
            "tam_pop": 100, "pe": 0.15, "pm": 0.25, "rho": 0.85,
            "raio_cluster": 0.5, "z_limiar": 1.645, "votos_limiar": 5,
            "max_hj_evals": 600,
        },
        "optuna_file": None,
    },
    "brkga_scs_img": {
        "solver": rodar_brkga_scs_img,
        "subdir": "brkga_scs_img",
        "params": {},
        "optuna_file": None,
    },
    "brkga_scs_dmn": {
        "solver": rodar_brkga_scs_dmn,
        "subdir": "brkga_scs_dmn",
        "params": {},
        "optuna_file": None,
    },
    "brkga_scs_vm_optuna": {
        "solver": rodar_brkga_scs_vm,
        "subdir": "brkga_scs_vm",
        "params": {},
        "optuna_file": "best_params_brkga_scs_vm.json",
    },
    "brkga_scs_vr_optuna": {
        "solver": rodar_brkga_scs_vr,
        "subdir": "brkga_scs_vr",
        "params": {},
        "optuna_file": "best_params_brkga_scs_vr.json",
    },
}

_interrupcao_solicitada = False


def _tratar_sinal_interrupcao(signum: int, _frame: Any) -> None:
    global _interrupcao_solicitada
    nome = signal.Signals(signum).name
    if not _interrupcao_solicitada:
        _interrupcao_solicitada = True
        print(
            f"\n[AVISO] Sinal {nome} recebido. Nenhuma nova tarefa será iniciada; "
            "aguardando conclusão das execuções ativas.",
            flush=True,
        )
    else:
        print(f"\n[AVISO] Segundo sinal {nome} recebido. Encerrando imediatamente.", flush=True)
        raise KeyboardInterrupt


def carregar_parametros_variante(metodo: str) -> Dict[str, Any]:
    """Retorna os hiperparâmetros fixos ou carregados do JSON do Optuna para o método."""
    spec = VARIANTES[metodo]
    params = dict(spec["params"])
    if spec["optuna_file"]:
        json_path = OPTUNA_DIR / spec["optuna_file"]
        with open(json_path, "r", encoding="utf-8") as f:
            params.update(json.load(f)["best_params"])
    return params


def construir_catalogo(
    data_dir: Path,
    modo: str,
    instancias_especificas: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Lê os cabeçalhos das instâncias Taillard e retorna o catálogo ordenado."""
    arquivos = sorted(data_dir.glob("ta*.txt"))
    if not arquivos:
        raise FileNotFoundError(f"Nenhuma instância ta*.txt encontrada em {data_dir}")

    if instancias_especificas:
        alvo = {n if n.endswith(".txt") else f"{n}.txt" for n in instancias_especificas}
        arquivos = [f for f in arquivos if f.name in alvo]
        faltantes = sorted(alvo - {f.name for f in arquivos})
        if faltantes:
            raise FileNotFoundError(f"Instâncias não encontradas: {faltantes}")
    elif modo == "poucas":
        alvo = set(INSTANCIAS_POUCAS)
        arquivos = [f for f in arquivos if f.name in alvo]

    registros = []
    for f in arquivos:
        with open(f, "r", encoding="utf-8") as fp:
            linhas = [ln.strip() for ln in fp if ln.strip()]
        partes = linhas[1].split()
        registros.append({
            "arquivo": f.name,
            "id": int(f.stem.replace("ta", "")),
            "n_jobs": int(partes[0]),
            "n_machines": int(partes[1]),
            "bks_header": int(partes[3]),
            "path": str(f),
        })

    return (
        pd.DataFrame(registros)
        .sort_values(["n_jobs", "n_machines", "id"])
        .reset_index(drop=True)
    )


def resolver_csv_saida(
    metodo: str,
    modo: str,
    output_custom: Optional[str],
    instancias_especificas: Optional[List[str]],
) -> Path:
    """Resolve o caminho padrão de saída ou um caminho customizado relativo à raiz."""
    if output_custom:
        p = Path(output_custom)
        return p if p.is_absolute() else (ROOT / p)

    pasta = RESULTADOS_DIR / VARIANTES[metodo]["subdir"]
    if instancias_especificas:
        nome = f"results_{metodo}_custom_taillard.csv"
    elif modo == "poucas":
        nome = f"results_{metodo}_poucas_instancias_taillard.csv"
    else:
        nome = f"results_{metodo}_taillard.csv"
    return pasta / nome


def resolver_workers(workers_arg: str) -> int:
    """Resolve o número de processos paralelos ('auto' usa $PBS_NP quando disponível)."""
    if workers_arg.strip().lower() == "auto":
        pbs_np = os.environ.get("PBS_NP", "")
        return int(pbs_np) if pbs_np.isdigit() and int(pbs_np) >= 1 else 1
    n = int(workers_arg)
    if n < 1:
        raise ValueError("--workers deve ser >= 1")
    return n


def carregar_concluidos(csv_path: Path) -> Set[Tuple[str, str, int]]:
    """Retorna o conjunto (instancia, metodo, seed) já concluído com makespan válido."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return set()

    df = pd.read_csv(csv_path)
    df["makespan"] = pd.to_numeric(df["makespan"], errors="coerce")
    df_valido = df.dropna(subset=["instancia", "metodo", "seed", "makespan"])
    return {
        (str(row.instancia), str(row.metodo), int(row.seed))
        for row in df_valido.itertuples(index=False)
    }


def salvar_linha_csv(csv_path: Path, linha: Dict[str, Any]) -> None:
    """Persiste uma linha no CSV com flush e fsync para evitar perda em falhas ou corte de fila."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    arquivo_novo = not csv_path.exists() or csv_path.stat().st_size == 0

    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CAMPOS_CSV)
        if arquivo_novo:
            writer.writeheader()
        writer.writerow(linha)
        f.flush()
        os.fsync(f.fileno())


def ordenar_csv(csv_path: Path) -> None:
    """Remove eventuais duplicatas e reordena o CSV de forma atômica."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return

    df = pd.read_csv(csv_path)
    df["makespan"] = pd.to_numeric(df["makespan"], errors="coerce")
    df = (
        df.dropna(subset=["instancia", "metodo", "seed", "makespan"])
        .drop_duplicates(subset=["instancia", "metodo", "seed"], keep="last")
        .sort_values(["n_jobs", "n_machines", "instancia", "metodo", "seed"])
    )

    tmp_path = csv_path.with_suffix(".csv.tmp")
    df[CAMPOS_CSV].to_csv(tmp_path, index=False)
    os.replace(tmp_path, csv_path)


def registrar_erro(log_path: Path, tarefa: Dict[str, Any], tb: str) -> None:
    """Grava stack trace de falhas em arquivo separado sem poluir o CSV principal."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(
            f"[{ts}] instancia={tarefa['instancia']} metodo={tarefa['metodo']} "
            f"seed={tarefa['seed']}\n{tb}\n{'-' * 72}\n"
        )
        f.flush()
        os.fsync(f.fileno())


def _inicializar_worker() -> None:
    """Configura processos filhos para ignorar SIGINT e respeitar 1 thread por núcleo."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[var] = "1"


def executar_tarefa(tarefa: Dict[str, Any]) -> Dict[str, Any]:
    """Executa uma combinação (instância, método, seed) em processo isolado."""
    t0 = time.perf_counter()
    try:
        nj, nm, pt, dd = FlowShopEngine.carregar_instancia_taillard(tarefa["inst_path"])
        engine = FlowShopEngine(nj, nm, pt, dd)
        solver_fn = VARIANTES[tarefa["metodo"]]["solver"]

        result = solver_fn(
            engine,
            seed=tarefa["seed"],
            objective=tarefa["objective"],
            max_evals=tarefa["max_evals"],
            t_param=tarefa["t_param"],
            **tarefa["params"],
        )
        elapsed = time.perf_counter() - t0

        bks = tarefa["bks"]
        rpd = round(((float(result.best_cost) - bks) / bks) * 100.0, 3) if bks else float("nan")

        linha = {
            "instancia": tarefa["instancia"],
            "grupo": "Taillard",
            "n_jobs": tarefa["n_jobs"],
            "n_machines": tarefa["n_machines"],
            "metodo": tarefa["metodo"],
            "seed": tarefa["seed"],
            "makespan": float(result.best_cost),
            "tardiness": float(result.tardiness),
            "cfo_count": int(result.n_evaluations),
            "tempo_s": round(elapsed, 3),
            "lower_bound": float(result.lower_bound),
            "bks": bks,
            "rpd": rpd,
        }
        return {"ok": True, "tarefa": tarefa, "linha": linha, "elapsed": elapsed}

    except Exception:
        return {
            "ok": False,
            "tarefa": tarefa,
            "erro_tb": traceback.format_exc(),
            "elapsed": time.perf_counter() - t0,
        }


def exibir_resumo(csv_path: Path, metodo: str) -> None:
    """Imprime métricas agregadas das execuções válidas registradas no CSV."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return

    df = pd.read_csv(csv_path)
    df = df[df["metodo"] == metodo].dropna(subset=["makespan"])
    if df.empty:
        return

    print("-" * 78, flush=True)
    print(f"RESUMO ACUMULADO ({csv_path.name}):", flush=True)
    print(
        f"  Método: {metodo} | Execuções: {len(df)} ({df['instancia'].nunique()} instâncias)",
        flush=True,
    )
    print(
        f"  Makespan médio: {df['makespan'].mean():.1f} | "
        f"RPD médio: {df['rpd'].mean():.3f}% | "
        f"Tempo médio/exec: {df['tempo_s'].mean():.2f}s",
        flush=True,
    )
    print("-" * 78, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Executa experimentos BRKGA-SCS / BRKGA-RSC no benchmark Taillard.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--metodo", required=True, choices=list(VARIANTES.keys()))
    parser.add_argument("--catalogo", choices=["poucas", "completo"], default="poucas")
    parser.add_argument("--instancias", nargs="+", default=None, help="Instâncias específicas (sobrescreve --catalogo).")
    parser.add_argument("--workers", default="auto", help="Processos paralelos ou 'auto' ($PBS_NP).")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--seed-inicio", type=int, default=100)
    parser.add_argument("--t-param", type=float, default=60.0, help="Critério de tempo; <= 0 usa --max-evals.")
    parser.add_argument("--max-evals", type=int, default=50000)
    parser.add_argument("--objective", choices=["Makespan", "Tardiness"], default="Makespan")
    parser.add_argument("--output", default=None, help="Caminho customizado do CSV de saída.")
    parser.add_argument("--dry-run", action="store_true", help="Exibe o plano sem executar.")
    return parser.parse_args()


def iterar_resultados(tarefas: List[Dict[str, Any]], workers: int):
    """Executa as tarefas (sequencialmente ou em pool) e produz resultados conforme concluem."""
    if workers == 1:
        for tarefa in tarefas:
            if _interrupcao_solicitada:
                break
            yield executar_tarefa(tarefa)
        return

    fila = iter(tarefas)
    em_execucao = {}
    with ProcessPoolExecutor(max_workers=workers, initializer=_inicializar_worker) as pool:
        for _ in range(min(workers, len(tarefas))):
            tarefa = next(fila, None)
            if tarefa is not None:
                em_execucao[pool.submit(executar_tarefa, tarefa)] = tarefa

        while em_execucao:
            prontas, _ = wait(em_execucao.keys(), timeout=1.0, return_when=FIRST_COMPLETED)
            for fut in prontas:
                tarefa_orig = em_execucao.pop(fut)
                try:
                    yield fut.result()
                except Exception:
                    yield {
                        "ok": False,
                        "tarefa": tarefa_orig,
                        "erro_tb": traceback.format_exc(),
                        "elapsed": 0.0,
                    }

                if not _interrupcao_solicitada:
                    proxima = next(fila, None)
                    if proxima is not None:
                        em_execucao[pool.submit(executar_tarefa, proxima)] = proxima


def main() -> int:
    args = parse_args()
    workers = resolver_workers(args.workers)
    t_param = args.t_param if args.t_param > 0 else None
    params = carregar_parametros_variante(args.metodo)

    catalogo = construir_catalogo(DATA_DIR, args.catalogo, args.instancias)
    csv_path = resolver_csv_saida(args.metodo, args.catalogo, args.output, args.instancias)
    error_log_path = csv_path.parent / f"errors_{args.metodo}.log"

    bks_path = DATA_DIR / "taillard_bks.json"
    bks_db = json.loads(bks_path.read_text(encoding="utf-8")) if bks_path.exists() else {}

    concluidos = carregar_concluidos(csv_path)
    seeds = list(range(args.seed_inicio, args.seed_inicio + args.seeds))
    total_planejado = len(catalogo) * len(seeds)

    tarefas = [
        {
            "instancia": row.arquivo,
            "inst_path": row.path,
            "n_jobs": int(row.n_jobs),
            "n_machines": int(row.n_machines),
            "metodo": args.metodo,
            "seed": seed,
            "objective": args.objective,
            "max_evals": args.max_evals,
            "t_param": t_param,
            "params": params,
            "bks": bks_db.get(row.arquivo.replace(".txt", ""), int(row.bks_header)),
        }
        for row in catalogo.itertuples(index=False)
        for seed in seeds
        if (row.arquivo, args.metodo, seed) not in concluidos
    ]

    ja_concluidas = total_planejado - len(tarefas)

    print("=" * 78, flush=True)
    print(f"Método:    {args.metodo} | Catálogo: {'custom' if args.instancias else args.catalogo}", flush=True)
    print(f"CSV:       {csv_path}", flush=True)
    print(f"Execuções: {ja_concluidas} concluídas | {len(tarefas)} pendentes (total: {total_planejado})", flush=True)
    print(f"Workers:   {workers} | Parada: {'t_param=' + str(t_param) if t_param else 'max_evals=' + str(args.max_evals)}", flush=True)
    print("=" * 78, flush=True)

    if args.dry_run or not tarefas:
        if not tarefas:
            print("Todas as execuções planejadas já estão concluídas.", flush=True)
            exibir_resumo(csv_path, args.metodo)
        return 0

    signal.signal(signal.SIGTERM, _tratar_sinal_interrupcao)
    signal.signal(signal.SIGINT, _tratar_sinal_interrupcao)

    t0 = time.perf_counter()
    salvas = 0
    erros = 0

    try:
        for res in iterar_resultados(tarefas, workers):
            t = res["tarefa"]
            atual = ja_concluidas + salvas + erros + 1
            if res["ok"]:
                linha = res["linha"]
                salvar_linha_csv(csv_path, linha)
                salvas += 1
                print(
                    f"[{atual}/{total_planejado}] {t['instancia']} ({t['n_jobs']}x{t['n_machines']}) "
                    f"| seed={t['seed']} | makespan={linha['makespan']:.1f} "
                    f"| rpd={linha['rpd']:.3f}% | cfo={linha['cfo_count']} ({res['elapsed']:.2f}s)",
                    flush=True,
                )
            else:
                erros += 1
                registrar_erro(error_log_path, t, res["erro_tb"])
                msg_erro = res["erro_tb"].strip().splitlines()[-1]
                print(
                    f"[{atual}/{total_planejado}] {t['instancia']} | seed={t['seed']} "
                    f"| ERRO: {msg_erro} (ver {error_log_path.name})",
                    flush=True,
                )
    except KeyboardInterrupt:
        print("\nExecução interrompida. Resultados concluídos foram preservados no CSV.", flush=True)

    if salvas > 0:
        ordenar_csv(csv_path)

    duracao = time.perf_counter() - t0
    print("=" * 78, flush=True)
    print(
        f"Concluído em {duracao:.1f}s ({duracao / 60:.1f} min) | "
        f"Salvas: {salvas} | Erros: {erros} | Restantes: {len(tarefas) - salvas - erros}",
        flush=True,
    )
    print("=" * 78, flush=True)
    exibir_resumo(csv_path, args.metodo)
    return 0 if erros == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
