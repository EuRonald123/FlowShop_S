#!/bin/bash
#PBS -N flowshop_exp
#PBS -l nodes=NOME_DO_NO:ppn=8
#PBS -l walltime=04:00:00
#PBS -j oe
## Descomente abaixo caso o cluster exija informar a fila explicitamente:
##PBS -q NOME_DA_FILA

# =============================================================================
# Script de submissão PBS — Experimentos Flow Shop Scheduling
#
# Ajuste acima:
#   - nodes=NOME_DO_NO:ppn=N  -> nó específico e quantidade de núcleos reservados
#   - walltime=HH:MM:SS       -> limite de tempo permitido pela fila
#
# Uso padrão (editando as variáveis abaixo):
#   qsub submit_pbs.sh
#
# Uso sobrescrevendo método/catálogo sem editar o arquivo:
#   qsub -v METODO=brkga_rsc,CATALOGO=completo submit_pbs.sh
# =============================================================================

set -euo pipefail

# --- Configuração do ambiente Miniconda ---
CONDA_BASE="${HOME}/miniconda3"
CONDA_ENV="flowshop"

# --- Parâmetros do experimento (podem ser sobrescritos via -v no qsub/psub) ---
METODO="${METODO:-brkga_rsc}"
CATALOGO="${CATALOGO:-completo}"
SEEDS="${SEEDS:-5}"
SEED_INICIO="${SEED_INICIO:-100}"
T_PARAM="${T_PARAM:-60}"
WORKERS="${WORKERS:-auto}"

# Vai para o diretório raiz do projeto a partir de onde o job foi submetido
cd "${PBS_O_WORKDIR:-$(dirname "$0")}"

# Garante PBS_NP definido mesmo em versões antigas do PBS/Torque
if [ -z "${PBS_NP:-}" ] && [ -n "${PBS_NODEFILE:-}" ] && [ -f "${PBS_NODEFILE}" ]; then
    export PBS_NP
    PBS_NP="$(wc -l < "${PBS_NODEFILE}")"
fi

# Ativa o ambiente Miniconda em shell não-interativo
if [ ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
    echo "[ERRO] Miniconda não encontrado em: ${CONDA_BASE}" >&2
    exit 1
fi
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# Mantém 1 thread BLAS/OpenMP por processo worker
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "=============================================================================="
echo "Início:    $(date '+%Y-%m-%d %H:%M:%S')"
echo "Job ID:    ${PBS_JOBID:-local}"
echo "Nó:        $(hostname)"
echo "Núcleos:   ${PBS_NP:-1}"
echo "Diretório: $(pwd)"
echo "Python:    $(which python)"
echo "Config:    metodo=${METODO} catalogo=${CATALOGO} seeds=${SEEDS} workers=${WORKERS}"
echo "=============================================================================="

STATUS=0
python -u run_experiments.py \
    --metodo "${METODO}" \
    --catalogo "${CATALOGO}" \
    --seeds "${SEEDS}" \
    --seed-inicio "${SEED_INICIO}" \
    --t-param "${T_PARAM}" \
    --workers "${WORKERS}" || STATUS=$?

echo "=============================================================================="
echo "Término:   $(date '+%Y-%m-%d %H:%M:%S') | Código de saída: ${STATUS}"
echo "=============================================================================="
exit "${STATUS}"
