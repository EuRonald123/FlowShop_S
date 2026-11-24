#!/bin/bash
#PBS -N flowshop_exp
#PBS -l select=1:ncpus=24:mem=16gb
#PBS -l walltime=04:00:00
#PBS -l cput=24:00:00
#PBS -j oe

# Script de submissão PBS — Experimentos Flow Shop Scheduling
#
# Uso padrão:
#   qsub submit_pbs.sh
#
# Sobrescrevendo parâmetros sem editar o arquivo:
#   qsub -v METODO=brkga_rsc,CATALOGO=poucas,SEEDS=1,T_PARAM=5 submit_pbs.sh

set -euo pipefail

# --- Ambiente micromamba ---
MAMBA_BIN="${MAMBA_BIN:-${HOME}/.local/bin/micromamba}"
export MAMBA_ROOT_PREFIX="${HOME}/micromamba"
CONDA_ENV="flowshop"

METODO="${METODO:-brkga_rsc}"
CATALOGO="${CATALOGO:-completo}"
SEEDS="${SEEDS:-5}"
SEED_INICIO="${SEED_INICIO:-100}"
T_PARAM="${T_PARAM:-60}"
WORKERS="${WORKERS:-auto}"

# Vai para o diretório de onde o job foi submetido
cd "${PBS_O_WORKDIR:-$(dirname "$0")}"

PBS_NP="${PBS_NP:-${NCPUS:-1}}"
export PBS_NP

# Ativa o ambiente
if [ ! -x "${MAMBA_BIN}" ]; then
    echo "[ERRO] micromamba não encontrado em: ${MAMBA_BIN}" >&2
    exit 1
fi
set +u
eval "$("${MAMBA_BIN}" shell hook --shell bash)"
micromamba activate "${CONDA_ENV}"
set -u

# 1 thread BLAS/OpenMP por processo worker
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "*-*-*-*-*"
echo "Início:    $(date '+%Y-%m-%d %H:%M:%S')"
echo "Job ID:    ${PBS_JOBID:-local}"
echo "Nó:        $(hostname)"
echo "Núcleos:   ${PBS_NP} (NCPUS=${NCPUS:-?})"
echo "Diretório: $(pwd)"
echo "Python:    $(which python) ($(python --version 2>&1))"
echo "Config:    metodo=${METODO} catalogo=${CATALOGO} seeds=${SEEDS} t_param=${T_PARAM} workers=${WORKERS}"
echo "*-*-*-*-*"

STATUS=0
python -u run_experiments.py \
    --metodo "${METODO}" \
    --catalogo "${CATALOGO}" \
    --seeds "${SEEDS}" \
    --seed-inicio "${SEED_INICIO}" \
    --t-param "${T_PARAM}" \
    --workers "${WORKERS}" || STATUS=$?

echo "*-*-*-*-*"
echo "Término:   $(date '+%Y-%m-%d %H:%M:%S') | Código de saída: ${STATUS}"
echo "*-*-*-*-*"
exit "${STATUS}"
