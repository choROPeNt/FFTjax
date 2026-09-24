#!/usr/bin/env bash
#
# #SBATCH directives can't read slurm/.env (it's gitignored/user-specific),
# so pass --account on submission, e.g.:
#   sbatch --account="$(grep -oP '(?<=ACCOUNT=").*(?=")' slurm/.env)" slurm/run_sbatch.sh
#SBATCH -p capella
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH -c 6
#SBATCH --mem-per-cpu=10G
#SBATCH -t 06:00:00
#SBATCH -J fftjax
#SBATCH -o slurm-%j.out

set -euo pipefail

# Site-/account-specific settings -- not committed, see slurm/.env.example
ENV_FILE="$(dirname "${BASH_SOURCE[0]}")/.env"
if [[ ! -f "$ENV_FILE" ]]; then
    echo "Missing $ENV_FILE -- copy slurm/.env.example to slurm/.env and fill in your own values." >&2
    exit 1
fi
source "$ENV_FILE"
: "${PROJECT_DIR:?Set PROJECT_DIR in $ENV_FILE}"

VENV_PATH="${VENV_PATH:-.venv}"

cd "$PROJECT_DIR"

ml release/2026 GCC/14.3.0 Python/3.13.5 OpenMPI/5.0.8 CUDA/13.2.0
source "$VENV_PATH/bin/activate"

echo '--- Capella batch job starting ---'
echo Host: "$(hostname)"
echo Project: "$(pwd)"
echo Python: "$(command -v python)"

# python benchmark/benchmark_3/elastic_solve_vtu.py

exit 0
