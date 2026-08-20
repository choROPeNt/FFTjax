#!/usr/bin/env bash
set -euo pipefail

# Site-/account-specific settings -- not committed, see slurm/.env.example
ENV_FILE="$(dirname "${BASH_SOURCE[0]}")/.env"
if [[ ! -f "$ENV_FILE" ]]; then
    echo "Missing $ENV_FILE -- copy slurm/.env.example to slurm/.env and fill in your own values." >&2
    exit 1
fi
source "$ENV_FILE"
: "${ACCOUNT:?Set ACCOUNT in $ENV_FILE}"
: "${PROJECT_DIR:?Set PROJECT_DIR in $ENV_FILE}"

PARTITION="${PARTITION:-capella}"
TIME="${TIME:-06:00:00}"
CPUS="${CPUS:-6}"
MEM_PER_CPU="${MEM_PER_CPU:-10G}"
GPUS="${GPUS:-1}"
NODES="${NODES:-1}"
NTASKS="${NTASKS:-1}"
VENV_PATH="${VENV_PATH:-.venv}"

srun \
  -p "$PARTITION" \
  -N "$NODES" \
  -n "$NTASKS" \
  --gres="gpu:${GPUS}" \
  --gpus-per-task="$GPUS" \
  -c "$CPUS" \
  --mem-per-cpu="$MEM_PER_CPU" \
  -t "$TIME" \
  --account="$ACCOUNT" \
  --pty bash -lc "
    cd '$PROJECT_DIR'

    ml release/2026  GCC/14.3.0 Python/3.13.5 OpenMPI/5.0.8 CUDA/13.2.0
    source '$VENV_PATH/bin/activate'

    echo '--- Capella interactive session ready ---'
    echo Host: \$(hostname)
    echo Project: \$(pwd)
    echo Python: \$(command -v python)

    echo 'CUDA:'
    nvidia-smi || true

    exec bash -i
  "