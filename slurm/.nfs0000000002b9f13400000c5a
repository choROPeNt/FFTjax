#!/usr/bin/env bash
set -euo pipefail

PARTITION="capella"
ACCOUNT="p_haftfaeden"
TIME="06:00:00"
CPUS=6
MEM_PER_CPU="10G"
GPUS=1
NODES=1
NTASKS=1

PROJECT_DIR="$HOME/projects/alpha-capella/FFTjax"
VENV_PATH=".venv"

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

    ml release/24.10 GCC/13.3.0 Python/3.12.3 CUDA/12.8.0
    source '$VENV_PATH/bin/activate'

    echo '--- Capella interactive session ready ---'
    echo Host: \$(hostname)
    echo Project: \$(pwd)
    echo Python: \$(command -v python)

    echo 'CUDA:'
    nvidia-smi || true

    exec bash -i
  "