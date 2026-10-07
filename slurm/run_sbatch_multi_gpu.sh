#!/usr/bin/env bash
#
# Real multi-GPU run of the pmap domain-decomposition work (operators/
# fft_distributed.py, Gamma0Operator/solve_lippmann_schwinger/solve_mechanics's
# n_devices auto-detection, cg_solve_pmap) -- exercises it on ACTUAL GPUs,
# not the XLA_FLAGS-simulated device count used for local development (see
# test/test_operators_fft_distributed.py's own docstring for that pattern).
#
# #SBATCH directives can't read slurm/.env (it's gitignored/user-specific),
# so pass --account on submission, e.g.:
#   sbatch --account="$(grep -oP '(?<=ACCOUNT=").*(?=")' slurm/.env)" slurm/run_sbatch_multi_gpu.sh
#SBATCH -p capella
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:2
#SBATCH --gpus-per-task=2
#SBATCH -c 12
#SBATCH --mem-per-cpu=4G
#SBATCH -t 03:00:00
#SBATCH -J fftjax-multigpu
#SBATCH -o out/slurm-multigpu-%j.out

set -euo pipefail

# Site-/account-specific settings -- not committed, see slurm/.env.example
# Under sbatch the script runs from a copy in /var/spool/slurmd/, so
# BASH_SOURCE doesn't point at slurm/ -- resolve from the submit dir instead
# (submit from the repo root, e.g. `sbatch slurm/run_sbatch_multi_gpu.sh`).
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    ENV_FILE="$SLURM_SUBMIT_DIR/slurm/.env"
else
    ENV_FILE="$(dirname "${BASH_SOURCE[0]}")/.env"
fi
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

echo '--- Capella multi-GPU batch job starting ---'
echo Host: "$(hostname)"
echo Project: "$(pwd)"
echo Python: "$(command -v python)"
echo 'CUDA:'
nvidia-smi || true

# The distributed test files size their own synthetic grid and skip-guard
# from DEVICES (same env var they use for local XLA_FLAGS-simulated
# testing) -- on real GPU hardware, the XLA_FLAGS CPU-device-count trick
# those files also set is simply ignored (that flag only affects the "host"
# CPU platform), so jax.local_device_count() reports the real 2 GPUs
# regardless; DEVICES=2 here only makes each test's own bookkeeping
# match the real allocation.
export DEVICES=2

echo '--- jax device visibility ---'
python -c "
import sys; sys.path.insert(0, 'src')
import utils.precision  # noqa: F401
import jax
print('backend:', jax.default_backend())
print('devices:', jax.devices())
print('local_device_count:', jax.local_device_count())
"

# echo '--- distributed correctness suite (real 2-GPU run) ---'
# python -m pytest \
#     test/test_operators_fft_distributed.py \
#     test/test_operators_projection_distributed.py \
#     test/test_problems_mechanics_distributed.py \
#     test/test_problems_mechanics_nonlinear_distributed.py \
#     test/test_solvers_elliptic_scalar_distributed.py \
#     test/test_solvers_krylov_cg_sharded.py \
#     -v

# Uncomment to also benchmark real geometry with n_devices auto-detecting
# the 2 GPUs above (needs --data-dir pointing at .vtu data on this node):
python benchmark/benchmark_3/elastic_solve.py --data-dir data/benchmark_3_

exit 0
