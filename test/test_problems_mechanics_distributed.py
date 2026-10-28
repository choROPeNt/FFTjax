"""
Correctness check for ``problems.mechanics.solve_mechanics``'s automatic
domain decomposition, across every formulation that's been wired for it --
the full CG solve, not just one FFT-heavy operator apply (see
``test_operators_projection_distributed.py`` for that narrower check on
just the Lippmann-Schwinger/Fourier-Galerkin family's shared
``Gamma0Operator`` core).

- "lippmann_schwinger" (pure strain BC) and "fourier_galerkin" (pure strain
  BC) both route through ``operators.projection.Gamma0Operator``, which
  auto-decomposes; no other code in either formulation changed.
- "displacement" (pure strain BC and mixed strain/stress BC, i.e.
  ``control`` nonzero) runs its whole CG under one ``shard_map``
  (``_solve_displacement_based_sharded``), with the DC-bin macroscopic-strain
  injection masked to device 0 -- the mixed-BC case exercises that injection
  with a genuinely nonzero, solved-for value, so it's the stronger check.

There is only ONE ``solve_mechanics`` now (no separate "distributed" entry
point) -- passing ``n_devices`` caps the device count each formulation
auto-detects via ``jax.local_device_count()``; ``n_devices=None`` (the
default) is exactly the original single-device solve on a single-device
machine.

Same isolation caveat as the other ``dev_pmap`` tests: run standalone, not
inside the full ``pytest test/`` sweep. Fake device count via ``DEVICES``
(default 4):

    DEVICES=2 python -m pytest test/test_problems_mechanics_distributed.py -v
    DEVICES=8 python -m pytest test/test_problems_mechanics_distributed.py -v
"""

import os

N_DEVICES = int(os.environ.get("DEVICES", "4"))
os.environ.setdefault("XLA_FLAGS", f"--xla_force_host_platform_device_count={N_DEVICES}")

import sys

sys.path.insert(0, "src")

import numpy as np
import pytest
from typing import cast

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)
import jax
import jax.numpy as jnp

from materialmodels.elastic.isotropic import LinearElasticIsotropic
from problems.mechanics import solve_mechanics
from solvers.solution import ElasticitySolution

# N[0] and N[1] must both be divisible by N_DEVICES -- scale the grid with it.
N = (2 * N_DEVICES, 2 * N_DEVICES, 8)
L = (1.0, 1.0, 1.0)
TOLER_LIN = 1e-8
MAXITER = 1000

EPS_BAR = jnp.array([
    [0.0,    1.0e-3, 0.0],
    [1.0e-3, 0.0,    0.0],
    [0.0,    0.0,    0.0],
])
# One stress-controlled pair (Poisson contraction on the (1,1) direction),
# only meaningful for formulation="displacement" -- exercises the DC-bin
# injection with a genuinely nonzero, solved-for value.
MIXED_CONTROL = ((0, 0, 0), (0, 1, 0), (0, 0, 0))

# (formulation, control) -- control is None (pure strain) unless noted.
# ("lippmann_schwinger", MIXED_CONTROL) exercises solve_lippmann_schwinger_mixed_bc's
# Michel et al 1999 outer loop, which just calls solve_lippmann_schwinger
# repeatedly -- confirms the new fully-sharded inner CG solve (see
# solvers.elliptic.vector.lippmann_schwinger._solve_lippmann_schwinger_sharded)
# composes correctly with that outer loop with zero code changes there.
CASES: list[tuple[str, tuple | None]] = [
    ("lippmann_schwinger", None),
    ("lippmann_schwinger", MIXED_CONTROL),
    ("fourier_galerkin",   None),
    ("fourier_galerkin",   MIXED_CONTROL),
    ("displacement",       None),
    ("displacement",       MIXED_CONTROL),
]


def _build_case():
    Nv = int(np.prod(N))
    rng = np.random.default_rng(0)
    phase = jnp.asarray(rng.integers(0, 2, size=Nv))
    matrix = LinearElasticIsotropic(E=3.0e3, nu=0.35, name="matrix")
    fiber = LinearElasticIsotropic(E=70.0e3, nu=0.20, name="fiber")
    return phase, [matrix, fiber]


def _solve(phase, materials, formulation, control, n_devices) -> ElasticitySolution:
    results = solve_mechanics(
        N, L, phase, materials, EPS_BAR, formulation=formulation, control=control,
        scheme="rotated", toler_lin=TOLER_LIN, maxiter=MAXITER, n_devices=n_devices,
    )
    return cast(ElasticitySolution, results[0].solution)


@pytest.mark.parametrize("formulation,control", CASES)
def test_n_devices_1_matches_unset(formulation, control):
    """n_devices=1 forces the single-device fallback branch, regardless
    of how many real/fake devices this process actually sees -- the path a
    single-GPU machine takes at the default (n_devices=None)."""
    phase, materials = _build_case()

    sol_default = _solve(phase, materials, formulation, control, n_devices=None)
    sol_forced = _solve(phase, materials, formulation, control, n_devices=1)

    assert bool(sol_default.converged) and bool(sol_forced.converged)
    if jax.local_device_count() == 1:
        np.testing.assert_allclose(np.asarray(sol_forced.eps), np.asarray(sol_default.eps), atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(np.asarray(sol_forced.sigma), np.asarray(sol_default.sigma), atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("formulation,control", CASES)
def test_distributed_solve_matches_single_device(formulation, control):
    """Real (or DEVICES-simulated) multi-device path through the full
    CG solve, checked against n_devices=1 as the oracle."""
    if jax.local_device_count() < N_DEVICES:
        pytest.skip(
            f"need {N_DEVICES} local devices (got {jax.local_device_count()}) -- "
            "XLA_FLAGS device count must be set before the first `import jax` in "
            "this process; run this file on its own, not inside the full `pytest test/` sweep"
        )
    phase, materials = _build_case()

    sol_single = _solve(phase, materials, formulation, control, n_devices=1)
    sol_dist = _solve(phase, materials, formulation, control, n_devices=None)

    assert bool(sol_single.converged) and bool(sol_dist.converged)
    np.testing.assert_allclose(np.asarray(sol_dist.eps), np.asarray(sol_single.eps), atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(sol_dist.sigma), np.asarray(sol_single.sigma), atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    for formulation, control in CASES:
        test_n_devices_1_matches_unset(formulation, control)
        if jax.local_device_count() >= N_DEVICES:
            test_distributed_solve_matches_single_device(formulation, control)
    print(f"ok -- backend={jax.default_backend()}  local_device_count={jax.local_device_count()}")
