"""
Correctness check for the scalar FFT-CG solvers' automatic domain
decomposition (``solvers/elliptic/scalar.py``): ``solve_damage_helmholtz_cg``
(homogeneous Gc), ``solve_damage_helmholtz_cg_het`` (per-voxel Gc, needs a
distributed gradient/divergence, not just a Laplacian), and
``solve_thermal_conduction`` (its own DC-bin macroscopic-gradient trick,
pure gradient-BC and mixed flux-BC). All three use
``operators.fft_distributed.distributed_fft_flat``/``distributed_ifft_flat``
as drop-in replacements for their own ``fft_``/``ifft_`` closures -- see
``test_problems_mechanics_distributed.py`` for the same primitive wired into
the vector (elasticity) solvers.

Same isolation caveat as the other ``dev_pmap`` tests: run standalone, not
inside the full ``pytest test/`` sweep. Fake device count via ``FAKE_DEVICES``
(default 4):

    FAKE_DEVICES=2 python -m pytest test/test_solvers_elliptic_scalar_distributed.py -v
    FAKE_DEVICES=8 python -m pytest test/test_solvers_elliptic_scalar_distributed.py -v
"""

import os

N_DEVICES = int(os.environ.get("FAKE_DEVICES", "4"))
os.environ.setdefault("XLA_FLAGS", f"--xla_force_host_platform_device_count={N_DEVICES}")

import sys

sys.path.insert(0, "src")

import numpy as np
import pytest

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)
import jax
import jax.numpy as jnp

from materialmodels.assembly import assemble_K_field
from materialmodels.thermal.isotropic import ThermalConductivityIsotropic
from operators.green import build_freq_grid
from solvers.elliptic.scalar import (
    solve_damage_helmholtz_cg, solve_damage_helmholtz_cg_het, solve_thermal_conduction,
)

# N[0] and N[1] must both be divisible by N_DEVICES -- scale the grid with it.
N = (2 * N_DEVICES, 2 * N_DEVICES, 8)
L = (1.0, 1.0, 1.0)
NV = int(np.prod(N))


def _require_multi_device():
    if jax.local_device_count() < N_DEVICES:
        pytest.skip(
            f"need {N_DEVICES} local devices (got {jax.local_device_count()}) -- "
            "XLA_FLAGS device count must be set before the first `import jax` in "
            "this process; run this file on its own, not inside the full `pytest test/` sweep"
        )


def _compare(fn, *args, tol_1=1e-10, tol_n=1e-6, **kwargs):
    """Run fn at max_devices=1 (single-device oracle) and max_devices=None
    (auto-detected) -- both must agree, at tol_1 if this process actually
    has only 1 device (byte-identical fallback), at the looser tol_n once
    real cross-device pmap arithmetic is exercised."""
    out_1 = fn(*args, max_devices=1, **kwargs)
    out_n = fn(*args, max_devices=None, **kwargs)
    tol = tol_1 if jax.local_device_count() == 1 else tol_n
    for a, b in zip(jax.tree_util.tree_leaves(out_1), jax.tree_util.tree_leaves(out_n)):
        np.testing.assert_allclose(np.asarray(b), np.asarray(a), atol=tol, rtol=tol)
    return out_n


def test_helmholtz_homogeneous_distributed_matches_single_device():
    _require_multi_device()
    xi_flat = build_freq_grid(N, L)
    rng = np.random.default_rng(0)
    H_field = jnp.asarray(np.abs(rng.standard_normal(NV)))
    d_prev = jnp.zeros(NV)

    d, converged = _compare(
        solve_damage_helmholtz_cg, H_field, xi_flat, N, 1.0, 1.0e-3, d_prev,
        toler_cg=1e-10, maxiter=500,
    )
    assert bool(converged)


def test_helmholtz_heterogeneous_distributed_matches_single_device():
    """Exercises the distributed gradient/divergence path (div(Gc(x) grad d)),
    not just the homogeneous Laplacian trick."""
    _require_multi_device()
    xi_flat = build_freq_grid(N, L)
    rng = np.random.default_rng(1)
    H_field = jnp.asarray(np.abs(rng.standard_normal(NV)))
    Gc = jnp.asarray(1.0e-3 * (1.0 + 0.5 * rng.integers(0, 2, size=NV)))
    d_prev = jnp.zeros(NV)

    d, converged = _compare(
        solve_damage_helmholtz_cg_het, H_field, xi_flat, N, 1.0, Gc, d_prev,
        toler_cg=1e-10, maxiter=500,
    )
    assert bool(converged)


def test_thermal_conduction_pure_gradient_bc_distributed_matches_single_device():
    _require_multi_device()
    xi_flat = build_freq_grid(N, L)
    rng = np.random.default_rng(2)
    phase = jnp.asarray(rng.integers(0, 2, size=NV))
    matrix = ThermalConductivityIsotropic(k=1.0, name="matrix")
    fiber = ThermalConductivityIsotropic(k=20.0, name="fiber")
    K_field = assemble_K_field([matrix, fiber], phase)
    grad_T_bar = jnp.array([1.0e-2, 0.0, 0.0])

    out = _compare(
        solve_thermal_conduction, N, K_field, xi_flat, grad_T_bar,
        toler_lin=1e-10, maxiter=500,
    )
    converged = out[-1]
    assert bool(converged)


def test_thermal_conduction_mixed_flux_bc_distributed_matches_single_device():
    """Nonzero control -- exercises the DC-bin macroscopic-gradient
    injection with a genuinely nonzero, solved-for value, not the
    degenerate all-zero case the pure-gradient-BC test above takes."""
    _require_multi_device()
    xi_flat = build_freq_grid(N, L)
    rng = np.random.default_rng(3)
    phase = jnp.asarray(rng.integers(0, 2, size=NV))
    matrix = ThermalConductivityIsotropic(k=1.0, name="matrix")
    fiber = ThermalConductivityIsotropic(k=20.0, name="fiber")
    K_field = assemble_K_field([matrix, fiber], phase)
    grad_T_bar = jnp.array([1.0e-2, 0.0, 0.0])
    control = (0, 1, 0)  # flux-controlled in y

    out = _compare(
        solve_thermal_conduction, N, K_field, xi_flat, grad_T_bar, control,
        toler_lin=1e-10, maxiter=500,
    )
    converged = out[-1]
    assert bool(converged)


if __name__ == "__main__":
    test_helmholtz_homogeneous_distributed_matches_single_device()
    test_helmholtz_heterogeneous_distributed_matches_single_device()
    test_thermal_conduction_pure_gradient_bc_distributed_matches_single_device()
    test_thermal_conduction_mixed_flux_bc_distributed_matches_single_device()
    print(f"ok -- backend={jax.default_backend()}  local_device_count={jax.local_device_count()}")
