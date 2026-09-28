"""
Correctness check for ``problems.mechanics.solve_displacement_based_nonlinear``'s
automatic domain decomposition -- the Newton-outer/CG-inner plasticity
driver, not just a linear CG solve (see ``test_problems_mechanics_distributed.py``
for the linear formulations). Same ``operators.fft_distributed.distributed_fft_flat``
primitive as ``solve_displacement_based``, wired into this driver's own
``fft_``/``ifft_`` closures -- ``local_update`` itself (the per-voxel J2
plasticity return mapping here) needs no changes, since it's already a plain
per-voxel map with no cross-voxel reduction (see
``materialmodels.inelastic.plasticity_j2.J2Plasticity.stress_and_tangent_field``).

Pushes the load well past yield so the plastic branch actually engages every
Newton iteration -- a linear-only check wouldn't exercise ``local_update``
being called with a genuinely evolving ``state`` across iterations.

Same isolation caveat as the other ``dev_pmap`` tests: run standalone, not
inside the full ``pytest test/`` sweep. Fake device count via ``FAKE_DEVICES``
(default 4):

    FAKE_DEVICES=2 python -m pytest test/test_problems_mechanics_nonlinear_distributed.py -v
    FAKE_DEVICES=8 python -m pytest test/test_problems_mechanics_nonlinear_distributed.py -v
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

from materialmodels.elastic.isotropic import LinearElasticIsotropic
from materialmodels.inelastic.plasticity_j2 import J2Plasticity
from operators.green import build_freq_grid
from problems.mechanics import solve_displacement_based_nonlinear

# N[0] and N[1] must both be divisible by N_DEVICES -- scale the grid with it.
N = (2 * N_DEVICES, 2 * N_DEVICES, 8)
L = (1.0, 1.0, 1.0)
TOLER_LIN = 1e-8
MAXITER_LIN = 1000
TOLER_NR = 1e-8
MAXITER_NR = 30

EPS_BAR = jnp.array([
    [0.0,   3.0e-3, 0.0],
    [3.0e-3, 0.0,   0.0],
    [0.0,   0.0,    0.0],
])


def _build_case():
    Nv = int(np.prod(N))
    rng = np.random.default_rng(0)
    phase = jnp.asarray(rng.integers(0, 2, size=Nv))

    fiber_el = LinearElasticIsotropic(E=70.0e3, nu=0.20, name="fiber")
    # Low sigma_y0 so yielding is robust across every grid size this file
    # runs at (N scales with N_DEVICES/FAKE_DEVICES -- see N's definition
    # above), not just the default FAKE_DEVICES=4 shape/phase pattern.
    matrix_pl = J2Plasticity(E=3.76e3, nu=0.39, sigma_y0=5.0, H=1.0e3, name="matrix-plastic")
    C_fiber = fiber_el.elastic_stiffness_tensor()

    def local_update(eps_field, state):
        eps_p_field, alpha_field = state
        sigma_pl, C_pl, (eps_p_new, alpha_new) = matrix_pl.stress_and_tangent_field(
            eps_field, eps_p_field, alpha_field
        )
        sigma_el = jnp.einsum("ijkl,klm->ijm", C_fiber, eps_field)
        C_el = jnp.broadcast_to(C_fiber[..., None], C_pl.shape)
        is_matrix = (phase == 0)
        sigma = jnp.where(is_matrix, sigma_pl, sigma_el)
        C_tan = jnp.where(is_matrix, C_pl, C_el)
        eps_p_out = jnp.where(is_matrix, eps_p_new, eps_p_field)
        alpha_out = jnp.where(is_matrix, alpha_new, alpha_field)
        return sigma, C_tan, (eps_p_out, alpha_out)

    state_init = (jnp.zeros((3, 3, Nv)), jnp.zeros(Nv))
    xi_flat = build_freq_grid(N, L)
    return xi_flat, local_update, state_init


def _solve(xi_flat, local_update, state_init, max_devices):
    eps, sigma, (_, alpha), converged, n_iter = solve_displacement_based_nonlinear(
        N, xi_flat, EPS_BAR, local_update, state_init,
        toler_lin=TOLER_LIN, maxiter_lin=MAXITER_LIN,
        toler_nr=TOLER_NR, maxiter_nr=MAXITER_NR,
        max_devices=max_devices,
    )
    return eps, sigma, alpha, converged, n_iter


def test_max_devices_1_matches_unset():
    """max_devices=1 forces the single-device fallback branch, regardless
    of how many real/fake devices this process actually sees -- the path a
    single-GPU machine takes at the default (max_devices=None)."""
    xi_flat, local_update, state_init = _build_case()

    eps_d, sigma_d, alpha_d, conv_d, n_iter_d = _solve(xi_flat, local_update, state_init, max_devices=None)
    eps_f, sigma_f, alpha_f, conv_f, n_iter_f = _solve(xi_flat, local_update, state_init, max_devices=1)

    assert bool(conv_d) and bool(conv_f)
    # Whether any voxel actually yields depends on grid size/phase pattern
    # (which vary with N_DEVICES here, see N's definition above) -- not a
    # meaningful dispatch check on its own, see
    # test_distributed_solve_matches_single_device for the real cross-device
    # numerical comparison, which does assert yielding on the fixed-shape
    # single-device-vs-distributed pair it directly compares.
    if jax.local_device_count() == 1:
        assert n_iter_d == n_iter_f
        np.testing.assert_allclose(np.asarray(eps_f), np.asarray(eps_d), atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(np.asarray(sigma_f), np.asarray(sigma_d), atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(np.asarray(alpha_f), np.asarray(alpha_d), atol=1e-10, rtol=1e-10)


def test_distributed_solve_matches_single_device():
    """Real (or FAKE_DEVICES-simulated) multi-device path through the full
    Newton/CG solve, checked against max_devices=1 as the oracle."""
    if jax.local_device_count() < N_DEVICES:
        pytest.skip(
            f"need {N_DEVICES} local devices (got {jax.local_device_count()}) -- "
            "XLA_FLAGS device count must be set before the first `import jax` in "
            "this process; run this file on its own, not inside the full `pytest test/` sweep"
        )
    xi_flat, local_update, state_init = _build_case()

    eps_s, sigma_s, alpha_s, conv_s, n_iter_s = _solve(xi_flat, local_update, state_init, max_devices=1)
    eps_d, sigma_d, alpha_d, conv_d, n_iter_d = _solve(xi_flat, local_update, state_init, max_devices=None)

    assert bool(conv_s) and bool(conv_d)
    assert int(jnp.sum(alpha_s > 1e-12)) > 0, "expected plastic yielding at this load level"
    assert n_iter_s == n_iter_d
    np.testing.assert_allclose(np.asarray(eps_d), np.asarray(eps_s), atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(sigma_d), np.asarray(sigma_s), atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(alpha_d), np.asarray(alpha_s), atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    test_max_devices_1_matches_unset()
    if jax.local_device_count() >= N_DEVICES:
        test_distributed_solve_matches_single_device()
    print(f"ok -- backend={jax.default_backend()}  local_device_count={jax.local_device_count()}")
