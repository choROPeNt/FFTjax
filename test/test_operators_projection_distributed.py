"""
Correctness check for ``operators.projection.Gamma0Operator``'s automatic
domain decomposition -- one level up from ``test_operators_fft_distributed.py``'s
raw-FFT check: this exercises the real per-voxel material field
(``materialmodels.assembly.assemble_C_field``) and the real reference
Green's operator (``operators.green.build_reference_green_operator``),
applied to the exact quantity ``solve_lippmann_schwinger`` builds as its CG
right-hand side (``-gamma0(sigma0 - stress_goal)``) -- not a synthetic
random field.

Gamma0Operator itself is ``x -> ifftn(green_op(fftn(x)))`` with no
cross-voxel reduction in ``green_op`` (a pointwise per-frequency
``ddot42`` contraction, see operators/projection.py's docstring) -- so
domain-decomposing it only requires the distributed FFT primitive, no
global reductions yet (those start mattering once CG's own dot products
are decomposed, a separate, later step).

There is only ONE class here now (no separate "distributed" variant) --
``Gamma0Operator`` auto-detects its device count from
``jax.local_device_count()`` (see operators/fft_distributed.py's
``choose_device_count``) and takes the original single-device code path at
one device, no configuration needed on real hardware. Two things are
checked, both against the SAME class, constructed two ways: (1) on this
real single-device machine, ``max_devices=1`` forces the fallback branch
(proving the dispatch itself, without needing multiple devices at all --
this is the path a laptop with one GPU actually takes), and (2) under
simulated multi-device (this file run standalone with FAKE_DEVICES set),
the auto-detected distributed path matches the same construction with
``max_devices=1`` as its oracle.

Same isolation caveat as ``test_operators_fft_distributed.py`` for check
(2): run standalone, not inside the full ``pytest test/`` sweep. Fake
device count is controlled by the ``FAKE_DEVICES`` env var (default 4) --
run as separate processes/jobs to compare device counts, e.g.:

    FAKE_DEVICES=2 python -m pytest test/test_operators_projection_distributed.py -v
    FAKE_DEVICES=4 python -m pytest test/test_operators_projection_distributed.py -v
    FAKE_DEVICES=8 python -m pytest test/test_operators_projection_distributed.py -v
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

from materialmodels.assembly import assemble_C_field
from materialmodels.elastic.isotropic import LinearElasticIsotropic
from operators.general_functions import ddot42
from operators.green import build_reference_green_operator
from operators.projection import Gamma0Operator

# N[0] and N[1] must both be divisible by N_DEVICES -- scale the grid with
# it so any FAKE_DEVICES value stays valid.
N = (2 * N_DEVICES, 2 * N_DEVICES, 8)
L = (1.0, 1.0, 1.0)


def _build_case():
    """Real per-voxel material field + real reference Green's operator on a
    small random two-phase grid, and the exact RHS quantity
    solve_lippmann_schwinger feeds into Gamma0Operator."""
    Nv = int(np.prod(N))
    rng = np.random.default_rng(0)
    phase = jnp.asarray(rng.integers(0, 2, size=Nv))

    matrix = LinearElasticIsotropic(E=3.0e3, nu=0.35, name="matrix")
    fiber = LinearElasticIsotropic(E=70.0e3, nu=0.20, name="fiber")
    materials = [matrix, fiber]

    C_field = assemble_C_field(materials, phase)
    green_op = build_reference_green_operator(N, L, materials, scheme="rotated")

    eps_bar = jnp.array([[1.0e-3, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    eps0 = jnp.ones((3, 3, Nv)) * eps_bar[:, :, None]
    sigma0 = ddot42(C_field, eps0)  # same RHS-precursor as solve_lippmann_schwinger
    return green_op, sigma0


def _direct_gamma0(green_op, sigma0: jnp.ndarray, n) -> jnp.ndarray:
    """Independent-of-Gamma0Operator reference: the exact same map
    (ifftn(green_op(fftn(x)))), computed by hand rather than through the
    class under test -- an oracle that doesn't move if Gamma0Operator's own
    single-device branch ever changes."""
    s = sigma0.shape
    sigma_hat = jnp.fft.fftn(sigma0.reshape(s[:-1] + n), axes=(-3, -2, -1)).reshape(s)
    out_hat = green_op(sigma_hat)
    return jnp.fft.ifftn(out_hat.reshape(s[:-1] + n), axes=(-3, -2, -1)).real.reshape(s)


def test_max_devices_1_matches_direct_computation():
    """max_devices=1 forces the single-device fallback branch, regardless
    of how many real/fake devices this process actually sees -- proving the
    dispatch itself is correct against an oracle computed independently of
    Gamma0Operator, without needing multiple devices at all. This is the
    path a laptop with one GPU actually takes at the default (max_devices=None)."""
    green_op, sigma0 = _build_case()

    gamma0 = Gamma0Operator(N, green_op, max_devices=1)
    assert gamma0.n_devices == 1

    expected = _direct_gamma0(green_op, sigma0, N)
    np.testing.assert_allclose(np.asarray(gamma0(sigma0)), np.asarray(expected), atol=1e-10, rtol=1e-10)


def test_distributed_gamma0_matches_single_device():
    """Real (or simulated, via FAKE_DEVICES) multi-device path -- same
    class, auto-detecting jax.local_device_count(), checked against the
    same class with max_devices=1 forced as the oracle."""
    if jax.local_device_count() < N_DEVICES:
        pytest.skip(
            f"need {N_DEVICES} local devices (got {jax.local_device_count()}) -- "
            "XLA_FLAGS device count must be set before the first `import jax` in "
            "this process; run this file on its own, not inside the full `pytest test/` sweep"
        )
    green_op, sigma0 = _build_case()

    gamma0_single = Gamma0Operator(N, green_op, max_devices=1)
    expected = gamma0_single(sigma0)

    gamma0_dist = Gamma0Operator(N, green_op)
    assert gamma0_dist.n_devices == N_DEVICES  # confirms auto-detection actually picked it up
    got = gamma0_dist(sigma0)

    np.testing.assert_allclose(np.asarray(got), np.asarray(expected), atol=1e-8, rtol=1e-8)


if __name__ == "__main__":
    test_max_devices_1_matches_direct_computation()
    if jax.local_device_count() >= N_DEVICES:
        test_distributed_gamma0_matches_single_device()
    print(f"ok -- backend={jax.default_backend()}  local_device_count={jax.local_device_count()}")
