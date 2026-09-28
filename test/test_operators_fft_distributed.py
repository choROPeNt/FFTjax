"""
Correctness check for ``operators.fft_distributed`` (dev_pmap branch
prototype, not wired into any solver yet) against the existing single-device
``jnp.fft.fftn``/``ifftn`` path used everywhere else in this project.

Deliberately kept OUT of the default ``pytest test/`` sweep: JAX's visible
device count is fixed process-wide on the first ``import jax`` in a pytest
session, so if another test file imports jax first, the
``XLA_FLAGS``/``xla_force_host_platform_device_count`` set below never takes
effect and this file would silently "pass" while only exercising 1 device
(n_devices=1, no actual cross-device ``all_to_all``). Run this file on its
own:

    python -m pytest test/test_operators_fft_distributed.py -v

Fake device count is controlled by the ``FAKE_DEVICES`` env var (default 4),
read *before* ``XLA_FLAGS`` is set below -- since that in turn must happen
before the first ``import jax`` in this process, a single external env var
is enough to get a genuinely different simulated device count per process;
no shell-level ``XLA_FLAGS`` export needed. To compare multiple device
counts, run this file as separate processes/jobs (JAX's device count can't
be changed mid-process), e.g.:

    FAKE_DEVICES=2 python -m pytest test/test_operators_fft_distributed.py -v
    FAKE_DEVICES=4 python -m pytest test/test_operators_fft_distributed.py -v
    FAKE_DEVICES=8 python -m pytest test/test_operators_fft_distributed.py -v

This is a golden-comparison test, not a port-and-verify against old
reference code: the distributed and single-device FFTs are the same
algorithm computed two ways, so they must agree near machine precision --
a real correctness oracle, not a parity check against a pre-refactor module.
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

from operators.fft_distributed import distributed_fft_flat, distributed_ifft_flat, pfft3d, pifft3d

# N[0] and N[1] must both be divisible by N_DEVICES -- scale the grid with
# it so any FAKE_DEVICES value stays valid.
N = (2 * N_DEVICES, 2 * N_DEVICES, 8)


def _require_fake_devices():
    if jax.local_device_count() < N_DEVICES:
        pytest.skip(
            f"need {N_DEVICES} local devices (got {jax.local_device_count()}) -- "
            "XLA_FLAGS device count must be set before the first `import jax` in "
            "this process; run this file on its own, not inside the full `pytest test/` sweep"
        )


def _split_slabs(x_full: jnp.ndarray) -> jnp.ndarray:
    """(nx, ny, nz) -> (n_devices, nx/n_devices, ny, nz), one slab per device."""
    return x_full.reshape(N_DEVICES, N[0] // N_DEVICES, *N[1:])


def _gather_slabs(x_sharded: jnp.ndarray) -> jnp.ndarray:
    """(n_devices, nx/n_devices, ny, nz) -> (nx, ny, nz)."""
    return x_sharded.reshape(N)


def test_pfft3d_matches_fftn():
    _require_fake_devices()
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.standard_normal(N))

    expected = jnp.fft.fftn(x, axes=(0, 1, 2))

    x_sharded = _split_slabs(x)
    out_sharded = jax.pmap(lambda s: pfft3d(s, axis_name="i"), axis_name="i")(x_sharded)
    got = _gather_slabs(out_sharded)

    np.testing.assert_allclose(np.asarray(got), np.asarray(expected), atol=1e-9, rtol=1e-9)


def test_pifft3d_matches_ifftn():
    _require_fake_devices()
    rng = np.random.default_rng(1)
    x = jnp.asarray(rng.standard_normal(N) + 1j * rng.standard_normal(N))

    expected = jnp.fft.ifftn(x, axes=(0, 1, 2))

    x_sharded = _split_slabs(x)
    out_sharded = jax.pmap(lambda s: pifft3d(s, axis_name="i"), axis_name="i")(x_sharded)
    got = _gather_slabs(out_sharded)

    np.testing.assert_allclose(np.asarray(got), np.asarray(expected), atol=1e-9, rtol=1e-9)


def test_round_trip_recovers_input():
    _require_fake_devices()
    rng = np.random.default_rng(2)
    x = jnp.asarray(rng.standard_normal(N))

    x_sharded = _split_slabs(x)
    fwd = jax.pmap(lambda s: pfft3d(s, axis_name="i"), axis_name="i")(x_sharded)
    back = jax.pmap(lambda s: pifft3d(s, axis_name="i"), axis_name="i")(fwd)
    got = _gather_slabs(back)

    np.testing.assert_allclose(np.asarray(got.real), np.asarray(x), atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(np.asarray(got.imag), 0.0, atol=1e-9)


def test_distributed_fft_flat_matches_fftn():
    """distributed_fft_flat is the general-purpose drop-in used by
    displacement_based.py/mechanics.py's nonlinear driver/scalar.py's own
    fft_ closures -- (*lead, Nv) flat in, auto-decomposed, gathered back to
    (*lead, Nv) flat out. Uses a (3, Nv) leading-axis field (like a
    displacement field) rather than the bare grid pfft3d itself is tested
    against above, to exercise the same reshape convention those solvers use."""
    _require_fake_devices()
    rng = np.random.default_rng(3)
    Nv = int(np.prod(N))
    x = jnp.asarray(rng.standard_normal((3, Nv)))

    expected = jnp.fft.fftn(x.reshape(3, *N), axes=(-3, -2, -1)).reshape(3, Nv)
    got = distributed_fft_flat(x, N)

    np.testing.assert_allclose(np.asarray(got), np.asarray(expected), atol=1e-9, rtol=1e-9)


def test_distributed_ifft_flat_matches_ifftn():
    _require_fake_devices()
    rng = np.random.default_rng(4)
    Nv = int(np.prod(N))
    x = jnp.asarray(rng.standard_normal((3, Nv)) + 1j * rng.standard_normal((3, Nv)))

    expected = jnp.fft.ifftn(x.reshape(3, *N), axes=(-3, -2, -1)).real.reshape(3, Nv)
    got = distributed_ifft_flat(x, N)

    np.testing.assert_allclose(np.asarray(got), np.asarray(expected), atol=1e-9, rtol=1e-9)


def test_distributed_fft_flat_max_devices_1_matches_direct():
    """max_devices=1 forces the fallback branch regardless of how many
    devices this process sees -- proving the dispatch itself, independent
    of the pmap path (the path a single-GPU machine actually takes)."""
    rng = np.random.default_rng(5)
    Nv = int(np.prod(N))
    x = jnp.asarray(rng.standard_normal((3, Nv)))

    expected = jnp.fft.fftn(x.reshape(3, *N), axes=(-3, -2, -1)).reshape(3, Nv)
    got = distributed_fft_flat(x, N, max_devices=1)

    np.testing.assert_allclose(np.asarray(got), np.asarray(expected), atol=1e-12, rtol=1e-12)


if __name__ == "__main__":
    test_distributed_fft_flat_max_devices_1_matches_direct()
    _require_fake_devices()
    test_pfft3d_matches_fftn()
    test_pifft3d_matches_ifftn()
    test_round_trip_recovers_input()
    test_distributed_fft_flat_matches_fftn()
    test_distributed_ifft_flat_matches_ifftn()
    print(f"all good -- backend={jax.default_backend()}  "
          f"local_device_count={jax.local_device_count()}")
