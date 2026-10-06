"""
Correctness check for ``materialmodels.assembly.assemble_C_field``'s
automatic domain decomposition -- builds the per-voxel stiffness field
directly as x-slabs under ``jax.pmap`` instead of materializing the full
``(3,3,3,3,Nv)`` tensor on one device first (see that function's own
docstring: this is what OOMs first on a large grid, before any solve even
starts -- benchmark/benchmark_3/elastic_solve.py's own investigation).

Two cases, both checked: (1) every material a constant tensor (the fast
gather-by-phase path), and (2) a per-voxel-oriented ``TransverseIsotropic``
mixed in (duck-typed via its ``.fiber_dir``/``.stiffness_field_oriented``
convention -- see ``_assemble_C_field_sharded``'s docstring), which is the
actual case that matters here: that material's own rotation is itself a
real memory/compute cost, not just the final per-phase gather, so a
correctness check that only exercised constant-tensor materials would miss
the whole point of sharding the assembly at all.

Same isolation caveat as the other ``dev_pmap`` tests: run standalone, not
inside the full ``pytest test/`` sweep. Fake device count via ``DEVICES``
(default 4):

    DEVICES=2 python -m pytest test/test_materialmodels_assembly_distributed.py -v
    DEVICES=4 python -m pytest test/test_materialmodels_assembly_distributed.py -v
"""

import os

N_DEVICES = int(os.environ.get("DEVICES", "4"))
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
from materialmodels.elastic.transverse_isotropic import TransverseIsotropic

# N[0] and N[1] must both be divisible by N_DEVICES -- scale the grid with it.
N = (2 * N_DEVICES, 2 * N_DEVICES, 8)
Nv = int(np.prod(N))


def _build_case_fast_path():
    """Every material a constant (3,3,3,3) tensor -- the gather-by-phase path."""
    rng = np.random.default_rng(0)
    phase = jnp.asarray(rng.integers(0, 2, size=Nv))
    matrix = LinearElasticIsotropic(E=3.0e3, nu=0.35, name="matrix")
    fiber = LinearElasticIsotropic(E=70.0e3, nu=0.20, name="fiber")
    return [matrix, fiber], phase


def _build_case_mixed_path():
    """A per-voxel-oriented TransverseIsotropic fibre mixed with a constant
    matrix -- exercises _LocalFieldMaterial's slice-before-rotate path, the
    actual memory/compute cost this sharding exists for."""
    rng = np.random.default_rng(1)
    phase = jnp.asarray(rng.integers(0, 2, size=Nv))
    matrix = LinearElasticIsotropic(E=3.0e3, nu=0.35, name="matrix")
    orientations = jnp.asarray(rng.normal(size=(3, Nv)))  # not pre-normalized, by design
    fiber = TransverseIsotropic(
        E_L=234000.0, E_T=15000.0, G_LT=15000.0, nu_LT=0.20, G_TT=7000.0,
        fiber_dir=orientations, name="fibre (per-voxel oriented)",
    )
    return [matrix, fiber], phase


@pytest.mark.parametrize("build_case", [_build_case_fast_path, _build_case_mixed_path],
                          ids=["fast_path", "mixed_path"])
def test_n_devices_1_matches_direct(build_case):
    """n_devices=1 forces the single-device fallback branch, regardless of
    how many real/fake devices this process actually sees -- proving the
    (n, n_devices) dispatch itself is correct against the original n=None
    call, without needing multiple devices at all."""
    materials, phase = build_case()

    expected = assemble_C_field(materials, phase)  # original, undecorated call
    got = assemble_C_field(materials, phase, n=N, n_devices=1)

    np.testing.assert_allclose(np.asarray(got), np.asarray(expected), atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("build_case", [_build_case_fast_path, _build_case_mixed_path],
                          ids=["fast_path", "mixed_path"])
def test_distributed_matches_single_device(build_case):
    """Real (or DEVICES-simulated) multi-device path, checked against
    n_devices=1 as the oracle."""
    if jax.local_device_count() < N_DEVICES:
        pytest.skip(
            f"need {N_DEVICES} local devices (got {jax.local_device_count()}) -- "
            "XLA_FLAGS device count must be set before the first `import jax` in "
            "this process; run this file on its own, not inside the full `pytest test/` sweep"
        )
    materials, phase = build_case()

    expected = assemble_C_field(materials, phase, n=N, n_devices=1)
    got = assemble_C_field(materials, phase, n=N)  # auto-detect -> N_DEVICES

    np.testing.assert_allclose(np.asarray(got), np.asarray(expected), atol=1e-8, rtol=1e-8)


if __name__ == "__main__":
    for build_case in (_build_case_fast_path, _build_case_mixed_path):
        test_n_devices_1_matches_direct(build_case)
        if jax.local_device_count() >= N_DEVICES:
            test_distributed_matches_single_device(build_case)
    print(f"ok -- backend={jax.default_backend()}  local_device_count={jax.local_device_count()}")
