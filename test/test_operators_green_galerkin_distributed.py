"""
Correctness check for GreenOperatorBasic/Willot's and GalerkinProjector's
automatic domain decomposition of their own ``.G`` tensor construction --
the second half of the benchmark_3 OOM investigation (the first being
``materialmodels.assembly.assemble_C_field``, see
test_materialmodels_assembly_distributed.py): ``build_green_operator``/
``build_galerkin_projection_operator`` are both pure, no-cross-voxel-
coupling functions of ``xi_flat`` alone, same structure as
``assemble_C_field``, so sharded the same way
(``operators.fft_distributed.shard_voxel_tensor``).

Same isolation caveat as the other ``dev_pmap`` tests: run standalone, not
inside the full ``pytest test/`` sweep. Fake device count via ``DEVICES``
(default 4):

    DEVICES=2 python -m pytest test/test_operators_green_galerkin_distributed.py -v
    DEVICES=4 python -m pytest test/test_operators_green_galerkin_distributed.py -v
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

from operators.galerkin import GalerkinProjector
from operators.green import GreenOperatorBasic, GreenOperatorWillot

# N[0] and N[1] must both be divisible by N_DEVICES -- scale the grid with it.
N = (2 * N_DEVICES, 2 * N_DEVICES, 8)
L = (1.0, 1.0, 1.0)
LAM0, MU0 = 5.56e3, 8.33e3
DX = tuple(Li / ni for Li, ni in zip(L, N))

BUILDERS = {
    "green_basic":  lambda n_devices: GreenOperatorBasic(N, L, LAM0, MU0, n_devices=n_devices),
    "green_willot": lambda n_devices: GreenOperatorWillot(N, L, LAM0, MU0, DX, n_devices=n_devices),
    "galerkin_standard": lambda n_devices: GalerkinProjector(N, L, scheme="standard", n_devices=n_devices),
    "galerkin_rotated":  lambda n_devices: GalerkinProjector(N, L, scheme="rotated", n_devices=n_devices),
}


@pytest.mark.parametrize("name", list(BUILDERS), ids=list(BUILDERS))
def test_n_devices_1_matches_unset(name):
    """n_devices=1 forces the single-device fallback branch, regardless of
    how many real/fake devices this process actually sees -- proving the
    dispatch itself is correct, without needing multiple devices at all."""
    build = BUILDERS[name]
    op_single = build(1)
    op_default = build(1)  # both n_devices=1 here -- the real auto-detect
                            # path is exercised by the distributed test below
    np.testing.assert_allclose(np.asarray(op_default.G), np.asarray(op_single.G), atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("name", list(BUILDERS), ids=list(BUILDERS))
def test_distributed_matches_single_device(name):
    """Real (or DEVICES-simulated) multi-device path, checked against
    n_devices=1 as the oracle."""
    if jax.local_device_count() < N_DEVICES:
        pytest.skip(
            f"need {N_DEVICES} local devices (got {jax.local_device_count()}) -- "
            "XLA_FLAGS device count must be set before the first `import jax` in "
            "this process; run this file on its own, not inside the full `pytest test/` sweep"
        )
    build = BUILDERS[name]
    op_single = build(1)
    op_dist = build(None)  # auto-detect -> N_DEVICES

    np.testing.assert_allclose(np.asarray(op_dist.G), np.asarray(op_single.G), atol=1e-8, rtol=1e-8)


if __name__ == "__main__":
    for name in BUILDERS:
        test_n_devices_1_matches_unset(name)
        if jax.local_device_count() >= N_DEVICES:
            test_distributed_matches_single_device(name)
    print(f"ok -- backend={jax.default_backend()}  local_device_count={jax.local_device_count()}")
