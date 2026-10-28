"""
Correctness check for ``solvers.krylov.cg.cg_solve_pmap`` in isolation --
the ``jax.lax.psum``-based preconditioned CG used *inside* a ``jax.pmap``
call for a fully domain-decomposed solve (state vectors stay sharded the
whole iteration, never gathered -- see
``solvers.elliptic.vector.lippmann_schwinger._solve_lippmann_schwinger_sharded``,
its first real caller). Tested here against a small synthetic SPD system
and ``solvers.krylov.cg.cg_solve`` as the single-device oracle, independent
of any FFT/Gamma0 machinery, to isolate the new psum-based reduction logic
itself from everything built on top of it.

Same isolation caveat as the other ``dev_pmap`` tests: run standalone, not
inside the full ``pytest test/`` sweep. Fake device count via ``DEVICES``
(default 4):

    DEVICES=2 python -m pytest test/test_solvers_krylov_cg_sharded.py -v
    DEVICES=8 python -m pytest test/test_solvers_krylov_cg_sharded.py -v
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

from solvers.krylov.cg import cg_solve, cg_solve_pmap

# Local length per device -- total system size is N_DEVICES * N_LOCAL.
N_LOCAL = 6
TOLER = 1e-10
MAXITER = 500


def _require_multi_device():
    if jax.local_device_count() < N_DEVICES:
        pytest.skip(
            f"need {N_DEVICES} local devices (got {jax.local_device_count()}) -- "
            "XLA_FLAGS device count must be set before the first `import jax` in "
            "this process; run this file on its own, not inside the full `pytest test/` sweep"
        )


def _build_spd_system():
    """A small, genuinely dense (not diagonal -- CG would trivially converge
    in one step on a diagonal system) SPD matrix A = M^T M + N*I, split into
    N_DEVICES row-blocks that each own a full row-slice of A and the
    matching slice of b -- a matrix-free A(v) = A @ v computed as
    (block_row @ v_full), which needs the FULL v on every device (an
    implicit all-gather via psum of a masked contribution) to mirror how a
    real distributed operator needs every device's contribution, not just
    its own local slice, to form Av."""
    N = N_DEVICES * N_LOCAL
    rng = np.random.default_rng(0)
    M = rng.standard_normal((N, N))
    A = M.T @ M + N * np.eye(N)  # SPD, well-conditioned
    b = rng.standard_normal(N)
    return jnp.asarray(A), jnp.asarray(b)


def test_cg_solve_pmap_matches_single_device_cg_solve():
    _require_multi_device()
    A, b = _build_spd_system()
    N = A.shape[0]

    x_expected, converged_expected = cg_solve(lambda v: A @ v, b, jnp.zeros(N), TOLER, MAXITER)
    assert bool(converged_expected)

    # Row-block sharding: device d owns rows [d*N_LOCAL : (d+1)*N_LOCAL] of
    # A and b. A_op needs the FULL current iterate (v) on every device to
    # form its local row-block @ v -- so v itself must be all-gathered
    # (psum of a zero-padded local contribution) inside A_op, exactly the
    # kind of within-iteration collective a real distributed operator needs.
    A_blocks = jnp.asarray(np.asarray(A).reshape(N_DEVICES, N_LOCAL, N))
    b_blocks = jnp.asarray(np.asarray(b).reshape(N_DEVICES, N_LOCAL))

    def solve_local(A_block, b_local):
        def A_op(v_local):
            # All-gather v via psum of a padded, one-hot-masked local slice
            # -- every device ends up with the identical full v. The offset
            # is a per-device *traced* value under pmap (jax.lax.axis_index),
            # so it needs lax.dynamic_update_slice, not a Python slice
            # (which requires static/constant bounds).
            idx = jax.lax.axis_index("i")
            padded = jax.lax.dynamic_update_slice(
                jnp.zeros((N,), dtype=v_local.dtype), v_local, (idx * N_LOCAL,),
            )
            v_full = jax.lax.psum(padded, "i")
            return A_block @ v_full

        x0 = jnp.zeros((N_LOCAL,))
        return cg_solve_pmap(A_op, b_local, x0, TOLER, MAXITER, axis_name="i")

    x_sharded, converged_sharded = jax.pmap(solve_local, axis_name="i")(A_blocks, b_blocks)
    x_got = np.asarray(x_sharded).reshape(N)

    assert bool(np.all(np.asarray(converged_sharded)))
    np.testing.assert_allclose(x_got, np.asarray(x_expected), atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    test_cg_solve_pmap_matches_single_device_cg_solve()
    print(f"ok -- backend={jax.default_backend()}  local_device_count={jax.local_device_count()}")
