"""
Standalone test for ElasticitySolver / ElasticitySolution
(solvers/elliptic/vector/base.py).

Three checks
------------
1. ElasticitySolver is a real ABC -- can't be instantiated directly, and a
   subclass that doesn't implement solve() can't either.
2. ElasticitySolution is a genuine JAX pytree (NamedTuple, not a plain
   dataclass) -- jax.tree_util.tree_flatten must see its four fields as
   leaves, so it survives jax.jit/grad/jvp the same way solvers.types.
   SolveState does.
3. A minimal concrete ElasticitySolver satisfies the contract end-to-end.

Usage
-----
    python -m pytest test/test_elliptic_vector_base.py
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.makedirs("output", exist_ok=True)

import sys
sys.path.insert(0, "src")

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)
import jax
import jax.numpy as jnp
import pytest

from solvers.elliptic.vector.base import ElasticitySolver, ElasticitySolution


# ── 1. ElasticitySolver is a real ABC ────────────────────────────────────────

with pytest.raises(TypeError):
    ElasticitySolver()


class _Incomplete(ElasticitySolver):
    pass


with pytest.raises(TypeError):
    _Incomplete()


# ── 2. ElasticitySolution is a genuine pytree ────────────────────────────────

sol = ElasticitySolution(
    eps=jnp.zeros((3, 3, 4)),
    sigma=jnp.ones((3, 3, 4)),
    delta=jnp.zeros((3, 3, 4)),
    converged=jnp.array(True),
)
leaves, treedef = jax.tree_util.tree_flatten(sol)
assert len(leaves) == 4, f"expected 4 leaves (eps, sigma, delta, converged), got {len(leaves)}"

rebuilt = jax.tree_util.tree_unflatten(treedef, [x * 2 for x in leaves])
assert jnp.array_equal(rebuilt.sigma, sol.sigma * 2), "pytree round-trip must preserve field mapping"


# ── 3. minimal concrete solver satisfies the contract ────────────────────────

class _Identity(ElasticitySolver):
    """Toy solver: eps = eps_bar broadcast, sigma = C:eps, no correction."""

    def __init__(self, Nv: int):
        self.Nv = Nv

    def solve(self, C_field, eps_bar, stress_goal=None):
        eps = jnp.ones((3, 3, self.Nv)) * eps_bar[:, :, None]
        sigma = jnp.einsum("ijklm,klm->ijm", C_field, eps)
        delta = jnp.zeros_like(eps)
        return ElasticitySolution(eps, sigma, delta, jnp.array(True))


Nv = 8
I2 = jnp.eye(3)
I4s = 0.5 * (jnp.einsum('ik,jl->ijkl', I2, I2) + jnp.einsum('il,jk->ijkl', I2, I2))
C_field = jnp.broadcast_to(60.0 * I4s[..., None], (3, 3, 3, 3, Nv))
eps_bar = jnp.eye(3) * 1e-3

result = _Identity(Nv).solve(C_field, eps_bar)
assert isinstance(result, ElasticitySolution)
assert result.eps.shape == (3, 3, Nv)
assert bool(result.converged)

print("test_elliptic_vector_base: all checks passed")
