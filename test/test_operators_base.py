"""
Standalone test for the LinearOperator ABC (operators/base.py) and its first
concrete implementations, GreenOperatorBasic / GreenOperatorWillot
(operators/green.py).

Three checks
------------
1. Composition and adjoint on toy scalar operators — verifies ``__matmul__``
   builds ``(A @ B)(x) == A(B(x))`` without evaluating either side eagerly,
   and that ``.T`` on a composition reverses order: ``(A @ B).T == B.T @ A.T``.
2. GreenOperatorBasic / GreenOperatorWillot parity — calling the class must
   give bit-identical results to the existing functional path
   (``build_green_operator`` + ``ddot42``) it wraps.
3. Self-adjointness — ``<G(x), y> == <x, G(y)>`` for random x, y, confirming
   the default ``.T -> self`` is a valid adjoint for the Green's operator
   (major-symmetric: Γ_ijkl = Γ_klij).

Usage
-----
    python -m pytest test/test_operators_base.py
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
sys.path.insert(0, "src")

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)
import jax.numpy as jnp
import numpy as np

from operators.base import LinearOperator
from operators.general_functions import ddot42
from operators.green import (
    GreenOperatorBasic,
    GreenOperatorWillot,
    build_freq_grid,
    build_green_operator,
)


# ── 1. composition and adjoint on toy operators ─────────────────────────────

class _Scale(LinearOperator):
    """x -> a * x; adjoint is itself since a is real."""

    def __init__(self, a: float):
        self.a = a

    def __call__(self, x):
        return self.a * x


class _Shift(LinearOperator):
    """Non-self-adjoint toy: right-multiply by a fixed matrix M along the last axis."""

    def __init__(self, M: jnp.ndarray):
        self.M = M

    def __call__(self, x):
        return x @ self.M

    @property
    def T(self):
        return _Shift(self.M.T)


rng = np.random.default_rng(0)
M = jnp.asarray(rng.normal(size=(4, 4)))
x = jnp.asarray(rng.normal(size=(4,)))

A = _Scale(2.0)
B = _Shift(M)
composed = A @ B

assert jnp.allclose(composed(x), A(B(x))), "A @ B must equal A(B(x))"

adjoint = composed.T
assert jnp.allclose(adjoint(x), B.T(A.T(x))), "(A @ B).T must equal B.T @ A.T"


# ── 2. GreenOperatorBasic / GreenOperatorWillot parity with the functional path ──

n = (4, 4, 4)
L = (1.0, 1.0, 1.0)
dx = tuple(Li / ni for Li, ni in zip(L, n))
lam0, mu0 = 60.0, 30.0
Nv = int(np.prod(n))

eps = jnp.asarray(rng.normal(size=(3, 3, Nv)))

xi_flat = build_freq_grid(n, L)

G_std_expected = build_green_operator(xi_flat, lam0, mu0, scheme='standard')
G_std_class = GreenOperatorBasic(n, L, lam0, mu0)
assert jnp.allclose(G_std_class.G, G_std_expected)
assert jnp.allclose(G_std_class(eps), ddot42(G_std_expected, eps)), \
    "GreenOperatorBasic.__call__ must match ddot42(build_green_operator(...), x)"

G_rot_expected = build_green_operator(xi_flat, lam0, mu0, scheme='rotated', dx=dx)
G_rot_class = GreenOperatorWillot(n, L, lam0, mu0, dx)
assert jnp.allclose(G_rot_class.G, G_rot_expected)
assert jnp.allclose(G_rot_class(eps), ddot42(G_rot_expected, eps)), \
    "GreenOperatorWillot.__call__ must match ddot42(build_green_operator(..., scheme='rotated'), x)"


# ── 3. self-adjointness of the Green's operator ─────────────────────────────

y = jnp.asarray(rng.normal(size=(3, 3, Nv)))

for op in (G_std_class, G_rot_class):
    lhs = jnp.sum(op(eps) * y)
    rhs = jnp.sum(eps * op.T(y))
    rel = abs(lhs - rhs) / (abs(lhs) + 1e-30)
    assert rel < 1e-10, f"{type(op).__name__} failed <Gx,y> == <x,Gy>: rel={rel}"

print("test_operators_base: all checks passed")
