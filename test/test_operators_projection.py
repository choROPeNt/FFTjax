"""
Standalone test for Gamma0Operator (operators/projection.py).

Three checks
------------
1. Parity — Gamma0Operator(n, GreenOperatorBasic(...))(x) must match the
   manual ifftn(ddot42(G, fftn(x))).real reimplementation, and likewise for
   GreenOperatorWillot.
2. Self-adjointness — <Gamma0(x), y> == <x, Gamma0(y)> for random real x, y.
3. Solver parity — applying Gamma0Operator to C_field:v must reproduce
   dstrain_nw_cg's inner A_op exactly (same FFT/Green pattern it already
   uses inline), confirming projection.py is a drop-in for what the existing
   solver hand-rolls.

Usage
-----
    python -m pytest test/test_operators_projection.py
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.makedirs("output", exist_ok=True)

import sys
sys.path.insert(0, "src")

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)
import jax.numpy as jnp
import numpy as np

from operators.general_functions import ddot42
from operators.green import GreenOperatorBasic, GreenOperatorWillot, build_freq_grid, build_green_operator
from operators.projection import Gamma0Operator


rng = np.random.default_rng(1)

n = (4, 4, 4)
L = (1.0, 1.0, 1.0)
dx = tuple(Li / ni for Li, ni in zip(L, n))
lam0, mu0 = 60.0, 30.0
Nv = int(np.prod(n))

x = jnp.asarray(rng.normal(size=(3, 3, Nv)))
y = jnp.asarray(rng.normal(size=(3, 3, Nv)))


def fft_(v):
    s = v.shape
    return jnp.fft.fftn(v.reshape(s[:-1] + n), axes=(-3, -2, -1)).reshape(s)


def ifft_(v):
    s = v.shape
    return jnp.fft.ifftn(v.reshape(s[:-1] + n), axes=(-3, -2, -1)).real.reshape(s)


# ── 1. parity with manual reimplementation, both schemes ────────────────────

for green_op in (GreenOperatorBasic(n, L, lam0, mu0), GreenOperatorWillot(n, L, lam0, mu0, dx)):
    gamma0 = Gamma0Operator(n, green_op)
    expected = ifft_(ddot42(green_op.G, fft_(x)))
    assert jnp.allclose(gamma0(x), expected), \
        f"Gamma0Operator parity failed for {type(green_op).__name__}"


# ── 2. self-adjointness ──────────────────────────────────────────────────────

for green_op in (GreenOperatorBasic(n, L, lam0, mu0), GreenOperatorWillot(n, L, lam0, mu0, dx)):
    gamma0 = Gamma0Operator(n, green_op)
    lhs = jnp.sum(gamma0(x) * y)
    rhs = jnp.sum(x * gamma0.T(y))
    rel = abs(lhs - rhs) / (abs(lhs) + 1e-30)
    assert rel < 1e-10, f"Gamma0Operator failed <Gx,y> == <x,Gy> for {type(green_op).__name__}: rel={rel}"


# ── 3. parity with dstrain_nw_cg's inline A_op (standard scheme) ────────────

C_field = jnp.asarray(rng.normal(size=(3, 3, 3, 3, Nv)))
v = jnp.asarray(rng.normal(size=(3, 3, Nv)))

green_op = GreenOperatorBasic(n, L, lam0, mu0)
gamma0 = Gamma0Operator(n, green_op)

Cv = jnp.einsum("ijklm,klm->ijm", C_field, v)
expected_A_op = ifft_(jnp.einsum("ijklm,klm->ijm", green_op.G, fft_(Cv)))

actual = gamma0(Cv)
assert jnp.allclose(actual, expected_A_op), \
    "Gamma0Operator(C:v) must match dstrain_nw_cg's inline A_op(v)"

print("test_operators_projection: all checks passed")
