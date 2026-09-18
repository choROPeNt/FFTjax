"""
Standalone test for operators.galerkin.build_galerkin_projection_operator --
the reference-medium-free Fourier-Galerkin compatibility projector.

Five checks
-----------
1. Major symmetry: Gs_ijkl == Gs_klij.
2. Minor symmetry: Gs_ijkl == Gs_jikl == Gs_ijlk.
3. Zero at the DC frequency (xi = 0).
4. Idempotency: Gs : Gs == Gs (a true projector).
5. Fixed-point / annihilation: a compatible field sym(n_hat outer v) is left
   unchanged by the projection; an n_hat-orthogonal (incompatible) field is
   projected to zero.

These are the exact properties verified with plain numpy before this
operator was implemented (see notes/FOURIER_GALERKIN.md) -- this ports that
check onto jax.numpy against the actual production function, for a handful
of frequencies from a real build_freq_grid grid rather than one hand-picked
direction.

Usage
-----
    python test/test_operators_green_galerkin.py
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
sys.path.insert(0, "src")

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)
import jax.numpy as jnp

from materialmodels.tensors import is_major_symmetric, is_minor_symmetric
from operators.galerkin import build_galerkin_projection_operator
from operators.green import build_freq_grid

n = (6, 5, 4)
L = (1.0, 1.0, 1.0)
xi_flat = build_freq_grid(n, L)
G = build_galerkin_projection_operator(xi_flat, scheme='standard')

# ── 1/2. symmetry ────────────────────────────────────────────────────────────

# is_major_symmetric/is_minor_symmetric expect a single (3,3,3,3) tensor --
# check every non-DC voxel individually.
xi_sq = jnp.sum(xi_flat ** 2, axis=0)
nonzero_idx = [i for i in range(xi_sq.shape[0]) if float(xi_sq[i]) > 0]

for i in nonzero_idx:
    Gi = G[..., i]  # (3, 3, 3, 3)
    assert is_major_symmetric(Gi), f"voxel {i}: major symmetry failed"
    assert is_minor_symmetric(Gi), f"voxel {i}: minor symmetry failed"

print("[1] major symmetry: PASSED (all non-DC voxels)")
print("[2] minor symmetry: PASSED (all non-DC voxels)")

# ── 3. zero at DC ─────────────────────────────────────────────────────────────

dc_idx = [i for i in range(xi_sq.shape[0]) if float(xi_sq[i]) == 0]
assert len(dc_idx) == 1, "expected exactly one DC (xi=0) voxel on this grid"
G_dc = G[..., dc_idx[0]]
assert float(jnp.max(jnp.abs(G_dc))) == 0.0, "projector must be exactly zero at xi=0"
print("[3] zero at DC: PASSED")

# ── 4. idempotency ────────────────────────────────────────────────────────────

GG = jnp.einsum('ijmnN,mnklN->ijklN', G, G)
err_idem = float(jnp.max(jnp.abs(GG - G)))
print(f"[4] idempotency: max|G:G - G| = {err_idem:.3e}")
assert err_idem < 1e-8, f"projector must satisfy G:G == G: err={err_idem:.3e}"
print("[4] PASSED")

# ── 5. fixed-point on compatible fields, annihilation on incompatible ones ────

max_fixed_err = 0.0
max_annih_err = 0.0
for i in nonzero_idx:
    xi_i = xi_flat[:, i]
    n_hat = xi_i / jnp.linalg.norm(xi_i)

    v = jnp.array([1.3, -0.7, 2.1])  # arbitrary, fixed across voxels
    A = 0.5 * (jnp.outer(n_hat, v) + jnp.outer(v, n_hat))     # compatible: sym(n_hat outer v)
    GA = jnp.einsum('ijkl,kl->ij', G[..., i], A)
    max_fixed_err = max(max_fixed_err, float(jnp.max(jnp.abs(GA - A))))

    # build an orthonormal basis {a, b} of the plane orthogonal to n_hat
    ref = jnp.array([1.0, 0.0, 0.0]) if abs(float(n_hat[0])) < 0.9 else jnp.array([0.0, 1.0, 0.0])
    a = jnp.cross(n_hat, ref)
    a = a / jnp.linalg.norm(a)
    b = jnp.cross(n_hat, a)
    Bt = 0.5 * (jnp.outer(a, b) + jnp.outer(b, a))             # incompatible: n_hat-orthogonal
    GB = jnp.einsum('ijkl,kl->ij', G[..., i], Bt)
    max_annih_err = max(max_annih_err, float(jnp.max(jnp.abs(GB))))

print(f"[5] compatible field fixed-point: max err = {max_fixed_err:.3e}")
print(f"[5] incompatible field annihilated: max err = {max_annih_err:.3e}")
assert max_fixed_err < 1e-8, f"compatible field must be a fixed point: err={max_fixed_err:.3e}"
assert max_annih_err < 1e-8, f"incompatible field must be annihilated: err={max_annih_err:.3e}"
print("[5] PASSED")

print("\ntest_operators_green_galerkin: all checks passed")
