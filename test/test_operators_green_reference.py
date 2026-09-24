"""
Standalone test for operators.green.build_reference_green_operator's
handling of a per-voxel-field material (TransverseIsotropic with fiber_dir
given as a (3, Nv) orientation field, e.g. straight from a TexGen VTU's own
YarnTangent field via utils.io.reader.read_vtu) -- previously crashed with a
jnp.stack shape mismatch: (3,3,3,3) from a plain material next to
(3,3,3,3,Nv) from a field-valued one. Fixed by spatially-averaging any
per-voxel elastic_stiffness_tensor() before the cross-material mean.

Three checks
------------
1. All-constant-tensor materials (the pre-existing case): matches a
   hand-computed mean lam0/mu0.
2. A uniform per-voxel field (every voxel the same orientation) gives
   *exactly* the same lam0/mu0 as passing that one direction as a plain
   scalar fiber_dir -- the field-averaging path and the scalar path must
   agree when the field has nothing to average over.
3. A genuinely varying per-voxel field matches a hand-computed reference:
   spatially average the field material's own stiffness first, then average
   across materials -- confirming *what* gets averaged, not just that
   nothing crashes.

Usage
-----
    python test/test_operators_green_reference.py
"""

import sys
sys.path.insert(0, "src")

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)
import jax.numpy as jnp
import numpy as np

from materialmodels.elastic.isotropic import LinearElasticIsotropic
from materialmodels.elastic.transverse_isotropic import TransverseIsotropic
from materialmodels.tensors import isotropic_equivalent_lame
from operators.green import build_reference_green_operator

n = (4, 4, 4)
L = (1.0, 1.0, 1.0)

matrix = LinearElasticIsotropic(E=3.5e3, nu=0.35, name="epoxy matrix")

# ── 1. all-constant-tensor materials (pre-existing case, unaffected) ───────

fiber_scalar = TransverseIsotropic(E_L=80.0e3, E_T=8.0e3, G_LT=4.0e3, nu_LT=0.25, G_TT=3.0e3,
                                    fiber_dir=[0.0, 0.0, 1.0], name="glass fiber")
green_scalar = build_reference_green_operator(n, L, [matrix, fiber_scalar], scheme="rotated")

C_mean_expected = jnp.mean(jnp.stack([matrix.elastic_stiffness_tensor(), fiber_scalar.elastic_stiffness_tensor()]), axis=0)
lam0_expected, mu0_expected = isotropic_equivalent_lame(C_mean_expected)
assert jnp.allclose(green_scalar.lam0, lam0_expected) and jnp.allclose(green_scalar.mu0, mu0_expected)
print(f"[1] constant-tensor materials: lam0={float(green_scalar.lam0):.4f}  mu0={float(green_scalar.mu0):.4f}")
print("[1] PASSED")

# ── 2. uniform per-voxel field matches the scalar-direction case exactly ───

Nv = int(np.prod(n))
uniform_field = jnp.tile(jnp.array([0.0, 0.0, 1.0])[:, None], (1, Nv))  # same direction, every voxel
fiber_field_uniform = TransverseIsotropic(E_L=80.0e3, E_T=8.0e3, G_LT=4.0e3, nu_LT=0.25, G_TT=3.0e3,
                                           fiber_dir=uniform_field, name="glass fiber (field)")
assert fiber_field_uniform.elastic_stiffness_tensor().shape == (3, 3, 3, 3, Nv), \
    "sanity: fiber_dir as a (3, Nv) field should make elastic_stiffness_tensor() per-voxel"

green_field_uniform = build_reference_green_operator(n, L, [matrix, fiber_field_uniform], scheme="rotated")
assert jnp.allclose(green_field_uniform.lam0, green_scalar.lam0, atol=1e-8)
assert jnp.allclose(green_field_uniform.mu0, green_scalar.mu0, atol=1e-8)
print(f"[2] uniform field vs. scalar direction: "
      f"lam0 diff={float(jnp.abs(green_field_uniform.lam0 - green_scalar.lam0)):.2e}, "
      f"mu0 diff={float(jnp.abs(green_field_uniform.mu0 - green_scalar.mu0)):.2e}")
print("[2] PASSED")

# ── 3. a genuinely varying field matches a hand-computed reference ─────────

rng = np.random.default_rng(0)
theta = rng.uniform(0.0, 2.0 * np.pi, Nv)
varying_field = jnp.array(np.stack([np.cos(theta), np.sin(theta), np.zeros(Nv)]))
fiber_field_varying = TransverseIsotropic(E_L=80.0e3, E_T=8.0e3, G_LT=4.0e3, nu_LT=0.25, G_TT=3.0e3,
                                           fiber_dir=varying_field, name="glass fiber (varying field)")

green_field_varying = build_reference_green_operator(n, L, [matrix, fiber_field_varying], scheme="rotated")

fiber_C_spatial_mean = jnp.mean(fiber_field_varying.elastic_stiffness_tensor(), axis=-1)
C_mean_ref = jnp.mean(jnp.stack([matrix.elastic_stiffness_tensor(), fiber_C_spatial_mean]), axis=0)
lam0_ref, mu0_ref = isotropic_equivalent_lame(C_mean_ref)

assert jnp.allclose(green_field_varying.lam0, lam0_ref, atol=1e-8)
assert jnp.allclose(green_field_varying.mu0, mu0_ref, atol=1e-8)
print(f"[3] varying field: lam0={float(green_field_varying.lam0):.4f}  mu0={float(green_field_varying.mu0):.4f}  "
      f"(matches spatial-mean-then-material-mean reference)")
print("[3] PASSED")

print("\ntest_operators_green_reference: all checks passed")
