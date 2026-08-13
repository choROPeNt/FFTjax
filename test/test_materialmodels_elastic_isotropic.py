"""
Standalone test for materialmodels.elastic.isotropic.LinearElasticIsotropic
and materialmodels.base.ConstitutiveModel.

Four checks
-----------
1. ConstitutiveModel is a real ABC -- can't be instantiated directly, and a
   subclass that doesn't implement stiffness_tensor() can't either.
2. LinearElasticIsotropic is-a ConstitutiveModel, and its stiffness_tensor/
   stiffness_voigt/stress_field/bulk_modulus/shear_modulus all match
   mat_models.elastic.LinearElasticIsotropic bit-identically -- this is a
   port onto the new ABC, not a reimplementation.
3. Drop-in compatibility with the existing assemble_C_field: a C_field built
   from the new class matches one built from the old class.
4. End-to-end: solve_mechanics on the composite RVE using the new class
   instead of the old one still reproduces tau_xy = 7.625369 MPa.

Usage
-----
    python -m pytest test/test_materialmodels_elastic_isotropic.py
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
sys.path.insert(0, "src")

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)
import jax.numpy as jnp
import pytest

from materialmodels.base import ConstitutiveModel
from materialmodels.elastic.isotropic import LinearElasticIsotropic as NewIsotropic
from mat_models.elastic import LinearElasticIsotropic as OldIsotropic, assemble_C_field


# ── 1. ConstitutiveModel is a real ABC ───────────────────────────────────────

with pytest.raises(TypeError):
    ConstitutiveModel()


class _Incomplete(ConstitutiveModel):
    pass


with pytest.raises(TypeError):
    _Incomplete()


# ── 2. bit-identical to the old class ────────────────────────────────────────

E, nu = 70.0e3, 0.20
new_mat = NewIsotropic(E=E, nu=nu, name="glass fiber")
old_mat = OldIsotropic(E=E, nu=nu, name="glass fiber")

assert isinstance(new_mat, ConstitutiveModel)
assert jnp.array_equal(new_mat.stiffness_tensor(), old_mat.stiffness_tensor())
assert jnp.array_equal(new_mat.stiffness_voigt(), old_mat.stiffness_voigt())
assert jnp.array_equal(new_mat.stiffness_voigt(engineering=True), old_mat.stiffness_voigt(engineering=True))
assert new_mat.bulk_modulus == old_mat.bulk_modulus
assert new_mat.shear_modulus == old_mat.shear_modulus
assert new_mat.lam == old_mat.lam and new_mat.mu == old_mat.mu

eps_test = jnp.zeros((3, 3, 1)).at[0, 0, 0].set(1e-3)
assert jnp.array_equal(new_mat.stress_field(eps_test), old_mat.stress_field(eps_test))


# ── 3. drop-in compatibility with assemble_C_field ───────────────────────────

phase = jnp.array([0, 1, 0, 1])
matrix_new = NewIsotropic(E=3.0e3, nu=0.35, name="epoxy matrix")
matrix_old = OldIsotropic(E=3.0e3, nu=0.35, name="epoxy matrix")

C_field_new = assemble_C_field([matrix_new, new_mat], phase)
C_field_old = assemble_C_field([matrix_old, old_mat], phase)
assert jnp.array_equal(C_field_new, C_field_old)


# ── 4. end-to-end parity through solve_mechanics ─────────────────────────────

from generation.rve import make_square_composite_rve
from problems.mechanics import solve_mechanics

phase_np, N, n, L, phi_act = make_square_composite_rve(
    phi=0.5, r_fiber=0.005, spacing=0.0002, N_min=32, nz=10,
)
phase_rve = jnp.array(phase_np.reshape(-1))

eps_bar = jnp.array([
    [0.0, 1.0e-3, 0.0],
    [1.0e-3, 0.0, 0.0],
    [0.0, 0.0, 0.0],
])

eps, sigma, delta, converged = solve_mechanics(
    n, L, phase_rve, [matrix_new, new_mat], eps_bar,
    formulation="lippmann_schwinger", scheme="rotated",
    toler_lin=1e-6, maxiter=1000,
)
tau_xy = float(jnp.mean(sigma[1, 0]))
assert bool(converged)
assert abs(tau_xy - 7.625369073063829) < 1e-6, f"tau_xy mismatch: got {tau_xy}"

print("test_materialmodels_elastic_isotropic: all checks passed")
print(f"  end-to-end tau_xy (avg) = {tau_xy:.6f} MPa")
