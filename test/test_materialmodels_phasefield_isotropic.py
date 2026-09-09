"""
Standalone test for PhaseFieldIsotropic (materialmodels/phasefield/isotropic.py)
-- the autodiff-derived Amor-split/AT2 phase-field material.

Five checks
-----------
1. Undamaged limit (d=0): stress_and_tangent must reproduce the plain
   elastic stiffness_tensor() exactly -- g(0) = 1, so psi reduces to the
   ordinary elastic energy and its Hessian must be the ordinary elastic C.
2. Compression's stress and volumetric tangent are never degraded: under
   pure hydrostatic COMPRESSION (tr(eps) < 0, no deviatoric part), sigma
   and the volumetric-volumetric part of C_tan (C_iijj, proportional to the
   bulk modulus K -- see the check's own derivation) must be identical at
   d=0 and d=1 -- the actual bug this class exists to fix, relative to
   materialmodels.phasefield.degradation.degrade_stiffness_field, which
   degrades the whole tensor regardless of sign. Checked directly against
   that function to show the difference is real, not just asserted.
   C_tan's *shear* block is NOT expected to match, and correctly so: the
   Amor split routes all deviatoric energy into psi_pos unconditionally
   (see psi_split's docstring), so an infinitesimal shear perturbation on
   top of a compressive state immediately engages the degraded branch --
   only the purely volumetric response has to survive compression intact.
3. Tension IS degraded: under hydrostatic TENSION, stress and tangent at
   d=1 (k_res small) must be much smaller than at d=0.
4. psi_pos matches materialmodels.phasefield.driving_force.strain_energy_amor_split
   exactly, for a handful of random strain states -- confirms the refactor
   didn't change the physics, only who derives what from it.
5. Finite-difference cross-check of C_tan against central differences of
   stress_and_tangent's own sigma, at a strain state deliberately away from
   the tr(eps)=0 kink (see PhaseFieldIsotropic.psi_split's docstring and
   notes/AUTODIFF_CONSTITUTIVE.md -- <x>_+ has a genuine, non-autodiff-
   related discontinuous second derivative exactly at tr(eps)=0).

Usage
-----
    python test/test_materialmodels_phasefield_isotropic.py
"""

import sys
sys.path.insert(0, "src")

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)
import jax.numpy as jnp
import numpy as np

from materialmodels.phasefield.isotropic import PhaseFieldIsotropic
from materialmodels.phasefield.degradation import degrade_stiffness_field
from materialmodels.phasefield.driving_force import strain_energy_amor_split

mat = PhaseFieldIsotropic(E=3.0e3, nu=0.35, Gc=1.0e-3, k_res=1e-6, name="epoxy")

# ── 1. undamaged limit (d=0) reproduces the plain elastic stiffness ─────────

eps_test = jnp.array([[1.0e-3, 2.0e-4, 0.0], [2.0e-4, -5.0e-4, 0.0], [0.0, 0.0, 3.0e-4]])
sigma0, C0, psi_pos0 = mat.stress_and_tangent(eps_test, jnp.array(0.0))
C_elastic = mat.stiffness_tensor()
sigma_elastic = jnp.einsum("ijkl,kl->ij", C_elastic, 0.5 * (eps_test + eps_test.T))

err_sigma = float(jnp.max(jnp.abs(sigma0 - sigma_elastic)))
err_C = float(jnp.max(jnp.abs(C0 - C_elastic)))
print(f"[1] d=0 limit: max|sigma diff|={err_sigma:.3e}  max|C diff|={err_C:.3e}")
assert err_sigma < 1e-9 and err_C < 1e-9, "d=0 should reproduce the plain elastic model exactly"
print("[1] PASSED")

# ── 2. pure compression is never degraded ───────────────────────────────────

eps_compression = jnp.diag(jnp.array([-2.0e-3, -2.0e-3, -2.0e-3]))  # hydrostatic compression
sigma_d0, C_d0, _ = mat.stress_and_tangent(eps_compression, jnp.array(0.0))
sigma_d1, C_d1, _ = mat.stress_and_tangent(eps_compression, jnp.array(1.0))

# Bulk modulus recovered from C_tan's full volumetric-volumetric contraction:
# for isotropic C_ijkl = lam*d_ij*d_kl + mu*(d_ik*d_jl+d_il*d_jk), summing
# i=j and k=l gives C_iijj = 9*lam + 6*mu = 9*(lam + 2*mu/3) = 9*K exactly.
K_d0 = float(jnp.einsum("iijj->", C_d0)) / 9.0
K_d1 = float(jnp.einsum("iijj->", C_d1)) / 9.0

err_sigma_comp = float(jnp.max(jnp.abs(sigma_d1 - sigma_d0)))
err_K_comp = abs(K_d1 - K_d0)
print(f"[2] compression, d=0 vs d=1: max|sigma diff|={err_sigma_comp:.3e}  "
      f"K(d=0)={K_d0:.4f}  K(d=1)={K_d1:.4f}  |K diff|={err_K_comp:.3e}")
assert err_sigma_comp < 1e-9, "pure compression's stress must not be degraded at all"
assert err_K_comp < 1e-6, "pure compression's volumetric tangent (bulk modulus) must not be degraded"

# Show the difference is real: degrade_stiffness_field (the current,
# uniform-degradation approach) DOES change the compressive stress here.
C_field_1vox = C_elastic[..., None]
d_1vox = jnp.array([1.0])
C_uniform_d1 = degrade_stiffness_field(C_field_1vox, d_1vox, k=mat.k_res)[..., 0]
sigma_uniform_d1 = jnp.einsum("ijkl,kl->ij", C_uniform_d1, eps_compression)
sigma_uniform_d0 = jnp.einsum("ijkl,kl->ij", C_elastic, eps_compression)
uniform_diff = float(jnp.max(jnp.abs(sigma_uniform_d1 - sigma_uniform_d0)))
print(f"    for comparison, degrade_stiffness_field's compressive sigma DOES change: "
      f"max|diff|={uniform_diff:.3e}  (this is the bug PhaseFieldIsotropic fixes)")
assert uniform_diff > 1.0, "expected the old uniform-degradation path to visibly change under compression"
print("[2] PASSED")

# ── 3. hydrostatic tension IS degraded ──────────────────────────────────────

eps_tension = jnp.diag(jnp.array([2.0e-3, 2.0e-3, 2.0e-3]))
sigma_t_d0, C_t_d0, psi_pos_t = mat.stress_and_tangent(eps_tension, jnp.array(0.0))
sigma_t_d1, C_t_d1, _ = mat.stress_and_tangent(eps_tension, jnp.array(1.0))

ratio = float(jnp.max(jnp.abs(sigma_t_d1)) / jnp.max(jnp.abs(sigma_t_d0)))
print(f"[3] tension, |sigma(d=1)| / |sigma(d=0)| = {ratio:.3e}  (expect ~k_res = {mat.k_res:.1e})")
assert ratio < 1e-4, "hydrostatic tension at d=1 should be heavily degraded"
assert float(psi_pos_t) > 0.0, "tension should register a nonzero driving force"
print("[3] PASSED")

# ── 4. psi_pos matches the existing strain_energy_amor_split exactly ──────

rng = np.random.default_rng(0)
max_psi_diff = 0.0
for _ in range(20):
    eps_r = jnp.array(rng.normal(scale=1.0e-3, size=(3, 3)))
    _, _, psi_pos_new = mat.stress_and_tangent(eps_r, jnp.array(0.3))
    psi_pos_ref, _ = strain_energy_amor_split(
        (0.5 * (eps_r + eps_r.T))[:, :, None], jnp.array([mat.lam]), jnp.array([mat.mu])
    )
    max_psi_diff = max(max_psi_diff, float(jnp.abs(psi_pos_new - psi_pos_ref[0])))
print(f"[4] max|psi_pos_new - strain_energy_amor_split| over 20 random states: {max_psi_diff:.3e}")
assert max_psi_diff < 1e-10, "psi_pos should match the existing Amor-split formula exactly"
print("[4] PASSED")

# ── 5. finite-difference cross-check of C_tan, away from the tr(eps)=0 kink ─

eps_probe = jnp.array([[1.5e-3, 3.0e-4, -2.0e-4], [3.0e-4, 8.0e-4, 1.0e-4], [-2.0e-4, 1.0e-4, 6.0e-4]])
d_probe = jnp.array(0.4)
assert abs(float(jnp.trace(eps_probe))) > 1.0e-4, "probe state must be well away from the tr(eps)=0 kink"

_, C_ad, _ = mat.stress_and_tangent(eps_probe, d_probe)

h = 1.0e-6
max_rel_diff = 0.0
for i in range(3):
    for j in range(3):
        d_eps = jnp.zeros((3, 3)).at[i, j].set(h)
        sigma_plus, _, _ = mat.stress_and_tangent(eps_probe + d_eps, d_probe)
        sigma_minus, _, _ = mat.stress_and_tangent(eps_probe - d_eps, d_probe)
        C_fd_col = (sigma_plus - sigma_minus) / (2.0 * h)
        rel_diff = float(jnp.max(jnp.abs(C_ad[:, :, i, j] - C_fd_col)) / (jnp.max(jnp.abs(C_fd_col)) + 1e-30))
        max_rel_diff = max(max_rel_diff, rel_diff)
print(f"[5] max relative diff, autodiff C_tan vs. central-difference: {max_rel_diff:.3e}")
assert max_rel_diff < 1e-4, "autodiff tangent should match finite differences away from the kink"
print("[5] PASSED")

print("\ntest_materialmodels_phasefield_isotropic: all checks passed")
