"""
Standalone test for learning.extractors.effective_modulus -- the mixed-BC
(free-lateral-surface) modulus/Poisson's-ratio extractor factored out of
notebooks/structure-property_phi-sweep.ipynb's hand-rolled control/
stress_goal/eps_bar block.

Four checks
-----------
1. Homogeneous, single-phase material: a free-lateral-surface uniaxial test
   on a spatially uniform material has an EXACT analytical answer -- the
   material's own E and nu -- since there's no heterogeneity for the mixed
   BC to actually do anything with. component=(0, 0).
2. Same homogeneous material, component=(1, 1): isotropy means loading any
   axis must reproduce the identical E/nu -- also a symmetry check that the
   control-mask construction isn't accidentally direction-specific.
3. component=(0, 1) (off-diagonal) must raise ValueError -- there's no
   free-surface Poisson's-ratio pair for a shear probe.
4. Two-phase composite: cross-checked bit-for-bit against a standalone,
   hand-inlined copy of notebooks/lin-elastic_mixed-BC.ipynb's own manual
   control/stress_goal/solve_mechanics/eps_bar_out block, confirming the
   extraction was faithfully factored out, not just written to resemble it.

Usage
-----
    python test/test_learning_extractors.py
"""

import sys
sys.path.insert(0, "src")

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)
import jax.numpy as jnp
import numpy as np

from generation.rve import make_square_composite_rve
from materialmodels.elastic.isotropic import LinearElasticIsotropic
from problems.mechanics import solve_mechanics
from learning.extractors import effective_modulus

# ── 1. homogeneous material, component=(0, 0): exact analytical answer ─────

n_homog = (8, 8, 8)
L_homog = (1.0, 1.0, 1.0)
Nv_homog = int(np.prod(n_homog))
phase_homog = jnp.zeros(Nv_homog, dtype=int)
mat = LinearElasticIsotropic(E=7000.0, nu=0.3, name="homogeneous")

E00, nu00, conv00 = effective_modulus(phase_homog, [mat], n_homog, L_homog, component=(0, 0))
print(f"[1] homogeneous, component=(0,0): E={E00:.4f}  nu_01={nu00[1]:.4f}  nu_02={nu00[2]:.4f}  converged={conv00}")
assert conv00
assert abs(E00 - mat.E) / mat.E < 1e-6, "a homogeneous material's effective E must equal its own E exactly"
assert abs(nu00[1] - mat.nu) < 1e-6 and abs(nu00[2] - mat.nu) < 1e-6, \
    "a homogeneous material's effective nu must equal its own nu exactly"
print("[1] PASSED")

# ── 2. homogeneous material, component=(1, 1): isotropy/symmetry check ─────

E11, nu11, conv11 = effective_modulus(phase_homog, [mat], n_homog, L_homog, component=(1, 1))
print(f"[2] homogeneous, component=(1,1): E={E11:.4f}  nu_10={nu11[0]:.4f}  nu_12={nu11[2]:.4f}  converged={conv11}")
assert conv11
assert abs(E11 - E00) < 1e-6, "loading a different axis of an isotropic material must give the identical E"
assert abs(nu11[0] - mat.nu) < 1e-6 and abs(nu11[2] - mat.nu) < 1e-6
print("[2] PASSED")

# ── 3. off-diagonal component must raise ────────────────────────────────────

try:
    effective_modulus(phase_homog, [mat], n_homog, L_homog, component=(0, 1))
    raise AssertionError("an off-diagonal component should have raised ValueError")
except ValueError:
    pass
print("[3] PASSED")

# ── 4. two-phase composite, cross-checked against a hand-inlined reference ─

phase_np, n, L, phi_act = make_square_composite_rve(phi=0.35, r_fiber=0.1, dx=0.02, N_min=20)
phase = jnp.array(phase_np.reshape(-1))
materials = [
    LinearElasticIsotropic(E=3500.0, nu=0.35, name="epoxy matrix"),
    LinearElasticIsotropic(E=70000.0, nu=0.20, name="glass fibre"),
]

E_new, nu_new, conv_new = effective_modulus(phase, materials, n, L, component=(0, 0), eps0=1.0e-3)

# hand-inlined reference: notebooks/lin-elastic_mixed-BC.ipynb's own logic
eps_bar_ref = jnp.array([[1.0e-3, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
control_ref = ((0, 0, 0), (0, 1, 0), (0, 0, 1))
stress_goal_ref = jnp.zeros((3, 3))
results_ref = solve_mechanics(
    n, L, phase, materials, eps_bar_ref,
    formulation="displacement", control=control_ref, stress_goal=stress_goal_ref,
    toler_lin=1e-6, maxiter=200,
)
sol_ref = results_ref[0].solution
E_ref = float(jnp.mean(sol_ref.sigma[0, 0])) / float(sol_ref.eps_bar[0, 0])
nu_ref_1 = float(-sol_ref.eps_bar[1, 1] / sol_ref.eps_bar[0, 0])
nu_ref_2 = float(-sol_ref.eps_bar[2, 2] / sol_ref.eps_bar[0, 0])

print(f"[4] composite: E_new={E_new:.4f}  E_ref={E_ref:.4f}  "
      f"nu_new={nu_new[1]:.4f}/{nu_new[2]:.4f}  nu_ref={nu_ref_1:.4f}/{nu_ref_2:.4f}  converged={conv_new}")
assert conv_new
assert abs(E_new - E_ref) < 1e-9, "effective_modulus must reproduce the hand-inlined reference exactly"
assert abs(nu_new[1] - nu_ref_1) < 1e-9 and abs(nu_new[2] - nu_ref_2) < 1e-9
print("[4] PASSED")

print("\ntest_learning_extractors: all checks passed")
