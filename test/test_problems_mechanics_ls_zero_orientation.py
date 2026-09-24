"""
End-to-end regression test: the Lippmann-Schwinger solve (either scheme)
must converge, and agree with the displacement formulation, on a composite
RVE where the oriented (TransverseIsotropic, fiber_dir given as a per-voxel
field) material's phase doesn't cover the whole grid -- the normal case for
real data, since utils.io.reader.read_vtu writes an exactly-zero orientation
vector at every voxel that ISN'T the oriented material's own phase (see its
own docstring).

The bug this guards against
----------------------------
materialmodels.tensors.rotation_from_direction used to do `d / norm(d)`
unconditionally -- 0/0 = NaN for a zero direction. TransverseIsotropic.
elastic_stiffness_tensor() (per-voxel field mode) is NaN at every such
voxel. materialmodels.assembly.assemble_C_field masks those out correctly
by phase (jnp.where), so formulation="displacement" -- which only ever
touches the assembled C_field -- was never affected. But
operators.green.build_reference_green_operator spatially averages each
material's OWN elastic_stiffness_tensor() (to build the Lippmann-Schwinger
reference medium) BEFORE any phase masking happens, over the WHOLE grid --
so the NaN at the oriented material's own "not really this phase" voxels
poisoned lam0/mu0, then the Green's operator, then the polarization
right-hand-side b for the CG solve. jax.scipy.sparse.linalg.cg's while_loop
condition is False for NaN, so it exits at iteration 0 and returns x=x0=0
-- delta=0 *exactly*, converged=False (nan <= nan), and the reported
"solution" collapses to the trivial eps=eps_bar-everywhere state (the Voigt/
parallel bound), which is why the two schemes agreed with each other
(neither did any real work) while disagreeing substantially with the
correctly-converging displacement formulation. Verified live: on a real
weave-derived RVE this was as far off as ~3x for the coarsest grid tested.

Fixed by making rotation_from_direction fall back to the identity rotation
for a zero/degenerate direction instead of NaN (see
test_materialmodels_tensors.py check 5 for the unit-level check;
this file is the full solve-level regression).

Usage
-----
    python test/test_problems_mechanics_ls_zero_orientation.py
"""

import sys
sys.path.insert(0, "src")

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)
import jax.numpy as jnp
import numpy as np

from generation.rve import make_random_composite_rve
from materialmodels.elastic.isotropic import LinearElasticIsotropic
from materialmodels.elastic.transverse_isotropic import TransverseIsotropic
from problems.mechanics import solve_mechanics

phase_np, n, L, phi_act, centres = make_random_composite_rve(
    phi=0.35, r_fiber=0.0035, dx=0.0006, size_in_r=10, nz=1, K=15, seed=67,
)
Nv = int(np.prod(n))
phase = jnp.array(phase_np.reshape(-1))

# read_vtu's own convention: a real direction at yarn (phase 1) voxels,
# EXACTLY zero everywhere else -- not "some other fixed direction".
orientations_np = np.zeros((3, Nv))
orientations_np[2, np.asarray(phase_np.reshape(-1)) == 1] = 1.0
orientations = jnp.array(orientations_np)

matrix = LinearElasticIsotropic(E=3.5e3, nu=0.35, name="epoxy matrix")
fiber = TransverseIsotropic(E_L=170.905e3, E_T=10.814e3, G_LT=6.9e3, nu_LT=0.245, G_TT=4.245e3,
                             fiber_dir=orientations, name="carbon fiber yarn")
materials = [matrix, fiber]

eps_bar = jnp.array([[1.0e-3, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

E_eff = {}
for label, formulation, scheme in [("ls_rotated", "lippmann_schwinger", "rotated"),
                                    ("ls_standard", "lippmann_schwinger", "standard"),
                                    ("displacement", "displacement", "rotated")]:
    results = solve_mechanics(n, L, phase, materials, eps_bar, formulation=formulation,
                               scheme=scheme, toler_lin=1e-4, maxiter=1000)
    sol = results[0].solution
    assert bool(sol.converged), f"{label}: expected convergence, got converged=False"
    assert not bool(jnp.any(jnp.isnan(sol.sigma))), f"{label}: NaN in the stress field"
    E_eff[label] = float(jnp.mean(sol.sigma[0, 0]) / jnp.mean(sol.eps[0, 0]))
    print(f"[{label}] converged=True  E_eff={E_eff[label]:.2f} MPa")

# The two Lippmann-Schwinger schemes and the displacement formulation solve
# the same physical problem three different numerical ways -- they should
# agree closely (a few tenths of a percent, consistent with
# test_problems_mechanics.py's own cross-formulation tolerance), not be
# identical (that would itself indicate the Green's operator isn't doing
# anything, i.e. exactly this bug) and not differ by tens of percent.
ref = E_eff["displacement"]
for label in ("ls_rotated", "ls_standard"):
    rel_diff = abs(E_eff[label] - ref) / ref
    print(f"  {label} vs. displacement: rel. diff = {rel_diff:.4%}")
    assert rel_diff < 0.01, f"{label} differs from displacement by {rel_diff:.2%}, expected < 1%"

assert E_eff["ls_rotated"] != E_eff["ls_standard"], \
    "the two Green's operator schemes giving a bit-identical answer is itself the bug's signature"

print("\ntest_problems_mechanics_ls_zero_orientation: all checks passed")
