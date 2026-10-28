"""
Validation script for the displacement-based Newton-CG solver
(``solvers.elliptic.vector.displacement_based.solve_displacement_based``).

Single-phase, homogeneous, isotropic 16x16x16 RVE — for a spatially uniform
material the solution must be a uniform field, so we can check against
closed-form elasticity results.

Case 1 — pure strain BC (control all zero): compare against the existing
strain-based solver ``solvers.elliptic.vector.lippmann_schwinger.solve_lippmann_schwinger``
for the same prescribed macroscopic strain. Both solve the same physical problem through
different unknowns (displacement vs. strain), so the resulting fields must
agree.

Case 2 — mixed BC, uniaxial stress: normal components stress-controlled with
target sigma_11 = 100 MPa, sigma_22 = sigma_33 = 0; shear components
strain-controlled at zero. The macroscopic strain solved for must match the
classical isotropic uniaxial-stress relations
    eps_11 = sigma_11 / E,   eps_22 = eps_33 = -nu * sigma_11 / E

Case 3 — heterogeneous, pure strain BC: a smooth (single-Fourier-mode, i.e.
band-limited) two-material blend, so both solvers converge to the same
discrete solution to CG tolerance. This is what actually exercises the
Newton-CG iterations for the displacement solver (cases 1-2 are homogeneous
and solve in 0-2 iterations). Note: for materials with sharp interfaces
(step-function phase fields), the displacement-based (Fourier-Galerkin) and
strain-based (Lippmann-Schwinger / Green's-operator) schemes are expected to
disagree at the level of a few percent — a well-known discretization
difference between the two families of FFT homogenization schemes, not a bug.

Case 4 — heterogeneous (sharp interface), mixed BC: the du<->sv cross-coupling
terms only enter the operator for *heterogeneous* materials (they vanish
identically for homogeneous C, which is why cases 1-2 above cannot catch a
sign error in that coupling). A random per-voxel two-phase field with high
stiffness contrast is a worst-case (sharp-interface) exercise of that
coupling.

Case 5 — heterogeneous, stress-controlled SHEAR: uniaxial tension of a 45-deg
layered cell with every other component (shear included) stress-free. First
checks that sm2sv is the adjoint of sv2sm, which keeps the shear rows of the
bordered operator symmetric, then solves the case end to end.

Usage
-----
    python test/test_displacement_nw_cg.py
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.makedirs("output", exist_ok=True)

import sys
sys.path.insert(0, "src")

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)
import jax
import jax.numpy as jnp
import numpy as np

from materialmodels.elastic.isotropic import LinearElasticIsotropic
from materialmodels.assembly import assemble_C_field
from operators.green        import build_freq_grid, GreenOperatorBasic
from solvers.elliptic.vector.lippmann_schwinger import solve_lippmann_schwinger
from solvers.elliptic.vector.displacement_based import solve_displacement_based
from solvers.elliptic.vector.mixed_bc import _active_pairs, sm2sv, sv2sm

jax.config.update("jax_enable_x64", True)

n  = (16, 16, 16)
L  = (1.0, 1.0, 1.0)
Nv = int(np.prod(n))

mat = LinearElasticIsotropic(E=210e3, nu=0.3, name="steel")
phase = jnp.zeros(Nv, dtype=int)
C_field = assemble_C_field([mat], phase)

xi_flat = build_freq_grid(n, L)

# ── Case 1 — pure strain BC ───────────────────────────────────────────────────
print("── Case 1: pure strain BC ───────────────────────────────────────────")

eps_bar = jnp.array([
    [1.0e-3, 0.0, 0.0],
    [0.0,    0.0, 0.0],
    [0.0,    0.0, 0.0],
])
control = ((0, 0, 0), (0, 0, 0), (0, 0, 0))
stress_goal_zero = jnp.zeros((3, 3))

green_op = GreenOperatorBasic(n, L, mat.lam, mat.mu)
eps_ref, sigma_ref, _, conv_ref = solve_lippmann_schwinger(
    n, C_field, green_op, eps_bar, toler_lin=1e-10, maxiter=2000
)

eps_u, sigma_u, delta_u, eps_bar_out, conv_u = solve_displacement_based(
    n, C_field, xi_flat, eps_bar, control, stress_goal_zero, toler_lin=1e-10, maxiter=2000
)

print(f"strain-based CG : converged={bool(conv_ref)}")
print(f"disp-based   CG : converged={bool(conv_u)}")

err_eps   = float(jnp.max(jnp.abs(eps_u - eps_ref)))
err_sigma = float(jnp.max(jnp.abs(sigma_u - sigma_ref)))
print(f"max |eps_disp - eps_strain|     = {err_eps:.3e}")
print(f"max |sigma_disp - sigma_strain| = {err_sigma:.3e}")

assert err_eps < 1e-8
assert err_sigma < 1e-4
assert jnp.allclose(eps_bar_out, eps_bar)
print("PASSED\n")

# ── Case 2 — mixed BC, uniaxial stress ────────────────────────────────────────
print("── Case 2: mixed BC, uniaxial stress ────────────────────────────────")

sigma_11_goal = 100.0
control = ((1, 0, 0), (0, 1, 0), (0, 0, 1))   # normal components stress-controlled
eps_bar_guess = jnp.zeros((3, 3))              # shear (strain-controlled) entries = 0
stress_goal = jnp.array([
    [sigma_11_goal, 0.0, 0.0],
    [0.0,           0.0, 0.0],
    [0.0,           0.0, 0.0],
])

eps_u, sigma_u, delta_u, eps_bar_out, conv_u = solve_displacement_based(
    n, C_field, xi_flat, eps_bar_guess, control, stress_goal, toler_lin=1e-10, maxiter=2000
)

eps_11_analytic = sigma_11_goal / mat.E
eps_22_analytic = -mat.nu * sigma_11_goal / mat.E

print(f"disp-based CG : converged={bool(conv_u)}")
print(f"eps_bar_out =\n{np.asarray(eps_bar_out)}")
print(f"eps_11: solved={float(eps_bar_out[0,0]):.6e}  analytic={eps_11_analytic:.6e}")
print(f"eps_22: solved={float(eps_bar_out[1,1]):.6e}  analytic={eps_22_analytic:.6e}")

sigma_mean = jnp.mean(sigma_u, axis=-1)
print(f"mean sigma_11 = {float(sigma_mean[0,0]):.4f}  (goal {sigma_11_goal})")
print(f"mean sigma_22 = {float(sigma_mean[1,1]):.4e}  (goal 0)")

assert abs(float(eps_bar_out[0, 0]) - eps_11_analytic) < 1e-8
assert abs(float(eps_bar_out[1, 1]) - eps_22_analytic) < 1e-8
assert abs(float(eps_bar_out[2, 2]) - eps_22_analytic) < 1e-8
assert abs(float(sigma_mean[0, 0]) - sigma_11_goal) < 1e-6
assert abs(float(sigma_mean[1, 1])) < 1e-6
print("PASSED\n")

# ── Case 3 — heterogeneous (smooth), pure strain BC ───────────────────────────
print("── Case 3: heterogeneous (smooth), pure strain BC ───────────────────")

mat2 = LinearElasticIsotropic(E=70e3, nu=0.33, name="aluminium")
xs = np.linspace(0.0, 1.0, n[0], endpoint=False)
X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
w = jnp.asarray(0.5 + 0.5 * np.sin(2.0 * np.pi * X)).ravel()   # smooth blend weight in [0,1]

C_field_het = (
    jnp.einsum("ijkl,m->ijklm", mat.elastic_stiffness_tensor(),  1.0 - w)
    + jnp.einsum("ijkl,m->ijklm", mat2.elastic_stiffness_tensor(), w)
)

lam0 = 0.5 * (mat.lam + mat2.lam)
mu0  = 0.5 * (mat.mu + mat2.mu)
green_op_het = GreenOperatorBasic(n, L, lam0, mu0)

control = ((0, 0, 0), (0, 0, 0), (0, 0, 0))

eps_ref, sigma_ref, _, conv_ref = solve_lippmann_schwinger(
    n, C_field_het, green_op_het, eps_bar, toler_lin=1e-12, maxiter=3000
)
eps_u, sigma_u, delta_u, eps_bar_out, conv_u = solve_displacement_based(
    n, C_field_het, xi_flat, eps_bar, control, stress_goal_zero, toler_lin=1e-12, maxiter=3000
)

print(f"strain-based CG : converged={bool(conv_ref)}")
print(f"disp-based   CG : converged={bool(conv_u)}")

err_eps_rel = float(jnp.max(jnp.abs(eps_u - eps_ref))) / float(eps_bar[0, 0])
print(f"max |eps_disp - eps_strain| / eps_11 = {err_eps_rel:.3e}")

assert err_eps_rel < 1e-3
print("PASSED\n")

# ── Case 4 — heterogeneous (sharp interface), mixed BC ────────────────────────
# Regression test: the du<->sv cross-coupling terms only enter the operator
# for *heterogeneous* materials (they vanish identically for homogeneous C,
# which is why cases 1-2 above cannot catch a sign error in that coupling).
# A random per-voxel two-phase field with high stiffness contrast is a
# worst-case (sharp-interface) exercise of that coupling.
print("── Case 4: heterogeneous (sharp interface), mixed BC ────────────────")

n_het = (8, 8, 8)
Nv_het = int(np.prod(n_het))
rng = np.random.default_rng(0)
phase_het = jnp.array(rng.integers(0, 2, size=Nv_het))
C_field_sharp = assemble_C_field([mat, mat2], phase_het)
xi_flat_het = build_freq_grid(n_het, L)

control = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
eps_bar_guess = jnp.zeros((3, 3))
sigma_11_goal = 50.0
stress_goal = jnp.array([
    [sigma_11_goal, 0.0, 0.0],
    [0.0,           0.0, 0.0],
    [0.0,           0.0, 0.0],
])

eps_u, sigma_u, delta_u, eps_bar_out, conv_u = solve_displacement_based(
    n_het, C_field_sharp, xi_flat_het, eps_bar_guess, control, stress_goal,
    toler_lin=1e-8, maxiter=500,
)

sigma_mean = jnp.mean(sigma_u, axis=-1)
print(f"disp-based CG : converged={bool(conv_u)}")
print(f"mean sigma_11 = {float(sigma_mean[0,0]):.4f}  (goal {sigma_11_goal})")
print(f"mean sigma_22 = {float(sigma_mean[1,1]):.4e}  mean sigma_33 = {float(sigma_mean[2,2]):.4e}  (goal 0)")

assert bool(conv_u)  # regression guard: the sign bug made this never converge
assert abs(float(sigma_mean[0, 0]) - sigma_11_goal) < 1e-4
assert abs(float(sigma_mean[1, 1])) < 1e-4
assert abs(float(sigma_mean[2, 2])) < 1e-4
print("PASSED\n")

# ── Case 5 — heterogeneous, stress-controlled SHEAR ──────────────────────────
# Uniaxial tension of an anisotropic cell: eps_11 prescribed, EVERY other
# component (shear included) stress-free. Layers inclined at 45 deg in the
# x-y plane couple normal loading into shear, so a correct solve needs a
# nonzero macroscopic eps_12.
print("── Case 5: heterogeneous, stress-controlled shear ───────────────────")

control = ((0, 1, 1), (1, 1, 1), (1, 1, 1))
pairs = _active_pairs(control)

# The bordered operator is symmetric only if sm2sv is the ADJOINT of sv2sm:
# <sm2sv(S), v> == <S, sv2sm(v)>_F for any S, symmetric or not. The old map
# (sm[i, j] alone) fails this on every shear pair. The elastic solve below
# converges either way, so this is the check that actually guards it.
S = jnp.array(rng.standard_normal((3, 3)))
v = jnp.array(rng.standard_normal(len(pairs)))
adj_err = abs(float(jnp.dot(sm2sv(S, pairs), v) - jnp.sum(S * sv2sm(v, pairs))))
print(f"adjoint identity error = {adj_err:.3e}")
assert adj_err < 1e-12

ix, iy, _ = np.meshgrid(*[np.arange(n_het[0])] * 3, indexing="ij")
phase_layers = jnp.array((((ix + iy) % 4) < 2).astype(int).ravel())
C_field_layers = assemble_C_field([mat, mat2], phase_layers)
eps_bar_guess = jnp.zeros((3, 3)).at[0, 0].set(1e-3)

eps_u, sigma_u, delta_u, eps_bar_out, conv_u = solve_displacement_based(
    n_het, C_field_layers, xi_flat_het, eps_bar_guess, control, jnp.zeros((3, 3)),
    toler_lin=1e-10, maxiter=2000,
)
sigma_mean = jnp.mean(sigma_u, axis=-1)
free = np.array(control, dtype=bool)
print(f"disp-based CG : converged={bool(conv_u)}")
print(f"mean sigma_11 = {float(sigma_mean[0, 0]):.4f}   max |free sigma| = "
      f"{float(jnp.max(jnp.abs(sigma_mean[free]))):.3e}")
print(f"macroscopic eps_12 = {float(eps_bar_out[0, 1]):.4e}  eps_22 = {float(eps_bar_out[1, 1]):.4e}")

assert bool(conv_u)
assert float(jnp.max(jnp.abs(sigma_mean[free]))) < 1e-6 * abs(float(sigma_mean[0, 0]))
assert abs(float(eps_bar_out[0, 1])) > 1e-5    # the layering must actually shear
print("PASSED")
