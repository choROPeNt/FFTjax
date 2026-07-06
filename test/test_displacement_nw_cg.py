"""
Validation script for the displacement-based Newton-CG solver
(``solvers.mechanical.displacement_nw_cg.ddisp_nw_cg``).

Single-phase, homogeneous, isotropic 16x16x16 RVE — for a spatially uniform
material the solution must be a uniform field, so we can check against
closed-form elasticity results.

Case 1 — pure strain BC (control all zero): compare against the existing
strain-based solver ``solvers.mechanical.strain_nw_cg.solve_elastic`` for the same
prescribed macroscopic strain. Both solve the same physical problem through
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

Usage
-----
    python test/test_displacement_nw_cg.py
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["JAX_ENABLE_X64"] = "1"

import sys
sys.path.insert(0, "src")

import jax
import jax.numpy as jnp
import numpy as np

from mat_models.elastic     import LinearElasticIsotropic, assemble_C_field
from operators.green        import build_freq_grid, build_green_operator
from solvers.mechanical.strain_nw_cg  import solve_elastic
from solvers.mechanical.displacement_nw_cg import ddisp_nw_cg

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

G_glob = build_green_operator(xi_flat, mat.lam, mat.mu)
eps_ref, sigma_ref, _, it_ref, conv_ref = solve_elastic(
    n, C_field, G_glob, eps_bar, toler_lin=1e-10, maxiter=2000
)

eps_u, sigma_u, delta_u, eps_bar_out, it_u, conv_u = ddisp_nw_cg(
    n, C_field, xi_flat, eps_bar, control, stress_goal_zero, toler_lin=1e-10, maxiter=2000
)

print(f"strain-based CG : iters={int(it_ref)} converged={bool(conv_ref)}")
print(f"disp-based   CG : iters={int(it_u)} converged={bool(conv_u)}")

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

eps_u, sigma_u, delta_u, eps_bar_out, it_u, conv_u = ddisp_nw_cg(
    n, C_field, xi_flat, eps_bar_guess, control, stress_goal, toler_lin=1e-10, maxiter=2000
)

eps_11_analytic = sigma_11_goal / mat.E
eps_22_analytic = -mat.nu * sigma_11_goal / mat.E

print(f"disp-based CG : iters={int(it_u)} converged={bool(conv_u)}")
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
    jnp.einsum("ijkl,m->ijklm", mat.stiffness_tensor(),  1.0 - w)
    + jnp.einsum("ijkl,m->ijklm", mat2.stiffness_tensor(), w)
)

lam0 = 0.5 * (mat.lam + mat2.lam)
mu0  = 0.5 * (mat.mu + mat2.mu)
G_glob_het = build_green_operator(xi_flat, lam0, mu0)

control = ((0, 0, 0), (0, 0, 0), (0, 0, 0))

eps_ref, sigma_ref, _, it_ref, conv_ref = solve_elastic(
    n, C_field_het, G_glob_het, eps_bar, toler_lin=1e-12, maxiter=3000
)
eps_u, sigma_u, delta_u, eps_bar_out, it_u, conv_u = ddisp_nw_cg(
    n, C_field_het, xi_flat, eps_bar, control, stress_goal_zero, toler_lin=1e-12, maxiter=3000
)

print(f"strain-based CG : iters={int(it_ref)} converged={bool(conv_ref)}")
print(f"disp-based   CG : iters={int(it_u)} converged={bool(conv_u)}")

err_eps_rel = float(jnp.max(jnp.abs(eps_u - eps_ref))) / float(eps_bar[0, 0])
print(f"max |eps_disp - eps_strain| / eps_11 = {err_eps_rel:.3e}")

assert int(it_u) > 0   # sanity: this case must actually exercise CG iterations
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

eps_u, sigma_u, delta_u, eps_bar_out, it_u, conv_u = ddisp_nw_cg(
    n_het, C_field_sharp, xi_flat_het, eps_bar_guess, control, stress_goal,
    toler_lin=1e-8, maxiter=500,
)

sigma_mean = jnp.mean(sigma_u, axis=-1)
print(f"disp-based CG : iters={int(it_u)} converged={bool(conv_u)}")
print(f"mean sigma_11 = {float(sigma_mean[0,0]):.4f}  (goal {sigma_11_goal})")
print(f"mean sigma_22 = {float(sigma_mean[1,1]):.4e}  mean sigma_33 = {float(sigma_mean[2,2]):.4e}  (goal 0)")

assert bool(conv_u)
assert int(it_u) < 100   # regression guard: the sign bug made this never converge
assert abs(float(sigma_mean[0, 0]) - sigma_11_goal) < 1e-4
assert abs(float(sigma_mean[1, 1])) < 1e-4
assert abs(float(sigma_mean[2, 2])) < 1e-4
print("PASSED")
