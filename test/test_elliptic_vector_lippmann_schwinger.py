"""
Standalone test for solve_lippmann_schwinger
(solvers/elliptic/vector/lippmann_schwinger.py).

Five checks
-----------
1. Correctness invariants on a synthetic random heterogeneous problem, both
   Green's-operator schemes -- independent of any other solver
   implementation (solvers.mechanical.strain_nw_cg, this module's former
   parity reference, is retired):
   (a) mean(eps) == eps_bar exactly. Gamma0Operator is zero at the DC
       (zero-frequency) mode by construction (see green.py's n_hat comment),
       so A(v) can't see v's DC component at all -- CG starts at x0=0 and
       never moves in that direction, so delta's mean is exactly 0.
   (b) the returned delta actually solves A(delta) = b to toler_lin,
       recomputed independently of cg_solve's own internal convergence
       tracking -- this would catch a bug where the returned value doesn't
       correspond to what was actually fed into the CG solve.
2. Real composite RVE (glass fiber / epoxy, same setup as
   notebooks/mechanics/lin-elastic_strain.ipynb) -- converges, and reproduces the
   known tau_xy (avg) = 7.625369 MPa result.
3. LippmannSchwingerSolver (the ElasticitySolver wrapper) reproduces
   solve_lippmann_schwinger's own output exactly -- it's a thin wrapper,
   this is a trivial-but-real check that it doesn't lose or reorder fields.
4. solve_lippmann_schwinger_mixed_bc (Michel et al 1999 outer-loop mixed-BC
   route, solvers/elliptic/vector/lippmann_schwinger.py) -- solve_
   lippmann_schwinger itself (checks 1-3 above) is untouched by this
   function's existence, this only checks the new outer loop:
   (a) reproduces the same closed-form homogeneous uniaxial-stress target as
       test_displacement_nw_cg.py's case 2 and test_elliptic_vector_
       fourier_galerkin.py's check 3.
   (b) agrees with formulation="displacement"'s mixed-BC result on the
       check-2 composite RVE within the same cross-formulation tolerance
       used elsewhere for that RVE.
   (c) a zero macro_stress_goal on the controlled directions (a free/
       traction-free surface, the common case -- e.g. notebooks/mechanics/
       lin-elastic_mixed-BC.ipynb) still reports converged=True once the
       residual is actually at noise level, on check 1's fast synthetic
       problem. Regression test for a real bug found while building that
       notebook's solver-comparison companion: the outer-loop's relative-
       tolerance check originally scaled by macro_stress_goal's own norm,
       which collapses to the function's ``+ 1e-30`` floor when that target
       is exactly zero, making convergence practically unreachable even
       though the physical answer was already correct (confirmed against
       (b)'s displacement cross-check on the composite RVE, before this was
       isolated down to a fast, targeted repro) -- fixed by scaling against
       the current mean-stress norm instead.
5. The outer-loop's default reference stiffness (``C0=None``) uses the true
   (anisotropic) voxel-mean ``jnp.mean(C_field, axis=-1)``, not green_op's
   own isotropized (lam0, mu0) reference medium. Regression test for a real
   divergence bug: with a transversely isotropic fibre (E_L/E_T ~ 15x) under
   mixed BC, the isotropized default made this UNDAMPED correction step
   diverge geometrically (eps_bar reaching ~1e21 within maxiter_outer,
   converged=False) instead of converging -- reproduced building the same
   notebook check 4c/4b came from, before this was isolated down to a fast
   synthetic repro. The true voxel-mean fixes it: converges cleanly and
   agrees with formulation="displacement" on the same problem.

Usage
-----
    python -m pytest test/test_elliptic_vector_lippmann_schwinger.py
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
from operators.green import GreenOperatorBasic, GreenOperatorWillot, build_freq_grid
from operators.projection import Gamma0Operator
from solvers.elliptic.vector.displacement_based import solve_displacement_based
from solvers.elliptic.vector.lippmann_schwinger import (
    solve_lippmann_schwinger, solve_lippmann_schwinger_mixed_bc, LippmannSchwingerSolver,
)


rng = np.random.default_rng(2)

n = (4, 4, 4)
L = (1.0, 1.0, 1.0)
dx = tuple(Li / ni for Li, ni in zip(L, n))
lam0, mu0 = 60.0, 30.0
Nv = int(np.prod(n))

# per-voxel major-symmetric positive-definite-ish stiffness (random, not
# physical -- only needs to be a valid 4th-order tensor for A_op to be
# well-posed for CG)
I2 = jnp.eye(3)
I4s = 0.5 * (jnp.einsum('ik,jl->ijkl', I2, I2) + jnp.einsum('il,jk->ijkl', I2, I2))
IxI = jnp.einsum('ij,kl->ijkl', I2, I2)
lam_field = jnp.asarray(rng.uniform(20.0, 80.0, size=(Nv,)))
mu_field = jnp.asarray(rng.uniform(10.0, 50.0, size=(Nv,)))
C_field = (lam_field[None, None, None, None, :] * IxI[..., None]
           + 2.0 * mu_field[None, None, None, None, :] * I4s[..., None])

eps_bar = jnp.array([
    [0.0, 1.0e-3, 0.0],
    [1.0e-3, 0.0, 0.0],
    [0.0, 0.0, 0.0],
])

toler_lin, maxiter = 1e-8, 1000


# ── 1. correctness invariants, both schemes ──────────────────────────────────

for green_op in (GreenOperatorBasic(n, L, lam0, mu0), GreenOperatorWillot(n, L, lam0, mu0, dx)):
    eps, sigma, delta, converged = solve_lippmann_schwinger(
        n, C_field, green_op, eps_bar, None, toler_lin, maxiter,
    )
    tag = type(green_op).__name__

    assert bool(converged), f"{tag}: solve must converge"
    assert jnp.allclose(jnp.mean(eps, axis=-1), eps_bar, atol=1e-10), \
        f"{tag}: mean(eps) must equal eps_bar exactly (Gamma0 is zero at DC)"

    gamma0 = Gamma0Operator(n, green_op)
    Adelta = gamma0(ddot42(C_field, delta)).reshape(-1)
    eps0 = jnp.ones((3, 3, Nv)) * eps_bar[:, :, None]
    bb = -gamma0(ddot42(C_field, eps0)).reshape(-1)
    resid = float(jnp.linalg.norm(Adelta - bb) / jnp.linalg.norm(bb))
    assert resid < toler_lin * 10, f"{tag}: A(delta) != b, relative residual {resid:.3e}"


# ── 2. real composite RVE (matches notebooks/mechanics/lin-elastic_strain.ipynb) ──────

from generation.rve import make_square_composite_rve
from materialmodels.elastic.isotropic import LinearElasticIsotropic
from materialmodels.assembly import assemble_C_field

phase_np, n_rve, L_rve, phi_act = make_square_composite_rve(
    phi=0.5, r_fiber=0.005, dx=0.0002, N_min=32, nz=10,
)
dx_rve = tuple(Li / ni for Li, ni in zip(L_rve, n_rve))
phase = jnp.array(phase_np.reshape(-1))

matrix = LinearElasticIsotropic(E=3.0e3, nu=0.35, name="epoxy matrix")
fiber = LinearElasticIsotropic(E=70.0e3, nu=0.20, name="glass fiber")
C_field_rve = assemble_C_field([matrix, fiber], phase)

lam0_rve = 0.5 * (matrix.lam + fiber.lam)
mu0_rve = 0.5 * (matrix.mu + fiber.mu)
green_op_rve = GreenOperatorWillot(n_rve, L_rve, lam0_rve, mu0_rve, dx_rve)

eps, sigma, delta, converged = solve_lippmann_schwinger(
    n_rve, C_field_rve, green_op_rve, eps_bar, None, toler_lin=1e-6, maxiter=1000,
)

tau_xy = float(jnp.mean(sigma[1, 0]))
assert bool(converged), "composite RVE solve must converge"
assert abs(tau_xy - 7.625369073063829) < 1e-6, f"tau_xy mismatch: got {tau_xy}, expected ~7.625369"


# ── 3. LippmannSchwingerSolver wraps solve_lippmann_schwinger exactly ───────
#
# allclose, not array_equal: these are two independent CG solves of the same
# problem, not two reads of one cached result -- bit-identical floats aren't
# guaranteed across separate calls on a multi-threaded CPU backend (verified
# flaky with array_equal: ~1/3 standalone runs differed at the ULP level).
# Numerical agreement well inside toler_lin is the real invariant.

solver = LippmannSchwingerSolver(n_rve, green_op_rve, toler_lin=1e-6, maxiter=1000)
result = solver.solve(C_field_rve, eps_bar)

assert bool(result.converged) == bool(converged)
assert jnp.allclose(result.eps, eps, atol=1e-12), "LippmannSchwingerSolver.eps must match solve_lippmann_schwinger"
assert jnp.allclose(result.sigma, sigma, atol=1e-8), "LippmannSchwingerSolver.sigma must match solve_lippmann_schwinger"
assert jnp.allclose(result.delta, delta, atol=1e-12), "LippmannSchwingerSolver.delta must match solve_lippmann_schwinger"

# ── 4. solve_lippmann_schwinger_mixed_bc (Michel et al 1999 outer loop) ─────

sigma_11_goal = 100.0
control_u = ((1, 0, 0), (0, 1, 0), (0, 0, 1))  # normal components stress-controlled
stress_goal_u = jnp.array([
    [sigma_11_goal, 0.0, 0.0],
    [0.0,           0.0, 0.0],
    [0.0,           0.0, 0.0],
])

# 4a. homogeneous closed-form target, same setup as
#     test_displacement_nw_cg.py's case 2 / test_elliptic_vector_
#     fourier_galerkin.py's check 3.
n_h, L_h = (16, 16, 16), (1.0, 1.0, 1.0)
Nv_h = int(np.prod(n_h))
mat_h = LinearElasticIsotropic(E=200e3, nu=0.3, name="steel")
phase_h = jnp.zeros((Nv_h,), dtype=int)
C_field_h = assemble_C_field([mat_h], phase_h)
dx_h = tuple(Li / ni for Li, ni in zip(L_h, n_h))
green_op_h = GreenOperatorWillot(n_h, L_h, mat_h.lam, mat_h.mu, dx_h)

eps_11_analytic = sigma_11_goal / mat_h.E
eps_22_analytic = -mat_h.nu * sigma_11_goal / mat_h.E

eps_h, sigma_h, delta_h, eps_bar_h, converged_h, n_iter_h = solve_lippmann_schwinger_mixed_bc(
    n_h, C_field_h, green_op_h, control_u, jnp.zeros((3, 3)), stress_goal_u,
    toler_lin=1e-10, maxiter=2000, toler_outer=1e-8, maxiter_outer=50,
)
sigma_mean_h = jnp.mean(sigma_h, axis=-1)

assert bool(converged_h), "mixed-BC solve must converge"
assert abs(float(eps_bar_h[0, 0]) - eps_11_analytic) < 1e-8
assert abs(float(eps_bar_h[1, 1]) - eps_22_analytic) < 1e-8
assert abs(float(sigma_mean_h[0, 0]) - sigma_11_goal) < 1e-6

print(f"[4a] mixed-BC closed form: eps_11={float(eps_bar_h[0,0]):.6e} "
      f"(analytic {eps_11_analytic:.6e}), n_iter_outer={n_iter_h} -- PASSED")

# 4b. heterogeneous composite RVE (check 2's own C_field_rve/green_op_rve),
#     cross-validated against formulation="displacement"'s bordered-CG
#     mixed-BC solve on the same problem. toler_outer is deliberately loose
#     (1e-2, not 4a's 1e-8): the outer correction's sensitivity matrix M is
#     built once from a FIXED reference stiffness, so convergence is linear
#     with a rate set by the material contrast (measured ~0.8-0.85/iteration
#     on this ~23x-contrast glass-fiber/epoxy RVE) -- getting all the way to
#     1e-6 would need >100 outer iterations (each a full inner CG solve);
#     1e-2 is already far tighter than the <10% cross-formulation tolerance
#     this check actually needs, and converges in well under maxiter_outer.
xi_flat_rve = build_freq_grid(n_rve, L_rve)

eps_ls_mixed, sigma_ls_mixed, delta_ls_mixed, eps_bar_ls_mixed, converged_ls_mixed, n_iter_ls_mixed = solve_lippmann_schwinger_mixed_bc(
    n_rve, C_field_rve, green_op_rve, control_u, jnp.zeros((3, 3)), stress_goal_u,
    toler_lin=1e-6, maxiter=2000, toler_outer=1e-2, maxiter_outer=60,
)
eps_disp, sigma_disp, delta_disp, eps_bar_disp, converged_disp = solve_displacement_based(
    n_rve, C_field_rve, xi_flat_rve, jnp.zeros((3, 3)), control_u, stress_goal_u,
    toler_lin=1e-6, maxiter=2000,
)

assert bool(converged_ls_mixed), "heterogeneous mixed-BC LS solve must converge"
eps_bar_diff = float(jnp.max(jnp.abs(eps_bar_ls_mixed - eps_bar_disp)))
eps_bar_scale = float(jnp.max(jnp.abs(eps_bar_disp))) + 1e-30
rel_diff_mixed = eps_bar_diff / eps_bar_scale
assert rel_diff_mixed < 0.10, (
    f"lippmann_schwinger vs. displacement mixed-BC eps_bar differ by {rel_diff_mixed:.1%}, expected <10%"
)

print(f"[4b] composite RVE mixed BC: eps_bar (LS, n_iter_outer={n_iter_ls_mixed}) = {np.asarray(eps_bar_ls_mixed).tolist()}")
print(f"                              eps_bar (displacement) = {np.asarray(eps_bar_disp).tolist()}")
print(f"     rel. diff = {rel_diff_mixed:.2%} -- PASSED")

# 4c. zero-target (free-surface) regression: macro_stress_goal == 0 on the
#     controlled directions must still report converged=True once the
#     residual is genuinely at noise level -- see docstring for the bug this
#     guards against. Reuses check 1's fast synthetic heterogeneous problem
#     (not the slow, high-contrast composite RVE above -- convergence SPEED
#     on a hard problem is a separate, already-understood property; this
#     check is specifically about the convergence-check's scale logic, which
#     doesn't need a slow problem to exercise) with a nonzero strain-
#     controlled eps_bar so the overall mean stress is genuinely nonzero
#     while the TARGET on the stress-controlled directions is exactly zero
#     -- an all-zero guess/target would trivially "converge" in 1 iteration
#     without exercising the outer correction at all.
green_op_c1 = GreenOperatorWillot(n, L, lam0, mu0, dx)
eps_bar_guess_free = jnp.array([[1.0e-3, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
eps_free, sigma_free, delta_free, eps_bar_free, converged_free, n_iter_free = solve_lippmann_schwinger_mixed_bc(
    n, C_field, green_op_c1, control_u, eps_bar_guess_free, jnp.zeros((3, 3)),
    toler_lin=1e-8, maxiter=2000, toler_outer=1e-6, maxiter_outer=50,
)
sigma_free_mean = jnp.mean(sigma_free, axis=-1)
assert bool(converged_free), (
    "zero macro_stress_goal (free surface) must still converge -- "
    f"n_iter_outer={n_iter_free}, sigma_yy={float(sigma_free_mean[1,1]):.3e}, "
    f"sigma_zz={float(sigma_free_mean[2,2]):.3e}"
)
sigma_xx_scale = abs(float(sigma_free_mean[0, 0])) + 1e-30
assert abs(float(sigma_free_mean[1, 1])) / sigma_xx_scale < 1e-4, "free surface: sigma_yy must be ~0"
assert abs(float(sigma_free_mean[2, 2])) / sigma_xx_scale < 1e-4, "free surface: sigma_zz must be ~0"

print(f"[4c] zero-target (free-surface) regression: n_iter_outer={n_iter_free}, "
      f"converged={bool(converged_free)}, sigma_yy={float(sigma_free_mean[1,1]):.3e} -- PASSED")

# ── 5. anisotropic material: outer loop must converge, not diverge ──────────
#
# Small synthetic two-phase problem (not a real RVE -- fast), with a genuinely
# anisotropic (transversely isotropic) phase: a flat interface along z (NOT
# the loaded x direction, to stay clear of notes/controll.md's unrelated
# sharp-interface-normal-loading checkerboard artifact -- this test is about
# the outer loop's reference-stiffness choice, not that). See docstring
# item 5 for the bug this guards against.
from materialmodels.elastic import TransverseIsotropic
from materialmodels.assembly import assemble_C_field as _assemble_C_field_aniso
from operators.green import build_reference_green_operator

n_aniso = (8, 8, 8)
L_aniso = (1.0, 1.0, 1.0)
phase_grid_aniso = np.zeros(n_aniso, dtype=int)
phase_grid_aniso[:, :, n_aniso[2] // 2:] = 1
phase_aniso = jnp.array(phase_grid_aniso.reshape(-1))

matrix_aniso = LinearElasticIsotropic(E=3.76e3, nu=0.39, name="epoxy matrix")
fiber_aniso = TransverseIsotropic(
    E_L=234000.0, E_T=15000.0, G_LT=15000.0, nu_LT=0.20, G_TT=7000.0, name="carbon fiber",
)
materials_aniso = [matrix_aniso, fiber_aniso]
C_field_aniso = _assemble_C_field_aniso(materials_aniso, phase_aniso)
green_op_aniso = build_reference_green_operator(n_aniso, L_aniso, materials_aniso, scheme="rotated")

control_aniso = ((0, 0, 0), (0, 1, 0), (0, 0, 1))  # xx strain-controlled; yy, zz free
eps_bar_guess_aniso = jnp.array([[1.0e-3, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
stress_goal_aniso = jnp.zeros((3, 3))

eps_a, sigma_a, delta_a, eps_bar_a, converged_a, n_iter_a = solve_lippmann_schwinger_mixed_bc(
    n_aniso, C_field_aniso, green_op_aniso, control_aniso, eps_bar_guess_aniso, stress_goal_aniso,
    toler_lin=1e-6, maxiter=2000, toler_outer=1e-4, maxiter_outer=100,
)
assert bool(converged_a), (
    f"anisotropic mixed-BC outer loop must converge, not diverge -- "
    f"n_iter_outer={n_iter_a}, eps_bar={np.asarray(eps_bar_a).tolist()}"
)
assert bool(jnp.all(jnp.abs(eps_bar_a) < 1.0)), (
    f"eps_bar magnitude must stay physically sane (< 1.0 strain), got "
    f"{np.asarray(eps_bar_a).tolist()} -- looks like the old divergence"
)

xi_flat_aniso = build_freq_grid(n_aniso, L_aniso)
eps_disp_a, sigma_disp_a, delta_disp_a, eps_bar_disp_a, converged_disp_a = solve_displacement_based(
    n_aniso, C_field_aniso, xi_flat_aniso, eps_bar_guess_aniso, control_aniso, stress_goal_aniso,
    toler_lin=1e-6, maxiter=2000,
)
assert bool(converged_disp_a), "displacement cross-check must converge"
eps_bar_diff_a = float(jnp.max(jnp.abs(eps_bar_a - eps_bar_disp_a)))
eps_bar_scale_a = float(jnp.max(jnp.abs(eps_bar_disp_a))) + 1e-30
rel_diff_a = eps_bar_diff_a / eps_bar_scale_a
assert rel_diff_a < 0.10, (
    f"anisotropic LS mixed-BC vs. displacement eps_bar differ by {rel_diff_a:.1%}, expected <10%"
)

print(f"[5] anisotropic material outer loop: n_iter_outer={n_iter_a}, "
      f"eps_bar={np.asarray(eps_bar_a).tolist()}")
print(f"    rel. diff from displacement = {rel_diff_a:.2%} -- PASSED")

print("test_elliptic_vector_lippmann_schwinger: all checks passed")
print(f"  composite RVE tau_xy (avg) = {tau_xy:.6f} MPa")
