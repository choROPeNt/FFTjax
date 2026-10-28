"""
Standalone test for solve_fourier_galerkin
(solvers/elliptic/vector/fourier_galerkin.py).

Five checks
-----------
1. Correctness invariants on the same kind of synthetic random heterogeneous
   problem as test_elliptic_vector_lippmann_schwinger.py's check 1, pure
   strain BC: mean(eps) == eps_bar exactly, A(delta) = b to tolerance
   (independently recomputed), and eps_bar_out == eps_bar exactly (no
   mixed-BC correction active).
2. Real composite RVE (glass fiber / epoxy, same setup as
   test_elliptic_vector_lippmann_schwinger.py's check 2) -- converges, and
   agrees with solve_lippmann_schwinger's tau_xy on the same RVE within the
   cross-formulation tolerance already used in test_problems_mechanics.py
   check 5 (<10%). Now a genuinely independent check since
   solve_fourier_galerkin no longer delegates to solve_lippmann_schwinger
   internally (it forked once mixed-BC support was added -- see
   notes/controll.md).
3. Mixed-BC closed-form recovery: same homogeneous single-phase RVE, same
   control/stress goal as test_displacement_nw_cg.py's case 2
   (n=(16,16,16), sigma_11 = 100 uniaxial stress) -- the fluctuation field is
   provably zero for a homogeneous material, so the closed-form isotropic
   uniaxial-stress relations are exact to CG tolerance, not just
   approximately close.
4. FourierGalerkinSolver (the ElasticitySolver wrapper) reproduces
   solve_fourier_galerkin's own output, including the populated eps_bar.
5. solve_lippmann_schwinger_dc (Kabel et al 2016 route, same DC-bin-identity
   core as solve_fourier_galerkin but applied to a reference-medium Green's
   operator instead) reproduces the same closed-form check 3 target --
   proves solvers.elliptic.vector.mixed_bc.solve_mixed_bc_dc_identity is
   genuinely operator-agnostic, not coincidentally correct for one operator.

Usage
-----
    python -m pytest test/test_elliptic_vector_fourier_galerkin.py
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.makedirs("output", exist_ok=True)

import sys
sys.path.insert(0, "src")

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)
import jax.numpy as jnp
import numpy as np

from operators.galerkin import build_galerkin_projector
from operators.general_functions import ddot42
from operators.green import GreenOperatorWillot
from operators.projection import Gamma0Operator
from solvers.elliptic.vector.fourier_galerkin import solve_fourier_galerkin, FourierGalerkinSolver
from solvers.elliptic.vector.lippmann_schwinger import solve_lippmann_schwinger, solve_lippmann_schwinger_dc


rng = np.random.default_rng(2)

n = (4, 4, 4)
L = (1.0, 1.0, 1.0)
Nv = int(np.prod(n))

# per-voxel major-symmetric positive-definite-ish stiffness (random, not
# physical -- only needs to be a valid 4th-order tensor for A_op to be
# well-posed for CG) -- same construction as
# test_elliptic_vector_lippmann_schwinger.py's check 1.
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


# ── 1. correctness invariants, pure strain BC ────────────────────────────────

galerkin_op = build_galerkin_projector(n, L, scheme='rotated')
eps, sigma, delta, eps_bar_out, converged = solve_fourier_galerkin(
    n, C_field, galerkin_op, eps_bar, toler_lin=toler_lin, maxiter=maxiter,
)

assert bool(converged), "pure-strain solve must converge"
assert jnp.allclose(jnp.mean(eps, axis=-1), eps_bar, atol=1e-10), \
    "mean(eps) must equal eps_bar exactly under pure strain BC"
assert jnp.allclose(eps_bar_out, eps_bar, atol=1e-10), \
    "eps_bar_out must equal eps_bar exactly under pure strain BC"

gamma0 = Gamma0Operator(n, galerkin_op)
Adelta = gamma0(ddot42(C_field, delta)).reshape(-1)
eps0 = jnp.ones((3, 3, Nv)) * eps_bar[:, :, None]
bb = -gamma0(ddot42(C_field, eps0)).reshape(-1)
resid = float(jnp.linalg.norm(Adelta - bb) / jnp.linalg.norm(bb))
assert resid < toler_lin * 10, f"A(delta) != b, relative residual {resid:.3e}"

print("[1] pure-strain invariants: PASSED")


# ── 2. real composite RVE, cross-formulation agreement with LS ──────────────

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

galerkin_op_rve = build_galerkin_projector(n_rve, L_rve, scheme='rotated')
eps_rve, sigma_rve, delta_rve, eps_bar_out_rve, converged_rve = solve_fourier_galerkin(
    n_rve, C_field_rve, galerkin_op_rve, eps_bar, toler_lin=1e-6, maxiter=1000,
)
tau_xy_fg = float(jnp.mean(sigma_rve[1, 0]))
assert bool(converged_rve), "composite RVE solve must converge"

lam0_rve = 0.5 * (matrix.lam + fiber.lam)
mu0_rve = 0.5 * (matrix.mu + fiber.mu)
green_op_rve = GreenOperatorWillot(n_rve, L_rve, lam0_rve, mu0_rve, dx_rve)
_, sigma_ls, _, _ = solve_lippmann_schwinger(
    n_rve, C_field_rve, green_op_rve, eps_bar, None, toler_lin=1e-6, maxiter=1000,
)
tau_xy_ls = float(jnp.mean(sigma_ls[1, 0]))
rel_diff = abs(tau_xy_fg - tau_xy_ls) / abs(tau_xy_ls)
assert rel_diff < 0.10, f"fourier_galerkin vs. lippmann_schwinger tau_xy differ by {rel_diff:.1%}, expected <10%"

print(f"[2] composite RVE: tau_xy (fourier_galerkin) = {tau_xy_fg:.6f} MPa, "
      f"tau_xy (lippmann_schwinger) = {tau_xy_ls:.6f} MPa, rel. diff = {rel_diff:.2%}")


# ── 3. mixed-BC closed-form recovery (same setup as test_displacement_nw_cg.py case 2) ──

n_h = (16, 16, 16)
L_h = (1.0, 1.0, 1.0)
Nv_h = int(np.prod(n_h))
mat_h = LinearElasticIsotropic(E=200e3, nu=0.3, name="steel")
phase_h = jnp.zeros((Nv_h,), dtype=int)
C_field_h = assemble_C_field([mat_h], phase_h)
galerkin_op_h = build_galerkin_projector(n_h, L_h, scheme='rotated')

sigma_11_goal = 100.0
control_u = ((1, 0, 0), (0, 1, 0), (0, 0, 1))  # normal components stress-controlled
stress_goal_u = jnp.array([
    [sigma_11_goal, 0.0, 0.0],
    [0.0,           0.0, 0.0],
    [0.0,           0.0, 0.0],
])
eps_11_analytic = sigma_11_goal / mat_h.E
eps_22_analytic = -mat_h.nu * sigma_11_goal / mat_h.E

eps_h, sigma_h, delta_h, eps_bar_h, converged_h = solve_fourier_galerkin(
    n_h, C_field_h, galerkin_op_h, jnp.zeros((3, 3)), control_u, stress_goal_u,
    toler_lin=1e-10, maxiter=2000,
)
sigma_mean_h = jnp.mean(sigma_h, axis=-1)

assert bool(converged_h), "mixed-BC solve must converge"
assert abs(float(eps_bar_h[0, 0]) - eps_11_analytic) < 1e-8
assert abs(float(eps_bar_h[1, 1]) - eps_22_analytic) < 1e-8
assert abs(float(eps_bar_h[2, 2]) - eps_22_analytic) < 1e-8
assert abs(float(sigma_mean_h[0, 0]) - sigma_11_goal) < 1e-6
assert abs(float(sigma_mean_h[1, 1])) < 1e-6

print(f"[3] mixed-BC closed form: eps_11={float(eps_bar_h[0,0]):.6e} "
      f"(analytic {eps_11_analytic:.6e}), eps_22={float(eps_bar_h[1,1]):.6e} "
      f"(analytic {eps_22_analytic:.6e}) -- PASSED")


# ── 4. FourierGalerkinSolver wraps solve_fourier_galerkin exactly ───────────

solver = FourierGalerkinSolver(n_h, galerkin_op_h, control=control_u, toler_lin=1e-10, maxiter=2000)
result = solver.solve(C_field_h, jnp.zeros((3, 3)), stress_goal_u)

assert bool(result.converged) == bool(converged_h)
assert result.eps_bar is not None
assert jnp.allclose(result.eps, eps_h, atol=1e-10), "FourierGalerkinSolver.eps must match solve_fourier_galerkin"
assert jnp.allclose(result.eps_bar, eps_bar_h, atol=1e-10), \
    "FourierGalerkinSolver.eps_bar must match solve_fourier_galerkin"

print("[4] FourierGalerkinSolver wrapper: PASSED")


# ── 5. solve_lippmann_schwinger_dc (Kabel route) reproduces the same closed form ──

green_op_h = GreenOperatorWillot(n_h, L_h, mat_h.lam, mat_h.mu, tuple(Li / ni for Li, ni in zip(L_h, n_h)))
eps_dc, sigma_dc, delta_dc, eps_bar_dc, converged_dc = solve_lippmann_schwinger_dc(
    n_h, C_field_h, green_op_h, jnp.zeros((3, 3)), control_u, stress_goal_u,
    toler_lin=1e-10, maxiter=2000,
)
assert bool(converged_dc)
assert abs(float(eps_bar_dc[0, 0]) - eps_11_analytic) < 1e-8
assert abs(float(eps_bar_dc[1, 1]) - eps_22_analytic) < 1e-8

print(f"[5] solve_lippmann_schwinger_dc (Kabel route): eps_11={float(eps_bar_dc[0,0]):.6e} -- PASSED "
      "(same DC-identity core, different operator, same closed form)")

print("\ntest_elliptic_vector_fourier_galerkin: all checks passed")
