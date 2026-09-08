"""
Standalone test for solve_displacement_based_nonlinear
(problems/mechanics.py) -- the Newton-outer/CG-inner driver for a nonlinear
per-voxel constitutive law. Lives in problems/, not solvers/, for the same
reason problems.fracture's staggered loop does -- see problems/mechanics.py's
module docstring.

Three checks
------------
1. Linear-problem reproduction: plugging a plain LINEAR local_update (a
   fixed C_field, sigma = C:eps, no state) into the nonlinear driver must
   reproduce solve_displacement_based's own answer to near machine
   precision -- the strongest available correctness check, since it's
   comparing against an independently-verified solver rather than trusting
   the new code in isolation.
2. A genuinely elastic load (small strain) on a two-phase RVE with a
   J2Plasticity matrix must leave every voxel's accumulated plastic strain
   exactly zero -- the yield surface should not activate below yield.
3. A load well past the matrix's yield stress must converge, produce no
   NaN, and leave a nonzero accumulated plastic strain in at least one
   matrix voxel (confirming the plastic branch actually engaged).

Usage
-----
    python -m pytest test/test_problems_mechanics_nonlinear.py
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
sys.path.insert(0, "src")

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)
import jax.numpy as jnp
import numpy as np

from generation.rve import make_square_composite_rve
from materialmodels.elastic.isotropic import LinearElasticIsotropic
from materialmodels.assembly import assemble_C_field
from materialmodels.inelastic.plasticity_j2 import J2Plasticity
from operators.green import build_freq_grid
from problems.mechanics import solve_displacement_based_nonlinear
from solvers.elliptic.vector.displacement_based import solve_displacement_based

phase_np, n, L, phi_act = make_square_composite_rve(
    phi=0.5, r_fiber=0.005, dx=0.0002, N_min=16, nz=1,
)
Nv = int(np.prod(n))
phase = jnp.array(phase_np.reshape(-1))
xi_flat = build_freq_grid(n, L)

# ── 1. linear-problem reproduction ───────────────────────────────────────────

matrix_el = LinearElasticIsotropic(E=3.76e3, nu=0.39, name="epoxy")
fiber_el  = LinearElasticIsotropic(E=70.0e3, nu=0.20, name="glass")
C_field   = assemble_C_field([matrix_el, fiber_el], phase)

eps_bar = jnp.array([[0.0, 1.0e-3, 0.0], [1.0e-3, 0.0, 0.0], [0.0, 0.0, 0.0]])
control = ((0, 0, 0), (0, 0, 0), (0, 0, 0))

eps_lin, sigma_lin, _, _, converged_lin = solve_displacement_based(
    n, C_field, xi_flat, eps_bar, control, jnp.zeros((3, 3)),
    toler_lin=1e-10, maxiter=2000,
)
assert bool(converged_lin), "reference linear solve did not converge"


def linear_local_update(eps_field, state):
    return jnp.einsum("ijklm,klm->ijm", C_field, eps_field), C_field, state


eps_nl, sigma_nl, _, converged_nl, n_iter = solve_displacement_based_nonlinear(
    n, xi_flat, eps_bar, linear_local_update, state_init=None,
    toler_lin=1e-10, maxiter_lin=2000, toler_nr=1e-10, maxiter_nr=10,
)
err_eps = float(jnp.max(jnp.abs(eps_nl - eps_lin)))
print(f"[1] linear reproduction: converged={converged_nl}  n_iter={n_iter}  "
      f"max|eps_nl - eps_lin|={err_eps:.3e}")
assert converged_nl, "nonlinear driver did not converge on a linear problem"
assert err_eps < 1e-8, f"nonlinear driver does not reproduce the known-good linear solve: {err_eps:.3e}"
print("[1] PASSED")

# ── 2/3. two-phase RVE with a plastic matrix ────────────────────────────────

matrix_pl = J2Plasticity(E=3.76e3, nu=0.39, sigma_y0=50.0, H=1.0e3, name="epoxy-plastic")
C_fiber   = fiber_el.stiffness_tensor()


def plastic_local_update(eps_field, state):
    eps_p_field, alpha_field = state
    sigma_pl, C_pl, (eps_p_new, alpha_new) = matrix_pl.stress_and_tangent_field(
        eps_field, eps_p_field, alpha_field
    )
    sigma_el = jnp.einsum("ijkl,klm->ijm", C_fiber, eps_field)
    C_el     = jnp.broadcast_to(C_fiber[..., None], C_pl.shape)
    is_matrix = (phase == 0)
    sigma = jnp.where(is_matrix, sigma_pl, sigma_el)
    C_tan = jnp.where(is_matrix, C_pl, C_el)
    eps_p_out = jnp.where(is_matrix, eps_p_new, eps_p_field)
    alpha_out = jnp.where(is_matrix, alpha_new, alpha_field)
    return sigma, C_tan, (eps_p_out, alpha_out)


state0 = (jnp.zeros((3, 3, Nv)), jnp.zeros(Nv))

# [2] genuinely elastic load -- alpha must stay exactly zero everywhere
eps_bar_small = jnp.array([[0.0, 1.0e-4, 0.0], [1.0e-4, 0.0, 0.0], [0.0, 0.0, 0.0]])
_, _, (_, alpha_small), converged_small, _ = solve_displacement_based_nonlinear(
    n, xi_flat, eps_bar_small, plastic_local_update, state0,
    toler_lin=1e-8, maxiter_lin=2000, toler_nr=1e-8, maxiter_nr=30,
)
print(f"[2] elastic load: converged={converged_small}  max|alpha|={float(jnp.max(alpha_small)):.3e}")
assert converged_small
assert float(jnp.max(alpha_small)) == 0.0, "yield surface activated below yield"
print("[2] PASSED")

# [3] load well past the matrix's yield stress
eps_bar_large = jnp.array([[0.0, 3.0e-3, 0.0], [3.0e-3, 0.0, 0.0], [0.0, 0.0, 0.0]])
sigma_large, eps_p_large, alpha_large = None, None, None
eps_large, sigma_large, (eps_p_large, alpha_large), converged_large, n_iter_large = \
    solve_displacement_based_nonlinear(
        n, xi_flat, eps_bar_large, plastic_local_update, state0,
        toler_lin=1e-8, maxiter_lin=2000, toler_nr=1e-8, maxiter_nr=50,
    )
matrix_alpha = alpha_large[phase == 0]
n_plastic = int(jnp.sum(matrix_alpha > 1e-12))
print(f"[3] plastic load: converged={converged_large}  n_iter={n_iter_large}  "
      f"plastic matrix voxels={n_plastic}/{matrix_alpha.shape[0]}  "
      f"no NaN={not bool(jnp.any(jnp.isnan(sigma_large)))}")
assert converged_large
assert not bool(jnp.any(jnp.isnan(sigma_large)))
assert n_plastic > 0, "expected at least one matrix voxel to yield at this load level"
print("[3] PASSED")

print("\ntest_problems_mechanics_nonlinear: all checks passed")
