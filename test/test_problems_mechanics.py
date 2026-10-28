"""
Standalone test for solve_mechanics (problems/mechanics.py).

Six checks
----------
1. End-to-end parity -- solve_mechanics on the same glass-fiber/epoxy
   composite RVE as notebooks/mechanics/lin-elastic_strain.ipynb and
   test_elliptic_vector_lippmann_schwinger.py must reproduce the known
   tau_xy (avg) = 7.625369 MPa result, going through the full wiring layer
   (assemble_C_field, reference-medium averaging, GreenOperatorWillot,
   solve_lippmann_schwinger) rather than each piece called by hand.
2. scheme="standard" also runs and converges (just checks it doesn't error
   and produces a finite, converged result -- no known reference value for
   the non-rotated scheme on this RVE).
3. formulation="displacement" also runs and converges on the same RVE, and
   agrees with the lippmann_schwinger result to within the few-percent
   discretization difference expected between the two schemes on a
   sharp-interface microstructure (see test_displacement_nw_cg.py's
   module docstring for why exact agreement isn't expected here).
4. Unknown formulation/scheme raise clear errors rather than silently
   doing the wrong thing. 4b: formulation="lippmann_schwinger" with
   all-zero control tolerates a leftover macroscopic (3,3) stress_goal
   instead of crashing -- regression test, see its own comment for the bug.
5. formulation="fourier_galerkin" also runs, converges, and agrees with
   both lippmann_schwinger and displacement to within the same
   cross-formulation tolerance as check 3 -- see
   test_operators_green_galerkin.py for the projector's own unit checks.
6. formulation="fourier_galerkin" with a nonzero ``control`` (mixed
   macroscopic strain/stress BC, see solvers.elliptic.vector.mixed_bc) no
   longer raises, and agrees with formulation="displacement"'s mixed-BC
   result on the same heterogeneous RVE within the same cross-formulation
   tolerance as check 3/5.

Usage
-----
    python -m pytest test/test_problems_mechanics.py
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.makedirs("output", exist_ok=True)

import sys
sys.path.insert(0, "src")

from typing import cast

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)
import jax.numpy as jnp
import pytest

from generation.rve import make_square_composite_rve
from materialmodels.elastic.isotropic import LinearElasticIsotropic
from problems.mechanics import solve_mechanics
from solvers.solution import ElasticitySolution


phase_np, n, L, phi_act = make_square_composite_rve(
    phi=0.5, r_fiber=0.005, dx=0.0002, N_min=32, nz=10,
)
phase = jnp.array(phase_np.reshape(-1))

matrix = LinearElasticIsotropic(E=3.0e3, nu=0.35, name="epoxy matrix")
fiber = LinearElasticIsotropic(E=70.0e3, nu=0.20, name="glass fiber")
materials = [matrix, fiber]

eps_bar = jnp.array([
    [0.0, 1.0e-3, 0.0],
    [1.0e-3, 0.0, 0.0],
    [0.0, 0.0, 0.0],
])


# ── 1. end-to-end parity with the known notebook result ─────────────────────

sol = cast(ElasticitySolution, solve_mechanics(
    n, L, phase, materials, eps_bar,
    formulation="lippmann_schwinger", scheme="rotated",
    toler_lin=1e-6, maxiter=1000,
)[0].solution)
tau_xy = float(jnp.mean(sol.sigma[1, 0]))
assert bool(sol.converged), "composite RVE solve must converge"
assert abs(tau_xy - 7.625369073063829) < 1e-6, f"tau_xy mismatch: got {tau_xy}, expected ~7.625369"
print(f"[1] rotated scheme: tau_xy = {tau_xy:.6f} MPa, converged={bool(sol.converged)}")


# ── 2. standard scheme runs and converges ────────────────────────────────────

sol_std = cast(ElasticitySolution, solve_mechanics(
    n, L, phase, materials, eps_bar,
    formulation="lippmann_schwinger", scheme="standard",
    toler_lin=1e-6, maxiter=2000,
)[0].solution)
assert bool(sol_std.converged), "standard-scheme solve must also converge"
assert jnp.all(jnp.isfinite(sol_std.sigma)), "standard-scheme stress must be finite"
print(f"[2] standard scheme: tau_xy = {float(jnp.mean(sol_std.sigma[1, 0])):.6f} MPa, "
      f"converged={bool(sol_std.converged)}")


# ── 3. displacement-based formulation runs, converges, and roughly agrees ────

sol_disp = cast(ElasticitySolution, solve_mechanics(
    n, L, phase, materials, eps_bar,
    formulation="displacement",
    toler_lin=1e-6, maxiter=2000,
)[0].solution)
tau_xy_disp = float(jnp.mean(sol_disp.sigma[1, 0]))
rel_diff = abs(tau_xy_disp - tau_xy) / abs(tau_xy)
assert bool(sol_disp.converged), "displacement-based solve must converge"
assert sol_disp.eps_bar is not None  # always populated for formulation="displacement"
assert jnp.allclose(sol_disp.eps_bar, eps_bar), "pure-strain BC: eps_bar must pass through unchanged"
assert rel_diff < 0.10, f"displacement vs. lippmann_schwinger tau_xy differ by {rel_diff:.1%}, expected <10%"
print(f"[3] displacement formulation: tau_xy = {tau_xy_disp:.6f} MPa, "
      f"converged={bool(sol_disp.converged)}, rel. diff from LS = {rel_diff:.2%}")


# ── 4. unknown formulation/scheme raise clear errors ─────────────────────────

with pytest.raises(ValueError):
    solve_mechanics(n, L, phase, materials, eps_bar, formulation="bogus")

with pytest.raises(ValueError):
    solve_mechanics(n, L, phase, materials, eps_bar, scheme="bogus")


# ── 4b. lippmann_schwinger + all-zero control tolerates a leftover macroscopic
#        (3,3) stress_goal instead of crashing ──────────────────────────────
#
# Regression test: formulation="lippmann_schwinger" has two solver classes
# with DIFFERENT stress_goal conventions -- LippmannSchwingerMixedBCSolver
# (control nonzero) takes a macroscopic (3,3) target, but the plain
# LippmannSchwingerSolver (control all-zero) takes an unrelated per-voxel
# (3,3, Nv)-or-None one. _solve_mechanics_step used to forward stress_goal
# unconditionally into whichever solver got picked -- harmless when control
# is nonzero, but a real crash (JAX shape-broadcasting error deep inside
# solve_lippmann_schwinger) when control is all-zero and the caller still
# passes a (3,3) array, e.g. reusing the same stress_goal variable across a
# mixed-BC cell and a pure-strain one (exactly what happened editing
# notebooks/mechanics/lin-elastic_solver-comparison.ipynb by hand). Fixed by
# not forwarding stress_goal at all when control has no active pairs.
sol_zero_control_leftover_goal = cast(ElasticitySolution, solve_mechanics(
    n, L, phase, materials, eps_bar,
    formulation="lippmann_schwinger", scheme="rotated",
    control=((0, 0, 0), (0, 0, 0), (0, 0, 0)), stress_goal=jnp.zeros((3, 3)),
    toler_lin=1e-6, maxiter=1000,
)[0].solution)
assert bool(sol_zero_control_leftover_goal.converged)
assert sol_zero_control_leftover_goal.eps_bar is None
assert jnp.allclose(sol_zero_control_leftover_goal.sigma, sol.sigma, atol=1e-8), \
    "a leftover macroscopic stress_goal must not change the pure-strain LS result"
print("[4b] lippmann_schwinger + all-zero control + leftover (3,3) stress_goal: PASSED")


# ── 5. fourier_galerkin formulation runs, converges, and roughly agrees ──────

sol_fg = cast(ElasticitySolution, solve_mechanics(
    n, L, phase, materials, eps_bar,
    formulation="fourier_galerkin", scheme="rotated",
    toler_lin=1e-6, maxiter=2000,
)[0].solution)
tau_xy_fg = float(jnp.mean(sol_fg.sigma[1, 0]))
rel_diff_ls = abs(tau_xy_fg - tau_xy) / abs(tau_xy)
rel_diff_disp = abs(tau_xy_fg - tau_xy_disp) / abs(tau_xy_disp)
assert bool(sol_fg.converged), "fourier_galerkin solve must converge"
assert sol_fg.eps_bar is not None  # always populated for formulation="fourier_galerkin"
assert jnp.allclose(sol_fg.eps_bar, eps_bar), "pure-strain BC: eps_bar must pass through unchanged"
assert rel_diff_ls < 0.10, f"fourier_galerkin vs. lippmann_schwinger tau_xy differ by {rel_diff_ls:.1%}, expected <10%"
assert rel_diff_disp < 0.10, f"fourier_galerkin vs. displacement tau_xy differ by {rel_diff_disp:.1%}, expected <10%"
print(f"[5] fourier_galerkin formulation: tau_xy = {tau_xy_fg:.6f} MPa, "
      f"converged={bool(sol_fg.converged)}, rel. diff from LS = {rel_diff_ls:.2%}, "
      f"from displacement = {rel_diff_disp:.2%}")


# ── 6. fourier_galerkin mixed-BC agrees with displacement's mixed-BC result ──

control_mixed = ((1, 0, 0), (0, 1, 0), (0, 0, 1))  # normal components stress-controlled
stress_goal_mixed = jnp.array([
    [50.0, 0.0, 0.0],
    [0.0,  0.0, 0.0],
    [0.0,  0.0, 0.0],
])

sol_fg_mixed = cast(ElasticitySolution, solve_mechanics(
    n, L, phase, materials, jnp.zeros((3, 3)),
    formulation="fourier_galerkin", scheme="rotated",
    control=control_mixed, stress_goal=stress_goal_mixed,
    toler_lin=1e-6, maxiter=2000,
)[0].solution)
sol_disp_mixed = cast(ElasticitySolution, solve_mechanics(
    n, L, phase, materials, jnp.zeros((3, 3)),
    formulation="displacement",
    control=control_mixed, stress_goal=stress_goal_mixed,
    toler_lin=1e-6, maxiter=2000,
)[0].solution)
assert bool(sol_fg_mixed.converged), "fourier_galerkin mixed-BC solve must converge"
eps_bar_diff = float(jnp.max(jnp.abs(sol_fg_mixed.eps_bar - sol_disp_mixed.eps_bar)))
eps_bar_scale = float(jnp.max(jnp.abs(sol_disp_mixed.eps_bar))) + 1e-30
assert eps_bar_diff / eps_bar_scale < 0.10, (
    f"fourier_galerkin vs. displacement mixed-BC eps_bar differ by "
    f"{eps_bar_diff / eps_bar_scale:.1%}, expected <10%"
)
print(f"[6] fourier_galerkin mixed BC: eps_bar =\n{jnp.asarray(sol_fg_mixed.eps_bar)}\n"
      f"    displacement mixed BC: eps_bar =\n{jnp.asarray(sol_disp_mixed.eps_bar)}\n"
      f"    rel. diff = {eps_bar_diff / eps_bar_scale:.2%}")

print("test_problems_mechanics: all checks passed")
