"""
Standalone test for solve_mechanics (problems/mechanics.py).

Four checks
-----------
1. End-to-end parity -- solve_mechanics on the same glass-fiber/epoxy
   composite RVE as notebooks/lin-elastic_strain.ipynb and
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
   doing the wrong thing.

Usage
-----
    python -m pytest test/test_problems_mechanics.py
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
sys.path.insert(0, "src")

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)
import jax.numpy as jnp
import pytest

from generation.rve import make_square_composite_rve
from materialmodels.elastic.isotropic import LinearElasticIsotropic
from problems.mechanics import solve_mechanics


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

sol = solve_mechanics(
    n, L, phase, materials, eps_bar,
    formulation="lippmann_schwinger", scheme="rotated",
    toler_lin=1e-6, maxiter=1000,
)
tau_xy = float(jnp.mean(sol.sigma[1, 0]))
assert bool(sol.converged), "composite RVE solve must converge"
assert abs(tau_xy - 7.625369073063829) < 1e-6, f"tau_xy mismatch: got {tau_xy}, expected ~7.625369"
print(f"[1] rotated scheme: tau_xy = {tau_xy:.6f} MPa, converged={bool(sol.converged)}")


# ── 2. standard scheme runs and converges ────────────────────────────────────

sol_std = solve_mechanics(
    n, L, phase, materials, eps_bar,
    formulation="lippmann_schwinger", scheme="standard",
    toler_lin=1e-6, maxiter=2000,
)
assert bool(sol_std.converged), "standard-scheme solve must also converge"
assert jnp.all(jnp.isfinite(sol_std.sigma)), "standard-scheme stress must be finite"
print(f"[2] standard scheme: tau_xy = {float(jnp.mean(sol_std.sigma[1, 0])):.6f} MPa, "
      f"converged={bool(sol_std.converged)}")


# ── 3. displacement-based formulation runs, converges, and roughly agrees ────

sol_disp = solve_mechanics(
    n, L, phase, materials, eps_bar,
    formulation="displacement",
    toler_lin=1e-6, maxiter=2000,
)
tau_xy_disp = float(jnp.mean(sol_disp.sigma[1, 0]))
rel_diff = abs(tau_xy_disp - tau_xy) / abs(tau_xy)
assert bool(sol_disp.converged), "displacement-based solve must converge"
assert jnp.allclose(sol_disp.eps_bar, eps_bar), "pure-strain BC: eps_bar must pass through unchanged"
assert rel_diff < 0.10, f"displacement vs. lippmann_schwinger tau_xy differ by {rel_diff:.1%}, expected <10%"
print(f"[3] displacement formulation: tau_xy = {tau_xy_disp:.6f} MPa, "
      f"converged={bool(sol_disp.converged)}, rel. diff from LS = {rel_diff:.2%}")


# ── 4. unknown formulation/scheme raise clear errors ─────────────────────────

with pytest.raises(ValueError):
    solve_mechanics(n, L, phase, materials, eps_bar, formulation="bogus")

with pytest.raises(ValueError):
    solve_mechanics(n, L, phase, materials, eps_bar, scheme="bogus")

print("test_problems_mechanics: all checks passed")
