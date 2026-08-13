"""
Standalone test for solve_mechanics (problems/mechanics.py).

Three checks
------------
1. End-to-end parity -- solve_mechanics on the same glass-fiber/epoxy
   composite RVE as notebooks/lin-elastic_strain.ipynb and
   test_elliptic_vector_lippmann_schwinger.py must reproduce the known
   tau_xy (avg) = 7.625369 MPa result, going through the full wiring layer
   (assemble_C_field, reference-medium averaging, GreenOperatorWillot,
   solve_lippmann_schwinger) rather than each piece called by hand.
2. scheme="standard" also runs and converges (just checks it doesn't error
   and produces a finite, converged result -- no known reference value for
   the non-rotated scheme on this RVE).
3. Unknown formulation/scheme raise clear errors rather than silently
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
from mat_models.elastic import LinearElasticIsotropic
from problems.mechanics import solve_mechanics


phase_np, N, n, L, phi_act = make_square_composite_rve(
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

eps, sigma, delta, converged = solve_mechanics(
    n, L, phase, materials, eps_bar,
    formulation="lippmann_schwinger", scheme="rotated",
    toler_lin=1e-6, maxiter=1000,
)
tau_xy = float(jnp.mean(sigma[1, 0]))
assert bool(converged), "composite RVE solve must converge"
assert abs(tau_xy - 7.625369073063829) < 1e-6, f"tau_xy mismatch: got {tau_xy}, expected ~7.625369"
print(f"[1] rotated scheme: tau_xy = {tau_xy:.6f} MPa, converged={bool(converged)}")


# ── 2. standard scheme runs and converges ────────────────────────────────────

eps_std, sigma_std, delta_std, converged_std = solve_mechanics(
    n, L, phase, materials, eps_bar,
    formulation="lippmann_schwinger", scheme="standard",
    toler_lin=1e-6, maxiter=2000,
)
assert bool(converged_std), "standard-scheme solve must also converge"
assert jnp.all(jnp.isfinite(sigma_std)), "standard-scheme stress must be finite"
print(f"[2] standard scheme: tau_xy = {float(jnp.mean(sigma_std[1, 0])):.6f} MPa, "
      f"converged={bool(converged_std)}")


# ── 3. unknown formulation/scheme raise clear errors ─────────────────────────

with pytest.raises(NotImplementedError):
    solve_mechanics(n, L, phase, materials, eps_bar, formulation="displacement")

with pytest.raises(ValueError):
    solve_mechanics(n, L, phase, materials, eps_bar, formulation="bogus")

with pytest.raises(ValueError):
    solve_mechanics(n, L, phase, materials, eps_bar, scheme="bogus")

print("test_problems_mechanics: all checks passed")
