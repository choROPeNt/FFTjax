"""
Standalone test for solvers/elliptic/scalar.py's solve_thermal_conduction --
the scalar (heat conduction) analogue of solvers/elliptic/vector/
displacement_based.py's solve_displacement_based, validated against known
analytical cases the way notes/TARGET_LAYOUT.md's tests/solvers/elliptic/
vector/ note calls for (a two-phase laminate bound), not just internal
self-consistency.

Three checks
------------
1. Homogeneous medium: the periodic fluctuation T' must vanish exactly (no
   heterogeneity to drive one), and the homogenized flux must equal
   -k * grad_T_bar exactly -- the trivial case any heterogeneous solver
   should reduce to.
2. Two-phase laminate (layers normal to x, equal volume fraction): the
   classic homogenization sanity check. Gradient driven along x (normal to
   the layers, a series thermal circuit) must reproduce the harmonic mean of
   k1/k2; driven along y (in-plane, a parallel circuit) must reproduce the
   arithmetic mean -- both closed-form, both directions checked in the same
   run so a sign or transpose error in the anisotropic conductivity handling
   would show up as only one of the two matching.
3. Mixed gradient/flux BC: driving grad_x while flux-controlling y and z to
   nonzero targets must hit those targets in the homogenized flux, and must
   leave the gradient-controlled x entry of the solved macroscopic gradient
   unchanged from what was prescribed.

Usage
-----
    python -m pytest test/test_solvers_elliptic_scalar_thermal.py
"""

import sys
sys.path.insert(0, "src")

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)
import jax.numpy as jnp
import numpy as np

from operators.green import build_freq_grid
from materialmodels.assembly import assemble_K_field
from materialmodels.thermal.isotropic import ThermalConductivityIsotropic
from solvers.elliptic.scalar import solve_thermal_conduction

# ── [1] homogeneous medium: T' == 0, flux_bar == -k * grad_T_bar exactly ────
n = (16, 4, 4)
L = (1.0, 1.0, 1.0)
xi_flat = build_freq_grid(n, L)
Nv = int(np.prod(n))

k1, k2 = 2.0, 8.0
K_homog = assemble_K_field([ThermalConductivityIsotropic(k1)], jnp.zeros(Nv, dtype=int))
grad_bar = jnp.array([1.0, 0.5, -0.3])
grad_T, flux, T_prime, delta, grad_bar_out, converged = solve_thermal_conduction(
    n, K_homog, xi_flat, grad_bar, toler_lin=1e-10, maxiter=2000,
)
assert bool(converged)
assert float(jnp.max(jnp.abs(T_prime))) < 1e-8, "homogeneous medium must have zero fluctuation"
flux_bar = jnp.mean(flux, axis=-1)
assert jnp.allclose(flux_bar, -k1 * grad_bar, atol=1e-8)
print(f"[1] homogeneous: max|T'|={float(jnp.max(jnp.abs(T_prime))):.2e}  "
      f"flux_bar={np.asarray(flux_bar)}")
print("[1] PASSED")

# ── [2] two-phase laminate: series (harmonic mean) vs parallel (arithmetic) ─
n = (32, 4, 4)
xi_flat = build_freq_grid(n, L)
phase_grid = np.zeros(n, dtype=int)
phase_grid[n[0] // 2:, :, :] = 1   # layers normal to x, equal volume fraction
phase = jnp.array(phase_grid.ravel())
materials = [ThermalConductivityIsotropic(k1, "phase0"), ThermalConductivityIsotropic(k2, "phase1")]
K_field = assemble_K_field(materials, phase)

vf = 0.5
k_eff_series   = 1.0 / (vf / k1 + vf / k2)   # gradient normal to layers
k_eff_parallel = vf * k1 + vf * k2            # gradient in-plane

grad_bar_x = jnp.array([1.0, 0.0, 0.0])
_, flux, _, _, _, converged = solve_thermal_conduction(
    n, K_field, xi_flat, grad_bar_x, toler_lin=1e-10, maxiter=2000,
)
assert bool(converged)
flux_bar = jnp.mean(flux, axis=-1)
k_eff_x = -float(flux_bar[0]) / float(grad_bar_x[0])
assert abs(k_eff_x - k_eff_series) < 1e-4, (k_eff_x, k_eff_series)
assert abs(float(flux_bar[1])) < 1e-8 and abs(float(flux_bar[2])) < 1e-8

grad_bar_y = jnp.array([0.0, 1.0, 0.0])
_, flux, _, _, _, converged = solve_thermal_conduction(
    n, K_field, xi_flat, grad_bar_y, toler_lin=1e-10, maxiter=2000,
)
assert bool(converged)
flux_bar = jnp.mean(flux, axis=-1)
k_eff_y = -float(flux_bar[1]) / float(grad_bar_y[1])
assert abs(k_eff_y - k_eff_parallel) < 1e-4, (k_eff_y, k_eff_parallel)

print(f"[2] laminate: k_eff series={k_eff_x:.4f} (analytical {k_eff_series:.4f})  "
      f"parallel={k_eff_y:.4f} (analytical {k_eff_parallel:.4f})")
print("[2] PASSED")

# ── [3] mixed gradient/flux BC hits its target, doesn't disturb the rest ────
n = (16, 16, 4)
xi_flat = build_freq_grid(n, L)
phase_grid = np.zeros(n, dtype=int)
phase_grid[n[0] // 2:, :, :] = 1
phase = jnp.array(phase_grid.ravel())
K_field = assemble_K_field(materials, phase)

control = (0, 1, 1)
flux_goal = jnp.array([0.0, -0.7, 0.3])
grad_T, flux, T_prime, delta, grad_bar_out, converged = solve_thermal_conduction(
    n, K_field, xi_flat, grad_bar_x, control=control, flux_goal=flux_goal,
    toler_lin=1e-10, maxiter=3000,
)
assert bool(converged)
flux_bar = jnp.mean(flux, axis=-1)
assert abs(float(flux_bar[1]) - float(flux_goal[1])) < 1e-4
assert abs(float(flux_bar[2]) - float(flux_goal[2])) < 1e-4
assert abs(float(grad_bar_out[0]) - float(grad_bar_x[0])) < 1e-10, \
    "the gradient-controlled entry must stay exactly what was prescribed"
print(f"[3] mixed BC: flux_bar={np.asarray(flux_bar)}  target={np.asarray(flux_goal)}  "
      f"grad_bar_out={np.asarray(grad_bar_out)}")
print("[3] PASSED")

print("\ntest_solvers_elliptic_scalar_thermal: all checks passed")
