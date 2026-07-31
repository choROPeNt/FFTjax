"""
Two-Phase Composite RVE -- Linear-Elastic Strain Solve

A minimal walkthrough of FFTjax's strain-based Newton-CG elastic solver
(solvers.mechanical.strain_nw_cg.solve_elastic) on a two-phase composite: a glass-fibre
reinforcement in an epoxy matrix, arranged in a square-packed pattern via
generation.rve.make_square_composite_rve, under a prescribed macroscopic shear strain.

Because the two phases have a large stiffness contrast (~23x), the reference-medium correction is
nontrivial -- the Newton-CG solve actually iterates, redistributing stress between the stiff
fibres and the compliant matrix (unlike a homogeneous material, where it wouldn't need to). Uses
the Willot "rotated" Green's-operator discretisation, which reduces spurious oscillations at phase
interfaces compared to the "standard" scheme. See lin_elastic_mixed_bc.py for the same
geometry/materials under a mixed strain/stress (free-lateral-surface tension) boundary condition
instead of this fully strain-controlled shear.

Saves its plot to docs/static/img/lin_elastic_strain.png for the
Examples page. Re-run this script and copy its printed output into
docs/docs/documentation/examples/lin-elastic-strain.md if the example ever changes.

Run from the repo root: python examples/lin_elastic_strain.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from generation.rve import make_square_composite_rve
from operators.green import build_freq_grid, build_green_operator
from mat_models.elastic import LinearElasticIsotropic, assemble_C_field
from solvers.mechanical.strain_nw_cg import solve_elastic
from post.fields import compute_displacement

print("JAX backend:", jax.default_backend())
print("Devices:", jax.devices())

# Composite RVE: square-packed 2-fibre geometry (Vf~0.5, 5 um fibre radius, 10-voxel-thick slab).
phase_np, N, n, L, phi_act = make_square_composite_rve(
    phi=0.5, r_fiber=0.005, spacing=0.0002, N_min=32, nz=10,
)
Nv = int(np.prod(n))

print("grid n :", n)
print("domain L [mm]:", tuple(float(Li) for Li in L))
print("fiber volume fraction (actual):", phi_act)

# Materials: glass fibre in an epoxy matrix -- a common, high-contrast (~23x) composite.
matrix = LinearElasticIsotropic(E=3.0e3, nu=0.35, name="epoxy matrix")
fiber = LinearElasticIsotropic(E=70.0e3, nu=0.20, name="glass fiber")

phase = jnp.array(phase_np.reshape(-1))  # 0 = matrix, 1 = fiber
C_field = assemble_C_field([matrix, fiber], phase)

print(matrix)
print(fiber)

# Frequency grid and Green's operator: reference medium is the average of the two phases'
# Lame parameters, a reasonable choice when neither phase dominates.
L_mm = tuple(float(Li) for Li in L)
dx = tuple(Li / ni for Li, ni in zip(L_mm, n))
xi_flat = build_freq_grid(n, L_mm)

lam0 = 0.5 * (matrix.lam + fiber.lam)
mu0 = 0.5 * (matrix.mu + fiber.mu)
G_glob = build_green_operator(xi_flat, lam0, mu0, scheme="rotated", dx=dx)

# Prescribe a macroscopic shear strain in the XY plane and solve.
eps_bar = jnp.array([
    [0.0, 1.0e-3, 0.0],
    [1.0e-3, 0.0, 0.0],
    [0.0, 0.0, 0.0],
])

eps, sigma, delta, converged = solve_elastic(
    n, C_field, G_glob, eps_bar, toler_lin=1e-6, maxiter=1000,
)

print("converged     :", bool(converged))
print("tau_xy (avg) :", float(jnp.mean(sigma[1, 0])), "MPa")

assert bool(converged)
print("\nPASSED -- Newton-CG converged on the heterogeneous composite.")

# Visualize: fiber phase and the resulting in-plane displacement field.
u_grid = compute_displacement(eps, eps_bar, xi_flat, n, dx)

extent = [0.0, n[0] * dx[0], 0.0, n[1] * dx[1]]  # physical [mm] extent

fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))

im = axes[0].imshow(phase_np[:, :, 0].T, origin="lower", cmap="gray_r", extent=extent)
axes[0].set_title(f"Fiber phase (Vf={phi_act:.3f})")
fig.colorbar(im, ax=axes[0])

im = axes[1].imshow(u_grid[:, :, 0, 0].T, origin="lower", cmap="plasma", extent=extent)
axes[1].set_title(r"Displacement $u_x$ [mm]")
fig.colorbar(im, ax=axes[1], format="%.1e")

im = axes[2].imshow(u_grid[:, :, 0, 1].T, origin="lower", cmap="plasma", extent=extent)
axes[2].set_title(r"Displacement $u_y$ [mm]")
fig.colorbar(im, ax=axes[2], format="%.1e")

for ax in axes:
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")

fig.tight_layout()

out_path = Path(__file__).resolve().parents[1] / "docs" / "static" / "img" / "lin_elastic_strain.png"
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=150)
print(f"\nSaved plot to {out_path}")
