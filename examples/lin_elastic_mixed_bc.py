"""
Two-Phase Composite RVE -- Mixed Strain/Stress Boundary Conditions

A more realistic walkthrough of FFTjax's mechanical solver: instead of prescribing the full
macroscopic strain tensor (as in lin_elastic_strain.py), this script prescribes a mixed boundary
condition -- displacement-controlled tension along x with free (traction-free) lateral surfaces
along y and z, the condition an actual tensile-test specimen is under (axial extension imposed at
the grips, lateral surfaces free to contract via Poisson's effect). The point is to measure the
composite's effective Poisson's ratios, which a fully strain-constrained case cannot show.

Uses solvers.mechanical.displacement_nw_cg.ddisp_nw_cg, the displacement-based solver -- required
here because the strain-based solver's cheaper mixed-BC variant (dstrain_nw_cg_mixed) is only
valid for homogeneous materials; our fibre/matrix composite is heterogeneous.

Saves its plot to docs/static/img/lin_elastic_mixed_bc.png for the Examples page. Re-run this
script and copy its printed output into
docs/docs/documentation/examples/lin-elastic-mixed-bc.md if the example ever changes.

Run from the repo root: python examples/lin_elastic_mixed_bc.py
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
from operators.green import build_freq_grid
from mat_models.elastic import LinearElasticIsotropic, assemble_C_field
from solvers.mechanical.displacement_nw_cg import ddisp_nw_cg
from post.fields import field_to_grid, compute_displacement

print("JAX backend:", jax.default_backend())
print("Devices:", jax.devices())

# Composite RVE: square-packed 2-fibre geometry, same as lin_elastic_strain.py's
# composite section, so the two examples are directly comparable.
phase_np, N, n, L, phi_act = make_square_composite_rve(
    phi=0.5, r_fiber=0.005, dx=0.0002, N_min=32, nz=10,
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

# Mixed boundary conditions: xx strain-controlled (tension), yy/zz stress-controlled
# (free lateral surfaces, target 0), shear strain-controlled at 0.
L_mm = tuple(float(Li) for Li in L)
dx = tuple(Li / ni for Li, ni in zip(L_mm, n))
xi_flat = build_freq_grid(n, L_mm)

eps_bar = jnp.array([
    [1.0e-3, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
])
control = (
    (0, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
)
stress_goal = jnp.array([
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
])

eps, sigma, delta, eps_bar_out, converged = ddisp_nw_cg(
    n, C_field, xi_flat, eps_bar, control, stress_goal, toler_lin=1e-6, maxiter=2000,
)

print("converged     :", bool(converged))
print("eps_bar (solved):")
print(np.array(eps_bar_out))
print()
print("sigma11 (avg) :", float(jnp.mean(sigma[0, 0])), "MPa")
print("sigma22 (avg) :", float(jnp.mean(sigma[1, 1])), "MPa  (target: 0, free surface)")
print("sigma33 (avg) :", float(jnp.mean(sigma[2, 2])), "MPa  (target: 0, free surface)")

assert abs(float(jnp.mean(sigma[1, 1]))) < 1e-3
assert abs(float(jnp.mean(sigma[2, 2]))) < 1e-3
print("\nPASSED -- free lateral surfaces converged to sigma22 = sigma33 = 0.")

nu_xy = float(-eps_bar_out[1, 1] / eps_bar_out[0, 0])
nu_xz = float(-eps_bar_out[2, 2] / eps_bar_out[0, 0])
print()
print("effective nu_xy = -eps22/eps11 :", nu_xy)
print("effective nu_xz = -eps33/eps11 :", nu_xz)

# Visualize: fiber phase and the resulting in-plane displacement field.
eps_grid = field_to_grid(eps, n)
u_grid = compute_displacement(eps, eps_bar_out, xi_flat, n, dx)

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

out_path = Path(__file__).resolve().parents[1] / "docs" / "static" / "img" / "lin_elastic_mixed_bc.png"
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=150)
print(f"\nSaved plot to {out_path}")
