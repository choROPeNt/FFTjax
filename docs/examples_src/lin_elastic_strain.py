"""
Linear-Elastic Strain Solve

A minimal walkthrough of FFTjax's strain-based Newton-CG elastic solver
(solvers.mechanical.strain_nw_cg.solve_elastic) on a homogeneous, isotropic
cube under a prescribed macroscopic strain.

Because the material is homogeneous and the reference medium is chosen to
match it exactly, the correction field is exactly zero -- the local strain
equals the prescribed macroscopic strain everywhere, and the local stress
follows directly from Hooke's law. This makes the result easy to check by
hand, so this script doubles as a sanity check of the install.

Saves its plot to docs/static/img/lin_elastic_strain.png for the
Examples page. Re-run this script and copy its printed output into
docs/docs/examples/lin-elastic-strain.md if the example ever changes.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from operators.green import build_freq_grid, build_green_operator
from mat_models.elastic import LinearElasticIsotropic, assemble_C_field
from solvers.mechanical.strain_nw_cg import solve_elastic

print("JAX backend:", jax.default_backend())
print("Devices:", jax.devices())

# Grid and material: a 32^3 voxel unit cube, single-phase isotropic steel.
n = (32, 32, 32)
L = (1.0, 1.0, 1.0)
Nv = int(np.prod(n))

material = LinearElasticIsotropic(E=210e3, nu=0.3, name="steel")
phase = jnp.zeros(Nv, dtype=int)  # homogeneous: single phase everywhere
C_field = assemble_C_field([material], phase)

print(material)

# Frequency grid and Green's operator: the reference medium (lam0, mu0) is
# set to the material's own Lame parameters -- exact for a homogeneous material.
xi_flat = build_freq_grid(n, L)
G_glob = build_green_operator(xi_flat, material.lam, material.mu)

print("xi_flat shape:", xi_flat.shape)
print("G_glob shape :", G_glob.shape)

# Prescribe a macroscopic strain and solve: a uniaxial strain of 1e-3 along x.
eps_bar = jnp.array([
    [1.0e-3, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
])

eps, sigma, delta, it, converged = solve_elastic(n, C_field, G_glob, eps_bar)

print("CG iterations:", int(it))
print("converged     :", bool(converged))

# Check against Hooke's law: for a homogeneous material the local strain must
# equal eps_bar everywhere, and the local stress must equal the direct
# Hooke's-law prediction C : eps_bar.
sigma_analytic = material.stress_field(jnp.broadcast_to(eps_bar[:, :, None], (3, 3, Nv)))

max_err_eps = float(jnp.max(jnp.abs(eps - eps_bar[:, :, None])))
max_err_sigma = float(jnp.max(jnp.abs(sigma - sigma_analytic)))

print(f"max |eps  - eps_bar|        = {max_err_eps:.3e}")
print(f"max |sigma - C:eps_bar|     = {max_err_sigma:.3e}")

assert max_err_eps < 1e-10
assert max_err_sigma < 1e-8
print("\nPASSED -- local fields match the analytic homogeneous solution.")

# Visualize: solved vs. analytic diagonal stress.
solved_diag = np.array([float(jnp.mean(sigma[i, i])) for i in range(3)])
analytic_diag = np.array([float(jnp.mean(sigma_analytic[i, i])) for i in range(3)])

labels = [r"$\sigma_{11}$", r"$\sigma_{22}$", r"$\sigma_{33}$"]
x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(5, 3.5))
ax.bar(x - width / 2, solved_diag, width, label="solved (FFT)")
ax.bar(x + width / 2, analytic_diag, width, label="analytic (Hooke's law)")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("stress [MPa]")
ax.set_title("Homogeneous cube under uniaxial strain")
ax.legend()
fig.tight_layout()

out_path = Path(__file__).resolve().parents[1] / "static" / "img" / "lin_elastic_strain.png"
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=150)
print(f"\nSaved plot to {out_path}")
