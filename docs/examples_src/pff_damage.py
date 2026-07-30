"""
Phase-Field Fracture (AT2) -- Minimal Walkthrough

A minimal walkthrough of FFTjax's staggered phase-field fracture solver: the
strain-based Newton-CG elastic solver (solvers.mechanical.strain_nw_cg
.solve_elastic) coupled with the AT2 Helmholtz damage solver
(solvers.damage.pff_damage.solve_helmholtz_cg) on a homogeneous, isotropic
cube with a small pre-damaged spherical seed at its centre.

The seed acts as a stress concentrator (like a void or pre-crack): under
increasing macroscopic tensile strain, damage nucleates at its surface and
grows outward, softening the macroscopic stress response -- the qualitative
signature of brittle fracture.

Saves its plot to docs/static/img/pff_damage.png for the Examples page.
Re-run this script and copy its printed output into
docs/docs/documentation/examples/phase-field.md if the example ever changes.
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
from mat_models.elastic import LinearElasticIsotropic, assemble_C_field, lame_from_C_field, strain_energy_amor_split
from solvers.mechanical.strain_nw_cg import solve_elastic
from solvers.damage.pff_damage import degradation, update_history, solve_helmholtz_cg

print("JAX backend:", jax.default_backend())
print("Devices:", jax.devices())

# Grid and material: a 32^3 voxel unit cube, single-phase brittle isotropic
# solid -- damage alone does the "cracking", no separate material phases.
n = (32, 32, 32)
L = (1.0, 1.0, 1.0)
Nv = int(np.prod(n))
dx = tuple(Li / ni for Li, ni in zip(L, n))

material = LinearElasticIsotropic(E=20e3, nu=0.2, name="brittle solid")
phase = jnp.zeros(Nv, dtype=int)
C_field = assemble_C_field([material], phase)
lam_vox, mu_vox = lame_from_C_field(C_field)

print(material)

xi_flat = build_freq_grid(n, L)
G_glob = build_green_operator(xi_flat, material.lam, material.mu)


def sphere_seed(n, L, vf):
    """Small central sphere, indicator field in [0, 1] -- a pre-damaged seed."""
    dx = tuple(Li / ni for Li, ni in zip(L, n))
    idx = np.indices(n)
    centers = np.stack([(idx[i] + 0.5) * dx[i] for i in range(3)], axis=0).reshape(3, -1)
    c0 = np.array([Li / 2 for Li in L])
    r = (3.0 * vf / (4.0 * np.pi)) ** (1.0 / 3.0)
    dist = np.linalg.norm(centers - c0[:, None], axis=0)
    return (dist <= r).astype(float)


# A fully-open (d=1) seed is a near-void, ~1e6 stiffness contrast against the
# k_res=1e-6 residual in `degradation()` -- the basic (non-accelerated)
# Lippmann-Schwinger CG scheme used by `solve_elastic` does not converge at
# that contrast within a practical iteration budget (verified: >2000 CG
# iterations, still not converged). Seeding at d=0.9 instead (~100x contrast,
# a strongly pre-damaged but not fully open seed) converges in O(10-100)
# iterations and still grows to d=1 under load.
d_init = jnp.array(sphere_seed(n, L, vf=0.015)) * 0.9
print("seed voxels:", int(jnp.sum(d_init > 0)), f"(Vf={float(jnp.mean(d_init > 0)):.4f})")

# AT2 parameters: length scale a few voxels wide, moderate toughness so the
# seed measurably softens the response within a handful of load steps.
l0 = 3.0 * dx[0]
Gc = 2.0e-2

TOLER_LIN = 1e-6
TOLER_HELM = 1e-4
MAXITER_CG = 3000
MAXITER_HELM = 300
MAXITER_STAGGER = 50
TOLER_STAGGER = 1e-3

d_field = d_init
H_field = jnp.zeros(Nv)

eps_dir = jnp.array([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
])
strain_levels = np.linspace(2.0e-4, 2.4e-3, 8)

results = []
for k, eps_scale in enumerate(strain_levels):
    eps_bar = float(eps_scale) * eps_dir
    d_st = d_field
    for it_st in range(1, MAXITER_STAGGER + 1):
        d_prev = d_st

        g = degradation(d_st)
        C_eff = g[None, None, None, None, :] * C_field

        eps, sigma, delta, it_mech, conv_mech = solve_elastic(
            n, C_eff, G_glob, eps_bar, toler_lin=TOLER_LIN, maxiter=MAXITER_CG,
        )

        psi_pos, _ = strain_energy_amor_split(eps, lam_vox, mu_vox)
        H_field = update_history(H_field, psi_pos)

        d_new, it_helm, conv_helm = solve_helmholtz_cg(
            H_field, xi_flat, n, l0, Gc, d_prev,
            toler_cg=TOLER_HELM, maxiter=MAXITER_HELM,
        )
        d_st = d_new

        change = float(jnp.max(jnp.abs(d_st - d_prev)))
        if change < TOLER_STAGGER:
            break

    assert bool(conv_mech), f"mechanical CG did not converge at step {k+1}"

    d_field = d_st
    sigma11_ave = float(jnp.mean(sigma[0, 0]))
    results.append({
        "eps11": float(eps_scale),
        "sigma11_ave": sigma11_ave,
        "max_d": float(jnp.max(d_field)),
        "staggered_iters": it_st,
    })
    print(f"step {k+1}/{len(strain_levels)}  eps11={float(eps_scale):.2e}  "
          f"sigma11_ave={sigma11_ave:8.4f} MPa  max(d)={float(jnp.max(d_field)):.4f}  "
          f"staggered_iters={it_st:2d}  mech_cg={int(it_mech):4d}  helm_cg={int(it_helm):3d}")

# Sanity checks: damage stays in [0, 1] and is monotone (irreversibility).
assert float(jnp.min(d_field)) >= 0.0 and float(jnp.max(d_field)) <= 1.0
assert float(jnp.min(d_field - d_init)) >= -1e-12, "damage decreased somewhere -- irreversibility violated"
print("\nPASSED -- mechanical CG converged at every step, damage stayed in [0, 1], "
      "and only grew (irreversibility held).")

# Visualize: macroscopic stress-strain curve (softening signature) and the
# final damage field through the seed's mid-plane.
eps11 = [r["eps11"] for r in results]
sigma11 = [r["sigma11_ave"] for r in results]

fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

axes[0].plot(eps11, sigma11, "o-")
axes[0].set_xlabel(r"$\bar\varepsilon_{11}$")
axes[0].set_ylabel(r"$\bar\sigma_{11}$ [MPa]")
axes[0].set_title("Macroscopic response")

d_grid = np.asarray(d_field).reshape(n)
im = axes[1].imshow(d_grid[n[0] // 2, :, :], vmin=0, vmax=1, cmap="inferno")
axes[1].set_title(f"Damage field, x={n[0] // 2} slice")
fig.colorbar(im, ax=axes[1], label="d")

fig.tight_layout()

out_path = Path(__file__).resolve().parents[1] / "static" / "img" / "pff_damage.png"
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=150)
print(f"\nSaved plot to {out_path}")
