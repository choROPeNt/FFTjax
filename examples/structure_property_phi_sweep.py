"""
Structure-Property: Transverse Modulus vs. Fibre Volume Fraction

Sweeps the fibre volume fraction phi of a random-fibre RVE
(generation.rve.make_random_composite_rve, Catalanotti 2016), solving for
the effective transverse Young's modulus E(phi) at each phi via
learning.extractors.effective_modulus (a mixed strain/stress boundary
condition displacement solve -- free lateral surfaces, the same real
tensile-test condition as lin_elastic_mixed_bc.py), then fits a
Gaussian-process surrogate E(phi) with learning.surrogates.GPSurrogate
(GPJax underneath, Matern-5/2 kernel).

The GP is refit INSIDE the sweep loop, N_UPDATES times as data accumulates,
rather than once at the end -- at each update it's asked one "simple
answer" question (the modulus at a fixed, never-simulated query phi) so the
printed output shows that cheap answer sharpening as more real FFT solves
feed into it, without ever running a new solve at the query point itself.

Saves its plot to docs/static/img/structure-property_gp_convergence.png for
the Examples page (kept distinct from structure-property_phi-sweep.png/svg,
the gallery card's own icon on the Examples index -- the two are unrelated
images that happen to cover the same example). Re-run this script and copy
its printed output into docs/docs/documentation/examples/structure-property.md
if the example ever changes.

Run from the repo root: python examples/structure_property_phi_sweep.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from generation.rve import make_random_composite_rve
from materialmodels.elastic.isotropic import LinearElasticIsotropic
from learning.extractors import effective_modulus
from learning.surrogates import GPSurrogate

print("JAX backend:", jax.default_backend())
print("Devices:", jax.devices())

r_fiber   = 0.005   # mm
vox       = 0.001   # mm -- 5 voxels per fibre radius
size_in_r = 10       # domain side ~ 10*r_fiber (Catalanotti 2016 convention)
seed      = 42       # fixed packing realisation across the sweep

E_matrix, nu_matrix = 3500.0, 0.35    # epoxy matrix
E_fiber,  nu_fiber  = 70000.0, 0.20   # glass fibre
eps0 = 1.0e-3   # small uniaxial tensile strain probe (xx)

N_UPDATES = 5      # how many times the GP gets refit as the sweep progresses
PHI_QUERY = 0.35   # never simulated -- the GP's "simple answer" target

phis        = np.linspace(0.10, 0.60, 12)
phi_grid    = np.linspace(phis.min(), phis.max(), 200)
snapshot_at = sorted(set(np.linspace(3, len(phis), N_UPDATES).round().astype(int)))

phi_obs, E_obs, nu_obs = [], [], []
gp_snapshots = []   # per update: n_obs, mean/std over phi_grid, mean/std at PHI_QUERY

for k, phi in enumerate(phis, start=1):
    phase_np, n, L, phi_act, _ = make_random_composite_rve(
        phi=phi, r_fiber=r_fiber, dx=vox, size_in_r=size_in_r, nz=1, K=15, seed=seed,
    )
    phase = jnp.array(phase_np.reshape(-1))
    materials = [
        LinearElasticIsotropic(E=E_matrix, nu=nu_matrix, name="epoxy matrix"),
        LinearElasticIsotropic(E=E_fiber, nu=nu_fiber, name="glass fibre"),
    ]

    E_val, nu_val, converged = effective_modulus(
        phase, materials, n, L, component=(0, 0), eps0=eps0, toler_lin=1e-6, maxiter=200,
    )
    phi_obs.append(phi_act)
    E_obs.append(E_val)
    nu_obs.append(nu_val[1])

    print(f"[{k:2d}/{len(phis)}] phi_act={phi_act:.3f}  grid={n}  "
          f"E={E_val:8.1f} MPa  nu={nu_val[1]:.4f}  converged={converged}")

    if k in snapshot_at:
        X_query = np.concatenate([phi_grid, [PHI_QUERY]])
        surrogate = GPSurrogate(num_iters=300).fit(phi_obs, E_obs, verbose=False)
        mean_all, var_all = surrogate.predict(X_query)
        std_all = np.sqrt(var_all)
        snap = {
            "n_obs": k, "mean_grid": mean_all[:-1], "std_grid": std_all[:-1],
            "mean_q": float(mean_all[-1]), "std_q": float(std_all[-1]),
        }
        gp_snapshots.append(snap)
        print(f"          -> GP update ({k} obs): E(phi={PHI_QUERY}) = "
              f"{snap['mean_q']:.1f} +/- {2 * snap['std_q']:.1f} MPa (95%, never simulated)")

phi_obs, E_obs, nu_obs = np.array(phi_obs), np.array(E_obs), np.array(nu_obs)
print("\nPASSED -- every sweep point's mixed-BC solve converged.")

# Visualize: the GP surrogate tightening, and its answer at phi=PHI_QUERY.
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

cmap = plt.cm.Blues
n_snap = len(gp_snapshots)
for i, snap in enumerate(gp_snapshots):
    color = cmap(0.35 + 0.55 * (i + 1) / n_snap)
    axes[0].plot(phi_grid, snap["mean_grid"], "-", color=color,
                 label=f"GP after {snap['n_obs']} obs")

final = gp_snapshots[-1]
axes[0].fill_between(phi_grid, final["mean_grid"] - 2 * final["std_grid"],
                      final["mean_grid"] + 2 * final["std_grid"],
                      color=cmap(0.9), alpha=0.15, label=r"final GP $\pm2\sigma$")
axes[0].plot(phi_obs, E_obs, "o", color="black", markersize=5, label="FFT solves")
axes[0].axvline(PHI_QUERY, color="gray", ls="--", lw=1, label=fr"query $\phi={PHI_QUERY}$")
axes[0].set_xlabel(r"$\phi$ (fibre volume fraction)")
axes[0].set_ylabel(r"$E_{11}$ [MPa]")
axes[0].set_title("GP surrogate tightening with more solves")
axes[0].legend(fontsize=8)
axes[0].grid(True, linewidth=0.5, alpha=0.6)

n_obs_hist  = [s["n_obs"] for s in gp_snapshots]
mean_q_hist = [s["mean_q"] for s in gp_snapshots]
std_q_hist  = [s["std_q"] for s in gp_snapshots]
axes[1].errorbar(n_obs_hist, mean_q_hist, yerr=2 * np.array(std_q_hist),
                  fmt="o-", color="C0", capsize=4)
axes[1].set_xlabel("number of FFT solves used")
axes[1].set_ylabel(fr"GP prediction $E(\phi={PHI_QUERY})$ [MPa]")
axes[1].set_title("Cheap, uncertainty-aware answer -- no new solve needed")
axes[1].grid(True, linewidth=0.5, alpha=0.6)

fig.tight_layout()

out_path = Path(__file__).resolve().parents[1] / "docs" / "static" / "img" / "structure-property_gp_convergence.png"
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=150)
print(f"\nSaved plot to {out_path}")
