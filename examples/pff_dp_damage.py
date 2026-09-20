"""
Phase-Field Fracture on a PLASTIC Matrix -- Minimal Walkthrough

The companion to examples/pff_damage.py, with the brittle solid swapped for a
Drucker-Prager elastoplastic one coupled to AT2 damage through the effective
stress (materialmodels.phasefield.drucker_prager.PhaseFieldDruckerPrager).

Same geometry as the brittle example -- a homogeneous cube with a small
pre-damaged spherical seed at its centre, pulled in uniaxial tension -- run
three ways, on one axis:

    brittle              PhaseFieldIsotropic: no plasticity at all
    plastic, beta_p = 0  a brittle crack in a plastic solid: yielding blunts
                         the crack by capping the elastic energy it can store
    plastic, beta_p > 0  ductile: accumulated plastic work ALSO drives the
                         crack, so a voxel that has yielded heavily cracks
                         sooner

Three curves is the point of the example -- but read WHERE they differ.

At these parameters all three share the same PEAK (8.234 MPa at eps_11 =
0.002), because the seed cracks the cube long before the matrix yields:
sigma_y0 is 56.1 MPa while the macroscopic response never exceeds ~8, so
fracture wins the race against yield and nothing has yielded at the peak
(alpha_max is still 0 there). Plasticity acts entirely on the POST-peak
branch, and in opposite directions:

    beta_p = 0   softens SLOWER than brittle (final 0.630 vs 0.128 MPa):
                 plastic dissipation near the crack tip holds residual load
    beta_p = 2   softens faster again (0.305 MPa): plastic work now feeds
                 the driving force, so the crack grows on its own dissipation

Whether plasticity moves the PEAK is a competition between Gc and sigma_y0,
not a property of the model: raise Gc (tougher) or lower sigma_y0 (weaker)
until yield precedes cracking and the peak shifts too. Worth changing one and
re-running to see it -- this configuration deliberately shows the other
regime, the one a stiff, tough matrix with a pre-existing flaw is actually in.

That the three peaks agree to four significant figures is also a check, not a
coincidence: while everything is elastic the two plastic models must reduce
exactly to the brittle one.

Two things to know when reading the output, both documented at length in
materialmodels/phasefield/drucker_prager.py:

  * plastic strain keeps accumulating inside a crack that transmits almost
    nothing, so `alpha_max` is not a meaningful material strain once damage
    has localized;
  * the mechanical (Newton) sub-solve is reliable through the load, the peak
    and the softening, but not past it -- this script stops at the end of the
    softening branch deliberately, and prints `mech` per increment so a
    non-converged one is visible rather than silently plotted.

Saves its plot to docs/static/img/pff_dp_damage.png for the Examples page.
Re-run this script and copy its printed output into
docs/docs/documentation/examples/phase-field-drucker-prager.md if it changes.

Run from the repo root: python examples/pff_dp_damage.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from materialmodels.phasefield.drucker_prager import PhaseFieldDruckerPrager
from materialmodels.phasefield.isotropic import PhaseFieldIsotropic
from problems.fracture import solve_fracture

print("JAX backend:", jax.default_backend())
print("Devices:", jax.devices())

# ── grid and loading ────────────────────────────────────────────────────────
n = (20, 20, 20)
L = (1.0, 1.0, 1.0)
Nv = int(np.prod(n))
l0 = 4.0 * L[0] / n[0]          # >= 4 * voxel size, the usual resolution rule
# Stops at the END of the softening branch, deliberately. Past that the
# mechanical sub-solve stops converging for a reason that is not established
# (see materialmodels/phasefield/drucker_prager.py), so there is nothing
# trustworthy to plot -- and the loop below breaks out at the first
# non-converged increment regardless, rather than drawing it.
EPS_MAX = 0.006
N_STEPS = 12

E, NU = 3.76e3, 0.39
SY, H_LIN = 56.1, 1.0e3
A_F, A_G = 0.13, 0.05           # epsilon_c/epsilon_t ~ 1.3, weak dilatancy
GC = 2.0e-2
# beta_p trades off directly against Gc: at 8.0 and this Gc the ductile case
# collapses within two increments, which shows the mechanism but makes an
# unreadable figure. 2.0 separates the three curves while keeping all of them
# on their converged branch.
BETA_P = 2.0


def sphere_seed(n, L, radius_vox):
    """Small central sphere, damaged -- a pre-crack / stress concentrator."""
    idx = np.indices(n)
    c = np.array([ni / 2 for ni in n])
    dist = np.linalg.norm(
        np.stack([idx[i] + 0.5 - c[i] for i in range(3)]).reshape(3, -1), axis=0)
    return (dist <= radius_vox).astype(float)


d_seed = jnp.array(sphere_seed(n, L, 2.5) * 0.9)
H_seed = jnp.zeros(Nv)
phase = jnp.zeros(Nv, dtype=int)
print(f"\nGrid   : {n}   seed voxels = {int(jnp.sum(d_seed > 0))}/{Nv}   l0 = {l0:.4g}")

CASES = [
    ("brittle", PhaseFieldIsotropic(E=E, nu=NU, Gc=GC, name="brittle")),
    ("plastic, beta_p=0",
     PhaseFieldDruckerPrager(E=E, nu=NU, Gc=GC, sigma_y0=SY, H=H_LIN,
                             a_f=A_F, a_g=A_G, beta_p=0.0, name="dp brittle")),
    (f"plastic, beta_p={BETA_P:g}",
     PhaseFieldDruckerPrager(E=E, nu=NU, Gc=GC, sigma_y0=SY, H=H_LIN,
                             a_f=A_F, a_g=A_G, beta_p=BETA_P, name="dp ductile")),
]

results = {}
for label, material in CASES:
    print(f"\n=== {label} ===")
    print(material)
    # The brittle material takes the linear mechanical sub-solve and the
    # plastic ones the Newton path -- solve_fracture picks per materials list,
    # so this loop needs no special-casing. formulation="displacement" is
    # required by the plastic cases and fine for the brittle one.
    d, Hh, state = d_seed, H_seed, None
    eps_hist, sig_hist, d_hist, al_hist = [0.0], [0.0], [float(jnp.max(d_seed))], [0.0]
    print(f"  {'eps_11':>8s}{'sigma_11':>11s}{'d_max':>8s}{'alpha_max':>11s}"
          f"{'stag':>6s}{'nr':>4s}{'mech':>6s}")
    for k in range(1, N_STEPS + 1):
        eps_bar = jnp.zeros((3, 3)).at[0, 0].set(EPS_MAX * k / N_STEPS)
        sol = solve_fracture(
            n, L, phase, [material], eps_bar, l0, None, d, Hh,
            formulation="displacement",
            toler_lin=1e-8, maxiter_cg=800,
            toler_helm=1e-6, maxiter_helm=300,
            eta=0.0, d_thres=0.95,
            toler_st_abs=1e-4, toler_st_rel=1e-5, maxiter_st=30,
            state_init=state, toler_nr=1e-7, maxiter_nr=40,
        )
        d, Hh = sol.d, sol.H
        if sol.eps_p is not None:
            state = (sol.eps_p, sol.alpha, sol.W_p)
        s11 = float(jnp.mean(sol.sigma[0, 0]))
        al = 0.0 if sol.alpha is None else float(jnp.max(sol.alpha))
        eps_hist.append(float(eps_bar[0, 0]))
        sig_hist.append(s11)
        d_hist.append(float(jnp.max(sol.d)))
        al_hist.append(al)
        nr = "-" if sol.iter_nr is None else f"{sol.iter_nr:d}"
        print(f"  {float(eps_bar[0, 0]):>8.4f}{s11:>11.4f}{float(jnp.max(sol.d)):>8.4f}"
              f"{al:>11.3e}{sol.iter_staggered:>6d}{nr:>4s}"
              f"{str(bool(sol.converged_mech)):>6s}")
        if not bool(sol.converged_mech):
            # Drop this increment and stop the curve here. A non-converged
            # mechanical solve is not an equilibrium state, so plotting it
            # would put a number on the figure that nothing stands behind --
            # and it is exactly where the negative stresses show up.
            for hist in (eps_hist, sig_hist, d_hist, al_hist):
                hist.pop()
            print(f"    -> mechanical solve did not converge; curve truncated at "
                  f"eps_11 = {eps_hist[-1]:.4f}")
            break
    results[label] = (eps_hist, sig_hist, d_hist, al_hist)
    peak = max(sig_hist)
    print(f"  peak sigma_11 = {peak:.3f} MPa at eps_11 = "
          f"{eps_hist[int(np.argmax(sig_hist))]:.4f}; final = {sig_hist[-1]:.3f} MPa")

# ── plot ────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
colors = {"brittle": "#444444"}
for label, (eh, sh, dh, ah) in results.items():
    style = dict(lw=2.0, marker="o", ms=3.5)
    if label == "brittle":
        style.update(color="#444444", ls="--")
    ax1.plot(np.array(eh) * 100, sh, label=label, **style)
    ax2.plot(np.array(eh) * 100, dh, label=label, **style)
ax1.set_xlabel(r"macroscopic strain $\bar\varepsilon_{11}$ [%]")
ax1.set_ylabel(r"macroscopic stress $\bar\sigma_{11}$ [MPa]")
ax1.set_title("homogenized response")
ax1.grid(alpha=0.3)
ax1.legend(fontsize=9)
ax2.set_xlabel(r"macroscopic strain $\bar\varepsilon_{11}$ [%]")
ax2.set_ylabel(r"$\max_x d$")
ax2.set_title("peak damage")
ax2.grid(alpha=0.3)
ax2.legend(fontsize=9)
fig.suptitle("Phase-field fracture: brittle vs Drucker-Prager matrix "
             f"($a_f$={A_F}, $a_g$={A_G}, $G_c$={GC:g})", fontsize=11)
fig.tight_layout()

out = Path(__file__).resolve().parents[1] / "docs" / "static" / "img" / "pff_dp_damage.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150)
print(f"\nSaved plot to {out}")

print("\nSummary -- note the peaks agree (nothing has yielded there);")
print("plasticity acts on the SOFTENING branch, in both directions:")
for label, (eh, sh, dh, ah) in results.items():
    print(f"  {label:<22s} peak={max(sh):8.3f} MPa   final={sh[-1]:8.3f} MPa   "
          f"d_max={dh[-1]:.4f}   alpha_max={ah[-1]:.3e}")
