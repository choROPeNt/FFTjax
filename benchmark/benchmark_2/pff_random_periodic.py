"""
PFF benchmark: multiple macroscopic load paths on a randomly-packed,
periodic two-phase composite RVE (generation.rve.make_random_composite_rve,
Catalanotti 2016), phase-field fracture (AT2 model) -- same random-fibre
geometry generator and transversely-isotropic carbon-fibre model as
notebooks/pff-damage.ipynb, but multiple independent load paths (tension and
compression along x so far) each compared against a reference curve from
assets/, at a single, given fibre volume fraction ``PHI`` (or ``--phi``).

Domain    : random-fibre RVE, size_in_r=15 (Catalanotti 2016 convention),
            nz=1 (plane-strain-like slab)
Materials : carbon fibre (transversely isotropic, E_L=234 GPa/E_T=15 GPa) in
            an epoxy matrix (isotropic, E=3.76 GPa, nu=0.39) -- same constants as
            notebooks/pff-damage.ipynb's carbon fibre.
PFF params: l0 = 3 voxels, Gc/k_res set per-material (matrix Gc=0.8e-3,
            k_res=1e-6, ordinary AT2; fiber Gc=1.6e-3, k_res=1.0,
            damage-immune -- same values as notebooks/pff-damage.ipynb) and
            gathered automatically by solve_fracture_incremental (Gc=None).
            The damage-immune fiber keeps a rigid load-bearing skeleton
            through failure, instead of softening alongside the matrix.
Boundary  : displacement-controlled -- formulation="displacement", the loaded
condition   diagonal component (e.g. eps_11 for tension_x/compression_x) is
            strain-controlled at the prescribed value, every other surface is
            left traction-free (stress_goal=0, control=1) -- the condition an
            actual tensile/compression specimen is under (grips impose the
            axial strain, lateral surfaces free to contract/expand via
            Poisson's effect), not an artificially strain-constrained cube.
Load paths: tension_x, compression_x (tension_y/shear_xy defined but commented
            out below -- no reference data for them yet) -- same random RVE,
            ramped independently (d/H reset between paths) so their responses
            are directly comparable on one plot. tension_x/compression_x are
            each overlaid against a reference curve from assets/, when the
            requested --phi matches one of the three phi values assets/ has
            data for (0.35, 0.55, 0.75).

Staggered scheme per increment -- see problems.fracture.solve_fracture's
docstring. Load-stepping is problems.fracture.solve_fracture_incremental
with stepping="fixed".

Usage
-----
    python benchmark/benchmark_2/pff_random_periodic.py
    python benchmark/benchmark_2/pff_random_periodic.py --phi 0.4
    python benchmark/benchmark_2/pff_random_periodic.py --loading tension_x
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
sys.path.insert(0, "src")

import argparse
import csv
from dataclasses import dataclass

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from generation.rve           import make_random_composite_rve
from materialmodels.elastic.isotropic import LinearElasticIsotropic
from materialmodels.elastic.transverse_isotropic import TransverseIsotropic
from post.fields              import homogenize
from utils.io.xdmf_writer     import IncrementalWriter
from problems.fracture        import solve_fracture_incremental

jax.config.update("jax_enable_x64", True)

# ── RVE generation (shared by every load path) ───────────────────────────────

PHI      = 0.55       # target fibre volume fraction -- the one value this run's
                      # geometry is generated at; override with --phi to compare
R_FIBER  = 0.0035     # mm
VOX      = 0.0001     # mm  target voxel size
SIZE_IN_R = 15        # domain side ~ 15*r_fiber (Catalanotti 2016 convention)
K_PERTURB = 15        # perturbation iterations (K>10 -> fully randomised)
SEED      = 67

EPS_TENSION_MAX = 1.0e-3    # target macroscopic strain at t=1, tension branch
EPS_COMP_MAX    = 1.0e-3    # target macroscopic strain at t=1, compression branch (magnitude)
l0_factor = 3.0      # phase-field length scale, in voxels

MATRIX = LinearElasticIsotropic(E=3.76e3, nu=0.39,
                                Gc=0.8e-3,
                                k_res=1e-6,   # damageable -- ordinary AT2 residual stiffness
                                name="epoxy matrix")
FIBER  = TransverseIsotropic(
    E_L=234000.0, E_T=15000.0, G_LT=15000.0, nu_LT=0.20, nu_TT=0.30,
    Gc=1.6e-3,
    k_res=1.0,    # damage-immune -- g(d) == 1 regardless of d, matching
                  # notebooks/pff-damage.ipynb's carbon fibre; keeps the fibre
                  # as a rigid load-bearing skeleton instead of softening
                  # alongside the matrix, which is what flattens the snap-back
    name="carbon fiber",
)
MATERIALS = [MATRIX, FIBER]
# Gc set per-material (above) so solve_fracture_incremental can gather the
# heterogeneous Gc field automatically -- pass Gc=None below, not a scalar.

toler_lin, maxiter_cg     = 1e-2, 2000   # maxiter_cg bumped vs. the reference-medium
                                          # formulation -- the displacement-based system's
                                          # extra stress-controlled unknowns need more CG
                                          # iterations (see notebooks/lin-elastic_mixed-BC.ipynb)
toler_helm, maxiter_helm  = 1e-2, 300
dt_step                   = 0.2   # 100 equal steps -- fine enough to resolve the
                                    # snap-through even at the larger compression strain
toler_st_abs, toler_st_rel, maxiter_st = 1e-2, 1e-3, 50

output   = "output/benchmark/random_periodic"
here     = os.path.dirname(os.path.abspath(__file__))
REF_DIR  = os.path.join(here, "assets")

# Reference curves in assets/ are only available at these three phi values
# (phi_{35,55,75}_load_xx_{tension,comp}.csv) -- keyed by the 2-decimal phi
# this benchmark was run at.
REF_PHI_LABELS = {0.35: "35", 0.55: "55", 0.75: "75"}


@dataclass
class LoadCase:
    eps_goal: jnp.ndarray
    i: int
    j: int
    comp_symbol: str
    stress_symbol: str
    label: str
    color: str
    ref_axis: str | None = None    # e.g. "xx" -- matches assets/phi_{..}_load_{ref_axis}_{ref_branch}.csv
    ref_branch: str | None = None  # "tension" or "comp"; None = no reference available
    plot_abs: bool = False         # take abs(eps)/abs(sigma) when plotting -- the
                                    # "comp" reference curves are stored as positive
                                    # magnitude even though the loading is compressive


LOADING_CASES: dict[str, LoadCase] = {
    "tension_x": LoadCase(
        eps_goal=EPS_TENSION_MAX * jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        i=0, j=0, comp_symbol="εxx", stress_symbol="σxx",
        label="tension x", color="tab:blue",
        ref_axis="xx", ref_branch="tension",
    ),
    "compression_x": LoadCase(
        eps_goal=-EPS_COMP_MAX * jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        i=0, j=0, comp_symbol="εxx", stress_symbol="σxx",
        label="compression x", color="tab:red",
        ref_axis="xx", ref_branch="comp", plot_abs=True,
    ),
    "shear_xy": LoadCase(
        eps_goal=EPS_TENSION_MAX * jnp.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        i=0, j=1, comp_symbol="εxy", stress_symbol="σxy",
        label="xy", color="tab:green",
        ref_axis="xy", ref_branch="shear",
    ),
}


STRESS_GOAL_ZERO = jnp.zeros((3, 3))   # every stress-controlled surface targets zero traction


def mixed_control(case: LoadCase) -> tuple[tuple[int, int, int], ...]:
    """
    Displacement-controlled boundary condition, all surfaces free except the
    loaded one:

    - Uniaxial (i == j, e.g. tension_x/compression_x): strain-controlled on
      the loaded diagonal component, stress-free (traction-free, control=1)
      on the other two normal directions. Shear stays strain-controlled at
      zero (no shear imposed).
    - Shear (i != j, e.g. shear_xy): strain-controlled on the loaded shear
      pair (i, j)/(j, i), stress-free on all three normal directions (every
      lateral surface free to expand/contract). The other, non-loaded shear
      pair stays strain-controlled at zero.
    """
    if case.i == case.j:
        diag = [1, 1, 1]
        diag[case.i] = 0
        return ((diag[0], 0, 0), (0, diag[1], 0), (0, 0, diag[2]))
    return ((1, 0, 0), (0, 1, 0), (0, 0, 1))


def load_reference(phi: float, case: LoadCase) -> np.ndarray | None:
    """(eps, sigma) reference array from assets/, or None if unavailable at this phi/case."""
    if case.ref_axis is None or case.ref_branch is None:
        return None
    label = REF_PHI_LABELS.get(round(phi, 2))
    if label is None:
        return None
    path = os.path.join(REF_DIR, f"phi_{label}_load_{case.ref_axis}_{case.ref_branch}.csv")
    if not os.path.exists(path):
        return None
    return np.loadtxt(path, delimiter=",", skiprows=1)


def build_rve(phi: float):
    phase_raw, n, L, phi_act, _centres = make_random_composite_rve(
        phi=phi, r_fiber=R_FIBER, dx=VOX, size_in_r=SIZE_IN_R, nz=1,
        K=K_PERTURB, seed=SEED,
    )
    Nv = int(np.prod(n))
    dx = tuple(Li / ni for Li, ni in zip(L, n))
    phase = jnp.array(phase_raw.reshape(-1))
    return phase_raw, phase, n, L, Nv, dx, phi_act


def run_case(name: str, case: LoadCase, phase_raw, phase, n, L, Nv, dx, phi_act) -> list[dict]:
    print(f"\n=== {name}  (phi_target={PHI:.3f}, phi_actual={phi_act:.4f}) ===")
    print(f"Grid     : {n}  (Nv = {Nv})   Domain: {tuple(f'{v:.4g}' for v in L)} mm")
    l0 = l0_factor * dx[0]
    print(f"PFF      : l0 = {l0:.4g} mm")
    for m in MATERIALS:
        print(f"           {m.name}: Gc = {m.Gc} MPa*mm  ->  Gc/l0 = {m.Gc/l0:.3g} MPa")

    jobname = f"pff_random_periodic_phi{PHI:.2f}_{name}"
    d_init = jnp.zeros((Nv,))
    H_init = jnp.zeros((Nv,))
    history: list[dict] = []

    def _report(r, write_time):
        sol = r.solution
        eps_bar, sigma_bar = homogenize(sol.eps, sol.sigma)
        print(
            f"  step {r.step:3d}  t={r.t:.3f}  "
            f"{case.comp_symbol}={float(eps_bar[case.i, case.j]):.2e}  "
            f"{case.stress_symbol}={float(sigma_bar[case.i, case.j]):.2f} MPa  "
            f"max(d)={float(jnp.max(sol.d)):.4f}  "
            f"st={sol.iter_staggered}  err_abs={sol.err_abs:.1e}  "
            f"time={r.wall_time + write_time:.1f}s"
        )
        history.append({
            "step": r.step, "time": float(r.t), "dt": float(r.dt),
            "eps_11": float(eps_bar[0, 0]), "eps_22": float(eps_bar[1, 1]),
            "eps_33": float(eps_bar[2, 2]), "eps_12": float(eps_bar[0, 1]),
            "sig_11": float(sigma_bar[0, 0]), "sig_22": float(sigma_bar[1, 1]),
            "sig_33": float(sigma_bar[2, 2]), "sig_12": float(sigma_bar[0, 1]),
            "max_d": float(jnp.max(sol.d)), "iter_st": sol.iter_staggered,
            "err_abs": sol.err_abs, "wall_time_s": r.wall_time + write_time,
        })

    with IncrementalWriter(f"{output}/{jobname}", grid_shape=n, grid_length=L) as w:
        w.write_increment(0, {
            "phase":             phase_raw.astype(np.float32),
            "displacement":      np.zeros((*n, 3), dtype=np.float64),
            "strain":            np.zeros((*n, 6), dtype=np.float64),
            "stress":            np.zeros((*n, 6), dtype=np.float64),
            "von_mises":         np.zeros(n, dtype=np.float64),
            "damage":            np.zeros(n, dtype=np.float64),
        }, time=0.0)

        solve_fracture_incremental(
            n, L, phase, MATERIALS, case.eps_goal, l0, None, d_init, H_init,
            stepping     = "fixed",
            dt_step      = dt_step,
            formulation  = "displacement",
            control      = mixed_control(case),
            stress_goal  = STRESS_GOAL_ZERO,
            toler_lin    = toler_lin, maxiter_cg=maxiter_cg,
            toler_helm   = toler_helm, maxiter_helm=maxiter_helm,
            toler_st_abs = toler_st_abs, toler_st_rel=toler_st_rel, maxiter_st=maxiter_st,
            writer       = w,
            on_increment = _report,
        )

    print(f"Written -> {output}/{jobname}.h5")
    print(f"           {output}/{jobname}.xdmf")

    csv_path = f"{output}/{jobname}_history.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    print(f"Written -> {csv_path}")

    eps_key = f"eps_{case.i+1}{case.j+1}"
    sig_key = f"sig_{case.i+1}{case.j+1}"
    eps = np.array([0.0] + [r[eps_key] for r in history])
    sig = np.array([0.0] + [r[sig_key] for r in history])
    is_shear = case.i != case.j
    if is_shear:
        eps = 2.0 * eps   # engineering shear strain gamma_ij = 2*eps_ij (tensor
                           # shear) -- to_voigt/eps_bar store tensor shear, but
                           # the reference curves report engineering shear
    if case.plot_abs:
        eps, sig = np.abs(eps), np.abs(sig)

    ref = load_reference(phi_act, case)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    if ref is not None:
        ax.plot(ref[:, 0], ref[:, 1], "k--", linewidth=1.2, label="reference")
    else:
        print(f"No reference curve for {name} at phi={phi_act:.2f} -- plotting FFTjax curve alone.")
    ax.plot(eps, sig, "b-o", markersize=3, linewidth=1.2, label="FFTjax")
    strain_sym = rf"\bar{{\gamma}}_{{{case.i+1}{case.j+1}}}" if is_shear else rf"\bar{{\varepsilon}}_{{{case.i+1}{case.j+1}}}"
    ax.set_xlabel(f"${strain_sym}$")
    ax.set_ylabel(rf"$\bar{{\sigma}}_{{{case.i+1}{case.j+1}}}$ [MPa]")
    ax.set_title(f"{case.label} -- phi={phi_act:.3f}")
    ax.legend()
    ax.grid(True, linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    plot_path = f"{output}/{jobname}_comparison.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Written -> {plot_path}")

    return history


def main():
    global PHI
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phi", type=float, default=PHI,
                         help=f"target fibre volume fraction (default: {PHI})")
    parser.add_argument("--loading", choices=[*LOADING_CASES, "all"], default="all",
                         help="which load path(s) to run (default: all, run sequentially)")
    args = parser.parse_args()
    PHI = args.phi

    os.makedirs(output, exist_ok=True)
    names = list(LOADING_CASES) if args.loading == "all" else [args.loading]

    phase_raw, phase, n, L, Nv, dx, phi_act = build_rve(PHI)

    # each run_case call writes its own {jobname}_comparison.png (FFTjax vs.
    # reference, when assets/ has one for this phi) -- see run_case.
    for name in names:
        run_case(name, LOADING_CASES[name], phase_raw, phase, n, L, Nv, dx, phi_act)


if __name__ == "__main__":
    main()
