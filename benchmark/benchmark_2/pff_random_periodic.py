"""
PFF benchmark: multiple macroscopic load paths on N random-seed realizations
of a randomly-packed, periodic two-phase composite RVE
(generation.rve.make_random_composite_rve, Catalanotti 2016), phase-field
fracture (AT2 model) -- same random-fibre geometry generator and
transversely-isotropic carbon-fibre model as notebooks/pff-damage.ipynb, but
multiple independent load paths (tension and compression along x, shear
along xy) each compared against a reference curve from assets/. Runs every
load path at every ``PHI_SWEEP`` fibre volume fraction by default (a fresh
set of RVEs generated per phi); pass ``--phi`` to run just one, or
``--realizations`` to change how many.

Realizations: N_REALIZATIONS independent RVEs (seeds SEED, SEED+1, ...) at
            the same phi, solved one at a time in a plain Python loop via
            problems.fracture.solve_fracture_incremental (early-exit,
            full writer/on_increment support). A jax.vmap version was tried
            first (batching all realizations into one traced call) but
            reverted: vmap requires every batch lane to take the same,
            data-independent control-flow path, so the staggered
            mechanics<->damage loop can't early-exit under it -- it has to
            run the full maxiter_st iterations every step regardless of
            actual convergence (solve_fracture_incremental_fixed still
            exists in problems/fracture.py for exactly that use case, just
            not a good fit here: for a handful of realizations that
            normally converge in a couple of staggered iterations, always
            running maxiter_st is far more wasted work than vmap saves by
            batching on-device).

Domain    : random-fibre RVE, size_in_r=15 (Catalanotti 2016 convention),
            nz=1 (plane-strain-like slab)
Materials : carbon fibre (transversely isotropic, E_L=234 GPa/E_T=15 GPa) in
            an epoxy matrix (isotropic, E=3.76 GPa, nu=0.39) -- same constants as
            notebooks/pff-damage.ipynb's carbon fibre.
PFF params: l0 = 3 voxels, Gc/k_res set per-material (matrix Gc=0.8e-3,
            k_res=1e-6, ordinary AT2; fiber Gc=1.6e-3, k_res=1.0,
            damage-immune -- same values as notebooks/pff-damage.ipynb) and
            gathered automatically (Gc=None). The damage-immune fiber keeps
            a rigid load-bearing skeleton through failure, instead of
            softening alongside the matrix.
Boundary  : displacement-controlled -- formulation="displacement", the loaded
condition   diagonal component (e.g. eps_11 for tension_x/compression_x) is
            strain-controlled at the prescribed value, every other surface is
            left traction-free (stress_goal=0, control=1) -- the condition an
            actual tensile/compression specimen is under (grips impose the
            axial strain, lateral surfaces free to contract/expand via
            Poisson's effect), not an artificially strain-constrained cube.
Load paths: tension_x, compression_x, shear_xy -- same RVE realizations (per
            phi), ramped independently (d/H reset between paths) so their
            responses are directly comparable. Each realization's curve is
            overlaid against the reference curve from assets/
            (phi_{35,55,75}_load_{xx,xy}_{tension,comp,shear}.csv).

Every realization gets its own full per-step XDMF/H5 export (via ``writer``)
and raises RuntimeError on non-convergence, same as solve_fracture_incremental
always does -- unlike the reverted vmap path, there's nothing suppressing that
here.

Usage
-----
    python benchmark/benchmark_2/pff_random_periodic.py                     # sweeps PHI_SWEEP
    python benchmark/benchmark_2/pff_random_periodic.py --phi 0.4           # just one phi
    python benchmark/benchmark_2/pff_random_periodic.py --loading tension_x
    python benchmark/benchmark_2/pff_random_periodic.py --realizations 8
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
from post.fields               import to_voigt
from utils.io.xdmf_writer     import IncrementalWriter
from problems.fracture        import solve_fracture_incremental

jax.config.update("jax_enable_x64", True)

# ── RVE generation (shared by every load path) ───────────────────────────────

PHI_SWEEP = [0.35, 0.55, 0.75]   # default sweep -- the phi values assets/ has reference data for
PHI      = PHI_SWEEP[-1]   # current phi being run -- set per iteration in main(); the
                            # module-level default only matters if run_case/build_rve
                            # are called directly (e.g. from a notebook/REPL), not via main()
R_FIBER  = 0.0035     # mm
VOX      = 0.0005     # mm  target voxel size
SIZE_IN_R = 15        # domain side ~ 15*r_fiber (Catalanotti 2016 convention)
K_PERTURB = 15        # perturbation iterations (K>10 -> fully randomised)
SEED      = None      # None = random seed, else int for reproducibility
N_REALIZATIONS = 1    # random-seed realizations (SEED, SEED+1, ...) batched via
                       # jax.vmap per (loading case, phi) -- see run_case

EPS_TENSION_MAX = 1.0e-2    # target macroscopic strain at t=1, tension branch
EPS_COMP_MAX    = 1.0e-2    # target macroscopic strain at t=1, compression branch (magnitude)
l0_factor = 3.0      # phase-field length scale, in voxels

MATRIX = LinearElasticIsotropic(E=3.76e3, nu=0.39,
                                Gc=1000,
                                k_res=1e-6,   # damageable -- ordinary AT2 residual stiffness
                                name="epoxy matrix")
FIBER  = TransverseIsotropic(
    E_L=234000.0, E_T=15000.0, G_LT=15000.0,
    nu_LT=0.20,
    G_TT=7000.0,      # transverse-transverse shear modulus, given directly --
                      # nu_TT is derived (~0.0714) instead of manually rounded
    Gc=1000,
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

# Reference curves in assets/ are only available at these three phi values,
# filenames like load_{ij}_{branch}_phi_{phi:.2f}_.csv -- see voigt_digit's
# docstring for the {ij} numbering convention.
REF_PHIS = {0.35, 0.55, 0.75}


@dataclass
class LoadCase:
    eps_goal: jnp.ndarray
    i: int
    j: int
    comp_symbol: str
    stress_symbol: str
    label: str
    color: str
    ref_branch: str | None = None  # "tension", "comp", or "shear"; None = no reference
                                    # available -- the {ij} digits are derived from
                                    # (i, j) directly (voigt_digit), not stored here
    plot_abs: bool = False         # take abs(eps)/abs(sigma) when plotting -- the
                                    # "comp" reference curves are stored as positive
                                    # magnitude even though the loading is compressive


LOADING_CASES: dict[str, LoadCase] = {
    # "tension_x": LoadCase(
    #     eps_goal=EPS_TENSION_MAX * jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    #     i=0, j=0, comp_symbol="εxx", stress_symbol="σxx",
    #     label="tension x", color="tab:blue",
    #     ref_branch="tension",
    # ),
    # "compression_x": LoadCase(
    #     eps_goal=-EPS_COMP_MAX * jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    #     i=0, j=0, comp_symbol="εxx", stress_symbol="σxx",
    #     label="compression x", color="tab:red",
    #     ref_branch="comp", plot_abs=True,
    # ),
    # "shear_xy": LoadCase(
    #     eps_goal=EPS_TENSION_MAX * jnp.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    #     i=0, j=1, comp_symbol="εxy", stress_symbol="σxy",
    #     label="xy", color="tab:green",
    #     ref_branch="shear",
    # ),
    "shear_zx": LoadCase(
        eps_goal=EPS_TENSION_MAX * jnp.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        i=0, j=2, comp_symbol="εzx", stress_symbol="σzx",
        label="zx", color="tab:green",
        ref_branch="shear",
    ),
}


STRESS_GOAL_ZERO = jnp.zeros((3, 3))   # every stress-controlled surface targets zero traction

# xyz axis convention for history/CSV columns and plot labels -- matches
# post.fields.to_voigt's index order (xx, yy, zz, xy, xz, yz), so
# VOIGT_LABELS[i] pairs 1:1 with to_voigt(tensor)[..., i].
_AXIS = "xyz"
VOIGT_LABELS = ("xx", "yy", "zz", "xy", "xz", "yz")


def voigt_label(i: int, j: int) -> str:
    """(i, j) tensor index -> its xyz Voigt label, e.g. (0, 2) -> 'xz'."""
    return _AXIS[i] + _AXIS[j]


def voigt_digit(axis: int) -> str:
    """
    0-indexed xyz axis -> the reference-data numbering used by assets/
    filenames (load_{ij}_...csv): 1=z, 2=y, 3=x -- the reverse of the usual
    x=1/y=2/z=3 -- i.e. digit = 3 - axis (x/0 -> '3', y/1 -> '2', z/2 -> '1').
    """
    return str(3 - axis)


def voigt_ref_digits(i: int, j: int) -> str:
    """(i, j) tensor index -> the two-digit {ij} label in a reference filename,
    e.g. (i=0,j=2) [x,z] -> digits (3,1) -> ascending -> '13'."""
    return "".join(sorted((voigt_digit(i), voigt_digit(j))))


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
    return ((0, 0, 0), (0, 0, 0), (0, 0, 0))


def load_reference(phi: float, case: LoadCase) -> np.ndarray | None:
    """(eps, sigma) reference array from assets/, or None if unavailable at this phi/case."""
    if case.ref_branch is None:
        return None
    phi_rounded = round(phi, 2)
    if phi_rounded not in REF_PHIS:
        return None
    digits = voigt_ref_digits(case.i, case.j)
    path = os.path.join(REF_DIR, f"load_{digits}_{case.ref_branch}_phi_{phi_rounded:.2f}_.csv")
    if not os.path.exists(path):
        return None
    return np.loadtxt(path, delimiter=",", skiprows=1)


def build_rve_realizations(phi: float, n_realizations: int = N_REALIZATIONS):
    """
    n_realizations independent random-fibre RVEs at the same phi (seeds
    SEED, SEED+1, ..., SEED+n_realizations-1). Grid shape n/L depends only
    on phi/r_fiber/dx/size_in_r, not on the seed (the RNG only perturbs
    fibre positions within a domain whose size is fixed before that step --
    see make_random_composite_rve), so every realization is guaranteed the
    same n/L -- checked here even though nothing downstream needs it to
    match anymore (no more vmap stacking), since a real drift would still
    indicate a bug in the generator.
    """
    phase_raw_list, phi_acts = [], []
    n = L = None
    for i in range(n_realizations):
        seed_i = None if SEED is None else SEED + i
        phase_raw, n_i, L_i, phi_act, _centres = make_random_composite_rve(
            phi=phi, r_fiber=R_FIBER, dx=VOX, size_in_r=SIZE_IN_R, nz=20,
            K=K_PERTURB, seed=seed_i,
        )
        if n is None:
            n, L = n_i, L_i
        elif n_i != n:
            raise RuntimeError(
                f"grid shape differs across realizations: {n_i} != {n} (seed={seed_i})"
            )
        phase_raw_list.append(phase_raw)
        phi_acts.append(phi_act)

    assert n is not None and L is not None, "n_realizations must be >= 1"
    Nv = int(np.prod(n))
    dx = tuple(Li / ni for Li, ni in zip(L, n))
    return phase_raw_list, n, L, Nv, dx, phi_acts


def run_case(name: str, case: LoadCase, phase_raw_list, n, L, Nv, dx, phi_acts, case_dir: str) -> list[dict]:
    n_real = len(phase_raw_list)
    phi_act_mean = float(np.mean(phi_acts))
    print(f"\n=== {name}  (phi_target={PHI:.3f}, phi_actual={phi_act_mean:.4f} +/- {np.std(phi_acts):.4f}, "
          f"{n_real} realizations) ===")
    print(f"Grid     : {n}  (Nv = {Nv})   Domain: {tuple(f'{v:.4g}' for v in L)} mm")
    l0 = l0_factor * dx[0]
    print(f"PFF      : l0 = {l0:.4g} mm")
    for m in MATERIALS:
        print(f"           {m.name}: Gc = {m.Gc} MPa*mm  ->  Gc/l0 = {m.Gc/l0:.3g} MPa")

    jobname = f"pff_random_periodic_phi{PHI:.2f}_{name}"
    control = mixed_control(case)
    print(control)
    history: list[dict] = []

    for k in range(n_real):
        seed_label = "random" if SEED is None else str(SEED + k)
        print(f"\n  --- realization {k}/{n_real - 1}  (seed {seed_label}) ---")
        phase_k = jnp.array(phase_raw_list[k].reshape(-1))
        d_init = jnp.zeros((Nv,))
        H_init = jnp.zeros((Nv,))
        real_jobname = f"{jobname}_real{k}"
        real_history: list[dict] = []

        def _report(r, write_time, k=k, real_history=real_history):
            sol = r.solution
            eps_bar = jnp.mean(sol.eps, axis=-1)
            sigma_bar = jnp.mean(sol.sigma, axis=-1)
            print(
                f"    step {r.step:3d}  t={r.t:.3f}  "
                f"{case.comp_symbol}={float(eps_bar[case.i, case.j]):.2e}  "
                f"{case.stress_symbol}={float(sigma_bar[case.i, case.j]):.2f} MPa  "
                f"max(d)={float(jnp.max(sol.d)):.4f}  "
                f"st={sol.iter_staggered}  err_abs={sol.err_abs:.1e}  "
                f"time={r.wall_time + write_time:.1f}s"
            )
            eps_voigt = np.asarray(to_voigt(np.asarray(eps_bar)))    # (6,) xx,yy,zz,xy,xz,yz
            sig_voigt = np.asarray(to_voigt(np.asarray(sigma_bar)))  # (6,) same order
            row = {
                "realization": k, "step": r.step, "time": float(r.t),
                **{f"eps_{lab}": float(eps_voigt[idx]) for idx, lab in enumerate(VOIGT_LABELS)},
                **{f"sig_{lab}": float(sig_voigt[idx]) for idx, lab in enumerate(VOIGT_LABELS)},
                "max_d": float(jnp.max(sol.d)), "iter_st": sol.iter_staggered,
                "converged_staggered": bool(sol.converged_staggered),
                "err_abs": float(sol.err_abs), "wall_time_s": r.wall_time + write_time,
            }
            history.append(row)
            real_history.append(row)

        with IncrementalWriter(f"{case_dir}/{real_jobname}", grid_shape=n, grid_length=L) as w:
            w.write_increment(0, {
                "phase":             phase_raw_list[k].astype(np.float32),
                "displacement":      np.zeros((*n, 3), dtype=np.float64),
                "strain":            np.zeros((*n, 6), dtype=np.float64),
                "stress":            np.zeros((*n, 6), dtype=np.float64),
                "von_mises":         np.zeros(n, dtype=np.float64),
                "damage":            np.zeros(n, dtype=np.float64),
            }, time=0.0)

            solve_fracture_incremental(
                n, L, phase_k, MATERIALS, case.eps_goal, l0, None, d_init, H_init,
                stepping     = "fixed",
                dt_step      = dt_step,
                formulation  = "displacement",
                control      = control,
                stress_goal  = STRESS_GOAL_ZERO,
                toler_lin    = toler_lin, maxiter_cg=maxiter_cg,
                toler_helm   = toler_helm, maxiter_helm=maxiter_helm,
                toler_st_abs = toler_st_abs, toler_st_rel=toler_st_rel, maxiter_st=maxiter_st,
                writer       = w,
                on_increment = _report,
            )
        print(f"  Written -> {case_dir}/{real_jobname}.h5 / .xdmf")

        real_csv_path = f"{case_dir}/{real_jobname}_history.csv"
        with open(real_csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=real_history[0].keys())
            writer.writeheader()
            writer.writerows(real_history)
        print(f"  Written -> {real_csv_path}")

    csv_path = f"{case_dir}/{jobname}_history.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    print(f"Written -> {csv_path}  (all {n_real} realizations combined)")

    # eps/sig extraction shared by both plots below -- rows: history entries
    # (already 0-indexed by realization/step), returns arrays starting at
    # (0, 0) (the unloaded state) with the shear-strain/abs-value
    # conventions applied.
    axis_label = voigt_label(case.i, case.j)
    eps_key = f"eps_{axis_label}"
    sig_key = f"sig_{axis_label}"
    is_shear = case.i != case.j

    def eps_sig(rows):
        eps = np.array([0.0] + [r[eps_key] for r in rows])
        sig = np.array([0.0] + [r[sig_key] for r in rows])
        if is_shear:
            eps = 2.0 * eps   # engineering shear strain gamma_ij = 2*eps_ij (tensor
                               # shear) -- to_voigt/eps_bar store tensor shear, but
                               # the reference curves report engineering shear
        if case.plot_abs:
            eps, sig = np.abs(eps), np.abs(sig)
        return eps, sig

    strain_sym = rf"\bar{{\gamma}}_{{{axis_label}}}" if is_shear else rf"\bar{{\varepsilon}}_{{{axis_label}}}"
    stress_sym = rf"\bar{{\sigma}}_{{{axis_label}}}"
    ref = load_reference(phi_act_mean, case)
    if ref is None:
        print(f"No reference curve for {name} at phi={phi_act_mean:.2f} -- plotting FFTjax curves alone.")

    # plot 1: every realization's curve overlaid, plus the reference
    fig, ax = plt.subplots(figsize=(6, 4.5))
    if ref is not None:
        ax.plot(ref[:, 0], ref[:, 1], "k--", linewidth=1.2, label="reference")
    for k in range(n_real):
        eps, sig = eps_sig([r for r in history if r["realization"] == k])
        ax.plot(eps, sig, "b-o", markersize=3, linewidth=1.0, alpha=0.6,
                 label="FFTjax" if k == 0 else None)
    ax.set_xlabel(f"${strain_sym}$")
    ax.set_ylabel(f"${stress_sym}$ [MPa]")
    ax.set_title(f"{case.label} -- phi={phi_act_mean:.3f} ({n_real} realizations)")
    ax.legend()
    ax.grid(True, linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    plot_path = f"{case_dir}/{jobname}_comparison.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Written -> {plot_path}")

    # plot 2: mean +/- 1 std across realizations, plus the reference. eps is
    # the strain-controlled (prescribed) direction here, so it's ~identical
    # across realizations -- averaging it is just for a single x per step,
    # not a meaningful statistic in its own right; the std band is on sig.
    eps_per_k = [eps_sig([r for r in history if r["realization"] == k])[0] for k in range(n_real)]
    sig_per_k = [eps_sig([r for r in history if r["realization"] == k])[1] for k in range(n_real)]
    eps_mean = np.mean(eps_per_k, axis=0)
    sig_mean = np.mean(sig_per_k, axis=0)
    sig_std  = np.std(sig_per_k, axis=0)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    if ref is not None:
        ax.plot(ref[:, 0], ref[:, 1], "k--", linewidth=1.2, label="reference")
    ax.fill_between(eps_mean, sig_mean - 3*sig_std, sig_mean + 3*sig_std,
                     color="tab:blue", alpha=0.25, label=r"$\pm$3 std")
    ax.plot(eps_mean, sig_mean, "b-o", markersize=3, linewidth=1.2, label=f"FFTjax mean (n={n_real})")
    ax.set_xlabel(f"${strain_sym}$")
    ax.set_ylabel(f"${stress_sym}$ [MPa]")
    ax.set_title(f"{case.label} -- phi={phi_act_mean:.3f}, mean $\\pm$ std ({n_real} realizations)")
    ax.legend()
    ax.grid(True, linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    mean_std_path = f"{case_dir}/{jobname}_mean_std.png"
    fig.savefig(mean_std_path, dpi=150)
    plt.close(fig)
    print(f"Written -> {mean_std_path}")

    return history


def main():
    global PHI
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phi", type=float, default=None,
                         help=f"single fibre volume fraction to run (default: sweep {PHI_SWEEP})")
    parser.add_argument("--loading", choices=[*LOADING_CASES, "all"], default="all",
                         help="which load path(s) to run (default: all, run sequentially)")
    parser.add_argument("--realizations", type=int, default=N_REALIZATIONS,
                         help=f"random-seed realizations solved per (loading case, phi) (default: {N_REALIZATIONS})")
    args = parser.parse_args()

    os.makedirs(output, exist_ok=True)
    names = list(LOADING_CASES) if args.loading == "all" else [args.loading]
    phis = [args.phi] if args.phi is not None else PHI_SWEEP

    # RVE geometry only depends on phi, not on the load case -- cache it so
    # looping load-case-outer/phi-inner (one subdir per load case) doesn't
    # regenerate the same realizations once per load case.
    rve_cache: dict[float, tuple] = {}

    def get_rve(phi):
        if phi not in rve_cache:
            rve_cache[phi] = build_rve_realizations(phi, args.realizations)
        return rve_cache[phi]

    # each run_case call writes its own {jobname}_comparison.png (FFTjax vs.
    # reference, when assets/ has one for this phi) -- see run_case. One
    # subdirectory per load case, phi swept within it.
    for name in names:
        case_dir = os.path.join(output, name)
        os.makedirs(case_dir, exist_ok=True)
        for phi in phis:
            PHI = phi
            phase_raw_list, n, L, Nv, dx, phi_acts = get_rve(phi)
            run_case(name, LOADING_CASES[name], phase_raw_list, n, L, Nv, dx, phi_acts, case_dir)


if __name__ == "__main__":
    main()
