"""
PFF benchmark: mode-I tension and/or mode-II shear on a single edge notch
plate, phase-field fracture (AT2 model). Combines what were pff_tension.py
and pff_shear.py -- same geometry/material/PFF setup, only the applied
macroscopic strain (and hence output jobname/reference file) differs.

Reference
---------
    Schneider & Kästner (2025)  https://doi.org/10.1111/ffe.14553

Domain      : 250 × 250 × 1 voxels,  L = [50, 50, 0.2] mm
Material    : steel  E = 210 GPa,  ν = 0.3
PFF params  : l₀ = 1.0 mm,  Gc = 2.7 MPa·mm
Pre-crack   : x ∈ [5, 15) mm  (i=[25,75) at 250² grid),  y = 125 (centre)
Load        : 100 equal increments, ramped to
    tension -- εxx → 1.11e-3   (pure-strain BC, ε₁₁ = ε₃₃ = 0)
    shear   -- εxy = εyx → 1.0e-3

Staggered scheme per increment -- see problems.fracture.solve_fracture's
docstring. Load-stepping is problems.fracture.solve_fracture_incremental
with stepping="fixed", dt_step=0.01 -- the original two scripts' hand-rolled
cutback loop was dead code (dt_min == dt_max == dt_init made every
"cutback" clip straight back to the same dt), equivalent to solve_fixed's
plain raise-on-first-non-convergence.

``--loading`` selects which case(s) to run:
    tension  -- only the mode-I case
    shear    -- only the mode-II case
    both (default) -- run both, one after the other

``--solvers`` selects which mechanical-solver configuration(s) of
SOLVER_CONFIGS below to run for each selected loading case -- same
(label, formulation, scheme) pattern as benchmark_3/elastic_solve_vtu.py's
SOLVER_CONFIGS, now exercising problems.fracture.solve_fracture's
formulation="fourier_galerkin" support alongside lippmann_schwinger (rotated
and standard) and displacement:
    all (default) -- every configured solver, sequentially
    a comma-separated subset of labels, e.g. --solvers ls_rotated,galerkin

Every selected (loading, solver) pair is a full independent 100-step
staggered solve -- nothing is shared across solvers beyond the common
geometry/material setup below. For each loading case, once all its solvers
have run, every solver's stress-strain curve is plotted together against
the reference curve on one combined figure, and a timing/summary table is
printed and written to CSV.

Not jax.vmap'd over loading cases or solvers: both solve_fracture's
staggered loop (break on convergence) and solve_fixed's load-stepping loop
(raise on non-convergence) are Python-level control flow over *traced*
convergence values -- vmap requires every batch lane to take the same code
path on different data, but different loadings/solvers can genuinely need a
different number of staggered iterations or fail to converge at a different
step. Batching them would need the staggered/stepping loops rewritten around
lax.while_loop/lax.cond with a fixed iteration budget -- out of scope here;
this script just runs every combination sequentially in one invocation.

Usage
-----
    python benchmark/benchmark_1/pff_single_notch.py
    python benchmark/benchmark_1/pff_single_notch.py --loading tension
    python benchmark/benchmark_1/pff_single_notch.py --loading shear
    python benchmark/benchmark_1/pff_single_notch.py --solvers ls_rotated,galerkin
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
sys.path.insert(0, "src")

import argparse
import csv
import time
import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from materialmodels.elastic.isotropic import LinearElasticIsotropic
from materialmodels.phasefield.isotropic import PhaseFieldIsotropic
from post.fields              import homogenize
from utils.io.xdmf_writer     import IncrementalWriter
from problems.fracture        import solve_fracture_incremental

jax.config.update("jax_enable_x64", True)

# ── Grid (shared by both loading cases) ──────────────────────────────────────

n  = (250, 250, 1)
L  = (50.0, 50.0, 0.2)   # mm
Nv = int(np.prod(n))
dx = tuple(Li / ni for Li, ni in zip(L, n))

materials = [
    # PhaseFieldIsotropic (Amor split + AT2, autodiff tangent) rather than
    # plain LinearElasticIsotropic + degrade_stiffness_field -- this
    # benchmark is this material's validation against the published
    # reference curve, see materialmodels/phasefield/isotropic.py.
    PhaseFieldIsotropic(E=210e3, nu=0.3, Gc=2.7, name="steel"),
    PhaseFieldIsotropic(E=1e-6*210e3,  nu=0.3, Gc=2.7,name="void"),
]

x_crack = (5.0, 15.0)   # mm: [start, end)
ms      = np.zeros(n, dtype=int)
j_crack = n[1] // 2
i_start = round(x_crack[0] / dx[0])
i_end   = round(x_crack[1] / dx[0])
ms[i_start:i_end, j_crack, :] = 1   # void (pre-crack)
phase   = jnp.array(ms.ravel())

l0 = 1.0        # mm  — phase-field length scale
Gc = 2.7        # MPa·mm  (= 2.7 N/mm, AT2 convention from reference)
d_thres = 0.95  # hybrid irreversibility threshold (Steinke & Kaliske 2019)

toler_lin, maxiter_cg     = 1e-2, 500
toler_helm, maxiter_helm  = 1e-3, 300
dt_step                   = 0.01   # 100 equal steps — see module docstring
toler_st_abs, toler_st_rel, maxiter_st = 1e-2, 1e-3, 200
eta = 1e-6   # damage-equation viscous regularisation (Fig. 3b, Schneider & Kästner 2025)

out_root_path = "output_"
output = f"{out_root_path}/benchmark/single_notch_plate"
here   = os.path.dirname(os.path.abspath(__file__))

LOADING_CASES = {
    "tension": dict(
        eps_goal=jnp.array([[0.0, 0.0, 0.0], [0.0, 1.11e-3, 0.0], [0.0, 0.0, 0.0]]),
        jobname="benchmark_pff_tension", ref_csv="ref_tension.csv",
        i=1, j=1, comp_label="eps_xx", stress_label="sig_xx",
        xlabel=r"$\bar{\varepsilon}_{xx}$", ylabel=r"$\bar{\sigma}_{xx}$ [MPa]",
        title="Mode-I tension — single edge notch plate",
    ),
    "shear": dict(
        eps_goal=jnp.array([[0.0, 1e-3, 0.0], [1e-3, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        jobname="benchmark_pff_shear", ref_csv="ref_shear.csv",
        i=0, j=1, comp_label="eps_xy", stress_label="sig_xy",
        xlabel=r"$\bar{\varepsilon}_{xy}$", ylabel=r"$\bar{\sigma}_{xy}$ [MPa]",
        title="Mode-II shear — single edge notch plate",
    ),
}

# (label, formulation, scheme) -- scheme selects GreenOperatorWillot vs.
# GreenOperatorBasic for "lippmann_schwinger", and the equivalent rotated vs.
# standard discretisation of the reference-medium-free projector for
# "fourier_galerkin" (operators.galerkin.GalerkinProjector); passed through
# unchanged for "displacement" too, where solve_fracture ignores it. Mirrors
# benchmark_3/elastic_solve_vtu.py's SOLVER_CONFIGS.
SOLVER_CONFIGS: list[tuple[str, str, str]] = [
    ("ls_rotated",   "lippmann_schwinger", "rotated"),
    ("ls_standard",  "lippmann_schwinger", "standard"),
    ("displacement", "displacement",       "rotated"),
    ("galerkin",     "fourier_galerkin",   "rotated"),
]

# (i, j) tensor index -> the "xyz" letter pair _report's history dict rows
# actually key their eps_{..}/sig_{..} entries by (eps_xx, eps_yy, ..., eps_xy,
# ...) -- NOT "eps_11"/"eps_22"/etc. LOADING_CASES' own i/j are already given
# in i<=j order, so this always lands on a key the dict actually has.
_AXIS = "xyz"


def run_case(name: str, case: dict, solver_label: str, formulation: str, scheme: str) -> list[dict]:
    print(f"\n=== {name}  [{solver_label}: formulation={formulation}, scheme={scheme}] ===")
    print(f"Grid     : {n}  (Nv = {Nv})   Domain: {L} mm")
    print(f"Crack    : x=[{x_crack[0]},{x_crack[1]}) mm  i=[{i_start},{i_end})  y={j_crack}  "
          f"({int((ms==1).sum())} voxels)")
    print(f"PFF      : l₀ = {l0} mm,  Gc = {Gc} MPa·mm  →  Gc/l₀ = {Gc/l0:.3g} MPa")
    print(f"Viscosity: η = {eta:.1e}  →  η/Δt = {eta/dt_step:.1e} MPa")

    i, j = int(case["i"]), int(case["j"])
    comp_label, stress_label = case["comp_label"], case["stress_label"]
    jobname = f"{case['jobname']}__{solver_label}"
    d_init = jnp.zeros((Nv,))
    H_init = jnp.zeros((Nv,))
    history: list[dict] = []

    print(f"{'step':>4} {'t':>6} {comp_label:>10} {stress_label + ' [MPa]':>13} "
          f"{'max(d)':>7} {'st':>3} {'err_abs':>9} {'err_rel':>9} "
          f"{'mech':>4} {'helm':>4} {'time [s]':>8}")

    def _report(r, write_time):
        sol = r.solution
        eps_bar, sigma_bar = homogenize(sol.eps, sol.sigma)
        conv_mech = bool(sol.converged_mech)
        conv_helm = bool(sol.converged_helm)
        print(
            f"{r.step:>4} {r.t:>6.3f} {float(eps_bar[i, j]):>10.2e} "
            f"{float(sigma_bar[i, j]):>13.2f} {float(jnp.max(sol.d)):>7.4f} "
            f"{sol.iter_staggered:>3} {sol.err_abs:>9.1e} {sol.err_rel:>9.1e} "
            f"{'ok' if conv_mech else 'FAIL':>4} {'ok' if conv_helm else 'FAIL':>4} "
            f"{r.wall_time + write_time:>8.1f}"
        )
        history.append({
            "step": r.step, "time": float(r.t), "dt": float(r.dt),
            "eps_xx": float(eps_bar[0, 0]), "eps_yy": float(eps_bar[1, 1]),
            "eps_zz": float(eps_bar[2, 2]), "eps_xy": float(eps_bar[0, 1]),
            "eps_xz": float(eps_bar[0, 2]), "eps_yz": float(eps_bar[1, 2]),
            "sig_xx": float(sigma_bar[0, 0]), "sig_yy": float(sigma_bar[1, 1]),
            "sig_zz": float(sigma_bar[2, 2]), "sig_xy": float(sigma_bar[0, 1]),
            "sig_xz": float(sigma_bar[0, 2]), "sig_yz": float(sigma_bar[1, 2]),
            "max_d": float(jnp.max(sol.d)), "iter_st": sol.iter_staggered,
            "err_abs": sol.err_abs, "err_rel": sol.err_rel,
            "converged_mech": conv_mech, "converged_helm": conv_helm,
            "wall_time_s": r.wall_time + write_time,
        })

    with IncrementalWriter(f"{output}/{jobname}", grid_shape=n, grid_length=L) as w:
        w.write_increment(0, {
            "phase":             ms.astype(np.float32),
            "displacement":      np.zeros((*n, 3), dtype=np.float64),
            "strain":            np.zeros((*n, 6), dtype=np.float64),
            "stress":            np.zeros((*n, 6), dtype=np.float64),
            "von_mises":         np.zeros(n, dtype=np.float64),
            "damage":            np.zeros(n, dtype=np.float64),
            "strain_energy_pos": np.zeros(n, dtype=np.float64),
        }, time=0.0)

        solve_fracture_incremental(
            n, L, phase, materials, case["eps_goal"], l0, Gc, d_init, H_init,
            stepping     = "fixed",
            dt_step      = dt_step,
            formulation  = formulation,
            scheme       = scheme,
            toler_lin    = toler_lin, maxiter_cg=maxiter_cg,
            toler_helm   = toler_helm, maxiter_helm=maxiter_helm,
            eta          = eta, d_thres=d_thres,
            toler_st_abs = toler_st_abs, toler_st_rel=toler_st_rel, maxiter_st=maxiter_st,
            writer       = w,
            on_increment = _report,
        )

    print(f"Written → {output}/{jobname}.h5")
    print(f"          {output}/{jobname}.xdmf")

    csv_path = f"{output}/{jobname}_history.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    print(f"Written → {csv_path}")

    return history


# Distinct marker/colour per solver so the combined curves stay legible.
_SOLVER_STYLE = {
    "ls_rotated":   dict(color="tab:blue",   marker="o"),
    "ls_standard":  dict(color="tab:orange", marker="s"),
    "displacement": dict(color="tab:green",  marker="^"),
    "galerkin":     dict(color="tab:red",    marker="d"),
}


def plot_combined(name: str, case: dict, results: dict[str, list[dict]]) -> None:
    """One figure per loading case: reference curve plus every solver's
    stress-strain curve, so all configured mechanical solvers can be
    compared directly against each other and against Schneider & Kästner
    (2025)."""
    i, j = int(case["i"]), int(case["j"])
    eps_key = f"eps_{_AXIS[i]}{_AXIS[j]}"
    sig_key = f"sig_{_AXIS[i]}{_AXIS[j]}"

    ref = np.loadtxt(os.path.join(here, case["ref_csv"]), delimiter=",", skiprows=1)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ref[:, 0], ref[:, 1], "k--", linewidth=1.2, label="Schneider & Kästner (2025)")
    for solver_label, history in results.items():
        style = _SOLVER_STYLE.get(solver_label, {})
        ax.plot([r[eps_key] for r in history], [r[sig_key] for r in history],
                 color=style.get("color"), marker=style.get("marker"),
                 markersize=3, linewidth=1.2, label=solver_label)
    ax.set_xlabel(case["xlabel"])
    ax.set_ylabel(case["ylabel"])
    ax.set_title(case["title"])
    ax.legend()
    ax.grid(True, linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    plot_path = f"{output}/{case['jobname']}_comparison_all_solvers.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Written → {plot_path}")


def print_summary(rows: list[dict]) -> None:
    print(f"\n{'loading':<8} {'solver':<13} {'formulation':<19} {'scheme':<9} "
          f"{'steps':>6} {'total time [s]':>15} {'final max(d)':>13} "
          f"{'peak stress [MPa]':>18} {'eps at peak':>12} {'mech fails':>10} {'helm fails':>10}")
    for r in rows:
        print(f"{r['loading']:<8} {r['solver_label']:<13} {r['formulation']:<19} {r['scheme']:<9} "
              f"{r['n_steps']:>6} {r['total_time_s']:>15.1f} {r['final_max_d']:>13.4f} "
              f"{r['peak_stress_MPa']:>18.2f} {r['eps_at_peak_stress']:>12.2e} "
              f"{r['n_mech_fail']:>10} {r['n_helm_fail']:>10}")

    summary_path = f"{output}/solver_comparison_summary.csv"
    with open(summary_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWritten → {summary_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--loading", choices=["tension", "shear", "both"], default="both",
                         help="which loading case(s) to run (default: both, run sequentially)")
    parser.add_argument("--solvers", default="all",
                         help="comma-separated SOLVER_CONFIGS labels to run, or 'all' (default): "
                              + ", ".join(label for label, _, _ in SOLVER_CONFIGS))
    args = parser.parse_args()

    os.makedirs(output, exist_ok=True)
    loading_names = list(LOADING_CASES) if args.loading == "both" else [args.loading]

    if args.solvers == "all":
        solver_configs = SOLVER_CONFIGS
    else:
        wanted = {s.strip() for s in args.solvers.split(",")}
        solver_configs = [c for c in SOLVER_CONFIGS if c[0] in wanted]
        missing = wanted - {c[0] for c in solver_configs}
        if missing:
            raise SystemExit(f"unknown --solvers label(s) {sorted(missing)}, expected one of "
                              f"{[label for label, _, _ in SOLVER_CONFIGS]}")

    summary_rows = []
    for loading_name in loading_names:
        case = LOADING_CASES[loading_name]
        results: dict[str, list[dict]] = {}
        for solver_label, formulation, scheme in solver_configs:
            t0 = time.perf_counter()
            history = run_case(loading_name, case, solver_label, formulation, scheme)
            total_time_s = time.perf_counter() - t0
            results[solver_label] = history

            i, j = int(case["i"]), int(case["j"])
            sig_key = f"sig_{_AXIS[i]}{_AXIS[j]}"
            eps_key = f"eps_{_AXIS[i]}{_AXIS[j]}"
            peak = max(history, key=lambda h: h[sig_key])
            summary_rows.append({
                "loading": loading_name, "solver_label": solver_label,
                "formulation": formulation, "scheme": scheme,
                "n_steps": len(history), "total_time_s": total_time_s,
                "final_max_d": history[-1]["max_d"],
                "peak_stress_MPa": peak[sig_key],
                "eps_at_peak_stress": peak[eps_key],
                "n_mech_fail": sum(not h["converged_mech"] for h in history),
                "n_helm_fail": sum(not h["converged_helm"] for h in history),
            })

        plot_combined(loading_name, case, results)

    print_summary(summary_rows)


if __name__ == "__main__":
    main()
