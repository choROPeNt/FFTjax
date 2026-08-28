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
    tension -- ε₂₂ → 1.11e-3   (pure-strain BC, ε₁₁ = ε₃₃ = 0)
    shear   -- ε₁₂ = ε₂₁ → 1.0e-3

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

Not jax.vmap'd over the two cases: both solve_fracture's staggered loop
(break on convergence) and solve_fixed's load-stepping loop (raise on
non-convergence) are Python-level control flow over *traced* convergence
values -- vmap requires every batch lane to take the same code path on
different data, but tension and shear can genuinely need a different
number of staggered iterations or fail to converge at a different step.
Batching them would need the staggered/stepping loops rewritten around
lax.while_loop/lax.cond with a fixed iteration budget -- out of scope here;
"both" just runs the two cases sequentially in one invocation.

Usage
-----
    python benchmark/single_notch_plate/pff_single_notch.py
    python benchmark/single_notch_plate/pff_single_notch.py --loading tension
    python benchmark/single_notch_plate/pff_single_notch.py --loading shear
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
sys.path.insert(0, "src")

import argparse
import csv
import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from materialmodels.elastic.isotropic import LinearElasticIsotropic
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
    LinearElasticIsotropic(E=210e3, nu=0.3, name="steel"),
    LinearElasticIsotropic(E=1e-6*210e3,  nu=0.3, name="void"),
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

output = "output/benchmark/single_notch_plate"
here   = os.path.dirname(os.path.abspath(__file__))

LOADING_CASES = {
    "tension": dict(
        eps_goal=jnp.array([[0.0, 0.0, 0.0], [0.0, 1.11e-3, 0.0], [0.0, 0.0, 0.0]]),
        jobname="benchmark_pff_tension", ref_csv="ref_tension.csv",
        i=1, j=1, comp_symbol="ε₂₂", stress_symbol="σ₂₂",
        xlabel=r"$\bar{\varepsilon}_{22}$", ylabel=r"$\bar{\sigma}_{22}$ [MPa]",
        title="Mode-I tension — single edge notch plate",
    ),
    "shear": dict(
        eps_goal=jnp.array([[0.0, 1e-3, 0.0], [1e-3, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        jobname="benchmark_pff_shear", ref_csv="ref_shear.csv",
        i=0, j=1, comp_symbol="ε₁₂", stress_symbol="σ₁₂",
        xlabel=r"$\bar{\varepsilon}_{12}$", ylabel=r"$\bar{\sigma}_{12}$ [MPa]",
        title="Mode-II shear — single edge notch plate",
    ),
}


def run_case(name: str, case: dict) -> None:
    print(f"\n=== {name} ===")
    print(f"Grid     : {n}  (Nv = {Nv})   Domain: {L} mm")
    print(f"Crack    : x=[{x_crack[0]},{x_crack[1]}) mm  i=[{i_start},{i_end})  y={j_crack}  "
          f"({int((ms==1).sum())} voxels)")
    print(f"PFF      : l₀ = {l0} mm,  Gc = {Gc} MPa·mm  →  Gc/l₀ = {Gc/l0:.3g} MPa")
    print(f"Viscosity: η = {eta:.1e}  →  η/Δt = {eta/dt_step:.1e} MPa")

    i, j = case["i"], case["j"]
    jobname = case["jobname"]
    d_init = jnp.zeros((Nv,))
    H_init = jnp.zeros((Nv,))
    history: list[dict] = []

    def _report(r, write_time):
        sol = r.solution
        eps_bar, sigma_bar = homogenize(sol.eps, sol.sigma)
        print(
            f"  step {r.step:3d}  t={r.t:.3f}  "
            f"{case['comp_symbol']}={float(eps_bar[i, j]):.2e}  "
            f"{case['stress_symbol']}={float(sigma_bar[i, j]):.2f} MPa  "
            f"max(d)={float(jnp.max(sol.d)):.4f}  "
            f"st={sol.iter_staggered}  err_abs={sol.err_abs:.1e}  err_rel={sol.err_rel:.1e}  "
            f"time={r.wall_time + write_time:.1f}s"
        )
        history.append({
            "step": r.step, "time": float(r.t), "dt": float(r.dt),
            "eps_11": float(eps_bar[0, 0]), "eps_22": float(eps_bar[1, 1]),
            "eps_33": float(eps_bar[2, 2]), "eps_12": float(eps_bar[0, 1]),
            "eps_13": float(eps_bar[0, 2]), "eps_23": float(eps_bar[1, 2]),
            "sig_11": float(sigma_bar[0, 0]), "sig_22": float(sigma_bar[1, 1]),
            "sig_33": float(sigma_bar[2, 2]), "sig_12": float(sigma_bar[0, 1]),
            "sig_13": float(sigma_bar[0, 2]), "sig_23": float(sigma_bar[1, 2]),
            "max_d": float(jnp.max(sol.d)), "iter_st": sol.iter_staggered,
            "err_abs": sol.err_abs, "err_rel": sol.err_rel,
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
            scheme       = "rotated",
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

    ref = np.loadtxt(os.path.join(here, case["ref_csv"]), delimiter=",", skiprows=1)
    eps_key = f"eps_{i+1}{j+1}"
    sig_key = f"sig_{i+1}{j+1}"

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ref[:, 0], ref[:, 1], "k--", linewidth=1.2, label="Schneider & Kästner (2025)")
    ax.plot([r[eps_key] for r in history], [r[sig_key] for r in history],
            "b-o", markersize=3, linewidth=1.2, label="FFTjax")
    ax.set_xlabel(case["xlabel"])
    ax.set_ylabel(case["ylabel"])
    ax.set_title(case["title"])
    ax.legend()
    ax.grid(True, linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    plot_path = f"{output}/{jobname}_comparison.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Written → {plot_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--loading", choices=["tension", "shear", "both"], default="both",
                         help="which loading case(s) to run (default: both, run sequentially)")
    args = parser.parse_args()

    os.makedirs(output, exist_ok=True)
    names = list(LOADING_CASES) if args.loading == "both" else [args.loading]
    for name in names:
        run_case(name, LOADING_CASES[name])


if __name__ == "__main__":
    main()
