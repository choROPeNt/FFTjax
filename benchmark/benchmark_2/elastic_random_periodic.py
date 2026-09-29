"""
Elastic-only companion to archiv/pff_random_periodic.py: linear elastic homogenization
on the SAME random-fibre RVE geometry and fibre/matrix elastic constants, but
with none of the phase-field machinery -- no Gc/k_res, no interphase, no
staggered mechanics<->damage loop, no load-stepping (a single macroscopic
strain is linear, so one CG solve is the whole answer). Just
problems.mechanics.solve_mechanics under the same free-lateral-surface mixed
BC (tension/compression along x, in-plane shear) as the PFF benchmark, one
solve per (load case, realization), reporting the resulting effective
modulus/shear modulus.

Domain    : random-fibre RVE, size_in_r=15 (Catalanotti 2016 convention),
            nz=20 -- same geometry constants as archiv/pff_random_periodic.py.
Materials : carbon fibre (transversely isotropic, E_L=234 GPa/E_T=15 GPa) in
            an epoxy matrix (isotropic, E=3.76 GPa, nu=0.39) -- same elastic
            constants as archiv/pff_random_periodic.py, without the phase-field
            (Gc/k_res) arguments, which this script never uses.
Boundary  : mixed strain/stress via ``control`` (problems.utils.loadcases.
            free_surface_control) -- the loaded component is strain-
            controlled, every other surface traction-free, same convention
            as archiv/pff_random_periodic.py.

Usage
-----
    python benchmark/benchmark_2/elastic_random_periodic.py
    python benchmark/benchmark_2/elastic_random_periodic.py --phi 0.35
    python benchmark/benchmark_2/elastic_random_periodic.py --loading tension_x --realizations 5
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
sys.path.insert(0, "src")

import argparse
import csv
from typing import cast

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)
import jax
import jax.numpy as jnp
import numpy as np

from generation.rve           import make_random_composite_rve
from materialmodels.elastic   import LinearElasticIsotropic, TransverseIsotropic
from problems.mechanics       import solve_mechanics
from problems.utils.loadcases import free_surface_control
from solvers.solution         import ElasticitySolution
from utils.io.xdmf_writer     import IncrementalWriter

jax.config.update("jax_enable_x64", True)

PHI_SWEEP      = [0.35, 0.55, 0.75]
R_FIBER        = 0.0035   # mm
VOX            = 0.00015  # mm  target voxel size
SIZE_IN_R      = 15       # domain side ~ 15*r_fiber (Catalanotti 2016 convention)
K_PERTURB      = 15       # perturbation iterations (K>10 -> fully randomised)
SEED           = None     # None = random seed, else int for reproducibility
N_REALIZATIONS = 5

EPS_MAX = 1.0e-2   # macroscopic strain magnitude for every load case -- linear elastic,
                   # so the response at any other strain is just a rescaling of this one

MATRIX = LinearElasticIsotropic(E=3.76e3, nu=0.39, name="epoxy matrix")
FIBER  = TransverseIsotropic(
    E_L=234000.0, E_T=15000.0, G_LT=15000.0, nu_LT=0.20, G_TT=7000.0,
    name="carbon fiber",
)
MATERIALS = [MATRIX, FIBER]

toler_lin, maxiter_cg = 1e-6, 2000

output = "output/benchmark/benchmark_2_elastic"

# name -> (i, j, sign, is_shear) -- sign flips tension_x into compression_x;
# is_shear picks the engineering-shear-consistent eps_bar (gamma/2 off-diagonal).
# Same five load cases as archiv/pff_random_periodic.py's LOADING_CASES.
LOAD_CASES: dict[str, tuple[int, int, float, bool]] = {
    "tension_z":     (2, 2,  1.0, False),
    "tension_x":     (0, 0,  1.0, False),
    "compression_x": (0, 0, -1.0, False),
    "shear_xy":      (0, 1,  1.0, True),
    "shear_zx":      (0, 2,  1.0, True),
}


def eps_bar_for(i: int, j: int, gamma: float, is_shear: bool) -> jnp.ndarray:
    """(3, 3) macroscopic strain for one load case -- engineering shear
    convention (gamma/2 tensor components) for i != j, matching
    problems.utils.loadcases.LoadCase.eps_bar."""
    eps = jnp.zeros((3, 3))
    if is_shear:
        eps = eps.at[i, j].set(gamma / 2.0).at[j, i].set(gamma / 2.0)
    else:
        eps = eps.at[i, i].set(gamma)
    return eps


def build_rve_realizations(phi: float, n_realizations: int = N_REALIZATIONS):
    """Same n_realizations-independent-RVEs-at-fixed-phi pattern as
    archiv/pff_random_periodic.py's own build_rve_realizations, minus the interphase
    this script never uses."""
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
    return phase_raw_list, n, L, phi_acts


def run_case(name: str, phase_raw_list, n, L, phi_acts, case_dir: str) -> list[dict]:
    i, j, sign, is_shear = LOAD_CASES[name]
    eps_bar = eps_bar_for(i, j, sign * EPS_MAX, is_shear)
    control = free_surface_control(i, j)
    n_real = len(phase_raw_list)
    phi_act_mean = float(np.mean(phi_acts))
    print(f"\n=== {name}  (phi_actual={phi_act_mean:.4f} +/- {np.std(phi_acts):.4f}, "
          f"{n_real} realizations) ===")
    print(f"Grid: {n}   Domain: {tuple(f'{v:.4g}' for v in L)} mm")

    rows: list[dict] = []
    for k in range(n_real):
        phase_k = jnp.array(phase_raw_list[k].reshape(-1))
        real_path = f"{case_dir}/{name}_real{k}"
        with IncrementalWriter(real_path, grid_shape=n, grid_length=L) as w:
            results = solve_mechanics(
                n, L, phase_k, MATERIALS, eps_bar,
                formulation="displacement", control=control,
                toler_lin=toler_lin, maxiter=maxiter_cg,
                writer=w,   # phase/strain/stress/von_mises/displacement fields -> {real_path}.h5/.xdmf
            )
        sol = cast(ElasticitySolution, results[0].solution)
        eps_bar_out = sol.eps_bar if sol.eps_bar is not None else eps_bar
        sigma_bar = jnp.mean(sol.sigma, axis=-1)

        driven_eps = float(eps_bar_out[i, j])
        driven_sig = float(sigma_bar[i, j])
        modulus = driven_sig / (driven_eps if not is_shear else 2.0 * driven_eps)
        label = "G_eff" if is_shear else "E_eff"

        print(f"  realization {k}: eps={driven_eps:.4e}  sigma={driven_sig:8.3f} MPa  "
              f"{label}={modulus:9.2f} MPa  converged={bool(sol.converged)}")
        rows.append({
            "realization": k, "phi_actual": phi_acts[k],
            "eps": driven_eps, "sigma": driven_sig, "modulus": modulus,
            "converged": bool(sol.converged),
        })

    moduli = [r["modulus"] for r in rows]
    print(f"  {label} = {np.mean(moduli):.2f} +/- {np.std(moduli):.2f} MPa "
          f"(mean +/- std over {n_real} realizations)")

    csv_path = f"{case_dir}/{name}_history.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Written -> {csv_path}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phi", type=float, default=None,
                         help=f"single fibre volume fraction to run (default: sweep {PHI_SWEEP})")
    parser.add_argument("--loading", choices=[*LOAD_CASES, "all"], default="all",
                         help="which load case(s) to run (default: all)")
    parser.add_argument("--realizations", type=int, default=N_REALIZATIONS,
                         help=f"random-seed realizations per (loading case, phi) (default: {N_REALIZATIONS})")
    args = parser.parse_args()

    os.makedirs(output, exist_ok=True)
    names = list(LOAD_CASES) if args.loading == "all" else [args.loading]
    phis = [args.phi] if args.phi is not None else PHI_SWEEP

    for phi in phis:
        phase_raw_list, n, L, phi_acts = build_rve_realizations(phi, args.realizations)
        case_dir = os.path.join(output, f"phi_{phi:.2f}")
        os.makedirs(case_dir, exist_ok=True)
        for name in names:
            run_case(name, phase_raw_list, n, L, phi_acts, case_dir)


if __name__ == "__main__":
    main()
