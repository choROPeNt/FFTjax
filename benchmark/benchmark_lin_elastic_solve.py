"""
Sweeps grid resolution on a fixed quad_rve (square-packed 2-fibre composite,
generation.rve.make_square_composite_rve) comparing FFTjax's two elastic
solver formulations, both driven through problems.mechanics.solve_mechanics
on the identical fully strain-controlled BC:
- lippmann_schwinger -- reference-medium CG correction (Willot 'rotated' scheme)
- displacement        -- true heterogeneous tangent, no reference medium

Glass fibre in an epoxy matrix (~23x stiffness contrast, same materials as
notebooks/lin-elastic_strain.ipynb) under a prescribed macroscopic shear
strain, so both solvers' CG correction does real work rather than converging
trivially in one step.

Writes results to docs/static/data/benchmark_lin_elastic_solve.json for the
interactive Benchmark page (docs/docs/documentation/benchmark.mdx).
"""
import sys
sys.path.insert(0, "src")

import json
import os
import time
import datetime
from typing import cast

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)
import jax
import jax.numpy as jnp
import numpy as np

from generation.rve import make_square_composite_rve
from materialmodels.elastic.isotropic import LinearElasticIsotropic
from problems.mechanics import solve_mechanics
from solvers.elliptic.vector.base import ElasticitySolution

GRID_SIZES = [16, 24, 32, 48, 64, 96]
PHI = 0.5           # target fibre volume fraction
R_FIBER = 0.005      # fibre radius [mm]
DX_COARSE = 1.0      # deliberately >> the RVE's side length, so N_min alone
                      # controls resolution (see make_square_composite_rve:
                      # N = max(N_min, ceil(L_side/dx)) -- a huge dx forces
                      # ceil(...) to 1, leaving N_min as the only lever)
TOLER_LIN = 1e-6
MAXITER = 2000
OUT_PATH = "docs/static/data/benchmark_lin_elastic_solve.json"

MATRIX = LinearElasticIsotropic(E=3.0e3, nu=0.35, name="epoxy matrix")
FIBER = LinearElasticIsotropic(E=70.0e3, nu=0.20, name="glass fiber")
MATERIALS = [MATRIX, FIBER]
EPS_BAR = jnp.array([
    [0.0, 1.0e-3, 0.0],
    [1.0e-3, 0.0, 0.0],
    [0.0, 0.0, 0.0],
])

# (label, formulation, scheme) -- scheme is ignored by solve_mechanics unless
# formulation == "lippmann_schwinger", so it's harmless to pass either way.
SOLVER_FORMULATIONS: list[tuple[str, str, str]] = [
    ("ls",   "lippmann_schwinger", "rotated"),
    ("disp", "displacement",       "rotated"),
]


def bench(N, repeats=3):
    phase_np, n, L, phi_act = make_square_composite_rve(
        phi=PHI, r_fiber=R_FIBER, dx=DX_COARSE, N_min=N, nz=N,
    )
    Nv = int(np.prod(n))
    phase = jnp.array(phase_np.reshape(-1))

    out = {"n": list(n), "elements": Nv, "volume_fraction": phi_act}

    for label, formulation, scheme in SOLVER_FORMULATIONS:
        # first call: trace + compile + run
        t0 = time.perf_counter()
        results = solve_mechanics(
            n, L, phase, MATERIALS, EPS_BAR, formulation=formulation, scheme=scheme,
            toler_lin=TOLER_LIN, maxiter=MAXITER,
        )
        sol = cast(ElasticitySolution, results[0].solution)
        sol.eps.block_until_ready()
        compile_ms = (time.perf_counter() - t0) * 1000.0
        converged = bool(sol.converged)

        # steady-state: already compiled
        t0 = time.perf_counter()
        for _ in range(repeats):
            results = solve_mechanics(
                n, L, phase, MATERIALS, EPS_BAR, formulation=formulation, scheme=scheme,
                toler_lin=TOLER_LIN, maxiter=MAXITER,
            )
            cast(ElasticitySolution, results[0].solution).eps.block_until_ready()
        run_ms = (time.perf_counter() - t0) * 1000.0 / repeats

        out[f"{label}_compile_ms"] = compile_ms
        out[f"{label}_run_ms"] = run_ms
        out[f"{label}_converged"] = converged

    return out


if __name__ == "__main__":
    results = []
    for N in GRID_SIZES:
        r = bench(N)
        results.append(r)
        print(f"n={N:>4}  Vf={r['volume_fraction']:.3f}  "
              f"ls: converged={r['ls_converged']}  compile={r['ls_compile_ms']:8.1f}ms  run={r['ls_run_ms']:8.1f}ms  |  "
              f"disp: converged={r['disp_converged']}  compile={r['disp_compile_ms']:8.1f}ms  run={r['disp_run_ms']:8.1f}ms")

    payload = {
        "generated": datetime.date.today().isoformat(),
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "ndim": 3,
        "material": {
            "matrix": {"name": MATRIX.name, "E": MATRIX.E, "nu": MATRIX.nu},
            "fiber": {"name": FIBER.name, "E": FIBER.E, "nu": FIBER.nu},
            "volume_fraction_target": PHI,
            "geometry": "quad_rve (square-packed 2-fibre)",
        },
        "solver": {
            "toler_lin": TOLER_LIN,
            "maxiter": MAXITER,
            "eps_bar": EPS_BAR.tolist(),
            "formulations": {label: formulation for label, formulation, _ in SOLVER_FORMULATIONS},
        },
        "results": results,
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {len(results)} results to {OUT_PATH}")
