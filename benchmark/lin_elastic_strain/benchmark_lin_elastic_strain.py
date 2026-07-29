"""
Sweeps the strain-based Newton-CG elastic solver (solvers.mechanical.strain_nw_cg
.solve_elastic) over grid size, for a simple two-phase composite: a spherical
steel inclusion (E=210e3, nu=0.3) in an aluminium matrix (E=70e3, nu=0.33),
Vf~15%, under a prescribed uniaxial macroscopic strain.

Writes results to docs/data/benchmark_lin_elastic_strain.json for the
interactive benchmark page (docs/benchmark.md).
"""
import sys
sys.path.insert(0, "src")
import os
os.environ["JAX_ENABLE_X64"] = "1"

import json
import time
import datetime

import jax
import jax.numpy as jnp
import numpy as np

from mat_models.elastic import LinearElasticIsotropic, assemble_C_field
from operators.green import build_freq_grid, build_green_operator
from solvers.mechanical.strain_nw_cg import solve_elastic

GRID_SIZES = [16, 24, 32, 48, 64, 96]
L = (1.0, 1.0, 1.0)
VF_TARGET = 0.15
TOLER_LIN = 1e-6
MAXITER = 1000
OUT_PATH = "docs/data/benchmark_lin_elastic_strain.json"

MATRIX = LinearElasticIsotropic(E=70e3, nu=0.33, name="aluminium")
INCLUSION = LinearElasticIsotropic(E=210e3, nu=0.3, name="steel")
EPS_BAR = jnp.array([
    [1.0e-3, 0.0, 0.0],
    [0.0,    0.0, 0.0],
    [0.0,    0.0, 0.0],
])


def sphere_phase(n, L, vf):
    """Centred spherical inclusion, phase=1 inside, 0 in the matrix."""
    dx = tuple(Li / ni for Li, ni in zip(L, n))
    idx = np.indices(n)
    centers = np.stack([(idx[i] + 0.5) * dx[i] for i in range(3)], axis=0).reshape(3, -1)
    c0 = np.array([Li / 2 for Li in L])
    r = (3.0 * vf / (4.0 * np.pi)) ** (1.0 / 3.0)
    dist = np.linalg.norm(centers - c0[:, None], axis=0)
    return (dist <= r).astype(int)


def bench(n, repeats=3):
    Nv = int(np.prod(n))
    phase = sphere_phase(n, L, VF_TARGET)
    vf_actual = float(phase.mean())

    C_field = assemble_C_field([MATRIX, INCLUSION], jnp.array(phase))
    xi_flat = build_freq_grid(n, L)
    lam0 = 0.5 * (MATRIX.lam + INCLUSION.lam)
    mu0 = 0.5 * (MATRIX.mu + INCLUSION.mu)
    G_glob = build_green_operator(xi_flat, lam0, mu0)

    # first call: trace + compile + run
    t0 = time.perf_counter()
    eps, sigma, delta, it, conv = solve_elastic(
        n, C_field, G_glob, EPS_BAR, toler_lin=TOLER_LIN, maxiter=MAXITER
    )
    eps.block_until_ready()
    compile_ms = (time.perf_counter() - t0) * 1000.0

    # steady-state: already compiled
    t0 = time.perf_counter()
    for _ in range(repeats):
        eps, sigma, delta, it, conv = solve_elastic(
            n, C_field, G_glob, EPS_BAR, toler_lin=TOLER_LIN, maxiter=MAXITER
        )
        eps.block_until_ready()
    run_ms = (time.perf_counter() - t0) * 1000.0 / repeats

    return {
        "n": list(n),
        "elements": Nv,
        "volume_fraction": vf_actual,
        "cg_iterations": int(it),
        "converged": bool(conv),
        "compile_ms": compile_ms,
        "run_ms": run_ms,
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
    }


if __name__ == "__main__":
    results = []
    for n in GRID_SIZES:
        r = bench((n, n, n))
        results.append(r)
        print(f"n={n:>4}  Vf={r['volume_fraction']:.3f}  "
              f"iters={r['cg_iterations']:>3}  converged={r['converged']}  "
              f"compile={r['compile_ms']:8.1f}ms  run={r['run_ms']:8.1f}ms")

    payload = {
        "generated": datetime.date.today().isoformat(),
        "backend": results[0]["backend"],
        "device": results[0]["device"],
        "ndim": 3,
        "material": {
            "matrix": {"name": MATRIX.name, "E": MATRIX.E, "nu": MATRIX.nu},
            "inclusion": {"name": INCLUSION.name, "E": INCLUSION.E, "nu": INCLUSION.nu},
            "volume_fraction_target": VF_TARGET,
            "geometry": "centred sphere",
        },
        "solver": {
            "toler_lin": TOLER_LIN,
            "maxiter": MAXITER,
            "eps_bar": EPS_BAR.tolist(),
        },
        "results": results,
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {len(results)} results to {OUT_PATH}")
