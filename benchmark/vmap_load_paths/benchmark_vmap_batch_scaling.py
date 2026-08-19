"""
Batch-size scaling benchmark for jax.vmap(solve_mechanics), companion to
notebooks/lin-elastic_strain_vmap.ipynb.

Every load path in the batch is the *same* macroscopic strain (in-plane
shear) replicated B times -- deliberately, so B is the only thing that
changes between runs. This isolates how solve time scales with batch size
alone, without also varying load-path diversity (which would confound "cost
of a bigger batch" with "cost of a harder problem").

Grid size, materials, and RVE geometry are fixed (same composite as
lin-elastic_strain.ipynb) -- only B varies.

For each B, three things are measured separately:
  - compile_ms   : first call to jax.jit(jax.vmap(solve_one)) at this B
                    (new B -> new input shape -> forces a fresh XLA trace)
  - vmap_run_ms  : mean (+ std) over REPEATS calls to the already-compiled
                   batched solve
  - loop_run_ms  : mean (+ std) over REPEATS calls to a Python loop of B
                   calls to a *singly*-jitted solve_one (compiled once,
                   outside the loop) -- the "just cache the compiled
                   single-example solve and call it B times" alternative,
                   for comparison against vmap.

Prints a summary table and writes benchmark/vmap_load_paths/results.json.
"""
import sys
sys.path.insert(0, "src")

import json
import os
import time
import datetime

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)
import jax
import jax.numpy as jnp
import numpy as np

from generation.rve import make_square_composite_rve
from materialmodels.elastic.isotropic import LinearElasticIsotropic
from problems.mechanics import solve_mechanics

BATCH_SIZES = [1, 2, 4, 8, 16, 32]
REPEATS     = 5
OUT_DIR     = "output/benchmark/vmap_load_paths"

RVE_KW = dict(phi=0.5, r_fiber=0.005, dx=0.0002, N_min=32, nz=1)
EPS_SHEAR = jnp.array([
    [0.0,    1.0e-3, 0.0],
    [1.0e-3, 0.0,    0.0],
    [0.0,    0.0,    0.0],
])
TOLER_LIN = 1e-6
MAXITER   = 1000


def time_ms(fn, repeats, sync):
    """Mean/std wall-clock time over `repeats` calls, in ms."""
    samples = []
    out = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        sync(out)
        samples.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(samples)), float(np.std(samples)), out


def bench(B, solve_one, solve_one_jit, n, L, phase, materials):
    eps_bar_batch = jnp.stack([EPS_SHEAR] * B)

    solve_batch = jax.jit(jax.vmap(solve_one))

    # compile: first call at this batch shape
    t0 = time.perf_counter()
    out = solve_batch(eps_bar_batch)
    jax.block_until_ready(out)
    compile_ms = (time.perf_counter() - t0) * 1000.0

    # vmap run: batched, already-compiled
    vmap_ms, vmap_std, out = time_ms(
        lambda: solve_batch(eps_bar_batch), REPEATS, jax.block_until_ready,
    )

    # loop baseline: single-example solve, jitted once, called B times
    def run_loop():
        return [solve_one_jit(EPS_SHEAR) for _ in range(B)]

    loop_ms, loop_std, loop_out = time_ms(
        run_loop, REPEATS, lambda outs: jax.block_until_ready(outs),
    )

    converged = bool(jnp.all(out.converged))
    tau_xy_batch = float(jnp.mean(out.sigma[:, 1, 0]))
    tau_xy_loop  = float(jnp.mean(loop_out[0].sigma[1, 0]))
    max_abs_diff = float(jnp.max(jnp.abs(out.sigma - jnp.stack([r.sigma for r in loop_out]))))

    return {
        "B": B,
        "converged": converged,
        "tau_xy_vmap": tau_xy_batch,
        "tau_xy_loop": tau_xy_loop,
        "max_abs_diff": max_abs_diff,
        "compile_ms": compile_ms,
        "vmap_run_ms": vmap_ms,
        "vmap_run_ms_std": vmap_std,
        "vmap_run_ms_per_path": vmap_ms / B,
        "loop_run_ms": loop_ms,
        "loop_run_ms_std": loop_std,
        "loop_run_ms_per_path": loop_ms / B,
    }


if __name__ == "__main__":
    phase_np, n, L, phi_act = make_square_composite_rve(**RVE_KW)
    phase = jnp.array(phase_np.reshape(-1))

    matrix = LinearElasticIsotropic(E=3.0e3,  nu=0.35, name="epoxy matrix")
    fiber  = LinearElasticIsotropic(E=70.0e3, nu=0.20, name="glass fiber")
    materials = [matrix, fiber]

    def solve_one(eps_bar):
        return solve_mechanics(
            n, L, phase, materials, eps_bar,
            scheme="rotated", toler_lin=TOLER_LIN, maxiter=MAXITER,
        )

    solve_one_jit = jax.jit(solve_one)
    solve_one_jit(EPS_SHEAR)  # warm up / compile once, outside all timed sections

    print(f"grid n={n}  Nv={int(np.prod(n))}  fiber Vf={phi_act:.4f}")
    print(f"{'B':>4}  {'compile_ms':>11}  {'vmap_ms':>10}  {'vmap/path':>10}  "
          f"{'loop_ms':>10}  {'loop/path':>10}  {'converged':>9}")

    results = []
    for B in BATCH_SIZES:
        r = bench(B, solve_one, solve_one_jit, n, L, phase, materials)
        results.append(r)
        print(f"{B:>4}  {r['compile_ms']:>11.1f}  {r['vmap_run_ms']:>10.1f}  "
              f"{r['vmap_run_ms_per_path']:>10.2f}  {r['loop_run_ms']:>10.1f}  "
              f"{r['loop_run_ms_per_path']:>10.2f}  {str(r['converged']):>9}")

    today = datetime.date.today().isoformat()
    payload = {
        "generated": today,
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "grid_n": list(n),
        "fiber_volume_fraction": phi_act,
        "repeats": REPEATS,
        "load_path": "in-plane shear, replicated B times per batch",
        "eps_bar": EPS_SHEAR.tolist(),
        "solver": {"toler_lin": TOLER_LIN, "maxiter": MAXITER, "scheme": "rotated"},
        "results": results,
    }

    out_path = os.path.join(OUT_DIR, f"results_{today}.json")
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {len(results)} results to {out_path}")
