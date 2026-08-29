"""
Multi-device scaling benchmark for jax.pmap(jax.vmap(solve_mechanics)),
companion to benchmark_vmap_batch_scaling.py (single-device batch-size
scaling) -- same fixed RVE/materials/load path, but here B is split across
n_devices physical devices (one pmap lane each, vmapped internally over its
B/n_devices share) instead of all landing on one device.

solve_mechanics needs no vmap/pmap-safe rewrite the way problems.fracture's
staggered PFF solve does: its CG solve (jax.scipy.sparse.linalg.cg) is
already built on jax.lax.while_loop, a real JAX control-flow primitive that
early-exits correctly per-lane under vmap/pmap -- unlike a Python-level
`if converged: break`, which cannot vary per batch lane inside a trace (see
problems.fracture.solve_fracture_fixed's docstring for why that solver
needed a dedicated, always-run-to-maxiter variant instead). So this is a
direct extension of the existing vmap benchmark, not a new solver path.

For each (B, n_devices) pair -- B divisible by every n_devices tested, so
the batch always splits evenly:
  - compile_ms         : first call to jax.pmap(jax.vmap(solve_one)) at this
                          (B, n_devices) shape (new shape -> fresh XLA trace,
                          once per device)
  - pmap_run_ms         : mean (+ std) over REPEATS calls to the
                          already-compiled pmap'd batch, using exactly
                          n_devices physical devices (jax.pmap(...,
                          devices=jax.local_devices()[:n_devices])) so the
                          sweep is genuinely 1-device vs 2-device vs
                          n-device on the *same* node, not just "however
                          many devices happen to be visible"
  - pmap_run_ms_per_path: pmap_run_ms / B -- the number to compare directly
                          against benchmark_vmap_batch_scaling.py's
                          vmap_run_ms_per_path at the same B, to see whether
                          spreading across devices actually beats batching
                          everything onto one

n_devices=1 is included deliberately as a sanity check: it should reproduce
single-device jax.vmap's per-path cost closely (same work, just wrapped in
an extra pmap layer with axis size 1), confirming pmap itself isn't adding
meaningful overhead before trusting the n_devices>1 numbers.

I have no multi-GPU node to run this against -- written and eyeballed for
correctness on whatever jax.local_device_count() reports locally (likely 1,
i.e. CPU), not benchmarked for real scaling. Treat the numbers from your
first real run as the actual answer, not this script's assumptions.

Prints a summary table and writes benchmark/pmap_device_scaling/results.json.
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

# Divisible by every candidate device count below (1, 2, 4, 8) so every
# (B, n_devices) pair in the sweep splits evenly -- no ragged remainder to
# special-case.
BATCH_SIZES = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
MAX_DEVICES_TO_TEST = 4   # sweep n_devices = 1, 2, 4 (capped at whatever's
                          # actually available) -- 4 to match a 4-GPU node;
                          # bump if you have more
REPEATS = 5
OUT_DIR = "output/benchmark/pmap_device_scaling"

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
    assert out is not None, "time_ms requires repeats >= 1"
    return float(np.mean(samples)), float(np.std(samples)), out


def bench(B, n_devices, devices, solve_one):
    per_device = B // n_devices
    eps_bar_batch = jnp.stack([EPS_SHEAR] * B).reshape(n_devices, per_device, 3, 3)

    solve_pmap = jax.pmap(jax.vmap(solve_one), devices=devices)

    # compile: first call at this (B, n_devices) shape, on these devices
    t0 = time.perf_counter()
    out = solve_pmap(eps_bar_batch)
    jax.block_until_ready(out)
    compile_ms = (time.perf_counter() - t0) * 1000.0

    # pmap run: already-compiled, split across n_devices physical devices
    pmap_ms, pmap_std, out = time_ms(
        lambda: solve_pmap(eps_bar_batch), REPEATS, jax.block_until_ready,
    )

    # solve_mechanics always returns list[IncrementResult] (one element
    # here, stepping="single") -- [0].solution is the (n_devices,
    # per_device, ...)-shaped ElasticitySolution.
    sol = cast(ElasticitySolution, out[0].solution)
    converged = bool(jnp.all(sol.converged))
    tau_xy = float(jnp.mean(sol.sigma[:, :, 1, 0]))

    return {
        "B": B,
        "n_devices": n_devices,
        "per_device": per_device,
        "converged": converged,
        "tau_xy_pmap": tau_xy,
        "compile_ms": compile_ms,
        "pmap_run_ms": pmap_ms,
        "pmap_run_ms_std": pmap_std,
        "pmap_run_ms_per_path": pmap_ms / B,
    }


if __name__ == "__main__":
    phase_np, n, L, phi_act = make_square_composite_rve(
        phi=0.5, r_fiber=0.005, dx=0.0002, N_min=32, nz=1,
    )
    phase = jnp.array(phase_np.reshape(-1))

    matrix = LinearElasticIsotropic(E=3.0e3,  nu=0.35, name="epoxy matrix")
    fiber  = LinearElasticIsotropic(E=70.0e3, nu=0.20, name="glass fiber")
    materials = [matrix, fiber]

    def solve_one(eps_bar):
        return solve_mechanics(
            n, L, phase, materials, eps_bar,
            scheme="rotated", toler_lin=TOLER_LIN, maxiter=MAXITER,
        )

    available = jax.local_device_count()
    n_devices_sweep = [d for d in (1, 2, 4, 8) if d <= min(MAX_DEVICES_TO_TEST, available)]
    print(f"grid n={n}  Nv={int(np.prod(n))}  fiber Vf={phi_act:.4f}")
    print(f"backend={jax.default_backend()}  local devices available={available}  "
          f"testing n_devices={n_devices_sweep}")
    if available < 2:
        print("NOTE: only one local device visible -- this run can't show real multi-device "
              "scaling, only that the pmap(n_devices=1) path itself works.")
    print(f"{'B':>5}  {'n_dev':>5}  {'compile_ms':>11}  {'pmap_ms':>10}  "
          f"{'pmap/path':>10}  {'converged':>9}")

    results = []
    for n_devices in n_devices_sweep:
        devices = jax.local_devices()[:n_devices]
        for B in BATCH_SIZES:
            if B % n_devices != 0:
                continue
            r = bench(B, n_devices, devices, solve_one)
            results.append(r)
            print(f"{B:>5}  {n_devices:>5}  {r['compile_ms']:>11.1f}  {r['pmap_run_ms']:>10.1f}  "
                  f"{r['pmap_run_ms_per_path']:>10.2f}  {str(r['converged']):>9}")

    today = datetime.date.today().isoformat()
    payload = {
        "generated": today,
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "local_device_count": available,
        "n_devices_tested": n_devices_sweep,
        "grid_n": list(n),
        "fiber_volume_fraction": phi_act,
        "repeats": REPEATS,
        "load_path": "in-plane shear, replicated B times per batch, split across n_devices",
        "eps_bar": EPS_SHEAR.tolist(),
        "solver": {"toler_lin": TOLER_LIN, "maxiter": MAXITER, "scheme": "rotated"},
        "results": results,
    }

    out_path = os.path.join(OUT_DIR, f"results_{today}.json")
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {len(results)} results to {out_path}")
