"""
Sweeps the frequency-grid construction (operators.green.set_freq_*) over a
range of grid sizes, comparing plain NumPy, eager JAX, and JIT-compiled JAX.

Writes results to docs/static/data/benchmark_set_freq.json for the interactive
Benchmark page (docs/docs/documentation/benchmark.mdx).
"""
import sys
sys.path.insert(0, "src")

import json
import os
import time
import datetime

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)
import jax
import numpy as np

from operators.green import set_freq_np, set_freq_jax, set_freq_jax_jit

GRID_SIZES = [8, 16, 24, 32, 48, 64, 96, 128, 160]
L = (1.0, 1.0, 1.0)
OUT_PATH = "docs/static/data/benchmark_set_freq.json"


def time_ms(fn, repeats=30, sync=None):
    t0 = time.perf_counter()
    out = None
    for _ in range(repeats):
        out = fn()
        if sync is not None:
            sync(out)
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / repeats, out


def bench(ndim, n, L, repeats_np=30, repeats_jax=30):
    np_ms, out_np = time_ms(lambda: set_freq_np(ndim, n, L), repeats=repeats_np)

    jax_ms, _ = time_ms(
        lambda: set_freq_jax(ndim, n, L),
        repeats=repeats_jax,
        sync=lambda x: x.block_until_ready(),
    )

    # compile
    t0 = time.perf_counter()
    out_jit = set_freq_jax_jit(ndim, n, L)
    out_jit.block_until_ready()
    jit_compile_ms = (time.perf_counter() - t0) * 1000.0

    # run
    jax_jit_ms, out_jit = time_ms(
        lambda: set_freq_jax_jit(ndim, n, L),
        repeats=repeats_jax,
        sync=lambda x: x.block_until_ready(),
    )

    max_abs = float(np.max(np.abs(np.array(out_jit) - out_np)))

    return {
        "ndim": ndim,
        "n": tuple(n),
        "numpy_ms": np_ms,
        "jax_eager_ms": jax_ms,
        "jax_jit_compile_ms": jit_compile_ms,
        "jax_jit_run_ms": jax_jit_ms,
        "max_abs_diff": max_abs,
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
    }


if __name__ == "__main__":
    results = []
    for n in GRID_SIZES:
        r = bench(ndim=3, n=(n, n, n), L=L)
        r["n"] = list(r["n"])
        r["elements"] = n ** 3
        results.append(r)
        print(f"n={n:>4}  numpy={r['numpy_ms']:.3f}ms  "
              f"jax_eager={r['jax_eager_ms']:.3f}ms  "
              f"jax_jit={r['jax_jit_run_ms']:.3f}ms")

    payload = {
        "generated": datetime.date.today().isoformat(),
        "backend": results[0]["backend"],
        "device": results[0]["device"],
        "ndim": 3,
        "results": results,
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {len(results)} results to {OUT_PATH}")
