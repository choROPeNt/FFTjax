"""
Sweeps operators.green.build_freq_grid -- the frequency-grid construction
used by every FFT-based solver in this project -- over a range of grid
sizes, comparing plain NumPy, eager JAX, and JIT-compiled JAX.

Writes results to docs/static/data/benchmark_set_freq.json for the interactive
Benchmark page (docs/docs/documentation/benchmark.mdx).

Grid sizes go up to 512 (134M voxels, ~3.2GB per (3, Nv) float64 array --
several such arrays live at once), which can exhaust memory on smaller
machines. Each size therefore runs in its own subprocess (--single below):
if the OS OOM-kills a subprocess (SIGKILL, uncatchable in-process) or it
raises MemoryError/XLA RESOURCE_EXHAUSTED, the parent sees a non-zero/negative
return code, stops the sweep there, and still writes out every size that
completed before it -- one grid size running out of memory doesn't lose the
results already collected.

Usage
-----
    python benchmark/benchmark_set_freq_sweep.py
    python benchmark/benchmark_set_freq_sweep.py --grid-sizes 16 32 64 --repeats 10 --out /tmp/out.json
"""
import sys
sys.path.insert(0, "src")

import argparse
import json
import os
import subprocess
import time
import datetime

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)
import jax
import numpy as np

from operators.green import build_freq_grid

GRID_SIZES = [8, 16, 24, 32, 48, 64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512]
L = (1.0, 1.0, 1.0)
REPEATS = 30
OUT_PATH = "docs/static/data/benchmark_set_freq.json"
OOM_EXIT_CODE = 137  # conventional 128 + SIGKILL(9), also used when we catch the error ourselves

# mark ALL shape-relevant args as static IMPORTANT for JIT compilation and performance
build_freq_grid_jit = jax.jit(build_freq_grid, static_argnames=("n", "L"))


## Plain-NumPy equivalent, for benchmarking -- same full-spectrum (fftn),
## real-valued-angular-frequency convention as build_freq_grid itself.
def set_freq_np(n, L):
    freqs = [np.fft.fftfreq(ni, d=Li / ni) * 2.0 * np.pi for ni, Li in zip(n, L)]
    grids = np.meshgrid(*freqs, indexing="ij")
    return np.stack([g.ravel() for g in grids])


def time_ms(fn, repeats=30, sync=None):
    t0 = time.perf_counter()
    out = None
    for _ in range(repeats):
        out = fn()
        if sync is not None:
            sync(out)
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / repeats, out


def bench(n, L, repeats=30):
    np_ms, out_np = time_ms(lambda: set_freq_np(n, L), repeats=repeats)

    jax_ms, _ = time_ms(
        lambda: build_freq_grid(n, L),
        repeats=repeats,
        sync=lambda x: x.block_until_ready(),
    )

    # compile
    t0 = time.perf_counter()
    out_jit = build_freq_grid_jit(n, L)
    out_jit.block_until_ready()
    jit_compile_ms = (time.perf_counter() - t0) * 1000.0

    # run
    jax_jit_ms, out_jit = time_ms(
        lambda: build_freq_grid_jit(n, L),
        repeats=repeats,
        sync=lambda x: x.block_until_ready(),
    )

    max_abs = float(np.max(np.abs(np.array(out_jit) - out_np)))

    return {
        "n": tuple(n),
        "numpy_ms": np_ms,
        "jax_eager_ms": jax_ms,
        "jax_jit_compile_ms": jit_compile_ms,
        "jax_jit_run_ms": jax_jit_ms,
        "max_abs_diff": max_abs,
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
    }


def _is_oom_error(exc: BaseException) -> bool:
    if isinstance(exc, MemoryError):
        return True
    msg = str(exc).upper()
    return "RESOURCE_EXHAUSTED" in msg or "OUT OF MEMORY" in msg


def _run_single(n: int) -> None:
    """
    Worker mode (``--single N``): run ``bench`` for exactly one grid size and
    print the result as one JSON line on stdout. Run in its own subprocess by
    ``main`` so a crash/OOM here only ends this one grid size, not the whole
    sweep. Reads module globals directly (L/REPEATS) -- __main__ has already
    applied any CLI overrides to them in this same (sub)process.
    """
    try:
        r = bench(n=(n, n, n), L=L, repeats=REPEATS)
    except Exception as exc:
        if _is_oom_error(exc):
            print(f"n={n}: out of memory ({exc})", file=sys.stderr)
            sys.exit(OOM_EXIT_CODE)
        raise
    r["n"] = list(r["n"])
    r["elements"] = n ** 3
    print(json.dumps(r))


def main(forward_args: list[str]) -> None:
    """``forward_args``: the CLI flags (minus --single) to re-pass to each
    worker subprocess, so it sees the same overrides this process resolved."""
    results = []
    for n in GRID_SIZES:
        proc = subprocess.run(
            [sys.executable, __file__, "--single", str(n), *forward_args],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            if proc.returncode < 0:
                reason = f"killed by signal {-proc.returncode} (likely OS OOM-kill)"
            elif proc.returncode == OOM_EXIT_CODE:
                reason = "out of memory"
            else:
                reason = f"exited with code {proc.returncode}"
            print(f"\nn={n:>4}: {reason} -- stopping sweep, keeping "
                  f"{len(results)} completed size(s)")
            if proc.stderr.strip():
                print(proc.stderr.strip().splitlines()[-1])
            break

        r = json.loads(proc.stdout)
        results.append(r)
        print(f"n={n:>4}  numpy={r['numpy_ms']:.3f}ms  "
              f"jax_eager={r['jax_eager_ms']:.3f}ms  "
              f"jax_jit={r['jax_jit_run_ms']:.3f}ms")

    if not results:
        print("No grid size completed -- nothing written.")
        return

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-sizes", type=int, nargs="+", default=GRID_SIZES)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--out", default=OUT_PATH)
    parser.add_argument("--single", type=int, default=None, metavar="N",
                         help="internal: run one grid size in a worker subprocess")
    args = parser.parse_args()

    GRID_SIZES = args.grid_sizes
    REPEATS    = args.repeats
    OUT_PATH   = args.out

    if args.single is not None:
        _run_single(args.single)
    else:
        main(["--repeats", str(REPEATS), "--out", OUT_PATH])
