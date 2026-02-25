import sys
sys.path.append("src")
import os
os.environ["JAX_ENABLE_X64"] = "1"




import jax
import jax.numpy as jnp
import numpy as np
import time

from operators.green import set_freq_np, set_freq_jax, set_freq_jax_jit


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
def pretty_print_bench(results):
    print("\n🚀 FFTjax Benchmark")
    print("=" * 82)
    print(f"{'ndim':>4} | {'grid':>15} | {'NumPy [ms]':>12} | {'JAX eager [ms]':>14} | {'JAX JIT [ms]':>12} | {'speedup':>8}")
    print("-" * 82)
    
    for r in results:
        speedup = r["numpy_ms"] / r["jax_jit_run_ms"]
        print(r["max_abs_diff"])

        grid = "×".join(map(str, r["n"]))
        print(
            f"{r['ndim']:>4} | "
            f"{grid:>15} | "
            f"{r['numpy_ms']:12.3f} | "
            f"{r['jax_eager_ms']:14.3f} | "
            f"{r['jax_jit_run_ms']:12.3f} | "
            f"{speedup:8.2f}x"
        )

    print("-" * 82)
    print(f"Backend: {results[0]['backend']}  |  Device: {results[0]['device']}")
    print("=" * 82)

if __name__ == "__main__":
    # Example usage
    results = bench(ndim=3, n=(64, 64, 64), L=(1.0, 1.0, 1.0))
    pretty_print_bench([results])