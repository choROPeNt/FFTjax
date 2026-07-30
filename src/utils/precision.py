"""
JAX precision and memory configuration, applied once at import time.

- Enables float64 (X64) on CPU/GPU, where this project's CG solvers assume
  double precision. Leaves JAX at its default float32 on TPU, since TPU has
  no native float64 support (double precision is emulated there and
  impractically slow -- see docs/docs/installation.mdx).
- Disables XLA's default GPU memory preallocation
  (XLA_PYTHON_CLIENT_PREALLOCATE), so JAX grows GPU memory usage as needed
  instead of grabbing ~75% of it upfront on first use.

Import this before creating any JAX arrays -- as the very first import in
any entry-point module, before `import jax`:

    import utils.precision  # noqa: F401 -- side effect: configures JAX

An explicit JAX_ENABLE_X64 in the environment is respected as-is (no
auto-detection override), so TPU users who want float64 despite the
performance cost can still force it.
"""
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax

if "JAX_ENABLE_X64" in os.environ:
    X64_ENABLED = os.environ["JAX_ENABLE_X64"].lower() in ("1", "true")
else:
    X64_ENABLED = jax.default_backend() != "tpu"
    jax.config.update("jax_enable_x64", X64_ENABLED)
