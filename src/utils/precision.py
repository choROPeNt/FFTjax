"""
JAX precision and memory configuration, applied once at import time.

- Loads a repo-root ``.env`` file, if present, into the environment --
  ``os.environ.setdefault`` per line, so a real shell-exported variable
  always wins over the file (same precedence rule as JAX_ENABLE_X64 below).
  Gitignored, machine-specific (e.g. available GPU memory differs per box);
  see ``.env.example`` for the keys this project actually reads.
- Enables float64 (X64) on CPU/GPU, where this project's CG solvers assume
  double precision. Leaves JAX at its default float32 on TPU, since TPU has
  no native float64 support (double precision is emulated there and
  impractically slow -- see docs/docs/documentation/installation.mdx).
- Disables XLA's default GPU memory preallocation
  (XLA_PYTHON_CLIENT_PREALLOCATE), so JAX grows GPU memory usage as needed
  instead of grabbing ~75% of it upfront on first use. Set
  XLA_PYTHON_CLIENT_MEM_FRACTION (a 0-1 fraction of total device memory, not
  an absolute byte count -- JAX has no such env var) to cap how far it's
  allowed to grow, in ``.env`` or the shell.

Import this before creating any JAX arrays -- as the very first import in
any entry-point module, before `import jax`:

    import utils.precision  # noqa: F401 -- side effect: configures JAX

An explicit JAX_ENABLE_X64 in the environment is respected as-is (no
auto-detection override), so TPU users who want float64 despite the
performance cost can still force it.
"""
import os
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    """Minimal KEY=VALUE .env loader -- setdefault only, comments (#) and
    blank lines skipped, optional surrounding quotes stripped. No
    interpolation, no multi-line values -- not needed for the handful of
    XLA/JAX flags this file actually sets."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key:
            os.environ.setdefault(key, value)


_load_dotenv(Path(__file__).resolve().parents[2] / ".env")

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax

if "JAX_ENABLE_X64" in os.environ:
    X64_ENABLED = os.environ["JAX_ENABLE_X64"].lower() in ("1", "true")
else:
    X64_ENABLED = jax.default_backend() != "tpu"
    jax.config.update("jax_enable_x64", X64_ENABLED)
