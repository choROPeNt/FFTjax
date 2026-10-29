"""
JAX precision and memory setup, applied on import. Import it first in every
entry-point module, before ``import jax``:

    import utils.precision  # noqa: F401

- loads a repo-root ``.env`` (shell variables win; see ``.env.example``)
- float64 on CPU/GPU (the CG solvers assume it), float32 on TPU (no native
  float64); an explicit JAX_ENABLE_X64 is respected
- no GPU memory preallocation; cap growth with XLA_PYTHON_CLIENT_MEM_FRACTION
  (a fraction of device memory, not bytes)
"""
import os
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    """Minimal KEY=VALUE loader: setdefault only, skips comments and blanks, strips quotes."""
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
