# 🚀 Getting Started

!!! warning "Not on PyPI yet"
    FFTjax is under active development and hasn't been published to PyPI. Install it directly from source in editable mode.

## 💿 Installation

=== ":simple-pypi: CPU"

    ```bash
    git clone https://github.com/choROPeNt/FFTjax.git
    cd FFTjax
    pip install -e .
    ```

=== ":simple-nvidia: GPU (CUDA)"

    ```bash
    git clone https://github.com/choROPeNt/FFTjax.git
    cd FFTjax
    pip install -e ".[cuda]"
    ```

=== ":material-chip: TPU"

    ```bash
    git clone https://github.com/choROPeNt/FFTjax.git
    cd FFTjax
    pip install -e ".[tpu]"
    ```

The `cuda`/`tpu` extras are declared in [`pyproject.toml`](https://github.com/choROPeNt/FFTjax/blob/main/pyproject.toml)
(`jax[cuda12]` / `jax[tpu]`) — one install command instead of installing the CPU build first and
upgrading it after.

Editable mode (`-e`) means local changes to `src/` are picked up immediately, without reinstalling — handy while the package is still moving fast.

## ✅ Verify

Check the package installed:

```bash
python -c "from importlib.metadata import version; print(version('fftjax'))"
```

Then run a full smoke test — builds a frequency grid, assembles an elastic stiffness field, and runs
one Newton-CG strain solve, so a pass means JAX, X64 precision, and the FFT solver machinery are all
actually working, not just importable:

```bash
python - <<'PY'
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from operators.green import build_freq_grid, build_green_operator
from mat_models.elastic import LinearElasticIsotropic, assemble_C_field
from solvers.mechanical.strain_nw_cg import solve_elastic

n, L = (8, 8, 8), (1.0, 1.0, 1.0)
mat = LinearElasticIsotropic(E=210e3, nu=0.3)
C_field = assemble_C_field([mat], jnp.zeros(8**3, dtype=int))
xi = build_freq_grid(n, L)
G = build_green_operator(xi, mat.lam, mat.mu)
eps_bar = jnp.array([[1e-3, 0, 0], [0, 0, 0], [0, 0, 0]])
eps, sigma, delta, it, conv = solve_elastic(n, C_field, G, eps_bar)

assert jnp.zeros(1).dtype == jnp.float64, "X64 precision not enabled"
assert bool(conv), "solver did not converge"
print("backend:", jax.default_backend(), "| devices:", jax.devices())
print("x64 enabled: True | solve converged in", int(it), "iterations")
print("FFTjax install OK")
PY
```

!!! tip "GPU/TPU install"
    On a GPU or TPU machine, `jax.devices()` above should list an accelerator, not `CpuDevice`. If it
    still shows CPU after installing the `cuda`/`tpu` extra, JAX likely fell back silently — check
    `pip show jaxlib` matches the accelerator build.

## Run the tests

```bash
python -m pytest test/
```
