# :zap: Benchmark

## Frequency grid construction

Scaling of `operators.green.set_freq` — the frequency-grid construction used by every FFT-based
solver — comparing plain NumPy, eager JAX, and JIT-compiled JAX across grid sizes from 8³ to 160³.

<div class="benchmark-block" data-src="../data/benchmark_set_freq.json" data-x="elements" data-x-label="grid elements (n³)" data-baseline="numpy_ms">
  <div class="benchmark-controls">
    <label>
      Metric
      <select class="benchmark-metric">
        <option value="time">Time [ms]</option>
        <option value="speedup">Speedup vs NumPy</option>
      </select>
    </label>

    <label class="benchmark-series">
      <input class="benchmark-series-input" type="checkbox" value="numpy_ms" data-label="NumPy" data-color="#8a8a8a" checked> NumPy
    </label>
    <label class="benchmark-series">
      <input class="benchmark-series-input" type="checkbox" value="jax_eager_ms" data-label="JAX eager" data-color="#ef6c00" checked> JAX eager
    </label>
    <label class="benchmark-series">
      <input class="benchmark-series-input" type="checkbox" value="jax_jit_run_ms" data-label="JAX JIT" data-color="#7c4dff" checked> JAX JIT
    </label>

    <button class="benchmark-view-toggle" title="Toggle chart/table view">▤</button>
  </div>

  <canvas class="benchmark-canvas" height="120"></canvas>
  <table class="benchmark-table"></table>

  <div class="benchmark-meta"></div>
</div>

!!! note "How to read it"
    - **JAX eager** re-traces the function on every call, so at small grid sizes it's dominated by
      Python/tracing overhead rather than FFT work — this is expected and why **JAX JIT** (traced once,
      then replayed) is the number that matters for repeated solves.
    - Toggle series on/off, switch between absolute time and speedup-vs-NumPy, or hit **▤** for a
      plain table. The chart follows the page's light/dark theme.

```bash
python benchmark/set_freq/benchmark_set_freq_sweep.py
```

## Linear-elastic strain solve

The strain-based Newton-CG solver ([`solvers.mechanical.strain_nw_cg.solve_elastic`](https://github.com/choROPeNt/FFTjax))
on a simple two-phase composite: a centred spherical steel inclusion (E=210 GPa, ν=0.3, Vf≈15%) in an
aluminium matrix (E=70 GPa, ν=0.33), under a prescribed uniaxial macroscopic strain. Unlike a
homogeneous material — where the Newton-CG correction is exactly zero — this contrast makes the
solver actually iterate, so **Compile** (first call: trace + XLA compile + run) and **Run**
(steady-state, already compiled) reflect real solve cost.

<div class="benchmark-block" data-src="../data/benchmark_lin_elastic_strain.json" data-x="elements" data-x-label="grid elements (n³)" data-extra-columns="cg_iterations:CG iters,volume_fraction:Vf">
  <div class="benchmark-controls">
    <label class="benchmark-series">
      <input class="benchmark-series-input" type="checkbox" value="compile_ms" data-label="Compile (first call)" data-color="#ef6c00" checked> Compile
    </label>
    <label class="benchmark-series">
      <input class="benchmark-series-input" type="checkbox" value="run_ms" data-label="Run (steady-state)" data-color="#7c4dff" checked> Run
    </label>

    <button class="benchmark-view-toggle" title="Toggle chart/table view">▤</button>
  </div>

  <canvas class="benchmark-canvas" height="120"></canvas>
  <table class="benchmark-table"></table>

  <div class="benchmark-meta"></div>
</div>

!!! note "How to read it"
    - CG iterations stay roughly flat (26-42) across grid sizes for this fixed contrast/volume
      fraction — the cost growth you see is per-iteration FFT cost scaling with grid size, not more
      iterations. See the table view (**▤**) for the exact iteration count per grid size.
    - This is CPU-only data on the machine that generated it; a GPU run would shift **Run** down
      substantially without changing iteration counts.

```bash
python benchmark/lin_elastic_strain/benchmark_lin_elastic_strain.py
```

## Reproducing

Both scripts overwrite their `docs/data/*.json` file with fresh numbers for your machine
(backend/device are recorded automatically via `jax.default_backend()` / `jax.devices()`).
