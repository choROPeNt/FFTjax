# TexGen VTU Elastic Solve Benchmark

`elastic_solve_vtu.py` runs `problems.mechanics.solve_mechanics` on every `*.vtu` file in a given
directory -- TexGen exports of the same geometry family (e.g. different fabric weights or export
resolutions), each carrying its own per-voxel fiber orientation field (`YarnTangent`) read via
`utils.io.reader.SimulationReader`. The yarn phase is `TransverseIsotropic` with
`fiber_dir="from_input"`, so the orientation field drives the material automatically -- no
per-file material config needed.

No `.vtu` data is bundled here (unlike `benchmark_2/assets/`) -- TexGen output is generated
externally and is typically too large/site-specific to commit. Point `--data-dir` at your own
export directory.

```bash
python benchmark/benchmark_3/elastic_solve_vtu.py --data-dir /path/to/texgen/exports
python benchmark/benchmark_3/elastic_solve_vtu.py --data-dir /path/to/texgen/exports --isolate
```

Tracks wall-clock read/jit/solve/write time, peak memory (host RSS via `resource`, plus JAX device
memory via `jax.devices()[0].memory_stats()` on GPU/TPU), and the homogenized modulus per file,
prints a summary table, and writes `output/benchmark/benchmark_3/results_<date>.json`.

`jit_time_s` vs. `solve_time_s`: `solve_mechanics` is called twice per file with identical inputs
-- the first call's time includes XLA trace + compile (`lax.while_loop` inside the CG solve always
compiles to XLA on first use for a given grid shape, even with no explicit `jax.jit` on
`solve_mechanics` itself), the second hits the now-warm compilation cache, giving a steady-state
solve time with the one-off compile cost no longer in it.

`E_eff_MPa` is the homogenized modulus at `LOADED_COMPONENT` (default `(0, 0)`, matching
`EPS_BAR`'s loaded direction), computed from the actual volume-averaged response
(`post.fields.homogenize`) rather than just the prescribed load -- no second solve needed. This is
the constrained (pure-strain-BC) modulus, not a free-surface engineering modulus; the latter (with
transverse Poisson's ratios) is `learning.extractors.effective_modulus`, which needs its own
separate mixed-BC solve and isn't reusable here.

Each file's solved fields -- `phase`, `yarn_index`, `orientation`, `strain`/`stress` (Voigt),
`von_mises`, `displacement` -- are also written to XDMF/HDF5 via
`utils.io.xdmf_writer.IncrementalWriter` -- this project's standard field-data output -- as
`output/benchmark/benchmark_3/<stem>.h5`/`.xdmf`, openable in ParaView with the `Xdmf3ReaderT`
reader.

`--isolate` re-runs the script once per file, each its own subprocess, for a true per-file
peak-memory reading -- without it, `host_peak_rss_mb` is the process's peak-so-far and only ever
grows across files, so later files' numbers include earlier ones' already-freed peaks.

Edit `MATERIALS_CFG`/`EPS_BAR` at the top of `elastic_solve_vtu.py` for your actual fiber/matrix
constants and load case -- the defaults are illustrative E-glass/epoxy values.
