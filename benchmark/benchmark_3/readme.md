# TexGen VTU Elastic Solve Benchmark

`elastic_solve.py` runs `problems.mechanics.solve_mechanics` on every `*.vtu` file in a given
directory -- TexGen exports of the same geometry family (e.g. different fabric weights or export
resolutions), each carrying its own per-voxel fiber orientation field (`YarnTangent` or
`Orientation`, depending on TexGen export version) read via `utils.io.reader.SimulationReader`.
The yarn phase is `TransverseIsotropic` with
`fiber_dir="from_input"`, so the orientation field drives the material automatically -- no
per-file material config needed.

No `.vtu` data is bundled here (unlike `benchmark_2/assets/`) -- TexGen output is generated
externally and is typically too large/site-specific to commit. Point `--data-dir` at your own
export directory.

```bash
python benchmark/benchmark_3/elastic_solve.py --data-dir /path/to/texgen/exports
```

Every solver config (`ls_rotated`, `ls_standard`, `displacement`, `galerkin`) automatically
domain-decomposes across `jax.local_device_count()` devices -- genuinely memory-scaling: `C_field`,
`G` and every CG state vector are built and kept as x-slab shards, never gathered onto one device.
On a single-device machine it's exactly the original single-device solve (verified in
`test/test_problems_mechanics_distributed.py`); each row's `n_devices_pmap` field shows how many
devices it used. Device memory columns report device 0 (each device holds ~1/`n_devices`).

Tracks wall-clock read/jit/solve/write time, peak memory (host RSS via `resource`, plus JAX device
memory via `jax.devices()[0].memory_stats()` on GPU/TPU), and the homogenized modulus per file,
prints a summary table, and writes `output/benchmark/benchmark_3/results_<date>.json`.

GPU memory columns (`n/a` on CPU/TPU). A small einsum/FFT warm-up runs before any problem data is
on the device, so the fixed CUDA/cuBLAS/cuFFT runtime cost (~130 MB) lands in `baseline_device_mb`
instead of being charged to the first stage:

- `C asm [MB]` (`c_assemble_device_mb`) / `solve [MB]` (`solve_device_mb`): each stage's own peak
  on top of what was already live when it started. `solve` is XLA compile + the first solve. A
  `<=` prefix (`*_exact: false` in the JSON) means the stage stayed under an earlier peak, so only
  an upper bound is known -- `peak_bytes_in_use` is a running maximum.
- `problem[MB]` (`problem_device_mb`): everything the first `solve_mechanics` call needs for this
  grid, i.e. its peak minus the baseline -- the number to report.
- `B/voxel` (`bytes_per_voxel`) and `max Nv` (`max_voxels_at_limit`): `problem_device_mb / Nv`,
  and the linear extrapolation of how many voxels fit under the allocator limit
  (`device_limit_mb`, i.e. `XLA_PYTHON_CLIENT_MEM_FRACTION` x device total).

On OOM, the printed failure shows the per-stage snapshots plus XLA's own "trying to allocate N
bytes" line -- in use at the crash plus that request is what the run actually needed. The full
per-stage breakdown is in each result's `mem_mb` dict in the JSON.

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

Every `(file, solver)` pair runs in its own subprocess, so `host_peak_rss_mb` is always a true
per-run peak (via `resource`) rather than a cumulative process-wide one, and JAX's GPU allocator
arena starts clean for every run instead of getting fragmented by earlier, differently-shaped
solves.

Edit `MATERIALS_CFG`/`EPS_BAR` at the top of `elastic_solve.py` for your actual fiber/matrix
constants and load case -- the defaults are illustrative E-glass/epoxy values.
