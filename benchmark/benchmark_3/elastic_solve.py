"""
Linear-elastic solve benchmark over a directory of TexGen-exported .vtu
microstructures -- same geometry family (e.g. different fabric weights or
TexGen export resolutions), looped over one at a time and, for each file,
over every (formulation, scheme) pair in SOLVER_CONFIGS -- Lippmann-
Schwinger with the rotated (GreenOperatorWillot) and standard
(GreenOperatorBasic) schemes, the displacement-based formulation (scheme
doesn't apply there, see solve_mechanics's own docstring), and the
reference-medium-free Fourier-Galerkin formulation (Vondrejc et al 2014;
see notes/FOURIER_GALERKIN.md).

Every formulation automatically domain-decomposes across
jax.local_device_count() devices -- genuinely memory-scaling: C_field, G
and every CG state vector are built and kept as x-slab shards, never
gathered onto one device (see materialmodels.assembly.assemble_C_field and
the *_sharded solvers). On a single-device machine it's exactly the
original single-device solve (see test/test_problems_mechanics_distributed.py).
Each row's ``n_devices_pmap`` field reports how many devices it used.
Device memory columns are device 0's (each device holds ~1/n_devices).

Tracks wall-clock read/jit/solve time and peak memory (host RSS, plus JAX
device memory on GPU/TPU) per (file, solver) pair. jit_time_s is
solve_mechanics's first call for that pair's grid shape (XLA trace + compile
+ run -- lax.while_loop inside the CG solve always compiles to XLA on first
use for a given shape, even with no explicit jax.jit on solve_mechanics
itself); solve_time_s is a second, identical call, which hits the now-warm
compilation cache -- steady-state solve time with jit overhead no longer in it.

Device memory (GPU/TPU only, None on CPU): warm_up_device runs before any
problem data is on the device, so the fixed runtime cost lands in
``baseline_device_mb`` instead of the first stage. ``c_assemble_device_mb``/
``solve_device_mb`` are each stage's own peak on top of what was live when
it started (see stage_peak_mb -- flagged ``*_exact: False`` when the stage
stayed under an earlier peak and only an upper bound is known).
``problem_device_mb`` is the first solve_mechanics call's peak minus that
baseline -- the grid's real device requirement -- with ``bytes_per_voxel``
and ``max_voxels_at_limit`` (linear extrapolation to the allocator limit,
``device_limit_mb``) derived from it.

No sample .vtu ships with this repo -- TexGen output is generated
externally, not committed. Point --data-dir at your own exports (see
readme.md in this folder).

The yarn phase is TransverseIsotropic with fiber_dir="from_input"
(materialmodels.factory), which picks up each file's own orientation field
automatically -- no per-file material config needed, only the geometry
changes. MATERIALS_CFG's constants are illustrative (typical E-glass
roving); edit them for your actual fiber/matrix.

Every (file, solver) pair runs in its own subprocess -- host_peak_rss_mb is
therefore always a true per-run peak (via `resource`), never a cumulative
process-wide one, and JAX's GPU allocator arena (which never returns freed
memory to the OS) starts clean for every run instead of getting fragmented
by earlier, differently-shaped solves.

Per-(file, solver) solved fields -- phase, yarn_index, orientation, strain,
stress (Voigt), von Mises stress, displacement -- are written to XDMF/HDF5
via utils.io.xdmf_writer.IncrementalWriter -- this project's standard
field-data output (see CLAUDE.md's IO conventions) -- one
<stem>__<solver_label>.h5/.xdmf pair per (file, solver) combination,
alongside the JSON summary in OUT_DIR, ONLY when --write-fields is passed
(off by default -- the write itself costs real time/device memory on top
of the solve at these grid sizes, not worth paying on every quick sweep).
Mirrors exactly what problems.mechanics.solve_mechanics's own writer=
support writes (field_to_grid/to_voigt/compute_displacement, same field
names) -- done by hand here instead of passed as writer= to that function,
so yarn_index (not a production field) can be added to the same increment
in one call.

Usage
-----
    python benchmark/benchmark_3/elastic_solve.py --data-dir data/texgen_exports
    python benchmark/benchmark_3/elastic_solve.py --data-dir data/texgen_exports --write-fields

Prints a summary table (one row per file x solver combination) and writes
one combined output/benchmark/benchmark_3/results_<date>.json; with
--write-fields, also <stem>__<solver_label>.h5/.xdmf per combination.
"""
import sys
sys.path.insert(0, "src")

import argparse
import datetime
import glob
import json
import os
import resource
import subprocess
import time
from typing import cast

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)
import jax
import jax.numpy as jnp
import numpy as np

from materialmodels.assembly import assemble_C_field
from materialmodels.factory import build_material
from operators.fft_distributed import choose_device_count
from post.fields import compute_displacement, field_to_grid, homogenize, to_voigt, von_mises
from problems.mechanics import solve_mechanics
from solvers.solution import ElasticitySolution
from utils.io.reader import SimulationReader
from utils.io.xdmf_writer import IncrementalWriter

OUT_DIR = "output/benchmark/benchmark_3"

# Phase 0 = matrix, phase 1 = yarn -- matches read_vtu's own convention, so
# this list's order must not change without checking that.
MATERIALS_CFG = [
    {"model": "isotropic_elastic", "E": 3.354e3, "nu": 0.38, "name": "epoxy matrix"},
    {"model": "transverse_isotropic", "E_L": 176.905e3, "E_T": 10.814e3, "G_LT": 6.9e3,
     "nu_LT": 0.245, "G_TT": 4.245e3, "fiber_dir": "from_input", "name": "carbon fiber yarn fvc75"},
]

EPS_BAR = jnp.array([
    [1.0e-3, 0.0, 0.0],
    [0.0,    0.0, 0.0],
    [0.0,    0.0, 0.0],
])
LOADED_COMPONENT = (0, 0)  # (i, j) into EPS_BAR -- which one the homogenized modulus divides by
TOLER_LIN = 1e-6
MAXITER = 1000

# (label, formulation, scheme) -- scheme selects GreenOperatorWillot vs.
# GreenOperatorBasic for "lippmann_schwinger", and the equivalent rotated
# vs. standard discretisation of the reference-medium-free projector for
# "fourier_galerkin" (operators.galerkin.GalerkinProjector); passed through
# unchanged for "displacement" too, where solve_mechanics ignores it.
SOLVER_CONFIGS: list[tuple[str, str, str]] = [
    ("ls_rotated",   "lippmann_schwinger", "rotated"),
    ("ls_standard",  "lippmann_schwinger", "standard"),
    ("displacement", "displacement",       "rotated"),
    ("galerkin",     "fourier_galerkin",   "rotated"),
]


def host_peak_rss_mb() -> float:
    """Peak resident set size so far in this process, in MB -- a true
    per-run peak, since every (file, solver) pair runs in its own
    subprocess (see module docstring)."""
    ru_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru_maxrss / (1024 * 1024) if sys.platform == "darwin" else ru_maxrss / 1024


def device_mem_mb() -> dict | None:
    """Current, peak-so-far and allocator-limit JAX device memory in MB --
    GPU/TPU only, None on CPU (and on any backend that doesn't implement
    memory_stats()). ``limit`` is what the allocator may grow to
    (XLA_PYTHON_CLIENT_MEM_FRACTION x device total), i.e. the real OOM
    threshold -- not the card's nominal capacity."""
    try:
        stats = jax.devices()[0].memory_stats()
    except Exception:
        return None
    if not stats or "peak_bytes_in_use" not in stats:
        return None
    mb = 1024.0 * 1024.0
    return {
        "in_use": stats.get("bytes_in_use", 0) / mb,
        "peak": stats["peak_bytes_in_use"] / mb,
        "limit": stats["bytes_limit"] / mb if "bytes_limit" in stats else None,
    }


def device_peak_mb() -> float | None:
    """Peak JAX device memory in MB -- GPU/TPU only, None on CPU (and on any
    backend that doesn't implement memory_stats())."""
    dev = device_mem_mb()
    return dev["peak"] if dev is not None else None


def warm_up_device() -> None:
    """Run a tiny einsum + FFT + while_loop on the device before any problem
    data exists, so the process's one-off GPU runtime cost (cuBLAS/cuFFT
    handles and workspaces, first-use XLA scratch) lands in the baseline
    instead of being charged to whichever stage happens to touch the GPU
    first. Without this, a small grid's first stage shows ~130 MB that has
    nothing to do with its own data."""
    @jax.jit
    def _warm(a):
        b = jnp.einsum("ijm,jkm->ikm", a, a)
        b = jnp.fft.ifftn(jnp.fft.fftn(b, axes=(0, 1, 2)), axes=(0, 1, 2)).real
        return jax.lax.while_loop(lambda c: c[1] < 3, lambda c: (c[0] * 0.5, c[1] + 1), (b, 0))[0]
    jax.block_until_ready(_warm(jnp.ones((3, 3, 64), dtype=EPS_BAR.dtype)))


def stage_peak_mb(before: dict | None, after: dict | None) -> tuple[float | None, bool]:
    """Device memory one stage needed at its own peak, on top of what was
    already live when it started: ``after.peak - before.in_use``.

    ``peak_bytes_in_use`` is a running maximum, so this is only resolved if
    the stage actually pushed the peak higher (``after.peak > before.peak``)
    -- returned with ``exact=True``. Otherwise the stage stayed under an
    earlier peak and all that's known is the upper bound
    ``before.peak - before.in_use`` (``exact=False``, printed as "<=").
    None on CPU/TPU."""
    if before is None or after is None:
        return None, False
    if after["peak"] > before["peak"]:
        return after["peak"] - before["in_use"], True
    return before["peak"] - before["in_use"], False


def mem_snapshot(path: str, label: str, n, stage: str) -> dict:
    """host peak-so-far, plus device current/peak-so-far/limit, right after
    one stage of bench_one. Printed immediately (so it shows up per
    realization the same way print_row's summary line does) and also
    returned for the JSON payload."""
    host_mb, dev = host_peak_rss_mb(), device_mem_mb()
    dev_s = (f"{dev['in_use']:.1f} MB in use / {dev['peak']:.1f} MB peak"
             if dev is not None else "n/a")
    print(f"    [{os.path.basename(path)} {label} {list(n)}] after {stage}: "
          f"host {host_mb:.1f} MB peak, device {dev_s}", file=sys.stderr)
    return {"host_peak_rss_mb": host_mb, "device": dev,
            "device_peak_mb": dev["peak"] if dev is not None else None}


def bench_one(path: str, label: str, formulation: str, scheme: str, write_fields: bool = False) -> dict:
    t0 = time.perf_counter()
    n, L, phase_np, orientations_np, yarn_index_np, vf_np, _, _ = SimulationReader(path).read()
    read_time_s = time.perf_counter() - t0
    Nv = int(np.prod(n))

    # Warm the device up *before* any problem data is on it -- see
    # warm_up_device. Everything measured after this is attributable to this
    # grid, and "baseline" is the fixed per-process runtime cost that stays
    # the same at any grid size.
    warm_up_device()
    mem_baseline = mem_snapshot(path, label, n, "baseline")

    phase = jnp.array(phase_np)
    orientations = jnp.array(orientations_np)
    materials = [build_material(m, orientations=orientations) for m in MATERIALS_CFG]

    # Explicit, standalone call purely to snapshot the material model's own
    # memory cost in isolation -- assemble_C_field is what actually builds
    # the (3,3,3,3,Nv) oriented stiffness field (this is where the OOM the
    # benchmark_3 investigation first found happens, well before the CG
    # solve itself -- an 18.5 GiB single allocation at 600x600x85, on every
    # formulation equally, since every one of them called this the same
    # unsharded way). n=n opts this into the same auto-sharded assembly
    # solve_mechanics's own internal call now uses (see assemble_C_field's
    # docstring) -- without it, this standalone diagnostic call would OOM
    # and abort the whole (file, solver) subprocess before ever reaching
    # the solve below, masking that the actual fix works. solve_mechanics
    # below builds its own C_field again internally (no way to hand it a
    # precomputed one through that API), so this genuinely costs an extra
    # materials-assembly pass -- acceptable for a benchmark script, not
    # something to do in production code.
    mem_before_materials = mem_snapshot(path, label, n, "inputs on device")
    jax.block_until_ready(assemble_C_field(materials, phase, n=n))
    # the returned C_field is dropped right away (never bound), so it is not
    # resident during the solve below -- solve_mechanics builds its own.
    mem_after_materials = mem_snapshot(path, label, n, "materials")

    # First call: XLA trace + compile (lax.while_loop inside the CG solve
    # always compiles to XLA on first use for a given shape, cached after)
    # plus that first solve. Second call: identical inputs, so the same
    # shape/dtype signature hits the now-warm compilation cache -- steady-
    # state solve time, with jit overhead no longer in it.
    # Every formulation auto-decomposes across jax.local_device_count()
    # devices -- n_devices_pmap just reports how many it actually picked.
    n_devices_pmap = choose_device_count(n)

    def _solve():
        results = solve_mechanics(n, L, phase, materials, EPS_BAR, formulation=formulation,
                                   scheme=scheme, toler_lin=TOLER_LIN, maxiter=MAXITER)
        sol = cast(ElasticitySolution, results[0].solution)
        jax.block_until_ready(sol.sigma)
        return sol

    t0 = time.perf_counter()
    _solve()
    jit_time_s = time.perf_counter() - t0
    mem_after_jit = mem_snapshot(path, label, n, "jit")

    t0 = time.perf_counter()
    sol = _solve()
    solve_time_s = time.perf_counter() - t0
    mem_after_solve = mem_snapshot(path, label, n, "solve")

    stem = f"{os.path.splitext(os.path.basename(path))[0]}__{label}"
    write_time_s = 0.0
    output_path = None
    mem_after_write = None
    if write_fields:
        t0 = time.perf_counter()
        eps_grid = field_to_grid(sol.eps, n)
        sigma_grid = field_to_grid(sol.sigma, n)
        # sol.eps_bar is populated only by formulation="displacement" (mixed
        # strain/stress BC) -- None for lippmann_schwinger's pure strain BC,
        # where the prescribed EPS_BAR is exactly the macroscopic strain
        # (stepping="single", never overridden here, always reaches t=1.0).
        eps_bar_u = sol.eps_bar if sol.eps_bar is not None else EPS_BAR
        u_grid = compute_displacement(sol.eps, eps_bar_u, n, L)
        with IncrementalWriter(os.path.join(OUT_DIR, stem), grid_shape=n, grid_length=L) as w:
            w.write_increment(0, {
                "phase":        np.asarray(phase_np).reshape(n).astype(np.float32),
                "yarn_index":   np.asarray(yarn_index_np).reshape(n).astype(np.int32),
                "orientation":  np.asarray(orientations_np).T.reshape(*n, 3).astype(np.float64),
                "strain":       to_voigt(eps_grid).astype(np.float64),
                "stress":       to_voigt(sigma_grid).astype(np.float64),
                "von_mises":    von_mises(sigma_grid).astype(np.float64),
                "displacement": np.asarray(u_grid).astype(np.float64),
            }, time=0.0)
        write_time_s = time.perf_counter() - t0
        output_path = os.path.join(OUT_DIR, stem + ".xdmf")
        mem_after_write = mem_snapshot(path, label, n, "write")

    # Homogenized modulus from the already-solved fields -- no second solve.
    # eps_bar_h/sigma_bar_h are the *actual* volume averages (homogenize()),
    # not just the prescribed EPS_BAR, so this stays correct even if
    # solve_mechanics didn't fully converge. This is the constrained
    # (pure-strain-BC) modulus, not a free-surface engineering modulus --
    # for the latter (with transverse Poisson's ratios), see
    # learning.extractors.effective_modulus, which needs its own separate
    # mixed-BC solve and isn't reusable here.
    i, j = LOADED_COMPONENT
    eps_bar_h, sigma_bar_h = homogenize(sol.eps, sol.sigma)
    E_eff_MPa = float(sigma_bar_h[i, j] / eps_bar_h[i, j])

    # Per-stage device memory: each stage's own peak on top of what was
    # already live when it started (see stage_peak_mb). c_assemble =
    # assemble_C_field alone; solve = XLA compile + the first solve together
    # (the "jit" call -- compile scratch and the first solve can't be
    # separated, the peak is a running max and never resets).
    c_assemble_device_mb, c_assemble_exact = stage_peak_mb(
        mem_before_materials["device"], mem_after_materials["device"])
    solve_device_mb, solve_exact = stage_peak_mb(
        mem_after_materials["device"], mem_after_jit["device"])

    # The number to report / extrapolate with: everything the first
    # solve_mechanics call (compile + solve -- the call that OOMs) needs on
    # the device *for this grid*, i.e. its peak minus the fixed runtime
    # baseline. Inputs (phase, orientations) are included, since they are
    # part of the problem. Only valid if the solve set the process peak
    # (always true once the grid is large enough to matter; flagged if not).
    dev_base, dev_jit = mem_baseline["device"], mem_after_jit["device"]
    if dev_base is not None and dev_jit is not None:
        problem_device_mb = dev_jit["peak"] - dev_base["peak"]
        problem_exact = solve_exact
        bytes_per_voxel = problem_device_mb * 1024.0 * 1024.0 / Nv
        limit_mb = dev_jit["limit"]
        # linear-in-Nv extrapolation: largest grid whose problem memory still
        # fits under the allocator limit on top of the fixed baseline
        max_voxels = (int((limit_mb - dev_base["peak"]) * 1024.0 * 1024.0 / bytes_per_voxel)
                      if limit_mb is not None and bytes_per_voxel > 0 else None)
    else:
        problem_device_mb = bytes_per_voxel = limit_mb = max_voxels = None
        problem_exact = False

    return {
        "path": path,
        "solver_label": label,
        "formulation": formulation,
        "scheme": scheme,
        "n": list(n),
        "Nv": int(phase.shape[0]),
        "n_devices_pmap": n_devices_pmap,
        "converged": bool(sol.converged),
        "E_eff_MPa": E_eff_MPa,
        "eps_bar_loaded": float(eps_bar_h[i, j]),
        "sigma_bar_loaded_MPa": float(sigma_bar_h[i, j]),
        "read_time_s": read_time_s,
        "jit_time_s": jit_time_s,
        "solve_time_s": solve_time_s,
        "write_time_s": write_time_s,
        "host_peak_rss_mb": host_peak_rss_mb(),
        "device_peak_mb": device_peak_mb(),
        "baseline_device_mb": dev_base["peak"] if dev_base is not None else None,
        "device_limit_mb": limit_mb,
        "c_assemble_device_mb": c_assemble_device_mb,
        "c_assemble_exact": c_assemble_exact,
        "solve_device_mb": solve_device_mb,
        "solve_exact": solve_exact,
        "problem_device_mb": problem_device_mb,
        "problem_exact": problem_exact,
        "bytes_per_voxel": bytes_per_voxel,
        "max_voxels_at_limit": max_voxels,
        "mem_mb": {
            "baseline": mem_baseline,
            "before_materials": mem_before_materials,
            "after_materials": mem_after_materials,
            "after_jit": mem_after_jit,
            "after_solve": mem_after_solve,
            "after_write": mem_after_write,
        },
        "output": output_path,
    }


def find_vtu_files(data_dir: str) -> list[str]:
    # Smallest first -- cheapest solves run (and fail, if something's wrong)
    # before the expensive ones burn any time.
    return sorted(glob.glob(os.path.join(data_dir, "*.vtu")), key=os.path.getsize)


def _mb(value: float | None, exact: bool = True) -> str:
    """MB column: "n/a" on CPU/TPU, "<=" prefix for an upper bound (a stage
    that stayed under an earlier peak, see stage_peak_mb)."""
    if value is None:
        return "n/a"
    return f"{value:.1f}" if exact else f"<={value:.1f}"


def print_row(r: dict) -> None:
    bpv = f"{r['bytes_per_voxel']:.0f}" if r["bytes_per_voxel"] is not None else "n/a"
    max_nv = (f"{r['max_voxels_at_limit'] / 1e6:.1f}M"
              if r["max_voxels_at_limit"] is not None else "n/a")
    print(f"{os.path.basename(r['path']):<34} {r['solver_label']:<13} {str(r['n']):>14} "
          f"{r['read_time_s']:>9.3f} {r['jit_time_s']:>9.3f} {r['solve_time_s']:>9.3f} "
          f"{r['host_peak_rss_mb']:>13.1f} {_mb(r['device_peak_mb']):>11} "
          f"{_mb(r['c_assemble_device_mb'], r['c_assemble_exact']):>12} "
          f"{_mb(r['solve_device_mb'], r['solve_exact']):>12} "
          f"{_mb(r['problem_device_mb'], r['problem_exact']):>12} {bpv:>9} {max_nv:>9} "
          f"{r['E_eff_MPa']:>12.1f} {str(r['converged']):>10}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True,
                         help="directory to scan for *.vtu TexGen exports")
    parser.add_argument("--write-fields", action="store_true",
                         help="also write per-(file, solver) XDMF/HDF5 field output "
                              "(phase, strain, stress, von Mises, displacement); off by "
                              "default, since the write itself costs real time/device "
                              "memory on top of the solve at these grid sizes")
    parser.add_argument("--single", default=None,
                         help=argparse.SUPPRESS)  # internal: one-run subprocess worker mode
    parser.add_argument("--solver-label", default=None,
                         help=argparse.SUPPRESS)  # internal: paired with --single
    args = parser.parse_args()

    if args.single is not None:
        # Worker invocation from the subprocess loop below: benchmark
        # exactly one (file, solver) pair, print its JSON result on stdout,
        # nothing else.
        label, formulation, scheme = next(
            c for c in SOLVER_CONFIGS if c[0] == args.solver_label
        )
        print(json.dumps(bench_one(args.single, label, formulation, scheme, write_fields=args.write_fields)))
        return

    paths = find_vtu_files(args.data_dir)
    if not paths:
        raise SystemExit(f"No .vtu files found in {args.data_dir!r}.")

    print(f"JAX backend: {jax.default_backend()}   device: {jax.devices()[0]}")
    dev = device_mem_mb()
    if dev is not None and dev["limit"] is not None:
        print(f"Device allocator limit: {dev['limit'] / 1024:.2f} GiB "
              f"(XLA_PYTHON_CLIENT_MEM_FRACTION={os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION', 'default 0.75')})")
    print(f"Found {len(paths)} .vtu file(s) in {args.data_dir}, "
          f"{len(SOLVER_CONFIGS)} solver config(s) each (one subprocess per run)")
    print(f"{'file':<34} {'solver':<13} {'grid':>14} {'read [s]':>9} {'jit [s]':>9} "
          f"{'solve [s]':>9} {'host RSS [MB]':>13} {'device [MB]':>11} "
          f"{'C asm [MB]':>12} {'solve [MB]':>12} {'problem[MB]':>12} "
          f"{'B/voxel':>9} {'max Nv':>9} "
          f"{'E_eff [MPa]':>12} {'converged':>10}")

    results = []
    for path in paths:
        for label, formulation, scheme in SOLVER_CONFIGS:
            cmd = [sys.executable, __file__, "--data-dir", args.data_dir,
                   "--single", path, "--solver-label", label]
            if args.write_fields:
                cmd.append("--write-fields")
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                # On OOM, XLA's own message says how much it tried to
                # allocate on top of what was already in use -- that sum,
                # not the in-use figure at the time of the crash, is what
                # the run actually needed. Show that plus the per-stage
                # snapshots instead of the whole traceback.
                lines = proc.stderr.splitlines()
                oom = [ln for ln in lines if "RESOURCE_EXHAUSTED" in ln or "trying to allocate" in ln]
                if oom:
                    stages = [ln for ln in lines if "] after " in ln]
                    print(f"  OOM: {path} [{label}]\n" + "\n".join(stages + oom[:3]))
                else:
                    print(f"  FAILED: {path} [{label}]\n{proc.stderr}")
                continue
            r = json.loads(proc.stdout.strip().splitlines()[-1])
            results.append(r)
            print_row(r)

    today = datetime.date.today().isoformat()
    payload = {
        "generated": today,
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "data_dir": args.data_dir,
        "materials": MATERIALS_CFG,
        "eps_bar": EPS_BAR.tolist(),
        "solver": {"toler_lin": TOLER_LIN, "maxiter": MAXITER, "configs": SOLVER_CONFIGS},
        "results": results,
    }

    out_path = os.path.join(OUT_DIR, f"results_{today}.json")
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {len(results)} results to {out_path}")


if __name__ == "__main__":
    main()
