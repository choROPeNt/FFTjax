"""
Linear-elastic solve benchmark over a directory of TexGen-exported .vtu
microstructures -- same geometry family (e.g. different fabric weights or
TexGen export resolutions), each carrying its own per-voxel fiber
orientation field (YarnTangent), looped over one at a time and, for each
file, over every (formulation, scheme) pair in SOLVER_CONFIGS -- Lippmann-
Schwinger with the rotated (GreenOperatorWillot) and standard
(GreenOperatorBasic) schemes, the displacement-based formulation (scheme
doesn't apply there, see solve_mechanics's own docstring), and the
reference-medium-free Fourier-Galerkin formulation (Vondrejc et al 2014;
see notes/FOURIER_GALERKIN.md). Tracks
wall-clock read/jit/solve time and peak memory (host RSS, plus JAX device
memory on GPU/TPU) per (file, solver) pair. jit_time_s is solve_mechanics's
first call for that pair's grid shape (XLA trace + compile + run --
lax.while_loop inside the CG solve always compiles to XLA on first use for a
given shape, even with no explicit jax.jit on solve_mechanics itself);
solve_time_s is a second, identical call, which hits the now-warm
compilation cache -- steady-state solve time with jit overhead no longer
in it.

No sample .vtu ships with this repo -- TexGen output is generated
externally, not committed. Point --data-dir at your own exports (see
readme.md in this folder).

The yarn phase is TransverseIsotropic with fiber_dir="from_input"
(materialmodels.factory), which picks up each file's own orientation field
automatically -- no per-file material config needed, only the geometry
changes. MATERIALS_CFG's constants are illustrative (typical E-glass
roving); edit them for your actual fiber/matrix.

Memory caveat: host_peak_rss_mb is the process's peak-so-far (via
`resource`), so it only ever grows across files run in the same process --
later files' numbers include earlier files' now-freed peaks, not an
isolated measurement. For a true per-file peak, run with --isolate instead,
which re-invokes this script once per (file, solver) pair as its own
subprocess.

Fragmentation/OOM caveat: JAX's GPU allocator never returns freed device
memory to the OS and never defragments its arena, so running every solver
config back to back in one process (the plain, no-flag mode) leaves memory
from earlier, differently-shaped solves fragmented across the arena by the
time a later, larger realization needs one big contiguous block -- often
failing well before the arena is nominally full (see notes on
XLA_PYTHON_CLIENT_MEM_FRACTION in .env.example). --isolate-solver is the
fix that costs the least subprocess-spawn/recompilation overhead: one fresh
subprocess per SOLVER CONFIG (not per file), looping over every realization
(--data-dir's .vtu files, smallest to largest) inside that one process, so
each solver gets a clean, unfragmented arena for its whole sweep, and a
crash on one (large) realization doesn't take down the other solvers' runs
or discard results already gathered for smaller realizations of the same
solver.

Per-(file, solver) solved fields -- phase, yarn_index, orientation, strain,
stress (Voigt), von Mises stress, displacement -- are also written to
XDMF/HDF5 via utils.io.xdmf_writer.IncrementalWriter -- this project's
standard field-data output (see CLAUDE.md's IO conventions) -- one
<stem>__<solver_label>.h5/.xdmf pair per (file, solver) combination,
alongside the JSON summary in OUT_DIR. Mirrors exactly what
problems.mechanics.solve_mechanics's own writer= support writes
(field_to_grid/to_voigt/compute_displacement, same field names) -- done by
hand here instead of passed as writer= to that function, so yarn_index (not
a production field) can be added to the same increment in one call. Pass
--no-fields to skip this entirely (summary table/JSON only) -- useful when
only the timing/memory numbers matter, since field_to_grid/compute_
displacement and the HDF5 write itself cost real time and disk on the
largest realizations.

Usage
-----
    python benchmark/benchmark_3/elastic_solve_vtu.py --data-dir data/texgen_exports
    python benchmark/benchmark_3/elastic_solve_vtu.py --data-dir data/texgen_exports --isolate
    python benchmark/benchmark_3/elastic_solve_vtu.py --data-dir data/texgen_exports --isolate-solver

Prints a summary table (one row per file x solver combination) and writes,
per combination, <stem>__<solver_label>.h5/.xdmf plus one combined
output/benchmark/benchmark_3/results_<date>.json.
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
    """Peak resident set size so far (this process), in MB. See the module
    docstring's memory caveat -- cumulative across the whole run, not
    isolated per file unless run with --isolate."""
    ru_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru_maxrss / (1024 * 1024) if sys.platform == "darwin" else ru_maxrss / 1024


def device_peak_mb() -> float | None:
    """Peak JAX device memory in MB -- GPU/TPU only, None on CPU (and on any
    backend that doesn't implement memory_stats())."""
    try:
        stats = jax.devices()[0].memory_stats()
    except Exception:
        return None
    peak = (stats or {}).get("peak_bytes_in_use")
    return peak / (1024 * 1024) if peak is not None else None


def mem_snapshot(path: str, label: str, n, stage: str) -> dict:
    """host/device peak-so-far right after one stage of bench_one (read,
    jit, solve, write) -- both are running peaks, not "current" usage, so a
    sequence of these across stages shows which stage first pushed the
    process peak to its current level, not just the single end-of-call
    number the returned row's host_peak_rss_mb/device_peak_mb already give.
    Printed immediately (so it shows up per realization the same way
    print_row's summary line does) and also returned for the JSON payload."""
    host_mb, dev_mb = host_peak_rss_mb(), device_peak_mb()
    dev_s = f"{dev_mb:.1f}" if dev_mb is not None else "n/a"
    print(f"    [{os.path.basename(path)} {label} {list(n)}] after {stage}: "
          f"host {host_mb:.1f} MB peak, device {dev_s} MB peak")
    return {"host_peak_rss_mb": host_mb, "device_peak_mb": dev_mb}


def bench_one(path: str, label: str, formulation: str, scheme: str, write_fields: bool = True) -> dict:
    t0 = time.perf_counter()
    n, L, phase_np, orientations_np, yarn_index_np, vf_np, _, _ = SimulationReader(path).read()
    read_time_s = time.perf_counter() - t0
    mem_after_read = mem_snapshot(path, label, n, "read")

    phase = jnp.array(phase_np)
    orientations = jnp.array(orientations_np)
    materials = [build_material(m, orientations=orientations) for m in MATERIALS_CFG]

    # Explicit, standalone call purely to snapshot the material model's own
    # memory cost in isolation -- assemble_C_field is what actually builds
    # the (3,3,3,3,Nv) oriented stiffness field (this is where the OOM in
    # the benchmark_3 investigation happens, well before the CG solve
    # itself). solve_mechanics below builds its own C_field again
    # internally (no way to hand it a precomputed one through that API), so
    # this genuinely costs an extra materials-assembly pass -- acceptable
    # for a benchmark script, not something to do in production code.
    jax.block_until_ready(assemble_C_field(materials, phase))
    mem_after_materials = mem_snapshot(path, label, n, "materials")

    # First call: XLA trace + compile (lax.while_loop inside the CG solve
    # always compiles to XLA on first use for a given shape, cached after --
    # even with no explicit jax.jit on solve_mechanics itself, same pattern
    # benchmark_lin_elastic_solve.py's compile_ms/run_ms split relies on)
    # plus that first solve. Second call: identical inputs, so the same
    # shape/dtype signature hits the now-warm compilation cache -- steady-
    # state solve time, with jit overhead no longer in it.
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
    output_path = None
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
    else:
        write_time_s = 0.0
        mem_after_write = None

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

    return {
        "path": path,
        "solver_label": label,
        "formulation": formulation,
        "scheme": scheme,
        "n": list(n),
        "Nv": int(phase.shape[0]),
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
        "mem_mb": {
            "after_read": mem_after_read,
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


def print_row(r: dict) -> None:
    dev = f"{r['device_peak_mb']:.1f}" if r["device_peak_mb"] is not None else "n/a"
    print(f"{os.path.basename(r['path']):<34} {r['solver_label']:<13} {str(r['n']):>14} "
          f"{r['read_time_s']:>9.3f} {r['jit_time_s']:>9.3f} {r['solve_time_s']:>9.3f} "
          f"{r['host_peak_rss_mb']:>13.1f} {dev:>11} {r['E_eff_MPa']:>12.1f} {str(r['converged']):>10}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True,
                         help="directory to scan for *.vtu TexGen exports")
    parser.add_argument("--isolate", action="store_true",
                         help="re-run this script once per (file, solver) pair, each its own "
                              "subprocess, for a true per-run peak-memory measurement instead "
                              "of a running process-wide peak")
    parser.add_argument("--isolate-solver", action="store_true",
                         help="re-run this script once per SOLVER CONFIG, each its own "
                              "subprocess looping over every realization (.vtu file) for that "
                              "solver -- a clean, unfragmented GPU allocator per solver, at far "
                              "less subprocess/recompilation overhead than --isolate, and a "
                              "crash on one realization doesn't discard results already "
                              "gathered for smaller ones or take down the other solvers")
    parser.add_argument("--no-fields", action="store_true",
                         help="skip writing the per-(file, solver) XDMF/HDF5 field output via "
                              "IncrementalWriter -- just the summary table/JSON, no phase/"
                              "strain/stress/displacement fields on disk")
    parser.add_argument("--single", default=None,
                         help=argparse.SUPPRESS)  # internal: one-run worker mode for --isolate
    parser.add_argument("--solver-label", default=None,
                         help=argparse.SUPPRESS)  # internal: paired with --single
    parser.add_argument("--only-solver", default=None,
                         help=argparse.SUPPRESS)  # internal: worker mode for --isolate-solver
    args = parser.parse_args()

    if args.single is not None:
        # Worker invocation from --isolate below: benchmark exactly one
        # (file, solver) pair, print its JSON result on stdout, nothing else.
        label, formulation, scheme = next(
            c for c in SOLVER_CONFIGS if c[0] == args.solver_label
        )
        print(json.dumps(bench_one(args.single, label, formulation, scheme,
                                    write_fields=not args.no_fields)))
        return

    if args.only_solver is not None:
        # Worker invocation from --isolate-solver below: benchmark every
        # realization for exactly one solver config, in this one process, one
        # JSON result per line on stdout (smallest file first, same order as
        # find_vtu_files) so the parent can salvage completed lines even if a
        # later, larger realization crashes this worker outright.
        label, formulation, scheme = next(
            c for c in SOLVER_CONFIGS if c[0] == args.only_solver
        )
        for path in find_vtu_files(args.data_dir):
            print(json.dumps(bench_one(path, label, formulation, scheme,
                                        write_fields=not args.no_fields)), flush=True)
        return

    paths = find_vtu_files(args.data_dir)
    if not paths:
        raise SystemExit(f"No .vtu files found in {args.data_dir!r}.")

    print(f"JAX backend: {jax.default_backend()}   device: {jax.devices()[0]}")
    mode_note = ("  (--isolate: one subprocess per run)" if args.isolate else
                 "  (--isolate-solver: one subprocess per solver)" if args.isolate_solver else "")
    print(f"Found {len(paths)} .vtu file(s) in {args.data_dir}, "
          f"{len(SOLVER_CONFIGS)} solver config(s) each" + mode_note)
    print(f"{'file':<34} {'solver':<13} {'grid':>14} {'read [s]':>9} {'jit [s]':>9} "
          f"{'solve [s]':>9} {'host RSS [MB]':>13} {'device [MB]':>11} "
          f"{'E_eff [MPa]':>12} {'converged':>10}")

    extra_flags = ["--no-fields"] if args.no_fields else []

    results = []
    if args.isolate_solver:
        # Outer loop over SOLVER CONFIGS, not files: one subprocess per
        # solver, each looping internally over every realization for that
        # solver alone -- a clean, unfragmented GPU allocator per solver,
        # and a crash partway through one solver's sweep still keeps every
        # realization's result that printed before it, plus every other
        # solver's full results.
        for label, formulation, scheme in SOLVER_CONFIGS:
            # Popen + live iteration over stdout, not subprocess.run -- run()
            # only hands stdout back once the child exits, so with several
            # realizations per solver every row would sit buffered and
            # invisible until the whole solver's sweep finished (or crashed).
            # Iterating the pipe as lines arrive prints each realization's
            # row the moment that worker process flushes it, same as the
            # non-isolated/--isolate paths already do.
            proc = subprocess.Popen(
                [sys.executable, __file__, "--data-dir", args.data_dir,
                 "--only-solver", label, *extra_flags],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            n_before = len(results)
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                results.append(r)
                print_row(r)
            proc.wait()
            if proc.returncode != 0:
                stderr = proc.stderr.read() if proc.stderr else ""
                print(f"  FAILED: solver {label!r} after {len(results) - n_before} "
                      f"realization(s)\n{stderr}")
    else:
        for path in paths:
            for label, formulation, scheme in SOLVER_CONFIGS:
                if args.isolate:
                    proc = subprocess.run(
                        [sys.executable, __file__, "--data-dir", args.data_dir,
                         "--single", path, "--solver-label", label, *extra_flags],
                        capture_output=True, text=True,
                    )
                    if proc.returncode != 0:
                        print(f"  FAILED: {path} [{label}]\n{proc.stderr}")
                        continue
                    r = json.loads(proc.stdout.strip().splitlines()[-1])
                else:
                    r = bench_one(path, label, formulation, scheme, write_fields=not args.no_fields)
                results.append(r)
                print_row(r)

    today = datetime.date.today().isoformat()
    payload = {
        "generated": today,
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "data_dir": args.data_dir,
        "isolated": args.isolate,
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
