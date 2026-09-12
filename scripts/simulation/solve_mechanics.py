"""
Linear-elastic mechanical homogenization on a loaded microstructure — YAML-driven.

Loads an existing microstructure (XDMF/HDF5, e.g. from
scripts/generation/generate_weave.py or generate_rve.py) via
utils.io.reader.SimulationReader, then solves the mechanical equilibrium
problem under a prescribed macroscopic strain via
problems.mechanics.solve_mechanics, which also owns writing each accepted
increment to the XDMF/HDF5 output. Geometry logic and solver
wiring both live outside this script -- see src/generation/,
src/materialmodels/, src/problems/mechanics.py, src/problems/incremental.py.

``stepping.mode`` selects how the target strain is reached:
  single    (default) -- one solve at the full eps_bar, no load stepping.
  fixed     -- equal load-fraction increments of size stepping.dt.
  automatic -- adaptive load-fraction step, grown on convergence, cut back
               and retried on non-convergence, Abaqus-*STATIC style.
Writes one XDMF/HDF5 increment per accepted step (time = load fraction
reached) -- for "single" that's just one increment at time=1.0. Progress
prints live as each increment converges, not only after the whole solve returns.

``control``/``stress_bar`` (formulation="displacement" only) prescribe a
mixed macroscopic strain/stress BC -- control's 1-entries mark which
directions are stress- rather than strain-controlled, stress_bar gives their
target (eps_bar is ignored on those entries). Omit both for pure strain BC.

Several load cases per run
--------------------------
A ``mechanics.loading`` block turns the config into a SWEEP: several load
cases on the one microstructure, each its own solve and its own output files
(suffixed ``_<case>``), with the geometry read and materials built once.
``cases: base6`` is the canonical set (uniaxial x/y/z, shear xy/xz/yz, in
post.fields.to_voigt's Voigt order); ``free_surfaces: true`` derives each
case's ``control`` mask per case -- loaded component strain-driven, every
unloaded surface traction-free -- which one fixed mask cannot express across
six cases. See problems.loadcases (shared with solve_inelastic.py) and
configs/simulation/sve_base6_uni.yaml.

A case here contributes its driven component, its PEAK gamma (engineering
strain: eps_bar[i,i] = gamma for a normal case, eps_bar[i,j] = gamma/2 for a
shear one) and its BC. The load PATH stays ``stepping``'s job, so a case's
cycle keys are ignored here -- unlike solve_inelastic.py, where the path is
the point. Without a ``loading`` block the single ``eps_bar`` tensor below is
the whole load and outputs are unsuffixed, exactly as before.

``write_fields`` controls per-voxel output, which at 152^3 costs ~0.4 GB and
several times the solve's own wall time per increment:
  true  (default) -- the zero-field step-0 reference plus every increment
  increments      -- every increment, no step-0 reference (identical in every
                     case of a sweep)
  false           -- no field output at all; stats, the metadata .h5 and the
                     per-sweep summary CSV are still written

String values in the YAML support {variable} interpolation:
  output:  "output/simulation"
  jobname: "{mechanics.input.stem}"

Usage
-----
    python scripts/simulation/solve_mechanics.py configs/simulation/mechanics_example.yaml
    python scripts/simulation/solve_mechanics.py configs/simulation/sve_base6_uni.yaml
    # one case out of a sweep, or just what would run:
    python scripts/simulation/solve_mechanics.py <config> --cases shear_xy
    python scripts/simulation/solve_mechanics.py <config> --list-cases

Output
------
    <output>/<jobname>[_<case>].h5
    <output>/<jobname>[_<case>].xdmf
    <output>/<jobname>[_<case>]_stats.npy   -- one structured-array row per
        accepted increment: step, t, dt, converged, wall_time, write_time,
        eps_bar_voigt (6,), sigma_bar_voigt (6,) -- np.load(path) to read.
    <output>/<jobname>_cases_summary.csv    -- sweeps only: one row per case
        with the final homogenized eps and sigma in Voigt order
        (11, 22, 33, 12, 13, 23), i.e. what an effective 6x6 is built from.
"""

import argparse
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import cast

import h5py
import numpy as np

sys.path.insert(0, "src")

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)
import jax.numpy as jnp

from materialmodels.factory import build_material
from post.fields import homogenize, to_voigt
from problems.loadcases import LoadCase, resolve_cases, select_cases
from problems.mechanics import solve_mechanics
from solvers.solution import ElasticitySolution
from utils.config import field_write_mode, load_config
from utils.io.reader import SimulationReader
from utils.io.xdmf_writer import IncrementalWriter

# per-increment solver + homogenization stats, saved to <stem>_stats.npy --
# one structured-array row per accepted increment.
_STATS_DTYPE = np.dtype([
    ("step", "i4"), ("t", "f8"), ("dt", "f8"), ("converged", "?"),
    ("wall_time", "f8"), ("write_time", "f8"),
    ("eps_bar_voigt", "f8", (6,)), ("sigma_bar_voigt", "f8", (6,)),
])


def main():
    parser = argparse.ArgumentParser(
        description="Solve linear-elastic homogenization on a loaded microstructure (XDMF/HDF5)"
    )
    parser.add_argument("config", type=Path, help="YAML configuration file")
    parser.add_argument("--cases", type=str, default=None,
                        help="comma-separated subset of mechanics.loading.cases to run, in "
                             "the given order (default: all of them)")
    parser.add_argument("--list-cases", action="store_true",
                        help="print the load cases this config resolves to and exit "
                             "(no geometry read, no solve)")
    parser.add_argument("--keep-going", action="store_true",
                        help="on a non-converged case, carry on with the remaining cases "
                             "instead of stopping (exit status is still nonzero)")
    args = parser.parse_args()

    cfg  = load_config(args.config)
    mcfg = cfg["mechanics"]
    scfg = mcfg.get("stepping", {})
    lcfg = mcfg.get("loading")
    print(f"Config : {args.config}")

    # ── load cases (optional) ───────────────────────────────────────────────
    # With a `loading` block this config is a SWEEP: several load cases on the
    # one microstructure, each its own solve and its own output files. Without
    # it, the single `eps_bar`/`control` pair below is the whole load, exactly
    # as before. Resolved before the geometry read so a config error costs
    # nothing to discover. See problems.loadcases.
    #
    # A case here contributes its driven component, its PEAK gamma and its BC;
    # the load PATH is `stepping`'s job (t*eps_bar), so a case's cycle keys
    # (n_load/n_unload/n_reload/gammas) are meaningless in this script and
    # ignored -- unlike in solve_inelastic.py, where the path is the point.
    cases: list[LoadCase] | None = None
    if lcfg is not None:
        try:
            cases = select_cases(
                resolve_cases(lcfg, control=mcfg.get("control"),
                              stress_bar=mcfg.get("stress_bar")),
                [s.strip() for s in args.cases.split(",")] if args.cases else None,
            )
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    elif args.cases:
        print("error: --cases needs a `mechanics.loading` block in the config.", file=sys.stderr)
        return 2
    if args.list_cases:
        if cases is None:
            print("  (no `mechanics.loading` block -- single eps_bar run, no load cases)")
        for case in cases or []:
            print(f"  {case.describe(path=False)}")
        return 0

    # ── load microstructure ──────────────────────────────────────────────────
    # input_L/input_dx are only needed for a metadata-less input (.npy, or a
    # bare .h5 with no n/L attrs) -- every other format (.xdmf/.vtu/.vti/.npz)
    # carries its own grid length and ignores these (with a warning if given
    # anyway). See utils.io.reader.SimulationReader.
    src = mcfg["input"]
    input_L  = tuple(mcfg["input_L"])  if mcfg.get("input_L")  else None
    input_dx = tuple(mcfg["input_dx"]) if mcfg.get("input_dx") else None
    phase_key = mcfg.get("phase_key", "phase")
    orientation_key = mcfg.get("orientation_key", "orientation")
    print(f"Input  : {src}")
    n, L, phase_np, orientations_np, _, vf_np, _, _ = SimulationReader(
        src, L=input_L, dx=input_dx, phase_key=phase_key, orientation_key=orientation_key,
    ).read()
    phase = jnp.array(phase_np)
    print(f"Grid   : {n}   phi = {float(np.mean(phase_np > 0)):.3f}")

    # ── materials (list, indexed by 0-based phase id) ───────────────────────
    # orientations: passed through so a phase config'd with `fiber_dir:
    # from_input` (transverse_isotropic only) picks up this geometry's own
    # per-voxel orientation field -- ignored by every other config.
    orientations = jnp.array(orientations_np)
    materials = [build_material(m, orientations=orientations) for m in mcfg["materials"]]
    for i, m in enumerate(materials):
        print(f"  phase {i}: {m}")

    # ── what to load with ────────────────────────────────────────────────────
    # Sweep: one run per resolved case (component + peak gamma + its own BC).
    # Single: the config's own eps_bar tensor and BC, exactly as before.
    control_cfg = mcfg.get("control")
    control = tuple(tuple(int(c) for c in row) for row in control_cfg) if control_cfg else None
    stress_bar_cfg = mcfg.get("stress_bar")
    stress_goal = jnp.array(stress_bar_cfg, dtype=jnp.float64) if stress_bar_cfg else None

    if cases is not None:
        runs = [(c.name,
                 jnp.asarray(c.eps_bar(c.gamma_max), dtype=jnp.float64),
                 c.control,
                 None if c.stress_bar is None else jnp.asarray(c.stress_bar, dtype=jnp.float64))
                for c in cases]
    else:
        runs = [(None, jnp.array(mcfg["eps_bar"], dtype=jnp.float64), control, stress_goal)]

    # ── output ────────────────────────────────────────────────────────────────
    output  = cfg["output"]
    jobname = cfg["jobname"]
    Path(output).mkdir(parents=True, exist_ok=True)
    write_fields = field_write_mode(mcfg.get("write_fields", True))
    mode = scfg.get("mode", "single")

    summaries = []
    for k, (name, eps_bar_run, control_run, stress_run) in enumerate(runs, start=1):
        stem = f"{output}/{jobname}_{name}" if name else f"{output}/{jobname}"
        if name:
            print(f"\n── load case {k}/{len(runs)}: {name} "
                  f"────────────────────────────────────")
        summaries.append(run_case(
            name=name, stem=stem, eps_bar=eps_bar_run, control=control_run,
            stress_goal=stress_run, n=n, L=L, phase=phase, phase_np=phase_np,
            materials=materials, orientations_np=orientations_np, mcfg=mcfg, scfg=scfg,
            mode=mode, write_fields=write_fields, src=src,
        ))
        if not summaries[-1]["ok"] and not args.keep_going:
            print("\nStopping after a non-converged case (--keep-going runs the rest anyway).")
            break

    if len(runs) > 1:
        _write_summary(f"{output}/{jobname}", summaries)

    print("\nOpen the .xdmf in ParaView with the 'Xdmf3ReaderT' reader."
          if write_fields != "none" else
          "\nNo field output was written (write_fields: false) -- see the _stats.npy / summary CSV.")
    return 0 if all(s["ok"] for s in summaries) and len(summaries) == len(runs) else 1


def _write_summary(stem: str, summaries: list[dict]) -> None:
    """
    One CSV row per case with the final homogenized strain and stress in Voigt
    order (11, 22, 33, 12, 13, 23) -- the sweep's actual deliverable, and what
    an effective 6x6 is assembled from. Per-case _stats.npy files keep the full
    per-increment history.
    """
    labels = ("11", "22", "33", "12", "13", "23")
    path = f"{stem}_cases_summary.csv"
    with open(path, "w") as f:
        f.write("case,converged," + ",".join(f"eps_{v}" for v in labels)
                + "," + ",".join(f"sig_{v}" for v in labels) + "\n")
        for s in summaries:
            f.write(f"{s['name']},{int(s['ok'])},"
                    + ",".join(f"{v:.8e}" for v in s["eps_voigt"]) + ","
                    + ",".join(f"{v:.8e}" for v in s["sig_voigt"]) + "\n")
    print(f"\n{'case':<14s}{'converged':>10s}" + "".join(f"{'sig_'+v:>13s}" for v in labels))
    for s in summaries:
        print(f"{s['name']:<14s}{str(s['ok']):>10s}"
              + "".join(f"{v:>13.4f}" for v in s["sig_voigt"]))
    print(f"\nWritten → {path}")


def run_case(*, name, stem, eps_bar, control, stress_goal, n, L, phase, phase_np,
             materials, orientations_np, mcfg, scfg, mode, write_fields, src) -> dict:
    """
    One load case: solve (single/fixed/automatic all go through the same API),
    write its increments, stats and metadata, and return a summary row.

    The writer is handed to solve_mechanics so it writes each accepted
    increment itself, as it is produced -- on_increment prints live for the
    same reason, instead of only after the whole solve returns.
    """
    stats_rows = []

    def _report(r, write_time):
        sol = cast(ElasticitySolution, r.solution)
        eps_bar_out, sigma_bar = homogenize(sol.eps, sol.sigma)
        print(f"  step {r.step:3d}  t={r.t:.4f}  dt={r.dt:.4f}  converged={bool(sol.converged)}  "
              f"solve={r.wall_time:.2f}s  write={write_time:.2f}s  "
              f"total={r.wall_time + write_time:.2f}s")
        # Voigt order: [11, 22, 33, 12, 13, 23]
        eps_v   = to_voigt(np.asarray(eps_bar_out))
        sigma_v = to_voigt(np.asarray(sigma_bar))
        print(f"           eps_bar   voigt = [{' '.join(f'{v: .4e}' for v in eps_v)}]")
        print(f"           sigma_bar voigt = [{' '.join(f'{v: .4e}' for v in sigma_v)}]")
        stats_rows.append((
            r.step, r.t, r.dt, bool(sol.converged),
            r.wall_time, write_time, eps_v, sigma_v,
        ))

    writer_cm = (IncrementalWriter(stem, grid_shape=n, grid_length=L)
                 if write_fields != "none" else nullcontext(None))
    with writer_cm as w:
        if w is not None and write_fields == "all":
            # step 0 -- undeformed initial condition (t=0), so the output has a
            # zero-strain reference frame before the first accepted increment.
            w.write_increment(0, {
                "phase":        phase_np.reshape(n).astype(np.float32),
                "strain":       np.zeros((*n, 6)),
                "stress":       np.zeros((*n, 6)),
                "von_mises":    np.zeros(n),
                "displacement": np.zeros((*n, 3)),
                "orientation":  orientations_np.T.reshape(*n, 3).astype(np.float64),
            }, time=0.0)

        results = solve_mechanics(
            n, L, phase, materials, eps_bar,
            stepping     = mode,
            formulation  = mcfg.get("formulation", "lippmann_schwinger"),
            scheme       = mcfg.get("scheme", "rotated"),
            control      = control,
            stress_goal  = stress_goal,
            toler_lin    = float(mcfg.get("toler_lin", 1e-6)),
            maxiter      = int(mcfg.get("maxiter", 1000)),
            dt           = scfg.get("dt"),
            dt_init      = float(scfg.get("dt_init", 0.1)),
            dt_min       = float(scfg.get("dt_min", 1e-4)),
            dt_max       = float(scfg.get("dt_max", 0.5)),
            factor_inc   = float(scfg.get("factor_inc", 1.5)),
            factor_dec   = float(scfg.get("factor_dec", 0.5)),
            max_cutbacks = int(scfg.get("max_cutbacks", 5)),
            max_steps    = int(scfg.get("max_steps", 1000)),
            writer       = w,
            orientation  = jnp.array(orientations_np),
            on_increment = _report,
        )

    final_sol = cast(ElasticitySolution, results[-1].solution)
    final = results[-1]
    with h5py.File(stem + ".h5", "a") as f:
        f.attrs["n"] = np.array(n, dtype=int)
        f.attrs["L"] = np.array(L, dtype=float)
        # initial conditions -- the prescribed loading and material setup that
        # produced this solve, so the output file is self-documenting even
        # without the original config.
        f.attrs["input"]       = str(src)
        f.attrs["eps_bar"]     = np.array(eps_bar, dtype=float)
        if name:
            f.attrs["load_case"] = name
        if control is not None:
            f.attrs["control"]    = np.array(control, dtype=int)
            f.attrs["stress_bar"] = np.array(
                stress_goal if stress_goal is not None else np.zeros((3, 3)), dtype=float)
        f.attrs["material_name"] = np.array([getattr(m, "name", "") for m in materials], dtype=object)
        f.attrs["material_repr"] = np.array([repr(m) for m in materials], dtype=object)
        f.attrs["formulation"] = mcfg.get("formulation", "lippmann_schwinger")
        f.attrs["scheme"]      = mcfg.get("scheme", "rotated")
        f.attrs["toler_lin"]   = float(mcfg.get("toler_lin", 1e-6))
        f.attrs["maxiter"]     = int(mcfg.get("maxiter", 1000))
        f.attrs["stepping_mode"] = mode
        f.attrs["converged"]   = bool(final_sol.converged)
        f.attrs["t_final"]     = float(final.t)

    stats = np.array(stats_rows, dtype=_STATS_DTYPE)
    stats_path = f"{stem}_stats.npy"
    np.save(stats_path, stats)

    if write_fields != "none":
        print(f"Written → {stem}.h5 / .xdmf")
    print(f"Written → {stats_path}  (structured array: {_STATS_DTYPE.names})")

    return {
        "name": name or "single", "ok": bool(final_sol.converged),
        "eps_voigt": stats["eps_bar_voigt"][-1], "sig_voigt": stats["sigma_bar_voigt"][-1],
        "wall": float(np.sum(stats["wall_time"])), "stem": stem,
    }


if __name__ == "__main__":
    sys.exit(main())
