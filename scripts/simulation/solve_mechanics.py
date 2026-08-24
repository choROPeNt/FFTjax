"""
Linear-elastic mechanical homogenization on a loaded microstructure — YAML-driven.

Loads an existing microstructure (XDMF/HDF5, e.g. from
scripts/generation/generate_weave.py or generate_rve.py) via
utils.io.reader.SimulationReader, then solves the mechanical equilibrium
problem under a prescribed macroscopic strain via
problems.mechanics.solve_mechanics_incremental, which also owns writing each
accepted increment to the XDMF/HDF5 output. Geometry logic and solver
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

String values in the YAML support {variable} interpolation:
  output:  "output/simulation"
  jobname: "{mechanics.input.stem}"

Usage
-----
    python scripts/simulation/solve_mechanics.py configs/simulation/mechanics_example.yaml

Output
------
    <output>/<jobname>.h5
    <output>/<jobname>.xdmf
"""

import argparse
import sys
from pathlib import Path
from typing import cast

import h5py
import numpy as np

sys.path.insert(0, "src")

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)
import jax.numpy as jnp

from materialmodels.factory import build_material
from problems.mechanics import solve_mechanics_incremental
from solvers.elliptic.vector.base import ElasticitySolution
from utils.config import load_config
from utils.io.reader import SimulationReader
from utils.io.xdmf_writer import IncrementalWriter


def main():
    parser = argparse.ArgumentParser(
        description="Solve linear-elastic homogenization on a loaded microstructure (XDMF/HDF5)"
    )
    parser.add_argument("config", type=Path, help="YAML configuration file")
    args = parser.parse_args()

    cfg  = load_config(args.config)
    mcfg = cfg["mechanics"]
    scfg = mcfg.get("stepping", {})
    print(f"Config : {args.config}")

    # ── load microstructure ──────────────────────────────────────────────────
    src = mcfg["input"]
    print(f"Input  : {src}")
    n, L, phase_np, orientations_np, _, vf_np, _, _ = SimulationReader(src).read()
    phase = jnp.array(phase_np)
    print(f"Grid   : {n}   phi = {float(np.mean(phase_np > 0)):.3f}")

    # ── materials (list, indexed by 0-based phase id) ───────────────────────
    materials = [build_material(m) for m in mcfg["materials"]]
    for i, m in enumerate(materials):
        print(f"  phase {i}: {m}")

    eps_bar_target = jnp.array(mcfg["eps_bar"], dtype=jnp.float64)

    # mixed strain/stress BC -- formulation="displacement" only (see
    # problems.mechanics.solve_mechanics docstring). control's 1-entries mark
    # which macroscopic directions are stress- rather than strain-controlled;
    # stress_bar gives their target stress, eps_bar is ignored there.
    control_cfg = mcfg.get("control")
    control = tuple(tuple(int(c) for c in row) for row in control_cfg) if control_cfg else None
    stress_bar_cfg = mcfg.get("stress_bar")
    stress_goal = jnp.array(stress_bar_cfg, dtype=jnp.float64) if stress_bar_cfg else None

    # ── output ────────────────────────────────────────────────────────────────
    output  = cfg["output"]
    jobname = cfg["jobname"]
    stem    = f"{output}/{jobname}"
    Path(output).mkdir(parents=True, exist_ok=True)

    # ── solve (single/fixed/automatic all go through the same API; the writer
    #    is handed in so solve_mechanics_incremental writes each accepted
    #    increment itself, as it's produced -- on_increment prints live for
    #    the same reason, instead of only after the whole solve returns) ────
    def _report(r, write_time):
        sol = cast(ElasticitySolution, r.solution)
        print(f"  step {r.step:3d}  t={r.t:.4f}  dt={r.dt:.4f}  converged={bool(sol.converged)}  "
              f"solve={r.wall_time:.2f}s  write={write_time:.2f}s  "
              f"total={r.wall_time + write_time:.2f}s")

    mode = scfg.get("mode", "single")
    with IncrementalWriter(stem, grid_shape=n, grid_length=L) as w:
        results = solve_mechanics_incremental(
            n, L, phase, materials, eps_bar_target,
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
        f.attrs["eps_bar"]     = np.array(eps_bar_target, dtype=float)
        if control is not None:
            f.attrs["control"]    = np.array(control, dtype=int)
            f.attrs["stress_bar"] = np.array(stress_goal, dtype=float)
        f.attrs["material_name"] = np.array([getattr(m, "name", "") for m in materials], dtype=object)
        f.attrs["material_repr"] = np.array([repr(m) for m in materials], dtype=object)
        f.attrs["formulation"] = mcfg.get("formulation", "lippmann_schwinger")
        f.attrs["scheme"]      = mcfg.get("scheme", "rotated")
        f.attrs["toler_lin"]   = float(mcfg.get("toler_lin", 1e-6))
        f.attrs["maxiter"]     = int(mcfg.get("maxiter", 1000))
        f.attrs["stepping_mode"] = mode
        f.attrs["converged"]   = bool(final_sol.converged)
        f.attrs["t_final"]     = float(final.t)

    print(f"Written → {stem}.h5 / .xdmf")
    print("Open the .xdmf in ParaView with the 'Xdmf3ReaderT' reader.")


if __name__ == "__main__":
    main()
