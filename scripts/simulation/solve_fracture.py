"""
Phase-field fracture (mechanics + AT2 damage) on a loaded microstructure — YAML-driven.

Loads an existing microstructure (XDMF/HDF5, e.g. from
scripts/generation/generate_weave.py or generate_rve.py) via
utils.io.reader.SimulationReader, then solves the staggered mechanics<->
phase-field problem under a prescribed macroscopic strain via
problems.fracture.solve_fracture_incremental, which also owns writing each
accepted increment to the XDMF/HDF5 output. Geometry/damage-model logic and
solver wiring both live outside this script -- see src/generation/,
src/materialmodels/, src/problems/fracture.py, src/problems/incremental.py.

Only isotropic materials are supported here (the Amor-split driving force
and AT2 residual stiffness gather both read .lam/.mu/.k_res -- see
materialmodels.phasefield.driving_force.lame_field's docstring); a
transversely isotropic fibre like scripts/simulation/solve_mechanics.py's
example can't be used until materialmodels/ grows that support.

``stepping.mode`` selects how the target strain is reached -- see
solve_mechanics.py's docstring, same three modes. ``stepping.dt_step`` (not
``dt``) is the load-fraction increment size for "fixed" -- solve_fracture's
own ``dt`` (fracture.eta/fracture.dt in the config) is the damage
equation's unrelated viscous-regularisation timestep.

Damage/history are carried in from the input file if present (an npz
preprocessor archive with d_init/H_init, see utils.io.reader.SimulationReader)
and threaded across increments internally; both start at zero for a fresh
XDMF/VTU geometry file with no prior damage state.

String values in the YAML support {variable} interpolation:
  output:  "output/simulation"
  jobname: "{fracture.input.stem}"

Usage
-----
    python scripts/simulation/solve_fracture.py configs/simulation/fracture_example.yaml

Output
------
    <output>/<jobname>.h5
    <output>/<jobname>.xdmf
    <output>/<jobname>_stats.npy   -- one structured-array row per accepted
        increment: step, t, dt, converged, converged_mech, converged_helm,
        iter_staggered, err_abs, err_rel, wall_time, write_time,
        eps_bar_voigt (6,), sigma_bar_voigt (6,), d_max -- np.load(path) to read.
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
from post.fields import homogenize, to_voigt
from problems.fracture import FractureSolution, solve_fracture_incremental
from utils.config import load_config
from utils.io.reader import SimulationReader
from utils.io.xdmf_writer import IncrementalWriter


def main():
    parser = argparse.ArgumentParser(
        description="Solve mechanics + AT2 phase-field damage on a loaded microstructure (XDMF/HDF5)"
    )
    parser.add_argument("config", type=Path, help="YAML configuration file")
    args = parser.parse_args()

    cfg  = load_config(args.config)
    fcfg = cfg["fracture"]
    scfg = fcfg.get("stepping", {})
    print(f"Config : {args.config}")

    # ── load microstructure (+ any prior damage state) ───────────────────────
    src = fcfg["input"]
    print(f"Input  : {src}")
    n, L, phase_np, orientations_np, _, vf_np, d_init_np, H_init_np = SimulationReader(src).read()
    phase = jnp.array(phase_np)
    d_init = jnp.array(d_init_np)
    H_init = jnp.array(H_init_np)
    print(f"Grid   : {n}   phi = {float(np.mean(phase_np > 0)):.3f}   "
          f"d_init max = {float(d_init_np.max()):.4f}")

    # ── materials (list, indexed by 0-based phase id; isotropic only -- see
    #    module docstring) ────────────────────────────────────────────────────
    materials = [build_material(m) for m in fcfg["materials"]]
    for i, m in enumerate(materials):
        print(f"  phase {i}: {m}  k_res={m.k_res}  Gc={m.Gc}")

    eps_bar_target = jnp.array(fcfg["eps_bar"], dtype=jnp.float64)

    # mixed strain/stress BC -- formulation="displacement" only (see
    # problems.fracture.solve_fracture docstring).
    control_cfg = fcfg.get("control")
    control = tuple(tuple(int(c) for c in row) for row in control_cfg) if control_cfg else None
    stress_bar_cfg = fcfg.get("stress_bar")
    stress_goal = jnp.array(stress_bar_cfg, dtype=jnp.float64) if stress_bar_cfg else None

    # ── output ────────────────────────────────────────────────────────────────
    output  = cfg["output"]
    jobname = cfg["jobname"]
    stem    = f"{output}/{jobname}"
    Path(output).mkdir(parents=True, exist_ok=True)

    # ── solve (single/fixed/automatic all go through the same API; the writer
    #    is handed in so solve_fracture_incremental writes each accepted
    #    increment itself, as it's produced -- on_increment prints live for
    #    the same reason, instead of only after the whole solve returns) ────
    _STATS_DTYPE = np.dtype([
        ("step", "i4"), ("t", "f8"), ("dt", "f8"), ("converged", "?"),
        ("converged_mech", "?"), ("converged_helm", "?"),
        ("iter_staggered", "i4"), ("err_abs", "f8"), ("err_rel", "f8"),
        ("wall_time", "f8"), ("write_time", "f8"),
        ("eps_bar_voigt", "f8", (6,)), ("sigma_bar_voigt", "f8", (6,)),
        ("d_max", "f8"),
    ])
    stats_rows = []

    def _report(r, write_time):
        sol = cast(FractureSolution, r.solution)
        eps_bar, sigma_bar = homogenize(sol.eps, sol.sigma)
        d_max = float(jnp.max(sol.d))
        print(f"  step {r.step:3d}  t={r.t:.4f}  dt={r.dt:.4f}  converged={bool(sol.converged)}  "
              f"staggered_iters={sol.iter_staggered:3d}  d_max={d_max:.4f}  "
              f"solve={r.wall_time:.2f}s  write={write_time:.2f}s  "
              f"total={r.wall_time + write_time:.2f}s")
        # Voigt order: [11, 22, 33, 12, 13, 23]
        eps_v   = to_voigt(np.asarray(eps_bar))
        sigma_v = to_voigt(np.asarray(sigma_bar))
        print(f"           eps_bar   voigt = [{' '.join(f'{v: .4e}' for v in eps_v)}]")
        print(f"           sigma_bar voigt = [{' '.join(f'{v: .4e}' for v in sigma_v)}]")
        stats_rows.append((
            r.step, r.t, r.dt, bool(sol.converged),
            bool(sol.converged_mech), bool(sol.converged_helm),
            sol.iter_staggered, sol.err_abs, sol.err_rel,
            r.wall_time, write_time, eps_v, sigma_v, d_max,
        ))

    mode = scfg.get("mode", "single")
    with IncrementalWriter(stem, grid_shape=n, grid_length=L) as w:
        # step 0 -- initial condition (t=0): whatever damage state was loaded
        # in (zero for a fresh geometry file), zero strain/stress/displacement.
        step0_fields = {
            "phase":        phase_np.reshape(n).astype(np.float32),
            "strain":       np.zeros((*n, 6)),
            "stress":       np.zeros((*n, 6)),
            "von_mises":    np.zeros(n),
            "displacement": np.zeros((*n, 3)),
            "damage":       d_init_np.reshape(n).astype(np.float64),
            "strain_energy_pos": np.zeros(n),
            "orientation":  orientations_np.T.reshape(*n, 3).astype(np.float64),
        }
        w.write_increment(0, step0_fields, time=0.0)

        results = solve_fracture_incremental(
            n, L, phase, materials, eps_bar_target,
            l0           = float(fcfg["l0"]),
            Gc           = float(fcfg["Gc"]) if fcfg.get("Gc") is not None else None,
            d_init       = d_init,
            H_init       = H_init,
            stepping     = mode,
            formulation  = fcfg.get("formulation", "lippmann_schwinger"),
            scheme       = fcfg.get("scheme", "rotated"),
            control      = control,
            stress_goal  = stress_goal,
            toler_lin    = float(fcfg.get("toler_lin", 1e-6)),
            maxiter_cg   = int(fcfg.get("maxiter_cg", 1000)),
            toler_helm   = float(fcfg.get("toler_helm", 1e-4)),
            maxiter_helm = int(fcfg.get("maxiter_helm", 300)),
            eta          = float(fcfg.get("eta", 0.0)),
            dt           = float(fcfg.get("dt", 1.0)),
            d_thres      = float(fcfg.get("d_thres", 0.95)),
            toler_st_abs = float(fcfg.get("toler_st_abs", 1e-2)),
            toler_st_rel = float(fcfg.get("toler_st_rel", 1e-3)),
            maxiter_st   = int(fcfg.get("maxiter_st", 200)),
            dt_step      = scfg.get("dt_step"),
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

    final_sol = cast(FractureSolution, results[-1].solution)
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
        f.attrs["formulation"] = fcfg.get("formulation", "lippmann_schwinger")
        f.attrs["scheme"]      = fcfg.get("scheme", "rotated")
        f.attrs["l0"]          = float(fcfg["l0"])
        f.attrs["material_Gc"] = np.array([m.Gc for m in materials], dtype=float)
        f.attrs["toler_lin"]   = float(fcfg.get("toler_lin", 1e-6))
        f.attrs["maxiter_cg"]  = int(fcfg.get("maxiter_cg", 1000))
        f.attrs["stepping_mode"] = mode
        f.attrs["converged"]   = bool(final_sol.converged)
        f.attrs["t_final"]     = float(final.t)
        f.attrs["d_max_final"] = float(jnp.max(final_sol.d))

    stats_path = f"{stem}_stats.npy"
    np.save(stats_path, np.array(stats_rows, dtype=_STATS_DTYPE))

    print(f"Written → {stem}.h5 / .xdmf")
    print(f"Written → {stats_path}  (structured array: {_STATS_DTYPE.names})")
    print("Open the .xdmf in ParaView with the 'Xdmf3ReaderT' reader.")


if __name__ == "__main__":
    main()
