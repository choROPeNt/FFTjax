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

Anisotropic materials (e.g. transverse_isotropic, like
scripts/simulation/solve_mechanics.py's example) are usable here too -- the
Amor-split driving force uses an isotropized (lambda, mu) proxy of each
material's elastic_stiffness_tensor() (materialmodels.phasefield.
driving_force.lame_field), exact for an isotropic material, an
approximation for an anisotropic one.

``stepping.mode`` selects how the target strain is reached -- see
solve_mechanics.py's docstring, same three modes. ``stepping.dt_step`` is
the load-fraction increment *size* for "fixed"; the damage equation's own
viscous-regularisation timestep (``fracture.eta``'s partner) isn't a config
value at all -- problems.fracture.solve_fracture_incremental derives it
itself as the real per-increment gap in load fraction, so it's always
correct even under "automatic" stepping's varying step size.

Damage/history are carried in from the input file if present (an npz
preprocessor archive with d_init/H_init, see utils.io.reader.SimulationReader)
and threaded across increments internally; both start at zero for a fresh
XDMF/VTU geometry file with no prior damage state.

Checkpoint / restart
--------------------
``fracture.checkpoint: true`` also writes, per increment, what a restart
needs beyond the usual output: the history field ``history`` (H), the
plastic strain tensor ``eps_p`` and the step size (group attribute ``dt``).
Any increment of such a run -- also one that was killed or is still running
-- can then be replayed by a second run with

    fracture:
      restart:
        file: output/simulation/<jobname of the first run>.h5
        increment: 12
        # dt_init: 0.01   # optional; default: the step the original run tried
        #                 # next, min(dt_12 * factor_inc, dt_max), so that
        #                 # increment 13 reproduces the original's
    jobname: "<a different jobname>"   # the restart file is never overwritten

and otherwise the same config (geometry, materials, solver settings; only
stepping.mode automatic). The run starts from that increment's state and
load fraction, writes it as its own first increment (numbered 12) and
continues with 13, 14, .... Increment 13 matches the original run's to
round-off (test/test_problems_fracture_restart.py).

String values in the YAML support {variable} interpolation:
  output:  "output/simulation"
  jobname: "{fracture.input.stem}"
  stats:   "{jobname}_stats.csv"   # optional, see Output below

Usage
-----
    python scripts/simulation/solve_fracture.py configs/simulation/fracture_example.yaml
    # progress detail (default: the config's top-level log_level, else
    # "increment" -- one line per step); see utils.monitor:
    python scripts/simulation/solve_fracture.py <config> --log-level sweep

Output
------
    <output>/<jobname>.h5
    <output>/<jobname>.xdmf
    <output>/<stats>   -- one row per accepted increment: step, t, dt,
        converged, converged_mech, converged_helm, iter_staggered, err_abs,
        err_rel, wall_time, write_time, eps_bar_voigt (6,),
        sigma_bar_voigt (6,), mises_stress, pressure, damage_mean,
        damage_max (homogenized via post.fields.macroscopic_response).
        ``stats`` is an optional config key, default "<jobname>_stats.npy"
        -- its extension (.npy or .csv) picks the format, see
        utils.io.stats.write_stats.
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

from materialmodels.base import PhaseFieldMaterial
from materialmodels.factory import build_material
from post.fields import homogenize, macroscopic_response
from problems.fracture import solve_fracture_incremental
from problems.utils.loadcases import BASE_COMPONENTS, driven_component, free_surface_control, voigt_label
from solvers.solution import FractureSolution
from utils.config import load_config
from utils.io.reader import SimulationReader, read_fracture_checkpoint, read_increment
from utils.io.stats import write_stats
from utils.io.xdmf_writer import IncrementalWriter
from utils.monitor import add_log_level_argument, get_monitor, monitor_from


def main():
    parser = argparse.ArgumentParser(
        description="Solve mechanics + AT2 phase-field damage on a loaded microstructure (XDMF/HDF5)"
    )
    parser.add_argument("config", type=Path, help="YAML configuration file")
    add_log_level_argument(parser)
    args = parser.parse_args()

    cfg  = load_config(args.config)
    fcfg = cfg["fracture"]
    scfg = fcfg.get("stepping", {})
    print(f"Config : {args.config}")

    # ── load microstructure (+ any prior damage state) ───────────────────────
    # input_L/input_dx are only needed for a metadata-less input (.npy, or a
    # bare .h5 with no n/L attrs) -- every other format (.xdmf/.vtu/.vti/.npz)
    # carries its own grid length and ignores these (with a warning if given
    # anyway). See utils.io.reader.SimulationReader.
    src = fcfg["input"]
    input_L  = tuple(fcfg["input_L"])  if fcfg.get("input_L")  else None
    input_dx = tuple(fcfg["input_dx"]) if fcfg.get("input_dx") else None
    phase_key = fcfg.get("phase_key", "phase")
    orientation_key = fcfg.get("orientation_key", "orientation")
    print(f"Input  : {src}")
    n, L, phase_np, orientations_np, _, vf_np, d_init_np, H_init_np = SimulationReader(
        src, L=input_L, dx=input_dx, phase_key=phase_key, orientation_key=orientation_key,
    ).read()
    phase = jnp.array(phase_np)
    d_init = jnp.array(d_init_np)
    H_init = jnp.array(H_init_np)
    print(f"Grid   : {n}   phi = {float(np.mean(phase_np > 0)):.3f}   "
          f"d_init max = {float(d_init_np.max()):.4f}")

    # ── materials (list, indexed by 0-based phase id; see module docstring) ──
    # cast: build_material() returns the thin ConstitutiveModel ABC (no
    # .k_res/.Gc); every concrete model this factory builds satisfies the
    # richer PhaseFieldMaterial protocol needed for the AT2 damage solve.
    # orientations: passed through so a phase config'd with `fiber_dir:
    # from_input` (transverse_isotropic only) picks up this geometry's own
    # per-voxel orientation field -- ignored by every other config.
    orientations = jnp.array(orientations_np)
    materials = [cast(PhaseFieldMaterial, build_material(m, orientations=orientations))
                 for m in fcfg["materials"]]
    for i, m in enumerate(materials):
        print(f"  phase {i}: {m}  k_res={m.k_res}  Gc={m.Gc}")

    # No hardcoded dtype: this script runs in whatever precision
    # utils.precision selected (JAX_ENABLE_X64), and pinning float64 here made
    # every fp32 run emit a truncation warning per array.
    eps_bar_target = jnp.array(fcfg["eps_bar"])

    # mixed strain/stress BC -- formulation="displacement" only (see
    # problems.fracture.solve_fracture docstring).
    # free_surfaces: true derives the mask the way problems.utils.loadcases does
    # for solve_mechanics/solve_inelastic: the one component eps_bar drives is
    # strain-controlled, every other macroscopic stress (shears included) is
    # held at stress_bar, i.e. zero by default. Same precedence as there: it
    # overrides a run-level `control`.
    control_cfg = fcfg.get("control")
    if fcfg.get("free_surfaces", False):
        control = free_surface_control(*driven_component(fcfg["eps_bar"]))
    else:
        control = tuple(tuple(int(c) for c in row) for row in control_cfg) if control_cfg else None
    if control is not None:
        free = [voigt_label(a, b) for a, b in BASE_COMPONENTS.values() if control[a][b]]  # Voigt order
        print(f"BC     : stress-controlled {', '.join(free) or 'none'}"
              + ("  (free_surfaces)" if fcfg.get("free_surfaces", False) else ""))
    stress_bar_cfg = fcfg.get("stress_bar")
    stress_goal = jnp.array(stress_bar_cfg) if stress_bar_cfg else None

    # ── output ────────────────────────────────────────────────────────────────
    output  = cfg["output"]
    jobname = cfg["jobname"]
    stem    = f"{output}/{jobname}"
    Path(output).mkdir(parents=True, exist_ok=True)
    # stats format is whatever this path's extension says (.npy or .csv) --
    # see utils.io.stats.write_stats. Default unchanged if `stats` is omitted.
    stats_path = Path(output) / cfg.get("stats", f"{jobname}_stats.npy")

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
        ("mises_stress", "f8"), ("pressure", "f8"),
        ("damage_mean", "f8"), ("damage_max", "f8"),
        ("iter_nr", "i4"), ("alpha_max", "f8"),
    ])
    stats_rows = []

    def _report(r, write_time):
        sol = cast(FractureSolution, r.solution)
        eps_bar, sigma_bar = homogenize(sol.eps, sol.sigma)
        resp = macroscopic_response(eps_bar, sigma_bar, scalars={"damage": sol.d})
        extra = ""
        if sol.alpha is not None:
            extra = (f"  nr={sol.iter_nr:2d}  mech={bool(sol.converged_mech)!s:5s}"
                     f"  alpha_max={float(jnp.max(sol.alpha)):.3e}")
        # Fields for the monitor's step line, which adds the clock and the
        # solve/write/total split -- see utils.monitor.
        mon = get_monitor()
        mon.note(f"t={r.t:.4f}  dt={r.dt:.4f}  converged={bool(sol.converged)}  "
                 f"staggered_iters={sol.iter_staggered:3d}  d_max={resp['damage_max']:.4f}{extra}",
                 kind="increment")
        # Voigt order: [11, 22, 33, 12, 13, 23]
        eps_v, sigma_v = resp["eps_bar"], resp["sigma_bar"]
        mon.detail(f"           eps_bar   voigt = [{' '.join(f'{v: .4e}' for v in eps_v)}]", kind="increment")
        mon.detail(f"           sigma_bar voigt = [{' '.join(f'{v: .4e}' for v in sigma_v)}]", kind="increment")
        mon.detail(f"           mises_stress = {resp['mises_stress']:.4e}   pressure = {resp['pressure']:.4e}",
                   kind="increment")
        stats_rows.append((
            r.step, r.t, r.dt, bool(sol.converged),
            bool(sol.converged_mech), bool(sol.converged_helm),
            sol.iter_staggered, sol.err_abs, sol.err_rel,
            r.wall_time, write_time, eps_v, sigma_v,
            resp["mises_stress"], resp["pressure"],
            resp["damage_mean"], resp["damage_max"],
            -1 if sol.iter_nr is None else int(sol.iter_nr),
            0.0 if sol.alpha is None else float(jnp.max(sol.alpha)),
        ))

    # A stateful phase-field material (phasefield_drucker_prager) switches
    # problems.fracture to its Newton mechanical sub-solve. Report it, because
    # it changes which controls matter and which entry points are usable.
    plastic = any(hasattr(m, "driving_force_field") for m in materials)
    if plastic:
        print(f"Solver : plastic phase present -> nonlinear (Newton) mechanical "
              f"sub-solve, toler_nr={float(fcfg.get('toler_nr', 1e-8)):g}, "
              f"maxiter_nr={int(fcfg.get('maxiter_nr', 50))}")
    # Opt-in extra output: the plastic strain TENSOR eps_p (Voigt 6), on top of
    # the scalar strain_p (= alpha) and plastic_work always written for a
    # plastic phase. Same key and meaning as
    # scripts/simulation/solve_inelastic.py's.
    write_eps_p = bool(fcfg.get("write_plastic_strain_tensor", False))
    if write_eps_p and plastic:
        print("Output : writing the plastic strain tensor eps_p (Voigt 6) per increment")

    mode = scfg.get("mode", "single")
    dt_init    = float(scfg.get("dt_init", 0.1))
    dt_max     = float(scfg.get("dt_max", 0.5))
    factor_inc = float(scfg.get("factor_inc", 1.5))

    # ── checkpoint / restart (see the module docstring) ─────────────────────
    checkpoint = bool(fcfg.get("checkpoint", False))
    if checkpoint:
        print("Output : checkpoint -- also writing history (H), eps_p and dt per increment")
    rcfg = fcfg.get("restart")
    restart = None
    state_init = eps_init = None
    t_start, first_step = 0.0, 1
    if rcfg:
        restart_file = Path(rcfg["file"]).with_suffix(".h5")
        restart_inc = int(rcfg["increment"])
        # The writer below opens its file with mode "w": restarting into the
        # run's own output would truncate the checkpoint, and the original run.
        if restart_file.resolve() == Path(stem + ".h5").resolve():
            raise ValueError(
                f"restart.file {restart_file} is this run's own output -- give the "
                "restart run a different jobname (the writer would overwrite it)")
        if mode != "automatic":
            raise ValueError(f"restart needs stepping.mode: automatic, got {mode!r}")
        restart = read_fracture_checkpoint(restart_file, restart_inc, n)
        if restart["phase"] is not None and not np.array_equal(
                restart["phase"].astype(int), np.asarray(phase_np).astype(int)):
            raise ValueError(f"restart.file {restart_file} was run on a different "
                             f"geometry than {src}")
        if (restart["state"] is not None) != plastic:
            raise ValueError(
                f"restart.file {restart_file} {'has' if restart['state'] is not None else 'has no'} "
                f"plastic state, but this config's materials are "
                f"{'' if plastic else 'not '}plastic -- not the same model")
        if restart["t"] >= 1.0:
            raise ValueError(f"increment {restart_inc} of {restart_file} is at t = 1, "
                             "the end of the load path -- nothing left to restart")
        d_init = jnp.array(restart["d"])
        H_init = jnp.array(restart["H"])
        t_start, first_step = restart["t"], restart_inc + 1
        # Step 0's strain is no solved field (zeros): a fresh run starts its
        # first sweep from the uniform eps_bar, so a restart from 0 does too.
        eps_init = jnp.array(restart["eps"]) if t_start > 0.0 else None
        if restart["state"] is not None:
            state_init = tuple(jnp.array(a) for a in restart["state"])
        # The step the original run tried next, so that the next increment
        # reproduces it -- see problems.incremental.solve_automatic.
        if rcfg.get("dt_init") is not None:
            dt_init = float(rcfg["dt_init"])
        elif restart["dt"] is not None:
            dt_init = min(restart["dt"] * factor_inc, dt_max)
        print(f"Restart: {restart_file} increment {restart_inc}  t = {t_start:.6f}  "
              f"d_max = {float(np.max(restart['d'])):.4f}  first dt = {dt_init:.6g}")

    # Older configs ask for per-sweep lines with fracture.log_sweeps: true;
    # that is log level "sweep" now, unless the CLI or log_level says otherwise.
    log_level = args.log_level
    if log_level is None and "log_level" not in cfg and fcfg.get("log_sweeps", False):
        log_level = "sweep"
    with monitor_from(cfg, log_level), IncrementalWriter(stem, grid_shape=n, grid_length=L) as w:
        if restart is not None:
            # The restart state, copied as written, as this run's first
            # increment -- same number and time as in the original run -- so
            # this file is a self-contained continuation, restartable from any
            # of its increments, this one included.
            fields_k, attrs_k = read_increment(restart_file, restart_inc)
            extra = {key: val for key, val in attrs_k.items() if key != "time"}
            w.write_increment(restart_inc, fields_k, time=restart["t"], attrs=extra or None)
        else:
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
            if checkpoint:
                step0_fields["history"] = H_init_np.reshape(n).astype(np.float64)
            # Same field set as the later increments, so ParaView sees one
            # consistent schema across the whole series rather than fields that
            # appear at increment 1.
            if plastic:
                step0_fields["strain_p"] = np.zeros(n)
                step0_fields["plastic_work"] = np.zeros(n)
                if write_eps_p or checkpoint:
                    step0_fields["eps_p"] = np.zeros((*n, 6))
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
            d_thres      = float(fcfg.get("d_thres", 0.95)),
            toler_st_abs = float(fcfg.get("toler_st_abs", 1e-2)),
            toler_st_rel = float(fcfg.get("toler_st_rel", 1e-3)),
            maxiter_st   = int(fcfg.get("maxiter_st", 200)),
            # Newton (mechanical) controls -- only used when a phase is a
            # stateful phase-field material, which is what makes
            # problems.fracture switch to its nonlinear sub-solve. Harmless
            # otherwise: the linear path ignores them.
            toler_nr     = float(fcfg.get("toler_nr", 1e-8)),
            maxiter_nr   = int(fcfg.get("maxiter_nr", 50)),
            # Backtracking line search in Newton (max step halvings); 0 =
            # full steps. See problems.fracture.solve_fracture's line_search.
            line_search  = int(fcfg.get("newton_line_search", 0)),
            # Krylov method of the Newton correction: "cg" (default),
            # "cg_best" (CG returning its best iterate when it hits maxiter_cg)
            # or "gmres" (~1.8x cost).
            linear_solver = str(fcfg.get("linear_solver", "cg")),
            # Damage bounds solved ("solve") or clipped ("clip", default), and
            # what irreversibility refers to: the previous sweep ("sweep",
            # default) or the increment's start ("increment"). See
            # problems.fracture.solve_fracture and sim/sim/14_root_cause.
            damage_bounds   = str(fcfg.get("damage_bounds", "clip")),
            irreversibility = str(fcfg.get("irreversibility", "sweep")),
            write_plastic_strain_tensor = write_eps_p,
            # Staggered accelerator (solvers.damage.anderson). None/"none" is
            # plain Picard, i.e. exactly the behaviour before it existed.
            accelerator  = fcfg.get("accelerator"),
            accel_depth  = int(fcfg.get("accel_depth", 5)),
            accel_beta   = float(fcfg.get("accel_beta", 1.0)),
            # Ceiling on damage. 1.0 = unmodified AT2; below 1 keeps the
            # degraded tangent non-degenerate at the cost of a crack that
            # never fully opens. See problems.fracture.solve_fracture.
            d_cap        = float(fcfg.get("d_cap", 1.0)),
            dt_step      = scfg.get("dt_step"),
            dt_init      = dt_init,
            dt_min       = float(scfg.get("dt_min", 1e-4)),
            dt_max       = dt_max,
            factor_inc   = factor_inc,
            factor_dec   = float(scfg.get("factor_dec", 0.5)),
            max_cutbacks = int(scfg.get("max_cutbacks", 5)),
            max_steps    = int(scfg.get("max_steps", 1000)),
            writer       = w,
            orientation  = jnp.array(orientations_np),
            on_increment = _report,
            # Checkpoint / restart (see the module docstring): the state a
            # restart starts from, None/0/1 for a fresh run.
            state_init   = state_init,
            eps_init     = eps_init,
            t_start      = t_start,
            first_step   = first_step,
            write_checkpoint = checkpoint,
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
        if checkpoint:
            f.attrs["checkpoint"] = True
        if restart is not None:
            f.attrs["restart_file"]      = str(restart_file)
            f.attrs["restart_increment"] = restart_inc
        f.attrs["converged"]   = bool(final_sol.converged)
        f.attrs["t_final"]     = float(final.t)
        f.attrs["d_max_final"] = float(jnp.max(final_sol.d))

    write_stats(stats_path, _STATS_DTYPE, stats_rows)

    print(f"Written → {stem}.h5 / .xdmf")
    print(f"Written → {stats_path}  ({stats_path.suffix[1:]}: {_STATS_DTYPE.names})")
    print("Open the .xdmf in ParaView with the 'Xdmf3ReaderT' reader.")


if __name__ == "__main__":
    main()
