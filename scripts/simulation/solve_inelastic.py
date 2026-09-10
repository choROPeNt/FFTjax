"""
Elastoplastic homogenization on a loaded microstructure — YAML-driven.

Loads an existing microstructure (XDMF/HDF5, e.g. from
scripts/generation/generate_weave.py or generate_rve.py) via
utils.io.reader.SimulationReader, builds a per-phase local_update via
materialmodels.assembly.assemble_local_update (any mix of plain elastic
phases and plastic phases), then drives a macroscopic strain through a
load / unload / reload cycle via
problems.mechanics.solve_displacement_based_nonlinear -- the Newton-outer/
CG-inner driver a stateful, strain-dependent-tangent material needs (unlike
scripts/simulation/solve_mechanics.py's fixed-C_field linear solve). Plastic
state (eps_p, alpha) is threaded across load steps by this script, never
stored on the material instances -- see materialmodels/inelastic/
plasticity_j2.py and notebooks/in-elastic_J2.ipynb for the underlying model
and its from-scratch demonstration.

The plastic model is chosen per phase in the config, not by this script:
``j2_plasticity`` (pressure-insensitive von Mises) and ``drucker_prager``
(pressure-sensitive, optionally non-associated) share a state convention
and a stress_and_tangent_field surface, so assemble_local_update treats
them interchangeably and swapping one for the other is a config edit --
see configs/simulation/inelastic_j2_example.yaml and
configs/simulation/inelastic_drucker_prager_example.yaml, which differ only
in the matrix phase's material block.

Plastic state is an ``(eps_p, alpha)`` pair per voxel -- see
materialmodels/inelastic/plasticity_j2.py. The ``strain_p`` field written
below is the SCALAR half of it for both models: the accumulated equivalent
plastic strain alpha, which drives hardening. The plastic strain TENSOR
eps_p is written as ``eps_p`` (Voigt 6, same convention as ``strain``), but
only when ``write_plastic_strain_tensor: true`` is set in the config -- off
by default, so existing output layouts are unchanged. Enable it when the
directionality of plastic flow matters, in particular its trace, the plastic
volume change, which alpha does not capture at all: Drucker-Prager's
non-associated flow (a_g > 0) dilates plastically, and that shows up only in
eps_p. scripts/postprocessing/homogenize.py picks the field up automatically
when it is present.

``control``/``stress_bar`` (both optional, omit for pure strain BC) give a
mixed macroscopic strain/stress BC, exactly as in solve_mechanics.py's
config: control's 1-entries mark directions that are stress- rather than
strain-controlled, stress_bar their target stress, and eps_bar is ignored
there. For a pressure-sensitive matrix (drucker_prager) this is not a
cosmetic choice -- clamping the transverse directions confines the matrix
and inflates the tension/compression yield asymmetry, see
configs/simulation/inelastic_drucker_prager_example.yaml.

``loading`` in the YAML controls the cycle:
  component  -- [i, j] macroscopic strain entry driven; default [0, 1]
                (shear). gamma is the ENGINEERING strain: a normal component
                gets eps_bar[i,i] = gamma, a shear one the symmetrized
                eps_bar[i,j] = eps_bar[j,i] = gamma/2 -- so one gamma_max is
                one load magnitude across a sweep of both (see
                problems.loadcases.LoadCase.eps_bar, which notes the one
                config shape this convention changed).
  gamma_max, n_load, n_unload, n_reload -- ramp 0 -> gamma_max -> -gamma_max
                -> gamma_max in that many equal steps each, plus the virgin
                gamma=0 state as step 0.

Several load cases per run
--------------------------
``loading.cases`` runs many load cases sequentially on ONE microstructure in
one process -- the geometry read, the material assembly and the frequency
grid are per-RVE, not per-case, so this is strictly cheaper than a shell
loop over near-identical configs, and there is one config to keep correct
instead of six. ``cases: base6`` is the canonical set (uniaxial x/y/z, shear
xy/xz/yz, in post.fields.to_voigt's Voigt order); a list may name individual
base cases or spell out arbitrary ones (own component, own control/
stress_bar, own cycle or an explicit ``gammas`` path). With
``free_surfaces: true`` each case's control mask is derived per case --
loaded component strain-driven, every unloaded surface traction-free --
which a single fixed ``control`` mask cannot express across six cases. See
problems.loadcases and configs/simulation/inelastic_base6_example.yaml.

Every case starts from virgin plastic state and writes its own output files
(suffixed ``_<case>``); plasticity is path-dependent, so cases are
independent runs that happen to share a microstructure, never a continued
load history. Without ``loading.cases`` the config is a single-case run and
the output paths are unsuffixed, exactly as before.

Writes one XDMF/HDF5 increment per step (including the virgin state), time =
step index -- NOT the applied strain, which reverses direction twice across
the cycle and so isn't a valid (monotonic) XDMF time axis; a viewer such as
ParaView silently mishandles duplicate/non-monotonic time values (verified
the hard way while building the reference notebook).

Usage
-----
    python scripts/simulation/solve_inelastic.py configs/simulation/inelastic_j2_example.yaml
    python scripts/simulation/solve_inelastic.py configs/simulation/inelastic_drucker_prager_example.yaml
    python scripts/simulation/solve_inelastic.py configs/simulation/inelastic_base6_example.yaml
    # one case out of a multi-case config (e.g. one SLURM array task per case):
    python scripts/simulation/solve_inelastic.py <config> --cases shear_xy
    # what would run, without running it:
    python scripts/simulation/solve_inelastic.py <config> --list-cases

Output
------
    <output>/<jobname>[_<case>].h5
    <output>/<jobname>[_<case>].xdmf
        one increment per step, fields: phase, displacement, strain, stress,
        von_mises, strain_p (= alpha), plus eps_p when
        write_plastic_strain_tensor is enabled.
    <output>/<jobname>[_<case>]_stats.npy   -- one structured-array row per step:
        step, gamma, converged, n_iter, wall_time, tau_avg (homogenized
        sigma[i,j]), plus the full homogenized stress and strain tensors as
        sig_00..sig_22 and eps_00..eps_22 -- the latter is what carries the
        strain solved for on stress-controlled directions under mixed BC,
        where eps_bar no longer describes the actual macroscopic strain.
        np.load(path) to read.
"""

import argparse
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, "src")

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)
import jax.numpy as jnp

from materialmodels.assembly import assemble_local_update, describe_materials
from materialmodels.factory import build_material
from operators.green import build_freq_grid
from post.fields import compute_displacement, field_to_grid, to_voigt, von_mises
from problems.loadcases import LoadCase, resolve_cases, select_cases, voigt_label
from problems.mechanics import solve_displacement_based_nonlinear
from utils.config import field_write_mode, load_config
from utils.io.reader import SimulationReader
from utils.io.xdmf_writer import IncrementalWriter

_STATS_DTYPE = np.dtype([
    ("step", "i4"), ("gamma", "f8"), ("converged", "?"), ("n_iter", "i4"),
    ("wall_time", "f8"), ("tau_avg", "f8"),
    # Full homogenized tensors, not just the driven component: under mixed
    # BC the macroscopic strain on stress-controlled directions is solved
    # for rather than prescribed, so eps_bar alone no longer describes the
    # load and the transverse response is the whole point of the run.
    ("sig", "f8", (3, 3)), ("eps", "f8", (3, 3)),
])


def run_case(case: LoadCase, *, stem, n, L, Nv, xi_flat, phase_np, materials,
             local_update, state0, solver, src, write_fields="all",
             write_eps_p=False) -> dict:
    """
    Run one load case's whole strain path and write its XDMF/HDF5, stats and
    metadata. Returns a summary row for the end-of-run table.

    Everything expensive that does NOT depend on the load case (geometry,
    materials, local_update, frequency grid) is built by the caller once and
    passed in; the only per-case state is the plastic state, which starts
    from ``state0`` every time -- see the module docstring.

    A non-converged step ends this case early and is reported in the returned
    summary (``ok``) rather than raised, so a sweep can keep the partial
    output and the caller decides whether to stop; the files written up to
    that point stay valid and complete for the steps they contain.
    """
    i_comp, j_comp = case.i, case.j
    control     = case.control
    stress_goal = None if case.stress_bar is None else jnp.asarray(case.stress_bar)
    # Zero where strain-controlled, one where stress-controlled -- used both to
    # keep eps_bar out of the solved-for directions and to keep the warm start
    # from double-counting them (see the load loop below).
    control_arr = jnp.zeros((3, 3)) if control is None else jnp.asarray(control, dtype=jnp.float64)

    print(f"Case   : {case.describe()}")

    stats_rows = []
    state = state0
    eps_prev_full = jnp.zeros((3, 3, Nv))
    ok, failed_at = True, None
    t_case = time.perf_counter()

    writer_cm = (IncrementalWriter(stem, grid_shape=n, grid_length=L)
                 if write_fields != "none" else nullcontext(None))
    with writer_cm as writer:
        for step, gamma in enumerate(case.gammas):
            t0 = time.perf_counter()
            eps_bar_step = jnp.asarray(case.eps_bar(gamma))

            # The virgin state is the trivial eps = sigma = 0 solution only if
            # nothing is being asked of it -- with a nonzero stress target
            # (a preload) even gamma = 0 is a real solve.
            trivial = (step == 0 and gamma == 0.0
                       and (stress_goal is None or not bool(jnp.any(stress_goal))))
            if trivial:
                eps_step, sigma_step, converged, n_iter = (
                    jnp.zeros((3, 3, Nv)), jnp.zeros((3, 3, Nv)), True, 0,
                )
            else:
                # Baseline carries only the strain-controlled directions --
                # matching what the solver builds internally, so delta_init
                # stays the zero-mean fluctuation it is meant to be.
                eps0_step = jnp.ones((3, 3, Nv)) * (eps_bar_step * (1.0 - control_arr))[:, :, None]
                # warm start: previous step's converged strain minus this step's
                # new baseline -- keeps Newton converging past yield, whichever
                # direction gamma moves (see problems.mechanics's own docstring).
                delta_init = eps_prev_full - eps0_step
                # Mixed-BC half of the same warm start: the previous step's
                # converged macroscopic strain on the stress-controlled
                # directions (e.g. the lateral contraction under axial
                # tension), which is a far better guess than 0 and is what
                # keeps the bordered Newton solve inside its basin. Masked by
                # control_arr so the strain-controlled entries stay with
                # eps0_step and are not counted twice.
                eps_bar_free_init = jnp.mean(eps_prev_full, axis=-1) * control_arr
                eps_step, sigma_step, state, converged, n_iter = solve_displacement_based_nonlinear(
                    n, xi_flat, eps_bar_step, local_update, state,
                    delta_init=delta_init,
                    control=control, stress_goal=stress_goal,
                    eps_bar_free_init=eps_bar_free_init,
                    **solver,
                )
            eps_prev_full = eps_step
            wall_time = time.perf_counter() - t0
            tau_avg = float(jnp.mean(sigma_step[i_comp, j_comp]))
            # The macroscopic strain actually reached -- equal to eps_bar_step
            # under pure strain BC (delta is zero-mean by construction), and
            # the solved-for answer on stress-controlled directions otherwise.
            sig_bar = np.array(jnp.mean(sigma_step, axis=-1))
            eps_bar_out = np.array(jnp.mean(eps_step, axis=-1))

            # Per-case, not a fixed set of components: under mixed BC what is
            # worth watching is this case's own driven strain, the strain
            # SOLVED FOR on its free directions (the lateral contraction, the
            # plastic dilatancy), and how well the stress target on those
            # directions is actually met -- the BC residual, which is the one
            # number that says the bordered system did its job.
            extra = f"  eps_{case.label}={eps_bar_out[i_comp, j_comp]: .5f}"
            if control is not None:
                free = [(a, b) for a in range(3) for b in range(a, 3) if control[a][b]]
                goal = np.zeros((3, 3)) if case.stress_bar is None else np.asarray(case.stress_bar)
                resid = max(abs(sig_bar[a, b] - goal[a, b]) for a, b in free) if free else 0.0
                extra += ("  free: "
                          + " ".join(f"eps_{voigt_label(a, b)}={eps_bar_out[a, b]: .5f}"
                                     for a, b in free)
                          + f"  |sig-goal|max={resid:.1e}")
            print(f"  step {step:3d}  gamma={gamma: .4f}  converged={converged}  "
                  f"n_iter={n_iter:2d}  tau_avg={tau_avg: .4f} MPa  wall={wall_time:.2f}s{extra}")
            stats_rows.append((step, float(gamma), bool(converged), int(n_iter), wall_time,
                               tau_avg, sig_bar, eps_bar_out))

            eps_p_step, alpha_step = state
            # Resolved macroscopic strain, not eps_bar_step: compute_displacement
            # adds the macroscopic part as eps_bar_ij x_j, and under mixed BC
            # eps_bar_step is zero on the stress-controlled directions, which
            # would drop the lateral contraction from the displacement field.
            # Identical to eps_bar_step under pure strain BC.
            if writer is not None and not (step == 0 and write_fields == "increments"):
                eps_grid   = field_to_grid(eps_step, n)
                sigma_grid = field_to_grid(sigma_step, n)
                u_grid = compute_displacement(eps_step, jnp.asarray(eps_bar_out), n, L)
                fields = {
                    "phase":        phase_np.reshape(n).astype(np.float64),
                    "displacement": u_grid.astype(np.float64),
                    "strain":       to_voigt(eps_grid).astype(np.float64),
                    "stress":       to_voigt(sigma_grid).astype(np.float64),
                    "von_mises":    von_mises(sigma_grid).astype(np.float64),
                    "strain_p":     np.array(alpha_step).reshape(n).astype(np.float64),
                }
                if write_eps_p:
                    # mandel=False, the same Voigt convention "strain" above uses,
                    # so eps_p and eps components are directly comparable (and the
                    # trace, the plastic volume change, is just the first three).
                    fields["eps_p"] = to_voigt(field_to_grid(eps_p_step, n)).astype(np.float64)
                # time = step index, not gamma -- see module docstring
                writer.write_increment(step, fields, time=float(step))

            if not converged:
                ok, failed_at = False, step
                print(f"  !! Newton did not converge at step {step}, gamma={gamma:.4f} "
                      f"-- stopping case {case.name!r} here")
                break

    _, alpha_final = state
    n_plastic = int(jnp.sum(alpha_final > 1e-12))
    with h5py.File(stem + ".h5", "a") as f:
        f.attrs["n"] = np.array(n, dtype=int)
        f.attrs["L"] = np.array(L, dtype=float)
        f.attrs["input"]          = str(src)
        f.attrs["material_name"]  = np.array([getattr(m, "name", "") for m in materials], dtype=object)
        f.attrs["material_repr"]  = np.array([repr(m) for m in materials], dtype=object)
        f.attrs["load_case"]      = case.name
        f.attrs["loading_component"] = np.array([i_comp, j_comp], dtype=int)
        f.attrs["gammas"]         = np.asarray(case.gammas, dtype=float)
        f.attrs["gamma_max"]      = case.gamma_max
        if control is not None:
            f.attrs["control"]    = np.array(control, dtype=int)
            f.attrs["stress_bar"] = np.array(
                case.stress_bar if case.stress_bar is not None else np.zeros((3, 3)), dtype=float)
        f.attrs["n_plastic_voxels"] = n_plastic
        f.attrs["max_alpha"]      = float(jnp.max(alpha_final))

    stats = np.array(stats_rows, dtype=_STATS_DTYPE)
    stats_path = f"{stem}_stats.npy"
    np.save(stats_path, stats)

    print(f"  plastic voxels: {n_plastic}/{Nv}   max accumulated plastic strain: "
          f"{float(jnp.max(alpha_final)):.4e}")
    print(f"  Written → {stem}.h5 / .xdmf   ({len(stats_rows)} increments)")
    print(f"  Written → {stats_path}  (structured array: {_STATS_DTYPE.names})")

    return {
        "name": case.name, "label": case.label, "ok": ok, "failed_at": failed_at,
        "n_steps": len(stats_rows), "wall": time.perf_counter() - t_case,
        "tau_peak": float(np.max(np.abs(stats["tau_avg"]))) if len(stats_rows) else 0.0,
        "n_plastic": n_plastic, "stem": stem,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Solve elastoplastic homogenization (J2 or Drucker-Prager, per "
                    "phase, chosen in the config) through a load/unload/reload "
                    "hysteresis cycle on a loaded microstructure (XDMF/HDF5)"
    )
    parser.add_argument("config", type=Path, help="YAML configuration file")
    parser.add_argument("--cases", type=str, default=None,
                        help="comma-separated subset of loading.cases to run, in the given "
                             "order (default: all of them) -- e.g. one case per SLURM array task")
    parser.add_argument("--list-cases", action="store_true",
                        help="print the load cases this config resolves to and exit "
                             "(no geometry read, no solve)")
    parser.add_argument("--keep-going", action="store_true",
                        help="on a non-converged case, carry on with the remaining cases "
                             "instead of stopping (exit status is still nonzero)")
    args = parser.parse_args()

    cfg  = load_config(args.config)
    icfg = cfg["inelastic"]
    lcfg = icfg.get("loading", {})
    print(f"Config : {args.config}")

    # ── load cases ──────────────────────────────────────────────────────────
    # Resolved up front, before anything expensive: a config error here (an
    # unknown case name, a component that its own control marks
    # stress-controlled) should cost nothing to discover.
    try:
        cases = select_cases(
            resolve_cases(lcfg, control=icfg.get("control"), stress_bar=icfg.get("stress_bar")),
            [s.strip() for s in args.cases.split(",")] if args.cases else None,
        )
    except ValueError as exc:
        # A config/CLI mistake, not a crash -- report it as one.
        print(f"error: {exc}", file=sys.stderr)
        return 2
    # Suffix output per case only when the config actually asked for cases --
    # a pre-existing single-case config keeps writing to <jobname>.h5.
    suffixed = "cases" in lcfg
    if args.list_cases:
        for case in cases:
            print(f"  {case.describe()}")
        return 0

    # ── load microstructure ──────────────────────────────────────────────────
    src = icfg["input"]
    input_L  = tuple(icfg["input_L"])  if icfg.get("input_L")  else None
    input_dx = tuple(icfg["input_dx"]) if icfg.get("input_dx") else None
    phase_key = icfg.get("phase_key", "phase")
    orientation_key = icfg.get("orientation_key", "orientation")
    print(f"Input  : {src}")
    n, L, phase_np, orientations_np, _, vf_np, _, _ = SimulationReader(
        src, L=input_L, dx=input_dx, phase_key=phase_key, orientation_key=orientation_key,
    ).read()
    Nv = int(np.prod(n))
    phase = jnp.array(phase_np)
    xi_flat = build_freq_grid(n, L)
    print(f"Grid   : {n}   phi = {float(np.mean(phase_np > 0)):.3f}")

    # ── materials (list, indexed by 0-based phase id; any mix of plain
    #    elastic, J2Plasticity and DruckerPrager models) ─────────────────────
    orientations = jnp.array(orientations_np)
    materials = [build_material(m, orientations=orientations) for m in icfg["materials"]]
    describe_materials(materials)

    local_update, state0 = assemble_local_update(materials, phase)

    solver = dict(
        toler_lin   = float(icfg.get("toler_lin", 1e-7)),
        maxiter_lin = int(icfg.get("maxiter_lin", 2000)),
        toler_nr    = float(icfg.get("toler_nr", 1e-7)),
        maxiter_nr  = int(icfg.get("maxiter_nr", 50)),
    )

    # ── output ────────────────────────────────────────────────────────────────
    output  = cfg["output"]
    jobname = cfg["jobname"]
    Path(output).mkdir(parents=True, exist_ok=True)
    # Per-voxel output is the dominant cost of a large sweep -- see
    # utils.config.field_write_mode.
    write_fields = field_write_mode(icfg.get("write_fields", True))

    # opt-in extra output field: the plastic strain TENSOR eps_p, alongside
    # the scalar "strain_p" (= alpha) always written -- see module docstring.
    write_eps_p = bool(icfg.get("write_plastic_strain_tensor", False))
    if write_eps_p:
        print("Output : writing the plastic strain tensor eps_p (Voigt 6) per increment")

    summaries = []
    for k, case in enumerate(cases, start=1):
        stem = f"{output}/{jobname}_{case.name}" if suffixed else f"{output}/{jobname}"
        print(f"\n── load case {k}/{len(cases)} ─────────────────────────────────────────")
        summaries.append(run_case(
            case, stem=stem, n=n, L=L, Nv=Nv, xi_flat=xi_flat, phase_np=phase_np,
            materials=materials, local_update=local_update, state0=state0,
            solver=solver, src=src, write_fields=write_fields,
            write_eps_p=write_eps_p,
        ))
        if not summaries[-1]["ok"] and not args.keep_going:
            print("\nStopping after a non-converged case (--keep-going runs the rest anyway).")
            break

    if len(cases) > 1:
        print(f"\n{'case':<14s}{'comp':>5s}{'steps':>7s}{'plastic':>9s}"
              f"{'|tau|_max':>11s}{'wall [s]':>10s}  status")
        for s in summaries:
            status = "ok" if s["ok"] else f"FAILED at step {s['failed_at']}"
            print(f"{s['name']:<14s}{s['label']:>5s}{s['n_steps']:>7d}{s['n_plastic']:>9d}"
                  f"{s['tau_peak']:>11.4f}{s['wall']:>10.1f}  {status}")
        not_run = len(cases) - len(summaries)
        if not_run:
            print(f"({not_run} case(s) not run)")

    print("\nOpen the .xdmf files in ParaView with the 'Xdmf3ReaderT' reader.")
    return 0 if all(s["ok"] for s in summaries) and len(summaries) == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
