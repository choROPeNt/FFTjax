"""
Elastoplastic homogenization on a loaded microstructure — YAML-driven.

Loads an existing microstructure (XDMF/HDF5, e.g. from
scripts/generation/generate_weave.py or generate_rve.py) via
utils.io.reader.SimulationReader, builds a per-phase local_update via
materialmodels.assembly.assemble_local_update (any mix of plain elastic
phases and plastic phases), then drives a macroscopic shear strain
through a load / unload / reload cycle via
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
in the matrix phase's material block. Note the ``strain_p`` field written
below is the accumulated equivalent plastic strain alpha for both.

``loading`` in the YAML controls the cycle:
  component  -- [i, j] macroscopic strain entry driven, symmetrized
                (eps_bar[i,j] = eps_bar[j,i] = gamma/2); default [0, 1] (shear).
  gamma_max, n_load, n_unload, n_reload -- ramp 0 -> gamma_max -> -gamma_max
                -> gamma_max in that many equal steps each, plus the virgin
                gamma=0 state as step 0.

Writes one XDMF/HDF5 increment per step (including the virgin state), time =
step index -- NOT the applied strain, which reverses direction twice across
the cycle and so isn't a valid (monotonic) XDMF time axis; a viewer such as
ParaView silently mishandles duplicate/non-monotonic time values (verified
the hard way while building the reference notebook).

Usage
-----
    python scripts/simulation/solve_inelastic.py configs/simulation/inelastic_j2_example.yaml
    python scripts/simulation/solve_inelastic.py configs/simulation/inelastic_drucker_prager_example.yaml

Output
------
    <output>/<jobname>.h5
    <output>/<jobname>.xdmf
    <output>/<jobname>_stats.npy   -- one structured-array row per step:
        step, gamma, converged, n_iter, wall_time, tau_avg (homogenized
        sigma[i,j]) -- np.load(path) to read.
"""

import argparse
import time
from pathlib import Path

import h5py
import numpy as np

import sys
sys.path.insert(0, "src")

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)
import jax.numpy as jnp

from materialmodels.assembly import assemble_local_update, describe_materials
from materialmodels.factory import build_material
from operators.green import build_freq_grid
from post.fields import compute_displacement, field_to_grid, to_voigt, von_mises
from problems.mechanics import solve_displacement_based_nonlinear
from utils.config import load_config
from utils.io.reader import SimulationReader
from utils.io.xdmf_writer import IncrementalWriter


def main():
    parser = argparse.ArgumentParser(
        description="Solve elastoplastic homogenization (J2 or Drucker-Prager, per "
                    "phase, chosen in the config) through a load/unload/reload "
                    "hysteresis cycle on a loaded microstructure (XDMF/HDF5)"
    )
    parser.add_argument("config", type=Path, help="YAML configuration file")
    args = parser.parse_args()

    cfg  = load_config(args.config)
    icfg = cfg["inelastic"]
    lcfg = icfg.get("loading", {})
    print(f"Config : {args.config}")

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

    # ── load / unload / reload cycle ────────────────────────────────────────
    i_comp, j_comp = lcfg.get("component", [0, 1])
    gamma_max  = float(lcfg.get("gamma_max", 0.03))
    n_load     = int(lcfg.get("n_load", 10))
    n_unload   = int(lcfg.get("n_unload", 15))
    n_reload   = int(lcfg.get("n_reload", 15))
    gammas_load   = np.linspace(0.0, gamma_max, n_load + 1)[1:]
    gammas_unload = np.linspace(gamma_max, -gamma_max, n_unload + 1)[1:]
    gammas_reload = np.linspace(-gamma_max, gamma_max, n_reload + 1)[1:]
    gammas_applied = np.concatenate([[0.0], gammas_load, gammas_unload, gammas_reload])

    toler_lin   = float(icfg.get("toler_lin", 1e-7))
    maxiter_lin = int(icfg.get("maxiter_lin", 2000))
    toler_nr    = float(icfg.get("toler_nr", 1e-7))
    maxiter_nr  = int(icfg.get("maxiter_nr", 50))

    # ── output ────────────────────────────────────────────────────────────────
    output  = cfg["output"]
    jobname = cfg["jobname"]
    stem    = f"{output}/{jobname}"
    Path(output).mkdir(parents=True, exist_ok=True)

    _STATS_DTYPE = np.dtype([
        ("step", "i4"), ("gamma", "f8"), ("converged", "?"), ("n_iter", "i4"),
        ("wall_time", "f8"), ("tau_avg", "f8"),
    ])
    stats_rows = []

    state = state0
    eps_prev_full = jnp.zeros((3, 3, Nv))
    with IncrementalWriter(stem, grid_shape=n, grid_length=L) as writer:
        for step, gamma in enumerate(gammas_applied):
            t0 = time.perf_counter()
            eps_bar_step = jnp.zeros((3, 3)).at[i_comp, j_comp].set(gamma / 2) \
                                             .at[j_comp, i_comp].set(gamma / 2)

            if step == 0:
                # virgin state: eps_bar_step is already zero, the trivial solution.
                eps_step, sigma_step, converged, n_iter = (
                    jnp.zeros((3, 3, Nv)), jnp.zeros((3, 3, Nv)), True, 0,
                )
            else:
                eps0_step = jnp.ones((3, 3, Nv)) * eps_bar_step[:, :, None]
                # warm start: previous step's converged strain minus this step's
                # new baseline -- keeps Newton converging past yield, whichever
                # direction gamma moves (see problems.mechanics's own docstring).
                delta_init = eps_prev_full - eps0_step
                eps_step, sigma_step, state, converged, n_iter = solve_displacement_based_nonlinear(
                    n, xi_flat, eps_bar_step, local_update, state,
                    toler_lin=toler_lin, maxiter_lin=maxiter_lin,
                    toler_nr=toler_nr, maxiter_nr=maxiter_nr,
                    delta_init=delta_init,
                )
                if not converged:
                    raise RuntimeError(f"Newton did not converge at step {step}, gamma={gamma:.4f}")
            eps_prev_full = eps_step
            wall_time = time.perf_counter() - t0
            tau_avg = float(jnp.mean(sigma_step[i_comp, j_comp]))

            print(f"  step {step:3d}  gamma={gamma: .4f}  converged={converged}  "
                  f"n_iter={n_iter:2d}  tau_avg={tau_avg: .4f} MPa  wall={wall_time:.2f}s")
            stats_rows.append((step, float(gamma), bool(converged), int(n_iter), wall_time, tau_avg))

            _, alpha_step = state
            eps_grid   = field_to_grid(eps_step, n)
            sigma_grid = field_to_grid(sigma_step, n)
            u_grid     = compute_displacement(eps_step, eps_bar_step, n, L)
            writer.write_increment(step, {
                "phase":        phase_np.reshape(n).astype(np.float64),
                "displacement": u_grid.astype(np.float64),
                "strain":       to_voigt(eps_grid).astype(np.float64),
                "stress":       to_voigt(sigma_grid).astype(np.float64),
                "von_mises":    von_mises(sigma_grid).astype(np.float64),
                "strain_p":     np.array(alpha_step).reshape(n).astype(np.float64),
            }, time=float(step))  # step index, not gamma -- see module docstring

    _, alpha_final = state
    n_plastic = int(jnp.sum(alpha_final > 1e-12))
    with h5py.File(stem + ".h5", "a") as f:
        f.attrs["n"] = np.array(n, dtype=int)
        f.attrs["L"] = np.array(L, dtype=float)
        f.attrs["input"]          = str(src)
        f.attrs["material_name"]  = np.array([getattr(m, "name", "") for m in materials], dtype=object)
        f.attrs["material_repr"]  = np.array([repr(m) for m in materials], dtype=object)
        f.attrs["loading_component"] = np.array([i_comp, j_comp], dtype=int)
        f.attrs["gamma_max"]      = gamma_max
        f.attrs["n_load"]         = n_load
        f.attrs["n_unload"]       = n_unload
        f.attrs["n_reload"]       = n_reload
        f.attrs["n_plastic_voxels"] = n_plastic
        f.attrs["max_alpha"]      = float(jnp.max(alpha_final))

    stats_path = f"{stem}_stats.npy"
    np.save(stats_path, np.array(stats_rows, dtype=_STATS_DTYPE))

    print(f"\nplastic voxels: {n_plastic}/{Nv}   max accumulated plastic strain: "
          f"{float(jnp.max(alpha_final)):.4e}")
    print(f"Written → {stem}.h5 / .xdmf   ({len(gammas_applied)} increments)")
    print(f"Written → {stats_path}  (structured array: {_STATS_DTYPE.names})")
    print("Open the .xdmf in ParaView with the 'Xdmf3ReaderT' reader.")


if __name__ == "__main__":
    main()
