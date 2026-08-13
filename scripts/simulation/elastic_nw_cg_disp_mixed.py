"""
Displacement-based elastic FFT load-stepping, driven by a YAML configuration
file, with support for mixed macroscopic strain/stress boundary conditions.

Unlike scripts/simulation/elastic_nw_cg_strain.py (strain-based solver, pure
strain BC only), this drives solvers.mechanical.displacement_nw_cg.ddisp_nw_cg,
which solves directly for the periodic displacement field and lets each
component of the macroscopic strain/stress tensor be independently
strain-controlled (prescribed eps_goal) or stress-controlled (prescribed
stress_goal), selected via loading.control in the YAML config:

    loading:
      control:       # 1 = stress-controlled, 0 = strain-controlled
        - [1, 0, 0]
        - [0, 1, 0]
        - [0, 0, 1]
      eps_goal:      # target strain at t=1 — only entries where control==0 matter
        - [0.0, 0.0, 0.0]
        - [0.0, 0.0, 0.0]
        - [0.0, 0.0, 0.0]
      stress_goal:   # target stress at t=1 — only entries where control==1 matter
        - [50.0, 0.0, 0.0]
        - [0.0,  0.0, 0.0]
        - [0.0,  0.0, 0.0]

Both targets are ramped linearly with the same pseudo-time fraction.

Usage
-----
    python scripts/simulation/elastic_nw_cg_disp_mixed.py configs/elastic_rve_mixed.yaml

Output
------
    <config.output>/<config.jobname>.h5
    <config.output>/<config.jobname>.xdmf
"""

import argparse
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
sys.path.insert(0, "src")

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)
import jax
import jax.numpy as jnp
import numpy as np
import time
from pathlib import Path

from mat_models.elastic    import (LinearElasticIsotropic,
                                   TransverseIsotropicFibre,
                                   assemble_C_field_smooth)
from operators.green       import build_freq_grid
from post.fields           import field_to_grid, von_mises, compute_displacement
from utils.io.xdmf_writer import IncrementalWriter
from post.fields          import to_voigt
from solvers.types         import SolveState, SolverSettings
from solvers.mechanical.displacement_nw_cg import ddisp_nw_cg
from utils.config          import load_config
from utils.io.reader         import SimulationReader

jax.config.update("jax_enable_x64", True)

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="Displacement-based elastic FFT solver — mixed strain/stress BC, config-driven via YAML"
)
parser.add_argument("config", type=Path, help="Path to YAML configuration file")
args = parser.parse_args()

cfg = load_config(args.config)
print(f"Config   : {args.config}")
print(f"Input    : {cfg['input']}")
print(f"Output   : {cfg['output']}/{cfg['jobname']}")

# ── Load geometry ─────────────────────────────────────────────────────────────

n, L, phase_np, orientations_np, _, vf_np, _, _ = SimulationReader(cfg["input"]).read()

Nv   = int(np.prod(n))
phi_ = float(np.mean(phase_np == 1))   # fraction of fibre voxels (binary)

print(f"Using jax-device : {jax.devices()[0].device_kind}")
print(f"Grid    : {n}   Nv = {Nv:,}")
print(f"Domain  : {tuple(f'{v:.4g}' for v in L)}")
print(f"phi_act : {phi_:.4f}")

# ── Materials ─────────────────────────────────────────────────────────────────

_MAT_TYPES = {
    "LinearElasticIsotropic": LinearElasticIsotropic,
    "TransverseIsotropicFibre": TransverseIsotropicFibre,
}

def _build_material(spec: dict):
    cls = _MAT_TYPES[spec["type"]]
    kw  = {k: v for k, v in spec.items() if k != "type"}
    return cls(**kw)

mat_cfg   = cfg["materials"]
materials = [_build_material(mat_cfg["matrix"]),
             _build_material(mat_cfg["fibre"])]

for m in materials:
    print(" ", m)

orientations = jnp.array(orientations_np)   # (3, Nv)

# phi_ind ∈ [0,1]: 0=matrix, 1=fibre centre.
# Vf_yarn=1.0 for solid fibres (RVE); set to tow packing fraction for textiles.
Vf_yarn  = float(cfg.get("Vf_yarn", 1.0))
phi_ind  = jnp.clip(jnp.array(vf_np, dtype=float) / Vf_yarn, 0.0, 1.0)
C_field  = assemble_C_field_smooth(materials, orientations, phi_ind)

# ── Frequency grid ────────────────────────────────────────────────────────────
# (no reference medium / Green's operator needed — ddisp_nw_cg applies the
#  true tangent stiffness directly)

slv     = cfg["solver"]
xi_flat = build_freq_grid(n, L)

# ── Loading ───────────────────────────────────────────────────────────────────

control_np  = np.asarray(cfg["loading"]["control"], dtype=int)
control     = tuple(tuple(int(v) for v in row) for row in control_np)   # static for jit
eps_goal    = jnp.array(cfg["loading"]["eps_goal"],    dtype=float)
stress_goal = jnp.array(cfg["loading"]["stress_goal"], dtype=float)

print(f"control (1=stress, 0=strain):\n{control_np}")

# ── Time-stepper parameters ───────────────────────────────────────────────────

ts           = cfg["timestepper"]
dt_init      = float(ts["dt_init"])
t_end        = float(ts["t_end"])
dt_min       = float(ts["dt_min"])
dt_max       = float(ts["dt_max"])
max_steps    = int(ts["max_steps"])
factor_inc   = float(ts["factor_inc"])
factor_dec   = float(ts["factor_dec"])
max_cutbacks = int(ts["max_cutbacks"])

# ── Solver settings ───────────────────────────────────────────────────────────

jobname = cfg["jobname"]
output  = cfg["output"]

settings = SolverSettings(
    ndim=3,
    n=n,
    L=L,
    toler_lin=float(slv["toler_lin"]),
    toler_nw=float(slv["toler_nw"]),
    maxiter_cg=int(slv["maxiter_cg"]),
    maxiter_nw=int(slv["maxiter_nw"]),
    jobname=jobname,
    output=output,
)
settings.add_load_step(
    control=jnp.array(control_np),
    strain_ave_goal=eps_goal,
    stress_ave_goal=stress_goal,
    timer=(dt_init, t_end, dt_min, dt_max),
)

# ── Initial state ─────────────────────────────────────────────────────────────

zero33 = jnp.zeros((3, 3))
zero_v = jnp.zeros((3, 3, Nv))

state = SolveState(
    strain_loc=zero_v,
    stress_loc=zero_v,
    tangent_glob=C_field,
    strain_ave=zero33,
    stress_ave=zero33,
    strain_ave_inc_goal=zero33,
    stress_ave_inc_goal=zero33,
    Deltastrain_loc=zero_v,
    Deltastress_loc=zero_v,
    stress_loc_goal=zero_v,
    deltastrain_loc=zero_v,
    time=0.0,
    dtime=dt_init,
    kinc=0,
    kstep=1,
    iter_nw=0,
    iter_cg=0,
    info=0,
    bb0n=1.0,
    pnewdt=1.0,
)

# ── Load-stepping loop ────────────────────────────────────────────────────────

os.makedirs(output, exist_ok=True)

dt   = state.dtime
t    = state.time
step = state.kinc

phase_vis   = phase_np.reshape(n).astype(np.float32)
orient_grid = orientations_np.T.reshape(*n, 3).astype(np.float32)
zero_grid   = np.zeros((*n, 6), dtype=np.float64)
zero_scal   = np.zeros(n,       dtype=np.float64)
zero_u      = np.zeros((*n, 3), dtype=np.float64)

with IncrementalWriter(f"{output}/{jobname}", grid_shape=n, grid_length=L) as w:

    w.write_increment(0, {
        "phase":        phase_vis,
        "orientation":  orient_grid,
        "displacement": zero_u,
        "strain":       zero_grid,
        "stress":       zero_grid,
        "von_mises":    zero_scal,
    }, time=0.0)

    while t < t_end and step < max_steps:

        dt = float(np.clip(dt, dt_min, dt_max))
        dt = min(dt, t_end - t)

        converged    = False
        t_step_start = time.perf_counter()

        for attempt in range(max_cutbacks + 1):
            t_frac       = float(t + dt) / t_end
            eps_bar_i    = t_frac * eps_goal
            stress_bar_i = t_frac * stress_goal
            eps_i, sigma_i, delta_i, eps_bar_out_i, conv_mech = ddisp_nw_cg(
                n, C_field, xi_flat, eps_bar_i, control, stress_bar_i,
                toler_lin=settings.toler_lin,
                maxiter=settings.maxiter_cg,
            )
            converged = bool(conv_mech)
            if converged:
                break
            dt = max(dt * factor_dec, dt_min)
            print(f"    cutback #{attempt+1}  dt → {dt:.6f}  (not converged)")

        if not converged:
            raise RuntimeError(
                f"Step {step+1} did not converge after {max_cutbacks} cutbacks "
                f"at t={t:.4f}"
            )

        t    += dt
        step += 1

        state = state._replace(
            strain_loc=eps_i,
            stress_loc=sigma_i,
            deltastrain_loc=delta_i,
            strain_ave=eps_bar_out_i,
            stress_ave=jnp.mean(sigma_i, axis=-1),
            time=t,
            dtime=dt,
            kinc=step,
            info=0 if converged else 1,
        )

        eps_grid   = field_to_grid(state.strain_loc, n)
        sigma_grid = field_to_grid(state.stress_loc, n)
        u_grid     = compute_displacement(state.strain_loc, state.strain_ave, n, L)

        w.write_increment(step, {
            "phase":        phase_vis,
            "orientation":  orient_grid,
            "displacement": u_grid.astype(np.float64),
            "strain":       to_voigt(eps_grid).astype(np.float64),
            "stress":       to_voigt(sigma_grid).astype(np.float64),
            "von_mises":    von_mises(sigma_grid).astype(np.float64),
        }, time=float(t))

        step_time = time.perf_counter() - t_step_start
        print(f"  step {step:2d}  t={t:.4f}  dt={dt:.4f}  "
              f"eps11={float(state.strain_ave[0,0]):.4e}  "
              f"sig11={float(state.stress_ave[0,0]):.3f} MPa  "
              f"time={step_time:.2f}s")

        dt = min(dt * factor_inc, dt_max)

print(f"\nWritten → {output}/{jobname}.h5")
print(f"          {output}/{jobname}.xdmf")
print("Open the .xdmf in ParaView with the 'Xdmf3ReaderT' reader.")
