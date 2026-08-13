"""
Elastic FFT load-stepping driven by a YAML configuration file.

All parameters (input path, materials, solver tolerances, loading, time
stepper) live in the YAML.  String values support {variable} interpolation
so paths and names can be derived from each other:

    input:   output/preprocessed/rve.xdmf
    jobname: "{input.stem}"                  # → "rve"
    output:  "output/simulation/{jobname}"   # → "output/simulation/rve"

Usage
-----
    python scripts/simulation/elastic_nw_cg_strain.py configs/elastic_rve.yaml

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
from operators.green       import build_freq_grid, GreenOperatorBasic, GreenOperatorWillot
from post.fields           import field_to_grid, von_mises, compute_displacement
from utils.io.xdmf_writer import IncrementalWriter
from post.fields          import to_voigt
from solvers.types         import SolveState, SolverSettings
from solvers.elliptic.vector.lippmann_schwinger import solve_lippmann_schwinger
from utils.config          import load_config
from utils.io_read         import read_simulation_input

jax.config.update("jax_enable_x64", True)

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="Elastic FFT solver — config-driven via YAML"
)
parser.add_argument("config", type=Path, help="Path to YAML configuration file")
args = parser.parse_args()

cfg = load_config(args.config)
print(f"Config   : {args.config}")
print(f"Input    : {cfg['input']}")
print(f"Output   : {cfg['output']}/{cfg['jobname']}")

# ── Load geometry ─────────────────────────────────────────────────────────────

n, L, phase_np, orientations_np, _, vf_np, _, _ = read_simulation_input(cfg["input"])

Nv   = int(np.prod(n))
dx   = tuple(Li / ni for Li, ni in zip(L, n))
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

# Reference medium: Voigt average using transverse fibre constants
def _lame(E, nu):
    return E * nu / ((1+nu)*(1-2*nu)), E / (2*(1+nu))

fib = materials[1]
mat = materials[0]
lam_fib, mu_fib = _lame(fib.E_T, fib.nu_TT)
lam0 = phi_ * lam_fib + (1-phi_) * mat.lam
mu0  = phi_ * mu_fib  + (1-phi_) * mat.mu
print(f"Reference : lam0={lam0/1e3:.2f} GPa  mu0={mu0/1e3:.2f} GPa")

# ── Green's operator ──────────────────────────────────────────────────────────

slv  = cfg["solver"]
xi_flat = build_freq_grid(n, L)
scheme = slv.get("scheme", "rotated")
if scheme == "rotated":
    green_op = GreenOperatorWillot(n, L, lam0, mu0, dx)
elif scheme in ("standard", "continuous"):
    green_op = GreenOperatorBasic(n, L, lam0, mu0)
else:
    raise ValueError(f"unknown scheme {scheme!r}, expected 'rotated' or 'standard'/'continuous'")

# ── Loading ───────────────────────────────────────────────────────────────────

eps_goal = jnp.array(cfg["loading"]["eps_goal"], dtype=float)

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
    control=jnp.zeros((3, 3)),
    strain_ave_goal=eps_goal,
    stress_ave_goal=jnp.zeros((3, 3)),
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

with IncrementalWriter(f"{output}/{jobname}", grid_shape=n, grid_spacing=dx) as w:

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
            eps_bar_i = float(t + dt) * eps_goal
            eps_i, sigma_i, delta_i, conv_mech = solve_lippmann_schwinger(
                n, C_field, green_op, eps_bar_i,
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
            strain_ave=jnp.mean(eps_i,   axis=-1),
            stress_ave=jnp.mean(sigma_i, axis=-1),
            time=t,
            dtime=dt,
            kinc=step,
            info=0 if converged else 1,
        )

        eps_grid   = field_to_grid(state.strain_loc, n)
        sigma_grid = field_to_grid(state.stress_loc, n)
        u_grid     = compute_displacement(state.strain_loc, eps_bar_i, xi_flat, n, dx)

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
              f"sig11={float(state.stress_ave[0,0]):.3f} MPa  "
              f"time={step_time:.2f}s")

        dt = min(dt * factor_inc, dt_max)

print(f"\nWritten → {output}/{jobname}.h5")
print(f"          {output}/{jobname}.xdmf")
print("Open the .xdmf in ParaView with the 'Xdmf3ReaderT' reader.")
