"""
Elastic benchmark: mode-I tension on a plate with a pre-crack.

Reproduces the elastic loading phase of the benchmark from
    Schneider & Kästner (2024)  https://doi.org/10.1111/ffe.14553

Domain   : 250 × 250 × 1 voxels,  L = [50, 50, 0.2] mm
Material : steel  E = 250 GPa,  ν = 0.3
           void   E ≈ 0 (crack)
Pre-crack: voxels x ∈ [20, 60),  y = 125 (centre),  all z
           — horizontal crack at mid-height, mode-I opening in Y
Load     : uniaxial strain ramp  ε₂₂ → 1.2 × 10⁻³,  10 equal increments

Note on boundary conditions
---------------------------
The reference simulation uses mixed control (ε₂₂ prescribed, σ₁₁ = σ₃₃ = 0).
This script prescribes ε₂₂ with ε₁₁ = ε₃₃ = 0 (pure-strain BC) as a first
step.  Mixed BC support will be added together with the PFF solver.

Usage
-----
    python scripts/benchmark_elastic_tension.py
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["JAX_ENABLE_X64"] = "1"

import sys
sys.path.insert(0, "src")

import jax
import jax.numpy as jnp
import numpy as np
import time

from mat_models.elastic    import LinearElasticIsotropic, assemble_C_field
from operators.green       import build_freq_grid, build_green_operator
from post.fields           import field_to_grid, von_mises, compute_displacement
from post.io               import IncrementalWriter, to_voigt
from solvers.elastic_nw_cg import solve_elastic
from solvers.types         import SolveState, SolverSettings
from mat_models.elastic    import (strain_energy_density,
                                   strain_energy_miehe_split,
                                   strain_energy_amor_split,
                                   lame_from_C_field)

jax.config.update("jax_enable_x64", True)

# ── Grid ──────────────────────────────────────────────────────────────────────

n  = (250, 250, 1)
L  = (50.0, 50.0, 0.2)   # mm
Nv = int(np.prod(n))
dx = tuple(Li / ni for Li, ni in zip(L, n))

# ── Materials ─────────────────────────────────────────────────────────────────

materials = [
    LinearElasticIsotropic(E=210e3, nu=0.3, name="steel"),
    LinearElasticIsotropic(E=1e-6,  nu=0.3, name="void"),   # crack / zero stiffness
]

# ── Microstructure — pre-crack as void voxels ─────────────────────────────────
# Horizontal crack at y-centre, spanning x ∈ [20, 60)
# phase 0 = steel,  phase 1 = void  (0-based, matches materials list)

ms       = np.zeros(n, dtype=int)
j_crack  = n[1] // 2           # y = 125
ms[20:60, j_crack, :] = 1      # void (crack)

phase = jnp.array(ms.ravel())

print(f"Grid     : {n}  (Nv = {Nv})")
print(f"Domain   : {L} mm")
print(f"Crack    : x=[20,60)  y={j_crack}  ({int((ms==1).sum())} voxels)")

# ── Stiffness field ───────────────────────────────────────────────────────────

C_field = assemble_C_field(materials, phase)   # (3,3,3,3, Nv)

# ── Reference medium — steel properties ───────────────────────────────────────
# Void contributes negligible stiffness so the Voigt average ≈ steel.

lam0 = materials[0].lam
mu0  = materials[0].mu
print(f"Reference: lam0={lam0/1e3:.2f} GPa  mu0={mu0/1e3:.2f} GPa")

xi_flat = build_freq_grid(n, L)
G_glob  = build_green_operator(xi_flat, lam0, mu0, scheme='rotated', dx=dx)

# ── Settings ──────────────────────────────────────────────────────────────────

eps_goal = jnp.array([
    [0.0,    0.0, 0.0],
    [0.0, 1.e-3, 0.0],   # ε₂₂ = 1.2 × 10⁻³ (uniaxial Y)
    [0.0,    0.0, 0.0],
])

settings = SolverSettings(
    ndim=3,
    n=n,
    L=L,
    toler_lin=1e-6,
    toler_nw=1e-4,
    maxiter_cg=500,
    maxiter_nw=300,
    jobname="benchmark_elastic_tension",
    output="output",
)
settings.add_load_step(
    control=jnp.zeros((3, 3)),          # pure strain control
    strain_ave_goal=eps_goal,
    stress_ave_goal=jnp.zeros((3, 3)),
    timer=(0.1, 1.0, 0.1, 0.1),        # 10 equal steps of dt=0.1
)

dt_init, t_end, dt_min, dt_max = settings.timer[0]

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

# ── Adaptive time-stepper parameters ─────────────────────────────────────────

factor_inc   = 1.5
factor_dec   = 0.5
max_cutbacks = 5
max_steps    = 20

# ── Load-stepping loop ────────────────────────────────────────────────────────

os.makedirs(settings.output, exist_ok=True)

dt   = state.dtime
t    = state.time
step = state.kinc

# write phase map once as a static numpy array for the writer
phase_grid       = ms.astype(np.float32)
lam_vox, mu_vox = lame_from_C_field(C_field)   # C_field is constant — compute once

with IncrementalWriter(
    f"{settings.output}/{settings.jobname}", grid_shape=n, grid_spacing=dx
) as w:

    # ── write undeformed initial state at t = 0 ───────────────────────────────
    zero_grid  = np.zeros((*n, 6), dtype=np.float64)
    zero_scal  = np.zeros(n,       dtype=np.float64)
    zero_u     = np.zeros((*n, 3), dtype=np.float64)
    w.write_increment(0, {
        "phase":             phase_grid,
        "displacement":      zero_u,
        "strain":            zero_grid,
        "stress":            zero_grid,
        "von_mises":                zero_scal,
        "strain_energy":            zero_scal,
        "strain_energy_pos_miehe":  zero_scal,
        "strain_energy_pos_amor":   zero_scal,
    }, time=0.0)

    while t < t_end and step < max_steps:

        dt = float(np.clip(dt, dt_min, dt_max))
        dt = min(dt, t_end - t)

        converged    = False
        t_step_start = time.perf_counter()

        for attempt in range(max_cutbacks + 1):
            eps_bar_i                     = float(t + dt) * eps_goal
            eps_i, sigma_i, delta_i, iter_mech, conv_mech = solve_elastic(
                n, C_field, G_glob, eps_bar_i,
                toler_lin=settings.toler_lin,
                maxiter=settings.maxiter_cg,
            )
            converged = bool(conv_mech)

            if converged:
                break

            dt = max(dt * factor_dec, dt_min)
            print(f"    cutback #{attempt + 1}  dt → {dt:.4f}  (mech iter={int(iter_mech)})")

        if not converged:
            raise RuntimeError(
                f"Step {step + 1} did not converge after {max_cutbacks} cutbacks "
                f"at t={t:.4f},  dt_min={dt_min}"
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
            info=0 if converged else int(iter_mech),
        )

        eps_grid   = field_to_grid(state.strain_loc, n)
        sigma_grid = field_to_grid(state.stress_loc, n)
        u_grid     = compute_displacement(state.strain_loc, eps_bar_i, xi_flat, n, dx)

        psi_e                  = strain_energy_density(state.strain_loc, C_field)
        psi_pos_miehe, _       = strain_energy_miehe_split(state.strain_loc, lam_vox, mu_vox)
        psi_pos_amor,  _       = strain_energy_amor_split( state.strain_loc, lam_vox, mu_vox)

        psi_e_grid      = np.asarray(psi_e          ).reshape(n)
        psi_miehe_grid  = np.asarray(psi_pos_miehe  ).reshape(n)
        psi_amor_grid   = np.asarray(psi_pos_amor   ).reshape(n)

        w.write_increment(step, {
            "phase":                   phase_grid,
            "displacement":            u_grid.astype(np.float64),
            "strain":                  to_voigt(eps_grid).astype(np.float64),
            "stress":              to_voigt(sigma_grid).astype(np.float64),
            "von_mises":           von_mises(sigma_grid).astype(np.float64),
            "strain_energy":            psi_e_grid.astype(np.float64),
            "strain_energy_pos_miehe":  psi_miehe_grid.astype(np.float64),
            "strain_energy_pos_amor":   psi_amor_grid.astype(np.float64),
        }, time=float(t))

        step_time = time.perf_counter() - t_step_start
        print(
            f"  step {step:2d}  t={t:.3f}  ε₂₂={float(state.strain_ave[1,1]):.2e}  "
            f"σ₂₂={float(state.stress_ave[1,1]):.2f} MPa  "
            f"CG={state.info}  time={step_time:.2f}s"
        )

        dt = min(dt * factor_inc, dt_max)

print(f"\nWritten → {settings.output}/{settings.jobname}.h5")
print(f"          {settings.output}/{settings.jobname}.xdmf")
print("Open the .xdmf in ParaView with the 'Xdmf3ReaderT' reader.")
