"""
Elastic FFT load-stepping script.

Builds a 2-phase composite RVE, applies a macroscopic strain ramp, and writes
all field increments (strain, stress, displacement, von Mises) to HDF5 + XDMF
for ParaView.

Usage
-----
    python scripts/elastic_nw_cg_strain.py

Output
------
    output/rve_loadsteps.h5
    output/rve_loadsteps.xdmf
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["JAX_ENABLE_X64"] = "1"

import sys
sys.path.insert(0, "src")

import jax
import jax.numpy as jnp
import time
import numpy as np

from generation.rve              import make_square_composite_rve
from mat_models.elastic          import (LinearElasticIsotropic,
                                         TransverseIsotropicFibre,
                                         assemble_C_field_oriented)
from operators.green             import build_freq_grid, build_green_operator
from post.fields                 import field_to_grid, von_mises, compute_displacement
from post.io                     import IncrementalWriter, to_voigt
from solvers.types               import SolveState, SolverSettings
from solvers.elastic_nw_cg      import solve_elastic

jax.config.update("jax_enable_x64", True)


# ── RVE setup ─────────────────────────────────────────────────────────────────

phi      = 0.6
r_fib_um = 3.5
vox_um   = 0.1
nz       = 10

materials = [
    LinearElasticIsotropic(E=3.5e3, nu=0.35, name="epoxy matrix"),
    # Transversely isotropic carbon fibre — reference axis: local Z = [0,0,1]
    # 5 constants from textbook / datasheet (all in MPa):
    #   E_L=230 GPa  E_T=15 GPa  G_LT=15 GPa  ν_LT=0.20  ν_TT=0.30
    TransverseIsotropicFibre(
        E_L=230e3, E_T=15e3, G_LT=15e3, nu_LT=0.20, nu_TT=0.30,
        name="carbon fibre",
    ),
]


phase_np, N, n, L, phi_ = make_square_composite_rve(phi, r_fib_um, vox_um, nz=nz)
phase = jnp.array(phase_np.ravel())
Nv    = int(np.prod(n))

# ── Per-voxel fibre orientation field ────────────────────────────────────────
# orientations[k, v] = component k of the unit fibre direction at voxel v.
# Only fibre voxels (phase == 1) carry a meaningful direction; matrix voxels
# are left as zero vectors (ignored by assemble_C_field_oriented for phase 0).
fiber_mask   = phase == 1                                # (Nv,) bool
orientations = jnp.zeros((3, Nv)).at[2, fiber_mask].set(1.0)   # Z only in fibres

# Example: tilt fibre phase by 10° toward X  (uncomment to try)
# theta = jnp.deg2rad(10.0)
# orientations = (jnp.zeros((3, Nv))
#                 .at[0, fiber_mask].set(jnp.sin(theta))
#                 .at[2, fiber_mask].set(jnp.cos(theta)))

C_field = assemble_C_field_oriented(materials, phase, orientations)

# Reference medium: volume-fraction (Voigt) average.
# For the transversely isotropic fibre we use its *transverse* Lamé constants
# (E_T, nu_TT) because the cross-section behaviour governs in-plane homogenisation.
def _lame_from_E_nu(E, nu):
    return E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))

fib  = materials[1]
lam_fib, mu_fib = _lame_from_E_nu(fib.E_T, fib.nu_TT)
mat  = materials[0]
lam0 = phi_ * lam_fib + (1 - phi_) * mat.lam
mu0  = phi_ * mu_fib  + (1 - phi_) * mat.mu

xi_flat = build_freq_grid(n, L)
G_glob  = build_green_operator(xi_flat, lam0, mu0)
dx      = tuple(float(Li / ni) for Li, ni in zip(L, n))

print(f"Using jax-device: {jax.devices()[0].device_kind}")
print(f"RVE: N={N}  nz={nz}  phi={phi_:.3f}")
print(f"Reference medium: lam0={lam0/1e3:.2f} GPa  mu0={mu0/1e3:.2f} GPa")
for m in materials:
    print(" ", m)

# ── Settings ──────────────────────────────────────────────────────────────────

eps_goal = jnp.array([
    [1.0e-3,  0.0, 0.0],
    [0.0, 0.0,  0.0],
    [0.0,  0.0,  0.0],
])  # pure shear  (ε̄₁₂ = ε̄₂₁ = 1×10⁻³ at t = 1)

settings = SolverSettings(
    ndim=3,
    n=n,
    L=L,
    toler_lin=1e-4,
    toler_nw=1e-7,
    maxiter_cg=1000,
    maxiter_nw=6,
    jobname="rve_loadsteps",
    output="output",
)
settings.add_load_step(
    control=jnp.zeros((3, 3)),           # 0 everywhere = pure strain control
    strain_ave_goal=eps_goal,
    stress_ave_goal=jnp.zeros((3, 3)),
    timer=(0.1, 1.0, 0.05, 0.2),        # (dt_init, t_end, dt_min, dt_max)
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
factor_inc   = 1.5   # multiply dt by this after a converged step
factor_dec   = 0.5   # multiply dt by this after a failed attempt
max_cutbacks = 5     # max dt reductions per step before aborting
max_steps    = 20

# ── Load-stepping loop ────────────────────────────────────────────────────────

os.makedirs(settings.output, exist_ok=True)

dt   = state.dtime
t    = state.time
step = state.kinc

orient_grid = np.asarray(orientations).T.reshape(*n, 3).astype(np.float32)

with IncrementalWriter(f"{settings.output}/{settings.jobname}", grid_shape=n, grid_spacing=dx) as w:

    # ── write undeformed initial state at t = 0 ───────────────────────────────
    zero_grid = np.zeros((*n, 6), dtype=np.float64)
    zero_scal = np.zeros(n,       dtype=np.float64)
    zero_u    = np.zeros((*n, 3), dtype=np.float64)
    w.write_increment(0, {
        "phase":        phase_np.astype(np.float32),
        "orientation":  orient_grid,
        "displacement": zero_u,
        "strain":       zero_grid,
        "stress":       zero_grid,
        "von_mises":    zero_scal,
    }, time=0.0)

    while t < t_end and step < max_steps:

        dt = float(np.clip(dt, dt_min, dt_max))
        dt = min(dt, t_end - t)

        # ── attempt the increment, cut back on failure ──────────────────────
        converged    = False
        t_step_start = time.perf_counter()
        for attempt in range(max_cutbacks + 1):
            eps_bar_i                     = float(t + dt) * eps_goal
            eps_i, sigma_i, delta_i, info = solve_elastic(
                n, C_field, G_glob, eps_bar_i,
                toler_lin=settings.toler_lin,
                maxiter=settings.maxiter_cg,
            )
            converged = (info is None or info == 0)

            if converged:
                break

            dt = max(dt * factor_dec, dt_min)
            print(f"    cutback #{attempt + 1}  dt → {dt:.6f}  (CG info={info})")

        if not converged:
            raise RuntimeError(
                f"Step {step + 1} did not converge after {max_cutbacks} cutbacks "
                f"at t={t:.4f}, min_dt={dt_min}"
            )

        # ── accept the increment ────────────────────────────────────────────
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
            info=0 if converged else info,
        )

        eps_grid   = field_to_grid(state.strain_loc, n)
        sigma_grid = field_to_grid(state.stress_loc, n)
        u_grid     = compute_displacement(state.strain_loc, eps_bar_i, xi_flat, n, dx)

        w.write_increment(step, {
            "phase":        phase_np.astype(np.float32),
            "orientation":  orient_grid,
            "displacement": u_grid.astype(np.float64),
            "strain":       to_voigt(eps_grid).astype(np.float64),
            "stress":       to_voigt(sigma_grid).astype(np.float64),
            "von_mises":    von_mises(sigma_grid).astype(np.float64),
        }, time=float(t))

        step_time = time.perf_counter() - t_step_start
        print(f"  step {step:2d}  t={t:.4f}  dt={dt:.4f}  "
              f"sig12={float(state.stress_ave[0, 1]):.3f} MPa  CG={state.info}  "
              f"time={step_time:.2f}s")

        dt = min(dt * factor_inc, dt_max)

print(f"\nWritten → {settings.output}/{settings.jobname}.h5")
print(f"          {settings.output}/{settings.jobname}.xdmf")
print("Open the .xdmf in ParaView with the 'Xdmf3ReaderT' reader.")
