"""
Phase-field fracture on a fibre-reinforced composite RVE.

Extends elastic_nw_cg_strain.py with the AT2 staggered PFF solver,
using all settings established in benchmark_pff_tension.py:
  - Willot rotated Green's operator
  - Hybrid irreversibility (dthres = 0.95, Steinke & Kaliske)
  - Viscous regularisation η = 1e-6  (Schneider & Kästner 2025, Fig. 3b)
  - Miehe spectral split for crack driving force ψ⁺

Reference : Schneider & Kästner (2025) https://doi.org/10.1111/ffe.14553

Units     : MPa, µm  →  Gc in MPa·µm = J/m²,  l₀ in µm

Usage
-----
    python scripts/pff_nw_cg_strain.py

Output
------
    output/rve_pff.h5
    output/rve_pff.xdmf
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

from generation.rve       import make_square_composite_rve
from mat_models.elastic   import (LinearElasticIsotropic, TransverseIsotropicFibre,
                                   assemble_C_field_oriented,
                                   strain_energy_miehe_split, lame_from_C_field)
from operators.green      import build_freq_grid, build_green_operator
from post.fields          import field_to_grid, von_mises, compute_displacement
from post.io              import IncrementalWriter, to_voigt
from solvers.anderson     import AndersonAccelerator
from solvers.elastic_nw_cg import solve_elastic
from solvers.pff_damage   import (degradation, update_history,
                                   solve_helmholtz_cg)
from solvers.types        import SolveState, SolverSettings

jax.config.update("jax_enable_x64", True)

# ── RVE geometry ──────────────────────────────────────────────────────────────

phi      = 0.6      # fibre volume fraction
r_fib_um = 3.5      # fibre radius  [µm]
vox_um   = 0.1      # voxel size    [µm]
nz       = 10       # voxels in z

# ── Materials (MPa) ───────────────────────────────────────────────────────────

materials = [
    LinearElasticIsotropic(E=3.5e3, nu=0.35, name="epoxy matrix"),
    TransverseIsotropicFibre(
        E_L=230e3, E_T=15e3, G_LT=15e3, nu_LT=0.20, nu_TT=0.30,
        name="carbon fibre",
    ),
]

phase_np, N, n, L, phi_ = make_square_composite_rve(phi, r_fib_um, vox_um, nz=nz)
phase = jnp.array(phase_np.ravel())
Nv    = int(np.prod(n))
dx    = tuple(float(Li / ni) for Li, ni in zip(L, n))

# ── Fibre orientations ────────────────────────────────────────────────────────

fiber_mask   = phase == 1
orientations = jnp.zeros((3, Nv)).at[2, fiber_mask].set(1.0)   # fibres along Z

C_field         = assemble_C_field_oriented(materials, phase, orientations)
lam_vox, mu_vox = lame_from_C_field(C_field)   # undegraded — fixed

# ── Reference medium (Voigt average of transverse moduli) ─────────────────────

def _lame(E, nu):
    return E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))

fib  = materials[1]
lam_fib, mu_fib = _lame(fib.E_T, fib.nu_TT)
mat  = materials[0]
lam0 = phi_ * lam_fib + (1 - phi_) * mat.lam
mu0  = phi_ * mu_fib  + (1 - phi_) * mat.mu

xi_flat = build_freq_grid(n, L)
G_glob  = build_green_operator(xi_flat, lam0, mu0, scheme='rotated', dx=dx)

print(f"Device    : {jax.devices()[0].device_kind}")
print(f"RVE       : N={N}  nz={nz}  φ={phi_:.3f}  Nv={Nv}")
print(f"Reference : lam0={lam0/1e3:.2f} GPa  mu0={mu0/1e3:.2f} GPa")
for m in materials:
    print(" ", m)

# ── PFF parameters ────────────────────────────────────────────────────────────
# Units: MPa·µm = J/m².  Epoxy Gc ≈ 50–150 J/m².  l₀ ≥ 4 × vox_um = 0.4 µm.

l0     = 0.4     # µm   phase-field length scale
Gc     = 100.0    # MPa·µm = J/m²  (epoxy matrix fracture energy)
dthres = 0.95    # hybrid irreversibility threshold
eta    = 1e-6    # numerical viscosity (paper Figure 3b)

print(f"PFF       : l₀={l0} µm  Gc={Gc} MPa·µm  dthres={dthres}  η={eta:.0e}")

# ── Macroscopic load ──────────────────────────────────────────────────────────

eps_goal = jnp.array([
    [1.0e-2, 0.0, 0.0],
    [0.0,    0.0, 0.0],
    [0.0,    0.0, 0.0],
])  # uniaxial ε₁₁

# ── Solver settings ───────────────────────────────────────────────────────────

settings = SolverSettings(
    ndim=3,
    n=n,
    L=L,
    toler_lin=1e-4,
    toler_nw=1e-4,
    maxiter_cg=500,
    maxiter_nw=300,
    jobname="rve_pff",
    output="output",
)
settings.add_load_step(
    control=jnp.zeros((3, 3)),
    strain_ave_goal=eps_goal,
    stress_ave_goal=jnp.zeros((3, 3)),
    timer=(0.05, 1.0, 0.005, 0.1),    # (dt_init, t_end, dt_min, dt_max)
)

dt_init, t_end, dt_min, dt_max = settings.timer[0]

# ── Staggered scheme parameters (Schneider & Kästner 2025) ───────────────────

toler_st_abs = 1e-2    # εa
toler_st_rel = 1e-3    # εr
maxiter_st   = 200
toler_helm   = 1e-3
maxiter_helm = 300
# Anderson mixing for the staggered fixed-point (matches reference PETSc Anderson).
anderson_depth = 5
anderson_beta  = 1.0

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

d_field = jnp.zeros((Nv,))   # damage field
H_field = jnp.zeros((Nv,))   # history variable H = max(ψ⁺)

# ── Adaptive time-stepper parameters ─────────────────────────────────────────

factor_inc   = 1.5
factor_dec   = 0.5
max_cutbacks = 5
max_steps    = 200

# ── Load-stepping loop ────────────────────────────────────────────────────────

os.makedirs(settings.output, exist_ok=True)

dt   = state.dtime
t    = state.time
step = state.kinc

orient_grid = np.asarray(orientations).T.reshape(*n, 3).astype(np.float32)
phase_grid  = phase_np.astype(np.float32)
zero_grid   = np.zeros((*n, 6), dtype=np.float64)
zero_scal   = np.zeros(n,       dtype=np.float64)
zero_u      = np.zeros((*n, 3), dtype=np.float64)

with IncrementalWriter(
    f"{settings.output}/{settings.jobname}", grid_shape=n, grid_spacing=dx
) as w:

    # ── initial undeformed state ──────────────────────────────────────────────
    w.write_increment(0, {
        "phase":             phase_grid,
        "orientation":       orient_grid,
        "displacement":      zero_u,
        "strain":            zero_grid,
        "stress":            zero_grid,
        "von_mises":         zero_scal,
        "damage":            zero_scal,
        "strain_energy_pos": zero_scal,
    }, time=0.0)

    while t < t_end and step < max_steps:

        dt = float(np.clip(dt, dt_min, dt_max))
        dt = min(dt, t_end - t)

        eps_bar_i      = float(t + dt) * eps_goal
        converged_mech = False
        t_step_start   = time.perf_counter()

        for attempt in range(max_cutbacks + 1):

            # ── staggered loop (Anderson-accelerated fixed point) ─────────────
            d_st = d_field
            H_st = H_field
            accel = AndersonAccelerator(depth=anderson_depth, beta=anderson_beta)

            for iter_st in range(1, maxiter_st + 1):
                d_in = d_st     # current iterate xₖ

                # 1. degrade stiffness
                g     = degradation(d_in)
                C_eff = g[None, None, None, None, :] * C_field

                # 2. mechanical CG solve
                eps_i, sigma_i, delta_i, iter_mech, conv_mech = solve_elastic(
                    n, C_eff, G_glob, eps_bar_i,
                    toler_lin=settings.toler_lin,
                    maxiter=settings.maxiter_cg,
                )

                # 3. crack driving force — undegraded Miehe spectral split
                psi_pos, _ = strain_energy_miehe_split(eps_i, lam_vox, mu_vox)

                # 4. hybrid history update
                H_st = update_history(H_st, psi_pos)

                # 5. Helmholtz CG solve for damage → G(d_in)
                d_out, iter_helm, conv_helm = solve_helmholtz_cg(
                    H_st, xi_flat, n, l0, Gc, d_in,
                    toler_cg=toler_helm,
                    maxiter=maxiter_helm,
                    eta=eta,
                    dt=dt,
                )

                # 6. staggered convergence — on the TRUE residual G(d_in) − d_in
                diff    = jnp.max(jnp.abs(d_out - d_in))
                err_abs = float(diff)
                err_rel = float(diff / (jnp.max(jnp.abs(d_out)) + 1e-30))
                if err_abs < toler_st_abs or err_rel < toler_st_rel:
                    d_st = d_out          # accept the consistent fixed-point value
                    break

                # 7. Anderson-accelerated next iterate, projected onto the
                #    admissible set: irreversibility floor d ≥ d_field, d ∈ [0,1]
                d_st = accel.step(d_in, d_out)
                d_st = jnp.clip(jnp.maximum(d_field, d_st), 0.0, 1.0)

            converged_mech = bool(conv_mech)
            if converged_mech:
                break

            dt = max(dt * factor_dec, dt_min)
            print(f"    cutback #{attempt + 1}  dt → {dt:.5f}  "
                  f"(mech iter={int(iter_mech)})")

        if not converged_mech:
            raise RuntimeError(
                f"Step {step + 1}: mechanical CG did not converge after "
                f"{max_cutbacks} cutbacks at t={t:.4f}"
            )

        # ── accept increment ──────────────────────────────────────────────────
        t       += dt
        step    += 1
        d_field  = d_st
        H_field  = H_st

        state = state._replace(
            strain_loc=eps_i,
            stress_loc=sigma_i,
            deltastrain_loc=delta_i,
            strain_ave=jnp.mean(eps_i,   axis=-1),
            stress_ave=jnp.mean(sigma_i, axis=-1),
            time=t,
            dtime=dt,
            kinc=step,
            info=0 if converged_mech else int(iter_mech),
        )

        eps_grid   = field_to_grid(state.strain_loc, n)
        sigma_grid = field_to_grid(state.stress_loc, n)
        u_grid     = compute_displacement(state.strain_loc, eps_bar_i, xi_flat, n, dx)
        d_grid     = np.asarray(d_field).reshape(n)
        psi_grid   = np.asarray(psi_pos).reshape(n)

        w.write_increment(step, {
            "phase":             phase_grid,
            "orientation":       orient_grid,
            "displacement":      u_grid.astype(np.float64),
            "strain":            to_voigt(eps_grid).astype(np.float64),
            "stress":            to_voigt(sigma_grid).astype(np.float64),
            "von_mises":         von_mises(sigma_grid).astype(np.float64),
            "damage":            d_grid.astype(np.float64),
            "strain_energy_pos": psi_grid.astype(np.float64),
        }, time=float(t))

        step_time = time.perf_counter() - t_step_start
        print(
            f"  step {step:3d}  t={t:.4f}  dt={dt:.4f}  "
            f"ε₁₁={float(state.strain_ave[0,0]):.2e}  "
            f"σ₁₁={float(state.stress_ave[0,0]):.2f} MPa  "
            f"max(d)={float(jnp.max(d_field)):.4f}  "
            f"st={iter_st}  err_abs={err_abs:.1e}  err_rel={err_rel:.1e}  "
            f"mech={int(iter_mech)}  helm={int(iter_helm)}  "
            f"time={step_time:.1f}s"
        )

        dt = min(dt * factor_inc, dt_max)

print(f"\nWritten → {settings.output}/{settings.jobname}.h5")
print(f"          {settings.output}/{settings.jobname}.xdmf")
print("Open the .xdmf in ParaView with the 'Xdmf3ReaderT' reader.")
