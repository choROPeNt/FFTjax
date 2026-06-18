"""
PFF benchmark: mode-I tension with phase-field fracture (AT2 model).

Extends benchmark_elastic_tension.py with a staggered mechanics–damage loop.

Reference
---------
    Schneider & Kästner (2024)  https://doi.org/10.1111/ffe.14553

Domain      : 250 × 250 × 1 voxels,  L = [50, 50, 0.2] mm
Material    : steel  E = 210 GPa,  ν = 0.3
PFF params  : l₀ = 1.0 mm,  Gc = 1.35 MPa·mm  (= 2.7/2 N/mm)
Pre-crack   : voxels x ∈ [20, 60),  y = 125 (centre)
Load        : uniaxial strain ramp  ε₂₂ → 1.2 × 10⁻³,  10 equal increments

Staggered scheme per increment
-------------------------------
    for each staggered iteration:
        1. Degrade stiffness:     C_eff = g(d) · C_field
        2. Mechanical solve:      (ε, σ) = solve_elastic(C_eff, G_glob, ε̄)
        3. Crack driving force:   ψ⁺ from undegraded Miehe spectral split
        4. Update history:        H = max(H_prev, ψ⁺)
        5. Damage solve:          d = solve_helmholtz_cg(H, ...)
        6. Converged?             max|d_new − d_old| < toler_st

Note on boundary conditions
---------------------------
Pure-strain BC (ε₁₁ = ε₃₃ = 0).  Mixed BC (σ₁₁ = σ₃₃ = 0) to be added.

Usage
-----
    python scripts/benchmark_pff_tension.py
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

from mat_models.elastic    import (LinearElasticIsotropic, assemble_C_field,
                                   strain_energy_miehe_split, lame_from_C_field)
from operators.green       import build_freq_grid, build_green_operator
from post.fields           import field_to_grid, von_mises, compute_displacement
from post.io               import IncrementalWriter, to_voigt
from solvers.elastic_nw_cg import solve_elastic
from solvers.pff_damage    import (degradation, update_history_hybrid,
                                   solve_helmholtz_cg)
from solvers.types         import SolveState, SolverSettings

jax.config.update("jax_enable_x64", True)

# ── Grid ──────────────────────────────────────────────────────────────────────

n  = (250, 250, 1)
L  = (50.0, 50.0, 0.2)   # mm
Nv = int(np.prod(n))
dx = tuple(Li / ni for Li, ni in zip(L, n))

# ── Materials ─────────────────────────────────────────────────────────────────

materials = [
    LinearElasticIsotropic(E=210e3, nu=0.3, name="steel"),
    LinearElasticIsotropic(E=1e-6,  nu=0.3, name="void"),
]

# ── Microstructure ────────────────────────────────────────────────────────────

ms      = np.zeros(n, dtype=int)
j_crack = n[1] // 2
ms[20:60, j_crack, :] = 1      # void (pre-crack)
phase   = jnp.array(ms.ravel())

print(f"Grid     : {n}  (Nv = {Nv})")
print(f"Domain   : {L} mm")
print(f"Crack    : x=[20,60)  y={j_crack}  ({int((ms==1).sum())} voxels)")

# ── Stiffness field (undegraded) ──────────────────────────────────────────────

C_field        = assemble_C_field(materials, phase)          # (3,3,3,3, Nv)
lam_vox, mu_vox = lame_from_C_field(C_field)                # undegraded λ, μ — fixed

# ── Reference medium — undegraded steel (G_glob stays fixed throughout) ───────

lam0 = materials[0].lam
mu0  = materials[0].mu
print(f"Reference: lam0={lam0/1e3:.2f} GPa  mu0={mu0/1e3:.2f} GPa")

xi_flat = build_freq_grid(n, L)
G_glob  = build_green_operator(xi_flat, lam0, mu0, scheme='rotated', dx=dx)

# ── PFF parameters ────────────────────────────────────────────────────────────

l0 = 1.0        # mm  — phase-field length scale
Gc = 2.7        # MPa·mm  (= 2.7 N/mm, AT2 convention from reference)

print(f"PFF      : l₀ = {l0} mm,  Gc = {Gc} MPa·mm"
      f"  →  Gc/l₀ = {Gc/l0:.3g} MPa")

# ── Solver settings ───────────────────────────────────────────────────────────

eps_goal = jnp.array([
    [0.0,    0.0, 0.0],
    [0.0, 1.2e-3, 0.0],   # ε₂₂ = 1.2 × 10⁻³
    [0.0,    0.0, 0.0],
])

settings = SolverSettings(
    ndim=3,
    n=n,
    L=L,
    toler_lin=1e-4,
    toler_nw=1e-4,
    maxiter_cg=500,
    maxiter_nw=300,
    jobname="benchmark_pff_tension",
    output="output",
)
settings.add_load_step(
    control=jnp.zeros((3, 3)),
    strain_ave_goal=eps_goal,
    stress_ave_goal=jnp.zeros((3, 3)),
    # 100 equal steps of dt=0.01 — matches paper Section 4.1 reference setup
    # (no adaptive stepping; brittle snap-through in one step is physically correct)
    timer=(0.01, 1.0, 0.001, 0.01), # dt_init, t_end, dt_min, dt_max
)

dt_init, t_end, dt_min, dt_max = settings.timer[0]

# Staggered scheme parameters — Schneider & Kästner (2024)
# Convergence when EITHER absolute OR relative criterion is met:
#   |d_new − d_old|_∞  < toler_st_abs     (εa = 1e-2)
#   |d_new − d_old|_∞ / (|d_new|_∞ + ε)  < toler_st_rel  (εr = 1e-3)
toler_st_abs  = 1e-2
toler_st_rel  = 1e-3
maxiter_st    = 200
toler_helm    = 1e-3
maxiter_helm  = 300
# Viscous regularisation η — Figure 3b of Schneider & Kästner (2025).
# η = 10⁻⁶  (same as kres).  With Δt = 0.01:  η/Δt = 10⁻⁴ MPa ≪ Gc/l₀ = 2.7 MPa.
# This is purely numerical stabilisation — it does NOT prevent the brittle snap-through.
eta = 1e-6
print(f"Viscosity: η = {eta:.1e}  →  η/Δt = {eta/dt_init:.1e} MPa  "
      f"(Gc/l₀ = {Gc/l0:.4g} MPa)")

# ── Initial mechanical state ──────────────────────────────────────────────────

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

# ── Initial damage and history (tracked separately from SolveState for now) ───

d_field = jnp.zeros((Nv,))   # damage field d ∈ [0, 1]
H_field = jnp.zeros((Nv,))   # history variable H = max(ψ⁺) over time

# ── Adaptive time-stepper parameters ─────────────────────────────────────────

factor_inc   = 1.5
factor_dec   = 0.5
max_cutbacks = 5
max_steps    = 1000

# ── Load-stepping loop ────────────────────────────────────────────────────────

os.makedirs(settings.output, exist_ok=True)

dt   = state.dtime
t    = state.time
step = state.kinc

phase_grid = ms.astype(np.float32)
zero_grid  = np.zeros((*n, 6), dtype=np.float64)
zero_scal  = np.zeros(n,       dtype=np.float64)
zero_u     = np.zeros((*n, 3), dtype=np.float64)

with IncrementalWriter(
    f"{settings.output}/{settings.jobname}", grid_shape=n, grid_spacing=dx
) as w:

    # ── write undeformed initial state at t = 0 ───────────────────────────────
    w.write_increment(0, {
        "phase":        phase_grid,
        "displacement": zero_u,
        "strain":       zero_grid,
        "stress":       zero_grid,
        "von_mises":    zero_scal,
        "damage":       zero_scal,
        "strain_energy_pos": zero_scal,
    }, time=0.0)

    while t < t_end and step < max_steps:

        dt = float(np.clip(dt, dt_min, dt_max))
        dt = min(dt, t_end - t)

        eps_bar_i    = float(t + dt) * eps_goal
        converged_mech = False
        t_step_start   = time.perf_counter()

        # ── cutback loop (mechanical convergence) ────────────────────────────
        for attempt in range(max_cutbacks + 1):

            # ── staggered iteration ──────────────────────────────────────────
            d_st   = d_field   # damage at start of this increment attempt
            H_st   = H_field

            for iter_st in range(1, maxiter_st + 1):
                d_prev_st = d_st

                # 1. degrade stiffness with current damage
                g     = degradation(d_st)                        # (Nv,)
                C_eff = g[None, None, None, None, :] * C_field   # (3,3,3,3,Nv)

                # 2. mechanical solve with degraded stiffness
                eps_i, sigma_i, delta_i, iter_mech, conv_mech = solve_elastic(
                    n, C_eff, G_glob, eps_bar_i,
                    toler_lin=settings.toler_lin,
                    maxiter=settings.maxiter_cg,
                )

                # 3. crack driving force from undegraded Miehe split
                psi_pos, _ = strain_energy_miehe_split(eps_i, lam_vox, mu_vox)

                # 4. update history variable — hybrid irreversibility (dthres=0.95)
                H_st = update_history_hybrid(H_st, psi_pos, d_st, dthres=0.95)

                # 5. damage solve — Helmholtz preconditioned CG
                d_st, iter_helm, conv_helm = solve_helmholtz_cg(
                    H_st, xi_flat, n, l0, Gc, d_prev_st,
                    toler_cg=toler_helm,
                    maxiter=maxiter_helm,
                    eta=eta,
                    dt=dt,
                )

                # 6. staggered convergence — either absolute or relative criterion
                diff      = jnp.max(jnp.abs(d_st - d_prev_st))
                err_abs   = float(diff)
                err_rel   = float(diff / (jnp.max(jnp.abs(d_st)) + 1e-30))
                if err_abs < toler_st_abs or err_rel < toler_st_rel:
                    break

            converged_mech = bool(conv_mech)

            if converged_mech:
                break

            dt = max(dt * factor_dec, dt_min)
            print(f"    cutback #{attempt + 1}  dt → {dt:.6f}  (mech iter={int(iter_mech)})")

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

        # ── post-processing ───────────────────────────────────────────────────
        eps_grid   = field_to_grid(state.strain_loc, n)
        sigma_grid = field_to_grid(state.stress_loc, n)
        u_grid     = compute_displacement(state.strain_loc, eps_bar_i, xi_flat, n, dx)
        d_grid     = np.asarray(d_field).reshape(n)
        psi_grid   = np.asarray(psi_pos).reshape(n)

        w.write_increment(step, {
            "phase":             phase_grid,
            "displacement":      u_grid.astype(np.float64),
            "strain":            to_voigt(eps_grid).astype(np.float64),
            "stress":            to_voigt(sigma_grid).astype(np.float64),
            "von_mises":         von_mises(sigma_grid).astype(np.float64),
            "damage":            d_grid.astype(np.float64),
            "strain_energy_pos": psi_grid.astype(np.float64),
        }, time=float(t))

        step_time = time.perf_counter() - t_step_start
        print(
            f"  step {step:2d}  t={t:.3f}  "
            f"ε₂₂={float(state.strain_ave[1,1]):.2e}  "
            f"σ₂₂={float(state.stress_ave[1,1]):.2f} MPa  "
            f"max(d)={float(jnp.max(d_field)):.4f}  "
            f"st={iter_st}  err_abs={err_abs:.1e}  err_rel={err_rel:.1e}  "
            f"mech={int(iter_mech)}  helm={int(iter_helm)}  time={step_time:.1f}s"
        )

        dt = min(dt * factor_inc, dt_max)

print(f"\nWritten → {settings.output}/{settings.jobname}.h5")
print(f"          {settings.output}/{settings.jobname}.xdmf")
print("Open the .xdmf in ParaView with the 'Xdmf3ReaderT' reader.")
