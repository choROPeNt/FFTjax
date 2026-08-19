"""
PFF benchmark: mode-II shear with phase-field fracture (AT2 model).

Mirrors pff_tension.py but applies shear loading instead of uniaxial tension.

Reference
---------
    Schneider & Kästner (2025)  https://doi.org/10.1111/ffe.14553

Domain      : 250 × 250 × 1 voxels,  L = [50, 50, 0.2] mm
Material    : steel  E = 210 GPa,  ν = 0.3
PFF params  : l₀ = 1.0 mm,  Gc = 2.7 MPa·mm
Pre-crack   : x ∈ [5, 15) mm  (i=[25,75) at 250² grid),  y = 125 (centre)
Load        : shear strain ramp  ε₁₂ = ε₂₁ → 1.0 × 10⁻³,  100 equal increments

Staggered scheme per increment (problems.fracture.solve_fracture, one call
per cutback attempt) -- see solvers.coupling.staggered for the loop itself:
    1. Degrade stiffness:     C_eff = g(d) · C_field
    2. Mechanical solve:      (ε, σ) = solve_lippmann_schwinger(C_eff, ...)
    3. Crack driving force:   ψ⁺ from undegraded Amor dev/vol split
    4. Update history:        hybrid irreversibility (Steinke & Kaliske 2019) —
                               H = max(H_prev, ψ⁺) if d ≥ d_thres, else H = ψ⁺
    5. Damage solve:          d = solve_damage_helmholtz_cg(H, ...)
    6. Converged?             max|d_new − d_old| < toler_st

Usage
-----
    python benchmark/single_notch_plate/pff_shear.py
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
sys.path.insert(0, "src")

import csv
import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import time

from materialmodels.elastic.isotropic import LinearElasticIsotropic
from post.fields              import field_to_grid, von_mises, compute_displacement, to_voigt
from utils.io.xdmf_writer     import IncrementalWriter
from problems.fracture        import solve_fracture

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

x_crack = (5.0, 15.0)   # mm: [start, end)

ms        = np.zeros(n, dtype=int)
j_crack   = n[1] // 2
i_start   = round(x_crack[0] / dx[0])
i_end     = round(x_crack[1] / dx[0])
ms[i_start:i_end, j_crack, :] = 1   # void (pre-crack)
phase     = jnp.array(ms.ravel())

print(f"Grid     : {n}  (Nv = {Nv})")
print(f"Domain   : {L} mm")
print(f"Crack    : x=[{x_crack[0]},{x_crack[1]}) mm  i=[{i_start},{i_end})  y={j_crack}  ({int((ms==1).sum())} voxels)")

# ── PFF parameters ────────────────────────────────────────────────────────────

l0 = 1.0        # mm  — phase-field length scale
Gc = 2.7        # MPa·mm  (= 2.7 N/mm, AT2 convention from reference)

print(f"PFF      : l₀ = {l0} mm,  Gc = {Gc} MPa·mm"
      f"  →  Gc/l₀ = {Gc/l0:.3g} MPa")

# Hybrid damage-/crack-like irreversibility threshold (Steinke & Kaliske 2019,
# as adopted by Schneider & Kästner 2025). History lock (H = max(H_prev, ψ⁺))
# only kicks in once d ≥ d_thres; below it H = ψ⁺ is left unrestricted so the
# pre-crack process zone doesn't over-widen.
d_thres = 0.95

# ── Solver settings ───────────────────────────────────────────────────────────

eps_goal = jnp.array([
    [0.0,  1e-3, 0.0],
    [1e-3, 0.0,  0.0],   # ε₁₂ = ε₂₁ = 1.0 × 10⁻³  (mode-II shear)
    [0.0,  0.0,  0.0],
])

toler_lin    = 1e-2
toler_helm   = 1e-3
maxiter_cg   = 500
maxiter_helm = 300
jobname      = "benchmark_pff_shear"
output       = "output/benchmark"

dt_init, t_end, dt_min, dt_max = 0.01, 1.0, 0.01, 0.01

# Staggered scheme parameters — Schneider & Kästner (2025)
toler_st_abs  = 1e-2
toler_st_rel  = 1e-3
maxiter_st    = 200
eta = 1e-6
print(f"Viscosity: η = {eta:.1e}  →  η/Δt = {eta/dt_init:.1e} MPa  "
      f"(Gc/l₀ = {Gc/l0:.4g} MPa)")

# ── Initial damage and history ────────────────────────────────────────────────

d_field = jnp.zeros((Nv,))
H_field = jnp.zeros((Nv,))

# ── Adaptive time-stepper parameters ─────────────────────────────────────────

factor_inc   = 1.5
factor_dec   = 0.5
max_cutbacks = 5
max_steps    = 1000

# ── Load-stepping loop ────────────────────────────────────────────────────────

os.makedirs(output, exist_ok=True)

dt   = dt_init
t    = 0.0
step = 0

phase_grid = ms.astype(np.float32)
zero_grid  = np.zeros((*n, 6), dtype=np.float64)
zero_scal  = np.zeros(n,       dtype=np.float64)
zero_u     = np.zeros((*n, 3), dtype=np.float64)

history: list[dict] = []

with IncrementalWriter(
    f"{output}/{jobname}", grid_shape=n, grid_length=L
) as w:

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

            sol = solve_fracture(
                n, L, phase, materials, eps_bar_i,
                l0, Gc, d_field, H_field,
                scheme="rotated",
                toler_lin=toler_lin, maxiter_cg=maxiter_cg,
                toler_helm=toler_helm, maxiter_helm=maxiter_helm,
                eta=eta, dt=dt, d_thres=d_thres,
                toler_st_abs=toler_st_abs, toler_st_rel=toler_st_rel, maxiter_st=maxiter_st,
            )

            converged_mech = bool(sol.converged_mech)

            if converged_mech:
                break

            dt = max(dt * factor_dec, dt_min)
            print(f"    cutback #{attempt + 1}  dt → {dt:.6f}  (not converged)")

        if not converged_mech:
            raise RuntimeError(
                f"Step {step + 1}: mechanical CG did not converge after "
                f"{max_cutbacks} cutbacks at t={t:.4f}"
            )

        # ── accept increment ──────────────────────────────────────────────────
        t       += dt
        step    += 1
        d_field  = sol.d
        H_field  = sol.H

        strain_ave = jnp.mean(sol.eps,   axis=-1)
        stress_ave = jnp.mean(sol.sigma, axis=-1)

        # ── post-processing ───────────────────────────────────────────────────
        eps_grid   = field_to_grid(sol.eps, n)
        sigma_grid = field_to_grid(sol.sigma, n)
        u_grid     = compute_displacement(sol.eps, eps_bar_i, n, L)
        d_grid     = np.asarray(sol.d).reshape(n)
        psi_grid   = np.asarray(sol.psi_pos).reshape(n)

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
            f"ε₁₂={float(strain_ave[0,1]):.2e}  "
            f"σ₁₂={float(stress_ave[0,1]):.2f} MPa  "
            f"max(d)={float(jnp.max(d_field)):.4f}  "
            f"st={sol.iter_staggered}  err_abs={sol.err_abs:.1e}  err_rel={sol.err_rel:.1e}  "
            f"time={step_time:.1f}s"
        )

        history.append({
            "step":       step,
            "time":       float(t),
            "dt":         float(dt),
            "eps_11":     float(strain_ave[0, 0]),
            "eps_22":     float(strain_ave[1, 1]),
            "eps_33":     float(strain_ave[2, 2]),
            "eps_12":     float(strain_ave[0, 1]),
            "eps_13":     float(strain_ave[0, 2]),
            "eps_23":     float(strain_ave[1, 2]),
            "sig_11":     float(stress_ave[0, 0]),
            "sig_22":     float(stress_ave[1, 1]),
            "sig_33":     float(stress_ave[2, 2]),
            "sig_12":     float(stress_ave[0, 1]),
            "sig_13":     float(stress_ave[0, 2]),
            "sig_23":     float(stress_ave[1, 2]),
            "max_d":      float(jnp.max(d_field)),
            "iter_st":    sol.iter_staggered,
            "err_abs":    sol.err_abs,
            "err_rel":    sol.err_rel,
            "wall_time_s": step_time,
        })

        dt = min(dt * factor_inc, dt_max)

print(f"\nWritten → {output}/{jobname}.h5")
print(f"          {output}/{jobname}.xdmf")
print("Open the .xdmf in ParaView with the 'Xdmf3ReaderT' reader.")

# ── CSV history ────────────────────────────────────────────────────────────────

csv_path = f"{output}/{jobname}_history.csv"
with open(csv_path, "w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=history[0].keys())
    writer.writeheader()
    writer.writerows(history)
print(f"Written → {csv_path}")

# ── Reference comparison plot ──────────────────────────────────────────────────

ref_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref_shear.csv")
ref = np.loadtxt(ref_path, delimiter=",", skiprows=1)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(ref[:, 0], ref[:, 1], "k--", linewidth=1.2, label="Schneider & Kästner (2025)")
ax.plot(
    [r["eps_12"] for r in history],
    [r["sig_12"] for r in history],
    "b-o", markersize=3, linewidth=1.2, label="FFTjax",
)
ax.set_xlabel(r"$\bar{\varepsilon}_{12}$")
ax.set_ylabel(r"$\bar{\sigma}_{12}$ [MPa]")
ax.set_title("Mode-II shear — single edge notch plate")
ax.legend()
ax.grid(True, linewidth=0.5, alpha=0.6)
fig.tight_layout()
plot_path = f"{output}/{jobname}_comparison.png"
fig.savefig(plot_path, dpi=150)
print(f"Written → {plot_path}")
