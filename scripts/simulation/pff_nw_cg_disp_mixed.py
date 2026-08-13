"""
Phase-field fracture (AT2) for a fibre composite RVE — config-driven via YAML,
with mixed macroscopic strain/stress boundary conditions.

Same staggered scheme as pff_nw_cg_strain.py (Amor split, viscous
regularisation, heterogeneous-Gc Helmholtz CG), but the mechanical solve
uses the displacement-based solvers.mechanical.displacement_nw_cg.ddisp_nw_cg
instead of the strain-based solver, so individual components of the
macroscopic strain/stress tensor can be independently strain-controlled
(prescribed eps_goal) or stress-controlled (prescribed stress_goal) via
loading.control in the YAML config — see configs/pff_rve_mixed.yaml.

Caveats vs. the pure-strain script
-----------------------------------
- No Willot "rotated" Green's-operator scheme: ddisp_nw_cg has no
  reference-medium Green's operator, so `solver.scheme` in the config is
  ignored for this script.
- ddisp_nw_cg rebuilds its frequency-domain preconditioner on every call
  (it depends on the current degraded stiffness), and the mechanical solve
  sits inside the innermost staggered loop (up to maxiter_st iterations per
  time step) — expect this to be slower per staggered iteration than the
  strain-based script.
- Load-controlled (stress-controlled) crack propagation can become unstable
  past peak load (snap-through); this staggered Newton-CG scheme has no
  arc-length control, so convergence may degrade or fail there.

Usage
-----
    python scripts/simulation/pff_nw_cg_disp_mixed.py configs/pff_rve_mixed.yaml

Output
------
    <output>/<jobname>.h5
    <output>/<jobname>.xdmf
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

from utils.config              import load_config
from utils.io_read              import read_simulation_input
from mat_models.elastic        import (LinearElasticIsotropic,
                                       TransverseIsotropicFibre,
                                       assemble_C_field_smooth,
                                       strain_energy_amor_split,
                                       lame_from_C_field)
from mat_models.micromechanics import yarn_properties
from operators.green           import build_freq_grid
from post.fields                import field_to_grid, von_mises, compute_displacement
from utils.io.xdmf_writer import IncrementalWriter
from post.fields          import to_voigt
from solvers.mechanical.displacement_nw_cg import ddisp_nw_cg
from solvers.damage.pff_damage        import (degradation, update_history_hybrid,
                                       solve_helmholtz_cg_het)
from solvers.types             import SolveState, SolverSettings

jax.config.update("jax_enable_x64", True)

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="PFF benchmark solver (Miehe split), mixed strain/stress BC — config-driven via YAML"
)
parser.add_argument("config", type=Path, help="Path to YAML configuration file")
parser.add_argument("--input", type=Path, default=None,
                    help="Override input file (used by active_learning.py for batch runs)")
parser.add_argument("--output", type=Path, default=None,
                    help="Override output directory (used by active_learning.py)")
args = parser.parse_args()

cfg = load_config(args.config)
if args.input is not None:
    cfg["input"]   = str(args.input)
    cfg["jobname"] = args.input.stem
if args.output is not None:
    cfg["output"] = str(args.output)
print(f"Config  : {args.config}")
print(f"Input   : {cfg['input']}")
print(f"Output  : {cfg['output']}/{cfg['jobname']}")

# ── Load geometry ─────────────────────────────────────────────────────────────

n, L, phase_np, orientations_np, _, vf_np, d_init_np, H_init_np = \
    read_simulation_input(cfg["input"])

Nv   = int(np.prod(n))
dx   = tuple(Li / ni for Li, ni in zip(L, n))
# ── Fill fiber orientations when not stored in the file ───────────────────────
_fiber_phase_cfg = int(cfg.get("fiber_phase", 1))
if not np.any(orientations_np != 0) and "fibre_dir" in cfg:
    d = np.array(cfg["fibre_dir"], dtype=float)
    d /= np.linalg.norm(d)
    fiber_vox = phase_np == _fiber_phase_cfg
    orientations_np = orientations_np.copy()
    orientations_np[:, fiber_vox] = d[:, None]
    print(f"  Orientations set to {d} for phase {_fiber_phase_cfg} "
          f"({int(fiber_vox.sum())} voxels)")

phi_ = float(np.mean(phase_np == 1))

print(f"Device  : {jax.devices()[0].device_kind}")
print(f"Grid    : {n}   Nv = {Nv:,}")
print(f"Domain  : {tuple(f'{v:.4g}' for v in L)}")
print(f"phi_act : {phi_:.4f}")

phase_jax    = jnp.array(phase_np)
orientations = jnp.array(orientations_np)
matrix_mask  = jnp.array(phase_np == 0)
void_mask    = jnp.array(phase_np == 2)

# ── Materials ─────────────────────────────────────────────────────────────────

def _build_material(spec: dict):
    t  = spec["type"]
    kw = {k: v for k, v in spec.items() if k != "type"}
    if t == "YarnProperties":
        Vf     = float(kw.pop("Vf_yarn"))
        E_m    = float(kw.pop("E_m"));  nu_m = float(kw.pop("nu_m"))
        E_fL   = float(kw.pop("E_fL")); E_fT  = float(kw.pop("E_fT"))
        G_fLT  = float(kw.pop("G_fLT"))
        nu_fLT = float(kw.pop("nu_fLT")); nu_fTT = float(kw.pop("nu_fTT"))
        E_L, E_T, G_LT, nu_LT, nu_TT = yarn_properties(
            Vf, E_fL, E_fT, G_fLT, nu_fLT, nu_fTT, E_m, nu_m
        )
        return TransverseIsotropicFibre(
            E_L=E_L, E_T=E_T, G_LT=G_LT, nu_LT=nu_LT, nu_TT=nu_TT,
            name=kw.get("name", "yarn"),
        )
    if t == "LinearElasticIsotropic":
        return LinearElasticIsotropic(
            name=str(kw.get("name", t)),
            **{k: float(v) for k, v in kw.items() if k != "name"},
        )
    if t == "TransverseIsotropicFibre":
        return TransverseIsotropicFibre(
            name=str(kw.get("name", t)),
            **{k: float(v) for k, v in kw.items() if k != "name"},
        )
    raise ValueError(f"Unknown material type: {t}")

mat_cfg    = cfg["materials"]
mat_matrix = _build_material(mat_cfg["matrix"])
mat_yarn   = _build_material(mat_cfg["yarn"])
for m in [mat_matrix, mat_yarn]:
    print(" ", m)

# ── Stiffness field ───────────────────────────────────────────────────────────

has_phase2      = bool(np.any(phase_np == 2))
phase2_mask     = jnp.array(phase_np == 2)
interphase_mask = jnp.zeros(Nv, dtype=bool)   # overridden below if interphase present
Vf_yarn     = float(cfg.get("Vf_yarn", 1.0))
vf_jax      = jnp.array(vf_np)

if has_phase2 and "interphase" in mat_cfg:
    # ── 3-phase: materials listed in phase-index order ────────────────────────
    fiber_phase = int(cfg.get("fiber_phase", 1))
    from mat_models.elastic import assemble_C_field_oriented
    mat_interphase = _build_material(mat_cfg["interphase"])
    print(" ", mat_interphase)
    if fiber_phase == 2:
        materials_3 = [mat_matrix, mat_interphase, mat_yarn]
    else:
        materials_3 = [mat_matrix, mat_yarn, mat_interphase]
    C_field = assemble_C_field_oriented(materials_3, phase_jax, orientations)
    print(f"  (3-phase assembly, fiber_phase={fiber_phase})")

elif has_phase2 and "void" in mat_cfg:
    # ── void/crack override ───────────────────────────────────────────────────
    C_field = assemble_C_field_smooth([mat_matrix, mat_yarn], orientations, vf_jax)
    mat_void   = _build_material(mat_cfg["void"])
    print(" ", mat_void)
    void_mask5 = jnp.broadcast_to(phase2_mask[None,None,None,None,:], C_field.shape)
    C_void5    = jnp.broadcast_to(mat_void.stiffness_tensor()[..., None], C_field.shape)
    C_field    = jnp.where(void_mask5, C_void5, C_field)

else:
    # ── 2-phase smooth blend ──────────────────────────────────────────────────
    C_field = assemble_C_field_smooth([mat_matrix, mat_yarn], orientations, vf_jax)

lam_vox, mu_vox = lame_from_C_field(C_field)

xi_flat = build_freq_grid(n, L)

# ── PFF parameters ────────────────────────────────────────────────────────────

pff      = cfg["pff"]
l0       = float(pff["l0"])
Gc_mat   = float(pff["Gc_mat"])
Gc_yarn  = Gc_mat * float(pff.get("Gc_yarn_factor", 1000))

eta      = float(pff.get("eta", 1e-6))

toler_st_abs = float(pff.get("toler_st_abs", 1e-2))
toler_st_rel = float(pff.get("toler_st_rel", 1e-3))
maxiter_st   = int(pff.get("maxiter_st", 200))
toler_helm   = float(pff.get("toler_helm", 1e-3))
maxiter_helm = int(pff.get("maxiter_helm", 300))
if has_phase2 and "interphase" in mat_cfg:
    Gc_interphase   = float(pff.get("Gc_interphase", Gc_mat * 0.5))
    fiber_mask_jax  = jnp.array(phase_np == fiber_phase)
    inter_phase_idx = 1 if fiber_phase == 2 else 2
    interphase_mask = jnp.array(phase_np == inter_phase_idx)
    Gc_field = jnp.where(fiber_mask_jax,  Gc_yarn,
               jnp.where(interphase_mask, Gc_interphase, Gc_mat))
    print(f"  Gc_interphase = {Gc_interphase:.3e}  MPa·mm")
else:
    phi_ind  = jnp.clip(vf_jax / Vf_yarn, 0.0, 1.0)
    Gc_field = (1.0 - phi_ind) * Gc_mat + phi_ind * Gc_yarn
    if has_phase2 and "void" in mat_cfg:
        Gc_field = jnp.where(phase2_mask, Gc_yarn, Gc_field)

print(f"PFF     : l₀={l0}  Gc_mat={Gc_mat}  Gc_yarn={Gc_yarn:.2e}"
      f"  η={eta:.0e}")

# ── Loading — mixed strain/stress control ─────────────────────────────────────
# control     : 1 = stress-controlled, 0 = strain-controlled (per tensor entry)
# eps_goal    : target macroscopic strain at t=1  (only entries where control==0 matter)
# stress_goal : target macroscopic stress at t=1  (only entries where control==1 matter)

control_np  = np.asarray(cfg["loading"]["control"], dtype=int)
control     = tuple(tuple(int(v) for v in row) for row in control_np)   # static for jit
eps_goal    = jnp.array(cfg["loading"]["eps_goal"],    dtype=float)
stress_goal = jnp.array(cfg["loading"]["stress_goal"], dtype=float)

print(f"control (1=stress, 0=strain):\n{control_np}")

ts           = cfg["timestepper"]
dt_init      = float(ts["dt_init"])
t_end        = float(ts["t_end"])
dt_min       = float(ts["dt_min"])
dt_max       = float(ts["dt_max"])
max_steps    = int(ts["max_steps"])
factor_inc   = float(ts.get("factor_inc", 1.5))
factor_dec   = float(ts.get("factor_dec", 0.5))
max_cutbacks = int(ts.get("max_cutbacks", 5))

jobname = cfg["jobname"]
output  = cfg["output"]

slv = cfg["solver"]
settings = SolverSettings(
    ndim=3, n=n, L=L,
    toler_lin=float(slv["toler_lin"]),
    toler_nw=float(slv.get("toler_nw", 1e-4)),
    maxiter_cg=int(slv["maxiter_cg"]),
    maxiter_nw=int(slv.get("maxiter_nw", 300)),
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
    strain_loc=zero_v, stress_loc=zero_v, tangent_glob=C_field,
    strain_ave=zero33, stress_ave=zero33,
    strain_ave_inc_goal=zero33, stress_ave_inc_goal=zero33,
    Deltastrain_loc=zero_v, Deltastress_loc=zero_v,
    stress_loc_goal=zero_v, deltastrain_loc=zero_v,
    time=0.0, dtime=dt_init, kinc=0, kstep=1,
    iter_nw=0, iter_cg=0, info=0, bb0n=1.0, pnewdt=1.0,
)

d_field = jnp.array(d_init_np)
H_field = jnp.array(H_init_np)

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
        "phase":             phase_vis,
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

        converged_mech = False
        t_step_start   = time.perf_counter()

        for attempt in range(max_cutbacks + 1):

            t_frac       = float(t + dt) / t_end
            eps_bar_i    = t_frac * eps_goal
            stress_bar_i = t_frac * stress_goal

            d_st = d_field
            H_st = H_field

            for iter_st in range(1, maxiter_st + 1):
                d_prev_st = d_st

                # 1. degrade matrix stiffness
                g     = jnp.where(matrix_mask, degradation(d_st), 1.0)
                C_eff = g[None, None, None, None, :] * C_field

                # 2. mechanical solve (mixed strain/stress BC)
                eps_i, sigma_i, delta_i, eps_bar_out_i, conv_mech = ddisp_nw_cg(
                    n, C_eff, xi_flat, eps_bar_i, control, stress_bar_i,
                    toler_lin=settings.toler_lin,
                    maxiter=settings.maxiter_cg,
                )

                # 3. crack driving force — Miehe spectral split, matrix only
                psi_pos, _ = strain_energy_amor_split(eps_i, lam_vox, mu_vox)
                damage_zone = (matrix_mask | interphase_mask) if (
                    has_phase2 and "interphase" in mat_cfg) else matrix_mask
                psi_pos    = jnp.where(damage_zone, psi_pos, 0.0)

                # 4. history
                H_st = update_history_hybrid(H_st, psi_pos, d_prev_st)

                # 5. Helmholtz CG (heterogeneous Gc)
                d_new, conv_helm = solve_helmholtz_cg_het(
                    H_st, xi_flat, n, l0, Gc_field, d_prev_st,
                    toler_cg=toler_helm, maxiter=maxiter_helm,
                    eta=eta, dt=dt,
                )

                d_st = d_new

                diff    = jnp.max(jnp.abs(d_st - d_prev_st))
                err_abs = float(diff)
                err_rel = float(diff / (jnp.max(jnp.abs(d_st)) + 1e-30))
                if err_abs < toler_st_abs or err_rel < toler_st_rel:
                    break

            converged_mech = bool(conv_mech)
            if converged_mech:
                break

            dt = max(dt * factor_dec, dt_min)
            print(f"    cutback #{attempt+1}  dt → {dt:.6f}  (mech converged={converged_mech})")

        if not converged_mech:
            raise RuntimeError(
                f"Step {step+1}: mechanical CG did not converge after "
                f"{max_cutbacks} cutbacks at t={t:.4f}"
            )

        t      += dt
        step   += 1
        d_field = d_st
        H_field = H_st

        state = state._replace(
            strain_loc=eps_i, stress_loc=sigma_i, deltastrain_loc=delta_i,
            strain_ave=eps_bar_out_i,
            stress_ave=jnp.mean(sigma_i, axis=-1),
            time=t, dtime=dt, kinc=step,
            info=0 if converged_mech else 1,
        )

        eps_grid   = field_to_grid(state.strain_loc, n)
        sigma_grid = field_to_grid(state.stress_loc, n)
        u_grid     = compute_displacement(state.strain_loc, state.strain_ave, xi_flat, n, dx)
        d_grid     = np.asarray(d_field).reshape(n)
        psi_grid   = np.asarray(psi_pos).reshape(n)

        w.write_increment(step, {
            "phase":             phase_vis,
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
            f"eps11={float(state.strain_ave[0,0]):.4e}  "
            f"sig11={float(state.stress_ave[0,0]):.2f} MPa  "
            f"max(d)={float(jnp.max(d_field)):.4f}  "
            f"st={iter_st}  err_abs={err_abs:.1e}  err_rel={err_rel:.1e}  "
            f"time={step_time:.1f}s"
        )

        dt = min(dt * factor_inc, dt_max)

print(f"\nWritten → {output}/{jobname}.h5")
print(f"          {output}/{jobname}.xdmf")
print("Open the .xdmf in ParaView with the 'Xdmf3ReaderT' reader.")
