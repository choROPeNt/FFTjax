"""
Phase-field fracture (AT2) for a TexGen woven composite RVE.
Matrix-only cracking: yarn voxels are assigned a very large Gc so they
never damage, while the matrix phase follows the standard PFF model.

Reads a TexGen voxel VTU export and extracts:
  - grid geometry
  - YarnIndex      (-1 = matrix, >= 0 = yarn number)
  - YarnTangent    (unit fibre direction per yarn voxel)

Material assignment
-------------------
  Matrix (YarnIndex == -1) : LinearElasticIsotropic  + AT2 damage (Gc_matrix)
  Yarns  (YarnIndex >= 0)  : TransverseIsotropicFibre, undamageable (Gc_yarn >> 1)

Staggered scheme per increment
-------------------------------
  for each staggered iteration:
    1. Degrade stiffness in matrix only:  C_eff = g(d) * C_matrix + C_yarn
    2. Mechanical solve:                  (ε, σ) = solve_elastic(C_eff, ...)
    3. Crack driving force (matrix only): ψ⁺ from undegraded Amor split, zeroed in yarn
    4. Update history:                    H = max(H_prev, ψ⁺)
    5. Damage solve (heterogeneous Gc):   d = solve_helmholtz_cg_het(H, Gc_field, ...)
    6. Converged?                         max|d_new - d_old| < toler_st

Usage
-----
    python scripts/pff_nw_cg_strain_texgen.py

Output
------
    output/texgen_pff.h5
    output/texgen_pff.xdmf
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
import xml.etree.ElementTree as ET

from mat_models.elastic        import (LinearElasticIsotropic,
                                       TransverseIsotropicFibre,
                                       assemble_C_field_oriented,
                                       strain_energy_amor_split,
                                       lame_from_C_field)
from mat_models.micromechanics import yarn_properties
from operators.green           import build_freq_grid, build_green_operator
from post.fields               import field_to_grid, von_mises, compute_displacement
from post.io                   import IncrementalWriter, to_voigt
from solvers.elastic_nw_cg     import solve_elastic
from solvers.pff_damage        import (degradation, update_history,
                                       solve_helmholtz_cg_het)
from solvers.types             import SolveState, SolverSettings

jax.config.update("jax_enable_x64", True)


# ── VTU reader ────────────────────────────────────────────────────────────────

def read_texgen_vtu(path: str):
    """
    Parse a TexGen voxel VTU (UnstructuredGrid of hexahedral cells).

    Returns
    -------
    n              : tuple (nx, ny, nz)
    L              : tuple (Lx, Ly, Lz)
    phase_np       : (Nv,) int    0 = matrix, 1 = yarn  (C-order)
    orientations   : (3, Nv)      YarnTangent unit vectors
    yarn_index_np  : (Nv,) int    raw YarnIndex (-1=matrix, >=0=yarn)
    """
    tree  = ET.parse(path)
    piece = tree.getroot().find('UnstructuredGrid/Piece')
    assert piece is not None, f"No <UnstructuredGrid/Piece> found in {path}"

    pts_node  = piece.find('Points/DataArray')
    conn_node = piece.find('Cells/DataArray[@Name="connectivity"]')
    assert pts_node  is not None, "No Points/DataArray in VTU"
    assert conn_node is not None, "No connectivity DataArray in VTU"

    pts  = np.fromstring(pts_node.text  or "", sep=' ').reshape(-1, 3)
    conn = np.fromstring(conn_node.text or "", sep=' ').astype(int).reshape(-1, 8)

    centroids = pts[conn].mean(axis=1)

    xu = np.unique(np.round(centroids[:, 0], 8))
    yu = np.unique(np.round(centroids[:, 1], 8))
    zu = np.unique(np.round(centroids[:, 2], 8))
    nx, ny, nz = len(xu), len(yu), len(zu)
    dx, dy, dz = float(xu[1]-xu[0]), float(yu[1]-yu[0]), float(zu[1]-zu[0])
    Lx, Ly, Lz = nx*dx, ny*dy, nz*dz

    ix_arr = np.round((centroids[:,0]-xu[0])/dx).astype(int)
    iy_arr = np.round((centroids[:,1]-yu[0])/dy).astype(int)
    iz_arr = np.round((centroids[:,2]-zu[0])/dz).astype(int)
    fft_idx = ix_arr*(ny*nz) + iy_arr*nz + iz_arr

    cd = piece.find('CellData')
    assert cd is not None, f"No <CellData> block found in {path}"
    arrays = {a.attrib['Name']: a for a in cd.findall('DataArray')}

    yarn_vtk = np.fromstring(arrays['YarnIndex'].text   or "", sep=' ').astype(int)
    tang_vtk = np.fromstring(arrays['YarnTangent'].text or "", sep=' ').reshape(-1, 3)

    Nv           = nx * ny * nz
    phase_flat   = np.empty(Nv, dtype=int)
    tangent_flat = np.zeros((Nv, 3))
    phase_flat[fft_idx]   = yarn_vtk
    tangent_flat[fft_idx] = tang_vtk

    phase_np     = np.where(phase_flat >= 0, 1, 0)
    orientations = tangent_flat.T.copy()

    return (nx, ny, nz), (Lx, Ly, Lz), phase_np, orientations, phase_flat


# ── Load VTU ──────────────────────────────────────────────────────────────────

vtu_path = "data/160_nn_2.vtu"
n, L, phase_np, orientations_np, yarn_index_np = read_texgen_vtu(vtu_path)

Nv  = int(np.prod(n))
dx  = tuple(Li / ni for Li, ni in zip(L, n))
print(f"Grid     : {n}   Nv = {Nv}")
print(f"Domain   : {tuple(f'{v:.4g}' for v in L)}  mm")

phase_jax    = jnp.array(phase_np)
orientations = jnp.array(orientations_np)

# boolean mask — True where voxel is matrix
matrix_mask = jnp.array(phase_np == 0)   # (Nv,)

phi_yarn = float(np.mean(phase_np))
print(f"Yarn vol. fraction: {phi_yarn:.3f}")

# ── Materials ─────────────────────────────────────────────────────────────────
# Constituent properties (MPa).  Adjust to your actual fibre/matrix system.

E_m,   nu_m   = 3.5e3, 0.35    # epoxy matrix
E_fL,  E_fT   = 230e3, 15e3    # carbon fibre: longitudinal / transverse
G_fLT          = 15e3           # fibre long.-trans. shear modulus
nu_fLT, nu_fTT = 0.20, 0.30    # fibre Poisson ratios
Vf_yarn        = 0.72           # fibre volume fraction within a yarn/tow

E_L, E_T, G_LT, nu_LT, nu_TT = yarn_properties(
    Vf_yarn, E_fL, E_fT, G_fLT, nu_fLT, nu_fTT, E_m, nu_m
)
print(f"Yarn (Vf={Vf_yarn:.2f}):  "
      f"E_L={E_L/1e3:.1f}  E_T={E_T/1e3:.2f}  "
      f"G_LT={G_LT/1e3:.2f}  nu_LT={nu_LT:.3f}  nu_TT={nu_TT:.3f}  GPa / —")

materials = [
    LinearElasticIsotropic(E=E_m, nu=nu_m, name="epoxy matrix"),
    TransverseIsotropicFibre(
        E_L=E_L, E_T=E_T, G_LT=G_LT, nu_LT=nu_LT, nu_TT=nu_TT,
        name="yarn",
    ),
]
for m in materials:
    print(" ", m)

# undegraded stiffness field — kept fixed; only matrix is degraded at runtime
C_field    = assemble_C_field_oriented(materials, phase_jax, orientations)
lam_vox, mu_vox = lame_from_C_field(C_field)   # undegraded Lamé per voxel

# ── Reference medium ──────────────────────────────────────────────────────────

def _lame(E, nu):
    return E * nu / ((1+nu)*(1-2*nu)), E / (2*(1+nu))

lam_yarn, mu_yarn = _lame(E_T, nu_TT)
lam_mat,  mu_mat  = _lame(E_m, nu_m)
lam0 = phi_yarn * lam_yarn + (1-phi_yarn) * lam_mat
mu0  = phi_yarn * mu_yarn  + (1-phi_yarn) * mu_mat
print(f"Reference: lam0={lam0/1e3:.2f} GPa  mu0={mu0/1e3:.2f} GPa")

xi_flat = build_freq_grid(n, L)
G_glob  = build_green_operator(xi_flat, lam0, mu0)

# ── PFF parameters ────────────────────────────────────────────────────────────

l0       = 0.05          # mm  phase-field length scale — set to ~2-3× voxel size
Gc_mat   = 5e-3          # MPa·mm  critical energy release rate of matrix
Gc_yarn  = 1e6 * Gc_mat  # effectively infinite — yarn never cracks

# spatially varying Gc: matrix gets Gc_mat, yarn gets Gc_yarn
Gc_field = jnp.where(matrix_mask, Gc_mat, Gc_yarn)

toler_st_abs = 1e-2
toler_st_rel = 1e-3
maxiter_st   = 200
toler_helm   = 1e-3
maxiter_helm = 300
eta          = 0.0       # viscosity — set > 0 if snap-through occurs

print(f"PFF      : l₀={l0} mm  Gc_mat={Gc_mat} MPa·mm  Gc_yarn={Gc_yarn:.1e}")

# ── Loading ───────────────────────────────────────────────────────────────────

eps_goal = jnp.array([
    [1.0e-3, 0.0, 0.0],
    [0.0,    0.0, 0.0],
    [0.0,    0.0, 0.0],
])

settings = SolverSettings(
    ndim=3,
    n=n,
    L=L,
    toler_lin=1e-4,
    toler_nw=1e-7,
    maxiter_cg=1000,
    maxiter_nw=6,
    jobname="texgen_pff",
    output="output",
)
settings.add_load_step(
    control=jnp.zeros((3, 3)),
    strain_ave_goal=eps_goal,
    stress_ave_goal=jnp.zeros((3, 3)),
    timer=(0.05, 1.0, 0.01, 0.1),
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

d_field = jnp.zeros(Nv)
H_field = jnp.zeros(Nv)

# ── Adaptive time-stepper ─────────────────────────────────────────────────────

factor_inc   = 1.5
factor_dec   = 0.5
max_cutbacks = 5
max_steps    = 200

os.makedirs(settings.output, exist_ok=True)

dt   = state.dtime
t    = state.time
step = state.kinc

phase_vis   = phase_np.reshape(n).astype(np.float32)
yarn_grid   = yarn_index_np.reshape(n).astype(np.float32)
orient_grid = orientations_np.T.reshape(*n, 3).astype(np.float32)
zero_grid   = np.zeros((*n, 6), dtype=np.float64)
zero_scal   = np.zeros(n,       dtype=np.float64)
zero_u      = np.zeros((*n, 3), dtype=np.float64)

with IncrementalWriter(
    f"{settings.output}/{settings.jobname}", grid_shape=n, grid_spacing=dx
) as w:

    w.write_increment(0, {
        "phase":             phase_vis,
        "yarn_index":        yarn_grid,
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

        t_step_start   = time.perf_counter()
        converged_mech = False

        for attempt in range(max_cutbacks + 1):

            d_st = d_field
            H_st = H_field

            for iter_st in range(1, maxiter_st + 1):
                d_prev_st = d_st

                # 1. degrade matrix stiffness only
                g     = jnp.where(matrix_mask, degradation(d_st), 1.0)
                C_eff = g[None, None, None, None, :] * C_field

                # 2. mechanical solve
                eps_i, sigma_i, delta_i, iter_mech, conv_mech = solve_elastic(
                    n, C_eff, G_glob, float(t + dt) * eps_goal,
                    toler_lin=settings.toler_lin,
                    maxiter=settings.maxiter_cg,
                )

                # 3. crack driving force — undegraded Amor split, matrix only
                psi_pos, _ = strain_energy_amor_split(eps_i, lam_vox, mu_vox)
                psi_pos    = jnp.where(matrix_mask, psi_pos, 0.0)

                # 4. update irreversibility history
                H_st = update_history(H_st, psi_pos)

                # 5. damage solve with spatially varying Gc
                d_st, iter_helm, conv_helm = solve_helmholtz_cg_het(
                    H_st, xi_flat, n, l0, Gc_field, d_prev_st,
                    toler_cg=toler_helm,
                    maxiter=maxiter_helm,
                    eta=eta,
                    dt=dt,
                )

                # 6. staggered convergence
                diff    = jnp.max(jnp.abs(d_st - d_prev_st))
                err_abs = float(diff)
                err_rel = float(diff / (jnp.max(jnp.abs(d_st)) + 1e-30))
                if err_abs < toler_st_abs or err_rel < toler_st_rel:
                    break

            converged_mech = bool(conv_mech)
            if converged_mech:
                break

            dt = max(dt * factor_dec, dt_min)
            print(f"    cutback #{attempt+1}  dt → {dt:.6f}  (mech={int(iter_mech)})")

        if not converged_mech:
            raise RuntimeError(
                f"Step {step+1}: mechanical CG did not converge after "
                f"{max_cutbacks} cutbacks at t={t:.4f}"
            )

        # ── accept increment ──────────────────────────────────────────────────
        t       += dt
        step    += 1
        d_field  = d_st
        H_field  = H_st

        eps_bar_i = float(t) * eps_goal
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
            "phase":             phase_vis,
            "yarn_index":        yarn_grid,
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
            f"  step {step:3d}  t={t:.4f}  "
            f"sig11={float(state.stress_ave[0,0]):.2f} MPa  "
            f"max(d)={float(jnp.max(d_field)):.4f}  "
            f"st={iter_st}  err_abs={err_abs:.1e}  "
            f"mech={int(iter_mech)}  helm={int(iter_helm)}  "
            f"time={step_time:.1f}s"
        )

        dt = min(dt * factor_inc, dt_max)

print(f"\nWritten → {settings.output}/{settings.jobname}.h5")
print(f"          {settings.output}/{settings.jobname}.xdmf")
print("Open the .xdmf in ParaView with the 'Xdmf3ReaderT' reader.")
