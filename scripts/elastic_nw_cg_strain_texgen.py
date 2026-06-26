"""
Elastic FFT load-stepping for a TexGen woven composite RVE.

Reads a TexGen voxel VTU export (data/test.vtu) and extracts:
  - grid geometry  (50 × 50 × 50 voxels, 2.0 × 2.0 × 0.42 mm)
  - YarnIndex      (-1 = matrix, ≥ 0 = yarn number)
  - YarnTangent    (unit fibre direction per yarn voxel)

Material assignment
-------------------
  Matrix (YarnIndex == -1) : LinearElasticIsotropic
  Yarns  (YarnIndex >= 0)  : TransverseIsotropicFibre
                             fibre direction = YarnTangent per voxel
                             (all yarns share the same elastic constants)

Usage
-----
    python scripts/elastic_nw_cg_strain_texgen.py

Output
------
    output/texgen_elastic.h5
    output/texgen_elastic.xdmf
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

from mat_models.elastic    import (LinearElasticIsotropic,
                                   TransverseIsotropicFibre,
                                   assemble_C_field_oriented)
from operators.green       import build_freq_grid, build_green_operator
from post.fields           import field_to_grid, von_mises, compute_displacement
from post.io               import IncrementalWriter, to_voigt
from solvers.types         import SolveState, SolverSettings
from solvers.elastic_nw_cg import solve_elastic

jax.config.update("jax_enable_x64", True)


# ── VTU reader ────────────────────────────────────────────────────────────────

def read_texgen_vtu(path: str):
    """
    Parse a TexGen voxel VTU (UnstructuredGrid of hexahedral cells).

    Returns
    -------
    n              : tuple (nx, ny, nz)
    L              : tuple (Lx, Ly, Lz)   physical domain size
    phase_np       : (Nv,) int             0 = matrix, 1 = yarn  (C-order)
    orientations   : (3, Nv) float         YarnTangent per voxel (C-order)
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

    centroids = pts[conn].mean(axis=1)   # (nc, 3)

    # unique sorted coordinates for each axis → grid size and spacing
    xu = np.unique(np.round(centroids[:, 0], 8))
    yu = np.unique(np.round(centroids[:, 1], 8))
    zu = np.unique(np.round(centroids[:, 2], 8))
    nx, ny, nz = len(xu), len(yu), len(zu)
    dx, dy, dz = float(xu[1] - xu[0]), float(yu[1] - yu[0]), float(zu[1] - zu[0])
    Lx, Ly, Lz = nx * dx, ny * dy, nz * dz

    # VTK cell index → (ix, iy, iz) via nearest grid point on uniform grid
    ix_arr = np.round((centroids[:, 0] - xu[0]) / dx).astype(int)
    iy_arr = np.round((centroids[:, 1] - yu[0]) / dy).astype(int)
    iz_arr = np.round((centroids[:, 2] - zu[0]) / dz).astype(int)

    # FFTjax C-order flat index for shape (nx, ny, nz): last index changes fastest
    fft_idx = ix_arr * (ny * nz) + iy_arr * nz + iz_arr

    # cell data arrays
    cd = piece.find('CellData')
    assert cd is not None, f"No <CellData> block found in {path}"
    arrays = {a.attrib['Name']: a for a in cd.findall('DataArray')}

    yarn_vtk = np.fromstring(arrays['YarnIndex'].text   or "", sep=' ').astype(int)
    tang_vtk = np.fromstring(arrays['YarnTangent'].text or "", sep=' ').reshape(-1, 3)

    Nv           = nx * ny * nz
    phase_flat   = np.empty(Nv, dtype=int)
    tangent_flat = np.zeros((Nv, 3))

    phase_flat[fft_idx]   = yarn_vtk        # -1 = matrix, ≥0 = yarn
    tangent_flat[fft_idx] = tang_vtk        # unit fibre direction (zeros for matrix)

    # remap: 0 = matrix, 1 = yarn  (required by assemble_C_field_oriented)
    phase_np     = np.where(phase_flat >= 0, 1, 0)
    orientations = tangent_flat.T.copy()    # (3, Nv)

    return (nx, ny, nz), (Lx, Ly, Lz), phase_np, orientations


# ── Load VTU ──────────────────────────────────────────────────────────────────

vtu_path = "data/160_nn_2.vtu"
n, L, phase_np, orientations_np = read_texgen_vtu(vtu_path)

Nv  = int(np.prod(n))
dx  = tuple(Li / ni for Li, ni in zip(L, n))

print(f"Grid     : {n}   Nv = {Nv}")
print(f"Domain   : {tuple(f'{v:.4g}' for v in L)}  mm")
print(f"Spacing  : {tuple(f'{v:.4g}' for v in dx)}  mm")

phase_jax    = jnp.array(phase_np)
orientations = jnp.array(orientations_np)

phi_yarn = float(np.mean(phase_np))
print(f"Yarn vol. fraction: {phi_yarn:.3f}")

# ── Materials ─────────────────────────────────────────────────────────────────
# Adjust constants to match your actual fibre/matrix system (units: MPa).

materials = [
    LinearElasticIsotropic(E=3.5e3, nu=0.35, 
                           name="epoxy matrix"
                           ),
    TransverseIsotropicFibre(
        E_L=230e3, E_T=15e3, G_LT=15e3, nu_LT=0.20, nu_TT=0.30,
        name="yarn",
    ),
]



for m in materials:
    print(" ", m)

C_field = assemble_C_field_oriented(materials, phase_jax, orientations)

# ── Reference medium (Voigt average, transverse yarn constants) ───────────────

def _lame(E, nu):
    return E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))

yarn = materials[1]
mat  = materials[0]
lam_yarn, mu_yarn = _lame(yarn.E_T, yarn.nu_TT)
lam0 = phi_yarn * lam_yarn + (1 - phi_yarn) * mat.lam
mu0  = phi_yarn * mu_yarn  + (1 - phi_yarn) * mat.mu

print(f"Reference: lam0={lam0/1e3:.2f} GPa  mu0={mu0/1e3:.2f} GPa")

xi_flat = build_freq_grid(n, L)
G_glob  = build_green_operator(xi_flat, lam0, mu0)

# ── Loading ───────────────────────────────────────────────────────────────────
# Uniaxial macroscopic strain in X1 direction; edit eps_goal as needed.

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
    jobname="texgen_elastic",
    output="output",
)
settings.add_load_step(
    control=jnp.zeros((3, 3)),
    strain_ave_goal=eps_goal,
    stress_ave_goal=jnp.zeros((3, 3)),
    timer=(0.1, 1.0, 0.05, 0.2),
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

# ── Adaptive time-stepper ─────────────────────────────────────────────────────

factor_inc   = 1.5
factor_dec   = 0.5
max_cutbacks = 5
max_steps    = 20

os.makedirs(settings.output, exist_ok=True)

dt   = state.dtime
t    = state.time
step = state.kinc

phase_vis   = phase_np.reshape(n).astype(np.float32)
orient_grid = orientations_np.T.reshape(*n, 3).astype(np.float32)
zero_grid   = np.zeros((*n, 6), dtype=np.float64)
zero_scal   = np.zeros(n,       dtype=np.float64)
zero_u      = np.zeros((*n, 3), dtype=np.float64)

with IncrementalWriter(
    f"{settings.output}/{settings.jobname}", grid_shape=n, grid_spacing=dx
) as w:

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
            eps_i, sigma_i, delta_i, iter_mech, conv_mech = solve_elastic(
                n, C_field, G_glob, eps_bar_i,
                toler_lin=settings.toler_lin,
                maxiter=settings.maxiter_cg,
            )
            converged = bool(conv_mech)
            if converged:
                break
            dt = max(dt * factor_dec, dt_min)
            print(f"    cutback #{attempt + 1}  dt → {dt:.6f}  (iter={int(iter_mech)})")

        if not converged:
            raise RuntimeError(
                f"Step {step + 1} did not converge after {max_cutbacks} cutbacks "
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
            info=0 if converged else int(iter_mech),
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
              f"sig11={float(state.stress_ave[0, 0]):.3f} MPa  "
              f"CG={int(iter_mech)}  time={step_time:.2f}s")

        dt = min(dt * factor_inc, dt_max)

print(f"\nWritten → {settings.output}/{settings.jobname}.h5")
print(f"          {settings.output}/{settings.jobname}.xdmf")
print("Open the .xdmf in ParaView with the 'Xdmf3ReaderT' reader.")
