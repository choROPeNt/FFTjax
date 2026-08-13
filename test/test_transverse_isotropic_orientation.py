"""
Test script for TransverseIsotropicFibre with uniform orientation in the XY plane.

Single-phase 128×128×1 RVE where every voxel has the same fibre direction
    d = [cos(alpha), sin(alpha), 0]
with alpha in degrees (default 45°).

For a spatially uniform material, the FFT solver must return a perfectly
uniform stress/strain field.  We verify this and compare the macroscopic
stress against the analytical result  σ̄ = C_rotated : ε̄.

Usage
-----
    python scripts/test_transverse_isotropic_orientation.py
    python scripts/test_transverse_isotropic_orientation.py --alpha 30
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
sys.path.insert(0, "src")

import argparse
import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)
import jax
import jax.numpy as jnp
import numpy as np

from mat_models.elastic    import TransverseIsotropicFibre, assemble_C_field_oriented
from operators.green       import build_freq_grid, GreenOperatorBasic
from post.fields           import field_to_grid, von_mises, compute_displacement
from utils.io.xdmf_writer import IncrementalWriter
from post.fields          import to_voigt
from solvers.elliptic.vector.lippmann_schwinger import solve_lippmann_schwinger

jax.config.update("jax_enable_x64", True)

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, default=45.0,
                    help="Fibre angle in degrees, measured from X toward Y (default 45)")
args = parser.parse_args()

alpha_deg = args.alpha
alpha_rad = float(np.deg2rad(alpha_deg))

# ── Grid ──────────────────────────────────────────────────────────────────────

n  = (128, 128, 1)
L  = (1.0, 1.0, 1.0)
Nv = int(np.prod(n))

# ── Material — single-phase, all fibres at angle alpha in XY ─────────────────
# Carbon fibre / epoxy-like constants (MPa)
mat = TransverseIsotropicFibre(
    E_L=230e3, E_T=15e3, G_LT=15e3, nu_LT=0.20, nu_TT=0.30,
    name="carbon fibre",
)

# Fibre direction: rotate e_z toward X by alpha in the XY plane
#   d = [cos(alpha), sin(alpha), 0]
d = jnp.array([np.cos(alpha_rad), np.sin(alpha_rad), 0.0])

phase        = jnp.zeros(Nv, dtype=int)          # single phase, index 0
orientations = jnp.stack([d] * Nv, axis=-1)      # (3, Nv) — same direction everywhere

C_field = assemble_C_field_oriented([mat], phase, orientations)  # (3,3,3,3, Nv)

# ── Reference medium — isotropic, from transverse-plane Lamé constants ────────
# For a single-phase problem the choice only affects CG convergence speed,
# not the solution.  We use the transverse (E_T, nu_TT) constants.
def _lame(E, nu):
    return E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))

lam0, mu0 = _lame(mat.E_T, mat.nu_TT)

xi_flat = build_freq_grid(n, L)
green_op = GreenOperatorBasic(n, L, lam0, mu0)

# ── Macroscopic strain — uniaxial in X ───────────────────────────────────────
eps_bar = jnp.array([
    [1.0e-3, 0.0, 0.0],
    [0.0,    0.0, 0.0],
    [0.0,    0.0, 0.0],
])

# ── Solve ─────────────────────────────────────────────────────────────────────
print(f"\nfibre angle  alpha = {alpha_deg}°")
print(f"fibre dir    d     = [{d[0]:.4f}, {d[1]:.4f}, {d[2]:.4f}]")
print(f"grid               = {n}   (Nv = {Nv})")
print(f"reference medium   lam0={lam0/1e3:.2f} GPa  mu0={mu0/1e3:.2f} GPa")
print(f"applied strain     ε₁₁ = {float(eps_bar[0,0]):.2e}")
print()

eps_loc, sigma_loc, delta, conv_mech = solve_lippmann_schwinger(
    n, C_field, green_op, eps_bar, toler_lin=1e-10, maxiter=2000
)

print(f"converged : {bool(conv_mech)}")

# ── Verify field uniformity ───────────────────────────────────────────────────
# For a spatially uniform stiffness the solution must be the uniform field
# eps_loc = eps_bar everywhere and stress follows σ = C_rot : eps_bar.
# Relative std should be at numerical precision (~1e-12).

eps_mean  = jnp.mean(eps_loc,   axis=-1)   # (3,3)
sigma_mean = jnp.mean(sigma_loc, axis=-1)  # (3,3)

eps_std   = jnp.std(eps_loc,   axis=-1)
sigma_std = jnp.std(sigma_loc, axis=-1)

print("── Field uniformity (single-phase → should be ~0) ──────────────────")
print(f"  max |std(ε)| / |ε̄₁₁|  = {float(jnp.max(eps_std))   / abs(float(eps_bar[0,0])):.2e}")
print(f"  max |std(σ)| / |σ̄₁₁|  = {float(jnp.max(sigma_std)) / abs(float(sigma_mean[0,0])):.2e}")

# ── Compare against analytical solution ──────────────────────────────────────
# σ̄_analytical = C_rotated : ε̄
C_rotated  = mat.stiffness_tensor_rotated(d)                       # (3,3,3,3)
sigma_anal = jnp.einsum("ijkl,kl->ij", C_rotated, eps_bar)        # (3,3)

err_sigma  = jnp.max(jnp.abs(sigma_mean - sigma_anal))
norm_sigma = jnp.max(jnp.abs(sigma_anal))

print()
print("── Macroscopic stress  σ̄ (MPa) ─────────────────────────────────────")
labels = [("11","22","33","12","13","23")[k]
          for k in range(6)]
voigt_idx = [(0,0),(1,1),(2,2),(0,1),(0,2),(1,2)]
for lbl, (i,j) in zip(labels, voigt_idx):
    s_fft  = float(sigma_mean[i,j])
    s_anal = float(sigma_anal[i,j])
    print(f"  σ{lbl}   FFT={s_fft:+.4f}   analytical={s_anal:+.4f}   err={abs(s_fft-s_anal):.2e}")

print()
print(f"  max component error : {float(err_sigma):.2e} MPa  "
      f"( {float(err_sigma/norm_sigma)*100:.4f}% of max |σ_anal| )")

print()
if float(err_sigma / norm_sigma) < 1e-6 and bool(conv_mech):
    print("PASS — stress matches analytical solution within 1e-6 relative tolerance")
else:
    print("FAIL — check CG convergence or stiffness rotation implementation")

# ── Write output ──────────────────────────────────────────────────────────────
dx          = tuple(Li / ni for Li, ni in zip(L, n))
jobname     = f"ti_orientation_alpha{int(alpha_deg):03d}"
output_dir  = "output"
os.makedirs(output_dir, exist_ok=True)

eps_grid    = field_to_grid(eps_loc,   n)       # (*n, 3, 3)
sigma_grid  = field_to_grid(sigma_loc, n)       # (*n, 3, 3)
u_grid      = compute_displacement(eps_loc, eps_bar, n, L)  # (*n, 3)
orient_grid = np.asarray(orientations).T.reshape(*n, 3).astype(np.float32)

# ── Ply-frame stress  σ_ply = R_ply^T · σ_global · R_ply ─────────────────────
# Ply axes:  e1 = fibre d,  e2 = transverse in-plane,  e3 = thickness (Z)
# R_ply columns are the ply basis vectors expressed in the global frame.
# For a ply at angle alpha in XY this is a pure rotation about Z.
ca, sa  = float(np.cos(alpha_rad)), float(np.sin(alpha_rad))
R_ply   = jnp.array([[ ca, -sa, 0.0],
                      [ sa,  ca, 0.0],
                      [0.0, 0.0, 1.0]])   # (3,3)  global ← ply

# σ_ply_ij = Σ_{ab} R_ply_{ai} σ_global_{ab} R_ply_{bj}   (contract both indices)
sigma_ply_loc  = jnp.einsum("ai,abm,bj->ijm", R_ply, sigma_loc, R_ply)  # (3,3,Nv)
sigma_ply_grid = field_to_grid(sigma_ply_loc, n)                         # (*n,3,3)

with IncrementalWriter(f"{output_dir}/{jobname}", grid_shape=n, grid_spacing=dx) as w:

    # ── write undeformed initial state at t = 0 ───────────────────────────────
    zero_grid = np.zeros((*n, 6), dtype=np.float64)
    zero_scal = np.zeros(n,       dtype=np.float64)
    zero_u    = np.zeros((*n, 3), dtype=np.float64)
    w.write_increment(0, {
        "orientation":  orient_grid,
        "displacement": zero_u,
        "strain":       zero_grid,
        "stress":       zero_grid,
        "stress_ply":   zero_grid,
        "von_mises":    zero_scal,
    }, time=0.0)

    # ── write solved state at t = 1 ───────────────────────────────────────────
    w.write_increment(1, {
        "orientation":  orient_grid,
        "displacement": u_grid.astype(np.float64),
        "strain":       to_voigt(eps_grid).astype(np.float64),
        "stress":       to_voigt(sigma_grid).astype(np.float64),
        "stress_ply":   to_voigt(sigma_ply_grid).astype(np.float64),
        "von_mises":    von_mises(sigma_grid).astype(np.float64),
    }, time=1.0)

print(f"\nWritten → {output_dir}/{jobname}.h5")
print(f"          {output_dir}/{jobname}.xdmf")
