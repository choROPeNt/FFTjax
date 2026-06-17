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
from jax import jit
from jax.scipy.sparse.linalg import cg as jax_cg
from functools import partial
import time
import numpy as np

from generation.rve              import make_square_composite_rve
from mat_models.elastic          import (LinearElasticIsotropic,
                                         TransverseIsotropicFibre,
                                         assemble_C_field_oriented)
from operators.green             import build_freq_grid, build_green_operator
from post.fields                 import field_to_grid, von_mises, compute_displacement
from post.io                     import IncrementalWriter, to_voigt

jax.config.update("jax_enable_x64", True)

# ── Solver ────────────────────────────────────────────────────────────────────

@partial(jit, static_argnames=("n_i", "maxiter"))
def solve_elastic(n_i, C_field, G_glob, eps_bar,
                  stress_goal=None, toler_lin=1e-10, maxiter=1000):
    Nv = int(np.prod(n_i))

    def fft_(x, _n=n_i):
        s = x.shape
        return jnp.fft.fftn(x.reshape(s[:-1] + _n), axes=(-3, -2, -1)).reshape(s)

    def ifft_(x, _n=n_i):
        s = x.shape
        return jnp.fft.ifftn(x.reshape(s[:-1] + _n), axes=(-3, -2, -1)).real.reshape(s)

    def A_op(v_flat, _C=C_field, _G=G_glob, _Nv=Nv, _f=fft_, _if=ifft_):
        v = v_flat.reshape(3, 3, _Nv)
        return _if(jnp.einsum("ijklm,klm->ijm", _G,
                   _f(jnp.einsum("ijklm,klm->ijm", _C, v)))).reshape(-1)

    sg     = jnp.zeros((3, 3, Nv)) if stress_goal is None else stress_goal
    eps0   = jnp.ones((3, 3, Nv)) * eps_bar[:, :, None]
    sigma0 = jnp.einsum("ijklm,klm->ijm", C_field, eps0)
    bb     = -ifft_(jnp.einsum("ijklm,klm->ijm", G_glob, fft_(sigma0 - sg))).reshape(-1)

    delta, info = jax_cg(A_op, bb, tol=toler_lin, maxiter=maxiter)
    eps   = eps0 + delta.reshape(3, 3, Nv)
    sigma = jnp.einsum("ijklm,klm->ijm", C_field, eps)
    return eps, sigma, info


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

print(f"RVE: N={N}  nz={nz}  phi={phi_:.3f}")
print(f"Reference medium: lam0={lam0/1e3:.2f} GPa  mu0={mu0/1e3:.2f} GPa")
for m in materials:
    print(" ", m)

# ── Load-step parameters ──────────────────────────────────────────────────────

eps_goal = jnp.array([
    [0.0,  0.0, 0.0],
    [0.0, 0.0,  0.0],
    [0.0,  0.0,  1.0e-3],
])  # pure shear  (ε̄₁₂ = ε̄₂₁ = 1×10⁻³ at t = 1)

# ── Increment control  [initial_dt, t_max, min_dt, max_dt, max_steps] ─────────
# Mirrors the ABAQUS *STATIC increment line:
#   initial_dt  first time increment
#   t_max       total load factor to reach  (1.0 = full eps_goal)
#   min_dt      smallest allowed increment
#   max_dt      largest allowed increment
#   max_steps   hard cap on number of increments
initial_dt = 0.1
t_max      = 1.0
min_dt     = 0.05
max_dt     = 0.2
max_steps  = 20

# ── Load-stepping loop ────────────────────────────────────────────────────────

os.makedirs("output", exist_ok=True)

# ── Variable time-stepper parameters ─────────────────────────────────────────
factor_inc   = 1.5   # multiply dt by this after a converged step
factor_dec   = 0.5   # multiply dt by this after a failed attempt
max_cutbacks = 5     # max dt reductions per step before aborting

# ── Load-stepping loop with adaptive dt ───────────────────────────────────────
t    = 0.0
dt   = initial_dt
step = 0

with IncrementalWriter("output/rve_loadsteps", grid_shape=n, grid_spacing=dx) as w:
    while t < t_max and step < max_steps:

        # clamp dt to allowed window and don't overshoot t_max
        dt = float(np.clip(dt, min_dt, max_dt))
        dt = min(dt, t_max - t)

        # ── attempt the increment, cut back on failure ──────────────────────
        converged = False
        t_step_start = time.perf_counter()
        for attempt in range(max_cutbacks + 1):
            eps_bar_i             = float(t + dt) * eps_goal
            eps_i, sigma_i, info  = solve_elastic(n, C_field, G_glob, eps_bar_i)
            converged             = (info is None or info == 0)

            if converged:
                break

            dt = max(dt * factor_dec, min_dt)
            print(f"    cutback #{attempt + 1}  dt → {dt:.6f}  (CG info={info})")

        if not converged:
            raise RuntimeError(
                f"Step {step + 1} did not converge after {max_cutbacks} cutbacks "
                f"at t={t:.4f}, min_dt={min_dt}"
            )

        # ── accept the increment ────────────────────────────────────────────
        t    += dt
        step += 1

        sigma_bar  = jnp.mean(sigma_i, axis=-1)
        eps_grid   = field_to_grid(eps_i,   n)
        sigma_grid = field_to_grid(sigma_i, n)
        u_grid     = compute_displacement(eps_i, eps_bar_i, xi_flat, n, dx)
        # Orientation vector reshaped to (*n, 3) for ParaView Vector attribute
        orient_grid = np.asarray(orientations).T.reshape(*n, 3).astype(np.float32)

        w.write_increment(step - 1, {
            "phase":        phase_np.astype(np.float32),
            "orientation":  orient_grid,
            "displacement": u_grid.astype(np.float64),
            "strain":       to_voigt(eps_grid).astype(np.float64),
            "stress":       to_voigt(sigma_grid).astype(np.float64),
            "von_mises":    von_mises(sigma_grid).astype(np.float64),
        }, time=float(t))

        step_time = time.perf_counter() - t_step_start
        print(f"  step {step:2d}  t={t:.4f}  dt={dt:.4f}  "
              f"sig12={float(sigma_bar[0, 1]):.3f} MPa  CG={info}  "
              f"time={step_time:.2f}s")

        # ── suggest next dt based on convergence quality ────────────────────
        # converged cleanly → grow dt; will be clamped to max_dt next iteration
        dt = min(dt * factor_inc, max_dt)

print(f"\nWritten → output/rve_loadsteps.h5")
print(f"          output/rve_loadsteps.xdmf")
print("Open the .xdmf in ParaView with the 'Xdmf3ReaderT' reader.")
