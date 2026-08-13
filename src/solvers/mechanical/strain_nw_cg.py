import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)

from typing import Tuple
from functools import partial
from math import prod

import jax.numpy as jnp
import numpy as np
from jax import jit

from solvers.krylov.cg import cg_solve


@partial(jit, static_argnames=("n_i", "maxiter"))
def dstrain_nw_cg(
    n_i:        Tuple,
    C_field:    jnp.ndarray,
    G_glob:     jnp.ndarray,
    eps_bar:    jnp.ndarray,
    stress_goal: jnp.ndarray | None = None,
    toler_lin:  float = 1e-4,
    maxiter:    int   = 1000,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Inner CG solve for one Newton step of the variational FFT elastic solver
    (small strains, Vondrejc / Lucarini–Segurado formulation) -- the
    Krylov-based (CG-accelerated) Lippmann-Schwinger scheme: trigonometric
    collocation (Zeman et al 2010; Vondrejc et al 2012, electrostatics) of
    Moulinec & Suquet's (1994, 1998) Lippmann-Schwinger equation, solved by
    CG instead of their original fixed-point "basic scheme". Extended to
    elasticity by Vondrejc et al (2014) -- matching this solver -- and
    further by Lucarini & Segurado (2018).

    Solves the linear system
        A(Δε) = b
    where
        A(v)  = iFFT( G̃ : FFT( C : v ) )     [Green–stiffness operator]
        b     = -iFFT( G̃ : FFT( C:ε₀ − σ_goal ) )   [projected residual]
        ε₀    = eps_bar broadcast to all voxels   [uniform initial guess]

    Parameters
    ----------
    n_i         : grid shape (nx, ny, nz) — must be static for JIT
    C_field     : (3, 3, 3, 3, Nv)      per-voxel stiffness (or tangent)
    G_glob      : (3, 3, 3, 3, Nv)      Green's operator in Fourier space
    eps_bar     : (3, 3)                prescribed macroscopic strain
    stress_goal : (3, 3, Nv) or None target stress field (None = zero, strain BC)
    toler_lin   : relative CG residual tolerance
    maxiter     : maximum CG iterations — must be static for JIT

    Returns
    -------
    eps        : (3, 3, Nv)   updated local strain  ε = ε₀ + Δε
    sigma      : (3, 3, Nv)   updated local stress  σ = C : ε
    delta      : (3, 3, Nv)   strain correction     Δε
    converged  : bool array   True if residual tolerance met
    """
    Nv = prod(n_i)

    # ── FFT helpers ───────────────────────────────────────────────────────────
    def fft_(x):
        s = x.shape
        return jnp.fft.fftn(x.reshape(s[:-1] + n_i), axes=(-3, -2, -1)).reshape(s)

    def ifft_(x):
        s = x.shape
        return jnp.fft.ifftn(x.reshape(s[:-1] + n_i), axes=(-3, -2, -1)).real.reshape(s)

    # ── Linear operator  A(v) = iFFT( G̃ : FFT( C : v ) ) ───────────────────
    def A_op(v_flat):
        v   = v_flat.reshape(3, 3, Nv)
        Cv  = jnp.einsum("ijklm,klm->ijm", C_field, v)
        GCv = jnp.einsum("ijklm,klm->ijm", G_glob, fft_(Cv))
        return ifft_(GCv).reshape(-1)

    # ── RHS  b = -iFFT( G̃ : FFT( C:ε₀ − σ_goal ) ) ────────────────────────
    sg     = jnp.zeros((3, 3, Nv)) if stress_goal is None else stress_goal
    eps0   = jnp.ones((3, 3, Nv)) * eps_bar[:, :, None]
    sigma0 = jnp.einsum("ijklm,klm->ijm", C_field, eps0)
    res0   = fft_(sigma0 - sg)
    bb     = -ifft_(jnp.einsum("ijklm,klm->ijm", G_glob, res0)).reshape(-1)

    # ── CG solve ──────────────────────────────────────────────────────────────
    x0 = jnp.zeros_like(bb)
    delta_flat, converged = cg_solve(A_op, bb, x0, toler_lin, maxiter)

    delta = delta_flat.reshape(3, 3, Nv)
    eps   = eps0 + delta
    sigma = jnp.einsum("ijklm,klm->ijm", C_field, eps)

    return eps, sigma, delta, converged


# simple alias — prefer dstrain_nw_cg in solver code, solve_elastic in scripts
solve_elastic = dstrain_nw_cg


# @partial(jit, static_argnames=("n_i", "control", "maxiter"))
# def dstrain_nw_cg_mixed(
#     n_i:        Tuple,
#     C_field:    jnp.ndarray,
#     G_glob:     jnp.ndarray,
#     eps_bar:    jnp.ndarray,
#     control:    Tuple[Tuple[int, ...], ...],
#     stress_ave_goal: jnp.ndarray,
#     toler_lin:  float = 1e-4,
#     maxiter:    int   = 1000,
# ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
#     """
#     Variational Newton-CG elastic solve with mixed macroscopic strain/stress
#     boundary conditions (Vondrejc et al. 2014 / Lucarini & Segurado 2018,
#     FFTMAD's ``variational_nw_cg_*`` family).

#     Unlike ``ddisp_nw_cg`` (which solves for an entirely separate displacement
#     field with its own preconditioner), this reuses the existing strain-based
#     Green's-operator CG solver almost unchanged. The trick: ``G_glob`` is
#     exactly zero at the zero-frequency (DC) point in the pure-strain scheme,
#     since the mean strain is fully prescribed there and the CG correction
#     never touches it. Overriding that otherwise-unused DC block with
#     ``I4s : control`` reclaims it as a free unknown exactly for the
#     stress-controlled tensor components — the correction field's own mean
#     value at those components becomes the solved macroscopic-strain
#     correction, with no extra unknowns, no separate preconditioner, and no
#     change to the well-conditioned Γ0-based operator elsewhere.

#     **HOMOGENEOUS MATERIALS ONLY as a one-shot solve.** The DC-override CG
#     operator is only symmetric (hence only reliably CG-solvable in one shot)
#     when ``mean(C:v) == C0:mean(v)`` for the trial field ``v`` — true for a
#     spatially constant ``C_field``, false in general for a heterogeneous one
#     (verified directly: for a two-phase composite the operator's symmetry
#     error is ~80%, and CG diverges to nonsense). FFTMAD's own reference
#     implementation only uses this trick *inside* an outer Newton iteration
#     that repeatedly re-solves and updates the stress target — never as a
#     single CG pass — which is not what this function does. Use
#     ``ddisp_nw_cg`` for heterogeneous materials with mixed BC instead.

#     Parameters
#     ----------
#     n_i         : grid shape (nx, ny, nz) — must be static for JIT
#     C_field     : (3, 3, 3, 3, Nv)  per-voxel stiffness (or tangent)
#     G_glob      : (3, 3, 3, 3, Nv)  Green's operator in Fourier space
#                   (its own DC block, index -1 == 0, is overridden internally —
#                   pass the same G_glob used for pure-strain BC)
#     eps_bar     : (3, 3)  prescribed macroscopic strain; entries where
#                   ``control == 1`` are ignored (solved for instead)
#     control     : (3, 3) static 0/1 mask, 1 = stress-controlled, 0 = strain-controlled
#     stress_ave_goal : (3, 3)  target macroscopic stress; only entries where
#                   ``control == 1`` are used
#     toler_lin   : relative CG residual tolerance
#     maxiter     : maximum CG iterations — must be static for JIT

#     Returns
#     -------
#     eps        : (3, 3, Nv)   updated local strain
#     sigma      : (3, 3, Nv)   updated local stress  σ = C : ε
#     delta      : (3, 3, Nv)   strain correction (fluctuation + macroscopic part)
#     eps_bar_out: (3, 3)       macroscopic strain, with stress-controlled
#                  entries filled in by the solve
#     converged  : bool array   True if residual tolerance met
#     """
#     Nv = prod(n_i)
#     control_arr = jnp.asarray(control, dtype=eps_bar.dtype)

#     I2  = jnp.eye(3, dtype=G_glob.dtype)
#     I4s = 0.5 * (jnp.einsum("ik,jl->ijkl", I2, I2)
#                  + jnp.einsum("il,jk->ijkl", I2, I2))
#     G_mixed = G_glob.at[:, :, :, :, 0].set(
#         jnp.einsum("ijkl,kl->ijkl", I4s, control_arr)
#     )

#     # ── FFT helpers ───────────────────────────────────────────────────────────
#     def fft_(x):
#         s = x.shape
#         return jnp.fft.fftn(x.reshape(s[:-1] + n_i), axes=(-3, -2, -1)).reshape(s)

#     def ifft_(x):
#         s = x.shape
#         return jnp.fft.ifftn(x.reshape(s[:-1] + n_i), axes=(-3, -2, -1)).real.reshape(s)

#     # ── Linear operator  A(v) = iFFT( G_mixed : FFT( C : v ) ) ─────────────
#     def A_op(v_flat):
#         v   = v_flat.reshape(3, 3, Nv)
#         Cv  = jnp.einsum("ijklm,klm->ijm", C_field, v)
#         GCv = jnp.einsum("ijklm,klm->ijm", G_mixed, fft_(Cv))
#         return ifft_(GCv).reshape(-1)

#     # ── RHS  b = -iFFT( G_mixed : FFT( C:ε₀ − σ_goal ) ) ───────────────────
#     eps0   = jnp.ones((3, 3, Nv)) * (eps_bar * (1.0 - control_arr))[:, :, None]
#     sg     = jnp.ones((3, 3, Nv)) * (stress_ave_goal * control_arr)[:, :, None]
#     sigma0 = jnp.einsum("ijklm,klm->ijm", C_field, eps0)
#     res0   = fft_(sigma0 - sg)
#     bb     = -ifft_(jnp.einsum("ijklm,klm->ijm", G_mixed, res0)).reshape(-1)

#     # ── CG solve ──────────────────────────────────────────────────────────────
#     x0 = jnp.zeros_like(bb)
#     delta_flat, converged = cg_solve(A_op, bb, x0, toler_lin, maxiter)

#     delta = delta_flat.reshape(3, 3, Nv)
#     eps   = eps0 + delta
#     sigma = jnp.einsum("ijklm,klm->ijm", C_field, eps)
#     eps_bar_out = eps_bar * (1.0 - control_arr) + jnp.mean(delta, axis=-1) * control_arr

#     return eps, sigma, delta, eps_bar_out, converged
