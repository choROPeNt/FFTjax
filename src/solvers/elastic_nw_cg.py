import os
os.environ["JAX_ENABLE_X64"] = "1"

from functools import partial

import jax.numpy as jnp
import numpy as np
from jax import jit
from jax.scipy.sparse.linalg import cg as jax_cg


@partial(jit, static_argnames=("n_i", "maxiter"))
def dstrain_nw_cg(
    n_i:        tuple,
    C_field:    jnp.ndarray,
    G_glob:     jnp.ndarray,
    eps_bar:    jnp.ndarray,
    stress_goal: jnp.ndarray | None = None,
    toler_lin:  float = 1e-4,
    maxiter:    int   = 1000,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, int | None]:
    """
    Inner CG solve for one Newton step of the variational FFT elastic solver
    (small strains, Vondrejc / Lucarini–Segurado formulation).

    Solves the linear system
        A(Δε) = b
    where
        A(v)  = iFFT( G̃ : FFT( C : v ) )     [Green–stiffness operator]
        b     = -iFFT( G̃ : FFT( C:ε₀ − σ_goal ) )   [projected residual]
        ε₀    = eps_bar broadcast to all voxels   [uniform initial guess]

    The solution Δε is the local strain correction; adding it to ε₀ gives
    a strain field that satisfies both equilibrium and the macroscopic BC.

    Parameters
    ----------
    n_i         : grid shape (nx, ny, nz) — must be static for JIT
    C_field     : (3, 3, 3, 3, Nv)  per-voxel stiffness (or tangent)
    G_glob      : (3, 3, 3, 3, Nv)  Green's operator in Fourier space
    eps_bar     : (3, 3)             prescribed macroscopic strain
    stress_goal : (3, 3, Nv) or None target stress field (None = zero, strain BC)
    toler_lin   : relative CG residual tolerance
    maxiter     : maximum CG iterations — must be static for JIT

    Returns
    -------
    eps   : (3, 3, Nv)  updated local strain  ε = ε₀ + Δε
    sigma : (3, 3, Nv)  updated local stress  σ = C : ε
    delta : (3, 3, Nv)  strain correction     Δε  (CG solution)
    info  : int         CG exit flag — 0 = converged
    """
    Nv = int(np.prod(n_i))

    # ── FFT helpers operating on flat voxel arrays (..., Nv) ─────────────────
    def fft_(x):
        s = x.shape
        return jnp.fft.fftn(x.reshape(s[:-1] + n_i), axes=(-3, -2, -1)).reshape(s)

    def ifft_(x):
        s = x.shape
        return jnp.fft.ifftn(x.reshape(s[:-1] + n_i), axes=(-3, -2, -1)).real.reshape(s)

    # ── Linear operator  A(v) = iFFT( G̃ : FFT( C : v ) ) ───────────────────
    def A_op(v_flat):
        v = v_flat.reshape(3, 3, Nv)
        Cv   = jnp.einsum("ijklm,klm->ijm", C_field, v)    # C : v
        GCv  = jnp.einsum("ijklm,klm->ijm", G_glob, fft_(Cv))  # G̃ : F(Cv)
        return ifft_(GCv).reshape(-1)

    # ── Right-hand side  b = -iFFT( G̃ : FFT( C:ε₀ − σ_goal ) ) ────────────
    sg     = jnp.zeros((3, 3, Nv)) if stress_goal is None else stress_goal
    eps0   = jnp.ones((3, 3, Nv)) * eps_bar[:, :, None]    # uniform initial field
    sigma0 = jnp.einsum("ijklm,klm->ijm", C_field, eps0)   # C : ε₀
    res0   = fft_(sigma0 - sg)                              # FFT of stress residual
    bb     = -ifft_(jnp.einsum("ijklm,klm->ijm", G_glob, res0)).reshape(-1)

    # ── CG solve for Δε ───────────────────────────────────────────────────────
    delta_flat, info = jax_cg(A_op, bb, tol=toler_lin, maxiter=maxiter)

    delta = delta_flat.reshape(3, 3, Nv)
    eps   = eps0 + delta
    sigma = jnp.einsum("ijklm,klm->ijm", C_field, eps)

    return eps, sigma, delta, info


# simple alias — prefer dstrain_nw_cg in solver code, solve_elastic in scripts
solve_elastic = dstrain_nw_cg
