import os
os.environ["JAX_ENABLE_X64"] = "1"

from typing import Tuple
from functools import partial
from math import prod

import jax.numpy as jnp
from jax import jit

from utils.cg import cg_count


def _active_pairs(control: Tuple[Tuple[int, ...], ...]) -> Tuple[Tuple[int, int], ...]:
    """Upper-triangular (i, j) index pairs, i<=j, where control[i][j] == 1."""
    return tuple(
        (i, j)
        for i in range(3)
        for j in range(i, 3)
        if control[i][j]
    )


def _nyquist_safe_xi(xi_flat: jnp.ndarray, n_i: Tuple) -> jnp.ndarray:
    """
    Zero the Nyquist-frequency component of ``xi_flat`` along each even grid
    dimension.

    The gradient/divergence operators below use ``xi`` to an odd power
    (unlike the strain-based solver's Green's operator, which only ever
    uses even powers ``ξξ``/``ξξξξ``). For an even-length dimension the
    Nyquist bin has no distinct negative-frequency partner, so its FFT
    coefficient must be real for the transform of a real signal to stay
    Hermitian-symmetric — multiplying it by an odd (purely imaginary) power
    of ``ξ`` breaks that symmetry and corrupts the real-space result.
    Zeroing it there is the standard fix used throughout FFT-Galerkin
    homogenization schemes.
    """
    idx   = [jnp.arange(ni) for ni in n_i]
    grids = jnp.meshgrid(*idx, indexing="ij")
    masks = [
        (g == ni // 2) if ni % 2 == 0 else jnp.zeros_like(g, dtype=bool)
        for ni, g in zip(n_i, grids)
    ]
    nyquist_mask = jnp.stack([m.ravel() for m in masks])
    return jnp.where(nyquist_mask, 0.0, xi_flat)


@partial(jit, static_argnames=("n_i", "control", "maxiter"))
def ddisp_nw_cg(
    n_i:        Tuple,
    C_field:    jnp.ndarray,
    xi_flat:    jnp.ndarray,
    eps_bar:    jnp.ndarray,
    control:    Tuple[Tuple[int, ...], ...],
    stress_goal: jnp.ndarray,
    toler_lin:  float = 1e-4,
    maxiter:    int   = 1000,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Displacement-based Newton-CG solve for one Newton step of the variational
    FFT elastic solver, supporting mixed strain/stress macroscopic boundary
    conditions (FFTMAD's ``displacement_nw_cg_small`` / ``du_loc_nw_CG``).

    Unlike the strain-based ``dstrain_nw_cg`` (which uses a fixed
    reference-medium Green's operator), this solves directly for a periodic
    displacement fluctuation field via the true (possibly heterogeneous)
    tangent stiffness ``C_field``, with no reference-medium approximation.
    The macroscopic-average strain correction for stress-controlled directions
    is solved for jointly with the displacement fluctuation, by embedding it
    in the (otherwise physically meaningless) zero-frequency mode of the
    Fourier-space strain field — the same trick used by the reference
    implementation, adapted here to a real-space CG unknown (full ``fftn``/
    ``ifftn``, no Hermitian-symmetry DOF reduction).

    Parameters
    ----------
    n_i         : grid shape (nx, ny, nz) — must be static for JIT
    C_field     : (3, 3, 3, 3, Nv)  per-voxel tangent stiffness
    xi_flat     : (3, Nv)           angular-frequency grid (``operators.green.build_freq_grid``)
    eps_bar     : (3, 3)  prescribed macroscopic strain; entries where
                  ``control == 1`` are ignored (solved for instead)
    control     : (3, 3) static 0/1 mask, 1 = stress-controlled, 0 = strain-controlled
    stress_goal : (3, 3)  target macroscopic stress; only entries where
                  ``control == 1`` are used
    toler_lin   : relative CG residual tolerance
    maxiter     : maximum CG iterations — must be static for JIT

    Returns
    -------
    eps        : (3, 3, Nv)   updated local strain
    sigma      : (3, 3, Nv)   updated local stress  σ = C : ε
    delta      : (3, 3, Nv)   strain correction (fluctuation + macroscopic part)
    eps_bar_out: (3, 3)       macroscopic strain, with stress-controlled
                 entries filled in by the solve
    iter_count : int array    CG iterations performed
    converged  : bool array   True if residual tolerance met
    """
    Nv = prod(n_i)
    pairs = _active_pairs(control)
    control_arr = jnp.asarray(control, dtype=eps_bar.dtype)

    iq = 1j * _nyquist_safe_xi(xi_flat, n_i)  # (3, Nv)

    # ── FFT helpers ───────────────────────────────────────────────────────────
    def fft_(x):
        s = x.shape
        return jnp.fft.fftn(x.reshape(s[:-1] + n_i), axes=(-3, -2, -1)).reshape(s)

    def ifft_(x):
        s = x.shape
        return jnp.fft.ifftn(x.reshape(s[:-1] + n_i), axes=(-3, -2, -1)).real.reshape(s)

    # ── pack/unpack the macroscopic-strain correction (stress-controlled only) ─
    def sv2sm(sv):
        sm = jnp.zeros((3, 3), dtype=sv.dtype)
        for k, (i, j) in enumerate(pairs):
            sm = sm.at[i, j].set(sv[k])
            sm = sm.at[j, i].set(sv[k])
        return sm

    def sm2sv(sm):
        if not pairs:
            return jnp.zeros((0,), dtype=sm.dtype)
        return jnp.stack([sm[i, j] for i, j in pairs])

    def unpack(x_flat):
        du = x_flat[: 3 * Nv].reshape(3, Nv)
        sv = x_flat[3 * Nv:]
        return du, sv

    def pack(du, sv):
        return jnp.concatenate([du.reshape(-1), sv])

    # ── strain from a displacement field, with macroscopic part embedded in
    #    the zero-frequency (DC) mode:  mean(eps_pert) = deps_bar_free ────────
    def strain_from_u(du, deps_bar_free):
        du_hat   = fft_(du)
        grad_hat = jnp.einsum("im,jm->ijm", du_hat, iq)
        eps_hat  = 0.5 * (grad_hat + jnp.transpose(grad_hat, (1, 0, 2)))
        eps_hat  = eps_hat.at[:, :, 0].set(Nv * deps_bar_free.astype(eps_hat.dtype))
        return ifft_(eps_hat)

    # ── Linear operator  A(du, dsv) = (div residual, mean-stress residual) ───
    def A_op(x_flat):
        du, sv        = unpack(x_flat)
        eps_trial     = strain_from_u(du, sv2sm(sv))
        sigma_trial   = jnp.einsum("ijklm,klm->ijm", C_field, eps_trial)
        sigma_hat     = fft_(sigma_trial)
        div_flat      = ifft_(jnp.einsum("ijm,jm->im", sigma_hat, iq))
        extra_out     = sm2sv(jnp.real(sigma_hat[:, :, 0]))
        return pack(div_flat, extra_out)

    # ── RHS from the prescribed (strain-controlled) baseline strain ─────────
    eps0      = jnp.ones((3, 3, Nv)) * (eps_bar * (1.0 - control_arr))[:, :, None]
    sigma0    = jnp.einsum("ijklm,klm->ijm", C_field, eps0)
    sigma0_hat = fft_(sigma0)
    bb_div    = -ifft_(jnp.einsum("ijm,jm->im", sigma0_hat, iq))
    bb_extra  = Nv * sm2sv(stress_goal) - sm2sv(jnp.real(sigma0_hat[:, :, 0]))
    bb        = pack(bb_div, bb_extra)

    # ── CG solve ──────────────────────────────────────────────────────────────
    x0 = jnp.zeros_like(bb)
    x_flat, iter_count, converged = cg_count(A_op, bb, x0, toler_lin, maxiter)

    du_sol, sv_sol = unpack(x_flat)
    deps_bar_free  = sv2sm(sv_sol)

    delta = strain_from_u(du_sol, deps_bar_free)
    eps   = eps0 + delta
    sigma = jnp.einsum("ijklm,klm->ijm", C_field, eps)
    eps_bar_out = eps_bar * (1.0 - control_arr) + deps_bar_free

    return eps, sigma, delta, eps_bar_out, iter_count, converged
