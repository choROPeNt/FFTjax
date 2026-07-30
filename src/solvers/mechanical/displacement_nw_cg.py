import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)

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
    C0:         jnp.ndarray | None = None,
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
    C0          : (3, 3, 3, 3) or None  reference stiffness used to build a
                  frequency-domain preconditioner ``(ξ·C0·ξ)⁻¹``. Without a
                  preconditioner CG needs O(10²-10³) iterations for realistic
                  (high-contrast) composites; default is the voxel-average of
                  ``C_field``.

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
        extra_out     = -sm2sv(jnp.real(sigma_hat[:, :, 0]))
        return pack(div_flat, extra_out)

    # ── RHS from the prescribed (strain-controlled) baseline strain ─────────
    eps0      = jnp.ones((3, 3, Nv)) * (eps_bar * (1.0 - control_arr))[:, :, None]
    sigma0    = jnp.einsum("ijklm,klm->ijm", C_field, eps0)
    sigma0_hat = fft_(sigma0)
    bb_div    = -ifft_(jnp.einsum("ijm,jm->im", sigma0_hat, iq))
    bb_extra  = sm2sv(jnp.real(sigma0_hat[:, :, 0])) - Nv * sm2sv(stress_goal)
    bb        = pack(bb_div, bb_extra)

    # ── Preconditioner  M ≈ A⁻¹, built from a reference (homogeneous) medium ──
    # "du" block: per-frequency acoustic tensor  K0(ξ) = ξ·C0·ξ  (3,3,Nv),
    # inverted pointwise. ξ=0 at the true DC bin *and* at every point where a
    # Nyquist-zeroed dimension (see _nyquist_safe_xi) makes all components
    # vanish simultaneously — K0 is singular there and must be regularized;
    # these are exactly the directions the operator itself is blind to
    # (a rigid-body-translation-like gauge freedom), so any regular value works.
    C0_ref  = jnp.mean(C_field, axis=-1) if C0 is None else C0
    K0_hat  = jnp.einsum("jm,ijkl,lm->ikm", iq.imag, C0_ref, iq.imag)
    null_pt = jnp.all(iq.imag == 0.0, axis=0)
    K0_hat  = jnp.where(null_pt[None, None, :], jnp.eye(3)[:, :, None], K0_hat)
    K0_inv  = jnp.moveaxis(jnp.linalg.inv(jnp.moveaxis(K0_hat, -1, 0)), 0, -1)

    def M_du(r_du):
        r_hat = fft_(r_du)
        z_hat = jnp.einsum("ikm,km->im", K0_inv, r_hat)
        return ifft_(z_hat)

    # "extra" block: self-coupling of the macroscopic-strain unknowns,
    # approximated with the same reference stiffness (-Nv * C0 : sv2sm(·)),
    # matching the sign of A_op's extra_out.
    if pairs:
        basis   = jnp.eye(len(pairs), dtype=eps_bar.dtype)
        P_extra = jnp.stack([
            -Nv * sm2sv(jnp.einsum("ijkl,kl->ij", C0_ref, sv2sm(basis[k])))
            for k in range(len(pairs))
        ], axis=1)

    def M(x_flat):
        du, sv = unpack(x_flat)
        z_sv = jnp.linalg.solve(P_extra, sv) if pairs else sv
        return pack(M_du(du), z_sv)

    # ── CG solve ──────────────────────────────────────────────────────────────
    x0 = jnp.zeros_like(bb)
    x_flat, iter_count, converged = cg_count(A_op, bb, x0, toler_lin, maxiter, M=M)

    du_sol, sv_sol = unpack(x_flat)
    deps_bar_free  = sv2sm(sv_sol)

    delta = strain_from_u(du_sol, deps_bar_free)
    eps   = eps0 + delta
    sigma = jnp.einsum("ijklm,klm->ijm", C_field, eps)
    eps_bar_out = eps_bar * (1.0 - control_arr) + deps_bar_free

    return eps, sigma, delta, eps_bar_out, iter_count, converged
