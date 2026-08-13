"""
Displacement-based Newton-CG solve for nonlinear (J2 plastic) materials.

Structurally the same FFT/preconditioner scaffolding as
``displacement_nw_cg.ddisp_nw_cg`` (same DC-embedding trick for
stress-controlled macroscopic directions, same reference-medium
preconditioner), generalized from a single one-shot linear CG solve into a
genuine outer Newton loop: at each Newton iteration the *true* nonlinear
stress (from ``mat_models.plastic.j2_return_map_field``) drives the
residual, while the *algorithmic consistent tangent* at the current strain
estimate is what the inner CG solve linearizes against -- not the same as
calling ``ddisp_nw_cg`` repeatedly with an updated ``C_field``, which would
silently re-derive the baseline strain from ``eps_bar`` every time instead
of carrying the accumulated strain field forward between Newton iterations.

Deliberately a separate module rather than a generalization of
``ddisp_nw_cg`` itself, matching this project's existing practice of
keeping new, less battle-tested solver variants alongside rather than
folding into an already-verified one (e.g. ``pff_damage.py`` next to the
mechanical solvers, not merged into them).
"""

from typing import Tuple
from functools import partial
from math import prod

import jax.numpy as jnp
from jax import jit

from solvers.krylov.cg import cg_solve
from mat_models.plastic import J2Plasticity, j2_return_map_field


def _active_pairs(control: Tuple[Tuple[int, ...], ...]) -> Tuple[Tuple[int, int], ...]:
    return tuple((i, j) for i in range(3) for j in range(i, 3) if control[i][j])


def _nyquist_safe_xi(xi_flat: jnp.ndarray, n_i: Tuple) -> jnp.ndarray:
    idx   = [jnp.arange(ni) for ni in n_i]
    grids = jnp.meshgrid(*idx, indexing="ij")
    masks = [
        (g == ni // 2) if ni % 2 == 0 else jnp.zeros_like(g, dtype=bool)
        for ni, g in zip(n_i, grids)
    ]
    nyquist_mask = jnp.stack([m.ravel() for m in masks])
    return jnp.where(nyquist_mask, 0.0, xi_flat)


def solve_plastic_disp_nw(
    n_i:        Tuple[int, int, int],
    material:   J2Plasticity,
    xi_flat:    jnp.ndarray,
    eps_bar:    jnp.ndarray,
    control:    Tuple[Tuple[int, ...], ...],
    stress_goal: jnp.ndarray,
    eps_prev:      jnp.ndarray,
    eps_p_prev:    jnp.ndarray,
    alpha_prev:    jnp.ndarray,
    toler_nw:   float = 1e-6,
    maxiter_nw: int   = 25,
    toler_lin:  float = 1e-6,
    maxiter_cg: int   = 2000,
    C0:         jnp.ndarray | None = None,
):
    """
    One load increment of the displacement-based Newton-CG solve for a J2
    plastic material, with mixed strain/stress macroscopic boundary
    conditions (same ``control``/``stress_goal`` convention as
    ``ddisp_nw_cg``).

    Parameters
    ----------
    n_i         : grid shape (nx, ny, nz)
    material    : J2Plasticity
    xi_flat     : (3, Nv)  angular-frequency grid
    eps_bar     : (3, 3)   prescribed macroscopic strain for THIS increment;
                  entries where ``control == 1`` are ignored (solved for)
    control     : (3, 3) static 0/1 mask, 1 = stress-controlled
    stress_goal : (3, 3)   target macroscopic stress for THIS increment
    eps_prev    : (3, 3, Nv)  converged local strain field from the END of
                  the previous increment -- the Newton warm start
    eps_p_prev  : (3, 3, Nv)  converged plastic strain field from the
                  previous increment (frozen during this increment's
                  Newton iterations, only advanced once Newton converges)
    alpha_prev  : (Nv,)       converged hardening variable, previous increment
    toler_nw    : Newton convergence tolerance on the relative strain update
    maxiter_nw  : max Newton iterations
    toler_lin, maxiter_cg, C0 : passed through to the inner CG solve each
                  Newton iteration (same meaning as in ``ddisp_nw_cg``)

    Returns
    -------
    eps         : (3, 3, Nv)   converged local strain field
    sigma       : (3, 3, Nv)   converged local stress field
    eps_p       : (3, 3, Nv)   updated plastic strain field (commit this
                  and ``alpha``/``eps`` as the next increment's ``*_prev``)
    alpha       : (Nv,)        updated hardening variable
    eps_bar_out : (3, 3)       macroscopic strain, stress-controlled entries filled in
    iter_nw     : int          Newton iterations used
    converged_nw: bool         True if the Newton loop met ``toler_nw``
    """
    Nv = prod(n_i)
    pairs = _active_pairs(control)
    control_arr = jnp.asarray(control, dtype=eps_bar.dtype)

    iq = 1j * _nyquist_safe_xi(xi_flat, n_i)

    def fft_(x):
        s = x.shape
        return jnp.fft.fftn(x.reshape(s[:-1] + n_i), axes=(-3, -2, -1)).reshape(s)

    def ifft_(x):
        s = x.shape
        return jnp.fft.ifftn(x.reshape(s[:-1] + n_i), axes=(-3, -2, -1)).real.reshape(s)

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

    def strain_from_u(du, deps_bar_free):
        du_hat   = fft_(du)
        grad_hat = jnp.einsum("im,jm->ijm", du_hat, iq)
        eps_hat  = 0.5 * (grad_hat + jnp.transpose(grad_hat, (1, 0, 2)))
        eps_hat  = eps_hat.at[:, :, 0].set(Nv * deps_bar_free.astype(eps_hat.dtype))
        return ifft_(eps_hat)

    # ── Newton state: total strain field + macroscopic strain estimate,
    #    both carried across Newton iterations (unlike ddisp_nw_cg, which
    #    re-derives eps0 from eps_bar fresh on every call). Strain-controlled
    #    directions are DATA (their mean must equal eps_bar exactly, not
    #    something Newton solves for -- d_eps from the CG correction always
    #    has exactly zero mean there, same DC-embedding trick as
    #    ddisp_nw_cg, so this only needs setting once, up front) --
    #    stress-controlled directions keep whatever mean eps_prev already
    #    had as the Newton warm start.
    eps_prev_mean = jnp.mean(eps_prev, axis=-1)                      # (3, 3)
    eps_prev_fluct = eps_prev - eps_prev_mean[:, :, None]             # zero-mean part
    eps0_mean = eps_bar * (1.0 - control_arr) + eps_prev_mean * control_arr
    eps_k = eps_prev_fluct + eps0_mean[:, :, None]

    eps_bar_free0 = eps_prev_mean * control_arr   # accumulated stress-controlled correction

    it_nw = 0
    converged_nw = jnp.array(False)
    sigma_k = None
    C0_ref_fixed = C0  # reference medium fixed across Newton iterations if given

    for it_nw in range(1, maxiter_nw + 1):
        sigma_k, _, _, C_algo_k = j2_return_map_field(material, eps_k, eps_p_prev, alpha_prev)

        # ── tangent operator at the current strain estimate ──────────────
        def A_op(x_flat, C_field=C_algo_k):
            du, sv        = unpack(x_flat)
            eps_trial     = strain_from_u(du, sv2sm(sv))
            sigma_trial   = jnp.einsum("ijklm,klm->ijm", C_field, eps_trial)
            sigma_hat     = fft_(sigma_trial)
            div_flat      = ifft_(jnp.einsum("ijm,jm->im", sigma_hat, iq))
            extra_out     = -sm2sv(jnp.real(sigma_hat[:, :, 0]))
            return pack(div_flat, extra_out)

        # ── Newton residual: TRUE nonlinear stress, not C_algo:eps_k ──────
        sigma_hat_k = fft_(sigma_k)
        res_div     = -ifft_(jnp.einsum("ijm,jm->im", sigma_hat_k, iq))
        ave_stress_k = jnp.real(sigma_hat_k[:, :, 0]) / Nv
        res_extra   = Nv * sm2sv(stress_goal - ave_stress_k)
        bb          = pack(res_div, res_extra)

        C0_ref = jnp.mean(C_algo_k, axis=-1) if C0_ref_fixed is None else C0_ref_fixed
        K0_hat  = jnp.einsum("jm,ijkl,lm->ikm", iq.imag, C0_ref, iq.imag)
        null_pt = jnp.all(iq.imag == 0.0, axis=0)
        K0_hat  = jnp.where(null_pt[None, None, :], jnp.eye(3)[:, :, None], K0_hat)
        K0_inv  = jnp.moveaxis(jnp.linalg.inv(jnp.moveaxis(K0_hat, -1, 0)), 0, -1)

        def M_du(r_du):
            r_hat = fft_(r_du)
            z_hat = jnp.einsum("ikm,km->im", K0_inv, r_hat)
            return ifft_(z_hat)

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

        x0 = jnp.zeros_like(bb)
        x_flat, conv_cg = cg_solve(A_op, bb, x0, toler_lin, maxiter_cg, M=M)

        d_du, d_sv = unpack(x_flat)
        d_eps_bar_free = sv2sm(d_sv)
        d_eps = strain_from_u(d_du, d_eps_bar_free)

        eps_k = eps_k + d_eps
        eps_bar_free0 = eps_bar_free0 + d_eps_bar_free

        rel_update = jnp.max(jnp.abs(d_eps)) / (jnp.max(jnp.abs(eps_k)) + 1e-30)
        converged_nw = rel_update < toler_nw
        if bool(converged_nw):
            break

    sigma_k, eps_p_k, alpha_k, _ = j2_return_map_field(material, eps_k, eps_p_prev, alpha_prev)
    eps_bar_out = eps_bar * (1.0 - control_arr) + eps_bar_free0

    return eps_k, sigma_k, eps_p_k, alpha_k, eps_bar_out, it_nw, converged_nw
