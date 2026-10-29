"""
Scalar elliptic FFT-CG solves: the AT2 phase-field damage equation
(``solve_damage_helmholtz_cg``, scalar or per-voxel Gc) and steady-state
thermal conduction homogenization (``solve_thermal_conduction``).
"""

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)

from functools import partial
from math import prod

import jax
import jax.numpy as jnp

from operators.fft_distributed import (
    choose_device_count, distributed_fft_flat, distributed_ifft_flat,
    gather_x_slabs, pfft3d_flat, pifft3d_flat, split_x_slabs,
)
from operators.green import nyquist_safe_xi
from solvers.krylov.cg import cg_solve, cg_solve_pmap


@partial(jax.jit, static_argnames=("n", "maxiter", "n_devices"))
def solve_damage_helmholtz_cg(
    H_field: jnp.ndarray,
    xi_flat: jnp.ndarray,
    n: tuple[int, ...],
    l0: float,
    Gc: float | jnp.ndarray,
    d_prev: jnp.ndarray,
    toler_cg: float = 1e-4,
    maxiter: int = 300,
    eta: float = 0.0,
    dt: float = 1.0,
    k: float | jnp.ndarray = 0.0,
    n_devices: int | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Preconditioned CG solve of the AT2 damage equation (Schneider & Kaestner 2025, Eq. 33):

        (Gc/l0 + eta/dt + 2(1-k)H) d - Gc l0 lap(d) = 2(1-k)H + (eta/dt) d_n

    Linear because -g'(d) = 2(1-k)(1-d) is affine in d. eta > 0 slows the
    damage rate so a crack grows gradually instead of snapping through. At
    k = 1 a voxel is immune to its own driving force. Irreversibility:
    d = max(d_prev, d_cg).

    Parameters
    ----------
    H_field  : (Nv,) driving force / history variable psi+
    xi_flat  : (ndim, Nv) frequencies from build_freq_grid
    n        : grid shape (static)
    l0       : phase-field length scale
    Gc       : critical energy release rate. Scalar: exact Laplacian, 2 FFTs
               per CG step. Per-voxel (Nv,): needs div(Gc grad d), not
               Gc*lap(d), 8 FFTs per step; pass a scalar when Gc is uniform
    d_prev   : (Nv,) previous damage: initial guess and irreversibility floor
    toler_cg : CG relative tolerance
    maxiter  : CG iteration cap (static)
    eta, dt  : viscosity and step size (eta = 0: no regularization)
    k        : residual stiffness k_res, scalar or (Nv,); must match the k
               that degraded the stiffness H_field came from
    n_devices: device cap for domain decomposition (None = all)

    Returns
    -------
    d         : (Nv,) damage in [0, 1]
    converged : bool
    """
    if jnp.ndim(Gc) > 0:
        return _solve_damage_helmholtz_cg_het(
            H_field, xi_flat, n, l0, Gc, d_prev, toler_cg, maxiter, eta, dt, k, n_devices,
        )

    resolved_n_devices = choose_device_count(n, n_devices)
    if resolved_n_devices > 1:
        return _solve_damage_helmholtz_cg_sharded(
            H_field, xi_flat, n, l0, Gc, d_prev, toler_cg, maxiter, eta, dt, k, resolved_n_devices,
        )

    xi_sq = jnp.sum(xi_flat ** 2, axis=0)
    eta_dt = eta / dt
    driving = 2.0 * (1.0 - k) * H_field

    def fft_(v):
        return distributed_fft_flat(v, n, n_devices=1)

    def ifft_(v_hat):
        return distributed_ifft_flat(v_hat, n, n_devices=1)

    def A_op(v_flat):
        lap_v = ifft_(-xi_sq * fft_(v_flat))
        mass_v = (Gc / l0 + eta_dt + driving) * v_flat
        return mass_v - Gc * l0 * lap_v

    driving_avg = jnp.mean(driving)
    P_denom = Gc / l0 + eta_dt + driving_avg + Gc * l0 * xi_sq

    def P_op(v_flat):
        return ifft_(fft_(v_flat) / P_denom)

    bb = driving + eta_dt * d_prev

    d_cg, converged = cg_solve(A_op, bb, d_prev, toler_cg, maxiter, M=P_op)

    d = jnp.maximum(d_prev, d_cg)
    return jnp.clip(d, 0.0, 1.0), converged


def _solve_damage_helmholtz_cg_sharded(H_field, xi_flat, n, l0, Gc, d_prev, toler_cg, maxiter, eta, dt, k, n_devices):
    """``n_devices > 1`` body of the scalar-Gc solve: split once, CG under one ``pmap``."""
    n_local = (n[0] // n_devices, n[1], n[2])
    eta_dt = eta / dt
    xi_sq = jnp.sum(xi_flat ** 2, axis=0)
    driving = 2.0 * (1.0 - k) * H_field
    driving_avg = jnp.mean(driving)
    P_denom = Gc / l0 + eta_dt + driving_avg + Gc * l0 * xi_sq

    driving_sharded = split_x_slabs(driving, n_devices)
    xi_sq_sharded = split_x_slabs(xi_sq, n_devices)
    d_prev_sharded = split_x_slabs(d_prev, n_devices)
    P_denom_sharded = split_x_slabs(P_denom, n_devices)

    def solve_local(driving_local, xi_sq_local, d_prev_local, P_denom_local):
        def fft_(v):
            return pfft3d_flat(v, n_local, "i")

        def ifft_(v_hat):
            return pifft3d_flat(v_hat, n_local, "i")

        def A_op(v_flat):
            lap_v = ifft_(-xi_sq_local * fft_(v_flat))
            mass_v = (Gc / l0 + eta_dt + driving_local) * v_flat
            return mass_v - Gc * l0 * lap_v

        def P_op(v_flat):
            return ifft_(fft_(v_flat) / P_denom_local)

        bb = driving_local + eta_dt * d_prev_local
        d_cg, converged = cg_solve_pmap(A_op, bb, d_prev_local, toler_cg, maxiter, axis_name="i", M=P_op)
        d_local = jnp.maximum(d_prev_local, d_cg)
        return jnp.clip(d_local, 0.0, 1.0), converged

    d_sharded, converged_sharded = jax.pmap(solve_local, axis_name="i")(
        driving_sharded, xi_sq_sharded, d_prev_sharded, P_denom_sharded,
    )
    return gather_x_slabs(d_sharded), converged_sharded[0]


@partial(jax.jit, static_argnames=("n", "maxiter", "n_devices"))
def _solve_damage_helmholtz_cg_het(
    H_field: jnp.ndarray,
    xi_flat: jnp.ndarray,
    n: tuple[int, ...],
    l0: float,
    Gc: jnp.ndarray,
    d_prev: jnp.ndarray,
    toler_cg: float = 1e-4,
    maxiter: int = 300,
    eta: float = 0.0,
    dt: float = 1.0,
    k: float | jnp.ndarray = 0.0,
    n_devices: int | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Per-voxel-Gc body of ``solve_damage_helmholtz_cg``:

        (Gc/l0 + eta/dt + 2(1-k)H) d - l0 div(Gc grad d) = 2(1-k)H + (eta/dt) d_n

    div(Gc grad d) = Gc lap(d) + grad(Gc).grad(d): putting a per-voxel Gc into
    Gc*lap(d) silently solves the wrong equation, interfaces included.
    Gradient and divergence are odd powers of xi, so they use nyquist_safe_xi.
    Preconditioned with the scalar operator at Gc0 = mean(Gc).
    """
    resolved_n_devices = choose_device_count(n, n_devices)
    if resolved_n_devices > 1:
        return _solve_damage_helmholtz_cg_het_sharded(
            H_field, xi_flat, n, l0, Gc, d_prev, toler_cg, maxiter, eta, dt, k, resolved_n_devices,
        )

    iq = 1j * nyquist_safe_xi(xi_flat, n)
    eta_dt = eta / dt
    driving = 2.0 * (1.0 - k) * H_field

    def fft_(v):
        return distributed_fft_flat(v, n, n_devices=1)

    def ifft_(v_hat):
        return distributed_ifft_flat(v_hat, n, n_devices=1)

    def fft_vec(v):
        return distributed_fft_flat(v, n, n_devices=1)

    def ifft_vec(v_hat):
        return distributed_ifft_flat(v_hat, n, n_devices=1)

    def div_Gc_grad(v_flat):
        """div(Gc grad v): gradient, real-space Gc multiply, divergence."""
        grad = ifft_vec(iq * fft_(v_flat)[None, :])
        q_hat = fft_vec(Gc[None, :] * grad)
        return ifft_(jnp.sum(iq * q_hat, axis=0))

    def A_op(v_flat):
        mass_v = (Gc / l0 + eta_dt + driving) * v_flat
        return mass_v - l0 * div_Gc_grad(v_flat)

    Gc0 = jnp.mean(Gc)
    xi_sq = jnp.sum(xi_flat ** 2, axis=0)
    driving_avg = jnp.mean(driving)
    P_denom = Gc0 / l0 + eta_dt + driving_avg + Gc0 * l0 * xi_sq

    def P_op(v_flat):
        return ifft_(fft_(v_flat) / P_denom)

    bb = driving + eta_dt * d_prev

    d_cg, converged = cg_solve(A_op, bb, d_prev, toler_cg, maxiter, M=P_op)

    d = jnp.maximum(d_prev, d_cg)
    return jnp.clip(d, 0.0, 1.0), converged


def _solve_damage_helmholtz_cg_het_sharded(H_field, xi_flat, n, l0, Gc, d_prev, toler_cg, maxiter, eta, dt, k, n_devices):
    """``n_devices > 1`` body of the per-voxel-Gc solve: split once, CG under one ``pmap``."""
    n_local = (n[0] // n_devices, n[1], n[2])
    iq = 1j * nyquist_safe_xi(xi_flat, n)
    eta_dt = eta / dt
    driving = 2.0 * (1.0 - k) * H_field
    Gc0 = jnp.mean(Gc)
    xi_sq = jnp.sum(xi_flat ** 2, axis=0)
    driving_avg = jnp.mean(driving)
    P_denom = Gc0 / l0 + eta_dt + driving_avg + Gc0 * l0 * xi_sq

    iq_sharded = split_x_slabs(iq, n_devices)
    Gc_sharded = split_x_slabs(Gc, n_devices)
    driving_sharded = split_x_slabs(driving, n_devices)
    d_prev_sharded = split_x_slabs(d_prev, n_devices)
    P_denom_sharded = split_x_slabs(P_denom, n_devices)

    def solve_local(iq_local, Gc_local, driving_local, d_prev_local, P_denom_local):
        def fft_(v):
            return pfft3d_flat(v, n_local, "i")

        def ifft_(v_hat):
            return pifft3d_flat(v_hat, n_local, "i")

        def div_Gc_grad(v_flat):
            grad = ifft_(iq_local * fft_(v_flat)[None, :])
            q_hat = fft_(Gc_local[None, :] * grad)
            return ifft_(jnp.sum(iq_local * q_hat, axis=0))

        def A_op(v_flat):
            mass_v = (Gc_local / l0 + eta_dt + driving_local) * v_flat
            return mass_v - l0 * div_Gc_grad(v_flat)

        def P_op(v_flat):
            return ifft_(fft_(v_flat) / P_denom_local)

        bb = driving_local + eta_dt * d_prev_local
        d_cg, converged = cg_solve_pmap(A_op, bb, d_prev_local, toler_cg, maxiter, axis_name="i", M=P_op)
        d_local = jnp.maximum(d_prev_local, d_cg)
        return jnp.clip(d_local, 0.0, 1.0), converged

    d_sharded, converged_sharded = jax.pmap(solve_local, axis_name="i")(
        iq_sharded, Gc_sharded, driving_sharded, d_prev_sharded, P_denom_sharded,
    )
    return gather_x_slabs(d_sharded), converged_sharded[0]


_ZERO_CONTROL_1D = (0, 0, 0)


@partial(jax.jit, static_argnames=("n", "control", "maxiter", "n_devices"))
def solve_thermal_conduction(
    n:          tuple[int, ...],
    K_field:    jnp.ndarray,
    xi_flat:    jnp.ndarray,
    grad_T_bar: jnp.ndarray,
    control:    tuple[int, int, int] = _ZERO_CONTROL_1D,
    flux_goal:  jnp.ndarray | None = None,
    toler_lin:  float = 1e-6,
    maxiter:    int = 1000,
    K0:         jnp.ndarray | None = None,
    n_devices: int | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Steady-state heat conduction with mixed gradient/flux BC; the scalar
    analogue of ``solve_displacement_based``.

    With T = grad_T_bar . x + T' (T' periodic) and q = -K grad T, solves

        div(K grad T') = -div(K grad_T_bar)

    by CG for ``pack(T', sv)``; the gradient correction of flux-controlled
    directions sits in the DC bin of the gradient.

    Parameters
    ----------
    n          : grid shape (static)
    K_field    : (3, 3, Nv) conductivity (materialmodels.assembly.assemble_K_field)
    xi_flat    : (3, Nv) frequencies
    grad_T_bar : (3,) macroscopic gradient; flux-controlled entries ignored
    control    : (3,) static 0/1 mask, 1 = flux-controlled
    flux_goal  : (3,) target flux, flux-controlled entries only (None = 0)
    toler_lin  : CG relative tolerance
    maxiter    : CG iteration cap (static)
    K0         : (3, 3) reference conductivity for the preconditioner
                 1/(xi.K0.xi); default: voxel mean of K_field
    n_devices  : device cap for the FFTs (None = all)

    Returns
    -------
    grad_T, flux   : (3, Nv) temperature gradient and heat flux
    T_prime        : (Nv,) periodic fluctuation, not the absolute temperature
    delta          : (3, Nv) gradient correction
    grad_T_bar_out : (3,) macroscopic gradient, flux-controlled entries solved
    converged      : bool
    """
    Nv = prod(n)
    active = tuple(i for i in range(3) if control[i])
    control_arr = jnp.asarray(control, dtype=grad_T_bar.dtype)

    iq = 1j * nyquist_safe_xi(xi_flat, n)

    def fft_(v):
        return distributed_fft_flat(v, n, n_devices=n_devices)

    def ifft_(v_hat):
        return distributed_ifft_flat(v_hat, n, n_devices=n_devices)

    def fft_vec(v):
        return distributed_fft_flat(v, n, n_devices=n_devices)

    def ifft_vec(v_hat):
        return distributed_ifft_flat(v_hat, n, n_devices=n_devices)

    def sv2full(sv):
        full = jnp.zeros((3,), dtype=sv.dtype)
        for idx, i in enumerate(active):
            full = full.at[i].set(sv[idx])
        return full

    def full2sv(full):
        if not active:
            return jnp.zeros((0,), dtype=full.dtype)
        return jnp.stack([full[i] for i in active])

    def unpack(x_flat):
        Tp = x_flat[:Nv]
        sv = x_flat[Nv:]
        return Tp, sv

    def pack(Tp, sv):
        return jnp.concatenate([Tp, sv])

    def grad_from_T(Tp, dgrad_bar_free):
        Tp_hat   = fft_(Tp)
        grad_hat = iq * Tp_hat[None, :]
        grad_hat = grad_hat.at[:, 0].set(Nv * dgrad_bar_free.astype(grad_hat.dtype))
        return ifft_vec(grad_hat)

    def A_op(x_flat):
        Tp, sv     = unpack(x_flat)
        grad_trial = grad_from_T(Tp, sv2full(sv))
        q_trial    = -jnp.einsum('ijm,jm->im', K_field, grad_trial)
        q_hat      = fft_vec(q_trial)
        div_flat   = ifft_(jnp.sum(iq * q_hat, axis=0))
        extra_out  = -full2sv(jnp.real(q_hat[:, 0]))
        return pack(div_flat, extra_out)

    grad0    = jnp.ones((3, Nv)) * (grad_T_bar * (1.0 - control_arr))[:, None]
    q0       = -jnp.einsum('ijm,jm->im', K_field, grad0)
    q0_hat   = fft_vec(q0)
    bb_div   = -ifft_(jnp.sum(iq * q0_hat, axis=0))
    fg       = jnp.zeros((3,), dtype=grad_T_bar.dtype) if flux_goal is None else flux_goal
    bb_extra = full2sv(jnp.real(q0_hat[:, 0])) - Nv * full2sv(fg)
    bb       = pack(bb_div, bb_extra)

    # k0 = xi.K0.xi vanishes at DC and Nyquist-zeroed points; the operator is
    # blind there too, so any regular value works.
    K0_ref  = jnp.mean(K_field, axis=-1) if K0 is None else K0
    k0_hat  = jnp.einsum('jm,jk,km->m', iq.imag, K0_ref, iq.imag)
    null_pt = jnp.all(iq.imag == 0.0, axis=0)
    k0_hat  = jnp.where(null_pt, 1.0, k0_hat)
    k0_inv  = 1.0 / k0_hat

    def M_T(r):
        return ifft_(k0_inv * fft_(r))

    if active:
        basis = jnp.eye(len(active), dtype=grad_T_bar.dtype)
        P_extra = jnp.stack([
            Nv * full2sv(jnp.einsum('ij,j->i', K0_ref, sv2full(basis[idx])))
            for idx in range(len(active))
        ], axis=1)

    def M(x_flat):
        Tp, sv = unpack(x_flat)
        z_sv = jnp.linalg.solve(P_extra, sv) if active else sv
        return pack(M_T(Tp), z_sv)

    x0 = jnp.zeros_like(bb)
    x_flat, converged = cg_solve(A_op, bb, x0, toler_lin, maxiter, M=M)

    T_prime, sv_sol = unpack(x_flat)
    dgrad_bar_free  = sv2full(sv_sol)

    delta  = grad_from_T(T_prime, dgrad_bar_free)
    grad_T = grad0 + delta
    flux   = -jnp.einsum('ijm,jm->im', K_field, grad_T)
    grad_T_bar_out = grad_T_bar * (1.0 - control_arr) + dgrad_bar_free

    return grad_T, flux, T_prime, delta, grad_T_bar_out, converged
