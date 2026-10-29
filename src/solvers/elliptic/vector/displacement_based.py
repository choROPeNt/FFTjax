"""
Displacement-based Newton-CG elastic solve with mixed strain/stress BC.

Solves for the periodic displacement fluctuation with the true tangent
``C_field`` (no reference medium). The macroscopic strain of stress-controlled
directions is solved jointly, carried in the zero-frequency mode of the strain.
"""

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)

from typing import Tuple
from functools import partial
from math import prod

import jax
import jax.numpy as jnp
from jax import jit, lax
from jax.sharding import PartitionSpec as P

from operators.fft_distributed import (
    AXIS, choose_device_count, pfft3d_flat, pifft3d_flat, shard_x_slabs, x_slab_mesh, x_slab_spec,
)
from operators.green import nyquist_safe_xi
from solvers.elliptic.vector.base import ElasticitySolver
from solvers.elliptic.vector.mixed_bc import (
    _ZERO_CONTROL, _active_pairs, sv2sm as _sv2sm, sm2sv as _sm2sv,
)
from solvers.krylov.cg import cg_solve, cg_solve_pmap
from solvers.solution import ElasticitySolution


def solve_displacement_based(
    n:           Tuple,
    C_field:     jnp.ndarray,
    xi_flat:     jnp.ndarray,
    eps_bar:     jnp.ndarray,
    control:     Tuple[Tuple[int, ...], ...],
    stress_goal: jnp.ndarray,
    toler_lin:   float = 1e-4,
    maxiter:     int = 1000,
    C0:          jnp.ndarray | None = None,
    n_devices: int | None = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Displacement-based solve; shards the whole CG across devices when more than one is available.

    Parameters
    ----------
    n           : grid shape (static)
    C_field     : (3, 3, 3, 3, Nv) tangent stiffness
    xi_flat     : (3, Nv) frequencies (operators.green.build_freq_grid)
    eps_bar     : (3, 3) macroscopic strain; stress-controlled entries ignored
    control     : (3, 3) static 0/1 mask, 1 = stress-controlled
    stress_goal : (3, 3) target stress, stress-controlled entries only
    toler_lin   : CG relative tolerance
    maxiter     : CG iteration cap (static)
    C0          : (3, 3, 3, 3) reference stiffness for the preconditioner
                  (xi.C0.xi)^-1; default: voxel mean of C_field
    n_devices   : device cap for domain decomposition (None = all)

    Returns
    -------
    eps, sigma, delta : (3, 3, Nv) strain, stress, strain correction
    eps_bar_out       : (3, 3) macroscopic strain, stress-controlled entries solved
    converged         : bool
    """
    resolved_n_devices = choose_device_count(n, n_devices)
    if resolved_n_devices > 1:
        return _solve_displacement_based_sharded(
            n, shard_x_slabs(C_field, resolved_n_devices), shard_x_slabs(xi_flat, resolved_n_devices),
            eps_bar, control, stress_goal, toler_lin, maxiter, C0, resolved_n_devices,
        )
    return _solve_displacement_based_local(
        n, C_field, xi_flat, eps_bar, control, stress_goal, toler_lin, maxiter, C0,
    )


@partial(jit, static_argnames=("n", "control", "maxiter"))
def _solve_displacement_based_local(n, C_field, xi_flat, eps_bar, control, stress_goal,
                                    toler_lin, maxiter, C0):
    """Single-device body of ``solve_displacement_based``."""
    Nv = prod(n)
    pairs = _active_pairs(control)
    control_arr = jnp.asarray(control, dtype=eps_bar.dtype)
    sv2sm = partial(_sv2sm, pairs=pairs)
    sm2sv = partial(_sm2sv, pairs=pairs)

    iq = 1j * nyquist_safe_xi(xi_flat, n)

    def fft_(x):
        s = x.shape
        return jnp.fft.fftn(x.reshape(s[:-1] + n), axes=(-3, -2, -1)).reshape(s)

    def ifft_(x):
        s = x.shape
        return jnp.fft.ifftn(x.reshape(s[:-1] + n), axes=(-3, -2, -1)).real.reshape(s)

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

    def A_op(x_flat):
        du, sv        = unpack(x_flat)
        eps_trial     = strain_from_u(du, sv2sm(sv))
        sigma_trial   = jnp.einsum("ijklm,klm->ijm", C_field, eps_trial)
        sigma_hat     = fft_(sigma_trial)
        div_flat      = ifft_(jnp.einsum("ijm,jm->im", sigma_hat, iq))
        extra_out     = -sm2sv(jnp.real(sigma_hat[:, :, 0]))
        return pack(div_flat, extra_out)

    eps0      = jnp.ones((3, 3, Nv)) * (eps_bar * (1.0 - control_arr))[:, :, None]
    sigma0    = jnp.einsum("ijklm,klm->ijm", C_field, eps0)
    sigma0_hat = fft_(sigma0)
    bb_div    = -ifft_(jnp.einsum("ijm,jm->im", sigma0_hat, iq))
    bb_extra  = sm2sv(jnp.real(sigma0_hat[:, :, 0])) - Nv * sm2sv(stress_goal)
    bb        = pack(bb_div, bb_extra)

    # K0 = xi.C0.xi is singular where all (Nyquist-zeroed) xi vanish; the
    # operator is blind there too (a translation gauge), so any regular value works.
    C0_ref  = jnp.mean(C_field, axis=-1) if C0 is None else C0
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
    x_flat, converged = cg_solve(A_op, bb, x0, toler_lin, maxiter, M=M)

    du_sol, sv_sol = unpack(x_flat)
    deps_bar_free  = sv2sm(sv_sol)

    delta = strain_from_u(du_sol, deps_bar_free)
    eps   = eps0 + delta
    sigma = jnp.einsum("ijklm,klm->ijm", C_field, eps)
    eps_bar_out = eps_bar * (1.0 - control_arr) + deps_bar_free

    return eps, sigma, delta, eps_bar_out, converged


@partial(jit, static_argnames=("n", "control", "maxiter", "n_devices"))
def _solve_displacement_based_sharded(n, C_field, xi_flat, eps_bar, control, stress_goal,
                                      toler_lin, maxiter, C0, n_devices):
    """
    ``n_devices > 1`` body of ``solve_displacement_based``, all under one ``shard_map``.

    The DC bin is local index 0 on device 0, so the macroscopic-strain
    injection and readout are masked to that device. The mean-strain unknowns
    also live only on device 0 (zeros elsewhere, kept by A, M and the RHS), so
    the psum'd CG dot products count them once.
    """
    Nv = prod(n)
    n_local = (n[0] // n_devices, n[1], n[2])
    pairs = _active_pairs(control)
    control_arr = jnp.asarray(control, dtype=eps_bar.dtype)
    sv2sm = partial(_sv2sm, pairs=pairs)
    sm2sv = partial(_sm2sv, pairs=pairs)

    xi_safe = nyquist_safe_xi(xi_flat, n)
    C0_ref = jnp.mean(C_field, axis=-1) if C0 is None else C0

    def solve_local(C_local, xi_local, C0_ref):
        Nv_local = C_local.shape[-1]
        on_dc = lax.axis_index(AXIS) == 0
        iq = 1j * xi_local

        def fft_(x):
            return pfft3d_flat(x, n_local, AXIS)

        def ifft_(x):
            return pifft3d_flat(x, n_local, AXIS)

        def unpack(x_flat):
            return x_flat[: 3 * Nv_local].reshape(3, Nv_local), x_flat[3 * Nv_local:]

        def pack(du, sv):
            return jnp.concatenate([du.reshape(-1), sv])

        def strain_from_u(du, deps_bar_free):
            du_hat   = fft_(du)
            grad_hat = jnp.einsum("im,jm->ijm", du_hat, iq)
            eps_hat  = 0.5 * (grad_hat + jnp.transpose(grad_hat, (1, 0, 2)))
            dc = jnp.where(on_dc, Nv * deps_bar_free.astype(eps_hat.dtype), eps_hat[:, :, 0])
            return ifft_(eps_hat.at[:, :, 0].set(dc))

        def A_op(x_flat):
            du, sv      = unpack(x_flat)
            eps_trial   = strain_from_u(du, sv2sm(sv))
            sigma_trial = jnp.einsum("ijklm,klm->ijm", C_local, eps_trial)
            sigma_hat   = fft_(sigma_trial)
            div_flat    = ifft_(jnp.einsum("ijm,jm->im", sigma_hat, iq))
            extra_out   = jnp.where(on_dc, -sm2sv(jnp.real(sigma_hat[:, :, 0])), 0.0)
            return pack(div_flat, extra_out)

        eps0       = jnp.ones((3, 3, Nv_local)) * (eps_bar * (1.0 - control_arr))[:, :, None]
        sigma0_hat = fft_(jnp.einsum("ijklm,klm->ijm", C_local, eps0))
        bb_div     = -ifft_(jnp.einsum("ijm,jm->im", sigma0_hat, iq))
        bb_extra   = jnp.where(
            on_dc, sm2sv(jnp.real(sigma0_hat[:, :, 0])) - Nv * sm2sv(stress_goal), 0.0)
        bb         = pack(bb_div, bb_extra)

        K0_hat  = jnp.einsum("jm,ijkl,lm->ikm", xi_local, C0_ref, xi_local)
        null_pt = jnp.all(xi_local == 0.0, axis=0)
        K0_hat  = jnp.where(null_pt[None, None, :], jnp.eye(3)[:, :, None], K0_hat)
        K0_inv  = jnp.moveaxis(jnp.linalg.inv(jnp.moveaxis(K0_hat, -1, 0)), 0, -1)

        if pairs:
            basis   = jnp.eye(len(pairs), dtype=eps_bar.dtype)
            P_extra = jnp.stack([
                -Nv * sm2sv(jnp.einsum("ijkl,kl->ij", C0_ref, sv2sm(basis[k])))
                for k in range(len(pairs))
            ], axis=1)

        def M(x_flat):
            du, sv = unpack(x_flat)
            z_sv = jnp.linalg.solve(P_extra, sv) if pairs else sv
            z_du = ifft_(jnp.einsum("ikm,km->im", K0_inv, fft_(du)))
            return pack(z_du, z_sv)

        x0 = jnp.zeros_like(bb)
        x_flat, converged = cg_solve_pmap(A_op, bb, x0, toler_lin, maxiter, axis_name=AXIS, M=M)

        du_sol, sv_sol = unpack(x_flat)
        deps_bar_free = sv2sm(lax.psum(sv_sol, AXIS))
        delta = strain_from_u(du_sol, deps_bar_free)
        eps   = eps0 + delta
        sigma = jnp.einsum("ijklm,klm->ijm", C_local, eps)
        eps_bar_out = eps_bar * (1.0 - control_arr) + deps_bar_free
        return eps, sigma, delta, eps_bar_out, converged

    return jax.shard_map(
        solve_local, mesh=x_slab_mesh(n_devices),
        in_specs=(x_slab_spec(5), x_slab_spec(2), P()),
        out_specs=(x_slab_spec(3), x_slab_spec(3), x_slab_spec(3), P(), P()),
    )(C_field, xi_safe, C0_ref)


class DisplacementBasedSolver(ElasticitySolver):
    """ElasticitySolver for ``solve_displacement_based``; mixed BC via ``control``."""

    def __init__(
        self,
        n:           Tuple[int, ...],
        xi_flat:     jnp.ndarray,
        control:     Tuple[Tuple[int, ...], ...] | None = None,
        toler_lin:   float = 1e-4,
        maxiter:     int = 1000,
        C0:          jnp.ndarray | None = None,
        n_devices: int | None = None,
    ):
        self.n = n
        self.xi_flat = xi_flat
        self.control = control if control is not None else _ZERO_CONTROL
        self.toler_lin = toler_lin
        self.maxiter = maxiter
        self.C0 = C0
        self.n_devices = n_devices

    def solve(
        self,
        C_field:     jnp.ndarray,
        eps_bar:     jnp.ndarray,
        stress_goal: jnp.ndarray | None = None,
    ) -> ElasticitySolution:
        sg = jnp.zeros((3, 3)) if stress_goal is None else stress_goal
        eps, sigma, delta, eps_bar_out, converged = solve_displacement_based(
            self.n, C_field, self.xi_flat, eps_bar, self.control, sg,
            toler_lin=self.toler_lin, maxiter=self.maxiter, C0=self.C0,
            n_devices=self.n_devices,
        )
        return ElasticitySolution(eps, sigma, delta, converged, eps_bar=eps_bar_out)
