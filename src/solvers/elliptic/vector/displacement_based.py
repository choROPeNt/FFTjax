"""
Displacement-based Newton-CG solve for the variational FFT elastic solver,
supporting mixed strain/stress macroscopic boundary conditions.

Unlike the strain-based ``solve_lippmann_schwinger`` (fixed reference-medium
Green's operator), this solves directly for a periodic displacement
fluctuation field via the true (possibly heterogeneous) tangent stiffness
``C_field``, with no reference-medium approximation. The macroscopic-average
strain correction for stress-controlled directions is solved for jointly
with the displacement fluctuation, by embedding it in the (otherwise
physically meaningless) zero-frequency mode of the Fourier-space strain
field.

Ported fresh from this project's earlier (pre-refactor) ``ddisp_nw_cg``,
itself a port of FFTMAD's ``displacement_nw_cg_small`` / ``du_loc_nw_CG``.
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
    Parameters
    ----------
    n           : grid shape (nx, ny, nz) -- must be static for JIT
    C_field     : (3, 3, 3, 3, Nv)  per-voxel tangent stiffness
    xi_flat     : (3, Nv)           angular-frequency grid (``operators.green.build_freq_grid``)
    eps_bar     : (3, 3)  prescribed macroscopic strain; entries where
                  ``control == 1`` are ignored (solved for instead)
    control     : (3, 3) static 0/1 mask, 1 = stress-controlled, 0 = strain-controlled
    stress_goal : (3, 3)  target macroscopic stress; only entries where
                  ``control == 1`` are used
    toler_lin   : relative CG residual tolerance
    maxiter     : maximum CG iterations -- must be static for JIT
    C0          : (3, 3, 3, 3) or None  reference stiffness used to build a
                  frequency-domain preconditioner ``(ξ·C0·ξ)⁻¹``. Without a
                  preconditioner CG needs O(10²-10³) iterations for realistic
                  (high-contrast) composites; default is the voxel-average of
                  ``C_field``.
    n_devices : caps the auto-detected device count
                  (``choose_device_count``). At > 1 the whole CG runs under
                  one ``jax.shard_map`` (``_solve_displacement_based_sharded``):
                  ``C_field`` and every CG vector stay x-slab sharded, never
                  gathered; returned fields stay sharded too. None (default)
                  uses jax.local_device_count(); at 1 this is the original
                  single-device solve.

    Returns
    -------
    eps        : (3, 3, Nv)   updated local strain
    sigma      : (3, 3, Nv)   updated local stress  σ = C : ε
    delta      : (3, 3, Nv)   strain correction (fluctuation + macroscopic part)
    eps_bar_out: (3, 3)       macroscopic strain, with stress-controlled
                 entries filled in by the solve
    converged  : bool array   True if residual tolerance met
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
    # pack/unpack the macroscopic-strain correction (stress-controlled only) --
    # shared implementation, see solvers.elliptic.vector.mixed_bc.
    sv2sm = partial(_sv2sm, pairs=pairs)
    sm2sv = partial(_sm2sv, pairs=pairs)

    iq = 1j * nyquist_safe_xi(xi_flat, n)  # (3, Nv)

    # ── FFT helpers ───────────────────────────────────────────────────────────
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
    # Nyquist-zeroed dimension (see operators.green.nyquist_safe_xi) makes all components
    # vanish simultaneously -- K0 is singular there and must be regularized;
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
    ``n_devices > 1`` body of ``solve_displacement_based``: same system and
    preconditioner as ``_solve_displacement_based_local``, with ``C_field``,
    ``iq``, ``K0_inv`` and the CG vectors held as local x-slabs under one
    ``jax.shard_map`` (``pfft3d_flat`` + ``cg_solve_pmap``).

    The global DC bin is local index 0 on device 0 (``axis_index == 0``), so
    the macroscopic-strain injection/readout is masked to that device. The
    K mean-strain unknowns appended to ``x`` likewise live only on device 0
    (zeros elsewhere, which A, M and the RHS preserve exactly), so the
    psum'd CG dot products count them once; a final psum replicates them.
    """
    Nv = prod(n)
    n_local = (n[0] // n_devices, n[1], n[2])
    pairs = _active_pairs(control)
    control_arr = jnp.asarray(control, dtype=eps_bar.dtype)
    sv2sm = partial(_sv2sm, pairs=pairs)
    sm2sv = partial(_sm2sv, pairs=pairs)

    xi_safe = nyquist_safe_xi(xi_flat, n)                             # (3, Nv), sharded
    C0_ref = jnp.mean(C_field, axis=-1) if C0 is None else C0         # cross-device mean

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
        deps_bar_free = sv2sm(lax.psum(sv_sol, AXIS))  # nonzero only on device 0
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
    """
    ElasticitySolver wrapping solve_displacement_based: displacement-based,
    periodic, supports mixed strain/stress macroscopic BC via ``control``.
    Formulation-specific setup (grid shape, frequency grid, mixed-BC control
    mask, CG tolerance) lives here in __init__; solve() takes only what's
    common to every ElasticitySolver.
    """

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
