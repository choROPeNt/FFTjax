"""
Mechanical (elasticity) problem: build C(x), pick a formulation, solve.

``solve_mechanics`` is the entry point (single or load-stepped, always a list
of IncrementResult). ``solve_displacement_based_nonlinear`` is the
Newton-CG driver for nonlinear local laws (e.g. plasticity); it takes the
caller's ``local_update`` instead of assembling materials.
"""

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)

import time
from functools import partial
from typing import Any, Callable, Tuple, cast
from math import prod

from jax import jit
import jax.numpy as jnp
import numpy as np

from materialmodels.assembly import assemble_C_field
from operators.fft_distributed import distributed_fft_flat, distributed_ifft_flat
from operators.galerkin import build_galerkin_projector
from operators.green import build_freq_grid, build_reference_green_operator, nyquist_safe_xi
from post.fields import compute_displacement, field_to_grid, to_voigt, von_mises
from problems.incremental import IncrementResult, solve_automatic, solve_fixed
from solvers.elliptic.vector.displacement_based import DisplacementBasedSolver
from solvers.elliptic.vector.fourier_galerkin import FourierGalerkinSolver
from solvers.elliptic.vector.lippmann_schwinger import LippmannSchwingerMixedBCSolver, LippmannSchwingerSolver
from solvers.elliptic.vector.mixed_bc import (
    _ZERO_CONTROL, _active_pairs, sv2sm as _sv2sm, sm2sv as _sm2sv,
)
from solvers.krylov.cg import cg_solve
from solvers.solution import ElasticitySolution
from utils.io.xdmf_writer import IncrementalWriter


def _solve_mechanics_step(
    n:           Tuple[int, ...],
    L:           Tuple[float, ...],
    phase:       jnp.ndarray,
    materials:   list,
    eps_bar:     jnp.ndarray,
    formulation: str = "lippmann_schwinger",
    scheme:      str = "rotated",
    control:     Tuple[Tuple[int, ...], ...] = _ZERO_CONTROL,
    stress_goal: jnp.ndarray | None = None,
    toler_lin:   float = 1e-6,
    maxiter:     int = 1000,
    n_devices: int | None = None,
) -> ElasticitySolution:
    """One solve at a fixed macroscopic strain; parameters as ``solve_mechanics``."""
    control_nonzero = any(any(row) for row in control)

    C_field = assemble_C_field(materials, phase, n=n, n_devices=n_devices)

    if formulation == "lippmann_schwinger":
        green_op = build_reference_green_operator(n, L, materials, scheme=scheme, n_devices=n_devices)

        if control_nonzero:
            solver = LippmannSchwingerMixedBCSolver(
                n, green_op, control, toler_lin, maxiter, n_devices=n_devices,
            )
            return solver.solve(C_field, eps_bar, stress_goal)
        else:
            # Not forwarded: here stress_goal is the mixed-BC (3, 3) target,
            # while LippmannSchwingerSolver expects a (3, 3, Nv) field.
            solver = LippmannSchwingerSolver(n, green_op, toler_lin, maxiter, n_devices=n_devices)
            return solver.solve(C_field, eps_bar)
    elif formulation == "displacement":
        xi_flat = build_freq_grid(n, L)
        solver = DisplacementBasedSolver(n, xi_flat, control, toler_lin, maxiter, n_devices=n_devices)
        return solver.solve(C_field, eps_bar, stress_goal)
    elif formulation == "fourier_galerkin":
        galerkin_op = build_galerkin_projector(n, L, scheme=scheme, n_devices=n_devices)

        solver = FourierGalerkinSolver(n, galerkin_op, control, toler_lin, maxiter, n_devices=n_devices)
        return solver.solve(C_field, eps_bar, stress_goal)
    else:
        raise ValueError(
            f"unknown formulation {formulation!r}, expected 'lippmann_schwinger', "
            f"'displacement', or 'fourier_galerkin'"
        )


def solve_mechanics(
    n:           Tuple[int, ...],
    L:           Tuple[float, ...],
    phase:       jnp.ndarray,
    materials:   list,
    eps_bar:     jnp.ndarray,
    stepping:    str = "single",
    formulation: str = "lippmann_schwinger",
    scheme:      str = "rotated",
    control:     Tuple[Tuple[int, ...], ...] | None = None,
    stress_goal: jnp.ndarray | None = None,
    toler_lin:   float = 1e-6,
    maxiter:     int = 1000,
    n_devices: int | None = None,
    dt:          float | None = None,
    dt_init:     float = 0.1,
    dt_min:      float = 1e-4,
    dt_max:      float = 0.5,
    factor_inc:  float = 1.5,
    factor_dec:  float = 0.5,
    max_cutbacks: int = 5,
    max_steps:   int = 1000,
    writer:       IncrementalWriter | None = None,
    orientation:  jnp.ndarray | None = None,
    on_increment: Callable[[IncrementResult, float], None] | None = None,
) -> list[IncrementResult]:
    """
    Mechanical equilibrium on a phase-labelled periodic grid under a
    macroscopic strain, optionally with mixed strain/stress control and load stepping.

    Parameters
    ----------
    n, L        : grid shape and domain size
    phase       : (Nv,) int phase index
    materials   : material per phase
    eps_bar     : (3, 3) macroscopic strain at t = 1; stress-controlled entries ignored
    stepping    : "single" (one solve), "fixed" (increments of ``dt``) or
                  "automatic" (adaptive with cutbacks, see problems.incremental)
    formulation : "lippmann_schwinger" (reference medium; mixed BC by outer
                  correction), "displacement" (true tangent, bordered CG) or
                  "fourier_galerkin" (no reference medium, DC-identity mixed BC)
    scheme      : "standard" or "rotated" (lippmann_schwinger, fourier_galerkin)
    control     : (3, 3) 0/1 mask, 1 = stress-controlled (None = pure strain)
    stress_goal : (3, 3) macroscopic target with ``control``; (3, 3, Nv) field
                  for pure-strain lippmann_schwinger
    toler_lin, maxiter : CG tolerance and iteration cap
    n_devices   : device cap for domain decomposition (None = all)
    dt, dt_init, dt_min, dt_max, factor_inc, factor_dec, max_cutbacks, max_steps :
                  load-stepping controls (``dt`` required for "fixed")
    writer      : IncrementalWriter, gets strain/stress/von_mises/displacement/
                  phase (and orientation) per accepted increment
    orientation : (3, Nv) written with the fields if given
    on_increment: callback ``(result, write_time)`` per accepted increment;
                  write_time is kept separate from result.wall_time (solve only)

    Returns
    -------
    list[IncrementResult], one per increment; ``.solution`` is an ElasticitySolution
    """
    control = control if control is not None else _ZERO_CONTROL

    def solve_fn(t: float) -> ElasticitySolution:
        return _solve_mechanics_step(
            n, L, phase, materials, t * eps_bar,
            formulation=formulation, scheme=scheme, control=control,
            stress_goal=stress_goal, toler_lin=toler_lin, maxiter=maxiter,
            n_devices=n_devices,
        )

    def _on_increment(result: IncrementResult) -> None:
        write_time = 0.0
        if writer is not None:
            t0 = time.perf_counter()
            sol = cast(ElasticitySolution, result.solution)
            eps_grid   = field_to_grid(sol.eps, n)
            sigma_grid = field_to_grid(sol.sigma, n)
            sigma_vm   = von_mises(sigma_grid)
            # The solved eps_bar, not the prescribed one: that has zeros on
            # stress-controlled entries and would drop e.g. Poisson contraction.
            eps_bar_u  = sol.eps_bar if sol.eps_bar is not None else result.t * eps_bar
            u_grid     = compute_displacement(sol.eps, eps_bar_u, n, L)
            fields = {
                "phase":        np.asarray(phase).reshape(n).astype(np.float32),
                "strain":       to_voigt(eps_grid).astype(np.float64),
                "stress":       to_voigt(sigma_grid).astype(np.float64),
                "von_mises":    sigma_vm.astype(np.float64),
                "displacement": u_grid.astype(np.float64),
            }
            if orientation is not None:
                fields["orientation"] = np.asarray(orientation).T.reshape(*n, 3).astype(np.float64)
            writer.write_increment(result.step, fields, time=result.t)
            write_time = time.perf_counter() - t0
        if on_increment is not None:
            on_increment(result, write_time)

    if stepping == "single":
        t0 = time.perf_counter()
        sol = solve_fn(1.0)
        result = IncrementResult(step=1, t=1.0, dt=1.0, solution=sol,
                                  wall_time=time.perf_counter() - t0)
        _on_increment(result)
        return [result]
    elif stepping == "fixed":
        if dt is None:
            raise ValueError("stepping='fixed' requires dt")
        return solve_fixed(solve_fn, dt, on_increment=_on_increment)
    elif stepping == "automatic":
        return solve_automatic(
            solve_fn, dt_init=dt_init, dt_min=dt_min, dt_max=dt_max,
            factor_inc=factor_inc, factor_dec=factor_dec,
            max_cutbacks=max_cutbacks, max_steps=max_steps,
            on_increment=_on_increment,
        )
    else:
        raise ValueError(
            f"unknown stepping {stepping!r}, expected 'single', 'fixed', or 'automatic'"
        )


def solve_displacement_based_nonlinear(
    n:                 Tuple[int, ...],
    xi_flat:           jnp.ndarray,
    eps_bar:           jnp.ndarray,
    local_update:      Callable[[jnp.ndarray, Any], Tuple[jnp.ndarray, jnp.ndarray, Any]],
    state_init:        Any,
    toler_lin:         float = 1e-6,
    maxiter_lin:       int = 1000,
    toler_nr:          float = 1e-8,
    maxiter_nr:        int = 50,
    C0:                jnp.ndarray | None = None,
    delta_init:        jnp.ndarray | None = None,
    control:           Tuple[Tuple[int, ...], ...] | None = None,
    stress_goal:       jnp.ndarray | None = None,
    eps_bar_free_init: jnp.ndarray | None = None,
    n_devices:       int | None = None,
):
    """
    Newton-CG solve of div(sigma(eps)) = 0 for a nonlinear local law, with
    optional mixed strain/stress BC (as solve_displacement_based).

    Each Newton iteration evaluates stress and tangent at the current strain
    via ``local_update`` and solves the linearized correction by CG (true
    Newton). With mixed BC the CG unknown is ``pack(du, sv)``.

    Parameters
    ----------
    n            : grid shape
    xi_flat      : (3, Nv) frequencies
    eps_bar      : (3, 3) macroscopic strain; stress-controlled entries ignored
    local_update : (eps_field, state) -> (sigma_field, C_tan_field, new_state)
    state_init   : state at the last converged increment
    toler_lin, maxiter_lin : CG tolerance and cap per Newton iteration
    toler_nr, maxiter_nr   : Newton relative tolerance and cap
    C0           : (3, 3, 3, 3) fixed preconditioner stiffness; None = mean
                   of the current tangent each iteration
    delta_init   : (3, 3, Nv) initial fluctuation; warm-start load steps with
                   the previous converged strain minus the new eps_bar (a zero
                   start deep in plasticity can fail to converge)
    control      : (3, 3) 0/1 mask, 1 = stress-controlled (None = pure strain)
    stress_goal  : (3, 3) target stress, stress-controlled entries only (None = 0)
    eps_bar_free_init : (3, 3) initial stress-controlled strain correction,
                   for warm starts like delta_init
    n_devices    : device cap for the FFTs (None = all)

    Returns
    -------
    eps, sigma : (3, 3, Nv)
    state      : final state (the last converged one if not converged)
    converged  : bool
    n_iter     : Newton iterations run

    The solved macroscopic strain is ``jnp.mean(eps, axis=-1)``.
    """
    Nv = prod(n)
    iq = 1j * nyquist_safe_xi(xi_flat, n)

    control = control if control is not None else _ZERO_CONTROL
    pairs = _active_pairs(control)
    control_arr = jnp.asarray(control, dtype=eps_bar.dtype)
    stress_goal = jnp.zeros((3, 3), dtype=eps_bar.dtype) if stress_goal is None else stress_goal

    def fft_(x):
        return distributed_fft_flat(x, n, n_devices=n_devices)

    def ifft_(x):
        return distributed_ifft_flat(x, n, n_devices=n_devices)

    def div_of(sigma_field):
        """div(sigma): (3, 3, Nv) -> (3, Nv)."""
        sigma_hat = fft_(sigma_field)
        return ifft_(jnp.einsum("ijm,jm->im", sigma_hat, iq))

    def strain_from_correction(dU):
        """Zero-mean symmetric gradient of dU; the mean is carried by eps_bar_free."""
        dU_hat   = fft_(dU)
        grad_hat = jnp.einsum("im,jm->ijm", dU_hat, iq)
        eps_hat  = 0.5 * (grad_hat + jnp.transpose(grad_hat, (1, 0, 2)))
        eps_hat  = eps_hat.at[:, :, 0].set(0.0)
        return ifft_(eps_hat)

    sv2sm = partial(_sv2sm, pairs=pairs)
    sm2sv = partial(_sm2sv, pairs=pairs)

    def unpack(x_flat):
        du = x_flat[: 3 * Nv].reshape(3, Nv)
        sv = x_flat[3 * Nv:]
        return du, sv

    def pack(du, sv):
        return jnp.concatenate([du.reshape(-1), sv])

    def strain_from_u(du, deps_bar_free):
        """Symmetric gradient of du with ``deps_bar_free`` in the DC bin (inside the CG solve)."""
        du_hat   = fft_(du)
        grad_hat = jnp.einsum("im,jm->ijm", du_hat, iq)
        eps_hat  = 0.5 * (grad_hat + jnp.transpose(grad_hat, (1, 0, 2)))
        eps_hat  = eps_hat.at[:, :, 0].set(Nv * deps_bar_free.astype(eps_hat.dtype))
        return ifft_(eps_hat)

    eps0  = jnp.ones((3, 3, Nv)) * (eps_bar * (1.0 - control_arr))[:, :, None]
    delta = jnp.zeros((3, 3, Nv)) if delta_init is None else delta_init
    eps_bar_free = jnp.zeros((3, 3), dtype=eps_bar.dtype) if eps_bar_free_init is None else eps_bar_free_init

    # delta must have zero mean or the average strain drifts from eps_bar; a
    # caller's warm start often doesn't, and can even fake convergence at eps = 0.
    @jit
    def _zero_mean_strain(eps_field):
        eps_hat = fft_(eps_field)
        eps_hat = eps_hat.at[:, :, 0].set(0.0)
        return ifft_(eps_hat)

    delta = _zero_mean_strain(delta)
    state = state_init
    converged = False
    n_iter = 0

    # The outer loop needs Python control flow, so jit wraps only the
    # FFT-heavy pieces: an eager pmap on committed arrays failed on 2 GPUs
    # ("Sharding passed to jit does not match").
    @jit
    def _residual(eps_k, state):
        sigma_k, C_tan_k, new_state = local_update(eps_k, state)
        residual_div   = div_of(sigma_k)
        sigma_sum      = jnp.real(fft_(sigma_k)[:, :, 0])
        residual_extra = sm2sv(sigma_sum) - Nv * sm2sv(stress_goal)
        resid_norm = jnp.sqrt(jnp.sum(residual_div ** 2) + jnp.sum(residual_extra ** 2))
        ref_norm   = jnp.linalg.norm(sigma_k) + 1e-30
        return sigma_k, C_tan_k, new_state, residual_div, residual_extra, resid_norm, ref_norm

    @jit
    def _cg_correction(C_tan_k, bb):
        def A_op(x_flat):
            du, sv      = unpack(x_flat)
            eps_trial   = strain_from_u(du, sv2sm(sv))
            sigma_trial = jnp.einsum("ijklm,klm->ijm", C_tan_k, eps_trial)
            sigma_hat   = fft_(sigma_trial)
            div_flat    = ifft_(jnp.einsum("ijm,jm->im", sigma_hat, iq))
            # Negated so the bordered operator is symmetric (div is minus the
            # gradient's adjoint). Unnegated it is asymmetric and indefinite,
            # and check 6's load ramp diverges; the solution is the same.
            extra_out   = -sm2sv(jnp.real(sigma_hat[:, :, 0]))
            return pack(div_flat, extra_out)

        C0_ref  = jnp.mean(C_tan_k, axis=-1) if C0 is None else C0
        K0_hat  = jnp.einsum("jm,ijkl,lm->ikm", iq.imag, C0_ref, iq.imag)
        null_pt = jnp.all(iq.imag == 0.0, axis=0)
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
            r_hat  = fft_(du)
            z_hat  = jnp.einsum("ikm,km->im", K0_inv, r_hat)
            z_du   = ifft_(z_hat)
            z_sv   = jnp.linalg.solve(P_extra, sv) if pairs else sv
            return pack(z_du, z_sv)

        x0 = jnp.zeros_like(bb)
        x_flat, cg_converged = cg_solve(A_op, bb, x0, toler_lin, maxiter_lin, M=M)

        du_sol, sv_sol = unpack(x_flat)
        delta_update = strain_from_correction(du_sol)
        return delta_update, sv_sol, cg_converged

    for n_iter in range(1, maxiter_nr + 1):
        eps_bar_free_bcast = jnp.ones((3, 3, Nv)) * eps_bar_free[:, :, None]
        eps_k = eps0 + eps_bar_free_bcast + delta
        sigma_k, C_tan_k, new_state, residual_div, residual_extra, resid_norm_j, ref_norm_j = (
            _residual(eps_k, state)
        )

        resid_norm = float(resid_norm_j)
        ref_norm   = float(ref_norm_j)
        if resid_norm / ref_norm < toler_nr:
            converged = True
            state = new_state
            break

        # +residual_extra: the extra row of A_op is negated, so its RHS is too.
        bb = pack((-residual_div).reshape(-1), residual_extra)
        delta_update, sv_sol, _cg_converged = _cg_correction(C_tan_k, bb)

        delta = delta + delta_update
        eps_bar_free = eps_bar_free + sv2sm(sv_sol)
        # ``state`` stays at the last converged increment: the return mapping
        # is defined relative to it. Advancing it per iteration ratchets
        # plastic flow; on a 152^3 RVE that made Newton diverge (check 4).
    else:
        pass

    eps_bar_free_bcast = jnp.ones((3, 3, Nv)) * eps_bar_free[:, :, None]
    eps_final = eps0 + eps_bar_free_bcast + delta
    sigma_final, _, _ = local_update(eps_final, state)
    return eps_final, sigma_final, state, converged, n_iter
