"""
Scalar elliptic FFT-CG solve. First user: the AT2 phase-field damage
sub-problem (Helmholtz-type screened Poisson equation).

Two variants: ``solve_damage_helmholtz_cg`` (homogeneous Gc, a single
scalar for the whole domain) and ``solve_damage_helmholtz_cg_het``
(heterogeneous Gc, an (Nv,) per-voxel field). They solve genuinely
different equations, not the same one at different generality -- see
the heterogeneous solver's docstring for why a spatially varying Gc
changes the diffusion term itself (Gc(x)*lap(d) is wrong; it has to be
div(Gc(x)*grad(d))), not just which value gets plugged in where.
"""

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)

from functools import partial
from math import prod

import jax
import jax.numpy as jnp

from operators.green import nyquist_safe_xi
from solvers.krylov.cg import cg_solve


@partial(jax.jit, static_argnames=("n", "maxiter"))
def solve_damage_helmholtz_cg(
    H_field: jnp.ndarray,
    xi_flat: jnp.ndarray,
    n: tuple[int, ...],
    l0: float,
    Gc: float,
    d_prev: jnp.ndarray,
    toler_cg: float = 1e-4,
    maxiter: int = 300,
    eta: float = 0.0,
    dt: float = 1.0,
    k: float | jnp.ndarray = 0.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Preconditioned CG solve of the AT2 Helmholtz damage equation with
    optional viscous regularisation (Schneider & Kästner 2025, Eq. 33).

    Stationarity of  Π = ∫ [g(d)ψ⁺ + ψ⁻] dV + Gc∫[d²/2l₀ + l₀/2|∇d|²] dV
    w.r.t. d gives  Gc/l₀·d − Gc·l₀·Δd = −g'(d)·ψ⁺. For the AT2 residual-
    stiffness degradation g(d) = (1−k)(1−d)² + k (see
    ``materialmodels.phasefield.degradation.degradation_at2``),
    −g'(d) = (1−k)(2−2d) is affine in d (verified against ``jax.grad`` of
    ``degradation_at2``), which is what makes this a closed-form linear CG
    solve rather than a per-voxel Newton iteration:

    Without viscosity (η = 0, default):
        (Gc/l₀ + 2(1−k)H) d  −  Gc·l₀ Δd  =  2(1−k)H

    With viscous regularisation (η > 0):
        (Gc/l₀ + η/Δt + 2(1−k)H) d  −  Gc·l₀ Δd  =  2(1−k)H + (η/Δt) dₙ

    At k=1 (damage-immune voxel, see ``materialmodels.elastic.isotropic.
    LinearElasticIsotropic``'s ``k_res``), both terms vanish exactly: local
    energy no longer drives that voxel's damage, matching g'(d)≡0 there --
    it can still be pulled up by the Laplacian term from neighbouring
    voxels, just not by its own ψ⁺.

    The η/Δt term adds resistance to rapid damage rate. Without it, damage
    snap-through is instantaneous and spatially diffuse (d jumps to 1 in a
    broad zone). With η > 0 the crack grows gradually from the notch tip,
    allowing it to fully sever the domain before σ drops to zero.

    Irreversibility is enforced post-solve: d = max(d_prev, d_cg).

    Parameters
    ----------
    H_field  : (Nv,)      crack driving force / history variable ψ⁺(x)
    xi_flat  : (ndim, Nv) angular-frequency grid from ``operators.green.build_freq_grid``
    n        : tuple       grid shape -- must be static for JIT
    l0       : float       phase-field length scale
    Gc       : float       critical energy release rate (homogeneous)
    d_prev   : (Nv,)      damage from the previous iteration (initial guess
                           AND irreversibility floor)
    toler_cg : float       CG relative residual tolerance
    maxiter  : int         max CG iterations -- must be static for JIT
    eta      : float       viscosity η (default 0 -> no regularisation)
    dt       : float       time-step size Δt (only used when η > 0)
    k        : float or (Nv,)  AT2 residual stiffness k_res -- must match
                           whatever k was used to degrade the stiffness
                           this increment's H_field came from (default 0.0,
                           the "pure" AT2 form with no residual stiffness --
                           callers with a nonzero/per-voxel k_res, e.g.
                           ``problems.fracture.solve_fracture``, must pass
                           it here too or the damage field drifts out of
                           sync with the true variational stationarity
                           condition -- negligibly for k~1e-6, not
                           negligibly as k approaches 1)

    Returns
    -------
    d          : (Nv,)        updated damage field in [0, 1]
    converged  : bool array   True if residual tolerance met
    """
    Nv = prod(n)
    xi_sq = jnp.sum(xi_flat ** 2, axis=0)   # |ξ|²  (Nv,) -- even power, no Nyquist issue
    eta_dt = eta / dt
    driving = 2.0 * (1.0 - k) * H_field   # -g'(d)·ψ⁺, affine-in-d term folded into the mass coefficient below

    def fft_(v):
        return jnp.fft.fftn(v.reshape(n)).reshape(Nv)

    def ifft_(v_hat):
        return jnp.fft.ifftn(v_hat.reshape(n)).real.reshape(Nv)

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


@partial(jax.jit, static_argnames=("n", "maxiter"))
def solve_damage_helmholtz_cg_het(
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
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Preconditioned CG solve of the AT2 Helmholtz damage equation for a
    spatially varying (per-voxel) critical energy release rate Gc(x).

    Differs from ``solve_damage_helmholtz_cg`` in exactly one place, but
    it's not a drop-in generalisation -- the variational derivative of the
    ``Gc(x)*(l0/2)|grad d|^2`` energy term is ``div(Gc(x)*grad(d))``, not
    ``Gc(x)*lap(d)`` (product rule: ``div(Gc grad d) = Gc*lap(d) +
    grad(Gc).grad(d)``, an extra term wherever Gc varies -- including at a
    sharp phase interface, not just for a smoothly-varying field). Plugging
    a per-voxel Gc(x) pointwise into the homogeneous solver's ``Gc*lap(d)``
    term would silently solve the wrong equation.

    Stationarity gives (everything but the diffusion term identical to the
    homogeneous case -- see that function's docstring for the derivation
    and the affine-in-d driving-force trick):

    Without viscosity (η = 0, default):
        (Gc(x)/l0 + 2(1−k)H) d  −  l0 div(Gc(x) grad d)  =  2(1−k)H

    With viscous regularisation (η > 0):
        (Gc(x)/l0 + η/Δt + 2(1−k)H) d − l0 div(Gc(x) grad d) = 2(1−k)H + (η/Δt) dₙ

    Numerically this costs 3 FFT round-trips per CG matvec instead of 1
    (gradient, real-space multiply by Gc(x), divergence) -- gradient and
    divergence are odd powers of ξ, so both need ``operators.green.
    nyquist_safe_xi`` (unlike the homogeneous operator's single even-power
    ``-|ξ|²`` Laplacian trick, which has no Nyquist issue and needs no
    real-space Gc multiply at all). The preconditioner is no longer exactly
    diagonal in Fourier space either (real-space multiplication by Gc(x)
    couples modes) -- approximated with a reference-medium Gc0 = mean(Gc),
    i.e. the homogeneous operator's own (exact, cheap) preconditioner,
    mirroring how ``solvers.elliptic.vector.displacement_based`` builds its
    preconditioner from a reference stiffness C0 = mean(C_field).

    Irreversibility is enforced post-solve: d = max(d_prev, d_cg).

    Parameters
    ----------
    H_field  : (Nv,)      crack driving force / history variable ψ⁺(x)
    xi_flat  : (ndim, Nv) angular-frequency grid from ``operators.green.build_freq_grid``
    n        : tuple       grid shape -- must be static for JIT
    l0       : float       phase-field length scale
    Gc       : (Nv,)      critical energy release rate, per voxel
    d_prev   : (Nv,)      damage from the previous iteration (initial guess
                           AND irreversibility floor)
    toler_cg : float       CG relative residual tolerance
    maxiter  : int         max CG iterations -- must be static for JIT
    eta      : float       viscosity η (default 0 -> no regularisation)
    dt       : float       time-step size Δt (only used when η > 0)
    k        : float or (Nv,)  AT2 residual stiffness k_res -- see
                           ``solve_damage_helmholtz_cg``'s docstring

    Returns
    -------
    d          : (Nv,)        updated damage field in [0, 1]
    converged  : bool array   True if residual tolerance met
    """
    Nv = prod(n)
    ndim = xi_flat.shape[0]
    iq = 1j * nyquist_safe_xi(xi_flat, n)   # (ndim, Nv) -- odd-power ξ, needs Nyquist zeroing
    eta_dt = eta / dt
    driving = 2.0 * (1.0 - k) * H_field

    def fft_(v):
        return jnp.fft.fftn(v.reshape(n)).reshape(Nv)

    def ifft_(v_hat):
        return jnp.fft.ifftn(v_hat.reshape(n)).real.reshape(Nv)

    def fft_vec(v):
        return jnp.fft.fftn(v.reshape(ndim, *n), axes=(-3, -2, -1)).reshape(ndim, Nv)

    def ifft_vec(v_hat):
        return jnp.fft.ifftn(v_hat.reshape(ndim, *n), axes=(-3, -2, -1)).real.reshape(ndim, Nv)

    def div_Gc_grad(v_flat):
        """div(Gc(x) * grad(v_flat)) via gradient -> real-space Gc multiply -> divergence."""
        grad = ifft_vec(iq * fft_(v_flat)[None, :])   # (ndim, Nv) real
        q_hat = fft_vec(Gc[None, :] * grad)             # Gc(x) applied in real space
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
