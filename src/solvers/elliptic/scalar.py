"""
Scalar elliptic FFT-CG solve. First user: the AT2 phase-field damage
sub-problem (Helmholtz-type screened Poisson equation). Homogeneous Gc
only -- add a heterogeneous-Gc variant here if a problem needs spatially
varying fracture toughness (see project memory for the divergence-form
operator that needs, unlike this one, Nyquist-zeroed ξ for its gradient/
divergence round-trip).
"""

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)

from functools import partial
from math import prod

import jax
import jax.numpy as jnp

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
