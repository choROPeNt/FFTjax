import os
os.environ["JAX_ENABLE_X64"] = "1"

import jax
import jax.numpy as jnp
from functools import partial
from math import prod

from utils.cg import cg_count

# Strain energy functions live in mat_models.elastic — re-exported here for
# convenience so solver code has a single import target.
from mat_models.elastic import (  # noqa: F401
    strain_energy_density,
    strain_energy_miehe_split,
    strain_energy_amor_split,
    strain_energy_spectral_split,   # backward-compatible alias for miehe
    lame_from_C_field,
)


# ── Degradation ───────────────────────────────────────────────────────────────

def degradation(d: jnp.ndarray, k: float = 1e-6) -> jnp.ndarray:
    """
    Quadratic degradation function  g(d) = (1 - k) * (1 − d)² + k.

    k is the numerical residual stiffness k_res.  The default 1e-6 matches
    the void-phase stiffness reduction used in Schneider & Kästner (2024).

    Parameters
    ----------
    d : (Nv,)   damage field ∈ [0, 1]
    k : float   residual stiffness k_res (default 1e-6)

    Returns
    -------
    g : (Nv,)
    """
    return (1 - k)*(1.0 - d) ** 2 + k


# ── History variable ──────────────────────────────────────────────────────────

def update_history(
    H_prev:  jnp.ndarray,
    psi_pos: jnp.ndarray,
) -> jnp.ndarray:
    """
    Enforce crack irreversibility: H = max(H_prev, ψ⁺).

    Parameters
    ----------
    H_prev  : (Nv,)   history variable from the previous increment
    psi_pos : (Nv,)   current tensile strain energy density ψ⁺

    Returns
    -------
    H : (Nv,)
    """
    return jnp.maximum(H_prev, psi_pos)


# ── Helmholtz preconditioned CG ───────────────────────────────────────────────

@partial(jax.jit, static_argnames=("n", "maxiter"))
def solve_helmholtz_cg(
    L_field:  jnp.ndarray,
    xi_flat:  jnp.ndarray,
    n:        tuple[int, ...],
    l0:       float,
    Gc:       float,
    d_prev:   jnp.ndarray,
    toler_cg: float = 1e-4,
    maxiter:  int   = 300,
    eta:      float = 0.0,
    dt:       float = 1.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Preconditioned CG solve of the AT2 Helmholtz damage equation with
    optional viscous regularisation (Schneider & Kästner 2025, Eq. 33).

    Without viscosity (η = 0, default):
        (Gc/l₀ + 2L) d  −  Gc·l₀ Δd  =  2L

    With viscous regularisation (η > 0):
        (Gc/l₀ + η/Δt + 2L) d  −  Gc·l₀ Δd  =  2L + (η/Δt) dₙ

    The term η/Δt adds resistance to rapid damage rate.  Without it the
    damage snap-through is instantaneous and spatially diffuse (d jumps to
    1 in a broad zone).  With η > 0 the crack grows gradually from the
    notch tip, allowing it to fully sever the domain before σ drops to zero.

    Irreversibility is enforced post-solve: d = max(d_prev, d_cg).

    Parameters
    ----------
    L_field  : (Nv,)      crack driving force ψ⁺(x)
    xi_flat  : (ndim, Nv) angular-frequency grid from ``build_freq_grid``
    n        : tuple       grid shape — must be static for JIT
    l0       : float       phase-field length scale
    Gc       : float       critical energy release rate
    d_prev   : (Nv,)      damage from the previous increment
                           (initial guess AND irreversibility floor)
    toler_cg : float       CG relative residual tolerance
    maxiter  : int         max CG iterations — must be static for JIT
    eta      : float       viscosity η  (default 0 → no regularisation)
    dt       : float       time-step size Δt (only used when η > 0)

    Returns
    -------
    d          : (Nv,)        updated damage field ∈ [0, 1]
    iter_count : int array    CG iterations performed
    converged  : bool array   True if residual tolerance met
    """
    Nv      = prod(n)
    xi_sq   = jnp.sum(xi_flat ** 2, axis=0)   # |ξ|²  (Nv,)
    eta_dt  = eta / dt                          # viscous mass contribution

    # ── forward / inverse FFT helpers ────────────────────────────────────────
    def fft_(v):
        return jnp.fft.fftn(v.reshape(n)).reshape(Nv)

    def ifft_(v_hat):
        return jnp.fft.ifftn(v_hat.reshape(n)).real.reshape(Nv)

    # ── A operator: (Gc/l₀ + η/Δt + 2L) v − Gc·l₀ Δv ──────────────────────
    def A_op(v_flat):
        lap_v  = ifft_(-xi_sq * fft_(v_flat))
        mass_v = (Gc / l0 + eta_dt + 2.0 * L_field) * v_flat
        return mass_v - Gc * l0 * lap_v

    # ── Fourier preconditioner with averaged mass ─────────────────────────────
    L_avg   = jnp.mean(L_field)
    P_denom = Gc / l0 + eta_dt + 2.0 * L_avg + Gc * l0 * xi_sq

    def P_op(v_flat):
        return ifft_(fft_(v_flat) / P_denom)

    # ── RHS: 2L + (η/Δt) dₙ ─────────────────────────────────────────────────
    bb = 2.0 * L_field + eta_dt * d_prev

    # ── solve ─────────────────────────────────────────────────────────────────
    d_cg, iter_count, converged = cg_count(A_op, bb, d_prev, toler_cg, maxiter, M=P_op)

    d = jnp.maximum(d_prev, d_cg)
    return jnp.clip(d, 0.0, 1.0), iter_count, converged
