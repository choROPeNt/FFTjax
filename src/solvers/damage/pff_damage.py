import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)

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


def update_history_hybrid(
    H_prev:  jnp.ndarray,
    psi_pos: jnp.ndarray,
    d_prev:  jnp.ndarray,
    d_thres: float = 0.95,
) -> jnp.ndarray:
    """
    Hybrid damage-/crack-like irreversibility (Steinke & Kaliske 2019, as
    adopted by Schneider & Kästner 2025, doi:10.1111/ffe.14553).

    Pure damage-like irreversibility (``update_history``: H = max(H_prev, ψ⁺)
    everywhere) locks in the history field as soon as any damage nucleates,
    which over-widens the diffuse process zone. The hybrid formulation only
    enforces the monotone history lock once a point has effectively cracked
    (d ≥ d_thres); below the threshold the driving force is left unrestricted
    so the pre-crack process zone can still relax:

        H = ψ⁺                     if d_prev < d_thres   (unrestricted)
        H = max(H_prev, ψ⁺)        if d_prev ≥ d_thres    (irreversible)

    Parameters
    ----------
    H_prev  : (Nv,)   history variable from the previous increment
    psi_pos : (Nv,)   current tensile strain energy density ψ⁺
    d_prev  : (Nv,)   damage field used to gate the switch (current best
                       estimate of d — e.g. from the previous staggered
                       iteration or the previous increment)
    d_thres : float   phase-field threshold above which irreversibility is
                       enforced (default 0.95)

    Returns
    -------
    H : (Nv,)
    """
    return jnp.where(d_prev >= d_thres, jnp.maximum(H_prev, psi_pos), psi_pos)


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


# ── Heterogeneous-Gc Helmholtz CG ─────────────────────────────────────────────

@partial(jax.jit, static_argnames=("n", "maxiter"))
def solve_helmholtz_cg_het(
    L_field:  jnp.ndarray,
    xi_flat:  jnp.ndarray,
    n:        tuple[int, ...],
    l0:       float,
    Gc_field: jnp.ndarray,
    d_prev:   jnp.ndarray,
    toler_cg: float = 1e-4,
    maxiter:  int   = 300,
    eta:      float = 0.0,
    dt:       float = 1.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Preconditioned CG solve for the AT2 damage equation with spatially varying
    fracture toughness Gc(x).

    The operator is in divergence form (correct variational derivative of
    ``∫ Gc(x) [d²/2l + l/2 |∇d|²] dx``):

        -l₀ ∇·(Gc(x) ∇d) + (Gc(x)/l₀ + η/Δt + 2H) d = 2H + (η/Δt) dₙ

    Gc lives inside the divergence — factoring it out is wrong for varying Gc.
    The operator is SPD (= Gᵀ diag(Gc·l₀) G + c(x), c > 0), so CG converges.

    Preconditioner: constant-coefficient reference-medium operator (same trick
    as Lippmann–Schwinger) with Gc₀ = geometric mean of Gc_field, inverted
    exactly in Fourier.  Geometric mean preconditions high-contrast fields
    better than arithmetic mean.

    Parameters
    ----------
    L_field  : (Nv,)       crack driving force ψ⁺(x)
    xi_flat  : (ndim, Nv)  angular-frequency grid from ``build_freq_grid``
    n        : tuple        grid shape — static for JIT
    l0       : float        phase-field length scale
    Gc_field : (Nv,)        spatially varying critical energy release rate
    d_prev   : (Nv,)        damage from previous increment
    toler_cg : float        CG relative residual tolerance
    maxiter  : int          max CG iterations — static for JIT
    eta      : float        viscosity η
    dt       : float        time-step size Δt

    Returns
    -------
    d          : (Nv,)
    iter_count : int array
    converged  : bool array
    """
    Nv     = prod(n)
    ndim   = xi_flat.shape[0]
    eta_dt = eta / dt

    def fft_(v):
        return jnp.fft.fftn(v.reshape(n)).reshape(Nv)

    def ifft_(v_hat):
        return jnp.fft.ifftn(v_hat.reshape(n)).real.reshape(Nv)

    xi_sq = jnp.sum(xi_flat ** 2, axis=0)   # full |ξ|² including Nyquist

    # Anti-symmetric xi: zero the Nyquist for each even-sized dimension.
    # The spectral gradient ifft(1j·ξ·V̂).real silently discards the Nyquist
    # (which is purely imaginary in Fourier), so fft(Gc·ifft(1j·ξ·V̂).real) ≠
    # Gc·1j·ξ·V̂ at Nyquist — the round-trip introduces error.  Zeroing the
    # Nyquist in ξ makes the round-trip exact for the correction term.
    xi_anti = []
    for i in range(ndim):   # unrolled at trace time — ndim and n are static
        if n[i] % 2 == 0:
            idx    = [slice(None)] * ndim
            idx[i] = n[i] // 2
            xi_i   = xi_flat[i].reshape(n).at[tuple(idx)].set(0.0).reshape(Nv)
        else:
            xi_i   = xi_flat[i]
        xi_anti.append(xi_i)
    xi_anti = jnp.stack(xi_anti)            # (ndim, Nv)

    # geometric-mean reference Gc₀ — better than arithmetic mean at high contrast
    Gc0  = jnp.exp(jnp.mean(jnp.log(Gc_field)))
    dGc  = Gc_field - Gc0                  # spatial Gc variation

    # Polarisation (Lippmann-Schwinger) decomposition:
    #   -l₀ ∇·(Gc ∇v) = -Gc₀·l₀·∇²v  -  l₀·∇·(δGc ∇v)
    #
    # Reference term:  exact scalar Laplacian with Gc₀ — diagonal in Fourier,
    #                  no round-trip, Nyquist handled correctly.
    # Correction term: only δGc = Gc − Gc₀ through the gradient round-trip.
    #                  For constant Gc (δGc = 0) this is identically zero, so
    #                  A_op reduces exactly to the scalar Laplacian.
    def A_op(v_flat):
        v_hat    = fft_(v_flat)
        # reference: exact spectral Laplacian with Gc₀
        ref      = (Gc0 / l0 + eta_dt + 2.0 * L_field) * v_flat + Gc0 * l0 * ifft_(xi_sq * v_hat)
        # correction: -l₀ ∇·(δGc ∇v) via real-space round-trip with Nyquist-zeroed ξ
        div_hat  = jnp.zeros(Nv, dtype=v_hat.dtype)
        for i in range(ndim):
            flux_i  = dGc * l0 * ifft_(1j * xi_anti[i] * v_hat)
            div_hat = div_hat + 1j * xi_anti[i] * fft_(flux_i)
        return ref + ifft_(-div_hat)

    L_avg   = jnp.mean(L_field)
    # Preconditioner: exact scalar operator with Gc₀, inverted in Fourier.
    # Gc₀/l₀ keeps the denominator > 0 at k=0.
    P_denom = Gc0 / l0 + eta_dt + 2.0 * L_avg + Gc0 * l0 * xi_sq

    def P_op(v_flat):
        return ifft_(fft_(v_flat) / P_denom)

    bb = 2.0 * L_field + eta_dt * d_prev

    d_cg, iter_count, converged = cg_count(A_op, bb, d_prev, toler_cg, maxiter, M=P_op)

    d = jnp.maximum(d_prev, d_cg)
    return jnp.clip(d, 0.0, 1.0), iter_count, converged
