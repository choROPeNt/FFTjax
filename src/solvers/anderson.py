"""
Accelerators for the staggered phase-field-fracture fixed-point  x = G(x).

Two classes are provided:

AndersonAccelerator
    Walker & Ni 2011 Type-II Anderson mixing.  Solves an *unconstrained* LS
    on *differences* of residuals:

        fₖ = G(xₖ) − xₖ
        γ  = argmin ‖fₖ − ΔF γ‖₂        (unconstrained)
        xₖ₊₁ = xₖ + β fₖ − (ΔX + β ΔF) γ

NGMRESAccelerator
    Nonlinear GMRES — matches DAMASK's outer -snes_type ngmres solver
    (Schneider & Kästner 2025, doi:10.1111/ffe.14553).  Solves a
    *constrained* LS directly on the residuals:

        α  = argmin ‖F α‖²   s.t.  1ᵀα = 1
        xₖ₊₁ = G α                       (G columns = G(xᵢ))

    The sum-to-1 constraint stabilises the solve when residuals become
    nearly collinear near the brittle snap-through.

    Full DAMASK config adds Anderson as the inner smoother
    (-snes_ngmres_anderson), generating a better trial point before each
    NGMRES function evaluation.  That variant requires one extra staggered
    sweep per step; NGMRESAccelerator implements the outer NGMRES loop only,
    at the same cost as AndersonAccelerator (one sweep per step).
"""

import os
os.environ["JAX_ENABLE_X64"] = "1"

import jax.numpy as jnp


class AndersonAccelerator:
    """
    Stateful Anderson-mixing accelerator for a Python-driven fixed-point loop.

    Call :meth:`reset` at the start of each new fixed-point problem, then feed
    successive ``(x, G(x))`` pairs to :meth:`step`, which returns the next
    accelerated iterate.

    Parameters
    ----------
    depth : int
        Window size m — number of past iterates retained.  Typical 3–5.
    beta  : float
        Mixing / damping factor β ∈ (0, 1].  β = 1 is undamped Anderson;
        smaller values blend toward plain Picard for robustness.
    reg   : float
        Tikhonov regularisation added to the normal-equations matrix to keep
        the least-squares solve well-posed when differences become collinear.
    """

    def __init__(self, depth: int = 5, beta: float = 1.0, reg: float = 1e-10):
        self.depth = depth
        self.beta  = beta
        self.reg   = reg
        self.reset()

    def reset(self) -> None:
        """Clear the iterate/residual history (new fixed-point problem)."""
        self._X: list[jnp.ndarray] = []   # iterates xₖ
        self._F: list[jnp.ndarray] = []   # residuals fₖ = G(xₖ) − xₖ

    def step(self, x: jnp.ndarray, gx: jnp.ndarray) -> jnp.ndarray:
        """
        Produce the next iterate from the current iterate ``x`` and its map
        value ``gx = G(x)``.

        The first call (empty history) returns the damped Picard step
        ``x + β (G(x) − x)``, so the first staggered sweep is unchanged when
        β = 1.
        """
        f = gx - x
        self._X.append(x)
        self._F.append(f)

        # trim to the most recent `depth` (+1, to form `depth` differences)
        if len(self._F) > self.depth + 1:
            self._X.pop(0)
            self._F.pop(0)

        if len(self._F) == 1:
            return x + self.beta * f

        # column-wise first differences over the window
        dX = jnp.stack([self._X[i + 1] - self._X[i]
                        for i in range(len(self._X) - 1)], axis=1)   # (Nv, m)
        dF = jnp.stack([self._F[i + 1] - self._F[i]
                        for i in range(len(self._F) - 1)], axis=1)   # (Nv, m)

        # γ = argmin ‖f − dF γ‖  via regularised normal equations
        AtA   = dF.T @ dF + self.reg * jnp.eye(dF.shape[1])
        Atb   = dF.T @ f
        gamma = jnp.linalg.solve(AtA, Atb)                          # (m,)

        # xₖ₊₁ = xₖ + β fₖ − (ΔX + β ΔF) γ
        return x + self.beta * f - (dX + self.beta * dF) @ gamma


# ── NGMRES ─────────────────────────────────────────────────────────────────────

class NGMRESAccelerator:
    """
    NGMRES (Nonlinear GMRES) for fixed-point x = G(x).

    Implements the outer -snes_type ngmres solver used in DAMASK
    (Schneider & Kästner 2025, doi:10.1111/ffe.14553).

    At each step the window of (xᵢ, G(xᵢ)) pairs grows by one entry; the
    constrained least-squares problem

        α  = argmin ‖F α‖²   subject to  1ᵀα = 1
        xₖ₊₁ = G α                       (G columns = G(xᵢ), F cols = G(xᵢ)−xᵢ)

    is then solved via regularised normal equations and a Lagrange-multiplier
    normalisation.  The sum-to-1 constraint (absent in Anderson) prevents the
    solution from drifting when residuals become collinear near snap-through,
    and guarantees exact recovery of any fixed-point x* = G(x*).

    API is drop-in compatible with AndersonAccelerator (same step / reset
    signatures, no beta parameter).  Cost is identical: one staggered sweep
    per outer iteration, no extra function evaluations.

    Parameters
    ----------
    depth : int
        Window size m — number of past (x, G(x)) pairs retained.
    reg   : float
        Tikhonov regularisation on FᵀF to handle near-collinear residuals.
    """

    def __init__(self, depth: int = 5, reg: float = 1e-10):
        self.depth = depth
        self.reg   = reg
        self.reset()

    def reset(self) -> None:
        """Clear window (call at the start of each new fixed-point problem)."""
        self._X: list[jnp.ndarray] = []   # iterates xᵢ
        self._G: list[jnp.ndarray] = []   # map values G(xᵢ)

    def step(self, x: jnp.ndarray, gx: jnp.ndarray) -> jnp.ndarray:
        """
        Return the NGMRES-optimal next iterate given xₖ and G(xₖ) = gx.
        On the first call (empty window) returns gx (plain Picard step).
        """
        self._X.append(x)
        self._G.append(gx)

        if len(self._X) > self.depth:
            self._X.pop(0)
            self._G.pop(0)

        m = len(self._X)
        if m == 1:
            return gx   # plain Picard — window not yet populated

        F = jnp.stack([self._G[i] - self._X[i] for i in range(m)], axis=1)  # (Nv, m)
        G = jnp.stack(self._G,                                   axis=1)      # (Nv, m)

        # Constrained LS solution via Lagrange multiplier:
        #   α = (FᵀF)⁻¹ 1 / (1ᵀ (FᵀF)⁻¹ 1)   ←→  argmin ‖Fα‖² s.t. 1ᵀα = 1
        AtA   = F.T @ F + self.reg * jnp.eye(m)
        ones  = jnp.ones(m)
        v     = jnp.linalg.solve(AtA, ones)   # (FᵀF)⁻¹ 1
        alpha = v / (ones @ v)                 # normalise: 1ᵀα = 1

        return G @ alpha                       # output combination Σ αᵢ G(xᵢ)
