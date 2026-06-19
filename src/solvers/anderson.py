"""
Anderson mixing (Anderson acceleration) for fixed-point iterations  x = G(x).

Used to accelerate the staggered phase-field-fracture iteration.  Plain
alternate minimisation (one mechanical solve + one damage solve per sweep)
converges agonisingly slowly near the brittle snap-through and can stall at a
spurious half-cracked near-fixed-point — the per-sweep change drops below the
staggered tolerance before the crack has severed the domain.

Anderson mixing combines a short history of iterates to extrapolate past that
stalled region, driving the iteration to the true (fully-severed) fixed point.
This matches the PETSc Anderson-mixing solver used in the reference DAMASK
setup of Schneider & Kästner (2025), https://doi.org/10.1111/ffe.14553.

Formulation (Walker & Ni 2011, type-II / "good" Anderson):

    fₖ = G(xₖ) − xₖ                              (residual)
    γ  = argmin ‖fₖ − ΔF γ‖₂                     (unconstrained least squares)
    xₖ₊₁ = xₖ + β fₖ − (ΔX + β ΔF) γ

with ΔX, ΔF the column-wise first differences of the last ``depth`` iterates
and residuals.  β = 1 reduces the update to  xₖ₊₁ = G(xₖ) − ΔG γ.
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
