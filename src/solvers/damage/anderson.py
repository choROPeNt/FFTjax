"""
Accelerators for the staggered mechanics<->damage fixed point.

Plain (Picard) staggering is the fixed-point iteration x_{k+1} = G(x_k) with
x the damage field and G one full sweep (mechanical solve -> driving force ->
history update -> damage solve). It converges linearly, and near the brittle
snap-through -- where the two sub-problems become strongly coupled -- the rate
degrades badly. Measured in this package on a 94x98 RVE with a plastic matrix,
the sweep count per load increment ran

    2  3  5  7  7  8  12  16  32  44

i.e. the last two increments alone cost more than the first eight combined,
and on a 152^3 grid a sweep is ~41 s, so 44 sweeps is half an hour for ONE
increment. That is what these accelerators exist to cut.

Both reuse only the (x_k, G(x_k)) pairs the loop already produces, so they
cost one sweep per outer iteration, exactly as Picard does -- no extra
function evaluations, no Jacobian.

    AndersonAccelerator -- Anderson mixing (Walker & Ni 2011, Type-II).
        UNCONSTRAINED least squares on windowed DIFFERENCES of the residual
        f_k = G(x_k) - x_k:

            gamma = argmin || f_k - dF gamma ||^2
            x_{k+1} = x_k + beta*f_k - (dX + beta*dF) gamma

        with dX, dF the matrices of successive differences of the stored x and
        f. beta in (0, 1] damps the step; Tikhonov regularization keeps the
        normal equations usable once the residual columns go nearly collinear,
        which is exactly what happens late in a snap-through.

    NGMRESAccelerator -- Nonlinear GMRES.
        CONSTRAINED least squares directly on the residuals,

            alpha = argmin || F alpha ||^2   subject to   1^T alpha = 1
            x_{k+1} = G alpha = sum_i alpha_i g_i

        i.e. it recombines past MAP VALUES rather than correcting the current
        iterate. This is DAMASK's outer ``-snes_type ngmres`` (Schneider &
        Kaestner 2025, doi:10.1111/ffe.14553). The sum-to-one constraint is
        what makes it robust to collinear residuals, and it guarantees exact
        recovery of a fixed point: if every stored g_i equals its x_i then any
        admissible alpha reproduces x*.

Shared API, so either drops into the staggered loop interchangeably:

    acc = AndersonAccelerator(depth=5, beta=1.0)      # or NGMRESAccelerator(depth=5)
    acc.reset()                                        # once per load increment
    for k in range(maxiter):
        g = G(x)                                       # one staggered sweep
        x = acc.step(x, g)                             # accelerated next iterate

``step`` returns the plain Picard value ``g`` on the first call (nothing to
extrapolate from yet), so an accelerator with depth 1 IS Picard and the
un-accelerated path is recovered exactly.

Two things these do that a textbook Anderson does not, both needed here
-----------------------------------------------------------------------
1. ``bounds`` / ``floor`` -- the iterate is a DAMAGE field. An extrapolated
   step is a linear combination with possibly negative weights, so it can
   leave [0, 1] or move damage BACKWARDS, neither of which is a physical
   damage field. Both are clamped: to ``bounds`` elementwise, and to
   ``floor`` (the increment's starting damage) so irreversibility survives
   acceleration. Without the floor the accelerator can un-crack voxels and
   the load increment converges to the wrong state.
2. A SAFEGUARD. Anderson has no descent guarantee and genuinely diverges
   when the window is ill-conditioned, so ``step`` monitors itself and falls
   back to Picard, restarting the history, when it is not helping.

   The test is RETROSPECTIVE: after an accelerated step, the next sweep
   reveals the residual that step actually produced, and if
   ||f_{k+1}|| > safeguard_factor * ||f_k|| the acceleration hurt. This costs
   nothing -- the sweep happens regardless -- and it is the only honest test
   available, because judging a step before taking it would need G(cand),
   i.e. the extra sweep these accelerators exist to avoid.

   A PROSPECTIVE test on step length (reject if ||cand - g|| exceeds some
   multiple of ||f||) was tried first and is WRONG: an accelerated step is
   supposed to travel further than the Picard step, so that rule rejects
   precisely the steps that work. Measured on a linear fixed point with
   spectral radius 0.97 it rejected 497 of 499 steps and gave exactly no
   speedup. Do not reintroduce it.

   Rejections are counted in ``n_rejected`` so a caller can see whether
   acceleration is earning its place.

Deliberately NumPy, not JAX
---------------------------
The accelerated update is O(depth^2) work on depth-length dot products of
(Nv,) vectors -- negligible against one staggered sweep (a Newton solve plus
a damage CG), and the least-squares solve is a (depth x depth) dense system.
Running it in NumPy keeps it out of the traced graph entirely, so it adds no
compilation and no jit boundary to the loop that calls it; the only cost is
pulling the two (Nv,) vectors to host per sweep, which is already happening
(the loop concretizes its convergence check anyway). Do NOT reach for JAX
here without a measurement: at depth 5 on 3.5M voxels this is ~0.1% of a
sweep.
"""

from __future__ import annotations

import numpy as np


class _AccelBase:
    """
    Shared history bookkeeping, clamping and safeguard for both accelerators.

    Subclasses implement ``_extrapolate(x, g)`` returning a candidate next
    iterate, or None to fall back to Picard.
    """

    def __init__(self, depth: int = 5, reg: float = 1e-10,
                 bounds: tuple[float, float] | None = (0.0, 1.0),
                 safeguard_factor: float = 2.0):
        if depth < 1:
            raise ValueError(f"depth={depth} must be >= 1 (depth 1 is plain Picard)")
        if reg < 0.0:
            raise ValueError(f"reg={reg} must be >= 0")
        if safeguard_factor <= 1.0:
            raise ValueError(
                f"safeguard_factor={safeguard_factor} must be > 1 -- it is the "
                "factor by which the residual may GROW after an accelerated "
                "step before that step is judged harmful, so <= 1 restarts on "
                "any step that is not monotonically improving, which Anderson "
                "legitimately is not"
            )
        self.depth = int(depth)
        self.reg = float(reg)
        self.bounds = bounds
        self.safeguard_factor = float(safeguard_factor)
        self.reset()

    def reset(self, floor: np.ndarray | None = None) -> None:
        """
        Drop the history. Call once per LOAD INCREMENT: the fixed point
        changes with the applied strain, so pairs from the previous increment
        extrapolate toward the wrong solution.

        ``floor`` is the increment's starting damage field, used as the
        irreversibility floor for every clamped iterate (see the module
        docstring). None disables the floor.
        """
        self._X: list[np.ndarray] = []
        self._G: list[np.ndarray] = []
        self._F: list[np.ndarray] = []
        self._floor = None if floor is None else np.asarray(floor, dtype=float).copy()
        self._norm_f_prev: float | None = None
        self._last_accelerated = False
        self.n_steps = 0
        self.n_rejected = 0
        self.n_accelerated = 0

    # ── clamping ────────────────────────────────────────────────────────
    def _clamp(self, x: np.ndarray) -> np.ndarray:
        if self._floor is not None:
            x = np.maximum(x, self._floor)
        if self.bounds is not None:
            x = np.clip(x, self.bounds[0], self.bounds[1])
        return x

    # ── the shared driver ───────────────────────────────────────────────
    def step(self, x, g):
        """
        One accelerated update.

        Parameters
        ----------
        x : (Nv,) the iterate this sweep started from
        g : (Nv,) G(x), the sweep's output (the plain Picard next iterate)

        Returns
        -------
        (Nv,) numpy array -- the next iterate, clamped to ``bounds`` and
        ``floor``. Equals ``g`` (Picard) on the first call, whenever the
        least-squares solve is degenerate, and whenever the safeguard rejects
        the extrapolation.
        """
        x = np.asarray(x, dtype=float).ravel()
        g = np.asarray(g, dtype=float).ravel()
        f = g - x
        norm_f = float(np.linalg.norm(f))
        self.n_steps += 1
        picard = self._clamp(g)

        # Retrospective safeguard, evaluated BEFORE this sweep's pair is used:
        # the residual we are looking at now is the verdict on the PREVIOUS
        # accelerated step. If it grew, that step hurt -- restart from Picard.
        if (self._last_accelerated and self._norm_f_prev is not None
                and norm_f > self.safeguard_factor * self._norm_f_prev):
            self.n_rejected += 1
            self._X, self._G, self._F = [x.copy()], [g.copy()], [f.copy()]
            self._norm_f_prev = norm_f
            self._last_accelerated = False
            return picard

        self._X.append(x.copy())
        self._G.append(g.copy())
        self._F.append(f.copy())
        if len(self._X) > self.depth:
            self._X.pop(0)
            self._G.pop(0)
            self._F.pop(0)
        self._norm_f_prev = norm_f

        if len(self._X) < 2:
            self._last_accelerated = False
            return picard        # nothing to extrapolate from yet

        cand = self._extrapolate()
        if cand is None or not np.all(np.isfinite(cand)):
            self.n_rejected += 1
            self._flush()
            self._last_accelerated = False
            return picard

        self.n_accelerated += 1
        self._last_accelerated = True
        return self._clamp(cand)

    def _flush(self) -> None:
        """Keep only the newest pair -- a rejected step means the window is
        misleading, but the latest sweep is still valid information."""
        self._X = self._X[-1:]
        self._G = self._G[-1:]
        self._F = self._F[-1:]

    def _extrapolate(self):
        raise NotImplementedError

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(depth={self.depth}, reg={self.reg:.1e}, "
                f"steps={self.n_steps}, accelerated={self.n_accelerated}, "
                f"rejected={self.n_rejected})")


class AndersonAccelerator(_AccelBase):
    """
    Anderson mixing, Type-II (Walker & Ni 2011).

    Parameters
    ----------
    depth : int    history window m. depth 1 is plain Picard.
    beta  : float  mixing/damping in (0, 1]. 1.0 is undamped Anderson; lower
            values trade rate for robustness, which is the usual remedy when
            the snap-through makes undamped steps overshoot.
    reg   : float  Tikhonov parameter, scaled by trace(dF^T dF)/m so it is
            dimensionless in the residual's own units rather than tied to the
            problem's stress scale.
    bounds, safeguard_factor : see _AccelBase.
    """

    def __init__(self, depth: int = 5, beta: float = 1.0, reg: float = 1e-10,
                 bounds: tuple[float, float] | None = (0.0, 1.0),
                 safeguard_factor: float = 2.0):
        if not 0.0 < beta <= 1.0:
            raise ValueError(f"beta={beta} must be in (0, 1]")
        self.beta = float(beta)
        super().__init__(depth=depth, reg=reg, bounds=bounds,
                         safeguard_factor=safeguard_factor)

    def _extrapolate(self):
        # Successive differences, columns oldest -> newest.
        dX = np.stack([self._X[i + 1] - self._X[i]
                       for i in range(len(self._X) - 1)], axis=1)
        dF = np.stack([self._F[i + 1] - self._F[i]
                       for i in range(len(self._F) - 1)], axis=1)
        f_k = self._F[-1]
        x_k = self._X[-1]

        # Normal equations with Tikhonov. Scaling the regularizer by the mean
        # diagonal keeps `reg` dimensionless: the residual here is a damage
        # difference, O(1e-3), so an absolute 1e-10 would be either inert or
        # dominant depending on where in the snap-through we are.
        A = dF.T @ dF
        m = A.shape[0]
        scale = float(np.trace(A)) / m if m else 0.0
        if not np.isfinite(scale) or scale <= 0.0:
            return None
        A = A + self.reg * scale * np.eye(m)
        b = dF.T @ f_k
        try:
            gamma = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return None
        # x_{k+1} = x_k + beta*f_k - (dX + beta*dF) gamma
        return x_k + self.beta * f_k - (dX + self.beta * dF) @ gamma


class NGMRESAccelerator(_AccelBase):
    """
    Nonlinear GMRES -- DAMASK's outer ``-snes_type ngmres``.

    Solves the CONSTRAINED problem

        alpha = argmin || F alpha ||^2   subject to   1^T alpha = 1

    and returns sum_i alpha_i g_i, a recombination of past MAP VALUES rather
    than a correction to the current iterate. Closed form via the normal
    equations and one Lagrange multiplier:

        alpha = (F^T F + reg I)^-1 1  /  1^T (F^T F + reg I)^-1 1

    The constraint is what makes this stable when the residual columns go
    nearly collinear (late snap-through), and it also makes a fixed point
    exact: with every g_i == x_i, any admissible alpha returns x*.

    Parameters
    ----------
    depth : int    history window. depth 1 is plain Picard.
    reg   : float  Tikhonov parameter, scaled by trace(F^T F)/m as in
            AndersonAccelerator.
    bounds, safeguard_factor : see _AccelBase.
    """

    def _extrapolate(self):
        F = np.stack(self._F, axis=1)          # residuals, not differences
        G = np.stack(self._G, axis=1)          # the map values to recombine
        A = F.T @ F
        m = A.shape[0]
        scale = float(np.trace(A)) / m if m else 0.0
        if not np.isfinite(scale) or scale <= 0.0:
            return None
        A = A + self.reg * scale * np.eye(m)
        ones = np.ones(m)
        try:
            y = np.linalg.solve(A, ones)
        except np.linalg.LinAlgError:
            return None
        denom = float(ones @ y)
        if not np.isfinite(denom) or abs(denom) < 1e-300:
            return None
        alpha = y / denom                       # sums to 1 by construction
        return G @ alpha


_ACCELERATORS = {
    "anderson": AndersonAccelerator,
    "ngmres":   NGMRESAccelerator,
}


def build_accelerator(kind: str | None, **kwargs):
    """
    Build an accelerator by name, or None for plain Picard -- the analogue of
    materialmodels.factory.build_material, so a YAML config can pick the
    accelerator the same way it picks a material.

        kind = None | "none" | "picard"   -> None (plain staggering)
        kind = "anderson"                 -> AndersonAccelerator(**kwargs)
        kind = "ngmres"                   -> NGMRESAccelerator(**kwargs)
    """
    if kind is None or str(kind).lower() in ("none", "picard", ""):
        return None
    key = str(kind).lower()
    if key not in _ACCELERATORS:
        raise ValueError(
            f"unknown staggered accelerator {kind!r}, expected one of "
            f"{list(_ACCELERATORS)} or 'none'"
        )
    return _ACCELERATORS[key](**kwargs)
