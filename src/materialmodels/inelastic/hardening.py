"""
Isotropic hardening laws sigma_y(alpha) for the plasticity models.

Every return mapping here reduces to one scalar equation for the multiplier:

    A - B dlam = sigma_y(alpha_prev + dlam)                              [*]

    J2 radial return   A = q_trial                    B = 3 mu
    DP cone            A = q_trial + 3 a_f p_trial    B = 3 mu + 9 K a_f a_g
    DP apex            A = 3 a_f p_trial              B = 9 K a_f a_g

so a new law needs no model change. With B > 0 and sigma_y non-decreasing,
[*] has exactly one root.

Autodiff: everything runs inside jacfwd and vmap. Discarded branches must
still have finite derivatives, hence the floored denominators. Segment
selection is discrete (zero derivative): the tangent is exact within a
segment and kinked at breakpoints.
"""

import jax.numpy as jnp
import numpy as np

_EPS_TINY = 1e-12

# Fixed count: a batched while_loop runs until the worst voxel converges anyway.
_NEWTON_ITERS = 8


class IsotropicHardening:
    """
    Base hardening law; subclasses get a working ``solve_return``.

    Subclasses define ``sigma_y(alpha)``, ``dsigma_y(alpha)`` (traced),
    ``sigma_y0`` and ``sigma_y_max`` (static; the ceiling, inf if unbounded).
    They may override ``solve_return`` (closed form) and ``affine_segments``.
    """

    sigma_y0: float
    # A finite ceiling limits the hydrostatic tension DruckerPrager can reach
    # when there is no dilatancy.
    sigma_y_max: float = float("inf")

    def sigma_y(self, alpha):
        raise NotImplementedError

    def dsigma_y(self, alpha):
        raise NotImplementedError

    def affine_segments(self):
        """
        ``(alpha_lo, sigma_lo, H, alpha_hi)`` of the intervals where sigma_y is
        affine, or None. DruckerPrager with ``a_tip > 0`` requires them.
        """
        return None

    def solve_return(self, A, B, alpha_prev):
        """
        Solve [*] for dlam >= 0 by bracketed Newton.

        R is strictly decreasing, so [0, max(R(0), 0)/B] always brackets the
        root; an elastic state gives an empty bracket and dlam = 0.
        """
        B_safe = jnp.maximum(B, _EPS_TINY)

        def residual(dlam):
            return A - B * dlam - self.sigma_y(alpha_prev + dlam)

        r0 = A - self.sigma_y(alpha_prev)
        hi = jnp.maximum(r0, 0.0) / B_safe
        lo = jnp.zeros_like(hi)
        dlam = 0.5 * (lo + hi)

        for _ in range(_NEWTON_ITERS):
            r = residual(dlam)
            dr = -B - self.dsigma_y(alpha_prev + dlam)
            lo = jnp.where(r > 0.0, dlam, lo)
            hi = jnp.where(r <= 0.0, dlam, hi)
            slope = jnp.where(jnp.abs(dr) > _EPS_TINY, dr, -1.0)
            dlam_newton = dlam - r / slope
            # Closed bracket test: once converged the iterate equals a bracket
            # end, and an open test would bisect away from the root.
            take = (dlam_newton >= lo) & (dlam_newton <= hi)
            dlam = jnp.where(take, dlam_newton, 0.5 * (lo + hi))

        return jnp.maximum(dlam, 0.0)


class LinearHardening(IsotropicHardening):
    """
    ``sigma_y = sigma_y0 + H alpha``; the default.

    ``solve_return`` is bit-for-bit the closed form the models used to inline.

    Parameters
    ----------
    sigma_y0 : initial yield stress, > 0
    H        : hardening modulus, >= 0 (0 = perfectly plastic)
    """

    def __init__(self, sigma_y0: float, H: float):
        self.sigma_y0 = float(sigma_y0)
        self.H = float(H)
        self.sigma_y_max = float("inf") if self.H > 0.0 else self.sigma_y0
        if self.sigma_y0 <= 0.0:
            raise ValueError(
                f"LinearHardening: sigma_y0={self.sigma_y0} must be > 0 -- a "
                "material that yields at zero stress has no elastic range"
            )
        if self.H < 0.0:
            raise ValueError(
                f"LinearHardening: H={self.H} < 0 -- softening is not supported "
                "(it makes the return mapping's residual non-monotone, and the "
                "global problem localizes with no regularization in this solver)"
            )

    def sigma_y(self, alpha):
        return self.sigma_y0 + self.H * alpha

    def dsigma_y(self, alpha):
        return jnp.full_like(jnp.asarray(alpha, dtype=float), self.H)

    def solve_return(self, A, B, alpha_prev):
        return jnp.maximum(A - self.sigma_y(alpha_prev), 0.0) / jnp.maximum(
            B + self.H, _EPS_TINY
        )

    def affine_segments(self):
        return (
            jnp.array([0.0]),
            jnp.array([self.sigma_y0]),
            jnp.array([self.H]),
            jnp.array([jnp.inf]),
        )

    def __repr__(self) -> str:
        return f"linear(sigma_y0={self.sigma_y0:.4g}, H={self.H:.4g})"


class PiecewiseLinearHardening(IsotropicHardening):
    """
    Tabulated sigma_y(alpha), linear between rows, perfectly plastic past the
    last one (Abaqus ``*PLASTIC`` convention).

    Each segment inverts [*] in closed form and exactly one is self-consistent,
    so no iteration is needed.

    Cover the strains you will reach: on the final plateau the global Newton
    converges much worse (as for H = 0). Compare the driver's max accumulated
    plastic strain with the last row.

    Parameters
    ----------
    table : ``(plastic_strain, yield_stress)`` rows, plastic strain first;
            at least two, the first at plastic strain 0
    """

    def __init__(self, table):
        tab = np.asarray([[float(a), float(s)] for a, s in table], dtype=float)
        if tab.ndim != 2 or tab.shape[1] != 2:
            raise ValueError(
                f"PiecewiseLinearHardening: table must be (plastic_strain, "
                f"yield_stress) pairs, got shape {tab.shape}"
            )
        if tab.shape[0] < 2:
            raise ValueError(
                f"PiecewiseLinearHardening: table has {tab.shape[0]} row(s), "
                "needs at least 2 -- a single row is perfect plasticity, which "
                "is LinearHardening(sigma_y0, H=0)"
            )
        a, s = tab[:, 0], tab[:, 1]
        # Swapped columns are otherwise silent; a real yield stress is never 0.
        if a[0] != 0.0:
            raise ValueError(
                f"PiecewiseLinearHardening: first row is at plastic strain "
                f"{a[0]}, must be 0.0. Columns are (plastic_strain, "
                "yield_stress) -- plastic strain FIRST; a table entered as "
                "(yield_stress, plastic_strain) fails exactly here"
            )
        if np.any(np.diff(a) <= 0.0):
            raise ValueError(
                "PiecewiseLinearHardening: plastic strain column must be "
                f"strictly increasing, got {a.tolist()}"
            )
        if np.any(s <= 0.0):
            raise ValueError(
                f"PiecewiseLinearHardening: yield stresses must be > 0, got {s.tolist()}"
            )
        if np.any(np.diff(s) < 0.0):
            raise ValueError(
                "PiecewiseLinearHardening: yield stress column must be "
                f"non-decreasing, got {s.tolist()} -- softening is not supported "
                "(it makes the return mapping's residual non-monotone, so the "
                "segment containing the root is no longer unique, and the global "
                "problem localizes with no regularization in this solver)"
            )

        H = np.zeros_like(s)
        H[:-1] = np.diff(s) / np.diff(a)
        alpha_hi = np.concatenate([a[1:], [np.inf]])

        self.table = tab
        self.sigma_y0 = float(s[0])
        self.sigma_y_max = float(s[-1])
        self._alpha_lo = jnp.asarray(a)
        self._sigma_lo = jnp.asarray(s)
        self._H = jnp.asarray(H)
        self._alpha_hi = jnp.asarray(alpha_hi)
        self.n_segments = int(tab.shape[0])

    def _segment_index(self, alpha):
        """Index of the segment containing ``alpha``, clamped."""
        idx = jnp.searchsorted(self._alpha_lo, alpha, side="right") - 1
        return jnp.clip(idx, 0, self.n_segments - 1)

    def sigma_y(self, alpha):
        j = self._segment_index(alpha)
        return self._sigma_lo[j] + self._H[j] * (alpha - self._alpha_lo[j])

    def dsigma_y(self, alpha):
        return self._H[self._segment_index(alpha)]

    def solve_return(self, A, B, alpha_prev):
        denom = jnp.maximum(B + self._H, _EPS_TINY)
        dlam = (A - self._sigma_lo - self._H * (alpha_prev - self._alpha_lo)) / denom
        alpha_new = alpha_prev + dlam

        # Closed bounds: a root exactly on a breakpoint is valid in both
        # neighbors; an open test can reject every segment there.
        valid = (
            (alpha_new >= jnp.maximum(self._alpha_lo, alpha_prev))
            & (alpha_new <= self._alpha_hi)
        )
        j = jnp.argmax(valid)

        # Elastic states have no valid segment; select the elastic answer.
        f_trial = A - self.sigma_y(alpha_prev)
        return jnp.where(f_trial > 0.0, jnp.maximum(dlam[j], 0.0), 0.0)

    def affine_segments(self):
        return (self._alpha_lo, self._sigma_lo, self._H, self._alpha_hi)

    def __repr__(self) -> str:
        pts = ", ".join(f"({a:.4g}, {s:.4g})" for a, s in self.table)
        return f"piecewise_linear[{self.n_segments} pts: {pts}, then plateau]"


_HARDENING = {
    "linear":           LinearHardening,
    "piecewise_linear": PiecewiseLinearHardening,
}


def build_hardening(cfg: dict) -> IsotropicHardening:
    """
    Hardening law from a config mapping; ``law`` is required (no silent default).

        hardening:
          law: piecewise_linear
          table:                # (plastic_strain, yield_stress)
            - [0.000, 56.1]
            - [0.020, 70.0]

    ``law: linear`` takes ``sigma_y0`` and ``H``.
    """
    if not isinstance(cfg, dict):
        raise ValueError(
            f"hardening: expected a mapping with a 'law' key, got {type(cfg).__name__}. "
            "A bare table is not accepted -- write 'law: piecewise_linear' and "
            "put the rows under 'table:'"
        )
    cfg = dict(cfg)
    law = cfg.pop("law", None)
    if law not in _HARDENING:
        raise ValueError(
            f"hardening: unknown law {law!r}, expected one of {list(_HARDENING)}"
        )

    if law == "piecewise_linear":
        table = cfg.pop("table", None)
        if table is None:
            raise ValueError("hardening: law 'piecewise_linear' requires a 'table' key")
        if cfg:
            raise ValueError(
                f"hardening: unexpected key(s) {list(cfg)} for law 'piecewise_linear'"
            )
        return PiecewiseLinearHardening(table)

    # YAML reads "1.0e3" as a string.
    try:
        return LinearHardening(**{k: float(v) for k, v in cfg.items()})
    except TypeError as exc:
        raise ValueError(f"hardening: law 'linear' -- {exc}") from None


def resolve_hardening(hardening, sigma_y0, H, who: str = "") -> IsotropicHardening:
    """
    One IsotropicHardening from a model's ``hardening`` or ``sigma_y0`` + ``H`` arguments.

    ``hardening`` may be an instance or a config mapping; giving it together
    with ``H``, or with a different ``sigma_y0``, raises.
    """
    tag = f"{who}: " if who else ""

    if hardening is None:
        if sigma_y0 is None or H is None:
            raise ValueError(
                f"{tag}needs a hardening law -- either sigma_y0=... and H=... for "
                "linear hardening, or hardening={'law': ...} (see "
                "materialmodels.inelastic.hardening.build_hardening)"
            )
        return LinearHardening(sigma_y0, H)

    if H is not None:
        raise ValueError(
            f"{tag}H and hardening are mutually exclusive -- H=... IS a hardening "
            "law (linear). Drop H, or drop hardening"
        )

    law = hardening if isinstance(hardening, IsotropicHardening) else build_hardening(hardening)

    if sigma_y0 is not None and float(sigma_y0) != law.sigma_y0:
        raise ValueError(
            f"{tag}sigma_y0={sigma_y0} contradicts the hardening law's virgin yield "
            f"stress {law.sigma_y0} -- they are the same quantity. Drop sigma_y0, or "
            "make the two agree"
        )
    return law
