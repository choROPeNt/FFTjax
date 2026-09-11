"""
Isotropic hardening laws sigma_y(alpha) for the plasticity models in
materialmodels.inelastic -- a pluggable extension point rather than a scalar
H hardcoded on each model.

Why a separate module. Every return mapping in this package reduces to the
SAME scalar equation for the plastic multiplier,

    A - B*dlam = sigma_y(alpha_prev + dlam)                              [*]

with only (A, B) differing per model and per branch:

    J2 radial return   A = q_trial                    B = 3*mu
    DP cone            A = q_trial + 3*a_f*p_trial    B = 3*mu + 9*K*a_f*a_g
    DP apex            A = 3*a_f*p_trial              B = 9*K*a_f*a_g

so [*] is the whole interface between a plasticity model and its hardening
law, and a new law needs no change to any model. That is the point of the
split: the models own the yield surface, this module owns sigma_y(alpha).

[*] always has exactly one root. B > 0 and sigma_y non-decreasing (enforced
below) make the residual R(dlam) = A - B*dlam - sigma_y(alpha_prev + dlam)
strictly decreasing, so a law that cannot invert [*] in closed form can
always fall back on the safeguarded Newton in IsotropicHardening.solve_return
-- which is exactly what a future smooth law (Voce, power-law) would inherit.
The two laws shipped here both override it with their exact closed forms.

Autodiff
--------
Everything here runs inside jax.jacfwd (the models derive their consistent
tangent by autodiff, see plasticity_j2.stress_and_tangent) and inside vmap
over voxels, so the rules the plasticity files already document apply
verbatim:

- every branch is evaluated and the unselected ones discarded by jnp.where /
  gather, and a discarded branch's derivative still has to be FINITE -- a NaN
  there poisons the kept branch's tangent. Hence the floored denominators
  below, which are load-bearing for the tangent even at states whose value
  never uses them.
- the segment selection in PiecewiseLinearHardening is discrete, so its
  derivative is zero. That is correct, not a lost term: d sigma_y/d alpha is
  genuinely piecewise constant. The tangent is exact WITHIN a segment and
  kinked at the breakpoints, which is what a piecewise-linear curve means. An
  increment that straddles a breakpoint may cost the global Newton an extra
  iteration; it does not cost it correctness.
"""

import jax.numpy as jnp
import numpy as np

_EPS_TINY = 1e-12

# Safeguarded-Newton iterations for the GENERIC solve_return fallback (no
# shipped law uses it -- both invert [*] exactly). Same fixed-count rationale
# as plasticity_drucker_prager._NEWTON_ITERS: a batched while_loop runs until
# every voxel in the grid has converged, so vmap pays the worst voxel anyway
# and a fixed count is cheaper and predictable.
_NEWTON_ITERS = 8


class IsotropicHardening:
    """
    Base class: a law supplies sigma_y(alpha) and its slope, and inherits a
    working solve_return.

    Subclasses MUST define
        sigma_y(alpha)   -- traced, current yield stress
        dsigma_y(alpha)  -- traced, d sigma_y / d alpha
        sigma_y0         -- static float, the virgin (alpha = 0) yield stress

    and MAY override
        solve_return(A, B, alpha_prev)  -- if [*] inverts in closed form
        affine_segments()               -- if the law is piecewise affine
    """

    sigma_y0: float

    def sigma_y(self, alpha):
        raise NotImplementedError

    def dsigma_y(self, alpha):
        raise NotImplementedError

    def affine_segments(self):
        """
        Static ``(alpha_lo, sigma_lo, H, alpha_hi)`` arrays describing
        intervals on which sigma_y is exactly affine, or ``None`` if the law
        is not piecewise affine.

        Only DruckerPrager's tip-rounded branch needs this: it runs its own
        Newton in q and needs the per-segment affine coefficients to set it
        up (see _stress_hyperbolic). A law returning None simply cannot be
        combined with ``a_tip > 0`` until that branch grows a general nested
        solve -- rejected at construction, not silently.
        """
        return None

    def solve_return(self, A, B, alpha_prev):
        """
        Solve [*] for dlam >= 0 -- generic safeguarded Newton, for any law
        that does not override it.

        The bracket is free: R is strictly decreasing, R(0) = A -
        sigma_y(alpha_prev), and since sigma_y is non-decreasing,
        R(dlam) <= R(0) - B*dlam, so dlam_hi = max(R(0), 0)/B always has
        R(dlam_hi) <= 0. An elastic state gives dlam_hi = 0 and every
        iteration below leaves dlam at exactly 0, which is the required
        elastic identity -- no separate branch.
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
            # R decreasing: r > 0 means the root lies ABOVE dlam.
            lo = jnp.where(r > 0.0, dlam, lo)
            hi = jnp.where(r <= 0.0, dlam, hi)
            slope = jnp.where(jnp.abs(dr) > _EPS_TINY, dr, -1.0)
            dlam_newton = dlam - r / slope
            # Closed bracket test, for the reason _solve_q_hyperbolic
            # documents: once converged the step is exactly 0 and the next
            # iterate EQUALS a bracket end, which an open test would reject,
            # bisecting away from the converged root.
            take = (dlam_newton >= lo) & (dlam_newton <= hi)
            dlam = jnp.where(take, dlam_newton, 0.5 * (lo + hi))

        return jnp.maximum(dlam, 0.0)


class LinearHardening(IsotropicHardening):
    """
    sigma_y(alpha) = sigma_y0 + H*alpha -- what both plasticity models did
    before this module existed, and still their default.

    solve_return below is literally the closed form those models inlined,
    with the same grouping of the same floats, so routing them through this
    class is bit-for-bit a no-op rather than a numerically-equivalent
    rewrite.

    Parameters
    ----------
    sigma_y0 : float  initial (virgin) yield stress, > 0
    H        : float  constant hardening modulus, >= 0 (0 = perfectly plastic)
    """

    def __init__(self, sigma_y0: float, H: float):
        self.sigma_y0 = float(sigma_y0)
        self.H = float(H)
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
        # max(f_trial, 0) makes dlam exactly 0 below yield, collapsing the
        # caller's formulas to the elastic identity -- the same trick, for
        # the same reason, as the code this replaces.
        return jnp.maximum(A - self.sigma_y(alpha_prev), 0.0) / jnp.maximum(
            B + self.H, _EPS_TINY
        )

    def affine_segments(self):
        # One segment covering everything -- affine, and NOT plateauing.
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
    Tabulated sigma_y(alpha), linearly interpolated between the rows and
    PERFECTLY PLASTIC beyond the last one (Abaqus' convention for a
    ``*PLASTIC`` table).

    Parameters
    ----------
    table : sequence of ``(plastic_strain, yield_stress)`` rows, plastic
            strain first -- the same two columns an Abaqus material card
            carries, in the same meaning. At least two rows; the first must
            sit at plastic strain 0.

    Why this is still closed-form. Inside segment j, sigma_y = sig_j +
    H_j*(alpha - alpha_j) is affine, so [*] inverts directly:

        dlam_j = (A - sig_j - H_j*(alpha_prev - alpha_j)) / (B + H_j)

    and because the residual is strictly decreasing, EXACTLY ONE segment has
    alpha_prev + dlam_j inside its own interval. So the solve is: evaluate
    every segment, keep the self-consistent one. No Newton iteration, no
    convergence tolerance -- unlike a smooth law, which is why a tabulated
    curve is the cheap way to leave linear hardening behind.

    Make the table cover the strains you will actually reach. Past the last
    row the plateau is PERFECTLY PLASTIC by construction, and perfect
    plasticity is markedly harder for the global Newton than any positive
    slope -- a load reversal that converges in 6 iterations while hardening
    can burn maxiter_nr once the voxels have run onto the plateau. That is
    not a property of this class: LinearHardening at H = 0 fails the same
    way, and earlier. The driver prints the max accumulated plastic strain
    per case, so compare it against this table's last row; if it is past it,
    extend the table rather than reaching for smaller steps.
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
        # The columns are (plastic strain, yield stress) and transposing them
        # is silent -- a stress where a strain belongs is still a number, and
        # the run just hardens absurdly. This check is what catches it: real
        # yield stresses are never 0, so a transposed table fails here.
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

        # n rows -> n segments: n-1 interpolated, plus a final half-infinite
        # PLATEAU (H = 0) so a voxel running past the calibrated data cannot
        # harden into extrapolated territory.
        H = np.zeros_like(s)
        H[:-1] = np.diff(s) / np.diff(a)
        alpha_hi = np.concatenate([a[1:], [np.inf]])

        self.table = tab
        self.sigma_y0 = float(s[0])
        self._alpha_lo = jnp.asarray(a)
        self._sigma_lo = jnp.asarray(s)
        self._H = jnp.asarray(H)
        self._alpha_hi = jnp.asarray(alpha_hi)
        self.n_segments = int(tab.shape[0])

    def _segment_index(self, alpha):
        """Index of the segment containing ``alpha`` (clamped at both ends)."""
        idx = jnp.searchsorted(self._alpha_lo, alpha, side="right") - 1
        return jnp.clip(idx, 0, self.n_segments - 1)

    def sigma_y(self, alpha):
        j = self._segment_index(alpha)
        return self._sigma_lo[j] + self._H[j] * (alpha - self._alpha_lo[j])

    def dsigma_y(self, alpha):
        return self._H[self._segment_index(alpha)]

    def solve_return(self, A, B, alpha_prev):
        # Every segment's candidate is computed and all but one discarded, so
        # the denominator is floored for the tangent's sake even though only
        # the degenerate B = 0 with H_j = 0 could actually vanish.
        denom = jnp.maximum(B + self._H, _EPS_TINY)
        dlam = (A - self._sigma_lo - self._H * (alpha_prev - self._alpha_lo)) / denom
        alpha_new = alpha_prev + dlam

        # Self-consistency: the candidate must land inside the segment that
        # produced it, and must not run backwards past alpha_prev. Bounds are
        # CLOSED at both ends -- a root sitting exactly on a breakpoint is
        # valid in both neighbours, which give the same dlam by continuity,
        # whereas an open test can reject every segment at that point.
        valid = (
            (alpha_new >= jnp.maximum(self._alpha_lo, alpha_prev))
            & (alpha_new <= self._alpha_hi)
        )
        j = jnp.argmax(valid)

        # Elastic states have no admissible segment at all (the root sits at
        # dlam < 0), so select the exact elastic answer rather than whatever
        # argmax fell back to. Same shape as the elastic guard in
        # DruckerPrager._stress_hyperbolic.
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
    Build a hardening law from a config dict, keyed by a "law" string -- the
    direct analogue of materialmodels.factory.build_material, so a YAML
    config can pick the hardening law the same way it picks the material
    model.

        hardening:
          law: linear
          sigma_y0: 56.1
          H: 1.0e3

        hardening:
          law: piecewise_linear
          table:                # (plastic_strain, yield_stress)
            - [0.000, 56.1]
            - [0.020, 70.0]
            - [0.100, 78.0]     # perfectly plastic beyond here

    ``law`` is REQUIRED and has no default, for the reason
    factory.build_material gives about fiber_dir: quietly assuming one here
    would be easy to get wrong without ever noticing.
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

    # YAML's float regex misses exponents without an explicit sign ("1.0e3"
    # loads as a str), so cast here -- same reason factory.build_material does.
    try:
        return LinearHardening(**{k: float(v) for k, v in cfg.items()})
    except TypeError as exc:
        raise ValueError(f"hardening: law 'linear' -- {exc}") from None


def resolve_hardening(hardening, sigma_y0, H, who: str = "") -> IsotropicHardening:
    """
    Normalize a plasticity model's ``(hardening | sigma_y0 + H)`` constructor
    pair into one IsotropicHardening.

    Both plasticity models call this, so the backwards-compatible
    ``sigma_y0=..., H=...`` spelling and the new ``hardening=...`` one behave
    identically in both, and the mutual-exclusion errors read the same.

    Accepts an IsotropicHardening instance, a config dict (built via
    build_hardening), or None with sigma_y0/H given.
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
