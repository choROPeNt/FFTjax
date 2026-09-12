"""
Drucker-Prager elastoplastic material model, pluggable isotropic hardening.

A pressure-sensitive generalization of materialmodels.inelastic.plasticity_j2
-- deliberately a DROP-IN REPLACEMENT for J2Plasticity: same statefulness
convention ((eps_p, alpha) per voxel, threaded by the CALLER across load
increments, never stored on this instance), same stress /
stress_and_tangent / stress_and_tangent_field surface, so
materialmodels.assembly.assemble_local_update and
problems.mechanics.solve_displacement_based_nonlinear pick it up unchanged
(assemble_local_update duck-types on stress_and_tangent_field, it does not
test for J2Plasticity). Only the constructor parameters differ: two extra
coefficients a_f (friction) and a_g (dilatancy) on top of J2's
(E, nu, sigma_y0, H) -- and both models take the same pluggable hardening
law (materialmodels.inelastic.hardening), linear by default. With a_f = a_g = 0 this model IS J2 -- every formula
below collapses to plasticity_j2's, verified in
test/test_materialmodels_drucker_prager.py.

stiffness_tensor() still returns the elastic-only stiffness (for a caller
that just wants an elastic reference, e.g. a fixed preconditioner); it is
not the elastoplastic tangent -- see stress_and_tangent, same as J2.

See notes/AUTODIFF_CONSTITUTIVE.md (Proposal A) for why the tangent is
derived by autodiff rather than hand-coded, and the two gotchas both
plasticity files work around: strain symmetrization (stress() does it
internally) and jnp.where/division NaN-poisoning a branch that should have
zero value but not zero gradient. This model adds a THIRD, its own: the
cone/apex return branch (see stress()), where the discarded branch must
still evaluate to a finite number, not just an unused one.

``a_tip`` (default 0.0 = the classical sharp cone, bit-for-bit the code
above) rounds the cone's tensile vertex into a hyperbola, which is what to
reach for when a confined, pressure-sensitive solve stalls: at the sharp
vertex the return sets s = 0 exactly, so the deviatoric block of the
consistent tangent collapses and the tangent goes SINGULAR -- not a bug,
just what a vertex implies. Measured on a confined uniaxial path
(E=1400, nu=0.39, sigma_y0=17, H=250, a_f=0.13, a_g=0.05) the smallest
eigenvalue of the tangent's symmetric part falls 104 -> 41 -> 0.00 over
eps_11 = 0.024 -> 0.030 and stays at 0 beyond; with a_tip = 0.1*sigma_y0 it
holds at 55 there and 13 at eps_11 = 0.060, while the stress response moves
by under 1%. See stress_hyperbolic below for the return mapping.
"""

import jax
import jax.numpy as jnp

from materialmodels.base import ConstitutiveModel
from materialmodels.inelastic.hardening import IsotropicHardening, resolve_hardening

_EPS_TINY = 1e-12

# Safeguarded-Newton iterations for the tip-rounded return (a_tip > 0). A
# fixed, unrolled count rather than a while_loop: jacfwd DOES differentiate
# through lax.while_loop, but a batched while_loop runs until EVERY voxel in
# the grid has converged, so vmap pays the worst voxel anyway and a fixed
# count is both cheaper and predictable. Swept over 9238 plastic states
# spanning E, H, a_f, a_g, a_tip/sigma_y0 in [0.01, 0.3] and the full
# pressure range, 6 iterations reach machine precision from the closed-form
# sharp-cone starting guess; 8 is that with margin.
_NEWTON_ITERS = 8


class DruckerPrager(ConstitutiveModel):
    """
    Rate-independent Drucker-Prager plasticity, linear isotropic hardening,
    optionally non-associated -- closed-form return mapping (cone + apex),
    no local Newton solve.

    Yield surface, in terms of the von Mises equivalent stress
    q = sqrt(3/2 s:s) and the mean stress p = tr(sigma)/3:

        f = q + 3*a_f*p - sigma_y(alpha)

    i.e. a cone in principal-stress space whose axis is the hydrostatic
    line: hydrostatic TENSION (p > 0) promotes yield, hydrostatic
    COMPRESSION suppresses it. Setting a_f = 0 recovers the
    pressure-insensitive von Mises cylinder exactly.

    Plastic flow follows the potential g = q + 3*a_g*p, so a_g controls
    plastic dilatancy (volume change under plastic flow) independently of
    a_f. a_g = a_f is associated flow (the default); a_g < a_f is the usual
    non-associated choice for geomaterials and polymers, whose real
    dilatancy is much smaller than their pressure sensitivity. a_g = 0 gives
    purely deviatoric (volume-preserving) plastic flow with a still
    pressure-sensitive yield surface.

    Parameters
    ----------
    E, nu    : float  isotropic elastic constants
    sigma_y0 : float  initial (virgin) yield stress -- the SHEAR-meridian
               intercept above, NOT a uniaxial tensile strength once a_f > 0
               (see the conversions below)
    H        : float  constant isotropic hardening modulus (0 = perfectly plastic)
    hardening : IsotropicHardening | dict | None  the hardening law, as an
               instance or a ``{"law": ...}`` config dict -- e.g. a tabulated
               ``piecewise_linear`` curve. Mutually exclusive with H; omit
               both and sigma_y0/H are required, which is the original
               spelling and still the default. Note a law that is not
               piecewise affine cannot currently be combined with a_tip > 0
               (see __init__).
    a_f      : float  friction coefficient, 0 <= a_f < 1 (0 = von Mises)
    a_g      : float | None  dilatancy coefficient, 0 <= a_g <= a_f;
               None (default) means a_g = a_f, i.e. associated flow
    name     : str    optional label
    k_res, Gc : as in LinearElasticIsotropic -- phase-field bookkeeping only,
                unrelated to the plastic return mapping itself.

    Choosing a_f and sigma_y0
    -------------------------
    The uniaxial yield stresses implied by the surface above are

        sigma_t = sigma_y0 / (1 + a_f)      (tension)
        sigma_c = sigma_y0 / (1 - a_f)      (compression)

    so from a measured tension/compression asymmetry m = sigma_c/sigma_t
    (~1.3 for a typical epoxy, >2 for concrete):

        a_f      = (m - 1) / (m + 1)
        sigma_y0 = sigma_t * (1 + a_f)

    a_f < 1 is required for a finite compressive strength -- a_f >= 1 makes
    the material unyieldable in compression, and is rejected in __init__.

    From a Mohr-Coulomb friction angle phi, cohesion c and dilatancy angle
    psi instead, matching the OUTER (compressive-meridian) cone:

        a_f      = 2 sin(phi) / (3 - sin(phi))
        a_g      = 2 sin(psi) / (3 - sin(psi))
        sigma_y0 = 6 c cos(phi) / (3 - sin(phi))

    (the inner/extension-meridian match replaces every 3 - sin by 3 + sin).
    """

    def __init__(self, E: float, nu: float, sigma_y0: float | None = None,
                 H: float | None = None, a_f: float | None = None,
                 a_g: float | None = None, a_tip: float = 0.0,
                 hardening: "IsotropicHardening | dict | None" = None,
                 name: str = "", k_res: float = 1e-6, Gc: float | None = None):
        self.E = float(E)
        self.nu = float(nu)
        self.hardening = resolve_hardening(hardening, sigma_y0, H, f"DruckerPrager {name!r}")
        # The virgin yield stress is a property of every law, not just the
        # linear one, and the a_tip check below needs it.
        self.sigma_y0 = self.hardening.sigma_y0
        # a_f has no meaningful default -- 0.0 would silently turn this into
        # von Mises -- so it stays required, just not positionally.
        if a_f is None:
            raise ValueError(
                f"DruckerPrager {name!r}: a_f is required. It is the friction "
                "coefficient; a_f = 0 is pressure-INsensitive von Mises, which "
                "is J2Plasticity, so there is no safe default to assume here"
            )
        self.a_f = float(a_f)
        self.a_g = float(a_f) if a_g is None else float(a_g)
        self.a_tip = float(a_tip)
        self.name = name
        self.k_res = float(k_res)
        self.Gc = float(Gc) if Gc is not None else None
        self.lam = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        self.mu = self.E / (2.0 * (1.0 + self.nu))
        self.K = self.lam + 2.0 * self.mu / 3.0

        # a_f, a_g are static Python floats, never traced, so these are real
        # errors raised at construction -- not something the return mapping
        # has to defend against per voxel inside a jit.
        if not 0.0 <= self.a_f < 1.0:
            raise ValueError(
                f"DruckerPrager {self.name!r}: a_f={self.a_f} outside [0, 1) -- "
                "a_f < 0 inverts the pressure sensitivity (compression would "
                "promote yield), and a_f >= 1 gives an infinite compressive "
                "yield stress sigma_y0/(1 - a_f)"
            )
        if self.a_g < 0.0:
            raise ValueError(
                f"DruckerPrager {self.name!r}: a_g={self.a_g} < 0 -- negative "
                "dilatancy (plastic compaction under yield) is not supported"
            )
        # a_tip is likewise a static Python float -- validated once, here.
        if self.a_tip < 0.0:
            raise ValueError(
                f"DruckerPrager {self.name!r}: a_tip={self.a_tip} < 0 -- the tip "
                "rounding is a stress-dimensioned radius, a_tip = 0 being the "
                "classical sharp cone"
            )
        if self.a_tip > 0.0 and self.a_f == 0.0:
            raise ValueError(
                f"DruckerPrager {self.name!r}: a_tip={self.a_tip} with a_f=0 -- a "
                "pressure-INsensitive surface is a cylinder with no vertex to "
                "round, and the hyperbola would only shrink the yield stress to "
                f"sqrt(sigma_y^2 - a_tip^2). Drop a_tip, or set a_f > 0."
            )
        # The tip-rounded branch solves for q per AFFINE segment of
        # sigma_y (see _stress_hyperbolic). A law that is not piecewise
        # affine would need a genuine nested solve there, which is not built
        # -- so say so here rather than produce silently wrong stresses.
        if self.a_tip > 0.0 and self.hardening.affine_segments() is None:
            raise ValueError(
                f"DruckerPrager {self.name!r}: a_tip={self.a_tip} with hardening "
                f"law {self.hardening!r}, which is not piecewise affine -- the "
                "tip-rounded return solves one affine segment at a time and has "
                "no general nested solve yet. Use a linear or piecewise_linear "
                "law, or drop a_tip"
            )
        if self.a_tip >= self.sigma_y0:
            raise ValueError(
                f"DruckerPrager {self.name!r}: a_tip={self.a_tip} >= "
                f"sigma_y0={self.sigma_y0} -- the rounded surface meets the "
                "hydrostatic axis at p_tip = (sigma_y - a_tip)/(3*a_f), so "
                "a_tip >= sigma_y leaves zero (or negative) hydrostatic tensile "
                "strength. Typical values are a few percent of sigma_y0."
            )

    def stiffness_tensor(self) -> jnp.ndarray:
        """(3, 3, 3, 3) elastic stiffness -- NOT the elastoplastic tangent."""
        d = jnp.eye(3)
        return (self.lam * jnp.einsum('ij,kl->ijkl', d, d)
                + self.mu * (jnp.einsum('ik,jl->ijkl', d, d)
                             + jnp.einsum('il,jk->ijkl', d, d)))

    def stress(
        self,
        eps: jnp.ndarray,
        eps_p_prev: jnp.ndarray,
        alpha_prev: jnp.ndarray,
    ) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]:
        """
        One-voxel Drucker-Prager return mapping: sharp cone (``a_tip = 0``,
        the default) or tip-rounded hyperbola (``a_tip > 0``).

        ``a_tip`` is a static Python float, never traced, so this dispatch
        resolves once at trace time -- jit sees only the selected branch,
        and ``a_tip = 0`` runs exactly the code it always did, at no added
        cost and bit-for-bit unchanged.

        Parameters
        ----------
        eps        : (3,3)  total strain -- symmetrized internally
        eps_p_prev : (3,3)  plastic strain from the previous converged state
        alpha_prev : ()     accumulated equivalent plastic strain (scalar)

        Returns
        -------
        sigma          : (3,3)
        (eps_p, alpha) : updated state, same shapes as the _prev inputs
        """
        if self.a_tip > 0.0:
            return self._stress_hyperbolic(eps, eps_p_prev, alpha_prev)
        return self._stress_sharp(eps, eps_p_prev, alpha_prev)

    def _stress_sharp(
        self,
        eps: jnp.ndarray,
        eps_p_prev: jnp.ndarray,
        alpha_prev: jnp.ndarray,
    ) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]:
        """
        One-voxel closed-form Drucker-Prager return mapping.

        Two returns, selected per voxel -- both closed-form for the linear
        and piecewise-linear hardening laws:

        SMOOTH CONE -- the J2 radial return plus a volumetric correction.
        As in plasticity_j2, the elastic branch is not an explicit
        jnp.where: max(f_trial, 0) makes the plastic multiplier dlam exactly
        zero below yield, which collapses every cone formula to the elastic
        identity (sigma = sigma_trial, state unchanged).

        APEX -- the pressure-sensitive surface is a CONE, not a cylinder, so
        a trial state deep in hydrostatic tension can sit "past the vertex",
        where the smooth return would overshoot into q < 0, i.e. flip the
        deviator's direction rather than shrink it. J2 has no such region
        (its own q stays positive for every trial state), which is why
        plasticity_j2 needs no second branch. Here the vertex return sets
        s = 0 exactly and solves the same consistency condition for the
        hardening increment alone -- equation [*] at A = 3*a_f*p_trial,
        B = 9*K*a_f*a_g -- parametrized by dalpha rather than by the
        volumetric plastic strain so that a_g = 0 (no dilatancy) stays finite
        instead of dividing by zero. Its denominator vanishes only for the
        genuinely ill-posed
        combination H = 0 with a_f*a_g = 0 -- perfectly plastic, no
        dilatancy, yet pressure-sensitive: no return to the vertex exists
        at all, since purely deviatoric flow cannot change p. That is
        floored rather than raised on, so a config that never actually
        reaches the vertex still runs.

        Both branches are evaluated at every voxel and one is discarded by
        jnp.where. jnp.where propagates the DISCARDED branch's derivative as
        a multiply-by-zero, so a non-finite value there poisons the kept
        branch's tangent with NaN -- hence the floored apex denominator and
        the q_safe guard below are load-bearing for the tangent even at
        voxels whose value never uses them.

        Parameters
        ----------
        eps        : (3,3)  total strain -- symmetrized internally, not
                     required to be pre-symmetrized by the caller (needed
                     for a correct autodiff tangent, see stress_and_tangent)
        eps_p_prev : (3,3)  plastic strain from the previous converged state
        alpha_prev : ()     accumulated equivalent plastic strain (scalar)

        Returns
        -------
        sigma          : (3,3)
        (eps_p, alpha) : updated state, same shapes as the _prev inputs
        """
        eps = 0.5 * (eps + eps.T)
        I3 = jnp.eye(3)

        eps_e_trial = eps - eps_p_prev
        sigma_trial = (self.lam * jnp.trace(eps_e_trial) * I3
                       + 2.0 * self.mu * eps_e_trial)
        p_trial = jnp.trace(sigma_trial) / 3.0
        s_trial = sigma_trial - p_trial * I3
        # sqrt(x) has an infinite/NaN gradient at x=0 -- q_trial=||s_trial||
        # hits exactly that point at a virgin/unstressed state (eps=eps_p=0),
        # a real case (a Newton solve's zero initial guess) not a corner
        # case. Clamping the radicand away from 0 makes the gradient exactly
        # 0 there instead of NaN. Same guard, same reason, as plasticity_j2.
        s_sq = jnp.sum(s_trial ** 2)
        q_trial = jnp.sqrt(1.5 * jnp.maximum(s_sq, _EPS_TINY))
        q_safe = jnp.where(q_trial > _EPS_TINY, q_trial, 1.0)

        # ── smooth-cone return ──────────────────────────────────────────
        # Consistency along the cone: the trial q is relaxed by 3*mu*dlam
        # and the trial p by 3*K*a_g*dlam, so the cone is exactly equation
        # [*] of materialmodels.inelastic.hardening at
        #     A = q_trial + 3*a_f*p_trial,  B = 3*mu + 9*K*a_f*a_g
        # and the hardening law inverts it -- in closed form for the linear
        # and piecewise-linear laws, which is what keeps this branch free of
        # any local iteration.
        B_cone = 3.0 * self.mu + 9.0 * self.K * self.a_f * self.a_g
        dlam = self.hardening.solve_return(
            q_trial + 3.0 * self.a_f * p_trial, B_cone, alpha_prev)
        # The return is radial in the deviatoric plane, so the flow
        # direction 1.5*s/q is the same evaluated at the trial or at the
        # updated state -- exactly J2's `1.5 * s_trial / q_safe`, with the
        # a_g*I3 dilatancy term as the only addition.
        n_dev = 1.5 * s_trial / q_safe
        s_cone = s_trial * (1.0 - 3.0 * self.mu * dlam / q_safe)
        sigma_cone = s_cone + (p_trial - 3.0 * self.K * self.a_g * dlam) * I3
        eps_p_cone = eps_p_prev + dlam * (n_dev + self.a_g * I3)
        alpha_cone = alpha_prev + dlam

        # ── apex (vertex) return ────────────────────────────────────────
        # Reached iff the cone return would drive q negative. Same equation
        # [*] as the cone, at A = 3*a_f*p_trial (the vertex has q = 0) and
        # B = 9*K*a_f*a_g. Outside the apex region the value below is
        # discarded; it only has to stay FINITE for the tangent's sake, which
        # is why solve_return floors its denominator -- the one combination
        # that vanishes is H = 0 with a_f*a_g = 0.
        B_apex = 9.0 * self.K * self.a_f * self.a_g
        dalpha_apex = self.hardening.solve_return(
            3.0 * self.a_f * p_trial, B_apex, alpha_prev)
        # s = 0: the whole trial deviator is absorbed as plastic strain,
        # s_trial/(2*mu) being its elastic-strain equivalent.
        sigma_apex = (p_trial - 3.0 * self.K * self.a_g * dalpha_apex) * I3
        eps_p_apex = (eps_p_prev + s_trial / (2.0 * self.mu)
                      + self.a_g * dalpha_apex * I3)
        alpha_apex = alpha_prev + dalpha_apex

        use_apex = (q_trial - 3.0 * self.mu * dlam) < 0.0
        sigma = jnp.where(use_apex, sigma_apex, sigma_cone)
        eps_p = jnp.where(use_apex, eps_p_apex, eps_p_cone)
        alpha = jnp.where(use_apex, alpha_apex, alpha_cone)

        return sigma, (eps_p, alpha)

    def _solve_q_hyperbolic(self, q_trial, xi_trial, xi_0, c, k):
        """
        Safeguarded (bracketed) Newton for the updated ``q``, the ONE scalar
        unknown the tip-rounded return reduces to. See _stress_hyperbolic
        for where the residual comes from.

            Psi(q) = q*(xi*(1 + k) - k*xi_0) - q_trial*xi,  xi = sqrt(q^2 + a^2)

        written multiplicatively (no division by xi) purely for scaling --
        the divided form is the same root but loses three orders of
        magnitude of residual at the same iteration count.

        Elementwise in (xi_0, c, k), so _stress_hyperbolic evaluates every
        affine segment of the hardening law in one call; q_trial and xi_trial
        stay scalar and broadcast.

        The bracket [0, q_trial] is GUARANTEED to contain exactly one root
        whenever the state is plastic: Psi(0) = -q_trial*a_tip < 0 and
        Psi(q_trial) = q_trial*k*f_trial > 0. Verified by sampling over 9238
        plastic states (bracket valid and root unique in all of them), which
        is why bisection is a sound fallback here rather than a guess.

        Newton alone is NOT enough: Psi is non-monotone in ~30% of those
        states (its slope can turn negative near q = 0 when the trial
        pressure sits far past the tip), so an unguarded Newton step can
        leave the bracket. Hence the bisection fallback -- taken on 0.4% of
        steps in the sweep, but load-bearing on those.

        Note the bracket test below is CLOSED (>= lo, <= hi), not open. Once
        Newton converges, Psi is exactly 0, the step is exactly 0 and the
        next iterate equals the bracket end just assigned -- an open test
        rejects it and bisects AWAY from the converged root, silently
        capping accuracy at ~1e-1 relative. Found the hard way.
        """
        a = self.a_tip

        def psi_and_slope(q):
            # xi >= a_tip > 0 always, so neither the sqrt nor either
            # division below can be singular -- unlike the sharp cone, whose
            # q_safe guard exists exactly because its xi IS q.
            xi = jnp.sqrt(q * q + a * a)
            psi = q * (xi * (1.0 + k) - k * xi_0) - q_trial * xi
            dpsi = (1.0 + k) * (xi + q * q / xi) - k * xi_0 - q_trial * q / xi
            return psi, dpsi

        # Start from the sharp-cone closed form -- exact in the a_tip -> 0
        # limit, so the rounding is only ever a small correction to solve for.
        dlam0 = jnp.maximum(xi_trial - xi_0, 0.0) / (3.0 * self.mu + c)
        q = jnp.clip(q_trial - 3.0 * self.mu * dlam0, 0.0, q_trial)
        lo = jnp.zeros_like(q_trial)
        hi = q_trial

        for _ in range(_NEWTON_ITERS):
            psi, dpsi = psi_and_slope(q)
            lo = jnp.where(psi < 0.0, q, lo)
            hi = jnp.where(psi >= 0.0, q, hi)
            slope = jnp.where(jnp.abs(dpsi) > _EPS_TINY, dpsi, 1.0)
            q_newton = q - psi / slope
            take = (q_newton >= lo) & (q_newton <= hi)
            q = jnp.where(take, q_newton, 0.5 * (lo + hi))
        return q

    def _stress_hyperbolic(
        self,
        eps: jnp.ndarray,
        eps_p_prev: jnp.ndarray,
        alpha_prev: jnp.ndarray,
    ) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]:
        """
        One-voxel tip-rounded return mapping -- ONE smooth surface, with no
        cone/apex branch anywhere in it.

            f = sqrt(q^2 + a_tip^2) + 3*a_f*p - sigma_y(alpha)

        a hyperbola asymptotic to the sharp cone (recovered exactly as
        a_tip -> 0) that crosses the hydrostatic axis at the finite tensile
        pressure p_tip = (sigma_y - a_tip)/(3*a_f) instead of running to a
        vertex. The plastic potential is rounded with the SAME a_tip,

            g = sqrt(q^2 + a_tip^2) + 3*a_g*p,

        which is what keeps the flow DIRECTION smooth too; rounding only the
        yield surface would leave the corner in the flow rule and defeat the
        point.

        Why this is worth a local solve. At the sharp vertex the return sets
        s = 0 exactly, so the whole deviatoric block of the consistent
        tangent vanishes and the tangent is singular -- correct for a
        vertex, and fatal for the CG inside
        problems.mechanics.solve_displacement_based_nonlinear. Rounded, the
        return lands at a small but NONZERO q, the tangent keeps a finite
        deviatoric stiffness, and CG has a solvable operator again.

        The return mapping, and why it is one scalar equation
        ----------------------------------------------------
        Flow is dg/dsigma = 1.5*s/xi + a_g*I with xi = sqrt(q^2 + a_tip^2),
        whose deviatoric part is still PARALLEL to s -- the return stays
        radial in the deviatoric plane, exactly as in the sharp cone. So

            s*(1 + 3*mu*dlam/xi) = s_trial          (deviatoric)
            p = p_trial - 3*K*a_g*dlam              (volumetric)

        and the consistency condition f = 0 then makes xi AFFINE in dlam:

            xi = xi_0 + c*dlam,   xi_0 = sigma_y - 3*a_f*p_trial,
                                  c    = H + 9*K*a_f*a_g

        so dlam is recoverable from q alone, and the deviatoric equation
        collapses to the single scalar residual solved in
        _solve_q_hyperbolic.

        That affine relation is what makes the solve one-dimensional, and it
        survives any PIECEWISE-affine hardening law: on segment j the same
        algebra gives xi = xi_0j + c_j*dlam with

            xi_0j = sig_j + H_j*(alpha_prev - alpha_j) - 3*a_f*p_trial
            c_j   = H_j + 9*K*a_f*a_g

        so the branch solves every segment and keeps the self-consistent one,
        reusing the identical scalar residual. A genuinely SMOOTH sigma_y
        (Voce, power-law) would instead make xi implicit in dlam and need a
        2x2 local solve -- not built, and rejected in __init__ rather than
        silently approximated.

        c > 0 is floored for the same reason hardening.solve_return floors
        its own denominator, and it is the same degenerate combination:
        H = 0 with a_f*a_g = 0, perfectly plastic and non-dilatant yet
        pressure-sensitive, where no return to the tip exists at all.

        Parameters / Returns: as stress().
        """
        eps = 0.5 * (eps + eps.T)
        I3 = jnp.eye(3)
        a = self.a_tip

        eps_e_trial = eps - eps_p_prev
        sigma_trial = (self.lam * jnp.trace(eps_e_trial) * I3
                       + 2.0 * self.mu * eps_e_trial)
        p_trial = jnp.trace(sigma_trial) / 3.0
        s_trial = sigma_trial - p_trial * I3
        # Same guard, same reason, as _stress_sharp: q_trial = ||s_trial||
        # sits exactly on sqrt's singular point at a virgin state.
        s_sq = jnp.sum(s_trial ** 2)
        q_trial = jnp.sqrt(1.5 * jnp.maximum(s_sq, _EPS_TINY))
        q_safe = jnp.where(q_trial > _EPS_TINY, q_trial, 1.0)

        xi_trial = jnp.sqrt(q_trial ** 2 + a * a)
        f_trial = xi_trial + 3.0 * self.a_f * p_trial - self.hardening.sigma_y(alpha_prev)

        # One set of affine coefficients PER SEGMENT of sigma_y, all shape
        # (n_seg,) -- a single segment for LinearHardening, so this reduces
        # to the scalar solve it replaces with identical arithmetic.
        alpha_lo, sigma_lo, H_seg, _ = self.hardening.affine_segments()
        xi_0 = sigma_lo + H_seg * (alpha_prev - alpha_lo) - 3.0 * self.a_f * p_trial
        c = jnp.maximum(H_seg + 9.0 * self.K * self.a_f * self.a_g, _EPS_TINY)
        k = 3.0 * self.mu / c

        # Pick the segment holding the root BEFORE solving, so the (expensive)
        # local Newton runs once instead of once per segment. Solving all of
        # them and keeping the self-consistent one is correct but wasteful:
        # measured on 200k voxels, stress_and_tangent_field costs
        # 3.80 + 5.64*n_seg ms (R^2 = 0.99 over n_seg = 2..12), against a flat
        # ~3.9 ms for the sharp cone at any n_seg -- and jacfwd carries
        # tangents through every discarded segment's iterations too, a further
        # ~7.6x on what is thrown away.
        #
        # The test compares, at each interior breakpoint, the plastic
        # multiplier the two conditions each demand there:
        #
        #   CONSISTENCY reaches the breakpoint alpha_b after dlam_c =
        #   alpha_b - alpha_prev, and fixes the surface it must land on,
        #       xi_c = sigma_y(alpha_b) - 3*a_f*p_trial + 9*K*a_f*a_g*dlam_c
        #   which needs no solve: sigma_y(alpha_b) is the tabulated value.
        #
        #   The RADIAL RETURN, to shrink q_trial onto that same surface
        #   (q_c = sqrt(xi_c^2 - a_tip^2)), needs dlam_r = xi_c*(q_trial -
        #   q_c)/(3*mu*q_c), straight from q*(1 + 3*mu*dlam/xi) = q_trial.
        #
        # dlam_r > dlam_c means the return still has further to go at that
        # breakpoint, i.e. the root lies above it -- so the number of
        # breakpoints where that holds IS the index of the segment containing
        # the root.
        #
        # This ordering is provably monotone, which a residual in dlam is NOT:
        # xi_c increases with alpha_b (sigma_y non-decreasing, dilatancy term
        # increasing), and d/d(xi) of xi*(q_trial - q)/q is
        # -q_trial*a_tip^2/(xi^2 - a_tip^2)^(3/2) - 1 < 0, so dlam_r strictly
        # decreases while dlam_c strictly increases. Their difference changes
        # sign exactly once. The first version of this tested the sign of the
        # radial-return residual R(dlam) instead, which is cheaper still but
        # non-monotone for steep hardening with a_g = 0 -- it picked the wrong
        # segment and moved stresses by 130 MPa. The sweep below is what
        # caught it.
        #
        # Evaluated at the INTERIOR breakpoints only (alpha_lo[1:], where
        # sigma_y is exactly the tabulated sigma_lo[1:]): the last segment's
        # upper bound is +inf, and with its plateau slope H = 0 the product
        # 0*inf would be NaN. Excluding it costs nothing -- a root in no
        # earlier segment is in the last one, which the count yields anyway.
        brk_dlam = jnp.maximum(alpha_lo[1:] - alpha_prev, 0.0)
        # xi < a_tip is unreachable (xi = sqrt(q^2 + a_tip^2) >= a_tip) and
        # happens at breakpoints whose hardening cannot meet a strongly
        # tensile p_trial. Clamped, q_c -> 0 makes dlam_r huge, correctly
        # reading as "the root is further up".
        brk_xi = jnp.maximum(
            sigma_lo[1:] - 3.0 * self.a_f * p_trial
            + 9.0 * self.K * self.a_f * self.a_g * brk_dlam, a)
        brk_q = jnp.sqrt(jnp.maximum(brk_xi ** 2 - a * a, _EPS_TINY))
        brk_dlam_r = brk_xi * (q_trial - brk_q) / (3.0 * self.mu * brk_q)
        # Integer, so it carries no tangent: neither the sqrt nor the division
        # above can poison the consistent tangent however small brk_q gets,
        # and the selection contributes exactly the zero derivative a
        # piecewise-linear law should. An empty slice (LinearHardening) sums
        # to 0, leaving the single-segment path bit-for-bit unchanged.
        j = jnp.sum(brk_dlam_r > brk_dlam)

        xi_0_j, c_j, k_j = xi_0[j], c[j], k[j]
        q_seg = self._solve_q_hyperbolic(q_trial, xi_trial, xi_0_j, c_j, k_j)
        # dlam from the RADIAL RETURN, q*(1 + 3*mu*dlam/xi) = q_trial, rather
        # than by inverting the affine consistency relation as
        # (xi - xi_0)/c. The two agree at the solution, but c = H +
        # 9*K*a_f*a_g vanishes for a non-dilatant (a_g = 0) material on a
        # PERFECTLY PLASTIC segment, and every table ends in one by the
        # plateau convention -- so the (xi - xi_0)/c form divides a rounding
        # error by _EPS_TINY there and leaves the state off the yield surface
        # by O(100 MPa). The radial form has no c in it and stays exact;
        # its own degeneracy, q -> 0, is the tip itself, already guarded.
        xi_seg = jnp.sqrt(q_seg ** 2 + a * a)
        q_den = jnp.maximum(q_seg, _EPS_TINY)
        dlam_seg = jnp.maximum(
            xi_seg * (q_trial - q_seg) / (3.0 * self.mu * q_den), 0.0)

        # Elastic states have no root inside [0, q_trial] (Psi < 0 across the
        # whole bracket), so the solve's bisection would creep toward q_trial
        # without reaching it. Select the exact elastic answer instead -- the
        # discarded solve value is still finite, bounded in [0, q_trial], so
        # it cannot poison this branch's tangent.
        q_new = jnp.where(f_trial > 0.0, q_seg, q_trial)
        dlam = jnp.where(f_trial > 0.0, dlam_seg, 0.0)

        xi_new = jnp.sqrt(q_new ** 2 + a * a)

        # q is homogeneous of degree 1 in s and the return is radial, so the
        # updated deviator is just the trial one rescaled.
        s_new = s_trial * (q_new / q_safe)
        p_new = p_trial - 3.0 * self.K * self.a_g * dlam
        sigma = s_new + p_new * I3
        eps_p = eps_p_prev + dlam * (1.5 * s_new / xi_new + self.a_g * I3)
        alpha = alpha_prev + dlam

        return sigma, (eps_p, alpha)

    def stress_and_tangent(
        self,
        eps: jnp.ndarray,
        eps_p_prev: jnp.ndarray,
        alpha_prev: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]:
        """
        One-voxel stress, consistent tangent, and updated state -- tangent
        via jax.jacfwd on stress() rather than a hand-derived formula.

        Note the tangent has no major symmetry (C_ijkl != C_klij) whenever
        a_g != a_f: a non-associated flow rule genuinely has a non-symmetric
        consistent tangent, and that is correct here, not a bug -- do not
        "fix" it by symmetrizing, which would break the quadratic Newton
        convergence this tangent exists to provide. It does mean
        problems.mechanics.solve_displacement_based_nonlinear applies CG to
        a non-symmetric operator (its A_op wraps C_tan unsymmetrized), which
        is formally outside CG's assumptions. In practice the asymmetry is a
        modest perturbation of a strongly symmetric-positive operator and
        Newton still converges -- verified in
        test/test_materialmodels_drucker_prager.py, check 6, at a_f = 0.13
        with a_g = 0.05. A strongly non-associated model (a_g far below a_f)
        that stalls in the inner solve is hitting this, not a bug in the
        return mapping; associated flow (a_g = a_f, the default) keeps the
        tangent symmetric and CG on solid ground.

        sigma is threaded through as part of jacfwd's aux output (has_aux
        only returns (jacobian, aux), never the primal value) so this needs
        just one autodiff pass, not a second plain call to stress().

        Returns
        -------
        sigma  : (3,3)
        C_tan  : (3,3,3,3)
        (eps_p, alpha) : updated state
        """
        def _fn(e):
            sigma, new_state = self.stress(e, eps_p_prev, alpha_prev)
            return sigma, (sigma, new_state)

        C_tan, (sigma, new_state) = jax.jacfwd(_fn, has_aux=True)(eps)
        return sigma, C_tan, new_state

    def stress_and_tangent_field(
        self,
        eps_field: jnp.ndarray,
        eps_p_field: jnp.ndarray,
        alpha_field: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]:
        """
        Per-voxel stress/tangent/state field, vmapped over
        stress_and_tangent -- the method assemble_local_update duck-types
        on, identical in signature and shapes to J2Plasticity's.

        Parameters
        ----------
        eps_field   : (3, 3, Nv)
        eps_p_field : (3, 3, Nv)
        alpha_field : (Nv,)

        Returns
        -------
        sigma_field : (3, 3, Nv)
        C_field     : (3, 3, 3, 3, Nv)
        (eps_p_field, alpha_field) : updated state, same shapes as inputs
        """
        eps_vox   = eps_field.transpose(2, 0, 1)    # (Nv, 3, 3)
        eps_p_vox = eps_p_field.transpose(2, 0, 1)  # (Nv, 3, 3)

        sigma_vox, C_vox, (eps_p_new_vox, alpha_new) = jax.vmap(
            self.stress_and_tangent
        )(eps_vox, eps_p_vox, alpha_field)

        sigma_field  = sigma_vox.transpose(1, 2, 0)        # (3, 3, Nv)
        C_field      = C_vox.transpose(1, 2, 3, 4, 0)      # (3, 3, 3, 3, Nv)
        eps_p_field  = eps_p_new_vox.transpose(1, 2, 0)    # (3, 3, Nv)
        return sigma_field, C_field, (eps_p_field, alpha_new)

    def __repr__(self) -> str:
        tag = f" ({self.name})" if self.name else ""
        flow = "associated" if self.a_g == self.a_f else "non-associated"
        if self.a_tip > 0.0:
            p_tip = (self.sigma_y0 - self.a_tip) / (3.0 * self.a_f)
            tip = f", a_tip={self.a_tip:.3g} (rounded, p_tip={p_tip:.3g})"
        else:
            tip = " (sharp tip)"
        return (f"DruckerPrager{tag}: E={self.E:.3g}, nu={self.nu:.3g}, "
                f"hardening={self.hardening}, "
                f"a_f={self.a_f:.3g}, a_g={self.a_g:.3g} ({flow}){tip}")
