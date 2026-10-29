"""
Drucker-Prager plasticity with pluggable isotropic hardening; a drop-in for
J2Plasticity (same state convention and methods), with a_f = a_g = 0 reducing
to J2 exactly.

Limits worth knowing:

- Unreachable states: with a_g = 0 plastic flow cannot change p, so a tensile
  trial pressure is reached only by the surface growing out to it. If sigma_y
  has a ceiling (H = 0, or a table's plateau), states with
  3 a_f p_trial > sigma_y_max - a_tip have no return at all; they come back
  as NaN, never as a plausible wrong stress. The real fix would be a tension
  cut-off (second surface), not built.
- Sharp vertex: the apex return sets s = 0, so the tangent is singular there.
  ``a_tip > 0`` rounds the tip into a hyperbola and keeps it regular (min
  eigenvalue 55 instead of 0 on a confined path, stresses within 1%).
"""

import sys
import warnings

import jax
import jax.numpy as jnp
import numpy as np

from materialmodels.base import ConstitutiveModel
from materialmodels.inelastic.hardening import IsotropicHardening, resolve_hardening

_EPS_TINY = 1e-12

# Fixed count for the tip-rounded return: 6 iterations reach machine precision
# over a 9238-state sweep; 8 adds margin. A batched while_loop would run to the
# worst voxel anyway.
_NEWTON_ITERS = 8


class DruckerPrager(ConstitutiveModel):
    """
    Rate-independent Drucker-Prager plasticity, optionally non-associated.

        yield     f = q + 3 a_f p - sigma_y(alpha),   q = sqrt(3/2 s:s), p = tr(sigma)/3
        potential g = q + 3 a_g p

    Tension promotes yield; a_f = 0 is von Mises. a_g sets dilatancy: a_g = a_f
    associated (default), a_g < a_f the usual choice for polymers.

    Parameters
    ----------
    E, nu     : elastic constants
    sigma_y0  : initial yield stress on the shear meridian (not the uniaxial
                tensile strength once a_f > 0)
    H         : linear hardening modulus (0 = perfectly plastic)
    hardening : IsotropicHardening or ``{"law": ...}`` dict, instead of H;
                with a_tip > 0 it must be piecewise affine
    a_f       : friction coefficient, 0 <= a_f < 1, required
    a_g       : dilatancy, 0 <= a_g <= a_f; None = a_f
    a_tip     : tip-rounding radius (0 = sharp cone)
    name      : display label
    k_res, Gc : phase-field parameters (see LinearElasticIsotropic)

    Choosing parameters
    -------------------
    Uniaxial strengths: sigma_t = sigma_y0 / (1 + a_f), sigma_c = sigma_y0 / (1 - a_f).
    From m = sigma_c / sigma_t: a_f = (m - 1)/(m + 1), sigma_y0 = sigma_t (1 + a_f).
    From Mohr-Coulomb (phi, c, psi), outer cone:
    a_f = 2 sin(phi)/(3 - sin(phi)), a_g = 2 sin(psi)/(3 - sin(psi)),
    sigma_y0 = 6 c cos(phi)/(3 - sin(phi)).
    """

    def __init__(self, E: float, nu: float, sigma_y0: float | None = None,
                 H: float | None = None, a_f: float | None = None,
                 a_g: float | None = None, a_tip: float = 0.0,
                 hardening: "IsotropicHardening | dict | None" = None,
                 name: str = "", k_res: float = 1e-6, Gc: float | None = None):
        self.E = float(E)
        self.nu = float(nu)
        self.hardening = resolve_hardening(hardening, sigma_y0, H, f"DruckerPrager {name!r}")
        self.sigma_y0 = self.hardening.sigma_y0
        # No default: a_f = 0 would silently make this von Mises.
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

        # Static conditioning of the tip-rounded solve: c = H + 9 K a_f a_g per
        # segment, k = 3 mu / c. c == 0 segments are solved in closed form.
        segs = self.hardening.affine_segments()
        self._c_zero = None
        self._newton_iters = _NEWTON_ITERS
        if self.a_tip > 0.0 and segs is not None:
            c_static = (np.asarray(segs[2], dtype=float)
                        + 9.0 * self.K * self.a_f * self.a_g)
            self._c_zero = c_static <= 0.0
            nz = c_static[c_static > 0.0]
            if nz.size:
                k_max = 3.0 * self.mu / float(nz.min())
                # Large k falls back to bisection, one iteration per halving:
                # at k ~ 1.5e15, 8 iterations leave |f| ~ 3e-3, 16 reach 8e-12.
                if k_max > 1.0e3:
                    self._newton_iters = min(
                        _NEWTON_ITERS + 4 * int(np.ceil(np.log10(k_max / 1.0e3))), 40)

        # Warn, don't raise: below the limit pressure this is a valid
        # material, and every tabulated law at a_g = 0 has a ceiling.
        if self._can_be_unreachable():
            p_lim = (self.hardening.sigma_y_max - self.a_tip) / (3.0 * self.a_f)
            why = (f"sigma_y is capped at {self.hardening.sigma_y_max:.4g} "
                   f"(perfectly plastic)" if self.hardening.sigma_y_max == self.sigma_y0
                   else f"sigma_y saturates at {self.hardening.sigma_y_max:.4g} "
                        f"(its table plateaus)")
            print(
                f"WARNING: DruckerPrager {self.name!r}: a_g = 0 (no dilatancy) with a "
                f"hardening law whose {why}.\n"
                f"         Plastic flow cannot change p and the surface cannot grow past "
                f"that ceiling, so trial\n"
                f"         states above p = {p_lim:.4g} MPa have no admissible return and "
                f"are returned as NaN.\n"
                f"         Fine below that pressure; give a_g > 0, an unbounded (linear, "
                f"H > 0) law, or a\n"
                f"         tension cut-off if the run needs to go past it.",
                file=sys.stderr)

    def _can_be_unreachable(self) -> bool:
        """Whether any trial state can lack a return: a_f > 0, a_g = 0, finite sigma_y ceiling."""
        return bool(self.a_f > 0.0 and self.a_g == 0.0
                    and np.isfinite(self.hardening.sigma_y_max))

    def _unreachable(self, p_trial):
        """
        Per-voxel mask of unreachable trial states, or None (static) when there are none.

        A return needs sigma_y >= a_tip + 3 a_f p_trial; elastic states always
        satisfy it, so no plasticity gate is needed.
        """
        if not self._can_be_unreachable():
            return None
        return self.hardening.sigma_y_max < self.a_tip + 3.0 * self.a_f * p_trial

    @staticmethod
    def _nan_out(bad, sigma, eps_p, alpha):
        """
        NaN out unreachable states so they fail the Newton step.

        Covers stress() only: its derivative there is zero, not NaN, so
        stress_and_tangent NaNs C_tan separately.
        """
        if bad is None:
            return sigma, eps_p, alpha
        nan = jnp.asarray(jnp.nan, dtype=sigma.dtype)
        return (jnp.where(bad, nan, sigma),
                jnp.where(bad, nan, eps_p),
                jnp.where(bad, nan, alpha))

    def elastic_stiffness_tensor(self) -> jnp.ndarray:
        """(3, 3, 3, 3) elastic stiffness, not the elastoplastic tangent."""
        d = jnp.eye(3)
        return (self.lam * jnp.einsum('ij,kl->ijkl', d, d)
                + self.mu * (jnp.einsum('ik,jl->ijkl', d, d)
                             + jnp.einsum('il,jk->ijkl', d, d)))

    def stiffness_tensor(
        self,
        eps: jnp.ndarray,
        eps_p_prev: jnp.ndarray,
        alpha_prev: jnp.ndarray,
    ) -> jnp.ndarray:
        """(3, 3, 3, 3) elastoplastic tangent at (eps, state); non-symmetric if a_g != a_f."""
        _, C_tan, _ = self.stress_and_tangent(eps, eps_p_prev, alpha_prev)
        return C_tan

    def stress(
        self,
        eps: jnp.ndarray,
        eps_p_prev: jnp.ndarray,
        alpha_prev: jnp.ndarray,
    ) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]:
        """
        Return mapping of one voxel: sharp cone (``a_tip = 0``) or rounded tip.

        Parameters
        ----------
        eps        : (3, 3) total strain, symmetrized internally
        eps_p_prev : (3, 3) plastic strain of the last converged state
        alpha_prev : scalar accumulated plastic strain

        Returns
        -------
        sigma          : (3, 3)
        (eps_p, alpha) : updated state
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
        Sharp-cone return of one voxel, closed form: cone or apex, selected per voxel.

        Cone: J2's radial return plus a volumetric correction; dlam = 0 below
        yield gives the elastic case. Apex: past the vertex the cone return
        would flip the deviator (q < 0), so s = 0 and only the hardening
        increment is solved, parametrized by dalpha so a_g = 0 stays finite.

        Both branches are always evaluated: the discarded one must stay
        finite or it poisons the kept tangent (hence q_safe and the floored
        denominators).
        """
        eps = 0.5 * (eps + eps.T)
        I3 = jnp.eye(3)

        eps_e_trial = eps - eps_p_prev
        sigma_trial = (self.lam * jnp.trace(eps_e_trial) * I3
                       + 2.0 * self.mu * eps_e_trial)
        p_trial = jnp.trace(sigma_trial) / 3.0
        s_trial = sigma_trial - p_trial * I3
        # sqrt has a NaN gradient at 0, which q_trial hits at a virgin state.
        s_sq = jnp.sum(s_trial ** 2)
        q_trial = jnp.sqrt(1.5 * jnp.maximum(s_sq, _EPS_TINY))
        q_safe = jnp.where(q_trial > _EPS_TINY, q_trial, 1.0)

        B_cone = 3.0 * self.mu + 9.0 * self.K * self.a_f * self.a_g
        dlam = self.hardening.solve_return(
            q_trial + 3.0 * self.a_f * p_trial, B_cone, alpha_prev)
        n_dev = 1.5 * s_trial / q_safe
        s_cone = s_trial * (1.0 - 3.0 * self.mu * dlam / q_safe)
        sigma_cone = s_cone + (p_trial - 3.0 * self.K * self.a_g * dlam) * I3
        eps_p_cone = eps_p_prev + dlam * (n_dev + self.a_g * I3)
        alpha_cone = alpha_prev + dlam

        B_apex = 9.0 * self.K * self.a_f * self.a_g
        dalpha_apex = self.hardening.solve_return(
            3.0 * self.a_f * p_trial, B_apex, alpha_prev)
        sigma_apex = (p_trial - 3.0 * self.K * self.a_g * dalpha_apex) * I3
        eps_p_apex = (eps_p_prev + s_trial / (2.0 * self.mu)
                      + self.a_g * dalpha_apex * I3)
        alpha_apex = alpha_prev + dalpha_apex

        use_apex = (q_trial - 3.0 * self.mu * dlam) < 0.0
        sigma = jnp.where(use_apex, sigma_apex, sigma_cone)
        eps_p = jnp.where(use_apex, eps_p_apex, eps_p_cone)
        alpha = jnp.where(use_apex, alpha_apex, alpha_cone)

        sigma, eps_p, alpha = self._nan_out(
            self._unreachable(p_trial), sigma, eps_p, alpha)
        return sigma, (eps_p, alpha)

    def _solve_q_hyperbolic(self, q_trial, xi_trial, xi_0, c, k):
        """
        Bracketed Newton for the updated q of the rounded return:

            Psi(q) = q (xi (1 + k) - k xi_0) - q_trial xi,   xi = sqrt(q^2 + a_tip^2)

        [0, q_trial] holds exactly one root for a plastic state. Psi is
        non-monotone in ~30% of states, so plain Newton can leave the bracket;
        bisection is the fallback. The bracket test is closed: an open one
        rejects a converged iterate and caps accuracy at ~1e-1.
        """
        a = self.a_tip

        def psi_and_slope(q):
            xi = jnp.sqrt(q * q + a * a)
            psi = q * (xi * (1.0 + k) - k * xi_0) - q_trial * xi
            dpsi = (1.0 + k) * (xi + q * q / xi) - k * xi_0 - q_trial * q / xi
            return psi, dpsi

        dlam0 = jnp.maximum(xi_trial - xi_0, 0.0) / (3.0 * self.mu + c)
        q = jnp.clip(q_trial - 3.0 * self.mu * dlam0, 0.0, q_trial)
        lo = jnp.zeros_like(q_trial)
        hi = q_trial

        for _ in range(self._newton_iters):
            psi, dpsi = psi_and_slope(q)
            lo = jnp.where(psi < 0.0, q, lo)
            hi = jnp.where(psi >= 0.0, q, hi)
            slope = jnp.where(jnp.abs(dpsi) > _EPS_TINY, dpsi, 1.0)
            q_newton = q - psi / slope
            take = (q_newton >= lo) & (q_newton <= hi)
            q = jnp.where(take, q_newton, 0.5 * (lo + hi))

        # Tangent from one Newton step at the root (implicit function theorem),
        # not through the loop: the Newton-vs-bisection choice at a converged
        # root depends on rounding that jit+vmap batching changes, and
        # bisection carries earlier iterates' tangents (196x off in check 12).
        q = jax.lax.stop_gradient(q)
        psi, dpsi = psi_and_slope(q)
        return q - psi / jnp.where(jnp.abs(dpsi) > _EPS_TINY, dpsi, 1.0)

    def _stress_hyperbolic(
        self,
        eps: jnp.ndarray,
        eps_p_prev: jnp.ndarray,
        alpha_prev: jnp.ndarray,
    ) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]:
        """
        Tip-rounded return of one voxel, one smooth surface:

            f = sqrt(q^2 + a_tip^2) + 3 a_f p - sigma_y,   g = sqrt(q^2 + a_tip^2) + 3 a_g p

        Rounding the potential too keeps the flow direction smooth. The return
        stays radial, and consistency makes xi = xi_0 + c dlam affine on each
        hardening segment (xi_0j = sig_j + H_j (alpha_prev - alpha_j) - 3 a_f
        p_trial, c_j = H_j + 9 K a_f a_g), so one scalar solve in q remains.
        A smooth (non-affine) sigma_y would need a 2x2 solve; it is rejected
        in __init__.
        """
        eps = 0.5 * (eps + eps.T)
        I3 = jnp.eye(3)
        a = self.a_tip

        eps_e_trial = eps - eps_p_prev
        sigma_trial = (self.lam * jnp.trace(eps_e_trial) * I3
                       + 2.0 * self.mu * eps_e_trial)
        p_trial = jnp.trace(sigma_trial) / 3.0
        s_trial = sigma_trial - p_trial * I3
        s_sq = jnp.sum(s_trial ** 2)
        q_trial = jnp.sqrt(1.5 * jnp.maximum(s_sq, _EPS_TINY))
        q_safe = jnp.where(q_trial > _EPS_TINY, q_trial, 1.0)

        xi_trial = jnp.sqrt(q_trial ** 2 + a * a)
        f_trial = xi_trial + 3.0 * self.a_f * p_trial - self.hardening.sigma_y(alpha_prev)

        alpha_lo, sigma_lo, H_seg, _ = self.hardening.affine_segments()
        xi_0 = sigma_lo + H_seg * (alpha_prev - alpha_lo) - 3.0 * self.a_f * p_trial
        c = jnp.maximum(H_seg + 9.0 * self.K * self.a_f * self.a_g, _EPS_TINY)
        k = 3.0 * self.mu / c

        # Pick the root's segment before solving (one Newton instead of one per
        # segment, 4.4x faster). At each interior breakpoint compare the
        # multiplier consistency needs (dlam_c) with the one the radial return
        # needs to reach that surface (dlam_r); dlam_r > dlam_c means the root
        # lies above, so the count of such breakpoints is the segment index.
        # Monotone in the breakpoint, unlike the sign of the residual R(dlam),
        # which picked wrong segments (130 MPa off) with steep hardening and
        # a_g = 0. Interior breakpoints only: the plateau's 0 * inf is NaN.
        brk_dlam = jnp.maximum(alpha_lo[1:] - alpha_prev, 0.0)
        brk_xi = jnp.maximum(
            sigma_lo[1:] - 3.0 * self.a_f * p_trial
            + 9.0 * self.K * self.a_f * self.a_g * brk_dlam, a)
        brk_q = jnp.sqrt(jnp.maximum(brk_xi ** 2 - a * a, _EPS_TINY))
        brk_dlam_r = brk_xi * (q_trial - brk_q) / (3.0 * self.mu * brk_q)
        j = jnp.sum(brk_dlam_r > brk_dlam)

        xi_0_j, c_j, k_j = xi_0[j], c[j], k[j]
        q_seg = self._solve_q_hyperbolic(q_trial, xi_trial, xi_0_j, c_j, k_j)
        # c == 0 (perfectly plastic, non-dilatant segment) pins xi at xi_0:
        # q in closed form. Static, so free for any other material.
        if self._c_zero is not None and bool(self._c_zero.any()):
            q_pin = jnp.clip(jnp.sqrt(jnp.maximum(xi_0_j ** 2 - a * a, _EPS_TINY)),
                             0.0, q_trial)
            q_seg = jnp.where(jnp.asarray(self._c_zero)[j], q_pin, q_seg)
        # dlam from the radial return, not (xi - xi_0)/c: c vanishes on a
        # non-dilatant plateau and that form left states O(100 MPa) off the surface.
        xi_seg = jnp.sqrt(q_seg ** 2 + a * a)
        q_den = jnp.maximum(q_seg, _EPS_TINY)
        dlam_seg = jnp.maximum(
            xi_seg * (q_trial - q_seg) / (3.0 * self.mu * q_den), 0.0)

        # Elastic states have no root in the bracket; select the elastic answer.
        q_new = jnp.where(f_trial > 0.0, q_seg, q_trial)
        dlam = jnp.where(f_trial > 0.0, dlam_seg, 0.0)

        xi_new = jnp.sqrt(q_new ** 2 + a * a)

        s_new = s_trial * (q_new / q_safe)
        p_new = p_trial - 3.0 * self.K * self.a_g * dlam
        sigma = s_new + p_new * I3
        eps_p = eps_p_prev + dlam * (1.5 * s_new / xi_new + self.a_g * I3)
        alpha = alpha_prev + dlam

        sigma, eps_p, alpha = self._nan_out(
            self._unreachable(p_trial), sigma, eps_p, alpha)
        return sigma, (eps_p, alpha)

    def stress_and_tangent(
        self,
        eps: jnp.ndarray,
        eps_p_prev: jnp.ndarray,
        alpha_prev: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]:
        """
        Stress, consistent tangent (``jax.jacfwd`` of ``stress``) and updated state of one voxel.

        For a_g != a_f the tangent is genuinely non-symmetric; do not
        symmetrize it (that breaks quadratic Newton convergence). CG then runs
        on a mildly non-symmetric operator, which works in practice (check 6);
        a strongly non-associated model may stall there. Unreachable states
        are NaN in every output.

        Returns
        -------
        sigma          : (3, 3)
        C_tan          : (3, 3, 3, 3)
        (eps_p, alpha) : updated state
        """
        def _fn(e):
            sigma, new_state = self.stress(e, eps_p_prev, alpha_prev)
            return sigma, (sigma, new_state)

        C_tan, (sigma, new_state) = jax.jacfwd(_fn, has_aux=True)(eps)
        if self._can_be_unreachable():
            # A constant NaN has a zero tangent, so _nan_out alone leaves a
            # singular zero block. A non-constant NaN inside stress() would
            # instead NaN the reverse-mode gradient of every reachable state.
            C_tan = jnp.where(jnp.all(jnp.isfinite(sigma)), C_tan,
                              jnp.asarray(jnp.nan, dtype=C_tan.dtype))
        return sigma, C_tan, new_state

    def stress_and_tangent_field(
        self,
        eps_field: jnp.ndarray,
        eps_p_field: jnp.ndarray,
        alpha_field: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]:
        """
        ``stress_and_tangent`` per voxel (the method assemble_local_update looks for).

        Parameters
        ----------
        eps_field, eps_p_field : (3, 3, Nv)
        alpha_field            : (Nv,)

        Returns
        -------
        sigma_field                : (3, 3, Nv)
        C_field                    : (3, 3, 3, 3, Nv)
        (eps_p_field, alpha_field) : updated state
        """
        eps_vox   = eps_field.transpose(2, 0, 1)
        eps_p_vox = eps_p_field.transpose(2, 0, 1)

        sigma_vox, C_vox, (eps_p_new_vox, alpha_new) = jax.vmap(
            self.stress_and_tangent
        )(eps_vox, eps_p_vox, alpha_field)

        sigma_field  = sigma_vox.transpose(1, 2, 0)
        C_field      = C_vox.transpose(1, 2, 3, 4, 0)
        eps_p_field  = eps_p_new_vox.transpose(1, 2, 0)
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
