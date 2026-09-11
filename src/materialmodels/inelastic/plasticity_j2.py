"""
J2 (von Mises) elastoplastic material model, pluggable isotropic hardening.

Unlike every other ConstitutiveModel in materialmodels/, this one is not
stateless -- elastoplastic stress/tangent need (eps, eps_p, alpha), not just
eps. State (eps_p, alpha) is per-voxel and threaded by the CALLER across load
increments, never stored on this instance -- same convention as
materialmodels.phasefield's d/H damage state relative to a plain elastic
model. stiffness_tensor() still returns the elastic-only stiffness (for a
caller that just wants an elastic reference, e.g. a fixed preconditioner);
it is not the elastoplastic tangent, which depends on state and is never
constant -- see stress_and_tangent for that.

See notes/AUTODIFF_CONSTITUTIVE.md (Proposal A) for why the tangent is
derived by autodiff here rather than a hand-coded consistent tangent, and
its two verified gotchas this file works around: strain symmetrization
(stress() does it internally) and jnp.where/division NaN-poisoning a branch
that should have zero value but not zero gradient.
"""

import jax
import jax.numpy as jnp

from materialmodels.base import ConstitutiveModel
from materialmodels.inelastic.hardening import IsotropicHardening, resolve_hardening

_EPS_TINY = 1e-12


class J2Plasticity(ConstitutiveModel):
    """
    Rate-independent J2 (von Mises) plasticity, radial return, with the
    isotropic hardening law sigma_y(alpha) supplied as a plug-in.

    The radial return is exactly equation [*] of
    materialmodels.inelastic.hardening with (A, B) = (q_trial, 3*mu), so this
    model never sees the hardening law's shape: linear, tabulated, or a
    future smooth law all enter through the same solve_return call below.

    Parameters
    ----------
    E, nu    : float  isotropic elastic constants
    sigma_y0 : float  initial (virgin) yield stress -- with H, the linear law
    H        : float  constant isotropic hardening modulus (0 = perfectly plastic)
    hardening : IsotropicHardening | dict | None  the hardening law, as an
               instance or a ``{"law": ...}`` config dict. Mutually exclusive
               with H; omit both and sigma_y0/H are required, which is the
               original spelling and still the default.
    name     : str    optional label
    k_res, Gc : as in LinearElasticIsotropic -- phase-field bookkeeping only,
                unrelated to the plastic return mapping itself.
    """

    def __init__(self, E: float, nu: float, sigma_y0: float | None = None,
                 H: float | None = None,
                 hardening: "IsotropicHardening | dict | None" = None,
                 name: str = "", k_res: float = 1e-6, Gc: float | None = None):
        self.E = float(E)
        self.nu = float(nu)
        self.hardening = resolve_hardening(hardening, sigma_y0, H, f"J2Plasticity {name!r}")
        # Kept as an attribute because it is a property of every law, not just
        # the linear one -- the virgin yield stress.
        self.sigma_y0 = self.hardening.sigma_y0
        self.name = name
        self.k_res = float(k_res)
        self.Gc = float(Gc) if Gc is not None else None
        self.lam = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        self.mu = self.E / (2.0 * (1.0 + self.nu))

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
        One-voxel closed-form radial return.

        The elastic and plastic branches are NOT selected by an explicit
        jnp.where on the whole update -- the hardening law returns dgamma
        exactly zero in the elastic case (see solve_return), which correctly
        collapses every formula below to the elastic identity (s_new =
        s_trial, eps_p unchanged) with no separate branch needed. Only the
        q_trial division needs an explicit safe-guard (see q_safe) since its
        gradient is singular at q_trial=0 even where dgamma=0 makes the
        value's dependence on it vanish.

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
        p = jnp.trace(sigma_trial) / 3.0
        s_trial = sigma_trial - p * I3
        # sqrt(x) has an infinite/NaN gradient at x=0 -- q_trial=||s_trial||
        # hits exactly that point at a virgin/unstressed state (eps=eps_p=0),
        # a real case (a Newton solve's zero initial guess) not a corner
        # case. Clamping the radicand away from 0 makes the gradient exactly
        # 0 there instead of NaN (correct: deep in the elastic domain,
        # sigma_y0 > 0, so dgamma is 0 in a neighborhood of the origin too).
        s_sq = jnp.sum(s_trial ** 2)
        q_trial = jnp.sqrt(1.5 * jnp.maximum(s_sq, _EPS_TINY))

        # Equation [*] of materialmodels.inelastic.hardening at
        # (A, B) = (q_trial, 3*mu). LinearHardening inverts it with exactly
        # the closed form this line used to inline.
        dgamma = self.hardening.solve_return(q_trial, 3.0 * self.mu, alpha_prev)

        q_safe = jnp.where(q_trial > _EPS_TINY, q_trial, 1.0)
        s = s_trial * (1.0 - 3.0 * self.mu * dgamma / q_safe)
        sigma = s + p * I3

        eps_p = eps_p_prev + dgamma * 1.5 * s_trial / q_safe
        alpha = alpha_prev + dgamma

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
        Per-voxel stress/tangent/state field, vmapped over stress_and_tangent
        -- same vmap-over-the-trailing-voxel-axis pattern as
        TransverseIsotropic.stiffness_field_oriented.

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
        return (f"J2Plasticity{tag}: E={self.E:.3g}, nu={self.nu:.3g}, "
                f"hardening={self.hardening}")
