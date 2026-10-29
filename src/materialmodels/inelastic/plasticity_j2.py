"""
J2 (von Mises) plasticity with pluggable isotropic hardening.

Stateful: stress needs (eps, eps_p, alpha), and the caller threads the state
across increments. ``elastic_stiffness_tensor()`` is the elastic tensor only;
the tangent comes from autodiff (notes/AUTODIFF_CONSTITUTIVE.md, Proposal A).
"""

import jax
import jax.numpy as jnp

from materialmodels.base import ConstitutiveModel
from materialmodels.inelastic.hardening import IsotropicHardening, resolve_hardening

_EPS_TINY = 1e-12


class J2Plasticity(ConstitutiveModel):
    """
    Rate-independent J2 plasticity, radial return; the hardening law solves
    equation [*] of materialmodels.inelastic.hardening at (A, B) = (q_trial, 3 mu).

    Parameters
    ----------
    E, nu     : elastic constants
    sigma_y0  : initial yield stress (with H: linear hardening)
    H         : linear hardening modulus (0 = perfectly plastic)
    hardening : IsotropicHardening or ``{"law": ...}`` dict, instead of H
    name      : display label
    k_res, Gc : phase-field parameters (see LinearElasticIsotropic)
    """

    def __init__(self, E: float, nu: float, sigma_y0: float | None = None,
                 H: float | None = None,
                 hardening: "IsotropicHardening | dict | None" = None,
                 name: str = "", k_res: float = 1e-6, Gc: float | None = None):
        self.E = float(E)
        self.nu = float(nu)
        self.hardening = resolve_hardening(hardening, sigma_y0, H, f"J2Plasticity {name!r}")
        self.sigma_y0 = self.hardening.sigma_y0
        self.name = name
        self.k_res = float(k_res)
        self.Gc = float(Gc) if Gc is not None else None
        self.lam = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        self.mu = self.E / (2.0 * (1.0 + self.nu))

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
        """(3, 3, 3, 3) elastoplastic tangent at (eps, state)."""
        _, C_tan, _ = self.stress_and_tangent(eps, eps_p_prev, alpha_prev)
        return C_tan

    def stress(
        self,
        eps: jnp.ndarray,
        eps_p_prev: jnp.ndarray,
        alpha_prev: jnp.ndarray,
    ) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]:
        """
        Radial return of one voxel.

        No elastic/plastic branch: the hardening law returns dgamma = 0 when
        elastic, which reduces every formula to the elastic case. ``eps`` is
        symmetrized internally (needed for a correct autodiff tangent).

        Parameters
        ----------
        eps        : (3, 3) total strain
        eps_p_prev : (3, 3) plastic strain of the last converged state
        alpha_prev : scalar accumulated plastic strain

        Returns
        -------
        sigma          : (3, 3)
        (eps_p, alpha) : updated state
        """
        eps = 0.5 * (eps + eps.T)
        I3 = jnp.eye(3)

        eps_e_trial = eps - eps_p_prev
        sigma_trial = (self.lam * jnp.trace(eps_e_trial) * I3
                       + 2.0 * self.mu * eps_e_trial)
        p = jnp.trace(sigma_trial) / 3.0
        s_trial = sigma_trial - p * I3
        # sqrt has a NaN gradient at 0, which q_trial hits at a virgin state
        # (e.g. a zero Newton guess); clamping makes it exactly 0 instead.
        s_sq = jnp.sum(s_trial ** 2)
        q_trial = jnp.sqrt(1.5 * jnp.maximum(s_sq, _EPS_TINY))

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
        Stress, consistent tangent (``jax.jacfwd`` of ``stress``) and updated
        state of one voxel, in one autodiff pass.

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
        return sigma, C_tan, new_state

    def stress_and_tangent_field(
        self,
        eps_field: jnp.ndarray,
        eps_p_field: jnp.ndarray,
        alpha_field: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]:
        """
        ``stress_and_tangent`` per voxel.

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
        return (f"J2Plasticity{tag}: E={self.E:.3g}, nu={self.nu:.3g}, "
                f"hardening={self.hardening}")
