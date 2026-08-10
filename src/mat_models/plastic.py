"""
J2 (von Mises) plasticity with linear isotropic hardening, small strain.

Closed-form radial-return algorithm (de Souza Neto, Peric & Owen,
"Computational Methods for Plasticity", Box 8.1) -- no inner Newton loop is
needed because the yield function is linear in the plastic multiplier for
linear isotropic hardening. Returns the algorithmic (consistent) tangent
alongside the stress, since the outer Newton-CG solver needs that, not just
sigma = C:eps.

Same tensor convention as mat_models.elastic (full tensor strain, not
Voigt-engineering shear).
"""

import jax
import jax.numpy as jnp


class J2Plasticity:
    """
    Isotropic linear elastic / J2 plastic material with linear isotropic
    hardening.

    Parameters
    ----------
    E        : float  Young's modulus (any consistent unit, e.g. MPa)
    nu       : float  Poisson's ratio
    sigma_y0 : float  initial (von Mises) yield stress
    H        : float  linear isotropic hardening modulus
                       (yield stress grows as sigma_y0 + H * alpha)
    name     : str    optional label for display
    """

    def __init__(self, E: float, nu: float, sigma_y0: float, H: float, name: str = ""):
        self.E    = float(E)
        self.nu   = float(nu)
        self.sigma_y0 = float(sigma_y0)
        self.H    = float(H)
        self.name = name
        self.lam  = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        self.mu   = E / (2.0 * (1.0 + nu))
        self.K    = self.lam + 2.0 / 3.0 * self.mu

    def __repr__(self) -> str:
        return (f"J2Plasticity ({self.name}): E={self.E:.4g}, nu={self.nu}, "
                f"sigma_y0={self.sigma_y0:.4g}, H={self.H:.4g}, "
                f"lam={self.lam:.4g}, mu={self.mu:.4g}")

    def stiffness_tensor(self) -> jnp.ndarray:
        """Elastic (initial) stiffness tensor -- same as LinearElasticIsotropic."""
        d = jnp.eye(3)
        return (self.lam * jnp.einsum('ij,kl->ijkl', d, d)
                + self.mu * (jnp.einsum('ik,jl->ijkl', d, d)
                             + jnp.einsum('il,jk->ijkl', d, d)))

    def radial_return(
        self,
        eps:         jnp.ndarray,
        eps_p_prev:  jnp.ndarray,
        alpha_prev:  jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Single-material-point closed-form radial return.

        Parameters
        ----------
        eps        : (3, 3)  total strain at this increment
        eps_p_prev : (3, 3)  plastic strain carried in from the previous
                     increment (deviatoric by construction of the flow rule)
        alpha_prev : ()      accumulated equivalent plastic strain, previous
                     increment

        Returns
        -------
        sigma      : (3, 3)
        eps_p_new  : (3, 3)
        alpha_new  : ()
        C_algo     : (3, 3, 3, 3)  algorithmic (consistent) tangent d(sigma)/d(eps)
                     -- discontinuous across the yield surface by construction
                     (the algorithmic tangent, unlike the continuum tangent,
                     jumps from C_e to C_algo_plastic exactly at first yield --
                     expected, not a bug).
        """
        d = jnp.eye(3)
        lam, mu, K = self.lam, self.mu, self.K
        sigma_y0, H = self.sigma_y0, self.H

        # ── elastic predictor ────────────────────────────────────────────
        eps_e_trial = eps - eps_p_prev
        tr_eps_e    = jnp.trace(eps_e_trial)
        sigma_trial = lam * tr_eps_e * d + 2.0 * mu * eps_e_trial
        p           = (1.0 / 3.0) * jnp.trace(sigma_trial)      # hydrostatic (elastic always)
        s_trial     = sigma_trial - p * d                        # deviatoric trial stress

        s_trial_norm = jnp.sqrt(jnp.sum(s_trial * s_trial))
        q_trial      = jnp.sqrt(1.5) * s_trial_norm               # von Mises equivalent trial stress
        f_trial      = q_trial - (sigma_y0 + H * alpha_prev)      # trial yield function

        is_plastic = f_trial > 0.0

        # flow direction -- guarded against s_trial_norm == 0 (pure hydrostatic
        # trial state; only reached when is_plastic is already False there,
        # since q_trial=0 < sigma_y0 unless sigma_y0<=0, so the guard value
        # is never actually used on a real yielding path)
        safe_norm = jnp.where(s_trial_norm > 0.0, s_trial_norm, 1.0)
        n = s_trial / safe_norm

        dgamma = jnp.where(is_plastic, f_trial / (3.0 * mu + H), 0.0)

        # de Souza Neto, Peric & Owen, Box 8.1: N = sqrt(3/2) * s/||s|| is the
        # flow direction as it appears in the flow rule (eps_p_dot = gamma_dot
        # * N) -- NOT the Frobenius-unit n = s/||s|| used for the tangent's
        # N⊗N below. Using n's coefficient (3*mu*dgamma) here instead of N's
        # (2*mu*dgamma*sqrt(3/2)) was a bug: it broke q_new = q_trial -
        # 3*mu*dgamma, the defining property of dgamma. theta already encodes
        # the correct (verified) scaling, so build s_new from theta directly
        # instead of re-deriving it from n and risking the same mismatch.
        eps_p_new = eps_p_prev + dgamma * jnp.sqrt(1.5) * n
        alpha_new = alpha_prev + dgamma

        theta      = 1.0 - 3.0 * mu * dgamma / jnp.where(q_trial > 0.0, q_trial, 1.0)
        s_new      = theta * s_trial

        sigma_new = jnp.where(is_plastic, s_new + p * d, sigma_trial)
        eps_p_new = jnp.where(is_plastic, eps_p_new, eps_p_prev)
        alpha_new = jnp.where(is_plastic, alpha_new, alpha_prev)

        # ── algorithmic (consistent) tangent ─────────────────────────────
        # de Souza Neto, Peric & Owen, Box 8.1.
        I_xI  = jnp.einsum('ij,kl->ijkl', d, d)
        I_sym = 0.5 * (jnp.einsum('ik,jl->ijkl', d, d) + jnp.einsum('il,jk->ijkl', d, d))
        I_dev = I_sym - (1.0 / 3.0) * I_xI
        NxN   = jnp.einsum('ij,kl->ijkl', n, n)

        theta_bar  = 3.0 * mu / (3.0 * mu + H) - (1.0 - theta)

        C_elastic  = lam * I_xI + 2.0 * mu * I_sym
        C_plastic  = K * I_xI + 2.0 * mu * theta * I_dev - 2.0 * mu * theta_bar * NxN
        C_algo     = jnp.where(is_plastic, C_plastic, C_elastic)

        return sigma_new, eps_p_new, alpha_new, C_algo


def j2_return_map_field(
    material:    J2Plasticity,
    eps_field:      jnp.ndarray,
    eps_p_prev_field: jnp.ndarray,
    alpha_prev_field: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Per-voxel radial return, vmapped -- the field-level entry point.

    Parameters
    ----------
    material          : J2Plasticity
    eps_field         : (3, 3, Nv)
    eps_p_prev_field  : (3, 3, Nv)
    alpha_prev_field  : (Nv,)

    Returns
    -------
    sigma_field     : (3, 3, Nv)
    eps_p_new_field : (3, 3, Nv)
    alpha_new_field : (Nv,)
    C_algo_field    : (3, 3, 3, 3, Nv)
    """
    eps_vox   = eps_field.transpose(2, 0, 1)          # (Nv, 3, 3)
    eps_p_vox = eps_p_prev_field.transpose(2, 0, 1)   # (Nv, 3, 3)

    sigma_vox, eps_p_new_vox, alpha_new_field, C_algo_vox = jax.vmap(material.radial_return)(
        eps_vox, eps_p_vox, alpha_prev_field
    )

    sigma_field     = sigma_vox.transpose(1, 2, 0)               # (3, 3, Nv)
    eps_p_new_field = eps_p_new_vox.transpose(1, 2, 0)           # (3, 3, Nv)
    C_algo_field    = C_algo_vox.transpose(1, 2, 3, 4, 0)        # (3, 3, 3, 3, Nv)

    return sigma_field, eps_p_new_field, alpha_new_field, C_algo_field
