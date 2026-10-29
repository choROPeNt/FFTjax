"""
Strain-energy splits for phase-field fracture, shared by the field-level and
autodiff paths. Only Amor et al. (2009); a Miehe spectral split would go here.
"""

import jax.numpy as jnp


def amor_split(
    eps: jnp.ndarray,
    lam: float | jnp.ndarray,
    mu: float | jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Volumetric-deviatoric split of one voxel (Amor et al. 2009):

        K       = lam + 2 mu / 3
        psi_pos = K/2 <tr eps>_+^2 + mu eps_dev : eps_dev
        psi_neg = K/2 <tr eps>_-^2

    ``eps`` is symmetrized first: autodiff would otherwise treat eps_ij and
    eps_ji as independent.

    Returns
    -------
    psi_pos, psi_neg : scalars
    """
    eps = 0.5 * (eps + eps.T)
    K = lam + 2.0 * mu / 3.0
    tr_eps = jnp.trace(eps)
    eps_dev = eps - (tr_eps / 3.0) * jnp.eye(3)
    psi_pos = 0.5 * K * jnp.maximum(tr_eps, 0.0) ** 2 + mu * jnp.sum(eps_dev ** 2)
    psi_neg = 0.5 * K * jnp.minimum(tr_eps, 0.0) ** 2
    return psi_pos, psi_neg
