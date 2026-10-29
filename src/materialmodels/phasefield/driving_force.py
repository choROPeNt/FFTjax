"""Crack driving force psi+ (Amor split, per voxel) and the history variable H."""

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)

from collections.abc import Sequence

import jax
import jax.numpy as jnp

from materialmodels.base import ConstitutiveModel
from materialmodels.phasefield.splits import amor_split
from materialmodels.tensors import isotropic_equivalent_lame


def lame_field(materials: Sequence[ConstitutiveModel], phase: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Per-voxel Lame parameters for the Amor split, gathered by phase.

    Uses the isotropic projection of each material's elastic tensor: exact for
    isotropic materials, an approximation for anisotropic ones (the Amor split
    is defined for isotropic elasticity only).

    Returns
    -------
    lam_vox, mu_vox : (Nv,)
    """
    lam_mu = [isotropic_equivalent_lame(m.elastic_stiffness_tensor()) for m in materials]
    lam_stack = jnp.array([lm[0] for lm in lam_mu])
    mu_stack = jnp.array([lm[1] for lm in lam_mu])
    return lam_stack[phase], mu_stack[phase]


def strain_energy_amor_split(
    eps_loc: jnp.ndarray,
    lam_vox: jnp.ndarray,
    mu_vox: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    ``amor_split`` per voxel: (3, 3, Nv), (Nv,), (Nv,) -> psi_pos, psi_neg (Nv,).

    psi_pos + psi_neg is the elastic energy; only psi_pos drives cracking.
    """
    eps_vox = eps_loc.transpose(2, 0, 1)
    psi_pos, psi_neg = jax.vmap(amor_split)(eps_vox, lam_vox, mu_vox)
    return psi_pos, psi_neg


def update_history_hybrid(
    H_prev: jnp.ndarray,
    psi_pos: jnp.ndarray,
    d_prev: jnp.ndarray,
    d_thres: float = 0.95,
) -> jnp.ndarray:
    """
    Hybrid irreversibility (Steinke & Kaliske 2019; Schneider & Kaestner 2025):

        H = psi+                 if d_prev < d_thres
        H = max(H_prev, psi+)    otherwise

    Locking H only once a point has cracked lets the process zone relax and
    avoids the over-wide zone of a pure damage-like lock.

    Returns
    -------
    H : (Nv,)
    """
    return jnp.where(d_prev >= d_thres, jnp.maximum(H_prev, psi_pos), psi_pos)
