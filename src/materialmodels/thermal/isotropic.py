"""Isotropic thermal conductivity. Kept as (3, 3): Voigt form only pays off for rank-4 tensors."""

import jax.numpy as jnp

from materialmodels.base import ConductivityModel


class ThermalConductivityIsotropic(ConductivityModel):
    """
    Isotropic conductivity ``K = k I``.

    Parameters
    ----------
    k    : thermal conductivity
    name : display label
    """

    def __init__(self, k: float, name: str = ""):
        self.k    = float(k)
        self.name = name

    def conductivity_tensor(self) -> jnp.ndarray:
        """(3, 3) conductivity ``K_ij``."""
        return self.k * jnp.eye(3)

    def flux_field(self, grad_T: jnp.ndarray) -> jnp.ndarray:
        """Fourier's law ``q = -K grad T``, (3, Nv) -> (3, Nv)."""
        return -jnp.einsum('ij,jm->im', self.conductivity_tensor(), grad_T)

    def __repr__(self) -> str:
        tag = f" ({self.name})" if self.name else ""
        return f"ThermalConductivityIsotropic{tag}: k={self.k:.3g}"
