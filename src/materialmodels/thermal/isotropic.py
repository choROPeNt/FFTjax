"""
Isotropic thermal conductivity constitutive model.

No Voigt form here, unlike materialmodels.elastic.isotropic -- Voigt
notation exists to exploit the minor/major symmetry of a rank-4 elastic
stiffness (81 entries down to 21, or 6 for isotropy) against a *symmetric*
strain/stress pair. A rank-2 conductivity tensor relating flux to gradient
has no such structure to compress (post.fields.to_voigt/from_voigt are
strain/stress-specific for the same reason -- see their own docstrings), so
this stays in its native (3, 3) form throughout, matching
materialmodels.base.ConductivityModel's contract directly.
"""

import jax.numpy as jnp

from materialmodels.base import ConductivityModel


class ThermalConductivityIsotropic(ConductivityModel):
    """
    Isotropic thermal conductivity.

    Parameters
    ----------
    k    : float  Thermal conductivity (any consistent unit, e.g. W/(mm·K)).
    name : str    Optional label for display.
    """

    def __init__(self, k: float, name: str = ""):
        self.k    = float(k)
        self.name = name

    def conductivity_tensor(self) -> jnp.ndarray:
        """
        Full 2nd-order conductivity tensor K_ij, shape (3, 3).

        Used directly by the FFT solver::

            q = -jnp.einsum('ij,jm->im', K, grad_T)
        """
        return self.k * jnp.eye(3)

    def flux_field(self, grad_T: jnp.ndarray) -> jnp.ndarray:
        """Fourier's law: q = -K : grad(T) from a gradient field (3,Nv) -> (3,Nv)."""
        return -jnp.einsum('ij,jm->im', self.conductivity_tensor(), grad_T)

    def __repr__(self) -> str:
        tag = f" ({self.name})" if self.name else ""
        return f"ThermalConductivityIsotropic{tag}: k={self.k:.3g}"
