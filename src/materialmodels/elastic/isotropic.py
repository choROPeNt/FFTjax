"""
Isotropic linear elastic constitutive model.

Voigt index convention (Abaqus order):
    0 = 11,  1 = 22,  2 = 33,  3 = 12,  4 = 13,  5 = 23

All Voigt representations use *tensor* shear components (ε₁₂, not γ₁₂ = 2ε₁₂),
consistent with ``post.fields.to_voigt`` and the σ = C:ε einsum in the FFT solver
(see mat_models/elastic.py's module docstring for the full convention note --
this is a port of that module's ``LinearElasticIsotropic`` onto the
``ConstitutiveModel`` ABC, not a reimplementation; verified bit-identical in
``test/test_materialmodels_elastic_isotropic.py``).
"""

import jax.numpy as jnp
import numpy as np

from materialmodels.base import ConstitutiveModel


class LinearElasticIsotropic(ConstitutiveModel):
    """
    Isotropic linear elastic constitutive model.

    Parameters
    ----------
    E    : float  Young's modulus (any consistent unit, e.g. MPa)
    nu   : float  Poisson's ratio  (−1 < ν < 0.5)
    name : str    Optional label for display.
    """

    def __init__(self, E: float, nu: float, name: str = ""):
        self.E    = float(E)
        self.nu   = float(nu)
        self.name = name
        self.lam  = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        self.mu   = E / (2.0 * (1.0 + nu))

    # ------------------------------------------------------------------
    # Stiffness representations
    # ------------------------------------------------------------------

    def stiffness_tensor(self) -> jnp.ndarray:
        """
        Full 4th-order stiffness tensor C_ijkl, shape (3, 3, 3, 3).

        Used directly by the FFT solver::

            sigma = jnp.einsum('ijkl,kl->ij', C, eps)
        """
        d = jnp.eye(3)
        return (self.lam * jnp.einsum('ij,kl->ijkl', d, d)
                + self.mu * (jnp.einsum('ik,jl->ijkl', d, d)
                             + jnp.einsum('il,jk->ijkl', d, d)))

    def stiffness_voigt(self, engineering: bool = False) -> jnp.ndarray:
        """
        6×6 Voigt stiffness matrix C_IJ, shape (6, 6).

        Parameters
        ----------
        engineering : bool
            False (default) — tensor shear convention (ε₁₂, shear block = 2μ),
            compatible with ``post.fields.to_voigt`` and the FFT solver output.
            True — engineering / Abaqus-UMAT convention (γ₁₂ = 2ε₁₂, shear block = μ).
        """
        lam, mu = self.lam, self.mu
        shear_factor = mu if engineering else 2.0 * mu
        C = np.zeros((6, 6))
        C[:3, :3] = lam
        C[0, 0] += 2 * mu;  C[1, 1] += 2 * mu;  C[2, 2] += 2 * mu
        C[3, 3] = shear_factor
        C[4, 4] = shear_factor
        C[5, 5] = shear_factor
        return jnp.array(C)

    def stress_voigt(self, eps_voigt: jnp.ndarray, engineering: bool = False) -> jnp.ndarray:
        """Compute Voigt stress from Voigt strain (..., 6) → (..., 6)."""
        return eps_voigt @ self.stiffness_voigt(engineering=engineering).T

    def stress_field(self, eps: jnp.ndarray) -> jnp.ndarray:
        """Compute stress from full-tensor strain field (3,3,Nv) → (3,3,Nv)."""
        return jnp.einsum('ijkl,klm->ijm', self.stiffness_tensor(), eps)

    @property
    def bulk_modulus(self) -> float:
        return self.E / (3.0 * (1.0 - 2.0 * self.nu))

    @property
    def shear_modulus(self) -> float:
        return self.mu

    def __repr__(self) -> str:
        tag = f" ({self.name})" if self.name else ""
        return (f"LinearElasticIsotropic{tag}: "
                f"E={self.E:.3g}, nu={self.nu:.3g}, "
                f"lam={self.lam:.3g}, mu={self.mu:.3g}")
