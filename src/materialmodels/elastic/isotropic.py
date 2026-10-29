"""
Isotropic linear elastic model.

Voigt order 11, 22, 33, 12, 13, 23 with tensor shear strains (eps_12, not
gamma_12), as in post.fields.to_voigt.
"""

import jax.numpy as jnp
import numpy as np

from materialmodels.base import ConstitutiveModel


class LinearElasticIsotropic(ConstitutiveModel):
    """
    Isotropic linear elastic material.

    Parameters
    ----------
    E     : Young's modulus
    nu    : Poisson's ratio (-1 < nu < 0.5)
    name  : display label
    k_res : AT2 residual stiffness, fracture only; 1.0 makes the phase
            damage-immune
    Gc    : critical energy release rate, fracture only; no default because
            no toughness value is safe to assume
    """

    def __init__(self, E: float, nu: float, name: str = "", k_res: float = 1e-6,
                 Gc: float | None = None):
        self.E     = float(E)
        self.nu    = float(nu)
        self.name  = name
        self.k_res = float(k_res)
        self.Gc    = float(Gc) if Gc is not None else None
        self.lam   = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        self.mu    = E / (2.0 * (1.0 + nu))

    def elastic_stiffness_tensor(self) -> jnp.ndarray:
        """(3, 3, 3, 3) stiffness ``C_ijkl``; also the tangent."""
        d = jnp.eye(3)
        return (self.lam * jnp.einsum('ij,kl->ijkl', d, d)
                + self.mu * (jnp.einsum('ik,jl->ijkl', d, d)
                             + jnp.einsum('il,jk->ijkl', d, d)))

    def stiffness_voigt(self, engineering: bool = False) -> jnp.ndarray:
        """
        (6, 6) Voigt stiffness.

        engineering=False: tensor shear (shear block 2 mu, matches to_voigt);
        True: engineering shear (Abaqus, block mu).
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
        """Voigt stress from Voigt strain, (..., 6) -> (..., 6)."""
        return eps_voigt @ self.stiffness_voigt(engineering=engineering).T

    def stress_field(self, eps: jnp.ndarray) -> jnp.ndarray:
        """Stress from a (3, 3, Nv) strain field."""
        return jnp.einsum('ijkl,klm->ijm', self.elastic_stiffness_tensor(), eps)

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
