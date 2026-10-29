"""
Isotropic AT2 phase-field material: stress and tangent by autodiff of one energy

    psi(eps, d) = g(d) psi_pos(eps) + psi_neg(eps)

so the compressive part is never degraded. Degrading the whole stiffness by
g(d) (degrade_stiffness_field) would degrade it too.
"""

import jax
import jax.numpy as jnp

from materialmodels.elastic.isotropic import LinearElasticIsotropic
from materialmodels.phasefield.degradation import degradation_at2
from materialmodels.phasefield.splits import amor_split


class PhaseFieldIsotropic(LinearElasticIsotropic):
    """
    LinearElasticIsotropic with Amor split and AT2 degradation; ``Gc`` is required.

    The damage ``d`` is passed in by the caller on every call; no state is stored.
    """

    def __init__(self, E: float, nu: float, Gc: float, name: str = "", k_res: float = 1e-6):
        super().__init__(E, nu, name=name, k_res=k_res, Gc=Gc)

    def psi_split(self, eps: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Undegraded ``(psi_pos, psi_neg)`` of one voxel (shared ``amor_split``)."""
        return amor_split(eps, self.lam, self.mu)

    def psi(self, eps: jnp.ndarray, d: jnp.ndarray) -> jnp.ndarray:
        """Degraded energy density of one voxel."""
        psi_pos, psi_neg = self.psi_split(eps)
        return degradation_at2(d, k=self.k_res) * psi_pos + psi_neg

    def stress_and_tangent(
        self, eps: jnp.ndarray, d: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        One voxel: ``sigma = dpsi/deps`` (grad) and ``C_tan = dsigma/deps`` (jacfwd).

        Returns
        -------
        sigma   : (3, 3)
        C_tan   : (3, 3, 3, 3)
        psi_pos : scalar driving force, for update_history_hybrid
        """
        def _stress_fn(e):
            def _psi_fn(e2):
                psi_pos, psi_neg = self.psi_split(e2)
                return degradation_at2(d, k=self.k_res) * psi_pos + psi_neg, psi_pos

            sigma, psi_pos = jax.grad(_psi_fn, has_aux=True)(e)
            return sigma, (sigma, psi_pos)

        C_tan, (sigma, psi_pos) = jax.jacfwd(_stress_fn, has_aux=True)(eps)
        return sigma, C_tan, psi_pos

    def stress(self, eps: jnp.ndarray, d: jnp.ndarray) -> jnp.ndarray:
        """``sigma`` of one voxel."""
        sigma, _, _ = self.stress_and_tangent(eps, d)
        return sigma

    def stiffness_tensor(self, eps: jnp.ndarray, d: jnp.ndarray) -> jnp.ndarray:
        """(3, 3, 3, 3) degraded tangent of one voxel."""
        _, C_tan, _ = self.stress_and_tangent(eps, d)
        return C_tan

    def stress_and_tangent_field(
        self, eps_field: jnp.ndarray, d_field: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        ``stress_and_tangent`` per voxel.

        Parameters
        ----------
        eps_field : (3, 3, Nv)
        d_field   : (Nv,)

        Returns
        -------
        sigma_field   : (3, 3, Nv)
        C_field       : (3, 3, 3, 3, Nv)
        psi_pos_field : (Nv,)
        """
        eps_vox = eps_field.transpose(2, 0, 1)
        sigma_vox, C_vox, psi_pos_field = jax.vmap(self.stress_and_tangent)(eps_vox, d_field)
        sigma_field = sigma_vox.transpose(1, 2, 0)
        C_field     = C_vox.transpose(1, 2, 3, 4, 0)
        return sigma_field, C_field, psi_pos_field

    def __repr__(self) -> str:
        tag = f" ({self.name})" if self.name else ""
        return (f"PhaseFieldIsotropic{tag}: E={self.E:.3g}, nu={self.nu:.3g}, "
                f"Gc={self.Gc:.3g}, k_res={self.k_res:.3g}")
