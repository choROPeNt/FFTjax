"""
Isotropic AT2 phase-field material: a LinearElasticIsotropic base plus the
Amor et al. (2009) tension/compression split and AT2 degradation, with the
degraded stress and tangent stiffness derived by automatic differentiation
of a single energy density rather than two independently hand-maintained
formulas.

Unifies what materialmodels.phasefield.driving_force's psi+/psi- split and
degradation.py's degrade_stiffness_field currently do separately:
degrade_stiffness_field applies g(d) to the *entire* stiffness tensor
uniformly, which does not reproduce the Amor split at the stress level (the
compressive volumetric branch gets degraded too, when physically it
shouldn't -- a crack doesn't heal or grow under compression). Here,
sigma = d(psi)/d(eps) and C_tan = d(sigma)/d(eps) both come from the same
psi(eps, d) = g(d)*psi_pos(eps) + psi_neg(eps), so the compressive branch
is structurally undegraded -- psi_neg never sees g(d) -- instead of
depending on two formulas staying in sync by hand.
"""

import jax
import jax.numpy as jnp

from materialmodels.elastic.isotropic import LinearElasticIsotropic
from materialmodels.phasefield.degradation import degradation_at2
from materialmodels.phasefield.splits import amor_split


class PhaseFieldIsotropic(LinearElasticIsotropic):
    """
    Same constructor as LinearElasticIsotropic (E, nu, name, k_res), except
    Gc is required here (not optional) -- a phase-field material with no
    critical energy release rate is a config mistake this class can catch
    immediately, rather than one that surfaces later, deep inside
    materialmodels.phasefield.degradation.Gc_field, the first time a real
    fracture solve actually needs it.

    d (the damage field) is never stored on this instance -- like
    plasticity's (eps_p, alpha), it's owned by the caller (the phase-field
    Helmholtz sub-solve in problems.fracture.solve_fracture) and passed in
    fresh each call. Unlike plasticity, there's no local state update here
    at all: stress_and_tangent is a plain function of the CURRENT (eps, d),
    nothing carried forward by this class between calls.
    """

    def __init__(self, E: float, nu: float, Gc: float, name: str = "", k_res: float = 1e-6):
        super().__init__(E, nu, name=name, k_res=k_res, Gc=Gc)

    def psi_split(self, eps: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Undegraded Amor volumetric-deviatoric split, one voxel: (3,3) ->
        (psi_pos, psi_neg) -- this class's own (lam, mu) plugged into the
        shared materialmodels.phasefield.splits.amor_split, the same
        function materialmodels.phasefield.driving_force.
        strain_energy_amor_split vmaps over the field-level API's
        (lam_vox, mu_vox); this is the sole place either path's formula
        lives, so the two can't drift apart. See amor_split's own docstring
        for the split itself and for why eps gets symmetrized first
        (required for stress_and_tangent's jax.jacfwd to come out correct).
        """
        return amor_split(eps, self.lam, self.mu)

    def psi(self, eps: jnp.ndarray, d: jnp.ndarray) -> jnp.ndarray:
        """Degraded energy density, one voxel: (3,3), scalar -> scalar.
        psi_neg has no d-dependence here -- the source of the compression
        fix, not a separate branch bolted on afterward."""
        psi_pos, psi_neg = self.psi_split(eps)
        return degradation_at2(d, k=self.k_res) * psi_pos + psi_neg

    def stress_and_tangent(
        self, eps: jnp.ndarray, d: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        One voxel: sigma = d(psi)/d(eps) via jax.grad, C_tan = d(sigma)/d(eps)
        via jax.jacfwd on that -- a Hessian of psi, one derivative order
        deeper than J2Plasticity's tangent (a potential-based model, not a
        return-mapping one). psi_pos (the Amor driving force) rides along
        as aux through both autodiff passes, so this is the only place that
        evaluates the split at all -- no redundant second pass to recover
        it, and no separate strain_energy_amor_split call needed upstream.

        Returns
        -------
        sigma   : (3,3)
        C_tan   : (3,3,3,3)
        psi_pos : ()      -- feed this straight into
                  materialmodels.phasefield.driving_force.update_history_hybrid
        """
        def _stress_fn(e):
            def _psi_fn(e2):
                psi_pos, psi_neg = self.psi_split(e2)
                return degradation_at2(d, k=self.k_res) * psi_pos + psi_neg, psi_pos

            sigma, psi_pos = jax.grad(_psi_fn, has_aux=True)(e)
            return sigma, (sigma, psi_pos)

        C_tan, (sigma, psi_pos) = jax.jacfwd(_stress_fn, has_aux=True)(eps)
        return sigma, C_tan, psi_pos

    def stress_and_tangent_field(
        self, eps_field: jnp.ndarray, d_field: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Per-voxel stress/tangent/driving-force field, vmapped over
        stress_and_tangent -- same vmap-over-the-trailing-voxel-axis
        pattern as J2Plasticity.stress_and_tangent_field.

        Parameters
        ----------
        eps_field : (3, 3, Nv)
        d_field   : (Nv,)

        Returns
        -------
        sigma_field : (3, 3, Nv)
        C_field     : (3, 3, 3, 3, Nv)
        psi_pos_field : (Nv,)
        """
        eps_vox = eps_field.transpose(2, 0, 1)  # (Nv, 3, 3)
        sigma_vox, C_vox, psi_pos_field = jax.vmap(self.stress_and_tangent)(eps_vox, d_field)
        sigma_field = sigma_vox.transpose(1, 2, 0)      # (3, 3, Nv)
        C_field     = C_vox.transpose(1, 2, 3, 4, 0)    # (3, 3, 3, 3, Nv)
        return sigma_field, C_field, psi_pos_field

    def __repr__(self) -> str:
        tag = f" ({self.name})" if self.name else ""
        return (f"PhaseFieldIsotropic{tag}: E={self.E:.3g}, nu={self.nu:.3g}, "
                f"Gc={self.Gc:.3g}, k_res={self.k_res:.3g}")
