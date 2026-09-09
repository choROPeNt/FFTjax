"""
Crack driving force for phase-field fracture: the tensile part ψ⁺ of the
elastic strain energy density, and the irreversibility law that turns ψ⁺
into a monotone history variable H.

The split itself (Amor et al. 2009 volumetric-deviatoric -- currently the
only one implemented; see materialmodels.phasefield.splits for where a
Miehe/spectral split would go) lives in materialmodels.phasefield.splits,
shared with materialmodels.phasefield.isotropic.PhaseFieldIsotropic's
autodiff path. This module only adds field-level (vmap) batching and the
irreversibility bookkeeping on top of it.
"""

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)

from collections.abc import Sequence

import jax
import jax.numpy as jnp

from materialmodels.base import ConstitutiveModel
from materialmodels.phasefield.splits import amor_split
from materialmodels.tensors import isotropic_equivalent_lame


def lame_field(materials: Sequence[ConstitutiveModel], phase: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Per-voxel Lamé constants for the Amor split, gathered by phase index --
    same hard (sharp-interface) assembly pattern as
    ``materialmodels.assembly.assemble_C_field``, but reduced to the scalar
    (λ, μ) pair the split needs instead of the full C tensor.

    Each material's (λ, μ) comes from ``isotropic_equivalent_lame`` on its
    own ``stiffness_tensor()`` -- exact for an isotropic material (recovers
    its own λ, μ bit-for-bit, verified in materialmodels.tensors), an
    isotropization for an anisotropic one (e.g. TransverseIsotropic).
    The Amor split itself is only defined for an isotropic elastic law, so
    an anisotropic phase's driving force is inherently approximate here --
    this is the same approximation problems.mechanics.solve_mechanics's
    lippmann_schwinger reference medium already makes for such materials,
    not a new one introduced by this function.

    Parameters
    ----------
    materials : list of ConstitutiveModel (any -- only needs .stiffness_tensor())
    phase     : (Nv,) int   phase index per voxel

    Returns
    -------
    lam_vox, mu_vox : (Nv,), (Nv,)
    """
    lam_mu = [isotropic_equivalent_lame(m.stiffness_tensor()) for m in materials]
    lam_stack = jnp.array([lm[0] for lm in lam_mu])
    mu_stack = jnp.array([lm[1] for lm in lam_mu])
    return lam_stack[phase], mu_stack[phase]


def strain_energy_amor_split(
    eps_loc: jnp.ndarray,
    lam_vox: jnp.ndarray,
    mu_vox: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Per-voxel volumetric-deviatoric energy split (Amor et al. 2009) --
    field-level vmap wrapper over materialmodels.phasefield.splits.
    amor_split, the shared single-voxel formula (see its docstring for the
    split itself). ψ⁺ + ψ⁻ reproduces the ordinary elastic energy exactly;
    only ψ⁺ drives crack growth -- a crack shouldn't heal or grow under
    compression.

    Parameters
    ----------
    eps_loc : (3, 3, Nv)   local strain, undegraded
    lam_vox : (Nv,)        Lamé λ per voxel -- use ``lame_field``
    mu_vox  : (Nv,)        shear modulus μ per voxel

    Returns
    -------
    psi_pos, psi_neg : (Nv,), (Nv,)
    """
    eps_vox = eps_loc.transpose(2, 0, 1)   # (Nv, 3, 3)
    psi_pos, psi_neg = jax.vmap(amor_split)(eps_vox, lam_vox, mu_vox)
    return psi_pos, psi_neg


def update_history_hybrid(
    H_prev: jnp.ndarray,
    psi_pos: jnp.ndarray,
    d_prev: jnp.ndarray,
    d_thres: float = 0.95,
) -> jnp.ndarray:
    """
    Hybrid damage-/crack-like irreversibility (Steinke & Kaliske 2019, as
    adopted by Schneider & Kästner 2025, doi:10.1111/ffe.14553).

    Pure damage-like irreversibility (H = max(H_prev, ψ⁺) everywhere) locks
    in the history field as soon as any damage nucleates, which over-widens
    the diffuse process zone. The hybrid formulation only enforces the
    monotone history lock once a point has effectively cracked (d ≥
    d_thres); below the threshold the driving force is left unrestricted so
    the pre-crack process zone can still relax:

        H = ψ⁺                     if d_prev < d_thres   (unrestricted)
        H = max(H_prev, ψ⁺)        if d_prev ≥ d_thres    (irreversible)

    Parameters
    ----------
    H_prev  : (Nv,)   history variable from the previous (staggered or
                       time) iteration
    psi_pos : (Nv,)   current tensile strain energy density ψ⁺
    d_prev  : (Nv,)   damage field used to gate the switch
    d_thres : float   phase-field threshold above which irreversibility is
                       enforced (default 0.95)

    Returns
    -------
    H : (Nv,)
    """
    return jnp.where(d_prev >= d_thres, jnp.maximum(H_prev, psi_pos), psi_pos)
