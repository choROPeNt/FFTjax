"""
Assemble per-voxel property fields from a list of ConstitutiveModel
instances and a phase-index field. Generic over any ConstitutiveModel, not
elastic-specific -- the same pattern applies to a conductivity or diffusivity
field once those model types exist.
"""

import jax.numpy as jnp

from materialmodels.base import ConstitutiveModel


def assemble_C_field(
    materials: list[ConstitutiveModel],
    phase: jnp.ndarray,
) -> jnp.ndarray:
    """
    Per-voxel stiffness field from a hard (sharp-interface) phase assignment.

    Each voxel gets exactly one material's stiffness tensor, selected by its
    phase index -- no interpolation/blending at interfaces. For smooth or
    orientation-dependent assembly, a separate assembler is needed (not
    built yet, deferred with the rest of materialmodels/elastic/).

    Parameters
    ----------
    materials : list of ConstitutiveModel, indexed by phase (0-based)
    phase     : (Nv,) int   phase index per voxel

    Returns
    -------
    C_field : (3, 3, 3, 3, Nv)
    """
    C_stack = jnp.stack([m.stiffness_tensor() for m in materials], axis=-1)  # (3,3,3,3,n_mats)
    return C_stack[..., phase]  # (3,3,3,3,Nv) -- gather by phase index
