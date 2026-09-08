"""
Assemble per-voxel property fields from a list of ConstitutiveModel
instances and a phase-index field. Generic over any ConstitutiveModel, not
elastic-specific -- the same pattern applies to a conductivity or diffusivity
field once those model types exist.
"""

from collections.abc import Sequence

import jax.numpy as jnp

from materialmodels.base import ConstitutiveModel


def assemble_C_field(
    materials: Sequence[ConstitutiveModel],
    phase: jnp.ndarray,
) -> jnp.ndarray:
    """
    Per-voxel stiffness field from a hard (sharp-interface) phase assignment.

    Each voxel gets exactly one material's stiffness tensor, selected by its
    phase index -- no interpolation/blending at interfaces. Most materials'
    ``stiffness_tensor()`` returns one constant ``(3,3,3,3)`` tensor, in
    which case this just broadcasts it to every voxel of that material's
    phase (the fast path below -- stack + gather, no per-voxel work at all).
    A material with a per-voxel-varying stiffness (e.g.
    ``TransverseIsotropic`` constructed with a per-voxel ``fiber_dir`` --
    see its docstring) instead returns an already-rotated ``(3,3,3,3,Nv)``
    field spanning the *whole* grid; only its entries at that material's own
    phase voxels are actually used, so it's fine for the field's other
    entries (voxels belonging to a different phase) to hold anything.
    Mixing both kinds of material in one ``materials:`` list works
    transparently -- callers never need to know or care which case applies.

    Parameters
    ----------
    materials : list of ConstitutiveModel, indexed by phase (0-based) -- any
                mix of constant-tensor and per-voxel-field materials
    phase     : (Nv,) int   phase index per voxel

    Returns
    -------
    C_field : (3, 3, 3, 3, Nv)
    """
    C_per_material = [m.stiffness_tensor() for m in materials]

    if all(C.ndim == 4 for C in C_per_material):
        # fast path: every material is one constant tensor.
        C_stack = jnp.stack(C_per_material, axis=-1)  # (3,3,3,3,n_mats)
        return C_stack[..., phase]  # (3,3,3,3,Nv) -- gather by phase index

    # mixed path: at least one material returns an already-per-voxel field.
    Nv = phase.shape[0]
    C_field = jnp.zeros((3, 3, 3, 3, Nv), dtype=C_per_material[0].dtype)
    for i, C_i in enumerate(C_per_material):
        if C_i.ndim == 4:
            C_i_field = jnp.broadcast_to(C_i[..., None], (3, 3, 3, 3, Nv))
        else:
            if C_i.shape[-1] != Nv:
                raise ValueError(
                    f"materials[{i}] ({materials[i]}) returned a per-voxel "
                    f"stiffness field spanning {C_i.shape[-1]} voxels, but the "
                    f"grid has Nv={Nv} -- a per-voxel fiber_dir must span the "
                    "whole grid, not just this material's own phase"
                )
            C_i_field = C_i
        C_field = jnp.where(phase == i, C_i_field, C_field)
    return C_field


def describe_materials(materials: Sequence[ConstitutiveModel]) -> None:
    """Print each material next to the phase index assemble_C_field/solve_mechanics assign it."""
    for i, m in enumerate(materials):
        print(f"phase {i}: {m}")
