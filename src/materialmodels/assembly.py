"""
Assemble per-voxel property fields from a list of ConstitutiveModel
instances and a phase-index field. Generic over any ConstitutiveModel, not
elastic-specific -- the same pattern applies to a conductivity or diffusivity
field once those model types exist.
"""

from collections.abc import Sequence

import jax.numpy as jnp

from materialmodels.base import ConstitutiveModel
from materialmodels.phasefield.degradation import degradation_at2


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


def assemble_local_update(materials: Sequence[ConstitutiveModel], phase: jnp.ndarray):
    """
    Build a ``local_update(eps_field, state) -> (sigma, C_tan, new_state)``
    callable for ``problems.mechanics.solve_displacement_based_nonlinear``
    from a per-phase materials list -- the stateful analogue of
    ``assemble_C_field``, generalizing the elastic-fiber/plastic-matrix
    ``local_update`` combinator from ``notebooks/in-elastic_J2.ipynb`` to any
    number of phases and any mix of stateless (plain ``ConstitutiveModel``,
    e.g. ``LinearElasticIsotropic``) and stateful (duck-typed via a
    ``stress_and_tangent_field(eps_field, eps_p_field, alpha_field)``
    method, e.g. ``J2Plasticity`` or ``DruckerPrager``) materials. The
    duck-typing is deliberate: a new stateful model needs no edit here, it
    just has to supply that one method with those shapes.

    State is one shared ``(eps_p_field, alpha_field)`` pair spanning the
    whole grid, same shapes ``J2Plasticity.stress_and_tangent_field`` uses
    -- each stateful material only ever updates its own phase's voxels (via
    ``jnp.where``); a purely elastic phase leaves that portion of the state
    untouched. This works even with several distinct plastic phases, since
    every voxel belongs to exactly one phase and each material's own state
    update never touches another phase's voxels.

    Parameters
    ----------
    materials : list of ConstitutiveModel, indexed by phase (0-based) -- any
                mix of stateless and stateful (stress_and_tangent_field) materials
    phase     : (Nv,) int   phase index per voxel

    Returns
    -------
    local_update : callable, state0 : (eps_p_field, alpha_field) initialized to zero
    """
    Nv = phase.shape[0]

    def local_update(eps_field, state):
        eps_p_field, alpha_field = state
        sigma  = jnp.zeros((3, 3, Nv))
        C_tan  = jnp.zeros((3, 3, 3, 3, Nv))
        eps_p_out = eps_p_field
        alpha_out = alpha_field

        for i, m in enumerate(materials):
            mask = (phase == i)
            if hasattr(m, "stress_and_tangent_field"):
                sigma_i, C_i, (eps_p_i, alpha_i) = m.stress_and_tangent_field(
                    eps_field, eps_p_field, alpha_field
                )
                eps_p_out = jnp.where(mask, eps_p_i, eps_p_out)
                alpha_out = jnp.where(mask, alpha_i, alpha_out)
            else:
                C_i = m.stiffness_tensor()
                sigma_i = jnp.einsum("ijkl,klm->ijm", C_i, eps_field)
                C_i = jnp.broadcast_to(C_i[..., None], (3, 3, 3, 3, Nv))
            sigma = jnp.where(mask, sigma_i, sigma)
            C_tan = jnp.where(mask, C_i, C_tan)

        return sigma, C_tan, (eps_p_out, alpha_out)

    state0 = (jnp.zeros((3, 3, Nv)), jnp.zeros(Nv))
    return local_update, state0


def assemble_pff_local_update(materials: Sequence[ConstitutiveModel], phase: jnp.ndarray):
    """
    Build a ``local_update(eps_field, d_field) -> (sigma, C_tan)`` callable
    for ``problems.fracture``'s staggered loop -- the phase-field analogue of
    ``assemble_local_update``, for materials whose degraded stress/tangent
    depend on both strain and damage rather than damage alone.

    Duck-types on ``hasattr(m, "psi_split")`` (only
    ``materialmodels.phasefield.isotropic.PhaseFieldIsotropic``-style
    materials have it) to call their autodiff
    ``stress_and_tangent_field(eps_field, d_field)``, discarding the
    ``psi_pos`` it also returns -- the staggered loop still gets the driving
    force from ``materialmodels.phasefield.driving_force.
    strain_energy_amor_split`` directly (same formula, verified bit-for-bit
    in ``test/test_materialmodels_phasefield_isotropic.py``), so it isn't
    needed here. A material without ``psi_split`` (plain
    ``LinearElasticIsotropic``, no Amor split) falls back to
    ``degradation_at2(d) * stiffness_tensor()``, reproducing
    ``materialmodels.phasefield.degradation.degrade_stiffness_field``'s
    per-phase behavior exactly -- so a ``materials`` list can freely mix
    old-style and new-style materials.

    Parameters
    ----------
    materials : list of ConstitutiveModel, indexed by phase (0-based) -- any
                mix of PhaseFieldIsotropic (psi_split) and plain elastic
                (stiffness_tensor + k_res) materials
    phase     : (Nv,) int   phase index per voxel

    Returns
    -------
    local_update : callable(eps_field: (3,3,Nv), d_field: (Nv,)) ->
                    (sigma_field: (3,3,Nv), C_tan_field: (3,3,3,3,Nv))
    """
    Nv = phase.shape[0]

    def local_update(eps_field, d_field):
        sigma = jnp.zeros((3, 3, Nv))
        C_tan = jnp.zeros((3, 3, 3, 3, Nv))

        for i, m in enumerate(materials):
            mask = (phase == i)
            if hasattr(m, "psi_split"):
                sigma_i, C_i, _psi_pos_i = m.stress_and_tangent_field(eps_field, d_field)
            else:
                g = degradation_at2(d_field, k=m.k_res)
                C_elastic = m.stiffness_tensor()
                C_i = g[None, None, None, None, :] * C_elastic[..., None]
                sigma_i = jnp.einsum("ijklm,klm->ijm", C_i, eps_field)
            sigma = jnp.where(mask, sigma_i, sigma)
            C_tan = jnp.where(mask, C_i, C_tan)

        return sigma, C_tan

    return local_update
