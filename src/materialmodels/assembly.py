"""Per-voxel property fields from a materials list and a phase-index field."""

from collections.abc import Sequence

import jax.numpy as jnp

from materialmodels.base import ConductivityModel, ConstitutiveModel
from materialmodels.phasefield.degradation import degradation_at2
from operators.fft_distributed import choose_device_count, shard_x_slabs, x_slab_shard_map, x_slab_spec


def assemble_C_field(
    materials: Sequence[ConstitutiveModel],
    phase: jnp.ndarray,
    n: tuple[int, ...] | None = None,
    n_devices: int | None = None,
) -> jnp.ndarray:
    """
    Per-voxel stiffness from a sharp phase assignment.

    Constant tensors are gathered by phase; a material with a per-voxel field
    (e.g. TransverseIsotropic with per-voxel ``fiber_dir``) contributes it at
    its own voxels. Nonlinear materials contribute their elastic tensor (see
    ``assemble_local_update`` for the tangent). Given ``n``, the field is built
    directly as x-slabs across devices: it is the largest per-voxel array.

    Parameters
    ----------
    materials : ConstitutiveModel per phase (0-based)
    phase     : (Nv,) int phase index
    n         : grid shape, enables the sharded build (None = single device)
    n_devices : device cap (None = all)

    Returns
    -------
    C_field : (3, 3, 3, 3, Nv), sharded as x-slabs on the multi-device path
    """
    if n is not None:
        resolved_n_devices = choose_device_count(n, n_devices)
        if resolved_n_devices > 1:
            return _assemble_C_field_sharded(materials, phase, resolved_n_devices)
    return _assemble_C_field_local(materials, phase)


def _assemble_C_field_local(
    materials: Sequence[ConstitutiveModel],
    phase: jnp.ndarray,
) -> jnp.ndarray:
    """Single-device assembly; also each device's slab in the sharded path."""
    C_per_material = [m.elastic_stiffness_tensor() for m in materials]

    if all(C.ndim == 4 for C in C_per_material):
        C_stack = jnp.stack(C_per_material, axis=-1)
        return C_stack[..., phase]

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


def _assemble_C_field_sharded(
    materials: Sequence[ConstitutiveModel],
    phase: jnp.ndarray,
    n_devices: int,
) -> jnp.ndarray:
    """
    ``C_field`` built per x-slab under ``shard_map`` and left sharded.

    A per-voxel material (``fiber_dir.ndim > 1`` plus ``stiffness_field_oriented``)
    has its orientation field sliced first, so the costly rotation is sharded too.
    """
    phase_sharded = shard_x_slabs(phase, n_devices)

    field_material_idx = [
        i for i, m in enumerate(materials)
        if getattr(m, "fiber_dir", None) is not None and m.fiber_dir.ndim > 1
    ]
    field_sharded = tuple(
        shard_x_slabs(materials[i].fiber_dir, n_devices) for i in field_material_idx
    )

    # Computed eagerly: the reference tensor goes through plain numpy
    # (voigt_to_tensor4), which fails on tracers inside shard_map.
    C_ref_by_idx = {
        i: materials[i].stiffness_tensor_rotated(jnp.array([0.0, 0.0, 1.0]))
        for i in field_material_idx
    }

    def local(phase_local, field_locals):
        field_by_idx = dict(zip(field_material_idx, field_locals))
        local_materials = [
            _LocalFieldMaterial(materials[i], field_by_idx[i], C_ref_by_idx[i])
            if i in field_by_idx else materials[i]
            for i in range(len(materials))
        ]
        return _assemble_C_field_local(local_materials, phase_local)

    return x_slab_shard_map(
        local, n_devices,
        in_specs=(x_slab_spec(1), tuple(x_slab_spec(f.ndim) for f in field_sharded)),
        out_specs=x_slab_spec(5),
    )(phase_sharded, field_sharded)


class _LocalFieldMaterial:
    """Per-slab stand-in for a per-voxel material: stiffness from the local
    orientation slice; every other attribute is passed through."""

    def __init__(self, material, field_local: jnp.ndarray, C_ref: jnp.ndarray):
        self._material = material
        self._field_local = field_local
        self._C_ref = C_ref

    def elastic_stiffness_tensor(self) -> jnp.ndarray:
        return self._material.stiffness_field_oriented(self._field_local, C_ref=self._C_ref)

    def __getattr__(self, name):
        return getattr(self._material, name)


def assemble_K_field(
    materials: Sequence[ConductivityModel],
    phase: jnp.ndarray,
) -> jnp.ndarray:
    """
    Per-voxel conductivity from a sharp phase assignment (assemble_C_field for rank 2).

    Parameters
    ----------
    materials : ConductivityModel per phase (0-based)
    phase     : (Nv,) int phase index

    Returns
    -------
    K_field : (3, 3, Nv)
    """
    K_per_material = [m.conductivity_tensor() for m in materials]

    if all(K.ndim == 2 for K in K_per_material):
        K_stack = jnp.stack(K_per_material, axis=-1)
        return K_stack[..., phase]

    Nv = phase.shape[0]
    K_field = jnp.zeros((3, 3, Nv), dtype=K_per_material[0].dtype)
    for i, K_i in enumerate(K_per_material):
        if K_i.ndim == 2:
            K_i_field = jnp.broadcast_to(K_i[..., None], (3, 3, Nv))
        else:
            if K_i.shape[-1] != Nv:
                raise ValueError(
                    f"materials[{i}] ({materials[i]}) returned a per-voxel "
                    f"conductivity field spanning {K_i.shape[-1]} voxels, but the "
                    f"grid has Nv={Nv}"
                )
            K_i_field = K_i
        K_field = jnp.where(phase == i, K_i_field, K_field)
    return K_field


def describe_materials(materials: Sequence[ConstitutiveModel | ConductivityModel]) -> None:
    """Print each material with its phase index."""
    for i, m in enumerate(materials):
        print(f"phase {i}: {m}")


def assemble_local_update(materials: Sequence[ConstitutiveModel], phase: jnp.ndarray):
    """
    ``local_update(eps_field, state) -> (sigma, C_tan, new_state)`` for
    ``solve_displacement_based_nonlinear``.

    Stateful materials are those with ``stress_and_tangent_field(eps,
    eps_p, alpha)`` (e.g. J2Plasticity, DruckerPrager); the rest are elastic.
    One shared ``(eps_p, alpha)`` state spans the grid; each material updates
    only its own phase's voxels.

    Parameters
    ----------
    materials : material per phase (0-based), any mix of stateless and stateful
    phase     : (Nv,) int phase index

    Returns
    -------
    local_update : callable
    state0       : zero ``(eps_p_field, alpha_field)``
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
                C_i = m.elastic_stiffness_tensor()
                sigma_i = jnp.einsum("ijkl,klm->ijm", C_i, eps_field)
                C_i = jnp.broadcast_to(C_i[..., None], (3, 3, 3, 3, Nv))
            sigma = jnp.where(mask, sigma_i, sigma)
            C_tan = jnp.where(mask, C_i, C_tan)

        return sigma, C_tan, (eps_p_out, alpha_out)

    state0 = (jnp.zeros((3, 3, Nv)), jnp.zeros(Nv))
    return local_update, state0


def assemble_pff_local_update(materials: Sequence[ConstitutiveModel], phase: jnp.ndarray):
    """
    ``local_update(eps_field, d_field) -> (sigma, C_tan)`` for the phase-field staggered loop.

    Materials with ``psi_split`` (PhaseFieldIsotropic) use their autodiff
    ``stress_and_tangent_field``; others use ``degradation_at2(d) * C``.

    Parameters
    ----------
    materials : material per phase (0-based)
    phase     : (Nv,) int phase index

    Returns
    -------
    local_update : callable, (3, 3, Nv) and (Nv,) -> (3, 3, Nv), (3, 3, 3, 3, Nv)
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
                C_elastic = m.elastic_stiffness_tensor()
                C_i = g[None, None, None, None, :] * C_elastic[..., None]
                sigma_i = jnp.einsum("ijklm,klm->ijm", C_i, eps_field)
            sigma = jnp.where(mask, sigma_i, sigma)
            C_tan = jnp.where(mask, C_i, C_tan)

        return sigma, C_tan

    return local_update
