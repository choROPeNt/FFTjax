"""
Assemble per-voxel property fields from a list of ConstitutiveModel
instances and a phase-index field. Generic over any ConstitutiveModel, not
elastic-specific -- the same pattern applies to a conductivity or diffusivity
field once those model types exist.
"""

from collections.abc import Sequence

import jax
import jax.numpy as jnp

from materialmodels.base import ConductivityModel, ConstitutiveModel
from materialmodels.phasefield.degradation import degradation_at2
from operators.fft_distributed import choose_device_count, gather_x_slabs, split_x_slabs


def assemble_C_field(
    materials: Sequence[ConstitutiveModel],
    phase: jnp.ndarray,
    n: tuple[int, ...] | None = None,
    n_devices: int | None = None,
) -> jnp.ndarray:
    """
    Per-voxel stiffness field from a hard (sharp-interface) phase assignment.

    Each voxel gets exactly one material's stiffness tensor, selected by its
    phase index -- no interpolation/blending at interfaces. Most materials'
    ``elastic_stiffness_tensor()`` returns one constant ``(3,3,3,3)`` tensor,
    in which case this just broadcasts it to every voxel of that material's
    phase (the fast path below -- stack + gather, no per-voxel work at all).
    A material with a per-voxel-varying stiffness (e.g.
    ``TransverseIsotropic`` constructed with a per-voxel ``fiber_dir`` --
    see its docstring) instead returns an already-rotated ``(3,3,3,3,Nv)``
    field spanning the *whole* grid; only its entries at that material's own
    phase voxels are actually used, so it's fine for the field's other
    entries (voxels belonging to a different phase) to hold anything.
    Mixing both kinds of material in one ``materials:`` list works
    transparently -- callers never need to know or care which case applies.
    Uses the constant elastic tensor even for a nonlinear (plastic/
    phase-field) material -- see assemble_local_update/
    assemble_pff_local_update for the state-dependent tangent instead.

    The returned ``(3,3,3,3,Nv)`` tensor is this project's single largest
    per-voxel array (81 components), and for a per-voxel-oriented material
    (``TransverseIsotropic.stiffness_field_oriented``) the rotation that
    builds it is itself a real compute/memory cost on top of that -- both
    are what first exhausts one device's memory on a large grid, well
    before any solve begins (see benchmark/benchmark_3/elastic_solve.py's
    own investigation: an 18.5 GiB single allocation attempt at a
    600x600x85 grid, on every formulation equally, since every one of them
    calls this function the same unsharded way). ``n`` (grid shape) and
    ``n_devices`` opt into building it directly as x-slabs instead -- same
    auto-dispatch convention as every FFT-based solver in this project
    (``operators.fft_distributed.choose_device_count``): ``n=None``
    (default) is the original single-device behaviour, unchanged; given
    ``n``, this resolves the same device count the solver that will
    consume this ``C_field`` resolves from the same ``(n, n_devices)`` pair,
    so the two stay in lockstep with no value threaded between them.

    Parameters
    ----------
    materials : list of ConstitutiveModel, indexed by phase (0-based) -- any
                mix of constant-tensor and per-voxel-field materials
    phase     : (Nv,) int   phase index per voxel
    n         : grid shape (nx, ny, nz), or None for the single-device path
    n_devices : caps the device count auto-detected for domain decomposition
                when ``n`` is given -- see ``choose_device_count``

    Returns
    -------
    C_field : (3, 3, 3, 3, Nv)
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
    """The original single-device assembly -- also what each device runs on
    its own x-slab under ``_assemble_C_field_sharded``."""
    C_per_material = [m.elastic_stiffness_tensor() for m in materials]

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


def _assemble_C_field_sharded(
    materials: Sequence[ConstitutiveModel],
    phase: jnp.ndarray,
    n_devices: int,
) -> jnp.ndarray:
    """
    Build ``C_field`` directly as ``n_devices`` x-slabs under ``jax.pmap``,
    so the full ``(3,3,3,3,Nv)`` tensor -- and, for a per-voxel-oriented
    material, the rotation that builds its own full-grid field -- is never
    materialized on one device. Purely per-voxel work with no cross-voxel
    coupling at all (unlike the FFT-based solves elsewhere in this project),
    so this needs no ``all_to_all``/transpose, just an independent
    ``_assemble_C_field_local`` call per slab.

    A material with a per-voxel field is duck-typed exactly as
    ``TransverseIsotropic`` exposes it: a ``.fiber_dir`` attribute with
    ``ndim > 1`` (a per-voxel orientation field spanning the whole grid,
    see that class's docstring) paired with a ``.stiffness_field_oriented(
    orientations)`` method taking an explicit orientation array rather than
    always reading ``self.fiber_dir``. That field is sliced to the local
    slab *before* the rotation runs, so the rotation itself is genuinely
    sharded too, not just the final per-phase gather -- the rotation is the
    actual cost ``stiffness_field_oriented``'s own docstring warns about at
    tens-of-millions of voxels, not the cheap gather a constant-tensor
    material needs. A constant-tensor material needs no slicing at all: its
    own ``elastic_stiffness_tensor()`` is the same tiny array on every
    device, recomputed independently by closure capture rather than passed
    through ``pmap``.
    """
    phase_sharded = split_x_slabs(phase, n_devices)  # (n_devices, Nv_local)

    field_material_idx = [
        i for i, m in enumerate(materials)
        if getattr(m, "fiber_dir", None) is not None and m.fiber_dir.ndim > 1
    ]
    field_sharded = tuple(
        split_x_slabs(materials[i].fiber_dir, n_devices) for i in field_material_idx
    )

    # The reference-frame tensor each per-voxel material's own
    # stiffness_field_oriented would otherwise recompute internally
    # (materials[i]._stiffness_tensor_reference()) goes through plain numpy
    # (materialmodels.tensors.voigt_to_tensor4 -- see that module's
    # docstring), which raises TracerArrayConversionError if recomputed
    # fresh inside the pmap trace below. Computed eagerly here instead, via
    # the public stiffness_tensor_rotated([0, 0, 1]) (the documented
    # reference axis -- see TransverseIsotropic's own docstring), and
    # passed into stiffness_field_oriented's C_ref parameter per-device, so
    # only the (jax.numpy, trace-safe) rotation itself runs under pmap.
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

    out_sharded = jax.pmap(local)(phase_sharded, field_sharded)
    return gather_x_slabs(out_sharded)


class _LocalFieldMaterial:
    """Thin per-slab stand-in for a per-voxel-oriented material (e.g.
    ``TransverseIsotropic``) inside ``_assemble_C_field_sharded``:
    ``elastic_stiffness_tensor()`` calls the wrapped material's own
    ``stiffness_field_oriented`` on the already-local-sliced orientation
    field and an eagerly-precomputed ``C_ref`` (see the comment above this
    class's construction site for why), instead of the wrapped material's
    full-grid ``self.fiber_dir`` -- everything else (``.k_res``, ``.Gc``,
    ``.name``, ...) is read straight through via ``__getattr__`` so this is
    transparent to any other caller convention (e.g.
    ``materialmodels.phasefield.degradation.k_res_field``) that doesn't
    care about orientation at all."""

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
    Per-voxel conductivity field from a hard (sharp-interface) phase
    assignment -- the ConductivityModel analogue of assemble_C_field (see
    its own docstring for the fast-path/mixed-path reasoning, identical here
    modulo tensor rank: (3,3) per material instead of (3,3,3,3)). No
    per-voxel-varying-material fast path is exercised by any thermal model
    yet (unlike TransverseIsotropic's per-voxel fiber_dir on the elastic
    side), but the mixed path is kept for parity -- an anisotropic
    conductivity model with its own per-voxel orientation would need no
    change here.

    Parameters
    ----------
    materials : list of ConductivityModel, indexed by phase (0-based)
    phase     : (Nv,) int   phase index per voxel

    Returns
    -------
    K_field : (3, 3, Nv)
    """
    K_per_material = [m.conductivity_tensor() for m in materials]

    if all(K.ndim == 2 for K in K_per_material):
        # fast path: every material is one constant tensor.
        K_stack = jnp.stack(K_per_material, axis=-1)  # (3,3,n_mats)
        return K_stack[..., phase]  # (3,3,Nv) -- gather by phase index

    # mixed path: at least one material returns an already-per-voxel field.
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
    """Print each material next to the phase index assemble_C_field/assemble_K_field assign it."""
    for i, m in enumerate(materials):
        print(f"phase {i}: {m}")


def assemble_local_update(materials: Sequence[ConstitutiveModel], phase: jnp.ndarray):
    """
    Build a ``local_update(eps_field, state) -> (sigma, C_tan, new_state)``
    callable for ``problems.mechanics.solve_displacement_based_nonlinear``
    from a per-phase materials list -- the stateful analogue of
    ``assemble_C_field``, generalizing the elastic-fiber/plastic-matrix
    ``local_update`` combinator from ``notebooks/mechanics/in-elastic_J2.ipynb`` to any
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
    ``degradation_at2(d) * elastic_stiffness_tensor()``, reproducing
    ``materialmodels.phasefield.degradation.degrade_stiffness_field``'s
    per-phase behavior exactly -- so a ``materials`` list can freely mix
    old-style and new-style materials.

    Parameters
    ----------
    materials : list of ConstitutiveModel, indexed by phase (0-based) -- any
                mix of PhaseFieldIsotropic (psi_split) and plain elastic
                (elastic_stiffness_tensor + k_res) materials
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
                C_elastic = m.elastic_stiffness_tensor()
                C_i = g[None, None, None, None, :] * C_elastic[..., None]
                sigma_i = jnp.einsum("ijklm,klm->ijm", C_i, eps_field)
            sigma = jnp.where(mask, sigma_i, sigma)
            C_tan = jnp.where(mask, C_i, C_tan)

        return sigma, C_tan

    return local_update
