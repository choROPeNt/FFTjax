"""
Transversely isotropic (fibre-reinforced) thermal conductivity material model.

Only 2 independent conductivity constants, unlike the elastic case's 5:
a rank-2 conductivity tensor's transverse plane is isotropic with nothing
left to independently constrain once k_T is fixed -- there is no analogue of
a Poisson-ratio-style coupling term or a "shear" conductivity. Mirrors
materialmodels.elastic.transverse_isotropic.TransverseIsotropic's
fiber_dir/rotation machinery directly: materialmodels.tensors.
rotation_from_direction is rank-agnostic (just builds R from a direction)
and is reused verbatim; the actual rotation is a plain rank-2 similarity
transform K' = R K R^T, not materialmodels.tensors.rotate_tensor4
(rank-4-specific, not applicable here).
"""

import jax
import jax.numpy as jnp

from materialmodels.base import ConductivityModel
from materialmodels.tensors import rotation_from_direction


def _rotate_tensor2(R: jnp.ndarray, K: jnp.ndarray) -> jnp.ndarray:
    """
    Rotate a 2nd-order tensor from a local frame to the global frame:
    K_global_ij = R_ia R_jb K_local_ab. materialmodels.tensors only has
    rotate_tensor4 (rank-4-specific, for the elastic stiffness); a rank-2
    rotation is this one-line similarity transform, not worth a shared
    helper of its own.
    """
    return jnp.einsum('ia,ab,jb->ij', R, K, R)


class ThermalConductivityTransverseIsotropic(ConductivityModel):
    """
    Transversely isotropic (fibre-reinforced) thermal conductivity.

    The *reference* fibre axis is local **Z = [0, 0, 1]** -- same convention
    as ``materialmodels.elastic.transverse_isotropic.TransverseIsotropic``.
    ``fiber_dir`` (constructor arg, default Z i.e. no rotation) is this
    material's own fixed orientation in the global frame --
    ``conductivity_tensor()`` always returns the tensor already rotated into
    it, so ``materialmodels.assembly.assemble_K_field`` (or any other caller
    that just calls ``.conductivity_tensor()``) gets a correctly oriented
    per-phase conductivity automatically, with no special-case rotation
    plumbing needed at the assembly layer. ``conductivity_tensor_rotated(fiber_dir)``
    and ``conductivity_field_oriented(orientations)`` are a different pair:
    they rotate the *reference* tensor by an explicitly given direction (one,
    or a per-voxel field), overriding this material's own stored
    ``fiber_dir`` rather than composing with it.

    2 independent constants:

    k_L : conductivity along the fibre (longitudinal)
    k_T : conductivity perpendicular to the fibre (transverse -- isotropic
          in that plane; e.g. a carbon fibre's k_L can be one to three
          orders of magnitude above its k_T, while E-glass is close to
          isotropic (k_L ~= k_T) and is equally well represented by
          ``materialmodels.thermal.isotropic.ThermalConductivityIsotropic``
          -- this model earns its keep once k_L != k_T actually matters)

    fiber_dir : (3,) unit vector (global frame), default None -> Z = [0, 0, 1]
                (identity rotation, i.e. conductivity_tensor() then returns
                exactly the reference-frame tensor). Not required to be
                pre-normalized -- rotation_from_direction normalizes
                internally.

                Also accepts a per-voxel field, (3, Nv) -- one direction per
                voxel, spanning the WHOLE grid (e.g. straight from
                utils.io.reader.read_vtu's orientations output), not just
                this material's own phase -- in which case
                conductivity_tensor() returns the (3, 3, Nv) rotated FIELD
                instead of a single (3, 3) tensor (vmapped internally, same
                machinery as conductivity_field_oriented).
                materialmodels.assembly.assemble_K_field detects this
                automatically and mixes it correctly with other,
                constant-tensor materials in the same materials: list.
    """

    def __init__(
        self,
        k_L: float,
        k_T: float,
        fiber_dir: jnp.ndarray | None = None,
        name: str = "",
    ):
        self.k_L = float(k_L)
        self.k_T = float(k_T)
        self.fiber_dir = (jnp.array([0.0, 0.0, 1.0]) if fiber_dir is None
                           else jnp.asarray(fiber_dir, dtype=float))
        self.name = name

    def _conductivity_tensor_reference(self) -> jnp.ndarray:
        """(3, 3) conductivity in the reference frame (fibre along Z), before
        this material's own ``fiber_dir`` is applied."""
        return jnp.diag(jnp.array([self.k_T, self.k_T, self.k_L]))

    def conductivity_tensor(self) -> jnp.ndarray:
        """
        Conductivity rotated into this material's own ``fiber_dir`` (default
        Z, i.e. identity rotation) -- the ConductivityModel-required
        no-argument method.

        Shape follows ``fiber_dir``: ``(3, 3)`` for a single direction (the
        common case), or ``(3, 3, Nv)`` if ``fiber_dir`` was given as a
        per-voxel field -- ``assemble_K_field`` handles both shapes
        automatically, including mixed with other materials.
        """
        if self.fiber_dir.ndim > 1:
            return self.conductivity_field_oriented(self.fiber_dir)
        R = rotation_from_direction(self.fiber_dir)
        return _rotate_tensor2(R, self._conductivity_tensor_reference())

    def conductivity_tensor_rotated(self, fiber_dir: jnp.ndarray) -> jnp.ndarray:
        """
        (3, 3) conductivity rotated so the fibre axis aligns with an
        explicitly given ``fiber_dir`` in the global frame -- overrides this
        material's own stored ``fiber_dir`` rather than composing with it
        (rotates the *reference* tensor, not ``conductivity_tensor()``'s
        already-rotated one). For a one-off direction different from this
        material's default; see ``conductivity_field_oriented`` for a
        per-voxel orientation field instead of a single direction.

        Parameters
        ----------
        fiber_dir : (3,) unit vector (global frame)
        """
        R = rotation_from_direction(jnp.asarray(fiber_dir, float))
        return _rotate_tensor2(R, self._conductivity_tensor_reference())

    def conductivity_field_oriented(self, orientations: jnp.ndarray) -> jnp.ndarray:
        """
        Per-voxel rotated conductivity field, vmapped over a fibre
        orientation field -- the field-valued counterpart of
        ``conductivity_tensor_rotated`` (one direction) for a spatially
        varying orientation (e.g. from ``utils.io.reader.read_vtu``'s
        ``orientations`` output). Like ``conductivity_tensor_rotated``, this
        overrides this material's own stored ``fiber_dir`` rather than
        composing with it.

        The reference-frame tensor is computed once and shared across every
        voxel; only the rotation itself (``rotation_from_direction`` +
        ``_rotate_tensor2``, both vmap-safe) runs per-voxel.

        Parameters
        ----------
        orientations : (3, Nv)  unit fibre direction per voxel (global frame)

        Returns
        -------
        K_field : (3, 3, Nv)
        """
        K_ref = self._conductivity_tensor_reference()

        def _one(d):
            R = rotation_from_direction(d)
            return _rotate_tensor2(R, K_ref)

        K_vox = jax.vmap(_one, in_axes=1)(orientations)   # (Nv, 3, 3)
        return jnp.moveaxis(K_vox, 0, -1)                   # (3, 3, Nv)

    def __repr__(self) -> str:
        tag = f" ({self.name})" if self.name else ""
        return f"ThermalConductivityTransverseIsotropic{tag}: k_L={self.k_L:.3g}  k_T={self.k_T:.3g}"
