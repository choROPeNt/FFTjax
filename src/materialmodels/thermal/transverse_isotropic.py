"""Transversely isotropic (fiber-reinforced) thermal conductivity; rotation K' = R K R^T."""

import jax
import jax.numpy as jnp

from materialmodels.base import ConductivityModel
from materialmodels.tensors import rotation_from_direction


def _rotate_tensor2(R: jnp.ndarray, K: jnp.ndarray) -> jnp.ndarray:
    """``K_ij = R_ia R_jb K_ab`` (local to global frame)."""
    return jnp.einsum('ia,ab,jb->ij', R, K, R)


class ThermalConductivityTransverseIsotropic(ConductivityModel):
    """
    Transversely isotropic conductivity, reference fiber axis Z (as TransverseIsotropic).

    ``conductivity_tensor()`` is rotated into ``fiber_dir``;
    ``conductivity_tensor_rotated``/``conductivity_field_oriented`` rotate
    the reference tensor to an explicit direction instead.

    Parameters
    ----------
    k_L, k_T  : conductivity along and across the fiber
    fiber_dir : (3,) direction (default Z), or (3, Nv) per-voxel field over
                the whole grid, which makes conductivity_tensor() (3, 3, Nv)
    name      : display label
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
        """(3, 3) conductivity with the fiber along Z."""
        return jnp.diag(jnp.array([self.k_T, self.k_T, self.k_L]))

    def conductivity_tensor(self) -> jnp.ndarray:
        """Conductivity rotated into ``fiber_dir``: (3, 3), or (3, 3, Nv) for a per-voxel field."""
        if self.fiber_dir.ndim > 1:
            return self.conductivity_field_oriented(self.fiber_dir)
        R = rotation_from_direction(self.fiber_dir)
        return _rotate_tensor2(R, self._conductivity_tensor_reference())

    def conductivity_tensor_rotated(self, fiber_dir: jnp.ndarray) -> jnp.ndarray:
        """(3, 3) reference conductivity rotated so the fiber points along ``fiber_dir``."""
        R = rotation_from_direction(jnp.asarray(fiber_dir, float))
        return _rotate_tensor2(R, self._conductivity_tensor_reference())

    def conductivity_field_oriented(self, orientations: jnp.ndarray) -> jnp.ndarray:
        """Reference conductivity rotated per voxel: (3, Nv) -> (3, 3, Nv)."""
        K_ref = self._conductivity_tensor_reference()

        def _one(d):
            R = rotation_from_direction(d)
            return _rotate_tensor2(R, K_ref)

        K_vox = jax.vmap(_one, in_axes=1)(orientations)
        return jnp.moveaxis(K_vox, 0, -1)

    def __repr__(self) -> str:
        tag = f" ({self.name})" if self.name else ""
        return f"ThermalConductivityTransverseIsotropic{tag}: k_L={self.k_L:.3g}  k_T={self.k_T:.3g}"
