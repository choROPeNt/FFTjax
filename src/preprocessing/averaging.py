"""Voxel averaging for interface voxels with a partial (smooth) phase volume fraction."""

from __future__ import annotations

from abc import ABC, abstractmethod

import jax.numpy as jnp


class VoxelAveraging(ABC):
    """
    Effective stiffness of interface voxels split between two phases by a
    smooth volume fraction. C_a, C_b: (3, 3, 3, 3) bulk or (3, 3, 3, 3, Nv).
    """

    @abstractmethod
    def average(self, C_a: jnp.ndarray, C_b: jnp.ndarray, vf: jnp.ndarray) -> jnp.ndarray:
        """Stiffness of a voxel with fraction ``vf`` (scalar or (Nv,)) of phase a."""
        ...


class ArithmeticAveraging(VoxelAveraging):
    """Volume-weighted (Voigt) mean: C_eff = vf * C_a + (1 - vf) * C_b."""

    def average(self, C_a: jnp.ndarray, C_b: jnp.ndarray, vf: jnp.ndarray) -> jnp.ndarray:
        vf = jnp.asarray(vf)
        if vf.ndim > 0 and C_a.ndim == 4:
            # Broadcasting pads on the left, so add the trailing voxel axis.
            C_a = C_a[..., None]
            C_b = C_b[..., None]
        return vf * C_a + (1.0 - vf) * C_b
