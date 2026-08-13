"""Voxel averaging for interface voxels with a partial (smooth) phase volume fraction."""

from __future__ import annotations

from abc import ABC, abstractmethod

import jax.numpy as jnp


class VoxelAveraging(ABC):
    """
    Combines two phases' stiffness tensors into one effective per-voxel value
    at interface voxels, where a single voxel spans a smooth volume fraction
    of each phase rather than being purely one or the other (e.g. a
    supersampled sub-voxel area fraction, as opposed to the hard-partitioned
    ``phase`` index materialmodels/assembly.py uses).

    Follows this project's per-voxel field convention (tensor indices first,
    voxel index trailing -- see materialmodels/assembly.py::assemble_C_field,
    operators/general_functions.py::ddot42): C_a/C_b are either a single bulk
    ``(3, 3, 3, 3)`` tensor (a material's ``stiffness_tensor()``, constant
    over all voxels) or already per-voxel ``(3, 3, 3, 3, Nv)``.
    """

    @abstractmethod
    def average(self, C_a: jnp.ndarray, C_b: jnp.ndarray, vf: jnp.ndarray) -> jnp.ndarray:
        """
        Effective stiffness for a voxel split between phase a (fraction vf)
        and phase b (fraction 1-vf).

        C_a, C_b : (3, 3, 3, 3) bulk, or (3, 3, 3, 3, Nv) per-voxel
        vf       : scalar, or (Nv,) volume fraction of phase a, in [0, 1]
        """
        ...


class ArithmeticAveraging(VoxelAveraging):
    """Volume-weighted (Voigt) mean: C_eff = vf * C_a + (1 - vf) * C_b."""

    def average(self, C_a: jnp.ndarray, C_b: jnp.ndarray, vf: jnp.ndarray) -> jnp.ndarray:
        vf = jnp.asarray(vf)
        if vf.ndim > 0 and C_a.ndim == 4:
            # bulk tensor + per-voxel fraction: add the trailing voxel axis
            # explicitly -- broadcasting can't infer it, since numpy/jax pad
            # missing dims on the *left*, not the right.
            C_a = C_a[..., None]
            C_b = C_b[..., None]
        return vf * C_a + (1.0 - vf) * C_b
