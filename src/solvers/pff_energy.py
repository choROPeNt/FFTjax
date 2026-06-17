import os
os.environ["JAX_ENABLE_X64"] = "1"

import jax.numpy as jnp

# Strain energy functions live in mat_models.elastic — re-exported here for
# convenience so solver code has a single import target.
from mat_models.elastic import (  # noqa: F401
    strain_energy_density,
    strain_energy_miehe_split,
    strain_energy_amor_split,
    strain_energy_spectral_split,   # backward-compatible alias for miehe
    lame_from_C_field,
)


def degradation(d: jnp.ndarray, k: float = 1e-7) -> jnp.ndarray:
    """
    Quadratic degradation function  g(d) = (1 − d)² + k.

    k is a small residual that keeps the stiffness matrix non-singular in
    fully damaged voxels.

    Parameters
    ----------
    d : (Nv,)   damage field ∈ [0, 1]
    k : float   residual stiffness (default 1e-7)

    Returns
    -------
    g : (Nv,)
    """
    return (1.0 - d) ** 2 + k
