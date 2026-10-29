"""Geometry helpers."""

from functools import partial

import jax.numpy as jnp
from jax import jit


@partial(jit, static_argnames=("n_phases",))
def phase_fraction(phase: jnp.ndarray, n_phases: int) -> jnp.ndarray:
    """Volume fraction of each phase 0..n_phases-1."""
    counts = jnp.bincount(phase.ravel(), length=n_phases)
    return counts / phase.size
