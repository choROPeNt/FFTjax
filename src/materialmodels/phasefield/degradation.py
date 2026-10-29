"""AT2 stiffness degradation and per-phase phase-field parameters."""

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)

from collections.abc import Sequence

import jax.numpy as jnp

from materialmodels.base import PhaseFieldMaterial


def degradation_at2(d: jnp.ndarray, k: float | jnp.ndarray = 1e-6) -> jnp.ndarray:
    """
    ``g(d) = (1 - k) (1 - d)^2 + k``.

    Parameters
    ----------
    d : (Nv,) damage in [0, 1]
    k : residual stiffness k_res, scalar or (Nv,); keeps d = 1 from going
        singular, k = 1 disables degradation

    Returns
    -------
    g : (Nv,)
    """
    return (1.0 - k) * (1.0 - d) ** 2 + k


def degrade_stiffness_field(
    C_field: jnp.ndarray,
    d: jnp.ndarray,
    k: float | jnp.ndarray = 0.0,
) -> jnp.ndarray:
    """
    ``C_eff = g(d) C_field`` for an already-assembled field (reused across
    staggered iterations instead of re-assembled).

    Parameters
    ----------
    C_field : (3, 3, 3, 3, Nv) undegraded stiffness
    d       : (Nv,) damage
    k       : residual stiffness, scalar or (Nv,)

    Returns
    -------
    C_eff : (3, 3, 3, 3, Nv)
    """
    g = degradation_at2(d, k=k)
    return g[None, None, None, None, :] * C_field


def k_res_field(materials: Sequence[PhaseFieldMaterial], phase: jnp.ndarray) -> jnp.ndarray:
    """Per-voxel ``k_res`` gathered by phase: (Nv,)."""
    k_stack = jnp.array([m.k_res for m in materials])
    return k_stack[phase]


def Gc_field(materials: Sequence[PhaseFieldMaterial], phase: jnp.ndarray) -> jnp.ndarray:
    """
    Per-voxel ``Gc`` gathered by phase: (Nv,).

    Raises if a material has no Gc: unlike k_res, no toughness is safe to
    assume. A non-uniform result needs div(Gc grad d), not Gc*lap(d) (they
    differ by grad(Gc).grad(d) at every interface);
    solve_damage_helmholtz_cg picks that form for a per-voxel Gc.
    """
    missing = [i for i, m in enumerate(materials) if getattr(m, "Gc", None) is None]
    if missing:
        raise ValueError(
            f"materials at index {missing} have no .Gc set -- every material "
            "used in a fracture solve needs a critical energy release rate"
        )
    Gc_stack = jnp.array([m.Gc for m in materials])
    return Gc_stack[phase]
