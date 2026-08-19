"""AT2 stiffness degradation law for phase-field fracture."""

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)

from collections.abc import Sequence

import jax.numpy as jnp

from materialmodels.base import ConstitutiveModel


def degradation_at2(d: jnp.ndarray, k: float | jnp.ndarray = 1e-6) -> jnp.ndarray:
    """
    Quadratic AT2 degradation function  g(d) = (1 - k) * (1 - d)² + k.

    k is the residual stiffness k_res, keeping fully-cracked voxels (d=1)
    from going exactly singular (or, at k=1, keeping a phase from degrading
    at all). Otherwise a pure function of the damage field -- knows nothing
    about materials or phases; callers combine it with an undegraded
    stiffness field via a plain elementwise multiply, e.g.
    ``g(d)[None, None, None, None, :] * C_field``.

    Parameters
    ----------
    d : (Nv,)          damage field in [0, 1]
    k : float or (Nv,) residual stiffness k_res -- scalar (homogeneous,
                        default 1e-6) or a per-voxel array, e.g. from
                        ``k_res_field`` for per-phase k_res, or any other
                        per-voxel field a caller builds directly.

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
    Apply AT2 degradation to an already-assembled undegraded stiffness
    field: ``C_eff = g(d) * C_field``. Kept in materialmodels/ (rather than
    inlined at the call site in solvers/coupling/staggered.py) so the
    materials layer owns how degradation combines with stiffness -- the
    caller only needs to know it *does*, not the broadcast shape.

    Deliberately takes the already-assembled ``C_field`` (from
    ``materialmodels.assembly.assemble_C_field``) rather than a materials
    list -- ``C_field`` doesn't change across staggered iterations while
    ``d`` does, so re-assembling per phase on every call here would redo
    that (more expensive) gather for no reason. Callers still gather
    ``k`` once via ``k_res_field`` (or pass a scalar) and hold onto it
    across iterations the same way.

    Parameters
    ----------
    C_field : (3, 3, 3, 3, Nv)   undegraded per-voxel stiffness
    d       : (Nv,)              damage field in [0, 1]
    k       : float or (Nv,)     AT2 residual stiffness k_res, see
                                  ``degradation_at2``

    Returns
    -------
    C_eff : (3, 3, 3, 3, Nv)
    """
    g = degradation_at2(d, k=k)
    return g[None, None, None, None, :] * C_field


def k_res_field(materials: Sequence[ConstitutiveModel], phase: jnp.ndarray) -> jnp.ndarray:
    """
    Per-voxel AT2 residual stiffness k_res, gathered from each material's
    ``.k_res`` attribute by phase index -- same hard (sharp-interface)
    assembly pattern as ``materialmodels.assembly.assemble_C_field`` and
    ``materialmodels.phasefield.driving_force.lame_field``.

    Parameters
    ----------
    materials : list of ConstitutiveModel with .k_res, indexed by phase
                (see materialmodels.elastic.isotropic.LinearElasticIsotropic)
    phase     : (Nv,) int   phase index per voxel

    Returns
    -------
    k_res : (Nv,)
    """
    k_stack = jnp.array([m.k_res for m in materials])
    return k_stack[phase]
