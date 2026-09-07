"""
Unified storage point for the per-problem solution NamedTuples returned by
solvers/problems ``solve`` entry points across this project.

Both are NamedTuples (not dataclasses) so they're automatically JAX
pytrees -- safe to pass through jax.jit/grad/jvp/vmap, same convention as
solvers.types.SolveState. A plain dataclass isn't a pytree without extra
registration, which would silently break autodiff through the solve.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp


class ElasticitySolution(NamedTuple):
    """Result of one elasticity solve (Lippmann-Schwinger or displacement-based)."""

    eps:       jnp.ndarray  # (3, 3, Nv)  local strain
    sigma:     jnp.ndarray  # (3, 3, Nv)  local stress
    delta:     jnp.ndarray  # (3, 3, Nv)  strain correction from the CG solve
    converged: jnp.ndarray  # bool array
    eps_bar:   jnp.ndarray | None = None  # (3, 3) macroscopic strain with any
                                           # stress-controlled entries filled in
                                           # by the solve -- None (default) for
                                           # solvers with no mixed-BC concept
                                           # (e.g. LippmannSchwingerSolver);
                                           # populated by DisplacementBasedSolver


class FractureSolution(NamedTuple):
    """
    Result of one staggered mechanics<->phase-field solve (one time
    increment, converged to the staggered tolerance or ``maxiter_st``
    exhausted).
    """

    eps:                 jnp.ndarray  # (3, 3, Nv)  local strain (degraded solve)
    sigma:               jnp.ndarray  # (3, 3, Nv)  local stress (degraded solve)
    delta:               jnp.ndarray  # (3, 3, Nv)  strain correction from the last CG solve
    d:                   jnp.ndarray  # (Nv,)       updated damage field
    H:                   jnp.ndarray  # (Nv,)       updated history variable
    psi_pos:             jnp.ndarray  # (Nv,)       tensile driving force (undegraded Amor split)
    converged_mech:      jnp.ndarray  # bool -- last mechanical CG solve
    converged_helm:      jnp.ndarray  # bool -- last damage CG solve
    converged_staggered: bool | jnp.ndarray  # staggered fixed-point converged within
                                              # maxiter_st -- concrete Python bool from
                                              # solve_fracture (early_exit=True), a JAX
                                              # bool array from solve_fracture_fixed
                                              # (early_exit=False, vmap-safe: no bool()
                                              # concretization -- see _staggered_loop)
    iter_staggered:      int          # staggered iterations actually run
    err_abs:             float | jnp.ndarray  # max|d_new - d_old| at the last iteration --
                                               # float (early_exit=True) or array (early_exit=False)
    err_rel:             float | jnp.ndarray  # err_abs / max|d_new| -- same float/array split
    eps_bar:             jnp.ndarray | None = None  # (3, 3) macroscopic strain with
                                                     # any stress-controlled entries
                                                     # filled in -- None unless
                                                     # formulation="displacement"

    @property
    def converged(self) -> bool | jnp.ndarray:
        """Satisfies problems.incremental's Solution protocol -- the staggered
        loop's own convergence, not the last mechanical/Helmholtz sub-solve."""
        return self.converged_staggered
