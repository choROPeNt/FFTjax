"""
Solution NamedTuples returned by the solvers. NamedTuples are JAX pytrees, so
they pass through jit/grad/vmap.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp


class ElasticitySolution(NamedTuple):
    """
    One elasticity solve.

    Attributes
    ----------
    eps, sigma : (3, 3, Nv) strain and stress
    delta      : (3, 3, Nv) strain correction from the CG solve
    converged  : bool
    eps_bar    : (3, 3) macroscopic strain with stress-controlled entries
                 solved; None for the pure-strain LippmannSchwingerSolver
    """

    eps:       jnp.ndarray
    sigma:     jnp.ndarray
    delta:     jnp.ndarray
    converged: jnp.ndarray
    eps_bar:   jnp.ndarray | None = None


class ThermalSolution(NamedTuple):
    """
    One steady-state thermal conduction solve.

    Attributes
    ----------
    grad_T     : (3, Nv) temperature gradient
    flux       : (3, Nv) heat flux q = -K : grad_T
    T_prime    : (Nv,) periodic temperature fluctuation; not the absolute
                 temperature (add grad_T_bar . x) and defined up to a constant
    delta      : (3, Nv) gradient correction from the CG solve
    converged  : bool
    grad_T_bar : (3,) macroscopic gradient with flux-controlled entries solved
    """

    grad_T:       jnp.ndarray
    flux:         jnp.ndarray
    T_prime:      jnp.ndarray
    delta:        jnp.ndarray
    converged:    jnp.ndarray
    grad_T_bar:   jnp.ndarray | None = None


class FractureSolution(NamedTuple):
    """
    One staggered mechanics/phase-field increment.

    Attributes
    ----------
    eps, sigma, delta   : (3, 3, Nv) degraded strain, stress, last strain correction
    d, H                : (Nv,) damage and history variable
    psi_pos             : (Nv,) tensile driving force (Amor split)
    converged_mech      : bool, last mechanical CG solve
    converged_helm      : bool, last damage CG solve
    converged_staggered : staggered loop converged; Python bool from
                          solve_fracture, JAX bool from solve_fracture_fixed
    iter_staggered      : staggered iterations run
    err_abs, err_rel    : max|d_new - d_old| and its ratio to max|d_new|
    eps_bar             : (3, 3) macroscopic strain with stress-controlled entries
                          solved; None for pure-strain lippmann_schwinger
    """

    eps:                 jnp.ndarray
    sigma:               jnp.ndarray
    delta:               jnp.ndarray
    d:                   jnp.ndarray
    H:                   jnp.ndarray
    psi_pos:             jnp.ndarray
    converged_mech:      jnp.ndarray
    converged_helm:      jnp.ndarray
    converged_staggered: bool | jnp.ndarray
    iter_staggered:      int
    err_abs:             float | jnp.ndarray
    err_rel:             float | jnp.ndarray
    eps_bar:             jnp.ndarray | None = None

    @property
    def converged(self) -> bool | jnp.ndarray:
        """Staggered-loop convergence (problems.incremental's Solution protocol)."""
        return self.converged_staggered
