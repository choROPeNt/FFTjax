"""Fourier-Galerkin elastic scheme (no reference medium)."""

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)

from typing import Tuple

import jax.numpy as jnp

from solvers.elliptic.vector.base import ElasticitySolver
from solvers.elliptic.vector.mixed_bc import _HasG, _ZERO_CONTROL, solve_mixed_bc_dc_identity
from solvers.solution import ElasticitySolution


def solve_fourier_galerkin(
    n:                 Tuple[int, ...],
    C_field:           jnp.ndarray,
    galerkin_op:       _HasG,
    eps_bar:           jnp.ndarray,
    control:           Tuple[Tuple[int, ...], ...] | None = None,
    macro_stress_goal: jnp.ndarray | None = None,
    toler_lin:         float = 1e-4,
    maxiter:           int = 1000,
    n_devices:       int | None = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Fourier-Galerkin scheme (Vondrejc et al 2014; Lucarini, Upadhyay & Segurado
    2022, alg. 5) with mixed strain/stress BC: ``solve_mixed_bc_dc_identity``
    with a GalerkinProjector.

    Parameters
    ----------
    n                 : grid shape (nx, ny, nz)
    C_field           : (3, 3, 3, 3, Nv) stiffness or tangent
    galerkin_op       : GalerkinProjector for this grid
    eps_bar           : (3, 3) macroscopic strain; stress-controlled entries ignored
    control           : (3, 3) 0/1 mask, 1 = stress-controlled (None = pure strain)
    macro_stress_goal : (3, 3) target stress, stress-controlled entries only (None = 0)
    toler_lin, maxiter: CG tolerance and iteration cap
    n_devices         : device cap for domain decomposition (None = all)

    Returns
    -------
    eps, sigma, delta : (3, 3, Nv) strain, stress, strain correction
    eps_bar_out       : (3, 3) macroscopic strain, stress-controlled entries solved
    converged         : bool
    """
    control = control if control is not None else _ZERO_CONTROL
    macro_stress_goal = jnp.zeros((3, 3)) if macro_stress_goal is None else macro_stress_goal
    return solve_mixed_bc_dc_identity(
        n, C_field, galerkin_op, control, eps_bar, macro_stress_goal, toler_lin, maxiter,
        n_devices=n_devices,
    )


class FourierGalerkinSolver(ElasticitySolver):
    """ElasticitySolver for ``solve_fourier_galerkin``; mixed BC via ``control``."""

    def __init__(
        self,
        n:           Tuple[int, ...],
        galerkin_op: _HasG,
        control:     Tuple[Tuple[int, ...], ...] | None = None,
        toler_lin:   float = 1e-4,
        maxiter:     int = 1000,
        n_devices: int | None = None,
    ):
        self.n = n
        self.galerkin_op = galerkin_op
        self.control = control if control is not None else _ZERO_CONTROL
        self.toler_lin = toler_lin
        self.maxiter = maxiter
        self.n_devices = n_devices

    def solve(
        self,
        C_field:     jnp.ndarray,
        eps_bar:     jnp.ndarray,
        stress_goal: jnp.ndarray | None = None,
    ) -> ElasticitySolution:
        eps, sigma, delta, eps_bar_out, converged = solve_fourier_galerkin(
            self.n, C_field, self.galerkin_op, eps_bar,
            self.control, stress_goal, self.toler_lin, self.maxiter,
            n_devices=self.n_devices,
        )
        return ElasticitySolution(eps, sigma, delta, converged, eps_bar=eps_bar_out)
