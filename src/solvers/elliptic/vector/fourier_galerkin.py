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
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Fourier-Galerkin elastic scheme (Vondrejc et al 2014; Lucarini, Upadhyay
    & Segurado 2022, doi:10.1088/1361-651X/ac34e1, sec. 3.3, algorithm 5),
    with mixed macroscopic strain/stress control (same paper, sec. 3.6,
    Lucarini & Segurado 2019a) via solvers.elliptic.vector.mixed_bc.
    solve_mixed_bc_dc_identity -- see that function's docstring for the
    DC-bin-identity derivation. Passing a GalerkinProjector
    (operators.galerkin) here instead of a GreenOperatorBasic/Willot
    (operators.green) is what makes this the Fourier-Galerkin scheme rather
    than the classical Krylov-based Lippmann-Schwinger one -- no reference
    medium is involved anywhere in this call.

    Unlike the pre-mixed-BC version of this function, this is no longer a
    pure delegation to solve_lippmann_schwinger: solving directly for the
    full field (rather than a correction around a fixed uniform eps0) is
    needed to let a stress-controlled direction's macroscopic mean be a
    genuine degree of freedom -- solve_mixed_bc_dc_identity implements that
    shared core, used identically by solve_lippmann_schwinger_dc for the
    reference-medium counterpart of this same scheme.

    Parameters
    ----------
    n           : grid shape (nx, ny, nz)
    C_field     : (3, 3, 3, 3, Nv)      per-voxel stiffness (or tangent)
    galerkin_op : GalerkinProjector, already built for this grid
                  (operators.galerkin.build_galerkin_projector)
    eps_bar     : (3, 3)                prescribed macroscopic strain;
                  entries where ``control == 1`` are ignored (solved for
                  instead)
    control     : (3, 3) 0/1 mask, 1 = stress-controlled, 0 = strain-controlled.
                  None (default) = pure strain BC (all zero).
    macro_stress_goal : (3, 3) or None  target macroscopic stress; only
                  entries where ``control == 1`` are used. None = zeros.
    toler_lin   : relative CG residual tolerance
    maxiter     : maximum CG iterations

    Returns
    -------
    eps        : (3, 3, Nv)   updated local strain  ε = ε₀ + Δε
    sigma      : (3, 3, Nv)   updated local stress  σ = C : ε
    delta      : (3, 3, Nv)   strain correction     Δε
    eps_bar_out: (3, 3)       macroscopic strain, with stress-controlled
                 entries filled in by the solve
    converged  : bool array   True if residual tolerance met
    """
    control = control if control is not None else _ZERO_CONTROL
    macro_stress_goal = jnp.zeros((3, 3)) if macro_stress_goal is None else macro_stress_goal
    return solve_mixed_bc_dc_identity(
        n, C_field, galerkin_op, control, eps_bar, macro_stress_goal, toler_lin, maxiter,
    )


class FourierGalerkinSolver(ElasticitySolver):
    """
    ElasticitySolver wrapping solve_fourier_galerkin: strain-based,
    periodic, no reference medium, supports mixed strain/stress macroscopic
    BC via ``control`` (same convention as DisplacementBasedSolver). Mirrors
    LippmannSchwingerSolver's constructor except for the operator it's built
    from -- see that class and operators.galerkin.GalerkinProjector.
    """

    def __init__(
        self,
        n:           Tuple[int, ...],
        galerkin_op: _HasG,
        control:     Tuple[Tuple[int, ...], ...] | None = None,
        toler_lin:   float = 1e-4,
        maxiter:     int = 1000,
    ):
        self.n = n
        self.galerkin_op = galerkin_op
        self.control = control if control is not None else _ZERO_CONTROL
        self.toler_lin = toler_lin
        self.maxiter = maxiter

    def solve(
        self,
        C_field:     jnp.ndarray,
        eps_bar:     jnp.ndarray,
        stress_goal: jnp.ndarray | None = None,
    ) -> ElasticitySolution:
        eps, sigma, delta, eps_bar_out, converged = solve_fourier_galerkin(
            self.n, C_field, self.galerkin_op, eps_bar,
            self.control, stress_goal, self.toler_lin, self.maxiter,
        )
        return ElasticitySolution(eps, sigma, delta, converged, eps_bar=eps_bar_out)
