import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)

from typing import Tuple

import jax.numpy as jnp

from operators.base import LinearOperator
from solvers.elliptic.vector.base import ElasticitySolver
from solvers.elliptic.vector.lippmann_schwinger import solve_lippmann_schwinger
from solvers.solution import ElasticitySolution


def solve_fourier_galerkin(
    n:           Tuple[int, ...],
    C_field:     jnp.ndarray,
    galerkin_op: LinearOperator,
    eps_bar:     jnp.ndarray,
    stress_goal: jnp.ndarray | None = None,
    toler_lin:   float = 1e-4,
    maxiter:     int = 1000,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Fourier-Galerkin elastic scheme (Vondrejc et al 2014; Lucarini, Upadhyay
    & Segurado 2022, doi:10.1088/1361-651X/ac34e1, sec. 3.3, algorithm 5).
    Delegates straight to solve_lippmann_schwinger: that function's CG
    operator/RHS composition (A(v) = Gamma0(C:v), b = -Gamma0(C:eps0 -
    stress_goal), eps = eps0 + delta) is algebraically identical to the
    paper's Fourier-Galerkin A(eps~)/b(x), the only difference being which
    Fourier-space kernel Gamma0Operator wraps. Passing a GalerkinProjector
    (operators.galerkin) here instead of a GreenOperatorBasic/Willot
    (operators.green) is what makes this the Fourier-Galerkin scheme rather
    than the classical Krylov-based Lippmann-Schwinger one -- no reference
    medium is involved anywhere in this call.

    Parameters
    ----------
    n           : grid shape (nx, ny, nz)
    C_field     : (3, 3, 3, 3, Nv)      per-voxel stiffness (or tangent)
    galerkin_op : LinearOperator        GalerkinProjector, already built for
                  this grid (operators.galerkin.build_galerkin_projector)
    eps_bar     : (3, 3)                prescribed macroscopic strain
    stress_goal : (3, 3, Nv) or None target stress field (None = zero, strain BC)
    toler_lin   : relative CG residual tolerance
    maxiter     : maximum CG iterations

    Returns
    -------
    eps        : (3, 3, Nv)   updated local strain  ε = ε₀ + Δε
    sigma      : (3, 3, Nv)   updated local stress  σ = C : ε
    delta      : (3, 3, Nv)   strain correction     Δε
    converged  : bool array   True if residual tolerance met
    """
    return solve_lippmann_schwinger(n, C_field, galerkin_op, eps_bar, stress_goal, toler_lin, maxiter)


class FourierGalerkinSolver(ElasticitySolver):
    """
    ElasticitySolver wrapping solve_fourier_galerkin: strain-based,
    periodic, no reference medium. Mirrors LippmannSchwingerSolver exactly
    except for the operator it's built from -- see that class and
    operators.galerkin.GalerkinProjector.
    """

    def __init__(
        self,
        n:           Tuple[int, ...],
        galerkin_op: LinearOperator,
        toler_lin:   float = 1e-4,
        maxiter:     int = 1000,
    ):
        self.n = n
        self.galerkin_op = galerkin_op
        self.toler_lin = toler_lin
        self.maxiter = maxiter

    def solve(
        self,
        C_field:     jnp.ndarray,
        eps_bar:     jnp.ndarray,
        stress_goal: jnp.ndarray | None = None,
    ) -> ElasticitySolution:
        eps, sigma, delta, converged = solve_fourier_galerkin(
            self.n, C_field, self.galerkin_op, eps_bar, stress_goal, self.toler_lin, self.maxiter,
        )
        return ElasticitySolution(eps, sigma, delta, converged)
