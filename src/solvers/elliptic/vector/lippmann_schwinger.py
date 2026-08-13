import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)

from typing import Tuple
from math import prod

import jax.numpy as jnp

from operators.base import LinearOperator
from operators.general_functions import ddot42
from operators.projection import Gamma0Operator
from solvers.elliptic.vector.base import ElasticitySolver, ElasticitySolution
from solvers.krylov.cg import cg_solve


def solve_lippmann_schwinger(
    n:           Tuple[int, ...],
    C_field:     jnp.ndarray,
    green_op:    LinearOperator,
    eps_bar:     jnp.ndarray,
    stress_goal: jnp.ndarray | None = None,
    toler_lin:   float = 1e-4,
    maxiter:     int = 1000,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Inner CG solve for one Newton step of the variational FFT elastic solver
    (small strains, Vondrejc / Lucarini-Segurado formulation) -- same
    Krylov-based Lippmann-Schwinger scheme as ``strain_nw_cg.dstrain_nw_cg``,
    rewired onto the operators/ LinearOperator stack (GreenOperatorBasic/
    Willot + Gamma0Operator) instead of a hand-rolled FFT/einsum A_op.
    Verified bit-identical to ``dstrain_nw_cg`` in
    ``test/test_elliptic_vector_lippmann_schwinger.py``.

    Solves the linear system
        A(Δε) = b
    where
        A(v)  = Gamma0(C:v)                     [Green-stiffness operator]
        b     = -Gamma0(C:ε₀ - σ_goal)          [projected residual]
        ε₀    = eps_bar broadcast to all voxels [uniform initial guess]

    Parameters
    ----------
    n           : grid shape (nx, ny, nz)
    C_field     : (3, 3, 3, 3, Nv)      per-voxel stiffness (or tangent)
    green_op    : LinearOperator        GreenOperatorBasic or GreenOperatorWillot,
                  already built for this grid and reference medium
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

    Not yet JIT-compiled at this level (unlike dstrain_nw_cg): green_op is a
    plain Python object, not a registered JAX pytree, so it can't be passed
    as a jax.jit argument without either being marked static (which would
    embed its G array as a compile-time constant, recompiling on every new
    instance) or registering LinearOperator subclasses as pytrees -- left as
    a follow-up, since ddot42 and cg_solve's internal while_loop are already
    individually JIT-compiled.
    """
    Nv = prod(n)
    gamma0 = Gamma0Operator(n, green_op)

    # ── Linear operator  A(v) = Gamma0(C:v) ─────────────────────────────────
    def A_op(v_flat):
        v = v_flat.reshape(3, 3, Nv)
        Cv = ddot42(C_field, v)
        return gamma0(Cv).reshape(-1)

    # ── RHS  b = -Gamma0(C:ε₀ − σ_goal) ──────────────────────────────────────
    sg     = jnp.zeros((3, 3, Nv)) if stress_goal is None else stress_goal
    eps0   = jnp.ones((3, 3, Nv)) * eps_bar[:, :, None]
    sigma0 = ddot42(C_field, eps0)
    bb     = -gamma0(sigma0 - sg).reshape(-1)

    # ── CG solve ──────────────────────────────────────────────────────────────
    x0 = jnp.zeros_like(bb)
    delta_flat, converged = cg_solve(A_op, bb, x0, toler_lin, maxiter)

    delta = delta_flat.reshape(3, 3, Nv)
    eps   = eps0 + delta
    sigma = ddot42(C_field, eps)

    return eps, sigma, delta, converged


class LippmannSchwingerSolver(ElasticitySolver):
    """
    ElasticitySolver wrapping solve_lippmann_schwinger: strain-based,
    periodic. Formulation-specific setup (grid shape, Green's operator, CG
    tolerance) lives here in __init__; solve() takes only what's common to
    every ElasticitySolver.
    """

    def __init__(
        self,
        n:         Tuple[int, ...],
        green_op:  LinearOperator,
        toler_lin: float = 1e-4,
        maxiter:   int = 1000,
    ):
        self.n = n
        self.green_op = green_op
        self.toler_lin = toler_lin
        self.maxiter = maxiter

    def solve(
        self,
        C_field:     jnp.ndarray,
        eps_bar:     jnp.ndarray,
        stress_goal: jnp.ndarray | None = None,
    ) -> ElasticitySolution:
        eps, sigma, delta, converged = solve_lippmann_schwinger(
            self.n, C_field, self.green_op, eps_bar, stress_goal, self.toler_lin, self.maxiter,
        )
        return ElasticitySolution(eps, sigma, delta, converged)
