import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)

from typing import Tuple
from math import prod

import jax.numpy as jnp

from operators.base import LinearOperator
from operators.general_functions import ddot42
from operators.green import GreenOperatorBasic
from operators.projection import Gamma0Operator
from solvers.elliptic.vector.base import ElasticitySolver
from solvers.elliptic.vector.mixed_bc import (
    _HasG, _ZERO_CONTROL, _active_pairs, sm2sv, solve_mixed_bc_dc_identity, sv2sm,
)
from solvers.krylov.cg import cg_solve
from solvers.solution import ElasticitySolution


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


def solve_lippmann_schwinger_dc(
    n:                 Tuple[int, ...],
    C_field:           jnp.ndarray,
    green_op:          _HasG,
    eps_bar:           jnp.ndarray,
    control:           Tuple[Tuple[int, ...], ...] | None = None,
    macro_stress_goal: jnp.ndarray | None = None,
    toler_lin:         float = 1e-4,
    maxiter:           int = 1000,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Kabel et al (2016) single-CG-solve mixed strain/stress macroscopic-BC
    route for Lippmann-Schwinger -- the same DC-bin identity patch
    solve_fourier_galerkin uses, applied to a reference-medium operator
    (GreenOperatorBasic/Willot) instead of a reference-medium-free Galerkin
    projector; see solvers.elliptic.vector.mixed_bc.
    solve_mixed_bc_dc_identity for the shared derivation, which is
    genuinely operator-agnostic between the two.

    Secondary to solve_lippmann_schwinger_mixed_bc (the Michel et al 1999
    outer-loop route, primary for this formulation since it leaves
    solve_lippmann_schwinger itself completely untouched) -- kept here as an
    independent, single-CG-solve cross-check on the same physics rather than
    the production entry point. See notes/controll.md.
    """
    control = control if control is not None else _ZERO_CONTROL
    macro_stress_goal = jnp.zeros((3, 3)) if macro_stress_goal is None else macro_stress_goal
    return solve_mixed_bc_dc_identity(
        n, C_field, green_op, control, eps_bar, macro_stress_goal, toler_lin, maxiter,
    )


def solve_lippmann_schwinger_mixed_bc(
    n:                 Tuple[int, ...],
    C_field:           jnp.ndarray,
    green_op:          GreenOperatorBasic,
    control:           Tuple[Tuple[int, ...], ...],
    eps_bar_guess:     jnp.ndarray,
    macro_stress_goal: jnp.ndarray,
    toler_lin:         float = 1e-4,
    maxiter:           int = 1000,
    toler_outer:       float = 1e-6,
    maxiter_outer:     int = 50,
    C0:                jnp.ndarray | None = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, int]:
    """
    Michel et al (1999) outer iterative correction for mixed strain/stress
    macroscopic BC: repeatedly calls the UNCHANGED solve_lippmann_schwinger,
    correcting ``eps_bar`` between calls until the mean stress on
    stress-controlled directions matches ``macro_stress_goal``. Primary
    mixed-BC route for this formulation -- zero risk to the existing
    reference-medium CG solve/preconditioner, at the cost of needing several
    outer iterations (each a full inner CG solve) rather than
    solve_lippmann_schwinger_dc's single one. See notes/controll.md.

    The correction is a Newton-like step built from a small (K, K) (K <= 6)
    system: ``M_active = <C0>_active`` restricted to the active
    stress-controlled (i, j) pairs (``_active_pairs``), the exact sensitivity
    of mean stress to a uniform macroscopic strain perturbation for a
    homogeneous material with reference stiffness ``C0`` (default: the true
    voxel-mean ``jnp.mean(C_field, axis=-1)``, not an isotropized reference
    medium -- see the C0_ref line below for why) -- for a homogeneous
    C_field this converges in a single outer iteration (the sensitivity is
    then exact, not approximate); for a heterogeneous one it's the standard
    fixed-point correction (Michel et al 1999's own algorithm, eq 106),
    converging over several outer iterations. This is an UNDAMPED fixed-point
    step (no line search/trust region) -- a poor enough ``C0`` approximation
    of the true sensitivity (e.g. an isotropized ``C0`` against a genuinely
    anisotropic material) can make it diverge rather than merely converge
    slowly; pass a better ``C0`` (or a genuinely anisotropic material's own
    voxel-mean, the default) if you see ``eps_bar`` blow up instead of settle.

    control=all-zero degenerates to exactly one call of the unchanged
    solve_lippmann_schwinger (n_iter_outer=1), returning eps_bar_guess as
    eps_bar_out unchanged -- no outer-loop overhead in the pure-strain case.

    Parameters
    ----------
    n, C_field, green_op   : see solve_lippmann_schwinger
    control                : (3, 3) 0/1 mask, 1 = stress-controlled
    eps_bar_guess           : (3, 3) initial macroscopic strain guess; entries
                              where ``control == 1`` are corrected by the
                              outer loop, entries where ``control == 0`` stay
                              fixed at this value throughout
    macro_stress_goal       : (3, 3) target macroscopic stress; only entries
                              where ``control == 1`` are used
    toler_lin, maxiter      : inner CG tolerance / iteration cap, forwarded
                              unchanged to every solve_lippmann_schwinger call
    toler_outer, maxiter_outer : outer-loop relative-residual tolerance and
                              iteration cap
    C0                      : (3, 3, 3, 3) or None -- reference stiffness for
                              the outer-loop correction's sensitivity matrix;
                              None (default) uses the true (anisotropic)
                              voxel-mean ``jnp.mean(C_field, axis=-1)``, NOT
                              green_op's own isotropized reference medium --
                              see the C0_ref line's comment for why (a poor
                              Jacobian approximation for an anisotropic
                              material can make this undamped correction
                              diverge, not just converge slowly)

    Returns
    -------
    eps, sigma, delta   : from the last inner solve_lippmann_schwinger call
    eps_bar_out         : (3, 3) macroscopic strain, stress-controlled
                          entries filled in by the outer loop
    converged           : bool -- inner CG converged AND outer residual
                          within toler_outer
    n_iter_outer        : int -- outer iterations actually run
    """
    pairs = _active_pairs(control)

    if not pairs:
        eps, sigma, delta, converged = solve_lippmann_schwinger(
            n, C_field, green_op, eps_bar_guess, toler_lin=toler_lin, maxiter=maxiter,
        )
        return eps, sigma, delta, eps_bar_guess, converged, 1

    # True (anisotropic) voxel-mean stiffness by default, NOT green_op's own
    # isotropized reference medium (_isotropic_stiffness(green_op.lam0,
    # green_op.mu0)) -- same default solve_displacement_based's own
    # preconditioner uses (jnp.mean(C_field, axis=-1) if C0 is None else C0).
    # Isotropizing away a genuinely anisotropic material's directional
    # stiffness (e.g. a transversely isotropic fibre, E_L/E_T ~ 15x here)
    # makes M a poor enough Jacobian approximation for this UNDAMPED
    # correction step that the outer loop can diverge geometrically instead
    # of just converging slowly -- reproduced on a carbon-fibre/epoxy RVE
    # under mixed BC: eps_bar blew up to ~1e21 within maxiter_outer using the
    # isotropized C0_ref, while the same case converges cleanly with this one.
    C0_ref = jnp.mean(C_field, axis=-1) if C0 is None else C0
    basis = jnp.eye(len(pairs), dtype=eps_bar_guess.dtype)
    M = jnp.stack(
        [sm2sv(jnp.einsum("ijkl,kl->ij", C0_ref, sv2sm(basis[k], pairs)), pairs)
         for k in range(len(pairs))],
        axis=1,
    )

    eps_bar = eps_bar_guess
    converged_outer = False

    for it in range(1, maxiter_outer + 1):
        eps, sigma, delta, converged = solve_lippmann_schwinger(
            n, C_field, green_op, eps_bar, toler_lin=toler_lin, maxiter=maxiter,
        )
        sigma_mean = jnp.mean(sigma, axis=-1)
        resid = sm2sv(macro_stress_goal - sigma_mean, pairs)
        # Scale by the CURRENT overall mean-stress norm, not macro_stress_goal's
        # own norm: a free-surface/traction-free target (macro_stress_goal == 0
        # on the controlled directions, e.g. lateral surfaces in a uniaxial
        # tension test) makes the target norm itself exactly zero, which would
        # collapse a goal-based relative tolerance to an unreachable floor even
        # once resid has reached floating-point noise -- the overall stress
        # tensor (dominated by whatever's on the strain-controlled directions,
        # e.g. the prescribed axial stress) is always a meaningful nonzero
        # scale instead. Same pattern as e.g. problems.fracture's staggered
        # loop, which scales its own convergence check by the current field's
        # magnitude, not a (possibly zero) target's.
        scale = jnp.linalg.norm(sigma_mean) + 1e-30
        if float(jnp.linalg.norm(resid)) < toler_outer * float(scale):
            converged_outer = True
            break
        eps_bar = eps_bar + sv2sm(jnp.linalg.solve(M, resid), pairs)

    return eps, sigma, delta, eps_bar, jnp.logical_and(converged, converged_outer), it


class LippmannSchwingerMixedBCSolver(ElasticitySolver):
    """
    ElasticitySolver wrapping solve_lippmann_schwinger_mixed_bc: an outer
    iterative correction (Michel et al 1999) around the UNCHANGED
    solve_lippmann_schwinger, supporting mixed strain/stress macroscopic BC
    via ``control`` (same convention as DisplacementBasedSolver/
    FourierGalerkinSolver) without touching the reference-medium CG solve
    itself. For the alternative single-CG-solve route (Kabel et al 2016),
    call solve_lippmann_schwinger_dc directly instead.
    """

    def __init__(
        self,
        n:             Tuple[int, ...],
        green_op:      GreenOperatorBasic,
        control:       Tuple[Tuple[int, ...], ...] | None = None,
        toler_lin:     float = 1e-4,
        maxiter:       int = 1000,
        toler_outer:   float = 1e-6,
        maxiter_outer: int = 50,
        C0:            jnp.ndarray | None = None,
    ):
        self.n = n
        self.green_op = green_op
        self.control = control if control is not None else _ZERO_CONTROL
        self.toler_lin = toler_lin
        self.maxiter = maxiter
        self.toler_outer = toler_outer
        self.maxiter_outer = maxiter_outer
        self.C0 = C0

    def solve(
        self,
        C_field:     jnp.ndarray,
        eps_bar:     jnp.ndarray,
        stress_goal: jnp.ndarray | None = None,
    ) -> ElasticitySolution:
        sg = jnp.zeros((3, 3)) if stress_goal is None else stress_goal
        eps, sigma, delta, eps_bar_out, converged, _n_iter_outer = solve_lippmann_schwinger_mixed_bc(
            self.n, C_field, self.green_op, self.control, eps_bar, sg,
            toler_lin=self.toler_lin, maxiter=self.maxiter,
            toler_outer=self.toler_outer, maxiter_outer=self.maxiter_outer, C0=self.C0,
        )
        return ElasticitySolution(eps, sigma, delta, converged, eps_bar=eps_bar_out)
