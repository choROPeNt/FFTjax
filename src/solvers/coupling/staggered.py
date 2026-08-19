"""
Staggered driver coupling the mechanical (elasticity) and phase-field
(damage) sub-problems. Concrete to this one coupling on purpose -- the
degrade-stiffness step (``g(d) * C_field``) is meaningless for other
physics pairings (e.g. thermal <-> diffusion), so it isn't worth hiding
behind a generic callback-driven driver until a second coupling actually
needs one.
"""

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)

from typing import NamedTuple, Tuple

import jax.numpy as jnp

from materialmodels.phasefield.degradation import degrade_stiffness_field
from materialmodels.phasefield.driving_force import strain_energy_amor_split, update_history_hybrid
from operators.base import LinearOperator
from solvers.elliptic.scalar import solve_damage_helmholtz_cg
from solvers.elliptic.vector.lippmann_schwinger import solve_lippmann_schwinger


class StaggeredFractureSolution(NamedTuple):
    """
    Result of one staggered mechanics<->phase-field solve (one time
    increment, converged to the staggered tolerance or ``maxiter_st``
    exhausted). A NamedTuple so it's a JAX pytree like ``ElasticitySolution``.
    """

    eps:                 jnp.ndarray  # (3, 3, Nv)  local strain (degraded solve)
    sigma:               jnp.ndarray  # (3, 3, Nv)  local stress (degraded solve)
    delta:               jnp.ndarray  # (3, 3, Nv)  strain correction from the last CG solve
    d:                   jnp.ndarray  # (Nv,)       updated damage field
    H:                   jnp.ndarray  # (Nv,)       updated history variable
    psi_pos:             jnp.ndarray  # (Nv,)       tensile driving force (undegraded Amor split)
    converged_mech:      jnp.ndarray  # bool -- last mechanical CG solve
    converged_helm:      jnp.ndarray  # bool -- last damage CG solve
    converged_staggered: bool         # staggered fixed-point converged within maxiter_st
    iter_staggered:      int          # staggered iterations actually run
    err_abs:             float        # max|d_new - d_old| at the last iteration
    err_rel:             float        # err_abs / max|d_new|


def solve_staggered_mechanics_phasefield(
    n:            Tuple[int, ...],
    C_field:      jnp.ndarray,
    green_op:     LinearOperator,
    eps_bar:      jnp.ndarray,
    xi_flat:      jnp.ndarray,
    l0:           float,
    Gc:           float,
    lam_vox:      jnp.ndarray,
    mu_vox:       jnp.ndarray,
    d_init:       jnp.ndarray,
    H_init:       jnp.ndarray,
    toler_lin:    float = 1e-6,
    maxiter_cg:   int = 1000,
    toler_helm:   float = 1e-4,
    maxiter_helm: int = 300,
    eta:          float = 0.0,
    dt:           float = 1.0,
    d_thres:      float = 0.95,
    k_res:        float | jnp.ndarray = 1e-6,
    toler_st_abs: float = 1e-2,
    toler_st_rel: float = 1e-3,
    maxiter_st:   int = 200,
) -> StaggeredFractureSolution:
    """
    Fixed-point iteration between the elastic sub-problem (degraded
    stiffness, fixed d) and the damage sub-problem (driving force from the
    undegraded elastic solution, fixed eps) for one prescribed macroscopic
    strain ``eps_bar`` (one time increment).

    Per iteration
    -------------
        1. Degrade stiffness:  C_eff = g(d) * C_field   (g = AT2 degradation)
        2. Mechanical solve:   (eps, sigma) = solve_lippmann_schwinger(C_eff, ...)
        3. Driving force:      psi+ from the undegraded Amor split
        4. History update:     hybrid irreversibility (Steinke & Kaliske 2019)
        5. Damage solve:       d = solve_damage_helmholtz_cg(H, ...)
        6. Converged?          max|d_new - d_old| < toler_st_abs (or relative)

    Parameters
    ----------
    n, C_field, green_op, eps_bar : mechanical sub-problem, see
        ``solve_lippmann_schwinger`` (C_field is the *undegraded* per-voxel
        stiffness -- assembled once outside the staggered loop)
    xi_flat, l0, Gc               : damage sub-problem, see
        ``solve_damage_helmholtz_cg``
    lam_vox, mu_vox               : undegraded per-voxel Lamé constants for
        the Amor driving-force split -- see
        ``materialmodels.phasefield.driving_force.lame_field``
    d_init, H_init                : damage / history carried in from the
        previous accepted time increment (also the irreversibility floor)
    toler_lin, maxiter_cg         : mechanical CG tolerance / cap
    toler_helm, maxiter_helm      : damage CG tolerance / cap
    eta, dt                       : viscous regularisation, see
        ``solve_damage_helmholtz_cg``
    d_thres                       : hybrid irreversibility threshold
    k_res                         : AT2 residual stiffness -- scalar (homogeneous)
        or (Nv,) per-voxel array, e.g. from
        ``materialmodels.phasefield.degradation.k_res_field`` for per-phase
        k_res, or any other per-voxel field a caller builds directly
    toler_st_abs, toler_st_rel    : staggered convergence tolerances
        (absolute or relative -- either satisfies convergence)
    maxiter_st                    : max staggered iterations

    Returns
    -------
    StaggeredFractureSolution
    """
    d_st = d_init
    H_st = H_init

    for iter_st in range(1, maxiter_st + 1):
        d_prev_st = d_st

        C_eff = degrade_stiffness_field(C_field, d_st, k=k_res)

        eps, sigma, delta, converged_mech = solve_lippmann_schwinger(
            n, C_eff, green_op, eps_bar,
            toler_lin=toler_lin, maxiter=maxiter_cg,
        )

        psi_pos, _ = strain_energy_amor_split(eps, lam_vox, mu_vox)
        H_st = update_history_hybrid(H_st, psi_pos, d_prev_st, d_thres=d_thres)

        d_st, converged_helm = solve_damage_helmholtz_cg(
            H_st, xi_flat, n, l0, Gc, d_prev_st,
            toler_cg=toler_helm, maxiter=maxiter_helm, eta=eta, dt=dt, k=k_res,
        )

        diff = jnp.max(jnp.abs(d_st - d_prev_st))
        err_abs = float(diff)
        err_rel = float(diff / (jnp.max(jnp.abs(d_st)) + 1e-30))
        if err_abs < toler_st_abs or err_rel < toler_st_rel:
            break

    return StaggeredFractureSolution(
        eps=eps, sigma=sigma, delta=delta, d=d_st, H=H_st, psi_pos=psi_pos,
        converged_mech=converged_mech, converged_helm=converged_helm,
        converged_staggered=(err_abs < toler_st_abs or err_rel < toler_st_rel),
        iter_staggered=iter_st, err_abs=err_abs, err_rel=err_rel,
    )
