"""
Wiring layer AND staggered driver for the fracture problem: mechanics +
AT2 phase-field damage on a phase-labelled periodic voxel grid, for one
prescribed macroscopic strain (one time increment).

The staggered fixed-point loop lives here rather than in solvers/ -- it's
not itself a reusable numerical algorithm the way solvers.krylov.cg.cg_solve
or solve_lippmann_schwinger are; it's specific composition of *this*
problem's physics choices (AT2 degradation, Amor split, hybrid
irreversibility). Keeping it here keeps solvers/ to its usual habit
(operate on prepared C_field/green_op/xi_flat arrays, no materialmodels
imports, no knowledge of "which degradation law" or "which energy split")
and keeps materialmodels/ imports confined to the one layer that's already
allowed to know about them.

Scope of this first pass, matching what's actually built on the
materialmodels/ and solvers/ stacks so far:
- Amor driving-force split, AT2 degradation, hybrid irreversibility,
  homogeneous Gc -- the two single-notch-plate benchmarks' scheme
  (Schneider & Kaestner 2025, doi:10.1111/ffe.14553).
- Damage/history (d, H) are carried across increments by the *caller*
  (this function takes d_init/H_init and returns d/H, it doesn't own the
  time-stepping loop) -- same division of responsibility as solve_mechanics
  not owning load-stepping either.
- ``control`` (mixed macroscopic strain/stress BC) only has meaning for
  formulation="displacement", mirroring problems/mechanics.py -- see that
  module's docstring for why lippmann_schwinger can't do it.
"""

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)

from typing import NamedTuple, Tuple

import jax.numpy as jnp

from materialmodels.assembly import assemble_C_field
from materialmodels.phasefield.degradation import degrade_stiffness_field, k_res_field
from materialmodels.phasefield.driving_force import lame_field, strain_energy_amor_split, update_history_hybrid
from operators.green import build_freq_grid, build_reference_green_operator
from solvers.elliptic.scalar import solve_damage_helmholtz_cg
from solvers.elliptic.vector.displacement_based import solve_displacement_based
from solvers.elliptic.vector.lippmann_schwinger import solve_lippmann_schwinger

_ZERO_CONTROL = ((0, 0, 0), (0, 0, 0), (0, 0, 0))


class FractureSolution(NamedTuple):
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
    eps_bar:             jnp.ndarray | None = None  # (3, 3) macroscopic strain with
                                                     # any stress-controlled entries
                                                     # filled in -- None unless
                                                     # formulation="displacement"


def solve_fracture(
    n:            Tuple[int, ...],
    L:            Tuple[float, ...],
    phase:        jnp.ndarray,
    materials:    list,
    eps_bar:      jnp.ndarray,
    l0:           float,
    Gc:           float,
    d_init:       jnp.ndarray,
    H_init:       jnp.ndarray,
    formulation:  str = "lippmann_schwinger",
    scheme:       str = "rotated",
    control:      Tuple[Tuple[int, ...], ...] | None = None,
    stress_goal:  jnp.ndarray | None = None,
    toler_lin:    float = 1e-6,
    maxiter_cg:   int = 1000,
    toler_helm:   float = 1e-4,
    maxiter_helm: int = 300,
    eta:          float = 0.0,
    dt:           float = 1.0,
    d_thres:      float = 0.95,
    k_res:        float | jnp.ndarray | None = None,
    toler_st_abs: float = 1e-2,
    toler_st_rel: float = 1e-3,
    maxiter_st:   int = 200,
) -> FractureSolution:
    """
    Solve one staggered mechanics<->phase-field increment under a
    prescribed macroscopic strain (and, for formulation="displacement",
    optionally mixed strain/stress control).

    Per staggered iteration
    ------------------------
        1. Degrade stiffness:  C_eff = g(d) * C_field   (g = AT2 degradation)
        2. Mechanical solve:   (eps, sigma) = solve_lippmann_schwinger(C_eff, ...)
                                or solve_displacement_based(C_eff, ...)
        3. Driving force:      psi+ from the undegraded Amor split
        4. History update:     hybrid irreversibility (Steinke & Kaliske 2019)
        5. Damage solve:       d = solve_damage_helmholtz_cg(H, ...)
        6. Converged?          max|d_new - d_old| < toler_st_abs (or relative)

    Parameters
    ----------
    n, L        : grid shape and physical domain size
    phase       : (Nv,) int      phase index per voxel (0-based)
    materials   : list           each implements .stiffness_tensor(), .lam, .mu
                  (see materialmodels.elastic.isotropic.LinearElasticIsotropic)
    eps_bar     : (3, 3)         prescribed macroscopic strain for this increment;
                  entries where ``control == 1`` are ignored (solved for instead)
    l0, Gc      : phase-field length scale and critical energy release rate
    d_init      : (Nv,)          damage carried in from the previous increment
    H_init      : (Nv,)          history variable carried in from the previous increment
    formulation : "lippmann_schwinger" (reference-medium, strain BC only) or
                  "displacement" (true heterogeneous tangent, supports mixed BC)
    scheme      : "standard" (GreenOperatorBasic) or "rotated" (GreenOperatorWillot)
                  -- only used by formulation="lippmann_schwinger"
    control     : (3, 3) 0/1 mask, 1 = stress-controlled, 0 = strain-controlled.
                  None (default) = pure strain BC (all zero). Only
                  formulation="displacement" can have any nonzero entries.
    stress_goal : (3, 3) macroscopic target stress, used only on
                  ``control``-marked entries -- only meaningful for
                  formulation="displacement"
    toler_lin, maxiter_cg     : mechanical CG tolerance / iteration cap
    toler_helm, maxiter_helm  : damage CG tolerance / iteration cap
    eta, dt                  : viscous regularisation of the damage equation
    d_thres                  : hybrid irreversibility threshold
    k_res                    : AT2 residual stiffness. None (default) gathers
                                each material's own .k_res per-phase (see
                                materialmodels.phasefield.degradation.k_res_field
                                -- e.g. LinearElasticIsotropic(..., k_res=1.0)
                                for a damage-immune phase). Pass a scalar or an
                                explicit (Nv,) array to override the per-phase
                                gather entirely (e.g. a smoothly-varying field).
    toler_st_abs, toler_st_rel, maxiter_st : staggered fixed-point convergence

    Returns
    -------
    FractureSolution(eps, sigma, delta, d, H, psi_pos, converged_mech,
    converged_helm, converged_staggered, iter_staggered, err_abs, err_rel,
    eps_bar)
    """
    if formulation not in ("lippmann_schwinger", "displacement"):
        raise ValueError(
            f"unknown formulation {formulation!r}, expected 'lippmann_schwinger' or 'displacement'"
        )

    control = control if control is not None else _ZERO_CONTROL
    control_nonzero = any(any(row) for row in control)
    if formulation == "lippmann_schwinger" and control_nonzero:
        raise ValueError(
            "formulation='lippmann_schwinger' cannot do stress-controlled "
            "macroscopic BC (control has nonzero entries) -- its reference-"
            "medium approach only supports pure strain BC; "
            "use formulation='displacement' instead"
        )
    stress_goal_arr = jnp.zeros((3, 3)) if stress_goal is None else stress_goal

    C_field = assemble_C_field(materials, phase)
    lam_vox, mu_vox = lame_field(materials, phase)
    xi_flat = build_freq_grid(n, L)
    k_res_arr = k_res_field(materials, phase) if k_res is None else k_res

    if formulation == "lippmann_schwinger":
        green_op = build_reference_green_operator(n, L, materials, scheme=scheme)

    d_st = d_init
    H_st = H_init
    eps_bar_cur = None

    for iter_st in range(1, maxiter_st + 1):
        d_prev_st = d_st

        C_eff = degrade_stiffness_field(C_field, d_st, k=k_res_arr)

        if formulation == "lippmann_schwinger":
            eps, sigma, delta, converged_mech = solve_lippmann_schwinger(
                n, C_eff, green_op, eps_bar,
                toler_lin=toler_lin, maxiter=maxiter_cg,
            )
        else:  # "displacement"
            eps, sigma, delta, eps_bar_cur, converged_mech = solve_displacement_based(
                n, C_eff, xi_flat, eps_bar, control, stress_goal_arr,
                toler_lin=toler_lin, maxiter=maxiter_cg,
            )

        psi_pos, _ = strain_energy_amor_split(eps, lam_vox, mu_vox)
        H_st = update_history_hybrid(H_st, psi_pos, d_prev_st, d_thres=d_thres)

        d_st, converged_helm = solve_damage_helmholtz_cg(
            H_st, xi_flat, n, l0, Gc, d_prev_st,
            toler_cg=toler_helm, maxiter=maxiter_helm, eta=eta, dt=dt, k=k_res_arr,
        )

        diff = jnp.max(jnp.abs(d_st - d_prev_st))
        err_abs = float(diff)
        err_rel = float(diff / (jnp.max(jnp.abs(d_st)) + 1e-30))
        if err_abs < toler_st_abs or err_rel < toler_st_rel:
            break

    return FractureSolution(
        eps=eps, sigma=sigma, delta=delta, d=d_st, H=H_st, psi_pos=psi_pos,
        converged_mech=converged_mech, converged_helm=converged_helm,
        converged_staggered=(err_abs < toler_st_abs or err_rel < toler_st_rel),
        iter_staggered=iter_st, err_abs=err_abs, err_rel=err_rel,
        eps_bar=eps_bar_cur,
    )
