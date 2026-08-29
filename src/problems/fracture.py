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
- Amor driving-force split, AT2 degradation, hybrid irreversibility --
  the two single-notch-plate benchmarks' scheme (Schneider & Kaestner 2025,
  doi:10.1111/ffe.14553). Gc may be homogeneous (a scalar, or a uniform
  per-material gather) or heterogeneous (a genuinely varying per-voxel
  field) -- solve_fracture dispatches to solvers.elliptic.scalar's
  solve_damage_helmholtz_cg or solve_damage_helmholtz_cg_het accordingly;
  see that module's docstring for why these solve different equations, not
  the same one at different generality.
- Damage/history (d, H) are carried across increments by the *caller*
  (this function takes d_init/H_init and returns d/H, it doesn't own the
  time-stepping loop) -- same division of responsibility as solve_mechanics
  not owning load-stepping either.
- ``control`` (mixed macroscopic strain/stress BC) only has meaning for
  formulation="displacement", mirroring problems/mechanics.py -- see that
  module's docstring for why lippmann_schwinger can't do it.
"""

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)

import time
from typing import Callable, NamedTuple, Tuple, cast

import jax.numpy as jnp
import numpy as np

from materialmodels.assembly import assemble_C_field
from materialmodels.phasefield.degradation import Gc_field, degrade_stiffness_field, k_res_field
from materialmodels.phasefield.driving_force import lame_field, strain_energy_amor_split, update_history_hybrid
from operators.green import build_freq_grid, build_reference_green_operator
from post.fields import compute_displacement, field_to_grid, to_voigt, von_mises
from problems.incremental import IncrementResult, solve_automatic, solve_fixed
from solvers.elliptic.scalar import solve_damage_helmholtz_cg, solve_damage_helmholtz_cg_het
from solvers.elliptic.vector.displacement_based import solve_displacement_based
from solvers.elliptic.vector.lippmann_schwinger import solve_lippmann_schwinger
from utils.io.xdmf_writer import IncrementalWriter

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


def solve_fracture(
    n:            Tuple[int, ...],
    L:            Tuple[float, ...],
    phase:        jnp.ndarray,
    materials:    list,
    eps_bar:      jnp.ndarray,
    l0:           float,
    Gc:           float | jnp.ndarray | None,
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
    l0          : phase-field length scale
    Gc          : critical energy release rate. None gathers each material's
                  own .Gc per-phase (see materialmodels.phasefield.
                  degradation.Gc_field -- raises if any present material's
                  .Gc is unset). Pass a scalar to use one value for every
                  phase, or an explicit (Nv,) array for a field not tied to
                  materials (e.g. a smoothly-varying one). A uniform Gc
                  (scalar, or an array/gather that happens to be uniform)
                  uses the cheaper homogeneous solver
                  (solve_damage_helmholtz_cg); a genuinely varying one uses
                  the divergence-form heterogeneous solver
                  (solve_damage_helmholtz_cg_het) -- see that module's
                  docstring for why these solve different equations.
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

    if Gc is None:
        Gc = Gc_field(materials, phase)
    # A per-voxel Gc that happens to be uniform collapses to the cheaper,
    # exact-preconditioner homogeneous solver -- only a genuinely varying
    # Gc needs the 3x-FFT divergence-form one (see solvers.elliptic.scalar).
    Gc_heterogeneous = isinstance(Gc, jnp.ndarray) and jnp.ndim(Gc) > 0
    if Gc_heterogeneous:
        Gc_min, Gc_max = float(jnp.min(Gc)), float(jnp.max(Gc))
        if Gc_min == Gc_max:
            Gc = Gc_min
            Gc_heterogeneous = False

    if formulation == "lippmann_schwinger":
        green_op = build_reference_green_operator(n, L, materials, scheme=scheme)
    else:
        green_op = None

    return _staggered_loop(
        n=n, l0=l0, formulation=formulation, control=control, stress_goal_arr=stress_goal_arr,
        C_field=C_field, lam_vox=lam_vox, mu_vox=mu_vox, xi_flat=xi_flat, k_res_arr=k_res_arr,
        Gc=Gc, Gc_heterogeneous=Gc_heterogeneous, green_op=green_op,
        eps_bar=eps_bar, d_init=d_init, H_init=H_init,
        toler_lin=toler_lin, maxiter_cg=maxiter_cg,
        toler_helm=toler_helm, maxiter_helm=maxiter_helm,
        eta=eta, dt=dt, d_thres=d_thres,
        toler_st_abs=toler_st_abs, toler_st_rel=toler_st_rel, maxiter_st=maxiter_st,
        early_exit=True,
    )


def _staggered_loop(
    n, l0, formulation, control, stress_goal_arr,
    C_field, lam_vox, mu_vox, xi_flat, k_res_arr, Gc, Gc_heterogeneous, green_op,
    eps_bar, d_init, H_init,
    toler_lin, maxiter_cg, toler_helm, maxiter_helm, eta, dt, d_thres,
    toler_st_abs, toler_st_rel, maxiter_st, early_exit: bool,
) -> FractureSolution:
    """
    The mechanics<->damage staggered fixed-point loop shared by solve_fracture
    (early_exit=True: Python-level break on convergence, err_abs/err_rel/
    converged_staggered as concrete Python float/bool -- exactly
    solve_fracture's original inline behaviour, moved here unchanged) and
    solve_fracture_fixed (early_exit=False: always runs exactly maxiter_st
    iterations, err_abs/err_rel/converged_staggered kept as JAX arrays
    throughout -- no float()/bool() concretization, no data-dependent Python
    control flow, so this path is safe under jax.vmap/jax.jit).
    """
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

        damage_solve = solve_damage_helmholtz_cg_het if Gc_heterogeneous else solve_damage_helmholtz_cg
        d_st, converged_helm = damage_solve(
            H_st, xi_flat, n, l0, Gc, d_prev_st,
            toler_cg=toler_helm, maxiter=maxiter_helm, eta=eta, dt=dt, k=k_res_arr,
        )

        diff = jnp.max(jnp.abs(d_st - d_prev_st))
        if early_exit:
            # concrete Python float/bool -- fine outside vmap/jit (eager),
            # which is the only context solve_fracture is ever called in.
            err_abs = float(diff)
            err_rel = float(diff / (jnp.max(jnp.abs(d_st)) + 1e-30))
            staggered_converged = err_abs < toler_st_abs or err_rel < toler_st_rel
            if staggered_converged:
                break
        else:
            # stay array-valued -- float()/bool() on a batched/traced value
            # would raise under vmap; every batch lane must run the same
            # (here: all maxiter_st) iterations regardless of its own
            # convergence, so there's nothing to break out of anyway.
            err_abs = diff
            err_rel = diff / (jnp.max(jnp.abs(d_st)) + 1e-30)
            staggered_converged = jnp.logical_or(err_abs < toler_st_abs, err_rel < toler_st_rel)

    return FractureSolution(
        eps=eps, sigma=sigma, delta=delta, d=d_st, H=H_st, psi_pos=psi_pos,
        converged_mech=converged_mech, converged_helm=converged_helm,
        converged_staggered=staggered_converged,
        iter_staggered=iter_st, err_abs=err_abs, err_rel=err_rel,
        eps_bar=eps_bar_cur,
    )


def solve_fracture_fixed(
    n:            Tuple[int, ...],
    L:            Tuple[float, ...],
    phase:        jnp.ndarray,
    materials:    list,
    eps_bar:      jnp.ndarray,
    l0:           float,
    Gc:           float | jnp.ndarray | None,
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
    vmap/jit-safe counterpart to solve_fracture: identical physics and
    parameters, but the staggered loop always runs exactly maxiter_st
    iterations -- no early break on convergence -- so every batch lane
    under jax.vmap takes the same, data-independent control flow (no
    per-lane Python bool()/float() concretization either). Trades some
    wasted staggered iterations (every increment always runs to
    maxiter_st) for actually being traceable/batchable.

    Convergence is NOT enforced or checked here -- inspect the returned
    FractureSolution's .converged_staggered/.converged_mech/.converged_helm
    yourself (per batch lane, after leaving vmap) instead of relying on a
    driver to raise for you.

    See solve_fracture's docstring for every parameter's meaning; unlike it,
    there's no early-exit path, so there's nothing else different here.
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

    # Gc_heterogeneous is decided WITHOUT inspecting any array derived from
    # `phase` (float()/bool() on a value that traces back to `phase` breaks
    # under jax.vmap, since `phase` is exactly the argument a caller batches
    # different RVE realizations over). Gc=None gathers per-material -- its
    # heterogeneity is a static property of `materials` (plain Python floats),
    # not something to inspect on the resulting per-voxel field. An explicit
    # array Gc is trusted at face value (no auto-collapse-if-uniform check,
    # unlike solve_fracture) -- pass a scalar yourself if you know it's
    # uniform and want the cheaper homogeneous solver.
    if Gc is None:
        Gc = Gc_field(materials, phase)
        Gc_values = {float(m.Gc) for m in materials if m.Gc is not None}
        Gc_heterogeneous = len(Gc_values) > 1
    else:
        Gc_heterogeneous = isinstance(Gc, jnp.ndarray) and jnp.ndim(Gc) > 0

    if formulation == "lippmann_schwinger":
        green_op = build_reference_green_operator(n, L, materials, scheme=scheme)
    else:
        green_op = None

    return _staggered_loop(
        n=n, l0=l0, formulation=formulation, control=control, stress_goal_arr=stress_goal_arr,
        C_field=C_field, lam_vox=lam_vox, mu_vox=mu_vox, xi_flat=xi_flat, k_res_arr=k_res_arr,
        Gc=Gc, Gc_heterogeneous=Gc_heterogeneous, green_op=green_op,
        eps_bar=eps_bar, d_init=d_init, H_init=H_init,
        toler_lin=toler_lin, maxiter_cg=maxiter_cg,
        toler_helm=toler_helm, maxiter_helm=maxiter_helm,
        eta=eta, dt=dt, d_thres=d_thres,
        toler_st_abs=toler_st_abs, toler_st_rel=toler_st_rel, maxiter_st=maxiter_st,
        early_exit=False,
    )


def solve_fracture_incremental_fixed(
    n:            Tuple[int, ...],
    L:            Tuple[float, ...],
    phase:        jnp.ndarray,
    materials:    list,
    eps_bar:      jnp.ndarray,
    l0:           float,
    Gc:           float | jnp.ndarray | None,
    d_init:       jnp.ndarray,
    H_init:       jnp.ndarray,
    dt_step:      float,
    formulation:  str = "lippmann_schwinger",
    scheme:       str = "rotated",
    control:      Tuple[Tuple[int, ...], ...] | None = None,
    stress_goal:  jnp.ndarray | None = None,
    toler_lin:    float = 1e-6,
    maxiter_cg:   int = 1000,
    toler_helm:   float = 1e-4,
    maxiter_helm: int = 300,
    eta:          float = 0.0,
    d_thres:      float = 0.95,
    k_res:        float | jnp.ndarray | None = None,
    toler_st_abs: float = 1e-2,
    toler_st_rel: float = 1e-3,
    maxiter_st:   int = 200,
) -> list[FractureSolution]:
    """
    vmap/jit-safe counterpart to solve_fracture_incremental(stepping="fixed"),
    built on solve_fracture_fixed instead of solve_fracture -- so both the
    staggered loop AND the load-stepping loop are free of data-dependent
    Python control flow (no bool()/float() on a convergence value, no break,
    no cutback, no raise-on-non-convergence). ``n_steps = round(1/dt_step)``
    is a static Python int, identical for every batch lane, so the whole
    thing unrolls to one fixed, uniform computation graph -- this is the
    function to jax.vmap over multiple RVE realizations (batch ``phase``,
    keep n/L/materials/eps_bar/... shared via in_axes=None), e.g. to sample
    several random-seed realizations at the same fibre volume fraction.

    Convergence is never checked or enforced -- unlike solve_fixed, this
    NEVER raises; a non-converged increment just keeps going with whatever
    (possibly still-transient) d/H it produced. Inspect each returned
    FractureSolution's .converged_staggered yourself, per batch lane, after
    leaving vmap.

    Returns list[FractureSolution], one per fixed step -- NOT
    list[IncrementResult] like solve_fracture_incremental: there's no
    wall-clock/writer/on_increment bookkeeping here, none of which make
    sense once this runs inside a jax.vmap batch. See solve_fracture's
    docstring for every other parameter's meaning.
    """
    if not (0.0 < dt_step <= 1.0):
        raise ValueError(f"dt_step must be in (0, 1], got {dt_step}")
    n_steps = max(1, round(1.0 / dt_step))
    dt_actual = 1.0 / n_steps

    d, H = d_init, H_init
    results = []
    for step in range(1, n_steps + 1):
        t = step * dt_actual
        sol = solve_fracture_fixed(
            n, L, phase, materials, t * eps_bar, l0, Gc, d, H,
            formulation=formulation, scheme=scheme, control=control, stress_goal=stress_goal,
            toler_lin=toler_lin, maxiter_cg=maxiter_cg,
            toler_helm=toler_helm, maxiter_helm=maxiter_helm,
            eta=eta, dt=dt_actual, d_thres=d_thres, k_res=k_res,
            toler_st_abs=toler_st_abs, toler_st_rel=toler_st_rel, maxiter_st=maxiter_st,
        )
        d, H = sol.d, sol.H
        results.append(sol)
    return results


def solve_fracture_incremental(
    n:            Tuple[int, ...],
    L:            Tuple[float, ...],
    phase:        jnp.ndarray,
    materials:    list,
    eps_bar:      jnp.ndarray,
    l0:           float,
    Gc:           float | jnp.ndarray | None,
    d_init:       jnp.ndarray,
    H_init:       jnp.ndarray,
    stepping:     str = "single",
    formulation:  str = "lippmann_schwinger",
    scheme:       str = "rotated",
    control:      Tuple[Tuple[int, ...], ...] | None = None,
    stress_goal:  jnp.ndarray | None = None,
    toler_lin:    float = 1e-6,
    maxiter_cg:   int = 1000,
    toler_helm:   float = 1e-4,
    maxiter_helm: int = 300,
    eta:          float = 0.0,
    d_thres:      float = 0.95,
    k_res:        float | jnp.ndarray | None = None,
    toler_st_abs: float = 1e-2,
    toler_st_rel: float = 1e-3,
    maxiter_st:   int = 200,
    dt_step:      float | None = None,
    dt_init:      float = 0.1,
    dt_min:       float = 1e-4,
    dt_max:       float = 0.5,
    factor_inc:   float = 1.5,
    factor_dec:   float = 0.5,
    max_cutbacks: int = 5,
    max_steps:    int = 1000,
    writer:       IncrementalWriter | None = None,
    orientation:  jnp.ndarray | None = None,
    on_increment: Callable[[IncrementResult, float], None] | None = None,
) -> list[IncrementResult]:
    """
    solve_fracture, load-stepped up to the target eps_bar (Abaqus-*STATIC
    style) via problems.incremental -- same pattern as
    problems.mechanics.solve_mechanics, with one addition: the
    damage/history state (d, H) is path-dependent, so it must be threaded
    from one accepted increment to the next rather than recomputed per call.

    That threading is done via a plain mutable closure over ``_state``,
    updated *only* inside ``on_increment`` -- which problems.incremental
    already guarantees fires exactly once per *accepted* increment, never
    on a rejected cutback attempt (see solve_automatic's docstring). This
    means solve_fn itself must stay read-only with respect to ``_state``:
    solve_automatic may call solve_fn several times per increment (one per
    cutback attempt) before accepting one, all of which must start from the
    same last-accepted (d, H), not from whatever a previous, rejected,
    too-large-dt attempt produced.

    ``dt_step`` is the load-stepping *target* increment size (problems.
    incremental's ``dt`` argument to solve_fixed) -- named differently here
    because solve_fracture's own ``dt`` parameter means something else (the
    damage equation's viscous-regularisation timestep). This function does
    not take that ``dt`` directly: it derives the *actual* per-increment
    value itself, as ``t - t_prev`` (the real gap between this increment's
    cumulative load fraction and the last *accepted* one), and passes that
    to solve_fracture. This matters because the real step size isn't always
    dt_step -- solve_fixed's last step can be shorter, and solve_automatic's
    varies every increment by construction -- so a fixed viscosity timestep
    would silently apply the wrong eta/dt strength on any increment whose
    real size differs from whatever value happened to be passed in.
    ``t_prev`` is tracked the same read-in-solve_fn/write-in-on_increment
    way as ``d``/``H``, for the same cutback-safety reason.

    ``stepping``, ``writer``, ``orientation``, ``on_increment`` -- see
    problems.mechanics.solve_mechanics's docstring; the only difference is
    the extra "damage" field written per increment here.
    All other parameters are solve_fracture's own.

    Returns
    -------
    list[IncrementResult] -- .solution is a FractureSolution per increment.
    """
    _state = {"d": d_init, "H": H_init, "t_prev": 0.0}

    def solve_fn(t: float) -> FractureSolution:
        dt_actual = t - _state["t_prev"]
        return solve_fracture(
            n, L, phase, materials, t * eps_bar, l0, Gc,
            d_init=_state["d"], H_init=_state["H"],
            formulation=formulation, scheme=scheme, control=control, stress_goal=stress_goal,
            toler_lin=toler_lin, maxiter_cg=maxiter_cg,
            toler_helm=toler_helm, maxiter_helm=maxiter_helm,
            eta=eta, dt=dt_actual, d_thres=d_thres, k_res=k_res,
            toler_st_abs=toler_st_abs, toler_st_rel=toler_st_rel, maxiter_st=maxiter_st,
        )

    def _on_increment(result: IncrementResult) -> None:
        sol = cast(FractureSolution, result.solution)
        _state["d"] = sol.d
        _state["H"] = sol.H
        _state["t_prev"] = result.t

        write_time = 0.0
        if writer is not None:
            t0 = time.perf_counter()
            eps_grid   = field_to_grid(sol.eps, n)
            sigma_grid = field_to_grid(sol.sigma, n)
            sigma_vm   = von_mises(sigma_grid)
            eps_bar_u  = sol.eps_bar if sol.eps_bar is not None else result.t * eps_bar
            u_grid     = compute_displacement(sol.eps, eps_bar_u, n, L)
            fields = {
                "phase":        np.asarray(phase).reshape(n).astype(np.float32),
                "strain":       to_voigt(eps_grid).astype(np.float64),
                "stress":       to_voigt(sigma_grid).astype(np.float64),
                "von_mises":    sigma_vm.astype(np.float64),
                "displacement": u_grid.astype(np.float64),
                "damage":       np.asarray(sol.d).reshape(n).astype(np.float64),
                "strain_energy_pos": np.asarray(sol.psi_pos).reshape(n).astype(np.float64),
            }
            if orientation is not None:
                fields["orientation"] = np.asarray(orientation).T.reshape(*n, 3).astype(np.float64)
            writer.write_increment(result.step, fields, time=result.t)
            write_time = time.perf_counter() - t0
        if on_increment is not None:
            on_increment(result, write_time)

    if stepping == "single":
        t0 = time.perf_counter()
        sol = solve_fn(1.0)
        result = IncrementResult(step=1, t=1.0, dt=1.0, solution=sol,
                                  wall_time=time.perf_counter() - t0)
        _on_increment(result)
        return [result]
    elif stepping == "fixed":
        if dt_step is None:
            raise ValueError("stepping='fixed' requires dt_step")
        return solve_fixed(solve_fn, dt_step, on_increment=_on_increment)
    elif stepping == "automatic":
        return solve_automatic(
            solve_fn, dt_init=dt_init, dt_min=dt_min, dt_max=dt_max,
            factor_inc=factor_inc, factor_dec=factor_dec,
            max_cutbacks=max_cutbacks, max_steps=max_steps,
            on_increment=_on_increment,
        )
    else:
        raise ValueError(
            f"unknown stepping {stepping!r}, expected 'single', 'fixed', or 'automatic'"
        )
