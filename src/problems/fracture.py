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
from typing import Callable, Tuple, cast

import jax.numpy as jnp
import numpy as np

from materialmodels.assembly import (assemble_C_field,
                                     assemble_inelastic_pff_local_update,
                                     assemble_pff_local_update)
from materialmodels.phasefield.degradation import Gc_field, k_res_field
from materialmodels.phasefield.driving_force import lame_field, strain_energy_amor_split, update_history_hybrid
from operators.green import build_freq_grid, build_reference_green_operator
from post.fields import compute_displacement, field_to_grid, to_voigt, von_mises
from problems.incremental import IncrementResult, solve_automatic, solve_fixed
from problems.mechanics import solve_displacement_based_nonlinear
from solvers.elliptic.scalar import solve_damage_helmholtz_cg, solve_damage_helmholtz_cg_het
from solvers.elliptic.vector.displacement_based import solve_displacement_based
from solvers.elliptic.vector.lippmann_schwinger import solve_lippmann_schwinger
from solvers.solution import FractureSolution
from utils.io.xdmf_writer import IncrementalWriter

_ZERO_CONTROL = ((0, 0, 0), (0, 0, 0), (0, 0, 0))


def _has_plastic_pff(materials) -> bool:
    """
    Whether any material is a STATEFUL phase-field material, i.e. carries
    plastic state as well as reacting to damage
    (materialmodels.phasefield.drucker_prager.PhaseFieldDruckerPrager).

    Duck-typed on ``driving_force_field``, the same marker
    materialmodels.assembly.assemble_inelastic_pff_local_update dispatches on
    -- so a future coupled model needs no edit here either. Static (plain
    Python over the materials list), never traced.
    """
    return any(hasattr(m, "driving_force_field") for m in materials)


def _require_newton_compatible(formulation: str, materials) -> None:
    """Raise for the combinations the nonlinear mechanical path cannot do."""
    if formulation != "displacement":
        names = [str(m) for m in materials if hasattr(m, "driving_force_field")]
        raise ValueError(
            f"formulation={formulation!r} cannot be used with a stateful "
            f"phase-field material ({names[0]}) -- its mechanical sub-solve is "
            "a Newton iteration on the true heterogeneous tangent, which only "
            "the displacement-based formulation provides; the "
            "lippmann_schwinger reference-medium form has no nonlinear path. "
            "Use formulation='displacement'"
        )


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
    state_init:   object | None = None,
    toler_nr:     float = 1e-8,
    maxiter_nr:   int = 50,
    d_cap:        float = 1.0,
    eps_init:     jnp.ndarray | None = None,
    line_search:  int = 0,
) -> FractureSolution:
    """
    Solve one staggered mechanics<->phase-field increment under a
    prescribed macroscopic strain (and, for formulation="displacement",
    optionally mixed strain/stress control).

    Per staggered iteration
    ------------------------
        1. Degraded tangent:   C_eff = pff_local_update(eps_prev, d) -- per-
                                material stress/tangent (materialmodels.
                                assembly.assemble_pff_local_update), evaluated
                                at the PREVIOUS iteration's eps (frozen-
                                tangent scheme -- see _staggered_loop's
                                docstring for why)
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
    d_cap       : float, ceiling on the damage field. 1.0 (default) is the
                  unmodified AT2 model, where a fully cracked voxel keeps only
                  ``k_res`` of its stiffness (1e-6 by default).

                  Below 1 this is a CONDITIONING regularization, and a
                  deliberate modelling change: a crack retains
                  g(d_cap) = (1 - d_cap)^2 + k_res of its stiffness, so at
                  d_cap = 0.975 that is 6.3e-4 -- roughly 600x stiffer than
                  k_res. The point is the tangent. As d -> 1 the degraded
                  tangent collapses toward k_res*C and the mechanical Newton
                  solve stops converging (the failure documented at length in
                  materialmodels/phasefield/drucker_prager.py, whose cause is
                  not established); holding d below 1 keeps the operator
                  non-degenerate.

                  The cost is physical: the crack never fully opens, so it
                  transmits a small spurious stress and the dissipated energy
                  is slightly under-counted. Choose d_cap as loose as the
                  solve tolerates -- it trades fidelity for solvability, and
                  a result obtained with it should say so.
    eps_init    : (3, 3, Nv) or None -- the strain field the FIRST staggered
                  sweep starts from, normally the previous accepted
                  increment's converged ``eps``. None (default) starts from
                  the uniform ``eps_bar``, with zero strain on every
                  stress-controlled direction.

                  On the newton path this is the Newton initial guess, and
                  it is not optional in practice. Measured on the 40^3
                  two-fibre cell (configs/simulation/
                  fracture_rve_square_2fibre_hotgap_transverse.yaml), t = 0.41
                  -> 0.45: from the uniform guess, which also drops the
                  converged -1.5% lateral contraction, the first Newton step
                  moves the strain by up to 0.17, flips every matrix voxel
                  into hydrostatic compression, and Newton then locks into a
                  period-2 cycle (residual 1.8e6 <-> 2.1e6) for all 30
                  iterations, sweep after sweep -- 29 sweeps, 1213 s, not
                  converged. From the previous increment's strain the same
                  increment converges in 5 sweeps of 4-6 Newton iterations
                  each, 51 s. On the linear path it is the frozen-tangent
                  evaluation point of the first sweep (see _staggered_loop).
    line_search : newton path only -- max step halvings of the backtracking
                  line search in problems.mechanics.
                  solve_displacement_based_nonlinear. 0 (default) takes full
                  Newton steps. In a cracked RVE the tangent is indefinite in
                  partly damaged compressed matrix voxels and CG stops at its
                  cap with relres 0.1-1, so full steps overshoot: measured on
                  the 60^3 bi cell (sim/sim/12_mini_eta, eps_22 2.05 ->
                  2.25%), the residual zigzagged (6.6e-4 -> 3.6e-3 -> 3.3e-4
                  -> ...) and Newton missed 1e-6 in 30 iterations in every
                  sweep from the second on. With 4 halvings it fell
                  monotonically and converged in all three sweeps tested.

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

    # A stateful phase-field material (PhaseFieldDruckerPrager) makes the
    # mechanical sub-problem genuinely nonlinear and path-dependent, so the
    # frozen-tangent linear solve cannot represent it at all -- switch to the
    # Newton driver. Decided from `materials`, which is static Python, so no
    # new config key is needed; see _staggered_loop for the two paths.
    mechanics = "newton" if _has_plastic_pff(materials) else "linear"
    if mechanics == "newton":
        _require_newton_compatible(formulation, materials)
        nl_local_update, nl_driving_force, state0 = (
            assemble_inelastic_pff_local_update(materials, phase))
        state_init = state0 if state_init is None else state_init
        # Fixed elastic reference for the CG preconditioner -- see
        # _staggered_loop's docstring for why not the default.
        C0 = jnp.mean(assemble_C_field(materials, phase), axis=-1)
        pff_local_update = None
    else:
        nl_local_update = nl_driving_force = C0 = None
        pff_local_update = assemble_pff_local_update(materials, phase)

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
        pff_local_update=pff_local_update, lam_vox=lam_vox, mu_vox=mu_vox, xi_flat=xi_flat, k_res_arr=k_res_arr,
        Gc=Gc, Gc_heterogeneous=Gc_heterogeneous, green_op=green_op,
        eps_bar=eps_bar, d_init=d_init, H_init=H_init,
        toler_lin=toler_lin, maxiter_cg=maxiter_cg,
        toler_helm=toler_helm, maxiter_helm=maxiter_helm,
        eta=eta, dt=dt, d_thres=d_thres,
        toler_st_abs=toler_st_abs, toler_st_rel=toler_st_rel, maxiter_st=maxiter_st,
        early_exit=True,
        mechanics=mechanics, nl_local_update=nl_local_update,
        nl_driving_force=nl_driving_force, state_init=state_init,
        toler_nr=toler_nr, maxiter_nr=maxiter_nr, C0=C0,
        d_cap=d_cap, eps_init=eps_init,
        line_search=line_search,
    )


def _staggered_loop(
    n, l0, formulation, control, stress_goal_arr,
    pff_local_update, lam_vox, mu_vox, xi_flat, k_res_arr, Gc, Gc_heterogeneous, green_op,
    eps_bar, d_init, H_init,
    toler_lin, maxiter_cg, toler_helm, maxiter_helm, eta, dt, d_thres,
    toler_st_abs, toler_st_rel, maxiter_st, early_exit: bool,
    mechanics: str = "linear",
    nl_local_update=None, nl_driving_force=None, state_init=None,
    toler_nr: float = 1e-8, maxiter_nr: int = 50, C0=None,
    d_cap: float = 1.0, eps_init=None,
    line_search: int = 0,
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

    Frozen-tangent mechanical sub-solve: ``pff_local_update``
    (materialmodels.assembly.assemble_pff_local_update) is evaluated at
    ``eps_prev`` -- the PREVIOUS staggered iteration's converged strain,
    seeded from eps_bar before the first iteration -- rather than the
    current, still-unknown ``eps``. A PhaseFieldIsotropic material's stress
    is nonlinear in eps even at fixed d (the Amor split), so a fully
    consistent tangent would need eps itself, which only the mechanical
    solve about to run produces; freezing it at the previous iterate keeps
    this loop's linear CG solvers (solve_lippmann_schwinger/
    solve_displacement_based) usable unchanged instead of a nonlinear
    Newton-CG mechanical sub-solve, exactly mirroring how ``d`` is already
    frozen within one mechanical solve and only updated once per staggered
    iteration. ``eps_prev`` is refreshed every iteration, so the loop still
    converges to the fully self-consistent solution at staggered
    convergence (eps_prev == eps there). Materials without ``psi_split``
    (plain LinearElasticIsotropic) ignore eps_prev entirely in
    ``pff_local_update``'s fallback branch, so this changes nothing for an
    all-elastic ``materials`` list.

    ``mechanics`` picks the mechanical sub-solve. "linear" is exactly the
    behaviour described above, unchanged:

    - "linear" -- one linear CG solve against the frozen degraded tangent
      (solve_lippmann_schwinger / solve_displacement_based).
    - "newton" -- problems.mechanics.solve_displacement_based_nonlinear, for a
      materials list holding a stateful phase-field material
      (materialmodels.phasefield.drucker_prager.PhaseFieldDruckerPrager),
      whose stress is genuinely path-dependent and so cannot be reached by a
      frozen-tangent linear solve at all. ``d`` is still frozen per sub-solve:
      the loop closes over it to hand that solver the two-argument
      (eps, state) signature it requires, which also means the tangent it
      differentiates is dsigma/deps at fixed d -- the right one here.

    Three things the "newton" branch has to get right, all about WHICH state
    each solve starts from:

    - ``state_init`` stays FROZEN across the whole staggered loop. Every
      staggered iteration re-runs the return mapping from the increment's last
      ACCEPTED plastic state, never from the previous staggered iterate's.
      Same argument, one level up, as the frozen-state comment inside
      solve_displacement_based_nonlinear's own Newton loop: advancing it per
      iteration ratchets plastic flow once per iteration and the stress drifts
      under the fixed point.
    - ``delta``/``eps_bar_free`` ARE carried across staggered iterations, as a
      warm start -- d changes between iterations and eps changes with it, so
      the previous iterate is much the better guess. Both are recovered from
      the returned eps, since that solver returns neither: delta is zero-mean
      by construction and eps0 = eps_bar*(1 - control), so mean(eps) is the
      resolved macroscopic strain and eps - mean(eps) the fluctuation. The
      first sweep takes the same split of ``eps_init``, the previous
      increment's strain -- see solve_fracture's docstring for what starting
      it from the uniform eps_bar instead does to Newton.
    - ``C0`` is passed as a FIXED elastic reference, which saves rebuilding and
      inverting the preconditioner every Newton iteration. It was expected to
      also help conditioning as the crack grows -- the default rebuilds from
      the voxel-mean of a partly cracked tangent, which looked like a poor
      reference for the intact material carrying the load. MEASURED, it makes
      no difference at all: on fixed localized-crack fields spanning
      d = 0.5 .. 0.9999 (a 1e6 stiffness contrast) the two choices give
      identical iteration counts and identical stresses to every printed
      digit. It is kept only for the saved work per iteration, not for
      robustness.
    """
    if mechanics not in ("linear", "newton"):
        raise ValueError(
            f"unknown mechanics {mechanics!r}, expected 'linear' or 'newton'"
        )
    d_st = d_init
    H_st = H_init
    eps_bar_cur = None
    Nv = lam_vox.shape[0]
    control_arr = jnp.asarray(control, dtype=eps_bar.dtype)

    state_st = state_init
    iter_nr = None
    if eps_init is None:
        eps_prev = jnp.ones((3, 3, Nv)) * eps_bar[:, :, None]
        delta_st = None
        eps_bar_free_st = None
    else:
        # Same mean/fluctuation split the loop applies to each sweep's result
        # below. Only the stress-controlled part of the mean is kept: the
        # strain-controlled entries are this increment's eps_bar, not the
        # previous one's, and the Newton solver re-projects delta to zero
        # mean regardless.
        eps_prev = eps_init
        eps_bar_init = jnp.mean(eps_init, axis=-1)
        delta_st = eps_init - eps_bar_init[:, :, None]
        eps_bar_free_st = control_arr * eps_bar_init

    for iter_st in range(1, maxiter_st + 1):
        d_prev_st = d_st

        if mechanics == "newton":
            def _nl(e, st, _d=d_st):
                return nl_local_update(e, _d, st)

            eps, sigma, state_new, converged_mech, iter_nr, res_nr = (
                solve_displacement_based_nonlinear(
                    n, xi_flat, eps_bar, _nl, state_init,
                    toler_lin=toler_lin, maxiter_lin=maxiter_cg,
                    toler_nr=toler_nr, maxiter_nr=maxiter_nr, C0=C0,
                    delta_init=delta_st, eps_bar_free_init=eps_bar_free_st,
                    control=control, stress_goal=stress_goal_arr,
                    line_search=line_search, return_residual=True,
                )
            )
            # A FAILED Newton returns the increment-START plastic state (its
            # contract for callers that ignore the flag). The driving force
            # below must not see that: it would build psi+ and W_p from
            # plastic flow that is already gone from the strain. Converged
            # sweeps then see this increment's plastic work and failed ones
            # do not, so H jumps on every converged sweep. Measured on the
            # 60^3 bi cell (eta 0.25, eps_22 ~ 2.9%, Newton at a 1-3e-5
            # floor): a four-sweep cycle -- Newton converges, err_abs jumps
            # to 3-5e-3, three failed sweeps settle it to 3e-4, repeat -- for
            # 168 sweeps. Return-map at the returned strain instead, so every
            # sweep's driving force is consistent with its own strain.
            if not bool(converged_mech):
                _, _, state_new = nl_local_update(eps, d_st, state_init)
            # Recover what that solver does not return, for the next
            # iteration's warm start and for FractureSolution -- see above.
            eps_bar_cur = jnp.mean(eps, axis=-1)
            delta = eps - eps_bar_cur[:, :, None]
            delta_st = delta
            eps_bar_free_st = control_arr * eps_bar_cur
            state_st = state_new
        elif formulation == "lippmann_schwinger":
            _, C_eff = pff_local_update(eps_prev, d_st)
            eps, sigma, delta, converged_mech = solve_lippmann_schwinger(
                n, C_eff, green_op, eps_bar,
                toler_lin=toler_lin, maxiter=maxiter_cg,
            )
        else:  # "displacement"
            _, C_eff = pff_local_update(eps_prev, d_st)
            eps, sigma, delta, eps_bar_cur, converged_mech = solve_displacement_based(
                n, C_eff, xi_flat, eps_bar, control, stress_goal_arr,
                toler_lin=toler_lin, maxiter=maxiter_cg,
            )
        eps_prev = eps

        if mechanics == "newton":
            # The driving force has to come from the materials here: psi+ is
            # built on the ELASTIC strain eps - eps_p (plastic strain stores no
            # energy), and it carries each phase's own beta_p*W_p term --
            # neither reconstructible from eps alone.
            drive, psi_pos = nl_driving_force(eps, state_st)
        else:
            drive, _ = strain_energy_amor_split(eps, lam_vox, mu_vox)
            psi_pos = drive
        H_st = update_history_hybrid(H_st, drive, d_prev_st, d_thres=d_thres)

        damage_solve = solve_damage_helmholtz_cg_het if Gc_heterogeneous else solve_damage_helmholtz_cg
        d_picard, converged_helm = damage_solve(
            H_st, xi_flat, n, l0, Gc, d_prev_st,
            toler_cg=toler_helm, maxiter=maxiter_helm, eta=eta, dt=dt, k=k_res_arr,
        )

        # Cap the damage BEFORE it becomes the iterate, so both the fixed
        # point and the accelerator see the capped map. d_cap < 1 leaves a
        # crack with g(d_cap) = (1 - d_cap)^2 + k_res residual stiffness
        # instead of k_res, which is what keeps the tangent from degenerating
        # -- see solve_fracture's docstring for the trade that makes.
        if d_cap < 1.0:
            d_picard = jnp.minimum(d_picard, d_cap)

        d_st = d_picard

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

    eps_p_out, alpha_out, W_p_out = (
        state_st if mechanics == "newton" else (None, None, None))
    if mechanics == "newton":
        # Fold the mechanical result into the increment's own convergence, so
        # solve_automatic's cutback can actually see it. FractureSolution.
        # converged returns converged_staggered alone, which is defensible on
        # the linear path (a failed CG there is rare and its damage update is
        # still a contraction) but not here: a failed Newton sub-solve returns
        # a non-equilibrium state -- observed as a small NEGATIVE macroscopic
        # stress under applied tension. Without this, the stepper accepts
        # that increment and carries the bad plastic state forward. The
        # failures behind that observation were cold Newton starts, which
        # eps_init removes (see solve_fracture), but Newton can still fail on
        # too large a step, so the fold stays.
        #
        # Only the newton path is changed; the linear path's reporting is left
        # exactly as it was.
        if early_exit:
            staggered_converged = bool(staggered_converged) and bool(converged_mech)
        else:
            staggered_converged = jnp.logical_and(
                staggered_converged, jnp.asarray(converged_mech))
    return FractureSolution(
        eps=eps, sigma=sigma, delta=delta, d=d_st, H=H_st, psi_pos=psi_pos,
        converged_mech=converged_mech, converged_helm=converged_helm,
        converged_staggered=staggered_converged,
        iter_staggered=iter_st, err_abs=err_abs, err_rel=err_rel,
        eps_bar=eps_bar_cur,
        eps_p=eps_p_out, alpha=alpha_out, W_p=W_p_out, iter_nr=iter_nr,
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
    # No stateful phase-field material here, and this is not a limitation that
    # can be worked around at this level: the nonlinear mechanical sub-solve
    # such a material needs is solve_displacement_based_nonlinear, which
    # concretizes its Newton residual to a Python float, so the path cannot be
    # traced -- and being traceable under jax.vmap is the only reason these
    # _fixed entry points exist. solve_fracture_incremental_fixed reaches this
    # check through here too.
    if _has_plastic_pff(materials):
        raise ValueError(
            "solve_fracture_fixed cannot take a stateful phase-field material "
            "(PhaseFieldDruckerPrager): its mechanical sub-solve is a Newton "
            "iteration whose residual is concretized to a Python float, so it "
            "cannot run under jax.jit or jax.vmap. Use solve_fracture or "
            "solve_fracture_incremental instead -- eager, with early exit and "
            "load-step cutback"
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

    pff_local_update = assemble_pff_local_update(materials, phase)
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
        pff_local_update=pff_local_update, lam_vox=lam_vox, mu_vox=mu_vox, xi_flat=xi_flat, k_res_arr=k_res_arr,
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
    toler_nr:     float = 1e-8,
    maxiter_nr:   int = 50,
    line_search:  int = 0,
    write_plastic_strain_tensor: bool = False,
    d_cap:        float = 1.0,
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
    # ``state`` (the plastic state, only for a stateful phase-field material)
    # is threaded here exactly like d/H/t_prev, and for the same reason: it is
    # path-dependent, and solve_automatic may call solve_fn several times per
    # increment before accepting one. Writing it ONLY in _on_increment -- which
    # problems.incremental guarantees fires once per ACCEPTED increment -- is
    # what keeps a rejected cutback attempt from advancing plastic flow.
    # ``eps`` is the last accepted increment's strain, the first sweep's
    # starting point in solve_fracture (eps_init) -- threaded the same way as
    # d/H, so a cutback retry restarts from the accepted strain and never from
    # a rejected attempt's.
    _state = {"d": d_init, "H": H_init, "t_prev": 0.0, "plastic": None, "eps": None}

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
            state_init=_state["plastic"], toler_nr=toler_nr, maxiter_nr=maxiter_nr,
            line_search=line_search,
            d_cap=d_cap, eps_init=_state["eps"],
        )

    def _on_increment(result: IncrementResult) -> None:
        sol = cast(FractureSolution, result.solution)
        _state["d"] = sol.d
        _state["H"] = sol.H
        _state["t_prev"] = result.t
        _state["eps"] = sol.eps
        if sol.eps_p is not None:
            _state["plastic"] = (sol.eps_p, sol.alpha, sol.W_p)

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
            # Plastic output, only when a stateful phase-field material made
            # the run plastic at all. Field names match
            # scripts/simulation/solve_inelastic.py's ("strain_p" = alpha, the
            # eps_p tensor behind the same opt-in) so
            # scripts/postprocessing/homogenize.py picks them up unchanged.
            if sol.alpha is not None:
                fields["strain_p"] = np.asarray(sol.alpha).reshape(n).astype(np.float64)
                fields["plastic_work"] = np.asarray(sol.W_p).reshape(n).astype(np.float64)
                if write_plastic_strain_tensor:
                    fields["eps_p"] = to_voigt(
                        field_to_grid(sol.eps_p, n)).astype(np.float64)
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
