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
from solvers.damage.anderson import build_accelerator
from solvers.elliptic.scalar import solve_damage_helmholtz_cg, solve_damage_helmholtz_cg_het
from solvers.elliptic.vector.displacement_based import solve_displacement_based
from solvers.elliptic.vector.lippmann_schwinger import solve_lippmann_schwinger
from solvers.solution import FractureSolution
from utils.io.xdmf_writer import IncrementalWriter

_ZERO_CONTROL = ((0, 0, 0), (0, 0, 0), (0, 0, 0))
# Consecutive sweeps with a converged damage field but a failed Newton solve
# after which _staggered_loop gives up on the increment -- see the comment at
# its use.
_NR_STALL_SWEEPS = 10   # Newton converged ~1 sweep in 4 at its floor on the bi cell;
                        # at 5, P(premature give-up) ~ 0.75^5 = 24%, at 10 ~ 6%


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
    accelerator:  object | None = None,
    d_cap:        float = 1.0,
    eps_init:     jnp.ndarray | None = None,
    log_sweeps:   bool = False,
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
    eta, dt                  : viscous regularisation of the damage equation,
                                (eta/dt)*(d - d_init) with d_init the
                                increment's starting damage. It is what
                                makes a load cutback stabilize a softening
                                increment: the staggered map's gain falls as
                                eta/dt grows. With eta ~ 0 an unstable
                                increment diverges at any dt -- see the
                                damage-solve comment in _staggered_loop.
                                A MODELLING choice, not a free solver knob:
                                it makes damage rate-dependent, with a lag
                                of roughly eta/(Gc/l0) in units of dt. dt
                                is the LOAD FRACTION here, so the same eta
                                means a different strain lag when eps_bar
                                changes
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
    accelerator : staggered accelerator instance (solvers.damage.anderson) or
                  None for plain Picard.
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
    log_sweeps  : print one line per staggered sweep (residual, d_max, and on
                  the newton path the Newton flag and iteration count).
                  Default False, which prints nothing.
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
        accelerator=accelerator, d_cap=d_cap, eps_init=eps_init,
        log_sweeps=log_sweeps, line_search=line_search,
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
    accelerator=None, d_cap: float = 1.0, eps_init=None,
    log_sweeps: bool = False, line_search: int = 0,
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

    # One increment = one fixed point, so the history must not carry across
    # increments: pairs from the previous applied strain extrapolate toward
    # the previous solution. d_init is the irreversibility floor -- an
    # extrapolated iterate is a linear combination with possibly negative
    # weights, so without the floor the accelerator can un-crack voxels.
    if accelerator is not None:
        # The accelerator clamps its own extrapolation, so it needs the same
        # ceiling -- otherwise it can step past d_cap and the cap below would
        # be doing the work of undoing it every sweep.
        accelerator.bounds = (0.0, float(d_cap))
        accelerator.reset(floor=np.asarray(d_init))

    nr_stall = 0
    for iter_st in range(1, maxiter_st + 1):
        d_prev_st = d_st
        t_sweep = time.perf_counter()

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

        # Two different "previous damage" fields go into the damage solve,
        # and they must not be the same one:
        #
        # - d_n, the VISCOUS reference, is d_init. The term (eta/dt)*(d - d_n)
        #   only regularizes if d_n is the last converged increment. With the
        #   previous sweep there (as this loop used to pass) it vanishes at
        #   the staggered fixed point, so eta regularizes nothing: it merely
        #   under-relaxes each sweep, and an under-relaxed map with an
        #   eigenvalue above 1 still diverges. Measured on the 152^3 UD RVE
        #   (configs/simulation/fracture_uni_transverse_pp.yaml),
        #   t = 0.525 -> 0.625, where the original run hit staggered=40:
        #   plain Picard DIVERGES (residual 4.3e-2 -> 8.6e-2 over sweeps 3-7,
        #   total damage accelerating), so no accelerator or cutback can
        #   rescue it while eta is inert. With d_n = d_init and eta = 0.5
        #   (eta/dt = 5) the same increment converges in 5 sweeps, residual
        #   6.3e-2 -> 7.8e-4 at ~0.33 per sweep. See solve_fracture's eta note.
        #
        # - The irreversibility FLOOR stays the previous sweep, d_prev_st. A
        #   floor at d_init lets damage heal back within the increment
        #   wherever hybrid irreversibility leaves H unrestricted (d below
        #   d_thres), and on a localizing crack that oscillates instead of
        #   converging. Measured on
        #   test_materialmodels_phasefield_drucker_prager's seeded cube,
        #   eps_11 = 0.004: floor d_prev_st converges in 22 monotone sweeps;
        #   floor d_init overshoots to d_max 0.992, heals back, and is still
        #   cycling at 1-3e-2 after 30. The price of the per-sweep floor is
        #   that overshoot cannot be taken back, so a converged d satisfies
        #   d >= D(d) rather than d = D(d).
        #
        # At eta = 0 this is bit-for-bit the solve the loop always did; at the
        # eta = 1e-6 the configs so far use, the viscous change is below any
        # printed digit.
        damage_solve = solve_damage_helmholtz_cg_het if Gc_heterogeneous else solve_damage_helmholtz_cg
        d_picard, converged_helm = damage_solve(
            H_st, xi_flat, n, l0, Gc, d_init,
            toler_cg=toler_helm, maxiter=maxiter_helm, eta=eta, dt=dt, k=k_res_arr,
            d_floor=d_prev_st,
        )

        # Cap the damage BEFORE it becomes the iterate, so both the fixed
        # point and the accelerator see the capped map. d_cap < 1 leaves a
        # crack with g(d_cap) = (1 - d_cap)^2 + k_res residual stiffness
        # instead of k_res, which is what keeps the tangent from degenerating
        # -- see solve_fracture's docstring for the trade that makes.
        if d_cap < 1.0:
            d_picard = jnp.minimum(d_picard, d_cap)

        # d_picard IS G(d_prev_st) -- one full sweep of the fixed point. The
        # accelerator recombines it with earlier sweeps (solvers.damage.
        # anderson); with none, d_st is d_picard and this path is bit-for-bit
        # the plain staggering it replaces.
        #
        # The accelerated iterate is floored at d_picard, i.e. it may add
        # damage on top of a Picard sweep but never take less. The
        # accelerator's own floor is d_init (reset above), and NGMRES
        # recombines with negative weights, so on its own it can take back
        # damage that an earlier sweep already locked in -- while every
        # Picard sweep after it re-floors at that lowered field. The two
        # floors disagree, and past the stress peak that is a cycle: on the
        # 60^3 mini RVEs (sim/sim/12_mini_eta, eta = 0.5) all three cases
        # stalled right after the peak, d_max going 0.975 -> 0.961 -> 0.975
        # and err_abs stuck at 2-6e-2 for 20-40 sweeps.
        # Flooring at d_prev_st instead is NOT enough: where the extrapolation
        # lies below d_prev_st but Picard wants to grow, max(cand, d_prev_st)
        # discards the Picard growth, and the loop stalled with an unchanged
        # residual for several sweeps. With d_picard (itself >= d_prev_st) as
        # the floor, every iterate makes at least Picard's progress and stays
        # <= d_cap, so it cannot cycle or stall; overshoot is permanent, the
        # same d >= D(d) price the Picard floor already pays.
        if accelerator is not None:
            d_st = jnp.maximum(
                jnp.asarray(accelerator.step(d_prev_st, d_picard), dtype=d_picard.dtype),
                d_picard)
        else:
            d_st = d_picard

        # Convergence is measured on the PICARD residual ||G(d) - d||, not on
        # the accelerated step ||d_st - d_prev_st||. They are the same thing
        # without an accelerator, but an extrapolated step is deliberately
        # LARGER than the Picard one, so testing the step would make a
        # converging accelerated run look divergent and never terminate.
        diff = jnp.max(jnp.abs(d_picard - d_prev_st))
        if early_exit:
            # concrete Python float/bool -- fine outside vmap/jit (eager),
            # which is the only context solve_fracture is ever called in.
            err_abs = float(diff)
            err_rel = float(diff / (jnp.max(jnp.abs(d_st)) + 1e-30))
            staggered_converged = err_abs < toler_st_abs or err_rel < toler_st_rel
            if log_sweeps:
                # One line per sweep, so a stalled attempt is not a black box:
                # which sweep, whether the residual is falling, whether
                # damage is running away, and whether Newton converged (and
                # in how many iterations) -- the last is what separated a
                # cycling cold-start Newton from a genuinely slow fixed point.
                nr_note = ("" if mechanics != "newton" else
                           f" newton={bool(converged_mech)}/{iter_nr} r={res_nr:.1e}")
                print(f"      [sweep {iter_st:3d}] {time.strftime('%H:%M:%S')} "
                      f"err_abs={err_abs:.3e} d_max={float(jnp.max(d_st)):.4f}"
                      f"{nr_note} ({time.perf_counter() - t_sweep:.1f}s)", flush=True)
            # On the Newton path, a converged damage field is not enough to
            # stop: if this sweep's Newton failed, the fold below rejects the
            # increment, and the cutback that follows cannot help -- the miss
            # is a Newton accuracy floor in the cracked voxels, not a step
            # that is too large. Sweep again instead; Newton restarts warm
            # from its own last iterate and usually converges. Measured on the
            # 60^3 uni RVE (sim/sim/12_mini_eta, eta = 0.25, eps_22 ~ 5.9%):
            # sweeps went newton fail / ok / fail with err_abs 3e-4 on the
            # last, and the same pattern repeated at dt/2, so two attempts
            # (11 min) were thrown away on increments whose damage had
            # converged. Bounded by maxiter_st like any other sweep.
            if staggered_converged and (mechanics != "newton" or bool(converged_mech)):
                break
            # ...but not indefinitely. Where Newton sits at an accuracy floor
            # just above toler_nr, re-sweeping a converged damage field never
            # ends: uni at eta 0.25, eps_22 ~ 7.5%, ran 90+ sweeps with
            # err_abs 1e-8..1e-6 and Newton 30/30 in every one (~3 h). After
            # NR_STALL_SWEEPS such sweeps in a row, stop and leave the verdict
            # to the fold below (rejected -> the stepper's cutback).
            nr_stall = nr_stall + 1 if staggered_converged else 0
            if nr_stall >= _NR_STALL_SWEEPS:
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
    accelerator:  str | object | None = None,
    accel_depth:  int = 5,
    accel_beta:   float = 1.0,
    d_cap:        float = 1.0,
    log_sweeps:   bool = False,
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
    # One accelerator instance for the whole run, reset per increment inside
    # _staggered_loop (the fixed point moves with the applied strain). Built
    # from a name so a YAML config can select it; an already-built instance is
    # passed through untouched.
    if accelerator is None or isinstance(accelerator, str):
        kw = {"depth": accel_depth}
        if accelerator is not None and str(accelerator).lower() == "anderson":
            kw["beta"] = accel_beta
        accel = build_accelerator(accelerator, **kw)
    else:
        accel = accelerator
    if accel is not None:
        print(f"Solver : staggered accelerator = {accel!r}")
    # The one dated stamp of the run. Every per-attempt stamp below is bare
    # HH:MM:SS local time, which is only unambiguous against this.
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}", flush=True)

    # ``eps`` is the last accepted increment's strain, the first sweep's
    # starting point in solve_fracture (eps_init) -- threaded the same way as
    # d/H, so a cutback retry restarts from the accepted strain and never from
    # a rejected attempt's.
    _state = {"d": d_init, "H": H_init, "t_prev": 0.0, "plastic": None, "eps": None}

    def solve_fn(t: float) -> FractureSolution:
        dt_actual = t - _state["t_prev"]
        # Log EVERY attempt, not just accepted increments. solve_automatic may
        # call this several times per increment (one per cutback) and
        # on_increment fires only on the accepted one, so without this a
        # cutback sequence is completely silent -- on a large grid that reads
        # as a hung run for tens of minutes at a time. Printed before the
        # solve, so the line appears while it is working, not after.
        #
        # The wall-clock stamp is what makes a log readable after the fact:
        # without it a stalled attempt is indistinguishable from a finished
        # one whose result line simply hasn't been written yet, and there is
        # no way to line the run up against nvidia-smi, the scheduler, or
        # anything else that happened on the machine. Local time, HH:MM:SS --
        # the date is printed once in the run banner ("Started:" above),
        # so a run that crosses midnight is still unambiguous.
        _state["attempt"] = _state.get("attempt", 0) + 1
        t_wall = time.perf_counter()
        print(f"    [attempt {_state['attempt']:3d}] {time.strftime('%H:%M:%S')} "
              f"t={t:.5f} dt={dt_actual:.5f} "
              f"from t_prev={_state['t_prev']:.5f} ...", flush=True)
        sol = solve_fracture(
            n, L, phase, materials, t * eps_bar, l0, Gc,
            d_init=_state["d"], H_init=_state["H"],
            formulation=formulation, scheme=scheme, control=control, stress_goal=stress_goal,
            toler_lin=toler_lin, maxiter_cg=maxiter_cg,
            toler_helm=toler_helm, maxiter_helm=maxiter_helm,
            eta=eta, dt=dt_actual, d_thres=d_thres, k_res=k_res,
            toler_st_abs=toler_st_abs, toler_st_rel=toler_st_rel, maxiter_st=maxiter_st,
            state_init=_state["plastic"], toler_nr=toler_nr, maxiter_nr=maxiter_nr,
            line_search=line_search,
            accelerator=accel, d_cap=d_cap, eps_init=_state["eps"],
            log_sweeps=log_sweeps,
        )
        acc_note = ("" if accel is None else
                    f" accel={accel.n_accelerated}/{accel.n_steps}"
                    f" rej={accel.n_rejected}")
        # Elapsed per ATTEMPT, which the accepted-increment line cannot give:
        # IncrementResult.wall_time covers every cutback attempt of a step
        # lumped together, so a rejected attempt's cost is otherwise invisible.
        print(f"    [attempt {_state['attempt']:3d}] {time.strftime('%H:%M:%S')} "
              f"-> converged={bool(sol.converged)} "
              f"staggered={sol.iter_staggered} "
              f"mech={bool(sol.converged_mech)} "
              f"d_max={float(jnp.max(sol.d)):.4f} err_abs={float(sol.err_abs):.3e}"
              f"{acc_note} ({time.perf_counter() - t_wall:.1f}s)", flush=True)
        return sol

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
