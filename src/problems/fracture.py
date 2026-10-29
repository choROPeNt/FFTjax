"""
Fracture problem: mechanics plus AT2 phase-field damage, solved by a
staggered loop for one prescribed macroscopic strain (one increment).

Amor split, AT2 degradation and hybrid irreversibility (Schneider & Kaestner
2025). The caller carries (d, H) across increments.
"""

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)

import time
from typing import Callable, Tuple, cast

import jax.numpy as jnp
import numpy as np

from materialmodels.assembly import assemble_pff_local_update
from materialmodels.phasefield.degradation import Gc_field, k_res_field
from materialmodels.phasefield.driving_force import lame_field, strain_energy_amor_split, update_history_hybrid
from operators.galerkin import build_galerkin_projector
from operators.green import build_freq_grid, build_reference_green_operator
from post.fields import compute_displacement, field_to_grid, to_voigt, von_mises
from problems.incremental import IncrementResult, solve_automatic, solve_fixed
from solvers.elliptic.scalar import solve_damage_helmholtz_cg
from solvers.elliptic.vector.displacement_based import solve_displacement_based
from solvers.elliptic.vector.fourier_galerkin import solve_fourier_galerkin
from solvers.elliptic.vector.lippmann_schwinger import solve_lippmann_schwinger, solve_lippmann_schwinger_mixed_bc
from solvers.elliptic.vector.mixed_bc import _ZERO_CONTROL
from solvers.solution import FractureSolution
from utils.io.xdmf_writer import IncrementalWriter


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
    One staggered mechanics/phase-field increment.

    Per iteration: degraded tangent at the previous strain, mechanical solve,
    driving force psi+ (Amor), history update (hybrid), damage solve, then
    stop when max|d_new - d_old| < toler_st_abs (or relative).

    Parameters
    ----------
    n, L        : grid shape and domain size
    phase       : (Nv,) int phase index
    materials   : material per phase
    eps_bar     : (3, 3) macroscopic strain; stress-controlled entries ignored
    l0          : phase-field length scale
    Gc          : None gathers each material's Gc; a scalar or (Nv,) overrides.
                  A uniform Gc is passed on as a scalar (exact-Laplacian solve)
    d_init, H_init : (Nv,) damage and history from the previous increment
    formulation : "lippmann_schwinger", "displacement" or "fourier_galerkin"
                  (see solve_mechanics)
    scheme      : "standard" or "rotated"
    control     : (3, 3) 0/1 mask, 1 = stress-controlled (None = pure strain)
    stress_goal : (3, 3) target stress, stress-controlled entries only
    toler_lin, maxiter_cg    : mechanical CG tolerance and cap
    toler_helm, maxiter_helm : damage CG tolerance and cap
    eta, dt     : viscous regularization of the damage equation
    d_thres     : hybrid irreversibility threshold
    k_res       : residual stiffness; None gathers each material's k_res
    toler_st_abs, toler_st_rel, maxiter_st : staggered convergence

    Returns
    -------
    FractureSolution
    """
    if formulation not in ("lippmann_schwinger", "displacement", "fourier_galerkin"):
        raise ValueError(
            f"unknown formulation {formulation!r}, expected 'lippmann_schwinger', "
            f"'displacement', or 'fourier_galerkin'"
        )

    control = control if control is not None else _ZERO_CONTROL
    stress_goal_arr = jnp.zeros((3, 3)) if stress_goal is None else stress_goal

    pff_local_update = assemble_pff_local_update(materials, phase)
    lam_vox, mu_vox = lame_field(materials, phase)
    xi_flat = build_freq_grid(n, L)
    k_res_arr = k_res_field(materials, phase) if k_res is None else k_res

    if Gc is None:
        Gc = Gc_field(materials, phase)
    if jnp.ndim(Gc) > 0:
        Gc_min, Gc_max = float(jnp.min(Gc)), float(jnp.max(Gc))
        if Gc_min == Gc_max:
            Gc = Gc_min

    if formulation == "lippmann_schwinger":
        elastic_op = build_reference_green_operator(n, L, materials, scheme=scheme)
    elif formulation == "fourier_galerkin":
        elastic_op = build_galerkin_projector(n, L, scheme=scheme)
    else:
        elastic_op = None

    return _staggered_loop(
        n=n, l0=l0, formulation=formulation, control=control, stress_goal_arr=stress_goal_arr,
        pff_local_update=pff_local_update, lam_vox=lam_vox, mu_vox=mu_vox, xi_flat=xi_flat, k_res_arr=k_res_arr,
        Gc=Gc, elastic_op=elastic_op,
        eps_bar=eps_bar, d_init=d_init, H_init=H_init,
        toler_lin=toler_lin, maxiter_cg=maxiter_cg,
        toler_helm=toler_helm, maxiter_helm=maxiter_helm,
        eta=eta, dt=dt, d_thres=d_thres,
        toler_st_abs=toler_st_abs, toler_st_rel=toler_st_rel, maxiter_st=maxiter_st,
        early_exit=True,
    )


def _staggered_loop(
    n, l0, formulation, control, stress_goal_arr,
    pff_local_update, lam_vox, mu_vox, xi_flat, k_res_arr, Gc, elastic_op,
    eps_bar, d_init, H_init,
    toler_lin, maxiter_cg, toler_helm, maxiter_helm, eta, dt, d_thres,
    toler_st_abs, toler_st_rel, maxiter_st, early_exit: bool,
) -> FractureSolution:
    """
    Staggered loop shared by solve_fracture (``early_exit=True``: Python break,
    float results) and solve_fracture_fixed (always ``maxiter_st`` iterations,
    array results, vmap/jit-safe).

    The tangent is frozen at the previous iteration's strain: the Amor split
    makes stress nonlinear in eps even at fixed d, and freezing it keeps the
    linear CG solvers usable. It converges to the consistent solution because
    eps_prev equals eps at staggered convergence.
    """
    d_st = d_init
    H_st = H_init
    eps_bar_cur = None
    Nv = lam_vox.shape[0]
    eps_prev = jnp.ones((3, 3, Nv)) * eps_bar[:, :, None]
    control_nonzero = any(any(row) for row in control)

    for iter_st in range(1, maxiter_st + 1):
        d_prev_st = d_st

        _, C_eff = pff_local_update(eps_prev, d_st)

        if formulation == "lippmann_schwinger":
            if control_nonzero:
                eps, sigma, delta, eps_bar_cur, converged_mech, _n_iter_outer = solve_lippmann_schwinger_mixed_bc(
                    n, C_eff, elastic_op, control, eps_bar, stress_goal_arr,
                    toler_lin=toler_lin, maxiter=maxiter_cg,
                )
            else:
                eps, sigma, delta, converged_mech = solve_lippmann_schwinger(
                    n, C_eff, elastic_op, eps_bar,
                    toler_lin=toler_lin, maxiter=maxiter_cg,
                )
        elif formulation == "fourier_galerkin":
            eps, sigma, delta, eps_bar_cur, converged_mech = solve_fourier_galerkin(
                n, C_eff, elastic_op, eps_bar, control, stress_goal_arr,
                toler_lin=toler_lin, maxiter=maxiter_cg,
            )
        else:
            eps, sigma, delta, eps_bar_cur, converged_mech = solve_displacement_based(
                n, C_eff, xi_flat, eps_bar, control, stress_goal_arr,
                toler_lin=toler_lin, maxiter=maxiter_cg,
            )
        eps_prev = eps

        psi_pos, _ = strain_energy_amor_split(eps, lam_vox, mu_vox)
        H_st = update_history_hybrid(H_st, psi_pos, d_prev_st, d_thres=d_thres)

        d_st, converged_helm = solve_damage_helmholtz_cg(
            H_st, xi_flat, n, l0, Gc, d_prev_st,
            toler_cg=toler_helm, maxiter=maxiter_helm, eta=eta, dt=dt, k=k_res_arr,
        )

        diff = jnp.max(jnp.abs(d_st - d_prev_st))
        if early_exit:
            err_abs = float(diff)
            err_rel = float(diff / (jnp.max(jnp.abs(d_st)) + 1e-30))
            staggered_converged = err_abs < toler_st_abs or err_rel < toler_st_rel
            if staggered_converged:
                break
        else:
            # float()/bool() would fail under vmap; every lane runs maxiter_st.
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
    vmap/jit-safe solve_fracture: always ``maxiter_st`` staggered iterations,
    no early exit and no convergence check. Inspect ``converged_staggered``
    per batch lane yourself. Parameters as solve_fracture.
    """
    if formulation not in ("lippmann_schwinger", "displacement", "fourier_galerkin"):
        raise ValueError(
            f"unknown formulation {formulation!r}, expected 'lippmann_schwinger', "
            f"'displacement', or 'fourier_galerkin'"
        )

    control = control if control is not None else _ZERO_CONTROL
    stress_goal_arr = jnp.zeros((3, 3)) if stress_goal is None else stress_goal

    pff_local_update = assemble_pff_local_update(materials, phase)
    lam_vox, mu_vox = lame_field(materials, phase)
    xi_flat = build_freq_grid(n, L)
    k_res_arr = k_res_field(materials, phase) if k_res is None else k_res

    # Uniformity comes from the materials, not the phase-derived array: a
    # float() on that breaks under vmap. An explicit array Gc is used as given.
    if Gc is None:
        Gc = Gc_field(materials, phase)
        Gc_values = {float(m.Gc) for m in materials if m.Gc is not None}
        if len(Gc_values) == 1:
            Gc = Gc_values.pop()

    if formulation == "lippmann_schwinger":
        elastic_op = build_reference_green_operator(n, L, materials, scheme=scheme)
    elif formulation == "fourier_galerkin":
        elastic_op = build_galerkin_projector(n, L, scheme=scheme)
    else:
        elastic_op = None

    return _staggered_loop(
        n=n, l0=l0, formulation=formulation, control=control, stress_goal_arr=stress_goal_arr,
        pff_local_update=pff_local_update, lam_vox=lam_vox, mu_vox=mu_vox, xi_flat=xi_flat, k_res_arr=k_res_arr,
        Gc=Gc, elastic_op=elastic_op,
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
    vmap/jit-safe solve_fracture_incremental(stepping="fixed"), built on
    solve_fracture_fixed: a static number of steps, no cutbacks, never raises.
    Use it to vmap over RVE realizations (batch ``phase``).

    Returns
    -------
    list[FractureSolution], one per step (no IncrementResult bookkeeping);
    check ``converged_staggered`` per lane yourself
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
    solve_fracture, load-stepped via problems.incremental (as solve_mechanics).

    (d, H) and t_prev are updated only in ``on_increment``, which fires once
    per accepted increment: cutback attempts must all start from the last
    accepted state. The damage viscosity step is the actual increment
    ``t - t_prev``, not ``dt_step`` (the last fixed step and every automatic
    step differ from it).

    Parameters as solve_fracture plus the load-stepping, writer and callback
    arguments of solve_mechanics; ``dt_step`` is the load increment (``dt``
    here means the viscosity step). The writer also gets "damage" and
    "strain_energy_pos".

    Returns
    -------
    list[IncrementResult] with FractureSolution results
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
