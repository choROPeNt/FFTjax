"""
Thermal (steady-state conduction) problem: build K(x), solve with mixed
gradient/flux BC. One formulation, a direct CG solve (no reference-medium
scheme exists for this equation). Writes the periodic fluctuation T', not T.
"""

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)

import time
from typing import Callable, Tuple, cast

import jax.numpy as jnp
import numpy as np

from materialmodels.assembly import assemble_K_field
from operators.green import build_freq_grid
from post.fields import vector_field_to_grid
from problems.incremental import IncrementResult, solve_automatic, solve_fixed
from solvers.elliptic.scalar import solve_thermal_conduction
from solvers.solution import ThermalSolution
from utils.io.xdmf_writer import IncrementalWriter

_ZERO_CONTROL = (0, 0, 0)


def _solve_thermal_step(
    n:          Tuple[int, ...],
    L:          Tuple[float, ...],
    phase:      jnp.ndarray,
    materials:  list,
    grad_T_bar: jnp.ndarray,
    control:    Tuple[int, int, int] = _ZERO_CONTROL,
    flux_goal:  jnp.ndarray | None = None,
    toler_lin:  float = 1e-6,
    maxiter:    int = 1000,
) -> ThermalSolution:
    """One solve at a fixed macroscopic gradient; parameters as ``solve_thermal``."""
    K_field = assemble_K_field(materials, phase)
    xi_flat = build_freq_grid(n, L)
    grad_T, flux, T_prime, delta, grad_T_bar_out, converged = solve_thermal_conduction(
        n, K_field, xi_flat, grad_T_bar, control=control, flux_goal=flux_goal,
        toler_lin=toler_lin, maxiter=maxiter,
    )
    return ThermalSolution(grad_T, flux, T_prime, delta, converged, grad_T_bar=grad_T_bar_out)


def solve_thermal(
    n:            Tuple[int, ...],
    L:            Tuple[float, ...],
    phase:        jnp.ndarray,
    materials:    list,
    grad_T_bar:   jnp.ndarray,
    stepping:     str = "single",
    control:      Tuple[int, int, int] | None = None,
    flux_goal:    jnp.ndarray | None = None,
    toler_lin:    float = 1e-6,
    maxiter:      int = 1000,
    dt:           float | None = None,
    dt_init:      float = 0.1,
    dt_min:       float = 1e-4,
    dt_max:       float = 0.5,
    factor_inc:   float = 1.5,
    factor_dec:   float = 0.5,
    max_cutbacks: int = 5,
    max_steps:    int = 1000,
    writer:       IncrementalWriter | None = None,
    on_increment: Callable[[IncrementResult, float], None] | None = None,
) -> list[IncrementResult]:
    """
    Steady-state conduction on a phase-labelled periodic grid under a
    macroscopic gradient, optionally with mixed gradient/flux control and
    load stepping (as solve_mechanics; for linear conduction "single" suffices).

    Parameters
    ----------
    n, L        : grid shape and domain size
    phase       : (Nv,) int phase index
    materials   : ConductivityModel per phase
    grad_T_bar  : (3,) macroscopic gradient at t = 1; flux-controlled entries ignored
    stepping    : "single", "fixed" or "automatic"
    control     : (3,) 0/1 mask, 1 = flux-controlled (None = pure gradient)
    flux_goal   : (3,) target flux, flux-controlled entries only (None = 0)
    toler_lin, maxiter : CG tolerance and iteration cap
    dt, dt_init, dt_min, dt_max, factor_inc, factor_dec, max_cutbacks, max_steps :
                  load-stepping controls
    writer      : IncrementalWriter for phase, gradient, flux and T'
    on_increment: callback ``(result, write_time)`` per accepted increment

    Returns
    -------
    list[IncrementResult] with ThermalSolution results
    """
    control = control if control is not None else _ZERO_CONTROL

    def solve_fn(t: float) -> ThermalSolution:
        return _solve_thermal_step(
            n, L, phase, materials, t * grad_T_bar,
            control=control, flux_goal=flux_goal, toler_lin=toler_lin, maxiter=maxiter,
        )

    def _on_increment(result: IncrementResult) -> None:
        write_time = 0.0
        if writer is not None:
            t0 = time.perf_counter()
            sol = cast(ThermalSolution, result.solution)
            grad_T_grid = vector_field_to_grid(sol.grad_T, n)
            flux_grid   = vector_field_to_grid(sol.flux, n)
            fields = {
                "phase":                  np.asarray(phase).reshape(n).astype(np.float32),
                "gradient":               grad_T_grid.astype(np.float64),
                "flux":                   flux_grid.astype(np.float64),
                "temperature_fluctuation": np.asarray(sol.T_prime).reshape(n).astype(np.float64),
            }
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
        if dt is None:
            raise ValueError("stepping='fixed' requires dt")
        return solve_fixed(solve_fn, dt, on_increment=_on_increment)
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
