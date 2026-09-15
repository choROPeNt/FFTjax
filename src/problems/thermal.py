"""
Thin wiring layer for the thermal (steady-state heat conduction) problem:
build K(x), pick a mixed gradient/flux macroscopic BC, solve, return the
fields -- the scalar analogue of problems.mechanics, mirrored structurally
rather than re-derived (see that module's own docstring for the reasoning
this one shares).

Scope of this first pass, matching what's actually built on the
materialmodels/ and solvers/ stacks so far:
- Only one formulation: solvers.elliptic.scalar.solve_thermal_conduction, a
  true heterogeneous-conductivity direct CG solve, no reference-medium fixed
  point. Unlike problems.mechanics there is no ``formulation`` switch to
  make -- no reference-medium Lippmann-Schwinger analogue exists for this
  equation in either this project or the FFTMAD reference implementation's
  own diffusion module (both treat heterogeneous div(k grad) as a direct CG
  solve).
- Only isotropic conductivity (materialmodels.thermal.isotropic.
  ThermalConductivityIsotropic) has been built; materialmodels.assembly.
  assemble_K_field is generic over any ConductivityModel, so an anisotropic
  one plugs in unchanged once it exists.
- No absolute-temperature-field reconstruction (writes the periodic
  fluctuation T', not T(x) -- see solvers.solution.ThermalSolution's own
  note on why, the same gauge-freedom reasoning problems.mechanics's
  post.fields.compute_displacement exists to handle on the elasticity side,
  just not built here yet).

One public entry point, ``solve_thermal``: its ``stepping`` argument picks a
single full-grad_T_bar solve or a load-stepped one (Abaqus-*STATIC style,
via problems.incremental -- reused verbatim, it's genuinely solver-agnostic)
-- always returns list[IncrementResult] regardless of ``stepping``, same
reason solve_mechanics does.
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
    """
    One thermal equilibrium solve at a fixed macroscopic gradient (no
    load-stepping) -- the per-increment building block solve_thermal closes
    over as its ``solve_fn``. See solve_thermal's docstring for all
    parameters.
    """
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
    Solve the thermal equilibrium problem on a phase-labelled periodic voxel
    grid under a prescribed macroscopic temperature gradient (optionally
    mixed gradient/flux control), optionally load-stepped up to grad_T_bar
    (Abaqus-*STATIC style).

    Parameters
    ----------
    n, L        : grid shape and physical domain size
    phase       : (Nv,) int      phase index per voxel (0-based)
    materials   : list           each implements .conductivity_tensor()
                  (see materialmodels.thermal.isotropic.ThermalConductivityIsotropic)
    grad_T_bar  : (3,)           macroscopic temperature gradient at t=1 (the
                  full load); entries where ``control == 1`` are ignored
                  (solved for instead)
    stepping    : "single"    -- one solve at the full grad_T_bar (default).
                  "fixed"     -- equal load-fraction increments of size ``dt``
                                 (problems.incremental.solve_fixed).
                  "automatic" -- adaptive load-fraction step, grown on
                                 convergence, cut back and retried on
                                 non-convergence (problems.incremental.
                                 solve_automatic; dt_init/_min/_max,
                                 factor_inc/_dec, max_cutbacks, max_steps all
                                 forwarded to it). For linear conduction
                                 (constant-in-T conductivity, the only kind
                                 built so far) an intermediate increment
                                 carries no information a scale factor
                                 doesn't -- "single" is the natural default,
                                 "fixed"/"automatic" exist for parity with
                                 solve_mechanics and for a future
                                 temperature-dependent conductivity.
    control     : (3,) 0/1 mask, 1 = flux-controlled, 0 = gradient-controlled.
                  None (default) = pure gradient BC (all zero).
    flux_goal   : (3,) macroscopic target flux, used only on
                  ``control``-marked entries. None (default) = zero.
    toler_lin, maxiter : CG tolerance / iteration cap
    dt, dt_init, dt_min, dt_max, factor_inc, factor_dec, max_cutbacks, max_steps :
                  load-stepping controls, forwarded to solve_fixed/
                  solve_automatic -- see problems.incremental. ``dt`` is
                  required for stepping="fixed"; the rest are only used by
                  stepping="automatic".
    writer        : if given, gets one write_increment(step, fields, time=t)
                  call per *accepted* increment (never for a failed cutback
                  attempt -- see problems.incremental's on_increment docs) --
                  phase, gradient, flux, and the periodic temperature
                  fluctuation T' (not the absolute temperature field, see
                  this module's own docstring). The caller still owns
                  opening/closing the writer.
    on_increment  : an additional, caller-supplied callback
                  ``(result, write_time) -> None`` invoked once per accepted
                  increment (after the writer, if any) -- see
                  solve_mechanics's identically-shaped parameter for the
                  full reasoning.

    Returns
    -------
    list[IncrementResult] -- always a list, even for stepping="single" (one
    element, t=1.0), so callers have one consistent return shape regardless
    of the stepping choice. Each element's ``.solution`` is a
    ThermalSolution; its ``.grad_T_bar`` is the macroscopic gradient with
    any flux-controlled entries filled in.
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
