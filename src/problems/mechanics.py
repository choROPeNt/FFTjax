"""
Thin wiring layer for the mechanical (elasticity) problem: pick a
formulation, build C(x), pick a reference medium (lippmann_schwinger) or a
mixed-BC control mask (displacement), solve, return the fields.

Scope of this first pass, matching what's actually built on the operators/
and solvers/ stacks so far:
- Reference-medium averaging (lippmann_schwinger only) is a plain arithmetic
  mean of the materials' Lame parameters (matches what
  notebooks/lin-elastic_strain.ipynb already does by hand). This is a
  placeholder for materialmodels/averaging.py's VoxelAveraging ABC, which
  doesn't exist yet either -- swap it in here once built, this function's
  signature shouldn't need to change.
- Builds a solver (LippmannSchwingerSolver or DisplacementBasedSolver, both
  ElasticitySolver) and calls .solve() -- not the plain solve_* functions --
  so the ABC actually gets exercised by its real callers.
- ``control`` (mixed macroscopic strain/stress BC) only has meaning for
  formulation="displacement" -- lippmann_schwinger's reference-medium
  approach can't do stress-controlled macroscopic directions, so a nonzero
  control there raises a clear error rather than silently ignoring it.
"""

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)

import time
from typing import Callable, Tuple, cast

import jax.numpy as jnp
import numpy as np

from materialmodels.assembly import assemble_C_field
from operators.green import build_freq_grid, build_reference_green_operator
from post.fields import compute_displacement, field_to_grid, to_voigt, von_mises
from problems.incremental import IncrementResult, solve_automatic, solve_fixed
from solvers.elliptic.vector.base import ElasticitySolution
from solvers.elliptic.vector.displacement_based import DisplacementBasedSolver
from solvers.elliptic.vector.lippmann_schwinger import LippmannSchwingerSolver
from utils.io.xdmf_writer import IncrementalWriter

_ZERO_CONTROL = ((0, 0, 0), (0, 0, 0), (0, 0, 0))


def solve_mechanics(
    n:           Tuple[int, ...],
    L:           Tuple[float, ...],
    phase:       jnp.ndarray,
    materials:   list,
    eps_bar:     jnp.ndarray,
    formulation: str = "lippmann_schwinger",
    scheme:      str = "rotated",
    control:     Tuple[Tuple[int, ...], ...] | None = None,
    stress_goal: jnp.ndarray | None = None,
    toler_lin:   float = 1e-6,
    maxiter:     int = 1000,
) -> ElasticitySolution:
    """
    Solve the mechanical equilibrium problem on a phase-labelled periodic
    voxel grid under a prescribed macroscopic strain (and, for
    formulation="displacement", optionally mixed strain/stress control).

    Parameters
    ----------
    n, L        : grid shape and physical domain size
    phase       : (Nv,) int      phase index per voxel (0-based)
    materials   : list           each implements .stiffness_tensor(), .lam, .mu
                  (see materialmodels.elastic.isotropic.LinearElasticIsotropic)
    eps_bar     : (3, 3)         prescribed macroscopic strain; entries where
                  ``control == 1`` are ignored (solved for instead)
    formulation : "lippmann_schwinger" (reference-medium, strain BC only) or
                  "displacement" (true heterogeneous tangent, supports mixed BC)
    scheme      : "standard" (GreenOperatorBasic) or "rotated" (GreenOperatorWillot)
                  -- only used by formulation="lippmann_schwinger"
    control     : (3, 3) 0/1 mask, 1 = stress-controlled, 0 = strain-controlled.
                  None (default) = pure strain BC (all zero). Only
                  formulation="displacement" can have any nonzero entries.
    stress_goal : formulation-specific -- see ElasticitySolver.solve's
                  docstring. lippmann_schwinger: (3, 3, Nv) per-voxel target
                  stress field or None (zero). displacement: (3, 3)
                  macroscopic target stress, used only on ``control``-marked
                  entries.
    toler_lin, maxiter : CG tolerance / iteration cap

    Returns
    -------
    ElasticitySolution(eps, sigma, delta, converged, eps_bar) -- a NamedTuple;
    ``eps_bar`` is None unless formulation="displacement", in which case it's
    the macroscopic strain with any stress-controlled entries filled in.
    """
    control = control if control is not None else _ZERO_CONTROL
    control_nonzero = any(any(row) for row in control)

    C_field = assemble_C_field(materials, phase)

    if formulation == "lippmann_schwinger":
        if control_nonzero:
            raise ValueError(
                "formulation='lippmann_schwinger' cannot do stress-controlled "
                "macroscopic BC (control has nonzero entries) -- its reference-"
                "medium approach only supports pure strain BC; "
                "use formulation='displacement' instead"
            )

        # Reference medium + Green's operator: see module docstring for why
        # this isn't materialmodels/averaging.py yet.
        green_op = build_reference_green_operator(n, L, materials, scheme=scheme)

        solver = LippmannSchwingerSolver(n, green_op, toler_lin, maxiter)
        return solver.solve(C_field, eps_bar, stress_goal)
    elif formulation == "displacement":
        xi_flat = build_freq_grid(n, L)
        solver = DisplacementBasedSolver(n, xi_flat, control, toler_lin, maxiter)
        return solver.solve(C_field, eps_bar, stress_goal)
    else:
        raise ValueError(
            f"unknown formulation {formulation!r}, expected 'lippmann_schwinger' or 'displacement'"
        )


def solve_mechanics_incremental(
    n:           Tuple[int, ...],
    L:           Tuple[float, ...],
    phase:       jnp.ndarray,
    materials:   list,
    eps_bar:     jnp.ndarray,
    stepping:    str = "single",
    formulation: str = "lippmann_schwinger",
    scheme:      str = "rotated",
    control:     Tuple[Tuple[int, ...], ...] | None = None,
    stress_goal: jnp.ndarray | None = None,
    toler_lin:   float = 1e-6,
    maxiter:     int = 1000,
    dt:          float | None = None,
    dt_init:     float = 0.1,
    dt_min:      float = 1e-4,
    dt_max:      float = 0.5,
    factor_inc:  float = 1.5,
    factor_dec:  float = 0.5,
    max_cutbacks: int = 5,
    max_steps:   int = 1000,
    writer:       IncrementalWriter | None = None,
    orientation:  jnp.ndarray | None = None,
    on_increment: Callable[[IncrementResult, float], None] | None = None,
) -> list[IncrementResult]:
    """
    solve_mechanics, optionally load-stepped up to the target eps_bar
    (Abaqus-*STATIC style) -- see problems.incremental for the driver.

    ``stepping``:
      "single"    -- one solve_mechanics call at the full eps_bar (default).
      "fixed"     -- equal load-fraction increments of size ``dt``
                     (problems.incremental.solve_fixed).
      "automatic" -- adaptive load-fraction step, grown on convergence, cut
                     back and retried on non-convergence
                     (problems.incremental.solve_automatic; dt_init/_min/_max,
                     factor_inc/_dec, max_cutbacks, max_steps all forwarded to it).

    ``writer``, if given, gets one write_increment(step, fields, time=t)
    call per *accepted* increment (never for a failed cutback attempt --
    see problems.incremental's on_increment docs) -- strain/stress/von_mises/
    displacement, plus phase (and orientation, if given). This is the only
    part of this function that knows about XDMF/HDF5 output; the caller
    still owns opening/closing the writer (its file path is a script/config
    concern, not this problem-wiring layer's).
    ``orientation`` : (3, Nv) float, only used for the ``writer`` output
    (constant across increments); pass None to omit it from the fields.
    ``on_increment`` : an additional, caller-supplied callback
    ``(result, write_time) -> None`` invoked once per accepted increment
    (after the writer, if any) -- e.g. a print/log callback, so progress is
    reported live as each increment converges rather than only after the
    whole solve returns. ``write_time`` is the wall-clock seconds spent in
    this function's own post-processing + write_increment call (0.0 if
    ``writer`` is None) -- kept separate from ``result.wall_time`` (which is
    the solve alone, measured in problems.incremental) so a caller can tell
    solve cost and write cost apart instead of only seeing their sum.

    All other parameters are solve_mechanics's own -- see its docstring.

    Returns
    -------
    list[IncrementResult] -- always a list, even for stepping="single" (one
    element, t=1.0), so callers have one consistent return shape
    regardless of the stepping choice. Each element's ``.solution`` is an
    ElasticitySolution.
    """
    def solve_fn(t: float) -> ElasticitySolution:
        return solve_mechanics(
            n, L, phase, materials, t * eps_bar,
            formulation=formulation, scheme=scheme, control=control,
            stress_goal=stress_goal, toler_lin=toler_lin, maxiter=maxiter,
        )

    def _on_increment(result: IncrementResult) -> None:
        write_time = 0.0
        if writer is not None:
            t0 = time.perf_counter()
            sol = cast(ElasticitySolution, result.solution)
            eps_grid   = field_to_grid(sol.eps, n)
            sigma_grid = field_to_grid(sol.sigma, n)
            sigma_vm   = von_mises(sigma_grid)
            # sol.eps_bar (displacement formulation only) is the *solved*
            # macroscopic strain -- stress-controlled entries filled in with
            # their actual result, e.g. Poisson contraction under a free
            # lateral surface. The prescribed eps_bar has zeros there instead,
            # so falling back to it would silently drop that contraction from
            # the displacement field's macroscopic part (sol.eps itself is
            # unaffected -- the solver embeds the true mean directly in its
            # DC frequency mode, so stress/modulus are correct either way).
            eps_bar_u  = sol.eps_bar if sol.eps_bar is not None else result.t * eps_bar
            u_grid     = compute_displacement(sol.eps, eps_bar_u, n, L)
            fields = {
                "phase":        np.asarray(phase).reshape(n).astype(np.float32),
                "strain":       to_voigt(eps_grid).astype(np.float64),
                "stress":       to_voigt(sigma_grid).astype(np.float64),
                "von_mises":    sigma_vm.astype(np.float64),
                "displacement": u_grid.astype(np.float64),
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
