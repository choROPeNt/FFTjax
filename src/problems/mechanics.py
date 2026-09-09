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
- One public entry point, ``solve_mechanics``: its ``stepping`` argument
  picks a single full-eps_bar solve or a load-stepped one (Abaqus-*STATIC
  style, via problems.incremental) -- it always returns list[IncrementResult]
  regardless of ``stepping``, so callers (including a jax.vmap/jax.jit'd
  one, see notebooks/lin-elastic_strain_vmap.ipynb) have one consistent
  return shape and don't need to special-case "single" against the two
  load-stepped modes.

``solve_displacement_based_nonlinear`` (bottom of this file) is a second,
separate entry point: a Newton-outer/CG-inner driver for a nonlinear local
constitutive law (e.g. materialmodels.inelastic.plasticity_j2.J2Plasticity),
where the tangent depends on the unknown strain and can't be assembled once
up front the way solve_mechanics's C_field is. It lives here rather than in
solvers/ for the same reason problems.fracture's staggered loop does (see
that module's docstring): it's an iterative scheme built by repeatedly
calling solvers/ primitives (here, one linear CG solve per Newton
iteration), not itself a reusable numerical algorithm the way
solvers.krylov.cg.cg_solve is -- keeping it here keeps solvers/ to its usual
habit (operate on prepared arrays, no outer-loop control flow of its own).
Unlike solve_mechanics, it does NOT go through materialmodels.assembly
(the caller's own ``local_update`` callable already resolves materials/phase
into a per-voxel stress+tangent), and it only supports pure strain BC (no
mixed strain/stress ``control``, unlike solve_displacement_based).
"""

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)

import time
from typing import Any, Callable, Tuple, cast
from math import prod

import jax.numpy as jnp
import numpy as np

from materialmodels.assembly import assemble_C_field
from operators.green import build_freq_grid, build_reference_green_operator, nyquist_safe_xi
from post.fields import compute_displacement, field_to_grid, to_voigt, von_mises
from problems.incremental import IncrementResult, solve_automatic, solve_fixed
from solvers.elliptic.vector.displacement_based import DisplacementBasedSolver
from solvers.elliptic.vector.lippmann_schwinger import LippmannSchwingerSolver
from solvers.krylov.cg import cg_solve
from solvers.solution import ElasticitySolution
from utils.io.xdmf_writer import IncrementalWriter

_ZERO_CONTROL = ((0, 0, 0), (0, 0, 0), (0, 0, 0))


def _solve_mechanics_step(
    n:           Tuple[int, ...],
    L:           Tuple[float, ...],
    phase:       jnp.ndarray,
    materials:   list,
    eps_bar:     jnp.ndarray,
    formulation: str = "lippmann_schwinger",
    scheme:      str = "rotated",
    control:     Tuple[Tuple[int, ...], ...] = _ZERO_CONTROL,
    stress_goal: jnp.ndarray | None = None,
    toler_lin:   float = 1e-6,
    maxiter:     int = 1000,
) -> ElasticitySolution:
    """
    One mechanical equilibrium solve at a fixed macroscopic strain (no
    load-stepping) -- the per-increment building block solve_mechanics
    closes over as its ``solve_fn``. See solve_mechanics's docstring for
    all parameters.
    """
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


def solve_mechanics(
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
    Solve the mechanical equilibrium problem on a phase-labelled periodic
    voxel grid under a prescribed macroscopic strain (and, for
    formulation="displacement", optionally mixed strain/stress control),
    optionally load-stepped up to eps_bar (Abaqus-*STATIC style).

    Parameters
    ----------
    n, L        : grid shape and physical domain size
    phase       : (Nv,) int      phase index per voxel (0-based)
    materials   : list           each implements .stiffness_tensor(), .lam, .mu
                  (see materialmodels.elastic.isotropic.LinearElasticIsotropic)
    eps_bar     : (3, 3)         macroscopic strain at t=1 (the full load);
                  entries where ``control == 1`` are ignored (solved for
                  instead)
    stepping    : "single"    -- one solve at the full eps_bar (default).
                  "fixed"     -- equal load-fraction increments of size ``dt``
                                 (problems.incremental.solve_fixed).
                  "automatic" -- adaptive load-fraction step, grown on
                                 convergence, cut back and retried on
                                 non-convergence (problems.incremental.
                                 solve_automatic; dt_init/_min/_max,
                                 factor_inc/_dec, max_cutbacks, max_steps all
                                 forwarded to it).
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
    dt, dt_init, dt_min, dt_max, factor_inc, factor_dec, max_cutbacks, max_steps :
                  load-stepping controls, forwarded to solve_fixed/
                  solve_automatic -- see problems.incremental. ``dt`` is
                  required for stepping="fixed"; the rest are only used by
                  stepping="automatic".
    writer        : if given, gets one write_increment(step, fields, time=t)
                  call per *accepted* increment (never for a failed cutback
                  attempt -- see problems.incremental's on_increment docs) --
                  strain/stress/von_mises/displacement, plus phase (and
                  orientation, if given). This is the only part of this
                  function that knows about XDMF/HDF5 output; the caller
                  still owns opening/closing the writer (its file path is a
                  script/config concern, not this problem-wiring layer's).
    orientation   : (3, Nv) float, only used for the ``writer`` output
                  (constant across increments); pass None to omit it from
                  the fields.
    on_increment  : an additional, caller-supplied callback
                  ``(result, write_time) -> None`` invoked once per accepted
                  increment (after the writer, if any) -- e.g. a print/log
                  callback, so progress is reported live as each increment
                  converges rather than only after the whole solve returns.
                  ``write_time`` is the wall-clock seconds spent in this
                  function's own post-processing + write_increment call (0.0
                  if ``writer`` is None) -- kept separate from
                  ``result.wall_time`` (which is the solve alone, measured
                  in problems.incremental) so a caller can tell solve cost
                  and write cost apart instead of only seeing their sum.

    Returns
    -------
    list[IncrementResult] -- always a list, even for stepping="single" (one
    element, t=1.0), so callers have one consistent return shape regardless
    of the stepping choice. Each element's ``.solution`` is an
    ElasticitySolution; its ``.eps_bar`` is None unless
    formulation="displacement", in which case it's the macroscopic strain
    with any stress-controlled entries filled in.
    """
    control = control if control is not None else _ZERO_CONTROL

    def solve_fn(t: float) -> ElasticitySolution:
        return _solve_mechanics_step(
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


def solve_displacement_based_nonlinear(
    n:            Tuple[int, ...],
    xi_flat:      jnp.ndarray,
    eps_bar:      jnp.ndarray,
    local_update: Callable[[jnp.ndarray, Any], Tuple[jnp.ndarray, jnp.ndarray, Any]],
    state_init:   Any,
    toler_lin:    float = 1e-6,
    maxiter_lin:  int = 1000,
    toler_nr:     float = 1e-8,
    maxiter_nr:   int = 50,
    C0:           jnp.ndarray | None = None,
    delta_init:   jnp.ndarray | None = None,
):
    """
    Newton-outer / CG-inner solve for a nonlinear local constitutive law:
    div(sigma(eps)) = 0 on a periodic voxel grid, prescribed macroscopic
    strain ``eps_bar``, pure strain BC only (no mixed strain/stress
    ``control`` -- that generalization, analogous to
    solve_displacement_based's, is not built here).

    Same FFT-based grad/div machinery as solve_displacement_based (frequency
    grid, Nyquist-safe xi, strain-from-fluctuation via the DC-bin trick),
    reused directly rather than re-derived, to avoid a second, possibly
    inconsistent implementation of the same operator. The difference is what
    drives each linear CG solve: solve_displacement_based solves once
    against a FIXED C_field; this solves repeatedly, at each outer (Newton)
    iteration evaluating the actual nonlinear stress and its tangent at the
    CURRENT trial strain via ``local_update``, using CG only for the
    *linearized correction* -- true Newton, not a fixed-point/secant scheme.

    ``local_update(eps_field, state) -> (sigma_field, C_tan_field, new_state)``
    is any per-voxel nonlinear stress/tangent/state map, already
    vmapped/materials-resolved by the caller (e.g. a per-phase combination of
    materialmodels.inelastic.plasticity_j2.J2Plasticity.stress_and_tangent_field
    for a plastic phase and a plain einsum against a constant elastic C for
    others -- see notebooks/in-elastic_J2.ipynb, section 4). ``state`` is
    caller-defined; this driver only carries it from iteration to iteration.

    Parameters
    ----------
    n            : grid shape (nx, ny, nz)
    xi_flat      : (3, Nv)  angular-frequency grid (operators.green.build_freq_grid)
    eps_bar      : (3, 3)   prescribed macroscopic strain (pure strain BC --
                   every component is prescribed, none solved for)
    local_update : eps_field (3,3,Nv), state -> (sigma_field (3,3,Nv),
                   C_tan_field (3,3,3,3,Nv), new_state)
    state_init   : initial per-voxel state (caller-defined; e.g. (eps_p, alpha)
                   from the end of the previous load increment)
    toler_lin, maxiter_lin : CG tolerance/cap for each Newton iteration's
                   linearized correction
    toler_nr, maxiter_nr   : Newton relative-residual tolerance and iteration cap
    delta_init   : (3,3,Nv) or None -- initial strain-fluctuation guess (i.e.
                   eps - eps_bar broadcast). None starts from zero fluctuation
                   (a uniform-strain guess) -- fine for a single, isolated
                   solve, but a genuinely bad starting point once far enough
                   into load-stepping that the true fluctuation field is
                   large (e.g. deep in a plastic regime): Newton then needs
                   many more iterations, or fails to converge in maxiter_nr,
                   purely from a poor initial guess, not a solver defect.
                   Load-stepping callers should warm-start each step with the
                   PREVIOUS step's converged full strain field minus this
                   step's new eps_bar broadcast -- see
                   notebooks/in-elastic_J2.ipynb, section 4, for the pattern.
                   Projected to zero mean internally regardless of what's
                   passed in (see below) -- the macroscopic strain is always
                   carried by eps0, never by delta.
    C0           : (3,3,3,3) or None -- fixed reference stiffness for the CG
                   preconditioner; None re-derives it from the CURRENT tangent
                   field every Newton iteration (voxel-mean of C_tan), matching
                   solve_displacement_based's default. Passing a fixed elastic
                   C0 avoids rebuilding/inverting the preconditioner each
                   iteration -- standard practice for FFT-based Newton-Krylov
                   plasticity, not yet benchmarked either way in this project.

    Returns
    -------
    eps       : (3, 3, Nv)
    sigma     : (3, 3, Nv)
    state     : final per-voxel state
    converged : bool -- True if the Newton residual met toler_nr within maxiter_nr
    n_iter    : int  -- Newton iterations actually run
    """
    Nv = prod(n)
    iq = 1j * nyquist_safe_xi(xi_flat, n)  # (3, Nv)

    def fft_(x):
        s = x.shape
        return jnp.fft.fftn(x.reshape(s[:-1] + n), axes=(-3, -2, -1)).reshape(s)

    def ifft_(x):
        s = x.shape
        return jnp.fft.ifftn(x.reshape(s[:-1] + n), axes=(-3, -2, -1)).real.reshape(s)

    def div_of(sigma_field):
        """div(sigma), (3,3,Nv) -> (3,Nv)."""
        sigma_hat = fft_(sigma_field)
        return ifft_(jnp.einsum("ijm,jm->im", sigma_hat, iq))

    def strain_from_correction(dU):
        """Pure-fluctuation strain from a Newton correction's displacement-like
        unknown -- zero mean (DC bin), unlike solve_displacement_based's
        strain_from_u (which embeds a macroscopic correction there instead):
        eps_bar is already fixed as the baseline here, nothing left to solve
        for in the mean."""
        dU_hat   = fft_(dU)
        grad_hat = jnp.einsum("im,jm->ijm", dU_hat, iq)
        eps_hat  = 0.5 * (grad_hat + jnp.transpose(grad_hat, (1, 0, 2)))
        eps_hat  = eps_hat.at[:, :, 0].set(0.0)
        return ifft_(eps_hat)

    eps0  = jnp.ones((3, 3, Nv)) * eps_bar[:, :, None]
    delta = jnp.zeros((3, 3, Nv)) if delta_init is None else delta_init
    # The macroscopic strain is carried entirely by eps0 above; delta must be
    # a zero-mean fluctuation on top of it, or the true average strain
    # silently drifts away from eps_bar. Newton's own corrections are always
    # zero-mean by construction (strain_from_correction zeros the DC bin
    # below), but a caller-supplied delta_init has no such guarantee -- e.g.
    # a naive load-stepping warm start using the previous step's full field
    # minus THIS step's new baseline has a nonzero mean whenever eps_bar
    # changed, and in the worst case that alone can satisfy div(sigma)=0
    # trivially (e.g. at eps=0) well short of the actual target strain,
    # reporting false convergence. Project unconditionally rather than
    # trusting the caller to have gotten this right.
    delta_hat = fft_(delta)
    delta_hat = delta_hat.at[:, :, 0].set(0.0)
    delta = ifft_(delta_hat)
    state = state_init
    converged = False
    n_iter = 0

    for n_iter in range(1, maxiter_nr + 1):
        eps_k = eps0 + delta
        sigma_k, C_tan_k, new_state = local_update(eps_k, state)

        residual   = div_of(sigma_k)                     # (3, Nv) -- want 0
        resid_norm = float(jnp.linalg.norm(residual))
        ref_norm   = float(jnp.linalg.norm(sigma_k)) + 1e-30
        if resid_norm / ref_norm < toler_nr:
            converged = True
            state = new_state
            break

        def A_op(x_flat):
            eps_trial   = strain_from_correction(x_flat.reshape(3, Nv))
            sigma_trial = jnp.einsum("ijklm,klm->ijm", C_tan_k, eps_trial)
            return div_of(sigma_trial).reshape(-1)

        C0_ref  = jnp.mean(C_tan_k, axis=-1) if C0 is None else C0
        K0_hat  = jnp.einsum("jm,ijkl,lm->ikm", iq.imag, C0_ref, iq.imag)
        null_pt = jnp.all(iq.imag == 0.0, axis=0)
        K0_hat  = jnp.where(null_pt[None, None, :], jnp.eye(3)[:, :, None], K0_hat)
        K0_inv  = jnp.moveaxis(jnp.linalg.inv(jnp.moveaxis(K0_hat, -1, 0)), 0, -1)

        def M(r_flat):
            r_hat = fft_(r_flat.reshape(3, Nv))
            z_hat = jnp.einsum("ikm,km->im", K0_inv, r_hat)
            return ifft_(z_hat).reshape(-1)

        bb = (-residual).reshape(-1)
        x0 = jnp.zeros_like(bb)
        Delta_flat, _cg_converged = cg_solve(A_op, bb, x0, toler_lin, maxiter_lin, M=M)

        # Delta_flat is the CG unknown -- a displacement-like field (3, Nv),
        # not a strain tensor -- the actual strain correction is its
        # symmetric gradient, same as A_op's own eps_trial above.
        delta = delta + strain_from_correction(Delta_flat.reshape(3, Nv))
        # ``state`` is deliberately NOT advanced here. It is the state at the
        # last CONVERGED load increment, and a return mapping is defined
        # relative to exactly that -- it must stay frozen for every Newton
        # iteration of this increment, with only ``delta`` (the unknown) moving.
        # Advancing it per iteration instead makes each iteration's return
        # start from the previous iterate's already-updated plastic strain, so
        # plastic flow ratchets up once per iteration and sigma drifts under
        # Newton's feet. Hardening masks it -- the drift self-limits as sigma_y
        # grows, so the residual still falls, just to a stalled ~1e-7 floor
        # instead of ~1e-10, at a fixed ~10 iterations per step regardless of
        # the load. Push the load far enough (a confined DP matrix past
        # eps_11 ~ 0.03) and it stops being masked: Newton descends about five
        # iterations, turns around and diverges geometrically to a garbage
        # fixed point, with the inner CG reporting success throughout and the
        # step size making no difference. Verified on a 152^3 tangled-fibre
        # RVE -- freezing state here took that step from divergence to
        # convergence in 8 iterations, and cut every earlier step from ~10 to
        # ~6 (see test/test_problems_mechanics_nonlinear.py, check 4).
    else:
        # loop exhausted maxiter_nr without an early break -- converged stays
        # False and ``state`` is still the last converged increment's, so a
        # caller that ignores the flag resumes from a consistent state rather
        # than from a half-converged iterate's.
        pass

    eps_final = eps0 + delta
    sigma_final, _, _ = local_update(eps_final, state)
    return eps_final, sigma_final, state, converged, n_iter
