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
into a per-voxel stress+tangent). It supports the same mixed strain/stress
``control``/``stress_goal`` convention as solve_displacement_based, ported
into the Newton loop: the macroscopic strain on stress-controlled
directions is its own Newton unknown (``eps_bar_free``), accumulated across
iterations separately from the fluctuation ``delta`` -- see
solve_displacement_based_nonlinear's own docstring for why that separation
matters.
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
from solvers.elliptic.vector.displacement_based import DisplacementBasedSolver, _active_pairs
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
    n:                 Tuple[int, ...],
    xi_flat:           jnp.ndarray,
    eps_bar:           jnp.ndarray,
    local_update:      Callable[[jnp.ndarray, Any], Tuple[jnp.ndarray, jnp.ndarray, Any]],
    state_init:        Any,
    toler_lin:         float = 1e-6,
    maxiter_lin:       int = 1000,
    toler_nr:          float = 1e-8,
    maxiter_nr:        int = 50,
    C0:                jnp.ndarray | None = None,
    delta_init:        jnp.ndarray | None = None,
    control:           Tuple[Tuple[int, ...], ...] | None = None,
    stress_goal:       jnp.ndarray | None = None,
    eps_bar_free_init: jnp.ndarray | None = None,
):
    """
    Newton-outer / CG-inner solve for a nonlinear local constitutive law:
    div(sigma(eps)) = 0 on a periodic voxel grid, prescribed macroscopic
    strain ``eps_bar``, optionally with mixed strain/stress macroscopic BC
    via ``control``/``stress_goal`` (same convention as
    solve_displacement_based); ``control`` omitted or all-zero (default) is
    pure strain BC.

    Same FFT-based grad/div machinery as solve_displacement_based (frequency
    grid, Nyquist-safe xi, strain-from-fluctuation via the DC-bin trick),
    reused directly rather than re-derived, to avoid a second, possibly
    inconsistent implementation of the same operator. The difference is what
    drives each linear CG solve: solve_displacement_based solves once
    against a FIXED C_field; this solves repeatedly, at each outer (Newton)
    iteration evaluating the actual nonlinear stress and its tangent at the
    CURRENT trial strain via ``local_update``, using CG only for the
    *linearized correction* -- true Newton, not a fixed-point/secant scheme.
    Mixed BC folds in the same way solve_displacement_based's bordered
    system does: the CG unknown at each Newton iteration is
    ``pack(du, sv)`` (fluctuation correction, macroscopic-stress-controlled
    strain correction), solved jointly against the current tangent.

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
    eps_bar      : (3, 3)   prescribed macroscopic strain; entries where
                   ``control == 1`` are ignored (solved for instead)
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
                   carried by eps0 + eps_bar_free, never by delta.
    C0           : (3,3,3,3) or None -- fixed reference stiffness for the CG
                   preconditioner; None re-derives it from the CURRENT tangent
                   field every Newton iteration (voxel-mean of C_tan), matching
                   solve_displacement_based's default. Passing a fixed elastic
                   C0 avoids rebuilding/inverting the preconditioner each
                   iteration -- standard practice for FFT-based Newton-Krylov
                   plasticity, not yet benchmarked either way in this project.
    control      : (3, 3) 0/1 mask, 1 = stress-controlled, 0 = strain-controlled.
                   None (default) = pure strain BC (all zero), reproducing
                   this function's original behavior exactly -- every
                   pack/unpack/sv helper below degenerates to a no-op when
                   there are no active (i, j) pairs, so this is a strict
                   superset of the pure-strain code, not a separate path.
    stress_goal  : (3, 3) macroscopic target stress; only entries where
                   ``control == 1`` are used. None = zeros.
    eps_bar_free_init : (3, 3) or None -- initial guess for the macroscopic
                   strain correction on stress-controlled directions (i.e.
                   the ``sv``/``deps_bar_free`` unknown, accumulated across
                   Newton iterations the same way ``delta`` accumulates its
                   own correction -- see below). None starts from zero.
                   Warm-starting this across load steps is the mixed-BC
                   analogue of ``delta_init`` (e.g. the previous, smaller
                   load level's converged lateral contraction is a much
                   better starting guess than 0 for the next level).

    Returns
    -------
    eps       : (3, 3, Nv)
    sigma     : (3, 3, Nv)
    state     : final per-voxel state
    converged : bool -- True if the Newton residual met toler_nr within maxiter_nr
    n_iter    : int  -- Newton iterations actually run

    The resolved macroscopic strain (only meaningful when ``control`` has
    active entries) is recoverable as ``jnp.mean(eps, axis=-1)`` -- not
    returned separately, to keep this function's return signature identical
    for every existing pure-strain caller.
    """
    Nv = prod(n)
    iq = 1j * nyquist_safe_xi(xi_flat, n)  # (3, Nv)

    control = control if control is not None else _ZERO_CONTROL
    pairs = _active_pairs(control)
    control_arr = jnp.asarray(control, dtype=eps_bar.dtype)
    stress_goal = jnp.zeros((3, 3), dtype=eps_bar.dtype) if stress_goal is None else stress_goal

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
        """Pure-fluctuation strain from a displacement-like unknown -- zero
        mean (DC bin) unconditionally. Used to extract delta's own update
        from a Newton correction's du block: the macroscopic-strain update
        (mixed BC) is carried entirely by eps_bar_free instead (see
        strain_from_u below), so this stays exactly as before regardless of
        whether control is active -- delta's zero-mean invariant is
        untouched by turning mixed BC on."""
        dU_hat   = fft_(dU)
        grad_hat = jnp.einsum("im,jm->ijm", dU_hat, iq)
        eps_hat  = 0.5 * (grad_hat + jnp.transpose(grad_hat, (1, 0, 2)))
        eps_hat  = eps_hat.at[:, :, 0].set(0.0)
        return ifft_(eps_hat)

    # ── mixed-BC bookkeeping, same convention as solve_displacement_based --
    #    pairs=() (pure strain) makes every one of these a no-op.
    def sv2sm(sv):
        sm = jnp.zeros((3, 3), dtype=sv.dtype)
        for k, (i, j) in enumerate(pairs):
            sm = sm.at[i, j].set(sv[k])
            sm = sm.at[j, i].set(sv[k])
        return sm

    def sm2sv(sm):
        if not pairs:
            return jnp.zeros((0,), dtype=sm.dtype)
        return jnp.stack([sm[i, j] for i, j in pairs])

    def unpack(x_flat):
        du = x_flat[: 3 * Nv].reshape(3, Nv)
        sv = x_flat[3 * Nv:]
        return du, sv

    def pack(du, sv):
        return jnp.concatenate([du.reshape(-1), sv])

    def strain_from_u(du, deps_bar_free):
        """Like strain_from_correction, but embeds deps_bar_free into the DC
        bin instead of zeroing it -- used only inside a Newton iteration's
        own linearized CG sub-solve, where the fluctuation and macroscopic
        corrections are coupled through the same tangent and must be solved
        for jointly. Reduces to strain_from_correction exactly when
        deps_bar_free is zero (in particular whenever pairs=())."""
        du_hat   = fft_(du)
        grad_hat = jnp.einsum("im,jm->ijm", du_hat, iq)
        eps_hat  = 0.5 * (grad_hat + jnp.transpose(grad_hat, (1, 0, 2)))
        eps_hat  = eps_hat.at[:, :, 0].set(Nv * deps_bar_free.astype(eps_hat.dtype))
        return ifft_(eps_hat)

    eps0  = jnp.ones((3, 3, Nv)) * (eps_bar * (1.0 - control_arr))[:, :, None]
    delta = jnp.zeros((3, 3, Nv)) if delta_init is None else delta_init
    eps_bar_free = jnp.zeros((3, 3), dtype=eps_bar.dtype) if eps_bar_free_init is None else eps_bar_free_init
    # The macroscopic strain is carried entirely by eps0 (strain-controlled
    # directions) + eps_bar_free (stress-controlled directions, solved for);
    # delta must be a zero-mean fluctuation on top of that, or the true
    # average strain silently drifts away from eps_bar. Newton's own
    # corrections are always zero-mean by construction (strain_from_correction
    # zeros the DC bin below), but a caller-supplied delta_init has no such
    # guarantee -- e.g. a naive load-stepping warm start using the previous
    # step's full field minus THIS step's new baseline has a nonzero mean
    # whenever eps_bar changed, and in the worst case that alone can satisfy
    # div(sigma)=0 trivially (e.g. at eps=0) well short of the actual target
    # strain, reporting false convergence. Project unconditionally rather
    # than trusting the caller to have gotten this right.
    delta_hat = fft_(delta)
    delta_hat = delta_hat.at[:, :, 0].set(0.0)
    delta = ifft_(delta_hat)
    state = state_init
    converged = False
    n_iter = 0

    for n_iter in range(1, maxiter_nr + 1):
        eps_bar_free_bcast = jnp.ones((3, 3, Nv)) * eps_bar_free[:, :, None]
        eps_k = eps0 + eps_bar_free_bcast + delta
        sigma_k, C_tan_k, new_state = local_update(eps_k, state)

        residual_div   = div_of(sigma_k)                  # (3, Nv) -- want 0
        sigma_sum      = jnp.real(fft_(sigma_k)[:, :, 0])  # DC bin = sum over voxels
        residual_extra = sm2sv(sigma_sum) - Nv * sm2sv(stress_goal)  # want 0

        resid_norm = float(jnp.sqrt(jnp.sum(residual_div ** 2) + jnp.sum(residual_extra ** 2)))
        ref_norm   = float(jnp.linalg.norm(sigma_k)) + 1e-30
        if resid_norm / ref_norm < toler_nr:
            converged = True
            state = new_state
            break

        def A_op(x_flat):
            du, sv      = unpack(x_flat)
            eps_trial   = strain_from_u(du, sv2sm(sv))
            sigma_trial = jnp.einsum("ijklm,klm->ijm", C_tan_k, eps_trial)
            sigma_hat   = fft_(sigma_trial)
            div_flat    = ifft_(jnp.einsum("ijm,jm->im", sigma_hat, iq))
            # Negated, exactly as solve_displacement_based's A_op does it --
            # the minus is what makes the bordered operator SYMMETRIC, not a
            # convention that can be re-derived per solver. div_of is -B^H
            # (the symmetric-gradient's adjoint, up to sign), so the du row's
            # coupling to sv is -B^H C S while the sv row's coupling to du is
            # +/- S^H C B; only the negated form is that block's transpose.
            # Unnegated, the operator is asymmetric (2e-1 relative on a 4^3
            # test grid, against 2e-16 negated) and its symmetric part gains
            # one positive eigenvalue per stress-controlled pair, i.e. turns
            # indefinite -- both of the assumptions cg_solve rests on. The
            # SOLUTION is unaffected either way (negating a row together with
            # its RHS entry is an identity, see bb below), so this is purely
            # about whether CG can find it: unnegated, check 6's load ramp
            # diverges outright at toler_lin=1e-8 (sigma33 -> -4.8e3 after 50
            # Newton iterations) and only finishes at 1e-6/300; negated, the
            # same ramp converges at 1e-8 in 6 Newton iterations.
            extra_out   = -sm2sv(jnp.real(sigma_hat[:, :, 0]))
            return pack(div_flat, extra_out)

        C0_ref  = jnp.mean(C_tan_k, axis=-1) if C0 is None else C0
        K0_hat  = jnp.einsum("jm,ijkl,lm->ikm", iq.imag, C0_ref, iq.imag)
        null_pt = jnp.all(iq.imag == 0.0, axis=0)
        K0_hat  = jnp.where(null_pt[None, None, :], jnp.eye(3)[:, :, None], K0_hat)
        K0_inv  = jnp.moveaxis(jnp.linalg.inv(jnp.moveaxis(K0_hat, -1, 0)), 0, -1)

        # "extra" block preconditioner: self-coupling of the macroscopic-
        # strain unknowns at the reference medium, sign-matched to A_op's
        # extra_out above (same as solve_displacement_based's own P_extra) --
        # keeping M positive definite against a negative definite A, which is
        # the pairing that makes preconditioned CG on this system equivalent
        # to CG on an SPD one.
        if pairs:
            basis   = jnp.eye(len(pairs), dtype=eps_bar.dtype)
            P_extra = jnp.stack([
                -Nv * sm2sv(jnp.einsum("ijkl,kl->ij", C0_ref, sv2sm(basis[k])))
                for k in range(len(pairs))
            ], axis=1)

        def M(x_flat):
            du, sv = unpack(x_flat)
            r_hat  = fft_(du)
            z_hat  = jnp.einsum("ikm,km->im", K0_inv, r_hat)
            z_du   = ifft_(z_hat)
            z_sv   = jnp.linalg.solve(P_extra, sv) if pairs else sv
            return pack(z_du, z_sv)

        # The extra block enters as +residual_extra, not -: A_op's extra row
        # is negated (see above), so its RHS is negated with it, which leaves
        # the equation it encodes -- mean(sigma_k + C_tan : deps) = stress_goal
        # -- exactly as it was.
        bb = pack((-residual_div).reshape(-1), residual_extra)
        x0 = jnp.zeros_like(bb)
        x_flat, _cg_converged = cg_solve(A_op, bb, x0, toler_lin, maxiter_lin, M=M)

        # x_flat's du block is a displacement-like field (3, Nv), not a
        # strain tensor -- the actual fluctuation correction is its symmetric
        # gradient, same as A_op's own eps_trial above; the sv block is the
        # macroscopic-strain correction, accumulated separately from delta.
        du_sol, sv_sol = unpack(x_flat)
        delta = delta + strain_from_correction(du_sol)
        eps_bar_free = eps_bar_free + sv2sm(sv_sol)
        # ``state`` is deliberately NOT advanced here. It is the state at the
        # last CONVERGED load increment, and a return mapping is defined
        # relative to exactly that -- it must stay frozen for every Newton
        # iteration of this increment, with only ``delta``/``eps_bar_free``
        # (the unknowns) moving. Advancing it per iteration instead makes
        # each iteration's return start from the previous iterate's
        # already-updated plastic strain, so plastic flow ratchets up once
        # per iteration and sigma drifts under Newton's feet. Hardening
        # masks it -- the drift self-limits as sigma_y grows, so the residual
        # still falls, just to a stalled ~1e-7 floor instead of ~1e-10, at a
        # fixed ~10 iterations per step regardless of the load. Push the load
        # far enough (a confined DP matrix past eps_11 ~ 0.03) and it stops
        # being masked: Newton descends about five iterations, turns around
        # and diverges geometrically to a garbage fixed point, with the inner
        # CG reporting success throughout and the step size making no
        # difference. Verified on a 152^3 tangled-fibre RVE -- freezing state
        # here took that step from divergence to convergence in 8 iterations,
        # and cut every earlier step from ~10 to ~6 (see
        # test/test_problems_mechanics_nonlinear.py, check 4).
    else:
        # loop exhausted maxiter_nr without an early break -- converged stays
        # False and ``state`` is still the last converged increment's, so a
        # caller that ignores the flag resumes from a consistent state rather
        # than from a half-converged iterate's.
        pass

    eps_bar_free_bcast = jnp.ones((3, 3, Nv)) * eps_bar_free[:, :, None]
    eps_final = eps0 + eps_bar_free_bcast + delta
    sigma_final, _, _ = local_update(eps_final, state)
    return eps_final, sigma_final, state, converged, n_iter
