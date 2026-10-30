"""
Abaqus-*STATIC-style incremental loading: wraps a per-increment solve
callable in a load-stepping loop, either fixed (equal increments of a given
size), automatic (adaptive step, grown after a converged increment, cut
back and retried after a non-converged one), or a prescribed path (an
explicit list of load points, any order -- e.g. a load/unload/reload
cycle, see solve_path).

Deliberately solver-agnostic: the per-increment solve is any callable
``solve_fn(t) -> object with a .converged usable via bool(...)``, where ``t`` is
the cumulative load fraction in (0, 1] to scale a target load by (the
caller's closure owns what "load" means -- e.g. ``lambda t:
_solve_mechanics_step(n, L, phase, materials, t * eps_bar, ...)``). Named
``t``/``dt`` (not ``lam``/``dlam``) to match the archived scripts'
pseudo-time convention (``t``, ``dt``, ``t_end``) even though there's no
physical time here, just a load proxy -- Abaqus's own *STATIC step calls
this "time" too.

Right now the only per-increment solve in this project is
problems.mechanics._solve_mechanics_step (pure linear elasticity -- always
converges in a single CG solve, so cutback only ever triggers on genuine CG
non-convergence). Once a nonlinear (Newton) solve exists, it plugs in here
unchanged -- this driver only reads ``.converged``, it has no idea what
happened inside.

``problems.mechanics.solve_mechanics`` is the one public entry point over
this driver -- its ``stepping`` argument picks "single" (one solve_fn call,
no stepping), "fixed", or "automatic" and always returns
list[IncrementResult] regardless of which.

``.converged`` is read via direct attribute access (``sol.converged``), not
``getattr(sol, "converged", True)`` -- measured ~30x slower in practice for
a JAX-pytree-registered NamedTuple like ElasticitySolution (something about
getattr's default-value fallback path interacts badly with its pytree
machinery). Since the contract already requires ``.converged`` to exist
(see the ``Solution`` protocol below), the defensive default was never
doing anything except quietly wrecking performance.
"""

from __future__ import annotations

import time
from typing import Callable, NamedTuple, Protocol, Sequence, runtime_checkable

from utils.monitor import Monitor, get_monitor


@runtime_checkable
class _Boolable(Protocol):
    def __bool__(self) -> bool: ...


@runtime_checkable
class Solution(Protocol):
    """What solve_fn(t) must return: anything with a .converged usable via bool(...)
    -- a plain bool or a 0-d JAX/numpy array both satisfy this. Declared as a
    read-only property (not a plain attribute) so ElasticitySolution's actual
    Array-typed field matches structurally -- a mutable attribute would need
    an exact (invariant) type match, which Array vs bool never is."""
    @property
    def converged(self) -> _Boolable: ...


class IncrementResult(NamedTuple):
    step:      int        # increment number, 1-based
    t:         float      # cumulative load fraction reached, 0 < t <= 1 (solve_path:
                           # the path value, any sign)
    dt:        float      # this increment's load-fraction size (solve_path: signed)
    solution:  Solution   # whatever solve_fn(t) returned
    wall_time: float      # seconds spent on this increment, wall-clock (includes
                           # any failed cutback attempts before the accepted one)


def solve_fixed(
    solve_fn: Callable[[float], Solution],
    dt: float,
    on_increment: Callable[[IncrementResult], None] | None = None,
) -> list[IncrementResult]:
    """
    Equal load-fraction increments of size ``dt`` from 0 to 1. ``dt`` is a
    target: the actual step count is ``round(1/dt)`` and the increments are
    then resized to exactly ``1/n_steps`` so every step is genuinely equal
    (the whole point of "fixed") -- e.g. dt=0.3 gives 3 steps of exactly
    1/3, not 0.3, 0.3, 0.3, 0.1. Raises RuntimeError on the first
    non-converged increment -- no cutback/retry.

    ``on_increment``, if given, is called exactly once per *accepted*
    increment (never on a failed attempt -- there's no cutback here, but the
    signature matches solve_automatic's for a common caller, e.g.
    problems.mechanics.solve_mechanics writing each accepted increment to
    disk as it's produced rather than only at the end).
    """
    if not (0.0 < dt <= 1.0):
        raise ValueError(f"dt must be in (0, 1], got {dt}")

    n_steps = max(1, round(1.0 / dt))
    dt_actual = 1.0 / n_steps
    mon = get_monitor()
    results = []
    for step in range(1, n_steps + 1):
        t = step * dt_actual
        with mon.stage("increment", step):
            t0 = time.perf_counter()
            sol, converged = _solve_attempt(mon, solve_fn, t, dt_actual, 1)
            wall_time = time.perf_counter() - t0
            if not converged:
                raise RuntimeError(f"Increment {step}/{n_steps} (t={t:.4f}) did not converge")
            result = IncrementResult(step=step, t=t, dt=dt_actual, solution=sol, wall_time=wall_time)
            results.append(result)
            if on_increment is not None:
                on_increment(result)
    return results


def _solve_attempt(mon: Monitor, solve_fn: Callable[[float], Solution], t: float, dt: float,
                   attempt: int) -> tuple[Solution, bool]:
    """
    One solve_fn call as a monitored "solve" stage -- one per attempt, so a
    cutback sequence shows every attempt (at log level "solve") rather than
    only the accepted one. ``bool(sol.converged)`` inside the stage is what
    forces the result to the host, so the stage's time is the solve's.
    """
    with mon.stage("solve", attempt, start=f"t={t:.5f}  dt={dt:.5f}") as st:
        sol = solve_fn(t)
        converged = bool(sol.converged)
        st.note(f"converged={converged}")
    return sol, converged


def solve_path(
    solve_fn: Callable[[float], Solution],
    path: Sequence[float],
    on_increment: Callable[[IncrementResult], None] | None = None,
    first_step: int = 1,
) -> list[IncrementResult]:
    """
    One increment per entry of ``path``, in order: ``solve_fn(path[k])`` for
    k = 0, 1, .... Unlike solve_fixed/solve_automatic the load parameter is
    not a fraction running monotonically to 1 -- ``path`` is whatever the
    caller's closure understands (e.g. the engineering strain gamma of a
    load/unload/reload cycle, see problems.utils.loadcases.cycle_gammas), and
    it may reverse direction. ``IncrementResult.t`` is the path value itself
    and ``.dt`` the signed change from the previous one (from 0 for the
    first entry); ``.step`` counts from ``first_step`` (1 by default, 0 for
    a path whose first entry is a reference state the output calls step 0,
    as solve_inelastic's virgin gamma = 0 is). No cutback: the path is prescribed, so there is no smaller
    step to retry with.

    A non-converged increment ends the path WITHOUT raising (solve_fixed
    raises): the results up to and including the failed one are returned,
    the failed one last, so the caller keeps the partial output and reads
    ``results[-1].solution.converged`` to tell. ``on_increment`` fires for
    that failed increment too -- unlike solve_automatic's, where a failed
    attempt is followed by a retry and is never final, here it is the last
    word on that load point, and recording it (stats, fields) is the
    diagnostic.
    """
    if len(path) == 0:
        raise ValueError("path is empty")

    mon = get_monitor()
    results = []
    t_prev = 0.0
    for step, t in enumerate(path, start=first_step):
        t = float(t)
        with mon.stage("increment", step):
            t0 = time.perf_counter()
            sol, converged = _solve_attempt(mon, solve_fn, t, t - t_prev, 1)
            wall_time = time.perf_counter() - t0
            result = IncrementResult(step=step, t=t, dt=t - t_prev, solution=sol, wall_time=wall_time)
            results.append(result)
            if on_increment is not None:
                on_increment(result)
        if not converged:
            break
        t_prev = t
    return results


def solve_automatic(
    solve_fn:     Callable[[float], Solution],
    dt_init:      float = 0.1,
    dt_min:       float = 1e-4,
    dt_max:       float = 0.5,
    factor_inc:   float = 1.5,
    factor_dec:   float = 0.5,
    max_cutbacks: int   = 5,
    max_steps:    int   = 1000,
    on_increment: Callable[[IncrementResult], None] | None = None,
) -> list[IncrementResult]:
    """
    Adaptive load-fraction step: grows by ``factor_inc`` after a converged
    increment (capped at ``dt_max``), shrinks by ``factor_dec`` and retries
    (up to ``max_cutbacks`` times) after a non-converged one (floored at
    ``dt_min``). Raises RuntimeError if an increment exhausts its cutback
    budget, or the run exceeds ``max_steps`` before reaching t=1.

    ``on_increment``, if given, is called exactly once per *accepted*
    increment -- never on a failed cutback attempt, since those don't
    represent real progress (e.g. shouldn't be written to disk as if they
    were).
    """
    if not (0.0 < dt_min <= dt_init <= dt_max):
        raise ValueError(
            f"require 0 < dt_min <= dt_init <= dt_max, got {dt_min}, {dt_init}, {dt_max}"
        )

    mon = get_monitor()
    results = []
    t, dt, step = 0.0, dt_init, 0

    while t < 1.0:
        if step >= max_steps:
            raise RuntimeError(f"Exceeded max_steps={max_steps} before reaching t=1 (t={t:.4f})")
        step += 1
        dt = min(dt, dt_max, 1.0 - t)   # don't overshoot t=1

        with mon.stage("increment", step):
            converged = False
            t0 = time.perf_counter()
            for attempt in range(max_cutbacks + 1):
                t_trial = t + dt
                sol, converged = _solve_attempt(mon, solve_fn, t_trial, dt, attempt + 1)
                if converged:
                    break
                dt_failed, dt = dt, max(dt * factor_dec, dt_min)
                # Shown at the default level: a cutback is rare, and without
                # this line a run of them looks like one very slow increment.
                mon.log("increment", f"cutback: t={t_trial:.5f} not converged, "
                                     + (f"retrying dt {dt_failed:.5f} -> {dt:.5f}"
                                        if attempt < max_cutbacks else "no cutbacks left"))
            wall_time = time.perf_counter() - t0

            if not converged:
                raise RuntimeError(
                    f"Increment {step} did not converge after {max_cutbacks} cutbacks "
                    f"at t={t:.4f} (dt floored at {dt_min})"
                )

            t = t_trial
            result = IncrementResult(step=step, t=t, dt=dt, solution=sol, wall_time=wall_time)
            results.append(result)
            if on_increment is not None:
                on_increment(result)
        dt = min(dt * factor_inc, dt_max)

    return results
