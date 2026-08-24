"""
Abaqus-*STATIC-style incremental loading: wraps a per-increment solve
callable in a load-stepping loop, either fixed (equal increments of a given
size) or automatic (adaptive step, grown after a converged increment, cut
back and retried after a non-converged one).

Deliberately solver-agnostic: the per-increment solve is any callable
``solve_fn(t) -> object with a .converged usable via bool(...)``, where ``t`` is
the cumulative load fraction in (0, 1] to scale a target load by (the
caller's closure owns what "load" means -- e.g. ``lambda t:
solve_mechanics(n, L, phase, materials, t * eps_bar, ...)``). Named ``t``/
``dt`` (not ``lam``/``dlam``) to match the archived scripts' pseudo-time
convention (``t``, ``dt``, ``t_end``) even though there's no physical time
here, just a load proxy -- Abaqus's own *STATIC step calls this "time" too.

Right now the only per-increment solve in this project is
``solve_mechanics`` (pure linear elasticity -- always converges in a single
CG solve, so cutback only ever triggers on genuine CG non-convergence).
Once a nonlinear (Newton) solve exists, it plugs in here unchanged -- this
driver only reads ``.converged``, it has no idea what happened inside.

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
from typing import Callable, NamedTuple, Protocol, runtime_checkable


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
    t:         float      # cumulative load fraction reached, 0 < t <= 1
    dt:        float      # this increment's load-fraction size
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
    problems.mechanics.solve_mechanics_incremental writing each accepted
    increment to disk as it's produced rather than only at the end).
    """
    if not (0.0 < dt <= 1.0):
        raise ValueError(f"dt must be in (0, 1], got {dt}")

    n_steps = max(1, round(1.0 / dt))
    dt_actual = 1.0 / n_steps
    results = []
    for step in range(1, n_steps + 1):
        t = step * dt_actual
        t0 = time.perf_counter()
        sol = solve_fn(t)
        converged = bool(sol.converged)
        wall_time = time.perf_counter() - t0
        if not converged:
            raise RuntimeError(f"Increment {step}/{n_steps} (t={t:.4f}) did not converge")
        result = IncrementResult(step=step, t=t, dt=dt_actual, solution=sol, wall_time=wall_time)
        results.append(result)
        if on_increment is not None:
            on_increment(result)
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

    results = []
    t, dt, step = 0.0, dt_init, 0

    while t < 1.0:
        if step >= max_steps:
            raise RuntimeError(f"Exceeded max_steps={max_steps} before reaching t=1 (t={t:.4f})")
        step += 1
        dt = min(dt, dt_max, 1.0 - t)   # don't overshoot t=1

        converged = False
        t0 = time.perf_counter()
        for attempt in range(max_cutbacks + 1):
            t_trial = t + dt
            sol = solve_fn(t_trial)
            converged = bool(sol.converged)
            if converged:
                break
            dt = max(dt * factor_dec, dt_min)
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
