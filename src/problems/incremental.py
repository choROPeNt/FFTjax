"""
Abaqus *STATIC-style load stepping around any ``solve_fn(t)``, where ``t`` is
the cumulative load fraction in (0, 1] and the result has ``.converged``.

``.converged`` is read directly, not via ``getattr(..., default)``: that was
~30x slower on JAX-pytree NamedTuples.
"""

from __future__ import annotations

import time
from typing import Callable, NamedTuple, Protocol, runtime_checkable


@runtime_checkable
class _Boolable(Protocol):
    def __bool__(self) -> bool: ...


@runtime_checkable
class Solution(Protocol):
    """
    What ``solve_fn`` returns: anything with a bool-able ``.converged``.

    A property, so an Array-typed field matches structurally.
    """
    @property
    def converged(self) -> _Boolable: ...


class IncrementResult(NamedTuple):
    """
    One accepted increment.

    Attributes
    ----------
    step      : 1-based increment number
    t         : cumulative load fraction, 0 < t <= 1
    dt        : load fraction of this increment
    solution  : what solve_fn(t) returned
    wall_time : seconds, including failed cutback attempts
    """

    step:      int
    t:         float
    dt:        float
    solution:  Solution
    wall_time: float


def solve_fixed(
    solve_fn: Callable[[float], Solution],
    dt: float,
    on_increment: Callable[[IncrementResult], None] | None = None,
) -> list[IncrementResult]:
    """
    ``round(1/dt)`` equal increments (dt = 0.3 gives three steps of 1/3).

    Raises RuntimeError on the first non-converged increment (no cutback).
    ``on_increment`` is called once per accepted increment.
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
    Adaptive steps: grow by ``factor_inc`` after convergence (up to ``dt_max``),
    shrink by ``factor_dec`` and retry after failure (down to ``dt_min``).

    Raises RuntimeError after ``max_cutbacks`` failures in one increment or
    ``max_steps`` increments. ``on_increment`` is called once per accepted
    increment, never for a failed attempt.
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
        dt = min(dt, dt_max, 1.0 - t)

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
