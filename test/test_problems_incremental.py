"""
Standalone test for problems/incremental.py's solve_path -- the prescribed-
path load stepping that scripts/simulation/solve_inelastic.py drives its
load / unload / reload cycle through.

Pure bookkeeping, no solve: solve_fn is a stub that records what it was
called with and returns a canned .converged, so the test costs milliseconds.

Four checks
-----------
1. One solve_fn call per path entry, in order, with the path value itself
   -- including a reversing path, which solve_fixed/solve_automatic (load
   fraction running monotonically to 1) cannot express.
2. IncrementResult bookkeeping: step counts from 1 (or first_step), t is
   the path value, dt the SIGNED change from the previous entry (from 0 for
   the first).
3. A non-converged entry ends the path without raising: the results up to
   and including it are returned, it is last, and on_increment fired for it
   too (the caller writes it out as the diagnostic) -- but nothing after it
   was solved.
4. An empty path is rejected rather than returning an empty list, which
   every caller would index with results[-1].

Usage
-----
    python -m pytest test/test_problems_incremental.py
"""

import sys
sys.path.insert(0, "src")

from typing import NamedTuple

from problems.incremental import solve_path


class _Sol(NamedTuple):
    t:         float
    converged: bool


def _stub(fail_at=None):
    calls = []

    def solve_fn(t):
        calls.append(t)
        return _Sol(t=t, converged=(fail_at is None or len(calls) != fail_at))
    return solve_fn, calls


path = [0.0, 0.01, 0.02, 0.0, -0.02, 0.01]

# ── [1] one call per entry, in order, path values passed through ────────────
solve_fn, calls = _stub()
seen = []
results = solve_path(solve_fn, path, on_increment=seen.append)
assert calls == path, f"solve_fn called with {calls}, expected {path}"
assert seen == results, "on_increment must fire once per increment, in order"
print("[1] PASSED")

# ── [2] step / t / dt bookkeeping ───────────────────────────────────────────
assert [r.step for r in results] == list(range(1, len(path) + 1))
assert [r.t for r in results] == path
expected_dt = [b - a for a, b in zip([0.0] + path[:-1], path)]
assert all(abs(r.dt - d) < 1e-15 for r, d in zip(results, expected_dt)), \
    f"dt {[r.dt for r in results]} != {expected_dt}"
assert all(r.solution.t == r.t for r in results)
assert [r.step for r in solve_path(_stub()[0], path, first_step=0)] == list(range(len(path))), \
    "first_step=0 numbers the reference entry step 0 (solve_inelastic's virgin state)"
print("[2] PASSED")

# ── [3] a failed entry ends the path, is returned last, and is reported ─────
solve_fn, calls = _stub(fail_at=4)
seen = []
results = solve_path(solve_fn, path, on_increment=seen.append)
assert calls == path[:4], f"nothing after the failed entry may be solved, got {calls}"
assert len(results) == 4 and seen == results
assert not results[-1].solution.converged and all(r.solution.converged for r in results[:-1])
print("[3] PASSED")

# ── [4] empty path rejected ─────────────────────────────────────────────────
try:
    solve_path(_stub()[0], [])
except ValueError:
    pass
else:
    raise AssertionError("expected a ValueError for an empty path")
print("[4] PASSED")

print("\ntest_problems_incremental: all checks passed")
