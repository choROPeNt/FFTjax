"""
Standalone test for utils/monitor.py -- the one place every solve layer
reports progress and timing through -- and for its wiring into
problems/incremental.py.

No solve happens here: solve_fn is a stub, output goes to a StringIO, so the
test costs milliseconds. What it guards is the output contract the drivers
and anyone reading a run log rely on.

Six checks
----------
1. No active monitor -> silent: library code that opens stages prints
   nothing (tests, notebooks, vmapped solves), and get_monitor() is back to
   silent after a run's ``with Monitor(...)`` exits.
2. Level filtering: a stage gets its own line only if its kind is at or
   above the level; deeper stages still show up in their parent's
   solve=/write=/total= breakdown.
3. Line layout: label + index, HH:MM:SS stamp, notes in order, timing last;
   detail() lines follow the closing line; note(kind=...) reaches the
   enclosing stage of that kind from inside a nested one.
4. The dated "Started:" banner and the Timing summary (counts per kind).
5. Through problems.incremental: solve_automatic shows a cutback at the
   default level, one "solve" child per attempt in the increment's
   breakdown (solve=2x...), and each attempt's own start/end lines only at
   level "solve".
6. An unknown level is rejected rather than silently printing nothing.

Usage
-----
    python -m pytest test/test_utils_monitor.py
"""

import io
import re
import sys
sys.path.insert(0, "src")

from typing import NamedTuple

from problems.incremental import solve_automatic, solve_path
from utils.monitor import Monitor, get_monitor

STAMP = r"\d\d:\d\d:\d\d"


class _Sol(NamedTuple):
    converged: bool


def _run(level, body):
    out = io.StringIO()
    with Monitor(level=level, stream=out):
        body()
    return out.getvalue().splitlines()


# ── [1] silent without an active monitor ────────────────────────────────────
def _nested():
    mon = get_monitor()
    with mon.stage("increment", 1):
        with mon.stage("solve", 1, start="t=1"):
            with mon.stage("newton", 1) as it:
                it.note("res=1e-3")
        with mon.stage("write"):
            pass
        mon.note("gamma= 0.0100", kind="increment")
        mon.detail("           sigma_bar = [...]", kind="increment")


buf = io.StringIO()
old_stdout, sys.stdout = sys.stdout, buf
try:
    _nested()
    solve_path(lambda t: _Sol(True), [0.5, 1.0])
finally:
    sys.stdout = old_stdout
assert buf.getvalue() == "", f"no active monitor must print nothing, got {buf.getvalue()!r}"
with Monitor(level="newton", stream=io.StringIO()) as m:
    assert get_monitor() is m
assert get_monitor() is not m, "get_monitor() must fall back to silent after the run exits"
print("[1] PASSED")

# ── [2] level filtering, breakdown always ───────────────────────────────────
lines = _run("increment", _nested)
body = [l for l in lines if not l.startswith(("Started:", "Timing :"))]
assert len(body) == 2, f"default level: one step line + its detail, got {body}"
step = body[0]
assert re.search(r"solve=\d+\.\d\ds  write=\d+\.\d\ds  total=\d+\.\d\ds$", step), step
assert not any("newton" in l for l in body)

lines = _run("newton", _nested)
kinds = [l.split()[0] for l in lines if not l.startswith(("Started:", "Timing :"))]
assert kinds == ["solve", "newton", "solve", "step", "sigma_bar"], kinds
print("[2] PASSED")

# ── [3] line layout ─────────────────────────────────────────────────────────
assert re.fullmatch(rf"  step   1  {STAMP}  gamma= 0\.0100  solve=\S+  write=\S+  total=\S+", step), step
assert body[1] == "           sigma_bar = [...]"
newton = next(l for l in _run("newton", _nested) if l.lstrip().startswith("newton"))
assert re.fullmatch(rf"      newton   1  {STAMP}  res=1e-3  \(\d+\.\d\ds\)", newton), newton
start = next(l for l in _run("solve", _nested) if l.endswith("..."))
assert re.fullmatch(rf"    solve   1  {STAMP}  t=1  \.\.\.", start), start
print("[3] PASSED")

# ── [4] banner and summary ──────────────────────────────────────────────────
lines = _run("increment", _nested)
assert re.match(r"Started: \d{4}-\d\d-\d\d \d\d:\d\d:\d\d", lines[0]), lines[0]
assert lines[-1].startswith("Timing : "), lines[-1]
for part in ("1 increment", "1 solve", "1 newton", "1 write", "wall"):
    assert part in lines[-1], (part, lines[-1])
print("[4] PASSED")

# ── [5] cutbacks through solve_automatic ────────────────────────────────────
def _cutback_run():
    calls = []

    def solve_fn(t):
        calls.append(t)
        return _Sol(converged=len(calls) != 1)      # first attempt fails
    solve_automatic(solve_fn, dt_init=0.5, dt_min=0.1, dt_max=0.5, factor_inc=1.0)

lines = _run("increment", _cutback_run)
assert any(re.fullmatch(rf"    {STAMP}  cutback: t=0\.50000 not converged, retrying dt 0\.50000 -> 0\.25000", l)
           for l in lines), lines
steps = [l for l in lines if l.lstrip().startswith("step")]
assert len(steps) == 4 and "solve=2x" in steps[0], steps
assert not any(l.lstrip().startswith("solve") for l in lines)
attempts = [l for l in _run("solve", _cutback_run) if l.lstrip().startswith("solve")]
assert len(attempts) == 2 * 5, f"start + end line for each of 5 attempts, got {attempts}"
assert "converged=False" in attempts[1] and "converged=True" in attempts[3]
print("[5] PASSED")

# ── [6] unknown level rejected ──────────────────────────────────────────────
try:
    Monitor(level="verbose")
except ValueError:
    pass
else:
    raise AssertionError("expected a ValueError for an unknown level")
print("[6] PASSED")

print("\ntest_utils_monitor: all checks passed")
