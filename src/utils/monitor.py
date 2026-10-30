"""
Progress and timing reports for every solve, from one place.

The solve layers -- problems.incremental's increments and cutback attempts,
problems.mechanics's Newton iterations, problems.fracture's staggered
sweeps, the field writes -- each open a timed *stage* here instead of
printing and timing on their own. A driver script activates one Monitor for
the run; how much of that nesting is printed is one setting, the deepest
level shown (``LEVELS``)::

    with Monitor(level="increment"):          # prints the dated "Started:" banner
        solve_mechanics(...)                  # library code: get_monitor().stage(...)
                                              # on exit: one timing summary line

Library code never takes a monitor argument -- it calls ``get_monitor()``,
which returns the active Monitor or, with none active (tests, notebooks, a
vmapped solve), a silent one. So nothing is threaded through solver
signatures, and a new solver gets the same reporting by opening its own
stage.

What a stage prints
-------------------
One line when it closes, if its level is shown::

    {indent}{label}  {HH:MM:SS}  {notes}  {timing}

``notes`` is whatever the code inside attached with ``note()`` -- the driver
owns its physics fields (gamma, tau_avg, d_max, ...), this module only owns
the layout and the clock. ``timing`` is the stage's own elapsed time; when
it had child stages it is broken down by child kind instead, e.g. an
increment's ``solve=6.06s  write=0.31s  total=6.37s`` (``solve=3x20.1s`` for
three cutback attempts). Child stages are reported in that breakdown even
when their own lines are not shown, which is what keeps the default output
one line per increment while still splitting solve from write cost.
``detail()`` lines are printed under the closing line (e.g. the homogenized
stress); ``start=`` makes a stage also print an opening line, for stages
long enough that "is it still working?" matters (a cutback attempt on a
large grid can run tens of minutes).

Every stamp is local HH:MM:SS against the one dated ``Started:`` banner, so
a run crossing midnight stays unambiguous and a log can be lined up against
nvidia-smi or the scheduler. Timings are host wall clock: they are only
meaningful once the stage's result has been forced to the host (JAX
dispatches asynchronously). The instrumented loops already do that at the
points they open stages (a ``float()`` of a residual, a ``bool()`` of a
converged flag), so no ``block_until_ready`` is added here.
"""

from __future__ import annotations

import contextvars
import sys
import time
from contextlib import contextmanager
from typing import IO, Iterator

# Shallowest to deepest. A Monitor at level L prints every stage whose kind is
# at or above L in this list; kinds not listed here ("write", ...) never get a
# line of their own and only appear in their parent's timing breakdown.
LEVELS = ("run", "increment", "solve", "sweep", "newton")
DEFAULT_LEVEL = "increment"

# Display label per kind, where the kind name is not what the output has
# always called it.
_LABELS = {"increment": "step"}


class Stage:
    """One open (or closed) timed stage -- what ``Monitor.stage`` yields."""

    __slots__ = ("kind", "index", "depth", "t0", "elapsed", "notes", "details", "children")

    def __init__(self, kind: str, index: int | None, depth: int):
        self.kind     = kind
        self.index    = index
        self.depth    = depth
        self.t0       = time.perf_counter()
        self.elapsed  = 0.0                       # set on close
        self.notes:    list[str] = []
        self.details:  list[str] = []
        # child kind -> [count, total seconds], in first-seen order
        self.children: dict[str, list] = {}

    def note(self, text: str) -> None:
        """Append a field to this stage's closing line."""
        self.notes.append(text)

    def detail(self, text: str) -> None:
        """Add a line printed under this stage's closing line."""
        self.details.append(text)


class Monitor:
    """
    Hierarchical, timed progress output. Use as a context manager to make it
    the active monitor (``get_monitor()``) for the duration of a run.

    Parameters
    ----------
    level  : deepest stage kind printed, one of ``LEVELS``. None prints
             nothing (stages are still timed, so ``Stage.elapsed`` stays
             usable).
    stream : where lines go, default sys.stdout (resolved at write time, so
             pytest's capture and redirections see it).
    """

    def __init__(self, level: str | None = DEFAULT_LEVEL, stream: IO[str] | None = None):
        if level is not None and level not in LEVELS:
            raise ValueError(f"log level must be one of {LEVELS} or None, got {level!r}")
        self.level   = level
        self._depth  = -1 if level is None else LEVELS.index(level)
        self._stream = stream
        self._stack: list[Stage] = []
        # kind -> [count, total seconds] over the whole run, for summary()
        self.totals: dict[str, list] = {}
        self._t_run  = time.perf_counter()
        self._token: contextvars.Token | None = None

    # ── activation ────────────────────────────────────────────────────────────
    def __enter__(self) -> "Monitor":
        self._token = _ACTIVE.set(self)
        self._t_run = time.perf_counter()
        if self.enabled("run"):
            self._write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        return self

    def __exit__(self, *exc) -> None:
        if self._token is not None:
            _ACTIVE.reset(self._token)
            self._token = None
        if self.enabled("run"):
            self._write(self.summary())

    # ── stages ────────────────────────────────────────────────────────────────
    def enabled(self, kind: str) -> bool:
        """Whether a stage of this kind gets a line of its own."""
        return kind in LEVELS and LEVELS.index(kind) <= self._depth

    @contextmanager
    def stage(self, kind: str, index: int | None = None, *,
              start: str | None = None) -> Iterator[Stage]:
        """
        Time the enclosed block as one stage of ``kind``, nested under
        whatever stage is open. ``index`` numbers it in its line (step,
        iteration, attempt). ``start``, if given, also prints an opening
        line with that text when the kind is shown.
        """
        st = Stage(kind, index, depth=len(self._stack))
        if start is not None and self.enabled(kind):
            self._write(self._line(st, [start, "..."]))
        self._stack.append(st)
        try:
            yield st
        finally:
            st.elapsed = time.perf_counter() - st.t0
            self._stack.pop()
            _add(self.totals, kind, st.elapsed)
            if self._stack:
                _add(self._stack[-1].children, kind, st.elapsed)
            if self.enabled(kind):
                self._write(self._line(st, st.notes + [_timing(st)]))
                for d in st.details:
                    self._write(d)

    def note(self, text: str, kind: str | None = None) -> None:
        """
        Attach a field to the innermost open stage, or to the innermost one
        of ``kind`` -- e.g. a driver's on_increment callback noting its
        physics fields onto the increment it runs inside. No-op with no
        matching stage open.
        """
        st = self._find(kind)
        if st is not None:
            st.note(text)

    def detail(self, text: str, kind: str | None = None) -> None:
        """As ``note``, for a whole line printed under the stage's closing line."""
        st = self._find(kind)
        if st is not None:
            st.detail(text)

    def log(self, kind: str, text: str) -> None:
        """One stamped line at ``kind``'s level and the current nesting,
        e.g. a cutback. Printed only if ``kind`` is shown."""
        if self.enabled(kind):
            indent = "  " * (len(self._stack) + 1)
            self._write(f"{indent}{time.strftime('%H:%M:%S')}  {text}")

    def summary(self) -> str:
        """``Timing : <count> <kind> <total> (<mean> avg) | ... | wall <s>``."""
        # Outermost first (LEVELS order), then unlisted kinds (write, ...).
        order = sorted(self.totals, key=lambda k: LEVELS.index(k) if k in LEVELS else len(LEVELS))
        parts = [f"{n} {kind} {_fmt_s(total)}" + (f" ({_fmt_s(total / n)} avg)" if n > 1 else "")
                 for kind in order for n, total in [self.totals[kind]]]
        parts.append(f"wall {_fmt_s(time.perf_counter() - self._t_run)}")
        return "Timing : " + " | ".join(parts)

    # ── internals ─────────────────────────────────────────────────────────────
    def _find(self, kind: str | None) -> Stage | None:
        for st in reversed(self._stack):
            if kind is None or st.kind == kind:
                return st
        return None

    def _line(self, st: Stage, fields: list[str]) -> str:
        label = _LABELS.get(st.kind, st.kind)
        if st.index is not None:
            label = f"{label} {st.index:3d}"
        indent = "  " * (st.depth + 1)
        return "  ".join([f"{indent}{label}", time.strftime("%H:%M:%S"), *fields])

    def _write(self, line: str) -> None:
        stream = self._stream if self._stream is not None else sys.stdout
        print(line, file=stream, flush=True)


def _add(table: dict[str, list], kind: str, seconds: float) -> None:
    entry = table.setdefault(kind, [0, 0.0])
    entry[0] += 1
    entry[1] += seconds


def _fmt_s(seconds: float) -> str:
    """Seconds below 10 min, then h:mm:ss -- a 6 h run reads as 6:12:40, not 22360.00s."""
    if seconds < 600:
        return f"{seconds:.2f}s"
    m, s = divmod(int(round(seconds)), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}"


def _timing(st: Stage) -> str:
    if not st.children:
        return f"({_fmt_s(st.elapsed)})"
    parts = [f"{kind}={_fmt_s(total)}" if n == 1 else f"{kind}={n}x{_fmt_s(total)}"
             for kind, (n, total) in st.children.items()]
    return "  ".join(parts + [f"total={_fmt_s(st.elapsed)}"])


_ACTIVE: contextvars.ContextVar[Monitor] = contextvars.ContextVar("fftjax_monitor")
_SILENT = Monitor(level=None)


def get_monitor() -> Monitor:
    """The active Monitor, or a silent one if no run activated any."""
    return _ACTIVE.get(_SILENT)


def add_log_level_argument(parser) -> None:
    """``--log-level`` for a driver's argparse parser -- overrides the config's ``log_level``."""
    parser.add_argument(
        "--log-level", choices=LEVELS, default=None,
        help=f"deepest solver stage to report (default: the config's log_level, "
             f"else {DEFAULT_LEVEL!r}): " + " < ".join(LEVELS),
    )


def monitor_from(cfg: dict, cli_level: str | None = None) -> Monitor:
    """A driver's Monitor: ``--log-level`` if given, else top-level ``log_level`` in the config."""
    return Monitor(level=cli_level or cfg.get("log_level", DEFAULT_LEVEL))
