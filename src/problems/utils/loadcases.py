"""
Macroscopic load cases (driven component, strain path, BC) for running many
cases on one microstructure from one config.

``BASE_SIX``: uniaxial xx, yy, zz and shear xy, xz, yz, in Voigt order, so
case k is column k of an effective 6x6. ``free_surfaces: false`` prescribes
the full strain (unit-strain cases); ``true`` drives one component and keeps
every other macroscopic stress at zero (specimen-like; for a
pressure-sensitive matrix the difference matters). Any case may also spell
out its own ``component``, ``control``/``stress_bar`` and path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

_AXIS = "xyz"

BASE_COMPONENTS: dict[str, tuple[int, int]] = {
    "uniaxial_x": (0, 0),
    "uniaxial_y": (1, 1),
    "uniaxial_z": (2, 2),
    "shear_xy":   (0, 1),
    "shear_xz":   (0, 2),
    "shear_yz":   (1, 2),
}
BASE_SIX: tuple[str, ...] = tuple(BASE_COMPONENTS)

_ALIASES: dict[str, str] = {
    _AXIS[i] + _AXIS[j]: name for name, (i, j) in BASE_COMPONENTS.items()
}

_SET_SHORTHANDS = {"base6", "base_six", "all"}


def voigt_label(i: int, j: int) -> str:
    """``(0, 2) -> 'xz'``."""
    return _AXIS[i] + _AXIS[j]


def strain_symbol(i: int, j: int) -> str:
    """``(0, 1) -> 'εxy'``."""
    return f"ε{voigt_label(i, j)}"


def stress_symbol(i: int, j: int) -> str:
    """``(0, 1) -> 'σxy'``."""
    return f"σ{voigt_label(i, j)}"


def free_surface_control(i: int, j: int) -> tuple[tuple[int, int, int], ...]:
    """
    ``control`` mask that strain-drives only (i, j)/(j, i); every other
    component, shears included, is stress-controlled (traction-free with a
    zero stress_bar).

    Free shears are well-posed: eps_bar is symmetric, so rotation is never an
    unknown. They matter wherever the RVE couples normal load into shear.
    """
    control = [[1, 1, 1] for _ in range(3)]
    control[i][j] = control[j][i] = 0
    return tuple(tuple(row) for row in control)


def cycle_gammas(gamma_max: float, n_load: int, n_unload: int, n_reload: int) -> np.ndarray:
    """Path 0 -> gamma_max -> -gamma_max -> gamma_max in the given step counts, starting at 0."""
    load   = np.linspace(0.0, gamma_max, n_load + 1)[1:]
    unload = np.linspace(gamma_max, -gamma_max, n_unload + 1)[1:]
    reload = np.linspace(-gamma_max, gamma_max, n_reload + 1)[1:]
    return np.concatenate([[0.0], load, unload, reload])


@dataclass(frozen=True)
class LoadCase:
    """
    One macroscopic load path; ``gammas`` starts at 0, ``control``/``stress_bar``
    as in problems.mechanics (None = pure strain).
    """
    name:       str
    i:          int
    j:          int
    gammas:     np.ndarray
    control:    tuple[tuple[int, int, int], ...] | None = None
    stress_bar: np.ndarray | None = None

    @property
    def label(self) -> str:
        """Label of the driven component, e.g. 'xy'."""
        return voigt_label(self.i, self.j)

    @property
    def gamma_max(self) -> float:
        return float(np.max(np.abs(self.gammas)))

    def eps_bar(self, gamma: float) -> np.ndarray:
        """
        Macroscopic strain at ``gamma``, engineering measure: eps_ii = gamma,
        or eps_ij = eps_ji = gamma/2, so one value means the same load size
        in every case.
        """
        eps = np.zeros((3, 3))
        if self.i == self.j:
            eps[self.i, self.i] = gamma
        else:
            eps[self.i, self.j] = gamma / 2
            eps[self.j, self.i] = gamma / 2
        return eps

    def describe(self, path: bool = True) -> str:
        """One-line summary; ``path=False`` omits the step count."""
        bc = "pure strain"
        if self.control is not None:
            free = ", ".join(
                f"sigma_{voigt_label(a, b)}="
                f"{0.0 if self.stress_bar is None else float(self.stress_bar[a][b]):g}"
                for a in range(3) for b in range(a, 3) if self.control[a][b]
            )
            bc = f"stress-controlled: {free}" if free else "pure strain (all-zero control)"
        steps = f"{len(self.gammas)} steps, " if path else ""
        return (f"{self.name:<12s} component [{self.i}, {self.j}] ({self.label})  "
                f"{steps}gamma_max={self.gamma_max:g}  {bc}")


def _component_of(name: str, spec: dict) -> tuple[int, int]:
    if spec.get("component") is not None:
        i, j = (int(c) for c in spec["component"])
        return i, j
    key = _ALIASES.get(name, name)
    if key not in BASE_COMPONENTS:
        raise ValueError(
            f"load case {name!r} is not one of the known base cases "
            f"({', '.join(BASE_SIX)}, or their xy-style labels) and gives no "
            f"`component: [i, j]` of its own -- add one to define it."
        )
    return BASE_COMPONENTS[key]


def _as_control(raw: Any) -> tuple[tuple[int, int, int], ...] | None:
    if raw is None:
        return None
    return tuple(tuple(int(c) for c in row) for row in raw)


def _as_stress_bar(raw: Any) -> np.ndarray | None:
    return None if raw is None else np.asarray(raw, dtype=float)


def resolve_cases(
    lcfg:       dict,
    control:    Any = None,
    stress_bar: Any = None,
) -> list[LoadCase]:
    """
    Load cases of a ``loading`` config block.

    Without ``cases`` the block itself is one case. ``cases`` is ``base6`` or
    a list of names (``shear_xy`` or ``xy``) and/or mappings; loading-level
    cycle parameters, ``control`` and ``stress_bar`` are defaults that a case
    overrides, and ``gammas: [...]`` gives an explicit path. ``control`` and
    ``stress_bar`` here are the run-level BC, used when a case has neither
    its own nor ``free_surfaces``.

    Raises ValueError if a case marks its own driven component
    stress-controlled: it would apply no load and fail to converge confusingly.
    """
    raw_cases = lcfg.get("cases")
    if raw_cases is None:
        specs: list[Any] = [{"component": list(lcfg.get("component", [0, 1]))}]
    elif isinstance(raw_cases, str):
        if raw_cases not in _SET_SHORTHANDS:
            raise ValueError(
                f"loading.cases: {raw_cases!r} is not a known shorthand "
                f"({', '.join(sorted(_SET_SHORTHANDS))}) -- give a list of case "
                f"names or mappings instead."
            )
        specs = list(BASE_SIX)
    elif isinstance(raw_cases, Sequence):
        specs = list(raw_cases)
    else:
        raise ValueError(f"loading.cases must be a string or a list, got {type(raw_cases).__name__}")

    cases: list[LoadCase] = []
    seen: set[str] = set()
    for spec in specs:
        if isinstance(spec, str):
            # Canonical names, so `[xx, uniaxial_x]` is caught as a duplicate.
            name, body = _ALIASES.get(spec, spec), {}
        elif isinstance(spec, dict):
            body = dict(spec)
            i_j = body.get("component")
            name = str(body.pop("name", None) or (
                voigt_label(int(i_j[0]), int(i_j[1])) if i_j is not None else ""))
            if not name:
                raise ValueError(
                    "a load case mapping needs either a `name` or a `component: [i, j]` "
                    "to be named after -- neither given."
                )
        else:
            raise ValueError(f"a load case must be a name or a mapping, got {spec!r}")

        if name in seen:
            raise ValueError(f"duplicate load case name {name!r} -- names become output filenames.")
        seen.add(name)

        i, j = _component_of(name, body)

        # Precedence: own control, free_surfaces, run-level control, pure strain.
        free_surfaces = bool(body.get("free_surfaces", lcfg.get("free_surfaces", False)))
        if body.get("control") is not None:
            case_control = _as_control(body["control"])
        elif free_surfaces:
            case_control = free_surface_control(i, j)
        else:
            case_control = _as_control(control)

        # free_surfaces sets only the mask; the target stays configurable
        # (e.g. a confining pressure) and defaults to zero.
        case_stress = _as_stress_bar(body.get("stress_bar", None))
        if case_stress is None:
            case_stress = _as_stress_bar(stress_bar)
        if case_control is not None and case_stress is None:
            case_stress = np.zeros((3, 3))

        if case_control is not None and case_control[i][j]:
            raise ValueError(
                f"load case {name!r} drives component [{i}, {j}] but marks it "
                f"stress-controlled in `control` -- eps_bar is ignored on "
                f"stress-controlled directions, so this case applies no load. Set "
                f"control[{i}][{j}] = 0, or drive a different component."
            )

        if body.get("gammas") is not None:
            gammas = np.asarray(body["gammas"], dtype=float).ravel()
            if gammas.size == 0:
                raise ValueError(f"load case {name!r}: `gammas` is empty.")
            if gammas[0] != 0.0:
                gammas = np.concatenate([[0.0], gammas])
        else:
            gammas = cycle_gammas(
                float(body.get("gamma_max", lcfg.get("gamma_max", 0.03))),
                int(body.get("n_load",   lcfg.get("n_load",   10))),
                int(body.get("n_unload", lcfg.get("n_unload", 15))),
                int(body.get("n_reload", lcfg.get("n_reload", 15))),
            )

        cases.append(LoadCase(name=name, i=i, j=j, gammas=gammas,
                              control=case_control, stress_bar=case_stress))

    return cases


def select_cases(cases: list[LoadCase], names: Sequence[str] | None) -> list[LoadCase]:
    """Cases named in ``names``, in that order (None keeps all)."""
    if not names:
        return cases
    by_name = {c.name: c for c in cases}
    missing = [nm for nm in names if nm not in by_name]
    if missing:
        raise ValueError(
            f"unknown load case(s) {', '.join(missing)} -- this config defines "
            f"{', '.join(by_name)}."
        )
    return [by_name[nm] for nm in names]
