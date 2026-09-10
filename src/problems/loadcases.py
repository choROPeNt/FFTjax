"""
Macroscopic load cases for a homogenization run -- the (driven component,
strain path, macroscopic BC) triple that is the ONLY thing distinguishing
one solve from the next when microstructure and materials are held fixed.

Split out of scripts/simulation/solve_inelastic.py so a driver can run many
load cases on one microstructure from one config and one process: the
geometry read, the material assembly and the frequency grid are per-RVE,
not per-case, and re-paying them once per case (as a shell loop over six
near-identical YAML files does) is pure overhead -- as is maintaining six
copies of everything the cases have in common.

The six base cases
------------------
``BASE_SIX`` is the canonical set: three uniaxial (xx, yy, zz) and three
shear (xy, xz, yz), ordered exactly like post.fields.to_voigt's Voigt index
order (11, 22, 33, 12, 13, 23, Abaqus convention), so case k of a base-six
sweep is column/row k of any effective 6x6 assembled from the results.

What "base" means for the BC is a choice the config has to make, because
both readings are legitimate and they answer different questions:

  free_surfaces: false (default)  -- pure macroscopic strain BC, i.e. the
      classic unit-strain cases. eps_bar is prescribed in full, every other
      entry clamped to zero. Reading sigma_bar off case k gives column k of
      the effective stiffness directly.
  free_surfaces: true             -- the loaded component is strain-driven
      and every surface that is not loaded is traction-free (see
      free_surface_control). This is the *specimen-like* case: uniaxial
      tension is free to contract laterally. For a pressure-sensitive
      matrix (drucker_prager) the difference is not cosmetic -- clamping
      the transverse directions confines the matrix and inflates the
      tension/compression yield asymmetry.

Arbitrary load cases
--------------------
A case is not restricted to the registry: any entry may spell out its own
``component``, ``control``/``stress_bar`` and either the load/unload/reload
cycle parameters or an explicit ``gammas`` path. The registry only supplies
defaults for the six names that recur often enough to deserve one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

_AXIS = "xyz"

# Ordered exactly like post.fields._VOIGT_IJ -- see the module docstring.
BASE_COMPONENTS: dict[str, tuple[int, int]] = {
    "uniaxial_x": (0, 0),
    "uniaxial_y": (1, 1),
    "uniaxial_z": (2, 2),
    "shear_xy":   (0, 1),
    "shear_xz":   (0, 2),
    "shear_yz":   (1, 2),
}
BASE_SIX: tuple[str, ...] = tuple(BASE_COMPONENTS)

# Voigt-label aliases ("xx", "xy", ...), so a config can name a case either way.
_ALIASES: dict[str, str] = {
    _AXIS[i] + _AXIS[j]: name for name, (i, j) in BASE_COMPONENTS.items()
}

# Shorthands for the whole set, usable as `cases: base6`.
_SET_SHORTHANDS = {"base6", "base_six", "all"}


def voigt_label(i: int, j: int) -> str:
    """(i, j) tensor index -> its xyz label, e.g. (0, 2) -> 'xz'."""
    return _AXIS[i] + _AXIS[j]


def free_surface_control(i: int, j: int) -> tuple[tuple[int, int, int], ...]:
    """
    Mixed-BC ``control`` mask for a "load this component, free every other
    surface" case -- 1 marks a stress-controlled (traction-free, given a
    zero stress_bar) direction, 0 a strain-controlled one.

    - Uniaxial (i == j): strain-controlled on the loaded diagonal entry,
      stress-free on the other two normal directions, so the RVE contracts
      laterally under axial tension instead of being confined. Shear stays
      strain-controlled at zero -- a free shear direction would leave the
      macroscopic rotation/shear undetermined rather than model anything.
    - Shear (i != j): strain-controlled on the loaded shear pair, stress-free
      on all three normal directions (every lateral surface free to expand or
      contract, e.g. against plastic dilatancy); the other, non-loaded shear
      pair stays strain-controlled at zero.

    Same rule as benchmark/benchmark_2/pff_random_periodic.py's local
    ``mixed_control`` -- that script predates this module and still carries
    its own copy against its own LoadCase type.
    """
    if i == j:
        diag = [1, 1, 1]
        diag[i] = 0
        return ((diag[0], 0, 0), (0, diag[1], 0), (0, 0, diag[2]))
    return ((1, 0, 0), (0, 1, 0), (0, 0, 1))


def cycle_gammas(gamma_max: float, n_load: int, n_unload: int, n_reload: int) -> np.ndarray:
    """
    The load / unload / reload path 0 -> gamma_max -> -gamma_max -> gamma_max
    in that many equal steps each, prefixed with the virgin gamma = 0 state.

    Every path this module produces starts at 0.0, including an explicit
    ``gammas`` list (0.0 is prepended if absent) -- callers rely on step 0
    being the unloaded state.
    """
    load   = np.linspace(0.0, gamma_max, n_load + 1)[1:]
    unload = np.linspace(gamma_max, -gamma_max, n_unload + 1)[1:]
    reload = np.linspace(-gamma_max, gamma_max, n_reload + 1)[1:]
    return np.concatenate([[0.0], load, unload, reload])


@dataclass(frozen=True)
class LoadCase:
    """
    One macroscopic load path on a fixed microstructure.

    ``gammas`` is the full applied path including the leading 0.0 virgin
    state; ``control``/``stress_bar`` are the mixed-BC pair in
    problems.mechanics's convention (None = pure strain BC).
    """
    name:       str
    i:          int
    j:          int
    gammas:     np.ndarray
    control:    tuple[tuple[int, int, int], ...] | None = None
    stress_bar: np.ndarray | None = None

    @property
    def label(self) -> str:
        """xyz label of the driven component, e.g. 'xy'."""
        return voigt_label(self.i, self.j)

    @property
    def gamma_max(self) -> float:
        return float(np.max(np.abs(self.gammas)))

    def eps_bar(self, gamma: float) -> np.ndarray:
        """
        Macroscopic strain at path value ``gamma``, with ``gamma`` the
        ENGINEERING measure in both cases:

          normal (i == j)  eps_bar[i,i] = gamma
          shear  (i != j)  eps_bar[i,j] = eps_bar[j,i] = gamma/2,
                           i.e. engineering shear gamma_ij = 2 eps_ij

        so "load to 5%" is one number that means the same size of load for
        every case in a sweep, instead of 5% axial strain in one case and
        2.5% in the next.

        Note this is a deliberate correction of the single-case convention
        scripts/simulation/solve_inelastic.py used before load cases existed,
        which halved the DIAGONAL entry too (eps_bar[i,i] = gamma/2). Nothing
        shipped drove a normal component through that path -- both example
        configs drive [0, 1] -- and the shear convention is unchanged, so
        only a hand-written `component: [i, i]` config is affected: it now
        reaches twice the strain it used to at the same gamma_max.
        """
        eps = np.zeros((3, 3))
        if self.i == self.j:
            eps[self.i, self.i] = gamma
        else:
            eps[self.i, self.j] = gamma / 2
            eps[self.j, self.i] = gamma / 2
        return eps

    def describe(self, path: bool = True) -> str:
        """
        One-line summary. ``path=False`` drops the step count, for a caller
        that takes only the peak gamma and owns the load path itself (see
        scripts/simulation/solve_mechanics.py).
        """
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
    Expand a ``loading`` config block into the list of load cases to run.

    Parameters
    ----------
    lcfg : the ``loading`` mapping. Without a ``cases`` key this yields
        exactly ONE case built from ``component``/``gamma_max``/``n_*`` --
        i.e. every pre-existing single-case config resolves unchanged.
        ``cases`` is either the shorthand ``base6`` or a list whose entries
        are registry names (``shear_xy``, or its ``xy`` alias) and/or
        mappings spelling a case out in full. Cycle parameters, ``control``
        and ``stress_bar`` given at ``loading`` level are the defaults each
        case inherits; anything in the case's own mapping overrides them,
        and ``gammas: [...]`` replaces the cycle with an explicit path.
    control, stress_bar : the run-level mixed BC (``inelastic.control`` /
        ``inelastic.stress_bar``), used for any case that neither carries its
        own nor asks for ``free_surfaces``.

    Raises
    ------
    ValueError : if a case drives a component that its own ``control`` marks
        stress-controlled -- eps_bar is ignored on stress-controlled
        directions, so such a case applies no load at all and every step is
        the trivial eps = 0 state, which then fails to converge for a
        confusing reason (the Newton criterion is relative to ||sigma||,
        which is ~0) rather than reporting the config error it actually is.
    """
    raw_cases = lcfg.get("cases")
    if raw_cases is None:
        # No `cases` key: the loading block IS the one case, named after the
        # component it drives -- every pre-existing config lands here.
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
            # A bare name is canonicalized ("xx" -> "uniaxial_x") so one
            # physical case has one identity and one output path however it was
            # spelled -- which is also what makes the duplicate check below
            # catch `cases: [xx, uniaxial_x]`. A mapping's explicit `name` is
            # the user's own label and is left exactly as written.
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

        # BC precedence, most specific first: the case's own control, else a
        # per-case/loading-level `free_surfaces` derivation, else the run-level
        # control (one fixed mask for every case), else pure strain.
        free_surfaces = bool(body.get("free_surfaces", lcfg.get("free_surfaces", False)))
        if body.get("control") is not None:
            case_control = _as_control(body["control"])
        elif free_surfaces:
            case_control = free_surface_control(i, j)
        else:
            case_control = _as_control(control)

        # free_surfaces derives the control MASK only -- the target on those
        # directions is still whatever the config says (a confining pressure
        # is a perfectly ordinary thing to want on an otherwise "free"
        # surface), defaulting to zero, i.e. traction-free.
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
    """
    Filter resolved cases down to ``names`` (order follows the request, so a
    SLURM array task or a rerun can pick one case out of a six-case config
    without editing it). ``None`` keeps all of them.
    """
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
