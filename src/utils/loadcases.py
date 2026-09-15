"""
Resolve a ``loading`` config block into the ONE macroscopic load case
(driven component, strain path, macroscopic BC) it describes.

Which load case runs is a config-file choice -- one config, one case, same
convention as every other single-purpose config in configs/simulation/. This
module exists only to spare solve_mechanics.py and solve_inelastic.py two
copies of the same small pieces of bookkeeping:

  - naming a driven component by [i, j] or by an xyz/base-case alias
    (``uniaxial_x``, ``shear_xy``, ``xy``, ...)
  - deriving the mixed-BC ``control`` mask a ``free_surfaces: true`` case
    needs (see ``free_surface_control``)
  - the ENGINEERING-strain convention for ``eps_bar(gamma)`` -- gamma itself
    on a normal component, gamma/2 (symmetrized) on a shear one, so the same
    gamma_max means the same size of load either way
  - the load/unload/reload path builder (``cycle_gammas``)

Parameters
----------
No config-level fan-out lives here: a ``loading`` block always resolves to
exactly one ``LoadCase``. Six load directions on one microstructure means
six config files (or a shell loop over them), not a ``cases:`` list this
module expands -- see configs/simulation/mechanics_example.yaml and
configs/simulation/inelastic_j2_example.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

_AXIS = "xyz"

# Ordered exactly like post.fields._VOIGT_IJ (11, 22, 33, 12, 13, 23, Abaqus
# convention) -- not load-bearing for a single case, but kept so a
# `component: uniaxial_x`-style config lines up with that convention too.
BASE_COMPONENTS: dict[str, tuple[int, int]] = {
    "uniaxial_x": (0, 0),
    "uniaxial_y": (1, 1),
    "uniaxial_z": (2, 2),
    "shear_xy":   (0, 1),
    "shear_xz":   (0, 2),
    "shear_yz":   (1, 2),
}
BASE_SIX: tuple[str, ...] = tuple(BASE_COMPONENTS)

# Voigt-label aliases ("xx", "xy", ...), so a config can name a component either way.
_ALIASES: dict[str, str] = {
    _AXIS[i] + _AXIS[j]: name for name, (i, j) in BASE_COMPONENTS.items()
}


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
        a normal or a shear component alike.
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


def _component_of(comp: Any) -> tuple[int, int, str]:
    """``loading.component`` -> (i, j, name) -- either a [i, j] pair (named
    after its xyz label) or a base-case/xyz-style alias string."""
    if isinstance(comp, str):
        key = _ALIASES.get(comp, comp)
        if key not in BASE_COMPONENTS:
            raise ValueError(
                f"loading.component: {comp!r} is not one of the known base "
                f"components ({', '.join(BASE_SIX)}, or their xy-style labels) -- "
                f"give an [i, j] pair instead to name an arbitrary one."
            )
        i, j = BASE_COMPONENTS[key]
        return i, j, key
    i, j = (int(c) for c in comp)
    return i, j, voigt_label(i, j)


def _as_control(raw: Any) -> tuple[tuple[int, int, int], ...] | None:
    if raw is None:
        return None
    return tuple(tuple(int(c) for c in row) for row in raw)


def _as_stress_bar(raw: Any) -> np.ndarray | None:
    return None if raw is None else np.asarray(raw, dtype=float)


def resolve_case(lcfg: dict, control: Any = None, stress_bar: Any = None) -> LoadCase:
    """
    Build the one ``LoadCase`` a ``loading`` config block describes.

    Parameters
    ----------
    lcfg : the ``loading`` mapping -- ``component`` ([i, j] or an alias like
        ``uniaxial_x``/``xy``; default [0, 1]), ``gamma_max``/``n_load``/
        ``n_unload``/``n_reload`` (or an explicit ``gammas`` path, 0.0
        prepended if missing), ``free_surfaces``, and optionally its own
        ``control``/``stress_bar``.
    control, stress_bar : the run-level mixed BC (e.g. ``inelastic.control``/
        ``inelastic.stress_bar``), used when ``lcfg`` doesn't set its own and
        doesn't ask for ``free_surfaces``.

    Raises
    ------
    ValueError : if the driven component is marked stress-controlled by its
        own ``control`` -- eps_bar is ignored on stress-controlled
        directions, so that config applies no load at all and would instead
        fail later inside Newton for a completely unrelated-looking reason.
    """
    i, j, name = _component_of(lcfg.get("component", [0, 1]))

    # BC precedence, most specific first: the block's own control, else
    # free_surfaces derivation, else the run-level control, else pure strain.
    free_surfaces = bool(lcfg.get("free_surfaces", False))
    if lcfg.get("control") is not None:
        case_control = _as_control(lcfg["control"])
    elif free_surfaces:
        case_control = free_surface_control(i, j)
    else:
        case_control = _as_control(control)

    # free_surfaces derives the control MASK only -- the target on those
    # directions is still whatever the config says (a confining pressure is
    # a perfectly ordinary thing to want on an otherwise "free" surface),
    # defaulting to zero, i.e. traction-free.
    case_stress = _as_stress_bar(lcfg.get("stress_bar", None))
    if case_stress is None:
        case_stress = _as_stress_bar(stress_bar)
    if case_control is not None and case_stress is None:
        case_stress = np.zeros((3, 3))

    if case_control is not None and case_control[i][j]:
        raise ValueError(
            f"loading drives component [{i}, {j}] but marks it stress-controlled "
            f"in `control` -- eps_bar is ignored on stress-controlled directions, "
            f"so this applies no load. Set control[{i}][{j}] = 0, or drive a "
            f"different component."
        )

    if lcfg.get("gammas") is not None:
        gammas = np.asarray(lcfg["gammas"], dtype=float).ravel()
        if gammas.size == 0:
            raise ValueError("loading: `gammas` is empty.")
        if gammas[0] != 0.0:
            gammas = np.concatenate([[0.0], gammas])
    else:
        gammas = cycle_gammas(
            float(lcfg.get("gamma_max", 0.03)),
            int(lcfg.get("n_load", 10)),
            int(lcfg.get("n_unload", 15)),
            int(lcfg.get("n_reload", 15)),
        )

    return LoadCase(name=name, i=i, j=j, gammas=gammas, control=case_control, stress_bar=case_stress)
