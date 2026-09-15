"""
Standalone test for utils/loadcases.py -- the config -> ONE load case
resolution shared by solve_mechanics.py and solve_inelastic.py.

Pure config/bookkeeping logic, so no solve happens here and the test costs
milliseconds.

Six checks
----------
1. component resolution: an [i, j] pair, and the xyz/base-case aliases
   (``uniaxial_x``, ``shear_xy``/``xy``), all agree with post.fields.to_voigt's
   index order.
2. free_surface_control derives a DIFFERENT mask for each of the four shapes
   (3 uniaxial + 1 shared shear mask) -- the loaded component stays
   strain-driven, every unloaded surface is traction-free.
3. Defaults: no `component` key resolves to the shear [0, 1] case; an
   explicit `gammas` path (0.0 prepended) and the default load/unload/reload
   cycle both work.
4. BC precedence: the block's own control > free_surfaces > run-level control.
5. Driving a component that `control` marks stress-controlled is rejected at
   resolve time -- such a config applies no load at all (eps_bar is ignored
   on stress-controlled directions), and left to run it fails later inside
   Newton for a completely unrelated-looking reason.
6. eps_bar is symmetrized, and gamma is the ENGINEERING measure for shear and
   normal components alike -- eps_bar[i,i] = gamma but eps_bar[i,j] = gamma/2
   -- so one gamma_max is one load magnitude for either.

Usage
-----
    python -m pytest test/test_utils_loadcases.py
"""

import sys
sys.path.insert(0, "src")

import numpy as np

from post.fields import _VOIGT_IJ
from utils.loadcases import (
    BASE_COMPONENTS, cycle_gammas, free_surface_control, resolve_case, voigt_label,
)

# ── [1] component resolution: [i, j] pairs and aliases ──────────────────────
for name, (i, j) in BASE_COMPONENTS.items():
    case = resolve_case({"component": name})
    assert (case.i, case.j) == (i, j) and case.name == name, name
assert (resolve_case({"component": "xy"}).i, resolve_case({"component": "xy"}).j) == (0, 1)
pair_case = resolve_case({"component": [1, 2]})
assert (pair_case.i, pair_case.j) == (1, 2) and pair_case.name == voigt_label(1, 2)
assert list(BASE_COMPONENTS.values()) == list(_VOIGT_IJ), (
    "BASE_COMPONENTS order must match to_voigt's order even though a single "
    "case no longer relies on it for a sweep"
)
print("[1] PASSED")

# ── [2] free_surface_control: 3 uniaxial masks + 1 shared shear mask ────────
assert free_surface_control(0, 0) == ((0, 0, 0), (0, 1, 0), (0, 0, 1))
assert free_surface_control(1, 1) == ((1, 0, 0), (0, 0, 0), (0, 0, 1))
assert free_surface_control(2, 2) == ((1, 0, 0), (0, 1, 0), (0, 0, 0))
for i, j in ((0, 1), (0, 2), (1, 2)):
    assert free_surface_control(i, j) == ((1, 0, 0), (0, 1, 0), (0, 0, 1)), (i, j)
    assert not free_surface_control(i, j)[i][j], "the driven component must stay strain-controlled"
print("[2] PASSED")

# ── [3] defaults: no component, explicit gammas, default cycle ──────────────
default = resolve_case({})
assert (default.i, default.j) == (0, 1) and default.control is None
cycled = resolve_case({"gamma_max": 0.03, "n_load": 10, "n_unload": 15, "n_reload": 15})
assert np.allclose(cycled.gammas, cycle_gammas(0.03, 10, 15, 15))
assert cycled.gammas[0] == 0.0 and np.isclose(cycled.gammas[10], 0.03)
explicit = resolve_case({"gammas": [0.005, 0.01, 0.04]})
assert np.allclose(explicit.gammas, [0.0, 0.005, 0.01, 0.04]), "0.0 prepended"
already_zero = resolve_case({"gammas": [0.0, 0.01]})
assert np.allclose(already_zero.gammas, [0.0, 0.01]), "not prepended twice"
print("[3] PASSED")

# ── [4] BC precedence: own control > free_surfaces > run-level ──────────────
run_level = ((0, 0, 0), (0, 1, 0), (0, 0, 1))
own = ((0, 0, 0), (0, 0, 0), (0, 0, 1))
assert resolve_case({"component": [0, 0]}, control=run_level).control == run_level
derived = resolve_case({"component": [0, 0], "free_surfaces": True}, control=run_level)
assert derived.control == free_surface_control(0, 0), "free_surfaces overrides run-level control"
explicit_control = resolve_case(
    {"component": [0, 0], "control": own, "stress_bar": [[0, 0, 0], [0, -5.0, 0], [0, 0, 0]]},
    control=run_level,
)
assert explicit_control.control == own and explicit_control.stress_bar[1][1] == -5.0
# free_surfaces derives the MASK; a run-level stress_bar is still the target on it
confined = resolve_case({"component": [0, 0], "free_surfaces": True},
                        stress_bar=[[0, 0, 0], [0, -2.0, 0], [0, 0, -2.0]])
assert confined.control == free_surface_control(0, 0) and confined.stress_bar[2][2] == -2.0
print("[4] PASSED")

# ── [5] driving a stress-controlled component is a config error ────────────
for bad_control in (free_surface_control(0, 0),):
    try:
        resolve_case({"component": [1, 1]}, control=bad_control)
    except ValueError as exc:
        assert "stress-controlled" in str(exc)
    else:
        raise AssertionError("expected a ValueError")
print("[5] PASSED")

# ── [6] eps_bar symmetry and the engineering-gamma convention ───────────────
shear_case = resolve_case({"component": "shear_xy"})
normal_case = resolve_case({"component": "uniaxial_x"})
eps_s = shear_case.eps_bar(0.02)
assert np.allclose(eps_s, eps_s.T)
assert eps_s[0, 1] == eps_s[1, 0] == 0.01 and np.isclose(np.trace(eps_s), 0.0)
eps_n = normal_case.eps_bar(0.02)
assert eps_n[0, 0] == 0.02 and np.count_nonzero(eps_n) == 1
assert eps_s[0, 1] * 2 == eps_n[0, 0], "shear and normal at the same gamma are the same load"
assert np.allclose(normal_case.eps_bar(0.0), 0.0)
print("[6] PASSED")

print("\ntest_utils_loadcases: all checks passed")
