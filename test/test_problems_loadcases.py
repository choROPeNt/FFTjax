"""
Standalone test for problems/loadcases.py -- the config -> load-case
resolution that lets one config (and one process) run many load cases on a
single microstructure.

Pure config/bookkeeping logic, so no solve happens here and the test costs
milliseconds: what it guards is that a six-case sweep means what its config
says it means, which is otherwise only discoverable hours into a run.

Seven checks
------------
1. base6 is the six canonical cases, in post.fields.to_voigt's Voigt order
   (11, 22, 33, 12, 13, 23) -- the ordering is load-bearing, since case k is
   meant to line up with row/column k of any effective 6x6 assembled later.
2. free_surfaces derives a DIFFERENT control mask per case (loaded component
   strain-driven, unloaded surfaces traction-free) -- the whole reason a
   sweep cannot just reuse one fixed `control` mask.
3. No `cases` key resolves to exactly one case built from the loading block
   itself -- the backwards-compatibility guarantee for every pre-existing
   single-case config.
4. Arbitrary cases: an explicit `component`, per-case cycle overrides, and an
   explicit `gammas` path (0.0 prepended, since step 0 must be the unloaded
   state the driver skips the solve for).
5. BC precedence: case control > free_surfaces > run-level control.
6. Driving a component that the case's own control marks stress-controlled
   is rejected at resolve time -- such a case applies no load at all (eps_bar
   is ignored on stress-controlled directions), and left to run it fails
   later inside Newton for a completely unrelated-looking reason.
7. eps_bar is symmetrized, and gamma is the ENGINEERING measure for shear and
   normal cases alike -- eps_bar[i,i] = gamma but eps_bar[i,j] = gamma/2 -- so
   one gamma_max is one load magnitude across a sweep of both, rather than 5%
   axial in one case and 2.5% in the next.

Usage
-----
    python -m pytest test/test_problems_loadcases.py
"""

import sys
sys.path.insert(0, "src")

import numpy as np

from post.fields import _VOIGT_IJ
from problems.loadcases import (
    BASE_COMPONENTS, BASE_SIX, cycle_gammas, free_surface_control,
    resolve_cases, select_cases, voigt_label,
)

# ── [1] base6 == the six Voigt components, in to_voigt's order ──────────────
cases = resolve_cases({"cases": "base6", "gamma_max": 0.02, "n_load": 2,
                       "n_unload": 0, "n_reload": 0})
assert [c.name for c in cases] == list(BASE_SIX)
assert [(c.i, c.j) for c in cases] == list(_VOIGT_IJ), (
    f"base6 order {[(c.i, c.j) for c in cases]} != to_voigt order {list(_VOIGT_IJ)}"
)
assert all(c.control is None for c in cases), "base6 must default to pure strain BC"
assert [c.name for c in resolve_cases({"cases": ["xx", "yz"]})] == ["uniaxial_x", "shear_yz"], \
    "Voigt-label aliases must resolve to the same cases as their canonical names"
print(f"[1] base6 = {[c.name for c in cases]}  components {[(c.i, c.j) for c in cases]}")
print("[1] PASSED")

# ── [2] free_surfaces derives a per-case mask ───────────────────────────────
free = resolve_cases({"cases": "base6", "free_surfaces": True})
by_name = {c.name: c for c in free}
assert by_name["uniaxial_x"].control == ((0, 0, 0), (0, 1, 0), (0, 0, 1))
assert by_name["uniaxial_y"].control == ((1, 0, 0), (0, 0, 0), (0, 0, 1))
assert by_name["uniaxial_z"].control == ((1, 0, 0), (0, 1, 0), (0, 0, 0))
for shear in ("shear_xy", "shear_xz", "shear_yz"):
    assert by_name[shear].control == ((1, 0, 0), (0, 1, 0), (0, 0, 1)), \
        f"{shear}: every normal direction should be traction-free under shear"
assert len({c.control for c in free}) == 4, \
    "the six cases must not all share one control mask (3 uniaxial + 1 shear mask)"
for c in free:
    assert c.stress_bar is not None and not np.any(c.stress_bar), \
        f"{c.name}: free surfaces default to a zero (traction-free) stress target"
    assert not c.control[c.i][c.j], f"{c.name}: the driven component must stay strain-controlled"
print(f"[2] uniaxial_x control={by_name['uniaxial_x'].control}  "
      f"shear_xy control={by_name['shear_xy'].control}")
print("[2] PASSED")

# ── [3] no `cases` key -> one case, straight from the loading block ─────────
single = resolve_cases({"component": [0, 1], "gamma_max": 0.03,
                        "n_load": 10, "n_unload": 15, "n_reload": 15})
assert len(single) == 1
assert (single[0].i, single[0].j) == (0, 1) and single[0].name == "xy"
assert single[0].control is None, "no control in the config means pure strain BC"
assert np.allclose(single[0].gammas, cycle_gammas(0.03, 10, 15, 15))
assert len(single[0].gammas) == 1 + 10 + 15 + 15
assert single[0].gammas[0] == 0.0 and np.isclose(single[0].gammas[10], 0.03)
# the default component, with no `component` key at all, is still the shear case
assert (resolve_cases({})[0].i, resolve_cases({})[0].j) == (0, 1)
print(f"[3] single case {single[0].name}: {len(single[0].gammas)} steps, "
      f"gamma_max={single[0].gamma_max}")
print("[3] PASSED")

# ── [4] arbitrary cases ─────────────────────────────────────────────────────
arb = resolve_cases({
    "gamma_max": 0.02, "n_load": 2, "n_unload": 0, "n_reload": 0,
    "cases": [
        "uniaxial_x",
        {"name": "monotonic_x", "component": [0, 0], "gammas": [0.005, 0.01, 0.04]},
        {"name": "deep_shear", "component": [0, 1], "gamma_max": 0.1, "n_load": 4},
        {"component": [1, 2]},                      # unnamed -> named after its label
    ],
})
assert [c.name for c in arb] == ["uniaxial_x", "monotonic_x", "deep_shear", "yz"]
assert np.allclose(arb[0].gammas, [0.0, 0.01, 0.02]), "inherits the loading-level cycle"
assert np.allclose(arb[1].gammas, [0.0, 0.005, 0.01, 0.04]), "explicit path, 0.0 prepended"
assert np.allclose(arb[2].gammas, cycle_gammas(0.1, 4, 0, 0)), "per-case overrides win"
assert arb[3].name == voigt_label(1, 2)
print(f"[4] {[(c.name, len(c.gammas)) for c in arb]}")
print("[4] PASSED")

# ── [5] BC precedence: case > free_surfaces > run-level ─────────────────────
run_level = ((0, 0, 0), (0, 1, 0), (0, 0, 1))
own = ((0, 0, 0), (0, 0, 0), (0, 0, 1))
prec = resolve_cases(
    {"cases": [
        "uniaxial_x",                                          # -> run-level control
        {"name": "derived", "component": [0, 0], "free_surfaces": True},
        {"name": "own", "component": [0, 0], "control": own,
         "stress_bar": [[0, 0, 0], [0, -5.0, 0], [0, 0, 0]]},
    ]},
    control=run_level, stress_bar=np.zeros((3, 3)),
)
assert prec[0].control == run_level
assert prec[1].control == free_surface_control(0, 0)
assert prec[2].control == own and prec[2].stress_bar[1][1] == -5.0
# a run-level control still applies when free_surfaces is off for that case
assert resolve_cases({"cases": ["uniaxial_x"]}, control=run_level)[0].control == run_level
# free_surfaces derives the MASK; a run-level stress_bar is still the target on it
confined = resolve_cases({"cases": ["uniaxial_x"], "free_surfaces": True},
                         stress_bar=[[0, 0, 0], [0, -2.0, 0], [0, 0, -2.0]])[0]
assert confined.control == free_surface_control(0, 0) and confined.stress_bar[2][2] == -2.0
print(f"[5] run-level={prec[0].control}  derived={prec[1].control}  own={prec[2].control}")
print("[5] PASSED")

# ── [6] driving a stress-controlled component is a config error ─────────────
for bad in (
    {"cases": [{"name": "bad", "component": [1, 1], "control": free_surface_control(0, 0)}]},
    {"component": [1, 1], "cases": None},   # run-level control, same conflict
):
    try:
        resolve_cases(bad, control=free_surface_control(0, 0))
    except ValueError as exc:
        assert "stress-controlled" in str(exc)
    else:
        raise AssertionError(f"expected a ValueError for {bad}")
# and the same conflict can never arise from the derived masks
for name, (i, j) in BASE_COMPONENTS.items():
    assert not free_surface_control(i, j)[i][j]
print("[6] driving a stress-controlled component rejected at resolve time")
print("[6] PASSED")

# ── [7] eps_bar symmetry and the engineering-gamma convention ───────────────
shear_case, normal_case = resolve_cases({"cases": ["shear_xy", "uniaxial_x"]})
eps_s = shear_case.eps_bar(0.02)
assert np.allclose(eps_s, eps_s.T)
assert eps_s[0, 1] == eps_s[1, 0] == 0.01 and np.isclose(np.trace(eps_s), 0.0)
eps_n = normal_case.eps_bar(0.02)
# gamma is the ENGINEERING strain: the normal entry is gamma itself, the shear
# entry gamma/2 (engineering shear = 2 eps_ij), so both cases above are "2%".
assert eps_n[0, 0] == 0.02 and np.allclose(eps_n[1:, 1:], 0.0)
assert np.count_nonzero(eps_n) == 1
assert eps_s[0, 1] * 2 == eps_n[0, 0], "shear and normal at the same gamma are the same load"
assert np.allclose(normal_case.eps_bar(0.0), 0.0)
print(f"[7] shear eps_bar[0,1]={eps_s[0, 1]}  uniaxial eps_bar[0,0]={eps_n[0, 0]}")
print("[7] PASSED")

# ── select_cases: subset in the requested order, unknown names rejected ─────
six = resolve_cases({"cases": "base6"})
assert [c.name for c in select_cases(six, ["shear_yz", "uniaxial_x"])] == ["shear_yz", "uniaxial_x"]
assert [c.name for c in select_cases(six, None)] == list(BASE_SIX)
try:
    select_cases(six, ["nope"])
except ValueError as exc:
    assert "nope" in str(exc)
else:
    raise AssertionError("expected a ValueError for an unknown case name")
try:
    resolve_cases({"cases": ["xx", "uniaxial_x"]})
except ValueError as exc:
    assert "duplicate" in str(exc)
else:
    raise AssertionError("expected a ValueError for a duplicate case name (same output path)")

print("\ntest_problems_loadcases: all checks passed")
