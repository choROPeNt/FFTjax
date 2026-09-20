"""
Standalone test for solvers.damage.anderson -- the Anderson and NGMRES
accelerators for the staggered mechanics<->damage fixed point.

Seven checks
------------
1. Depth 1 IS Picard, bit-for-bit. The un-accelerated path has to be
   recoverable exactly, or "turn the accelerator off" is not a real option.
2. A fixed point is EXACT: fed x* with G(x*) = x*, both must return x* and
   keep returning it. The docs claim this specifically for NGMRES's
   sum-to-one constraint; it is asserted for both.
3. Speedup on a linear fixed point with known solution, and -- the property
   that matters here -- speedup that GROWS as the map stalls. A staggered
   loop is cheap when it is easy (2 sweeps) and ruinous when it is hard (44),
   so an accelerator that only helps the easy case is worthless.
4. Correctness of the accelerated answer: same fixed point, shorter path.
5. Clamping. The iterate is a DAMAGE field, and an extrapolated step is a
   linear combination with possibly negative weights, so it must be clamped
   into [0, 1] and above the irreversibility floor. Without the floor an
   accelerator can un-crack voxels.
6. The safeguard, in all three of its regimes: silent while the iteration is
   converging (a safeguard that fires when things work throws away the whole
   speedup -- an earlier PROSPECTIVE version did exactly that, rejecting 497
   of 499 steps), silent when residual growth is below its threshold (a
   diverging MAP is not acceleration's fault), and firing above it, where the
   fallback must be exactly the Picard iterate.
7. Through the REAL coupled solver: problems.fracture.solve_fracture with an
   accelerator must reach the same state as plain Picard in fewer staggered
   sweeps. Checked in the pre-severance regime, which is the one this
   package's results are verified in (see
   materialmodels/phasefield/drucker_prager.py on why past-severance is not).

Usage
-----
    python test/test_solvers_damage_anderson.py
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
sys.path.insert(0, "src")

import utils.precision  # noqa: F401
import jax.numpy as jnp
import numpy as np

from materialmodels.phasefield.drucker_prager import PhaseFieldDruckerPrager
from problems.fracture import solve_fracture
from solvers.damage.anderson import (AndersonAccelerator, NGMRESAccelerator,
                                     build_accelerator)

rng = np.random.default_rng(0)


def _linear_map(n=200, lo=0.1, hi=0.97, seed=0):
    """x -> Mx + c with a known fixed point and a controlled spectral radius."""
    r = np.random.default_rng(seed)
    Q = np.linalg.qr(r.normal(size=(n, n)))[0]
    M = Q @ np.diag(np.linspace(lo, hi, n)) @ Q.T
    c = r.normal(size=n) * 0.01
    x_star = np.linalg.solve(np.eye(n) - M, c)
    return (lambda x: M @ x + c), x_star


def _iterate(G, acc, n, tol=1e-10, maxit=5000):
    x = np.zeros(n)
    if acc is not None:
        acc.reset()
    for k in range(1, maxit + 1):
        g = G(x)
        if np.linalg.norm(g - x) < tol:
            return k, x
        x = g if acc is None else acc.step(x, g)
    return maxit, x


def _both(**kw):
    return (("Anderson", AndersonAccelerator(**kw)),
            ("NGMRES", NGMRESAccelerator(**{k: v for k, v in kw.items()
                                            if k != "beta"})))


# ── 1. depth 1 is Picard, bit-for-bit ───────────────────────────────────────
G1, _ = _linear_map()
worst1 = 0.0
for name, acc in _both(depth=1, bounds=None):
    acc.reset()
    xa = np.zeros(200)
    xp = np.zeros(200)
    for _ in range(40):
        xa = acc.step(xa, G1(xa))
        xp = G1(xp)
        worst1 = max(worst1, float(np.max(np.abs(xa - xp))))
print(f"[1] depth 1 vs plain Picard over 40 steps: max|diff| = {worst1:.3e}")
assert worst1 == 0.0, worst1
print("[1] PASSED")


# ── 2. a fixed point is exact ───────────────────────────────────────────────
G2, xs2 = _linear_map()
worst2 = 0.0
for name, acc in _both(depth=5, bounds=None):
    acc.reset()
    x = xs2.copy()
    for _ in range(8):
        x = acc.step(x, G2(x))
    err = float(np.linalg.norm(x - xs2)) / float(np.linalg.norm(xs2))
    worst2 = max(worst2, err)
    print(f"[2] {name:9s} 8 steps started AT the fixed point: rel drift = {err:.2e}")
assert worst2 < 1e-12, worst2
print("[2] PASSED")


# ── 3. speedup, and speedup that GROWS with difficulty ──────────────────────
print("[3] sweeps to 1e-10 (the metric that decides whether this is worth it):")
speedups = {}
for tag, hi in (("radius 0.97", 0.97), ("radius 0.995 (stalled)", 0.995)):
    G, _ = _linear_map(hi=hi)
    k_pic, _ = _iterate(G, None, 200)
    line = f"    {tag:24s} Picard={k_pic:5d}"
    for name, acc in _both(depth=5, bounds=None):
        k, _ = _iterate(G, acc, 200)
        speedups[(tag, name)] = k_pic / k
        line += f"   {name}={k:4d} ({k_pic / k:4.1f}x)"
    print(line)
for name in ("Anderson", "NGMRES"):
    easy = speedups[("radius 0.97", name)]
    hard = speedups[("radius 0.995 (stalled)", name)]
    assert easy > 2.0, f"{name}: only {easy:.1f}x on the easy map"
    assert hard > easy, (
        f"{name}: speedup does not grow with difficulty ({easy:.1f}x -> "
        f"{hard:.1f}x). An accelerator that helps only the cheap iterations "
        "is not worth its complexity here"
    )
print("[3] PASSED")


# ── 4. same fixed point, shorter path ───────────────────────────────────────
G4, xs4 = _linear_map(hi=0.99)
worst4 = 0.0
for name, acc in _both(depth=5, bounds=None):
    k, x = _iterate(G4, acc, 200)
    err = float(np.linalg.norm(x - xs4)) / float(np.linalg.norm(xs4))
    worst4 = max(worst4, err)
    print(f"[4] {name:9s} converged in {k:4d} sweeps, rel |x - x*| = {err:.2e}")
assert worst4 < 1e-8, worst4
print("[4] PASSED")


# ── 5. clamping: bounds and the irreversibility floor ───────────────────────
floor5 = np.array([0.4, 0.4, 0.4, 0.4, 0.4])
acc5 = AndersonAccelerator(depth=4, bounds=(0.0, 1.0))
acc5.reset(floor=floor5)
x5 = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
# a map whose extrapolation deliberately overshoots far outside [0, 1]
seq5 = [np.array([-2.0, 0.2, 0.9, 3.0, 0.45]),
        np.array([-5.0, 0.1, 0.95, 9.0, 0.50]),
        np.array([-9.0, 0.05, 0.99, 20.0, 0.55])]
ok5 = True
for g in seq5:
    x5 = acc5.step(x5, g)
    ok5 &= bool(np.all(x5 >= floor5 - 1e-12)) and bool(np.all((x5 >= 0.0) & (x5 <= 1.0)))
print(f"[5] extrapolation driven far outside [0,1]: every iterate in bounds "
      f"and >= floor = {ok5}")
assert ok5, "clamping failed -- an accelerated damage field left [0,1] or healed"
# and with the floor disabled it is allowed to go below (proving the floor acts)
acc5b = AndersonAccelerator(depth=4, bounds=(0.0, 1.0))
acc5b.reset(floor=None)
x5b = np.array([0.5] * 5)
for g in seq5:
    x5b = acc5b.step(x5b, g)
assert float(np.min(x5b)) < float(np.min(floor5)), (
    "with floor=None the iterate never went below the floor value, so check 5 "
    "does not actually demonstrate that the floor is what held it up"
)
print("[5] PASSED")


# ── 6. the safeguard: fires on residual GROWTH, silent while converging ─────
# The safeguard watches ||f_{k+1}|| / ||f_k|| and restarts from Picard when it
# exceeds safeguard_factor. Note what it is NOT: it does not police a
# divergent MAP. A map whose residual grows 1.4x per step diverges, but the
# acceleration is not what made it do that, and firing there would be noise --
# verified below. It fires when the growth exceeds the factor it was given.
#
# (a) silent while converging -- already implied by checks 3/4 reaching a
#     speedup, asserted here directly.
Gq, _ = _linear_map(hi=0.9)
acc6a = AndersonAccelerator(depth=5, bounds=None, safeguard_factor=2.0)
_iterate(Gq, acc6a, 200)
print(f"[6] converging map: {acc6a.n_rejected} rejections over {acc6a.n_steps} "
      f"steps ({acc6a.n_accelerated} accelerated)")
assert acc6a.n_rejected == 0, (
    f"{acc6a.n_rejected} spurious rejections on a converging map -- a safeguard "
    "that fires while things are working throws away the speedup (an earlier, "
    "PROSPECTIVE step-length safeguard did exactly this: 497 rejections in 499 "
    "steps, zero speedup)"
)

# (b) below the threshold: a map growing 1.4x per step must NOT trip a 2.0
#     factor, because the accelerator is not the cause of that growth.
Gd_slow = lambda x: np.diag(np.linspace(1.05, 1.4, 50)) @ x + 0.01
acc6b = AndersonAccelerator(depth=5, bounds=None, safeguard_factor=2.0)
acc6b.reset()
x6b = np.ones(50) * 0.01
for _ in range(30):
    x6b = acc6b.step(x6b, Gd_slow(x6b))
print(f"    map growing 1.4x/step vs factor 2.0: {acc6b.n_rejected} rejections "
      f"(correctly below threshold)")
assert acc6b.n_rejected == 0, acc6b.n_rejected

# (c) above the threshold: growth beyond the factor must fire, fall back to the
#     Picard value, and keep the iterate finite.
Gd_fast = lambda x: np.diag(np.linspace(4.0, 6.0, 50)) @ x + 0.01
acc6c = AndersonAccelerator(depth=5, bounds=None, safeguard_factor=2.0)
acc6c.reset()
x6c = np.ones(50) * 1e-6
fell_back = False
for _ in range(25):
    g = Gd_fast(x6c)
    if not np.all(np.isfinite(g)):
        break
    n_rej_before = acc6c.n_rejected
    x6c = acc6c.step(x6c, g)
    if acc6c.n_rejected > n_rej_before:
        # a rejection must return exactly the Picard iterate
        fell_back = bool(np.max(np.abs(x6c - g)) == 0.0)
print(f"    map growing 5x/step vs factor 2.0: {acc6c.n_rejected} rejections, "
      f"fallback == Picard value = {fell_back}")
assert acc6c.n_rejected > 0, (
    "residual growing 5x per step never tripped a safeguard set at 2x -- the "
    "monitor is not watching the residual"
)
assert fell_back, "a rejected step did not return the plain Picard iterate"
print("[6] PASSED")


# ── 7. through the real coupled solver ──────────────────────────────────────
# Pre-severance (moderate damage), which is the regime this package's results
# are verified in. Gc is deliberately high enough to give a real peak-and-
# soften branch rather than an immediate snap to d ~ 1.
n7 = (8, 8, 8)
L7 = (1.0, 1.0, 1.0)
Nv7 = int(np.prod(n7))
l0_7 = 4.0 * L7[0] / n7[0]
mat7 = PhaseFieldDruckerPrager(E=1400.0, nu=0.39, Gc=2.0e-1, sigma_y0=17.0,
                               H=250.0, a_f=0.13, a_g=0.05, a_tip=5.61,
                               beta_p=1.0, sigma_crit=22.0)
phase7 = jnp.zeros(Nv7, dtype=int)
_idx = np.indices(n7)
_c = np.array([x / 2 for x in n7])
_dist = np.linalg.norm(
    np.stack([_idx[i] + 0.5 - _c[i] for i in range(3)]).reshape(3, -1), axis=0)
d7 = jnp.array((_dist <= 1.5).astype(float) * 0.9)
H7 = jnp.zeros(Nv7)
KW7 = dict(formulation="displacement", toler_lin=1e-8, maxiter_cg=200,
           toler_helm=1e-6, maxiter_helm=150, eta=0.0, d_thres=0.95,
           toler_nr=1e-7, maxiter_nr=20)

st7 = None
for k in range(1, 4):                      # walk to a loaded, damaged state
    eb = jnp.zeros((3, 3)).at[0, 0].set(0.002 * k)
    s = solve_fracture(n7, L7, phase7, [mat7], eb, l0_7, None, d7, H7,
                       state_init=st7, toler_st_abs=1e-4, toler_st_rel=1e-5,
                       maxiter_st=60, **KW7)
    d7, H7, st7 = s.d, s.H, (s.eps_p, s.alpha, s.W_p)
assert float(jnp.max(d7)) < 0.95, (
    f"warm-up already severed the specimen (d_max={float(jnp.max(d7)):.4f}) -- "
    "check 7 must run pre-severance to be meaningful"
)

eb7 = jnp.zeros((3, 3)).at[0, 0].set(0.008)
ref7 = solve_fracture(n7, L7, phase7, [mat7], eb7, l0_7, None, d7, H7,
                      state_init=st7, accelerator=None,
                      toler_st_abs=1e-7, toler_st_rel=1e-8, maxiter_st=200, **KW7)
sig_ref = float(jnp.mean(ref7.sigma[0, 0]))
d_ref = np.asarray(ref7.d)
print(f"[7] real coupled solver, pre-severance (d_max={float(jnp.max(d7)):.4f}):")
print(f"      {'Picard':16s} sweeps={ref7.iter_staggered:3d}  "
      f"sigma_11={sig_ref:.5f} MPa")
for name, acc in (("Anderson d=5", AndersonAccelerator(depth=5)),
                  ("NGMRES d=5", NGMRESAccelerator(depth=5))):
    s = solve_fracture(n7, L7, phase7, [mat7], eb7, l0_7, None, d7, H7,
                       state_init=st7, accelerator=acc,
                       toler_st_abs=1e-7, toler_st_rel=1e-8, maxiter_st=200, **KW7)
    sig = float(jnp.mean(s.sigma[0, 0]))
    dd = float(np.max(np.abs(np.asarray(s.d) - d_ref)))
    sp = ref7.iter_staggered / s.iter_staggered
    print(f"      {name:16s} sweeps={s.iter_staggered:3d}  sigma_11={sig:.5f} MPa  "
          f"({sp:.2f}x)  |dsigma|={abs(sig - sig_ref):.1e}  max|dd|={dd:.1e}")
    assert s.iter_staggered < ref7.iter_staggered, (
        f"{name} took {s.iter_staggered} sweeps against Picard's "
        f"{ref7.iter_staggered} -- no speedup on the real problem"
    )
    # Same fixed point. Tolerances are loose in absolute terms because the
    # staggered tolerance is 1e-7 on d and the stress is O(15 MPa).
    assert abs(sig - sig_ref) < 1e-3, (sig, sig_ref)
    assert dd < 1e-4, dd
print("[7] PASSED")


# ── factory ─────────────────────────────────────────────────────────────────
assert build_accelerator(None) is None
assert build_accelerator("none") is None
assert build_accelerator("picard") is None
assert isinstance(build_accelerator("anderson", depth=3), AndersonAccelerator)
assert isinstance(build_accelerator("ngmres", depth=3), NGMRESAccelerator)
for bad, needle in ((lambda: build_accelerator("bogus"), "unknown staggered"),
                    (lambda: AndersonAccelerator(depth=0), "depth"),
                    (lambda: AndersonAccelerator(beta=0.0), "beta"),
                    (lambda: AndersonAccelerator(safeguard_factor=1.0), "safeguard_factor")):
    try:
        bad()
        raise AssertionError(f"expected a ValueError mentioning {needle!r}")
    except ValueError as exc:
        assert needle in str(exc), f"wrong message for {needle!r}: {exc}"

print("\ntest_solvers_damage_anderson: all checks passed")
