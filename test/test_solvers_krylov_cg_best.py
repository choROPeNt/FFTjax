"""
Standalone test for solvers.krylov.cg.cg_solve_best -- CG that returns its
smallest-residual iterate when it stops at the iteration cap unconverged.

Checks
------
1. Converging solve: same answer as cg_solve, to round-off (same algorithm,
   same stopping test), flag True, with and without preconditioner and from
   a nonzero start vector.
2. Capped solve on an ill-conditioned SPD system: for EVERY cap the returned
   residual is <= |b| (the start vector x0 = 0 already has |b|) and <= the
   residual of cg_solve's last iterate; and there are caps where the last
   iterate's residual exceeds |b|, i.e. the case this function exists for
   actually occurs in the test.
3. Breakdown: a zero-curvature direction (p.Ap = 0 -> NaN step) returns the
   best finite iterate, not NaN.

Usage
-----
    python test/test_solvers_krylov_cg_best.py
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
sys.path.insert(0, "src")

import utils.precision  # noqa: F401 -- side effect: configures JAX
import jax
import jax.numpy as jnp
import numpy as np

from solvers.krylov.cg import cg_solve, cg_solve_best

jax.config.update("jax_enable_x64", True)
rng = np.random.default_rng(0)


def spd(nn, cond):
    """Dense SPD matrix with log-uniform spectrum over [1/cond, 1]."""
    q, _ = np.linalg.qr(rng.standard_normal((nn, nn)))
    lam = 10.0 ** np.linspace(-np.log10(cond), 0.0, nn)
    return jnp.asarray((q * lam) @ q.T)


def relres(A, b, x):
    return float(jnp.linalg.norm(b - A @ x) / jnp.linalg.norm(b))


# ── 1. converging solve: identical to cg_solve ───────────────────────────────
A1 = spd(60, 1e3)
b1 = jnp.asarray(rng.standard_normal(60))
M1 = jnp.diag(1.0 / jnp.diag(A1))
worst = 0.0
for M in (None, lambda v: M1 @ v):
    for x0 in (jnp.zeros(60), jnp.asarray(rng.standard_normal(60))):
        xa, ca = cg_solve(lambda v: A1 @ v, b1, x0, 1e-10, 500, M=M)
        xb, cb = cg_solve_best(lambda v: A1 @ v, b1, x0, 1e-10, 500, M=M)
        assert bool(ca) and bool(cb)
        worst = max(worst, float(jnp.max(jnp.abs(xa - xb)) / jnp.max(jnp.abs(xa))))
print(f"[1] converging solve: max relative difference to cg_solve = {worst:.1e}")
assert worst < 1e-12
print("[1] PASSED")

# ── 2. capped solve: never worse than the last iterate, never above |b| ──────
A2 = spd(400, 1e10)
b2 = jnp.asarray(rng.standard_normal(400))
n_spike, n_strict, worst_best = 0, 0, 0.0
for cap in range(1, 121):
    x_last, c_last = cg_solve(lambda v: A2 @ v, b2, jnp.zeros(400), 1e-12, cap)
    x_best, c_best = cg_solve_best(lambda v: A2 @ v, b2, jnp.zeros(400), 1e-12, cap)
    r_last, r_best = relres(A2, b2, x_last), relres(A2, b2, x_best)
    assert not bool(c_last) and not bool(c_best), "check 2 needs unconverged solves"
    # 1e-9 slack: the best iterate is picked on the recursively updated
    # residual, the comparison here is on the true one
    assert r_best <= 1.0 + 1e-9, f"cap {cap}: best iterate has relres {r_best:.3e} > 1"
    assert r_best <= r_last * (1.0 + 1e-9), f"cap {cap}: best {r_best:.3e} > last {r_last:.3e}"
    n_spike += r_last > 1.0
    n_strict += r_best < 0.5 * r_last
    worst_best = max(worst_best, r_best)
print(f"[2] caps 1..120 on cond 1e10: last iterate above |b| at {n_spike} caps, best iterate less than "
      f"half the last's residual at {n_strict}; largest best-iterate relres {worst_best:.3f}")
assert n_spike > 0, "check 2 is vacuous: the last iterate never exceeded |b|"
assert n_strict > 0
print("[2] PASSED")

# ── 3. breakdown returns the best finite iterate ─────────────────────────────
# A singular in the direction of the first search direction: p0 = b, A b = 0.
A3 = jnp.diag(jnp.asarray([0.0, 1.0, 2.0]))
b3 = jnp.asarray([1.0, 0.0, 0.0])
x3, c3 = cg_solve_best(lambda v: A3 @ v, b3, jnp.zeros(3), 1e-10, 10)
print(f"[3] breakdown: x = {np.asarray(x3)}  converged={bool(c3)}")
assert bool(jnp.all(jnp.isfinite(x3))) and not bool(c3)
print("[3] PASSED")

print("\ntest_solvers_krylov_cg_best: all checks passed")
