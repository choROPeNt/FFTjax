"""
Standalone test for calibration.calibrate.calibrate -- the Adam calibration
loop factored out of notebooks/lin-elastic_inverse-calibration.ipynb's own
calibration cell and its calibrate() comparison helper.

Four checks
-----------
1. Converges to the minimum of a simple quadratic loss under jax.jacfwd,
   and history[0] is exactly (param_init, loss(param_init)) -- confirming
   the (param, loss) pairing is "at the start of this step", not shifted.
2. loss_tol stops early: fewer than max_steps entries in history once the
   loss target is reached, and the returned param is history's last entry.
3. A hand-supplied finite-difference gradient (no jax.jacfwd at all)
   converges to the same minimum as check 1 -- calibrate() doesn't care how
   value_and_grad_fn computes its gradient, only that it returns one, the
   same substitution notebooks/lin-elastic_inverse-calibration.ipynb makes
   for its jax.jacfwd-vs-finite-difference comparison.
4. verbose=True doesn't change the optimization trajectory, only whether it
   prints -- checked by comparing final params/histories to check 1's run.

Usage
-----
    python test/test_calibration_calibrate.py
"""

import sys
sys.path.insert(0, "src")

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)
import jax
import jax.numpy as jnp

from calibration.calibrate import calibrate

TARGET = 42.0


def loss(p):
    return (p - TARGET) ** 2


value_and_grad_autodiff = jax.jit(lambda p: (loss(p), jax.jacfwd(loss)(p)))

# ── 1. converges to the true minimum; history[0] pairing is exact ──────────

param, history = calibrate(value_and_grad_autodiff, param_init=0.0, max_steps=200)
err = abs(param - TARGET)
print(f"[1] calibrated param={param:.4f}  target={TARGET}  |error|={err:.3e}  n_steps={len(history)}")
assert err < 1e-2, "Adam should converge close to the quadratic's minimum in 200 steps"
assert len(history) == 200
p0, l0 = history[0]
assert abs(p0 - 0.0) < 1e-12 and abs(l0 - float(loss(0.0))) < 1e-12, \
    "history[0] must be exactly (param_init, loss(param_init))"
print("[1] PASSED")

# ── 2. loss_tol stops early ─────────────────────────────────────────────────

loss_tol = 1.0  # loose -- (p - 42)^2 < 1 as soon as p is within 1 of the target
param_early, history_early = calibrate(
    value_and_grad_autodiff, param_init=0.0, max_steps=200, loss_tol=loss_tol,
)
print(f"[2] early-stopped after {len(history_early)} steps, param={param_early:.4f}, "
      f"final recorded loss={history_early[-1][1]:.4e} (tol={loss_tol})")
assert len(history_early) < 200, "loss_tol should stop well before max_steps on this easy problem"
assert history_early[-1][1] < loss_tol
assert abs(param_early - history_early[-1][0]) < 1e-9, \
    "returned param must equal history's last recorded param exactly"
print("[2] PASSED")

# ── 3. finite-difference gradient reaches the same minimum ─────────────────

h = 1e-3


def grad_fd(p):
    return (loss(p + h) - loss(p - h)) / (2.0 * h)


value_and_grad_fd = jax.jit(lambda p: (loss(p), grad_fd(p)))
param_fd, history_fd = calibrate(value_and_grad_fd, param_init=0.0, max_steps=200)
err_fd = abs(param_fd - TARGET)
print(f"[3] finite-difference calibrated param={param_fd:.4f}  |error|={err_fd:.3e}")
assert err_fd < 1e-2, "a finite-difference gradient should reach the same minimum"
print("[3] PASSED")

# ── 4. verbose=True doesn't change the trajectory ───────────────────────────

param_v, history_v = calibrate(
    value_and_grad_autodiff, param_init=0.0, max_steps=200, verbose=True, print_every=50,
)
assert abs(param_v - param) < 1e-9
assert history_v == history
print("[4] PASSED (verbose output above -- trajectory matches check [1] exactly)")

print("\ntest_calibration_calibrate: all checks passed")
