"""
Standalone test for calibration.calibrate -- calibrate_adam and
calibrate_newton, factored out of
notebooks/inverse_calibration/lin-elastic_inverse-calibration.ipynb's own
calibration cell and its comparison helpers.

calibrate_adam -- four checks
------------------------------
1. Converges to the minimum of a simple quadratic loss under jax.jacfwd,
   and history[0] is exactly (param_init, loss(param_init)) -- confirming
   the (param, loss) pairing is "at the start of this step", not shifted.
2. loss_tol stops early: fewer than max_steps entries in history once the
   loss target is reached, and the returned param is history's last entry.
3. A hand-supplied finite-difference gradient (no jax.jacfwd at all)
   converges to the same minimum as check 1 -- calibrate_adam() doesn't care
   how value_and_grad_fn computes its gradient, only that it returns one,
   the same substitution
   notebooks/inverse_calibration/lin-elastic_inverse-calibration.ipynb makes
   for its jax.jacfwd-vs-finite-difference comparison.
4. verbose=True doesn't change the optimization trajectory, only whether it
   prints -- checked by comparing final params/histories to check 1's run.

calibrate_newton -- three checks
---------------------------------
5. Exact quadratic loss: converges in exactly ONE step (Newton's own
   textbook property -- the local quadratic model IS the loss on an exact
   quadratic, so the very first step lands exactly on the minimum, up to
   float precision).
6. A smooth but non-quadratic loss (quartic well): still converges close to
   the minimum in far fewer steps than calibrate_adam needs on the same
   problem, and -- the actual point of adding it -- monotonically, with no
   overshoot/oscillation in the recorded history.
7. loss_tol stops early, same contract as calibrate_adam's check 2.

Usage
-----
    python test/test_calibration_calibrate.py
"""

import sys
sys.path.insert(0, "src")

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)
import jax
import jax.numpy as jnp
import numpy as np

from calibration.calibrate import calibrate_adam, calibrate_newton

TARGET = 42.0


def loss(p):
    return (p - TARGET) ** 2


value_and_grad_autodiff = jax.jit(lambda p: (loss(p), jax.jacfwd(loss)(p)))

# ── 1. converges to the true minimum; history[0] pairing is exact ──────────

param, history = calibrate_adam(value_and_grad_autodiff, param_init=0.0, max_steps=200)
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
param_early, history_early = calibrate_adam(
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
param_fd, history_fd = calibrate_adam(value_and_grad_fd, param_init=0.0, max_steps=200)
err_fd = abs(param_fd - TARGET)
print(f"[3] finite-difference calibrated param={param_fd:.4f}  |error|={err_fd:.3e}")
assert err_fd < 1e-2, "a finite-difference gradient should reach the same minimum"
print("[3] PASSED")

# ── 4. verbose=True doesn't change the trajectory ───────────────────────────

param_v, history_v = calibrate_adam(
    value_and_grad_autodiff, param_init=0.0, max_steps=200, verbose=True, print_every=50,
)
assert abs(param_v - param) < 1e-9
assert history_v == history
print("[4] PASSED (verbose output above -- trajectory matches check [1] exactly)")

# ── 5. exact quadratic: Newton converges in exactly one step ───────────────
# (loss_tol=1e-6 so the loop stops as soon as that first step's landing spot
# is recorded, rather than running out the full max_steps at the minimum)

param_n, history_n = calibrate_newton(loss, param_init=0.0, max_steps=10, loss_tol=1e-6)
err_n = abs(param_n - TARGET)
print(f"[5] Newton calibrated param={param_n:.6f}  |error|={err_n:.3e}  n_steps={len(history_n)}")
assert len(history_n) == 2, \
    "one Newton step should exactly solve an exact quadratic: history[0] is the start, " \
    "history[1] is already at the minimum"
assert err_n < 1e-8, "Newton should land on the quadratic's minimum to float precision"
print("[5] PASSED")

# ── 6. smooth non-quadratic loss: fewer steps than Adam, no oscillation ────

QUARTIC_TARGET = 5.0


def loss_quartic(p):
    # A gentle quartic well: still one clean minimum, but curvature isn't
    # constant the way it is for a quadratic, so Newton needs more than one
    # step here -- the point of this check is "still smooth", not "still 1".
    return (p - QUARTIC_TARGET) ** 4 + 0.5 * (p - QUARTIC_TARGET) ** 2


value_and_grad_quartic = jax.jit(lambda p: (loss_quartic(p), jax.jacfwd(loss_quartic)(p)))
param_adam_q, history_adam_q = calibrate_adam(
    value_and_grad_quartic, param_init=-5.0, max_steps=200, init_lr=5.0,
)
param_newton_q, history_newton_q = calibrate_newton(loss_quartic, param_init=-5.0, max_steps=30)

err_adam_q = abs(param_adam_q - QUARTIC_TARGET)
err_newton_q = abs(param_newton_q - QUARTIC_TARGET)
print(f"[6] quartic well: Adam {len(history_adam_q)} steps -> |error|={err_adam_q:.3e}, "
      f"Newton {len(history_newton_q)} steps -> |error|={err_newton_q:.3e}")
assert err_newton_q < 1e-4, "Newton should converge tightly on a smooth, single-minimum quartic"
assert len(history_newton_q) < len(history_adam_q), \
    "Newton should reach a tight minimum in far fewer steps than Adam's fixed schedule"

params_n = np.array([p for p, _ in history_newton_q])
errors_n = np.abs(params_n - QUARTIC_TARGET)
assert np.all(np.diff(errors_n) <= 1e-9), \
    f"Newton's distance to the target should shrink monotonically, no overshoot/bounce (errors={errors_n})"
print("[6] PASSED (Newton's distance to the target shrinks monotonically, no bounce)")

# ── 7. loss_tol stops early, same contract as calibrate_adam ───────────────

loss_tol_n = 1e-3
param_n_early, history_n_early = calibrate_newton(
    loss_quartic, param_init=-5.0, max_steps=30, loss_tol=loss_tol_n,
)
print(f"[7] Newton early-stopped after {len(history_n_early)} steps, "
      f"final recorded loss={history_n_early[-1][1]:.4e} (tol={loss_tol_n})")
assert len(history_n_early) < 30
assert history_n_early[-1][1] < loss_tol_n
print("[7] PASSED")

print("\ntest_calibration_calibrate: all checks passed")
