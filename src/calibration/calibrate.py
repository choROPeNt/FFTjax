"""
Parameter calibration against a differentiable forward model: invert one (or
a few) unknown parameters of an existing differentiable physical forward
model against reference data. Two calibrators, for two different regimes:

calibrate_adam    : first-order, any value_and_grad_fn (autodiff or a
                    hand-supplied finite-difference gradient), scalar or
                    array param -- the general-purpose default, but its
                    fixed-magnitude Adam steps overshoot and oscillate
                    around a well-conditioned minimum before the learning-
                    rate schedule damps it out (see
                    notebooks/inverse_calibration/lin-elastic_inverse-calibration.ipynb's
                    own convergence plot for exactly this).
calibrate_newton  : second-order, scalar param only, needs loss_fn itself
                    (always differentiated twice via jax.grad, not a
                    pluggable gradient source) -- converges in a handful of
                    iterations with no oscillation at all on a smooth,
                    locally-convex 1-D loss, since each step already
                    accounts for curvature instead of taking a fixed-size
                    step and hoping the schedule sorts it out. The common
                    case for calibrating one physical constant.

A sibling of learning/, not a member of it: learning.surrogates.GPSurrogate
fits a statistical model to sparse (X, y) samples and interpolates between
them, with no physical forward model involved. This instead inverts one
unknown parameter of an existing differentiable physical forward model
against real reference data -- a different problem (parameter estimation,
not surrogate regression), sharing only the general "JAX autodiff" toolset,
not any code, with learning/.

calibrate_adam was factored out of that notebook's own calibration loop and
its calibrate() comparison helper, which were the same optax optimizer/
schedule/history bookkeeping hand-copied twice -- one always ran a fixed
step budget, the other stopped early once a loss tolerance was reached; both
are now the one `loss_tol` argument below. calibrate_newton was added
alongside it once the Adam trajectory's oscillation turned out to be worth
an actual fix rather than just a smaller learning rate.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import optax


def calibrate_adam(
    value_and_grad_fn: Callable,
    param_init,
    max_steps: int = 100,
    loss_tol: float | None = None,
    init_lr: float = 800.0,
    transition_steps: int = 15,
    decay_rate: float = 0.7,
    verbose: bool = False,
    print_every: int = 10,
) -> tuple[float, list[tuple[float, float]]]:
    """
    Parameters
    ----------
    value_and_grad_fn : param -> (loss, grad), e.g.
                         jax.jit(lambda p: (loss(p), jax.jacfwd(loss)(p)))
    param_init        : initial parameter value (scalar or array)
    max_steps         : hard cap on optimizer steps
    loss_tol          : stop as soon as loss < loss_tol, without taking that
                         step's update -- None (default) always runs the
                         full max_steps, e.g. to inspect the whole
                         convergence history rather than stop at a target
    init_lr, transition_steps, decay_rate : optax.exponential_decay schedule
                         feeding optax.adam -- this project's calibration
                         notebook's own tuned values as the defaults
    verbose           : print progress every print_every steps
    print_every       : see verbose

    Returns
    -------
    param   : float -- final calibrated parameter
    history : list[(float, float)] -- (param, loss) at the start of every
              step actually taken; len(history) is the iteration count
    """
    schedule = optax.exponential_decay(
        init_value=init_lr, transition_steps=transition_steps, decay_rate=decay_rate,
    )
    optimizer = optax.adam(learning_rate=schedule)
    param = jnp.asarray(param_init, dtype=float)
    opt_state = optimizer.init(param)

    history: list[tuple[float, float]] = []
    for step in range(max_steps):
        loss_val, grad_val = value_and_grad_fn(param)
        history.append((float(param), float(loss_val)))
        if verbose and (step % print_every == 0 or step == max_steps - 1):
            print(f"step {step:3d}   param = {float(param):10.4f}   loss = {float(loss_val):.4e}")
        if loss_tol is not None and float(loss_val) < loss_tol:
            break
        updates, opt_state = optimizer.update(grad_val, opt_state, param)
        param = optax.apply_updates(param, updates)

    return float(param), history


def calibrate_newton(
    loss_fn: Callable[[float], float],
    param_init,
    max_steps: int = 20,
    loss_tol: float | None = None,
    damping: float = 0.0,
    verbose: bool = False,
    print_every: int = 5,
) -> tuple[float, list[tuple[float, float]]]:
    """
    Scalar Newton's method: param -= grad(loss)(param) / hessian(loss)(param)
    each step, both from jax.grad (differentiating loss_fn twice via
    autodiff, not finite differences) -- exact in one step on an exact
    quadratic, and a handful of steps on anything smooth and locally convex
    near the minimum. No learning-rate schedule to tune, because there's no
    fixed-size step to begin with: each step already accounts for the loss's
    local curvature.

    Only for a SCALAR param -- a vector Newton step needs the full Hessian
    matrix and a linear solve, out of scope for what "simple" means here.
    Reach for calibrate_adam (or a proper second-order optimizer, e.g.
    optax's LBFGS) once there's more than one unknown.

    Parameters
    ----------
    loss_fn   : param -> scalar loss (plain, not a value_and_grad_fn --
                Newton needs curvature too, so this always differentiates
                loss_fn itself via jax.grad twice rather than accepting a
                pluggable gradient source the way calibrate_adam does)
    param_init : initial parameter value (scalar)
    max_steps : hard cap on Newton iterations
    loss_tol  : stop as soon as loss < loss_tol, without taking that step's
                update -- None (default) always runs the full max_steps
    damping   : added to the Hessian before dividing (Levenberg-Marquardt
                style) -- 0 (default) is pure Newton; raise it if the loss
                is only weakly convex near the current point and pure
                Newton's raw 1/hessian step overshoots
    verbose, print_every : see calibrate_adam

    Returns
    -------
    param, history -- same shape as calibrate_adam
    """
    grad_fn = jax.grad(loss_fn)
    hess_fn = jax.grad(grad_fn)

    param = jnp.asarray(param_init, dtype=float)
    history: list[tuple[float, float]] = []
    for step in range(max_steps):
        loss_val = loss_fn(param)
        history.append((float(param), float(loss_val)))
        if verbose and (step % print_every == 0 or step == max_steps - 1):
            print(f"step {step:3d}   param = {float(param):10.4f}   loss = {float(loss_val):.4e}")
        if loss_tol is not None and float(loss_val) < loss_tol:
            break
        grad_val = grad_fn(param)
        hess_val = hess_fn(param)
        param = param - grad_val / (hess_val + damping)

    return float(param), history
