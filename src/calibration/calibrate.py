"""
Calibrate parameters of a differentiable forward model against reference data.

calibrate_adam   : first order, any gradient source, scalar or array; its
                   fixed-size steps oscillate around the minimum until the
                   schedule damps them
calibrate_newton : second order, scalar only; converges in a few steps
                   without oscillation on a smooth convex loss
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
    Adam with exponential learning-rate decay.

    Parameters
    ----------
    value_and_grad_fn : param -> (loss, grad)
    param_init        : initial value
    max_steps         : step cap
    loss_tol          : stop once loss < loss_tol (None = run all steps)
    init_lr, transition_steps, decay_rate : decay schedule
    verbose, print_every : progress output

    Returns
    -------
    param   : final value
    history : (param, loss) per step taken
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
    Scalar Newton: ``param -= grad / (hessian + damping)``, both by autodiff of loss_fn.

    Scalar only (a vector step needs the Hessian matrix); use calibrate_adam
    or LBFGS for more unknowns. ``damping`` > 0 tames overshoot on a weakly
    convex loss. Other arguments and returns as calibrate_adam.
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
