"""
Gradient-descent parameter calibration against a differentiable forward
model: an Adam (optax) loop over a scalar (or array) parameter, driven by
any value_and_grad_fn -- typically loss(param), jax.jacfwd(loss)(param) for
the common case, but a caller can substitute a finite-difference gradient
just as easily (see notebooks/lin-elastic_inverse-calibration.ipynb's own
jax.jacfwd-vs-finite-difference comparison, which calls this with both).

A sibling of learning/, not a member of it: learning.surrogates.GPSurrogate
fits a statistical model to sparse (X, y) samples and interpolates between
them, with no physical forward model involved. This instead inverts one
unknown parameter of an existing differentiable physical forward model
against real reference data by gradient descent -- a different problem
(parameter estimation, not surrogate regression), sharing only the general
"JAX autodiff + optax" toolset, not any code, with learning/.

Factored out of that notebook's own calibration loop and its calibrate()
comparison helper, which were the same optax optimizer/schedule/history
bookkeeping hand-copied twice -- one always ran a fixed step budget, the
other stopped early once a loss tolerance was reached; both are now the one
`loss_tol` argument below.
"""

from collections.abc import Callable

import jax.numpy as jnp
import optax


def calibrate(
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
