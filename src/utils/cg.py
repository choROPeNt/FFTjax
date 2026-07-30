import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)

import jax
import jax.numpy as jnp


def cg_count(
    A,
    b:       jnp.ndarray,
    x0:      jnp.ndarray,
    tol:     float,
    maxiter: int,
    M=None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Preconditioned CG with iteration counter, compatible with jax.jit.

    Implements conjugate gradients via jax.lax.while_loop so the iteration
    count is a traced integer returned alongside the solution.

    Parameters
    ----------
    A       : callable  linear operator  y = A(x)
    b       : (N,)      right-hand side
    x0      : (N,)      initial guess
    tol     : float     relative residual tolerance  ‖r‖ / ‖b‖ < tol
    maxiter : int       maximum iterations (must be static for JIT)
    M       : callable or None  preconditioner  z = M(r)  (M ≈ A⁻¹)

    Returns
    -------
    x          : (N,)        solution
    iter_count : int array   actual number of iterations performed
    converged  : bool array  True if ‖r‖ / ‖b‖ < tol at exit
    """
    M_op   = (lambda x: x) if M is None else M
    b_norm = jnp.linalg.norm(b)
    atol   = tol * b_norm

    r0  = b - A(x0)
    z0  = M_op(r0)
    p0  = z0
    rz0 = jnp.dot(r0, z0)

    def cond(state):
        _, r, _, _, k = state
        return (k < maxiter) & (jnp.linalg.norm(r) > atol)

    def body(state):
        x, r, p, rz, k = state
        Ap    = A(p)
        pAp   = jnp.dot(p, Ap)
        alpha = rz / jnp.where(pAp != 0.0, pAp, 1.0)
        x_new = x + alpha * p
        r_new = r - alpha * Ap
        z_new = M_op(r_new)
        rz_new = jnp.dot(r_new, z_new)
        beta   = rz_new / jnp.where(rz != 0.0, rz, 1.0)
        p_new  = z_new + beta * p
        return x_new, r_new, p_new, rz_new, k + 1

    x_f, r_f, _, _, k_f = jax.lax.while_loop(
        cond, body, (x0, r0, p0, rz0, jnp.int32(0))
    )

    converged = jnp.linalg.norm(r_f) <= atol
    return x_f, k_f, converged
