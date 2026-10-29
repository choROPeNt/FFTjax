"""Conjugate-gradient drivers shared by every solver."""

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)

import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg as _jax_cg


def cg_solve(A, b, x0, tol, maxiter, M=None):
    """
    ``jax.scipy.sparse.linalg.cg`` with an early-exit ``while_loop``, plus a
    convergence flag (JAX's own ``info`` is always None).

    Not reverse-differentiable: JAX's cg gradient is silently wrong on this
    project's operators. Use ``cg_solve_scan`` for ``jax.grad``.
    """
    x, _ = _jax_cg(A, b, x0=x0, tol=tol, maxiter=maxiter, M=M)
    converged = jnp.linalg.norm(b - A(x)) <= tol * jnp.linalg.norm(b)
    return x, converged


def cg_solve_diff(A, b, x0, tol, maxiter, M=None):
    """
    ``cg_solve`` with implicit differentiation (``jax.lax.custom_linear_solve``).

    Unsafe for every solver here: their operators are exactly singular (the
    DC mode, rigid-body translation), so the adjoint solve divides by a
    near-zero pivot and the gradient is wrong by orders of magnitude. Kept for
    non-singular operators; it does fix JAX cg's assumption that A is
    symmetric whenever b is real. Use ``cg_solve_scan`` instead.

    Returns
    -------
    x         : solution
    converged : forward-solve convergence only
    """
    def solve(matvec, rhs):
        return cg_solve(matvec, rhs, x0, tol, maxiter, M=M)

    return jax.lax.custom_linear_solve(A, b, solve, transpose_solve=solve, has_aux=True)


def cg_solve_scan(A, b, x0, tol, maxiter, M=None):
    """
    Differentiable preconditioned CG over a fixed-length ``lax.scan``.

    Autodiff follows the executed iterations, so the singular operators here
    are safe. Always runs ``maxiter`` steps (frozen once converged): slower
    than ``cg_solve``.
    """
    M_op = (lambda x: x) if M is None else M
    b_norm = jnp.linalg.norm(b)
    atol = tol * b_norm

    r0 = b - A(x0)
    z0 = M_op(r0)
    p0 = z0
    rz0 = jnp.dot(r0, z0)
    conv0 = jnp.linalg.norm(r0) <= atol

    def step(state, _):
        x, r, p, rz, conv = state
        conv_before = conv

        Ap = A(p)
        pAp = jnp.dot(p, Ap)
        alpha = rz / jnp.asarray(jnp.where(pAp != 0.0, pAp, 1.0))
        x_new = x + alpha * p
        r_new = r - alpha * Ap
        z_new = M_op(r_new)
        rz_new = jnp.dot(r_new, z_new)
        beta = rz_new / jnp.asarray(jnp.where(rz != 0.0, rz, 1.0))
        p_new = z_new + beta * p
        conv_new = jnp.linalg.norm(r_new) <= atol

        x_out = jnp.where(conv_before, x, x_new)
        r_out = jnp.where(conv_before, r, r_new)
        p_out = jnp.where(conv_before, p, p_new)
        rz_out = jnp.where(conv_before, rz, rz_new)
        conv_out = conv_before | conv_new
        return (x_out, r_out, p_out, rz_out, conv_out), None

    init = (x0, r0, p0, rz0, conv0)
    (x_f, r_f, _, _, converged), _ = jax.lax.scan(step, init, xs=None, length=maxiter)
    return x_f, converged


def cg_solve_pmap(A, b, x0, tol, maxiter, axis_name, M=None):
    """
    Preconditioned CG on device-local shards, inside ``pmap``/``shard_map``.

    Dot products and norms are ``psum``-reduced over ``axis_name``, so every
    device takes identical steps and no full vector is ever gathered.
    ``converged`` uses a fresh residual ``|b - A x| <= tol |b|``.

    Parameters
    ----------
    A, b, x0, M : as ``cg_solve``, on local shards
    axis_name   : collective axis to reduce over

    Returns
    -------
    x         : local shard of the solution
    converged : bool, identical on every device
    """
    def dot(u, v):
        return jax.lax.psum(jnp.sum(u * v), axis_name)

    def norm(u):
        return jnp.sqrt(dot(u, u))

    M_op = (lambda x: x) if M is None else M
    b_norm = norm(b)
    atol = tol * b_norm

    r0 = b - A(x0)
    z0 = M_op(r0)
    p0 = z0
    rz0 = dot(r0, z0)

    # The state holds the residual VECTOR r; cond_fun recomputes its norm.
    # Storing a scalar norm in that slot silently corrupted r_new before.
    def cond_fun(state):
        _, r, _, _, i = state
        return (norm(r) > atol) & (i < maxiter)

    def body_fun(state):
        x, r, p, rz, i = state
        Ap = A(p)
        pAp = dot(p, Ap)
        alpha = rz / jnp.where(pAp != 0.0, pAp, 1.0)
        x_new = x + alpha * p
        r_new = r - alpha * Ap
        z_new = M_op(r_new)
        rz_new = dot(r_new, z_new)
        beta = rz_new / jnp.where(rz != 0.0, rz, 1.0)
        p_new = z_new + beta * p
        return (x_new, r_new, p_new, rz_new, i + 1)

    init = (x0, r0, p0, rz0, 0)
    x_f, _, _, _, _ = jax.lax.while_loop(cond_fun, body_fun, init)

    converged = norm(b - A(x_f)) <= atol
    return x_f, converged
