import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)

import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg as _jax_cg


def cg_solve(A, b, x0, tol, maxiter, M=None):
    """
    Thin wrapper around jax.scipy.sparse.linalg.cg that also reports
    convergence, since the underlying JAX version's own ``info`` return is
    currently a permanent ``None`` placeholder (not yet implemented there).

    Uses ``jax.lax.while_loop`` internally, so it stops as soon as it
    converges rather than paying for ``maxiter`` on every call -- 3-20x
    faster in practice than a fixed-length scan with a generous ``maxiter``
    safety cap. The cost: this does **not** support reverse-mode autodiff
    (``jax.grad``) through the solve -- JAX's own ``cg`` gradient rule was
    tested against this project's FFT-based operators and found to give
    silently incorrect results (not an error, not NaN -- a wrong gradient,
    off by many orders of magnitude on a heterogeneous composite, and NaN
    on a degenerate zero-residual case). If a solve here ever needs to be
    differentiated with ``jax.grad``/``jax.jvp``, use ``cg_solve_scan``
    below (verified working on this project's actual operators, just much
    slower per call since it can't early-exit) -- **not** ``cg_solve_diff``,
    which looks like the obvious fix but is unsafe here; see its docstring.

    Shared by every Newton-CG solver in ``solvers/`` (mechanical and
    damage) -- kept here once rather than redefined per module.
    """
    x, _ = _jax_cg(A, b, x0=x0, tol=tol, maxiter=maxiter, M=M)
    converged = jnp.linalg.norm(b - A(x)) <= tol * jnp.linalg.norm(b)
    return x, converged


def cg_solve_diff(A, b, x0, tol, maxiter, M=None):
    """
    Differentiable counterpart of ``cg_solve`` via implicit differentiation
    (``jax.lax.custom_linear_solve``). Fixes one real, confirmed bug in
    ``jax.scipy.sparse.linalg.cg``'s own gradient rule -- but is still
    **UNSAFE for every solver in this project** (``dstrain_nw_cg``,
    ``ddisp_nw_cg``, the plastic and damage solvers) for a separate,
    unrelated reason. Use ``cg_solve_scan`` instead. Kept here, correctness
    verified independently, in case a future non-singular operator needs it
    -- see the two findings below before reaching for this function.

    Finding 1 -- the symmetry-assumption bug (fixed here, doesn't matter
    for finding 2): the forward value is identical to ``cg_solve`` (same
    fast ``while_loop`` CG). Gradients come from JAX auto-deriving the true
    transpose of ``A`` via ``jax.linear_transpose`` (every ``A_op`` here is
    plain ``einsum``/``fftn``/``ifftn``, all transposable), then solving the
    adjoint system with this same ``cg_solve`` -- not
    ``jax.scipy.sparse.linalg.cg``'s own implicit-diff rule, which
    (``jax/_src/scipy/sparse/linalg.py::_isolve``) sets
    ``symmetric = all(real_valued(b)) if check_symmetric else False`` --
    i.e. it assumes ``A`` is symmetric whenever ``b`` is real, with **no
    check that ``A`` actually is**. Verified on a synthetic non-symmetric
    linear system (probe-vector loss ``dot(w, x)``, NOT ``sum(x**2)`` --
    that masks the bug via the scalar identity ``x.T @ Ainv @ x == x.T @
    Ainv.T @ x``, true for *any* matrix): this function matches a dense
    ground truth to ~1e-13 relative error where the built-in ``cg``
    gradient is off by 1-3%.

    Finding 2 -- why this still can't be used here: every ``A_op`` in this
    project's mechanical/damage solvers is **exactly singular by
    construction** -- ``dstrain_nw_cg``'s Green's operator is identically
    zero at the DC (zero-frequency) mode (verified: ``G_glob[...,0]`` has
    norm 0.0) since the mean strain is prescribed separately as ``eps_bar``,
    not solved for; ``ddisp_nw_cg`` has the analogous rigid-body-translation
    null space (see its own preconditioner comment). The forward CG solve
    never notices, because it starts at ``x0=0`` and never explores the
    null direction. Implicit differentiation does notice: the backward pass
    solves an adjoint system whose right-hand side (the loss cotangent) has
    no reason to avoid that direction, so CG divides by a near-zero pivot.
    Reproduced directly: on the real ``dstrain_nw_cg`` operator this
    function's ``jax.grad`` was off by ~10 orders of magnitude (matching
    ``jax.scipy.sparse.linalg.cg``'s own gradient, similarly wrong) --
    confirmed as an operator-singularity problem, not a leftover transpose
    bug, by reconstructing the same small operator as a dense matrix and
    finding ``jnp.linalg.solve`` on it returns ``NaN``.

    ``cg_solve_scan`` avoids this because it never inverts ``A`` as an
    abstract linear map -- it differentiates the literal sequence of
    executed CG steps, which (like the forward solve) starts at zero and
    never touches the null direction either.

    Returns
    -------
    x         : solution, same pytree structure as ``b``
    converged : bool array -- ``cg_solve``'s convergence flag for the
                *forward* solve only; the backward-pass adjoint solve's own
                convergence cannot be surfaced to the caller at all, which
                is exactly what breaks in finding 2 above.
    """
    def solve(matvec, rhs):
        return cg_solve(matvec, rhs, x0, tol, maxiter, M=M)

    return jax.lax.custom_linear_solve(A, b, solve, transpose_solve=solve, has_aux=True)


def cg_solve_scan(A, b, x0, tol, maxiter, M=None):
    """
    Differentiable CG via a fixed-length ``jax.lax.scan`` instead of
    ``jax.lax.while_loop`` -- ``jax.grad``/``jax.jvp`` differentiate the
    literal sequence of executed iterations by ordinary autodiff (``scan``
    has a registered transpose rule; ``while_loop`` does not), so this needs
    no implicit-function-theorem machinery and no operator-invertibility
    assumption. That's what makes it safe on this project's structurally
    singular ``A_op``s where ``cg_solve_diff`` is not -- see that function's
    docstring for the full finding.

    Runs the full ``maxiter`` budget on every call (state is frozen once
    converged, so extra steps are no-ops, not wrong -- just not free): no
    early exit, unlike ``cg_solve``. Originally
    ``notebooks/lin-elastic_gradient-check.ipynb``'s ``cg_diff``, promoted
    here once validated against the real (heterogeneous) ``dstrain_nw_cg``
    operator, not just synthetic systems.
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
        conv_before = conv  # already converged entering this step -> freeze

        Ap = A(p)
        pAp = jnp.dot(p, Ap)
        alpha = rz / jnp.where(pAp != 0.0, pAp, 1.0)
        x_new = x + alpha * p
        r_new = r - alpha * Ap
        z_new = M_op(r_new)
        rz_new = jnp.dot(r_new, z_new)
        beta = rz_new / jnp.where(rz != 0.0, rz, 1.0)
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
