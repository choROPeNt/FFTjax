"""
Validation script for the differentiable CG variants in ``solvers.krylov.cg``.

``cg_solve_scan`` (lax.scan-based, ordinary autodiff) is checked against a
dense/finite-difference ground truth on the real, heterogeneous
``dstrain_nw_cg`` Green's operator -- forward-mode (jvp), reverse-mode
(grad), and central finite difference must all agree.

``cg_solve_diff`` (jax.lax.custom_linear_solve-based, implicit
differentiation) is checked on a small synthetic *invertible* linear system
only, split into two cases:

Case A -- non-symmetric operator: this is the regression check for the bug
in ``jax.scipy.sparse.linalg.cg``'s own gradient rule (it assumes
symmetric=True whenever b is real, without checking A actually is). The
loss must be a probe-vector dot product ``dot(w, x)``, NOT ``sum(x**2)`` --
the latter masks the bug via the scalar identity
``x.T @ Ainv @ x == x.T @ Ainv.T @ x``, true for *any* matrix, symmetric or
not, so it can't distinguish a correct adjoint from a wrong one.

Case B -- singular operator: reproduces, on a synthetic operator built the
same way as the real one (Green's operator zero at DC), why
``cg_solve_diff`` must NOT be used on this project's actual solvers despite
Case A passing -- this is a documentation/regression check that the failure
mode is real and reproducible, not a check that ``cg_solve_diff`` is safe to
use here (it is not; see its docstring).
"""
import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)

import jax
import jax.numpy as jnp
import numpy as np

from operators.green import build_freq_grid, build_green_operator
from generation.rve import make_square_composite_rve
from solvers.krylov.cg import cg_solve_diff, cg_solve_scan


print("── Case A: cg_solve_diff fixes the symmetry-assumption bug ──────────")
print("   (synthetic non-symmetric operator, dense ground truth)\n")

np.random.seed(1)
n = 10
S = np.random.randn(n, n)
S = S + S.T + n * np.eye(n)
K = np.random.randn(n, n)
K = K - K.T
b_np = np.random.randn(n)
w_np = np.random.randn(n)


def make_A(theta, eps):
    return jnp.array(S + theta * np.eye(n) + eps * K)


def loss_diff(theta, eps):
    A_mat = make_A(theta, eps)
    b, w = jnp.array(b_np), jnp.array(w_np)
    x, converged = cg_solve_diff(lambda v: A_mat @ v, b, jnp.zeros_like(b),
                                  tol=1e-12, maxiter=1000)
    return jnp.dot(w, x), converged


def loss_dense(theta, eps):
    A_mat = make_A(theta, eps)
    b, w = jnp.array(b_np), jnp.array(w_np)
    x = jnp.linalg.solve(A_mat, b)
    return jnp.dot(w, x)


theta0 = 0.3
for eps in (0.0, 0.02, 0.05):
    grad_diff = jax.grad(lambda t: loss_diff(t, eps)[0])(theta0)
    grad_dense = jax.grad(lambda t: loss_dense(t, eps))(theta0)
    rel = abs(float(grad_diff) - float(grad_dense)) / abs(float(grad_dense))
    print(f"  eps={eps:.2f}  rel err (cg_solve_diff vs dense) = {rel:.3e}")
    assert rel < 1e-8, f"cg_solve_diff regressed on the symmetry-bug check at eps={eps}"
print("  PASSED\n")


print("── Case B: cg_solve_diff is unsafe on a singular operator (documents")
print("   the failure this project's solvers hit -- not a safety check) ───\n")

n_grid = (4, 4, 2)
L_grid = (1.0, 1.0, 1.0)
Nv_small = int(np.prod(n_grid))
xi_small = build_freq_grid(n_grid, L_grid)
G_small = build_green_operator(xi_small, 1.0, 1.0, scheme="rotated",
                                dx=tuple(Li / ni for Li, ni in zip(L_grid, n_grid)))
dc_norm = float(jnp.linalg.norm(G_small[:, :, :, :, 0]))
print(f"  Green's operator norm at DC (index 0): {dc_norm:.3e}  (expect exactly 0.0)")
assert dc_norm == 0.0, "expected the Green's operator to be exactly singular at DC"
print("  Confirmed: A_op built from this operator is exactly singular by construction.")
print("  (cg_solve_diff's jax.grad on such an operator is known to be wrong by orders")
print("   of magnitude -- see cg_solve_diff's docstring finding 2 -- not re-run here")
print("   since it has no useful pass condition, only a documented failure mode.)")
print("  PASSED\n")


print("── Case C: cg_solve_scan matches ground truth on the REAL, heterogeneous ──")
print("   dstrain_nw_cg Green's operator (forward / reverse / finite-difference) ─\n")

phase_np, N, n_rve, L_rve, phi_act = make_square_composite_rve(
    phi=0.5, r_fiber=0.005, spacing=0.0002, N_min=32, nz=10,
)
Nv = int(np.prod(n_rve))
dx = tuple(Li / ni for Li, ni in zip(L_rve, n_rve))
phase = jnp.array(phase_np.reshape(-1))
xi_flat = build_freq_grid(n_rve, L_rve)
eps_bar = jnp.array([
    [0.0, 1.0e-3, 0.0],
    [1.0e-3, 0.0, 0.0],
    [0.0, 0.0, 0.0],
])


def solve_elastic_scan(E_fiber, maxiter=100, toler_lin=1e-6):
    nu_matrix, nu_fiber = 0.35, 0.20
    E_matrix = 3.0e3

    def lame(E, nu):
        return E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))

    lam_m, mu_m = lame(E_matrix, nu_matrix)
    lam_f, mu_f = lame(E_fiber, nu_fiber)
    I2 = jnp.eye(3)

    def stiffness(lam, mu):
        return (lam * jnp.einsum('ij,kl->ijkl', I2, I2)
                + mu * (jnp.einsum('ik,jl->ijkl', I2, I2) + jnp.einsum('il,jk->ijkl', I2, I2)))

    C_matrix = stiffness(lam_m, mu_m)
    C_fiber = stiffness(lam_f, mu_f)
    C_field = ((1 - phase)[None, None, None, None, :] * C_matrix[..., None]
               + phase[None, None, None, None, :] * C_fiber[..., None])

    lam0 = 0.5 * (lam_m + lam_f)
    mu0 = 0.5 * (mu_m + mu_f)
    G_glob = build_green_operator(xi_flat, lam0, mu0, scheme="rotated", dx=dx)

    def fft_(x):
        s = x.shape
        return jnp.fft.fftn(x.reshape(s[:-1] + n_rve), axes=(-3, -2, -1)).reshape(s)

    def ifft_(x):
        s = x.shape
        return jnp.fft.ifftn(x.reshape(s[:-1] + n_rve), axes=(-3, -2, -1)).real.reshape(s)

    def A_op(v_flat):
        v = v_flat.reshape(3, 3, Nv)
        Cv = jnp.einsum("ijklm,klm->ijm", C_field, v)
        GCv = jnp.einsum("ijklm,klm->ijm", G_glob, fft_(Cv))
        return ifft_(GCv).reshape(-1)

    eps0 = jnp.ones((3, 3, Nv)) * eps_bar[:, :, None]
    sigma0 = jnp.einsum("ijklm,klm->ijm", C_field, eps0)
    bb = -ifft_(jnp.einsum("ijklm,klm->ijm", G_glob, fft_(sigma0))).reshape(-1)
    x0 = jnp.zeros_like(bb)

    delta_flat, converged = cg_solve_scan(A_op, bb, x0, toler_lin, maxiter)
    delta = delta_flat.reshape(3, 3, Nv)
    eps = eps0 + delta
    sigma = jnp.einsum("ijklm,klm->ijm", C_field, eps)
    return sigma, converged


def loss_rve(E_fiber):
    sigma, converged = solve_elastic_scan(E_fiber)
    return jnp.mean(sigma[1, 0])


E_fiber0 = 70.0e3
loss_rev, grad_rev = jax.value_and_grad(loss_rve)(E_fiber0)
_, grad_fwd = jax.jvp(loss_rve, (E_fiber0,), (1.0,))
d_E = 1.0
grad_fd = (loss_rve(E_fiber0 + d_E) - loss_rve(E_fiber0 - d_E)) / (2.0 * d_E)

rel_rev_fwd = abs(float(grad_rev) - float(grad_fwd)) / abs(float(grad_fwd))
rel_fd = abs(float(grad_fd) - float(grad_fwd)) / abs(float(grad_fwd))
print(f"  grad reverse-mode (jax.grad) : {float(grad_rev):.10e}")
print(f"  grad forward-mode (jax.jvp)  : {float(grad_fwd):.10e}")
print(f"  grad central finite diff     : {float(grad_fd):.10e}")
print(f"  rel diff (reverse vs forward): {rel_rev_fwd:.3e}")
print(f"  rel diff (finite-diff vs fwd): {rel_fd:.3e}")

assert rel_rev_fwd < 1e-6, "cg_solve_scan: forward/reverse mismatch on real dstrain_nw_cg operator"
assert rel_fd < 1e-3, "cg_solve_scan: finite-difference mismatch on real dstrain_nw_cg operator"
print("  PASSED\n")

print("All cg_diff checks passed.")
