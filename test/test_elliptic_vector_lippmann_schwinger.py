"""
Standalone test for solve_lippmann_schwinger
(solvers/elliptic/vector/lippmann_schwinger.py).

Three checks
------------
1. Parity with dstrain_nw_cg on a synthetic random problem, standard scheme
   (GreenOperatorBasic) -- eps/sigma/delta/converged must match exactly,
   since both build the identical linear system, just via different
   plumbing (LinearOperator composition vs. hand-rolled einsum/FFT).
2. Same, rotated scheme (GreenOperatorWillot).
3. Real composite RVE (glass fiber / epoxy, same setup as
   notebooks/lin-elastic_strain.ipynb) -- converges, and reproduces the
   known tau_xy (avg) = 7.625 MPa result.

Usage
-----
    python -m pytest test/test_elliptic_vector_lippmann_schwinger.py
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
sys.path.insert(0, "src")

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)
import jax.numpy as jnp
import numpy as np

from operators.green import GreenOperatorBasic, GreenOperatorWillot
from solvers.mechanical.strain_nw_cg import dstrain_nw_cg
from solvers.elliptic.vector.lippmann_schwinger import solve_lippmann_schwinger


rng = np.random.default_rng(2)

n = (4, 4, 4)
L = (1.0, 1.0, 1.0)
dx = tuple(Li / ni for Li, ni in zip(L, n))
lam0, mu0 = 60.0, 30.0
Nv = int(np.prod(n))

# per-voxel major-symmetric positive-definite-ish stiffness (random, not
# physical -- only needs to be a valid 4th-order tensor for A_op to be
# well-posed for CG)
I2 = jnp.eye(3)
I4s = 0.5 * (jnp.einsum('ik,jl->ijkl', I2, I2) + jnp.einsum('il,jk->ijkl', I2, I2))
IxI = jnp.einsum('ij,kl->ijkl', I2, I2)
lam_field = jnp.asarray(rng.uniform(20.0, 80.0, size=(Nv,)))
mu_field = jnp.asarray(rng.uniform(10.0, 50.0, size=(Nv,)))
C_field = (lam_field[None, None, None, None, :] * IxI[..., None]
           + 2.0 * mu_field[None, None, None, None, :] * I4s[..., None])

eps_bar = jnp.array([
    [0.0, 1.0e-3, 0.0],
    [1.0e-3, 0.0, 0.0],
    [0.0, 0.0, 0.0],
])

toler_lin, maxiter = 1e-8, 1000


# ── 1. parity, standard scheme ───────────────────────────────────────────────

green_op = GreenOperatorBasic(n, L, lam0, mu0)

eps_ref, sigma_ref, delta_ref, conv_ref = dstrain_nw_cg(
    n, C_field, green_op.G, eps_bar, None, toler_lin, maxiter,
)
eps_new, sigma_new, delta_new, conv_new = solve_lippmann_schwinger(
    n, C_field, green_op, eps_bar, None, toler_lin, maxiter,
)

assert bool(conv_ref) and bool(conv_new), "both solves must converge"
assert jnp.allclose(eps_new, eps_ref, atol=1e-10), "eps mismatch (standard scheme)"
assert jnp.allclose(sigma_new, sigma_ref, atol=1e-6), "sigma mismatch (standard scheme)"
assert jnp.allclose(delta_new, delta_ref, atol=1e-10), "delta mismatch (standard scheme)"


# ── 2. parity, rotated (Willot) scheme ───────────────────────────────────────

green_op_w = GreenOperatorWillot(n, L, lam0, mu0, dx)

eps_ref, sigma_ref, delta_ref, conv_ref = dstrain_nw_cg(
    n, C_field, green_op_w.G, eps_bar, None, toler_lin, maxiter,
)
eps_new, sigma_new, delta_new, conv_new = solve_lippmann_schwinger(
    n, C_field, green_op_w, eps_bar, None, toler_lin, maxiter,
)

assert bool(conv_ref) and bool(conv_new), "both solves must converge (rotated)"
assert jnp.allclose(eps_new, eps_ref, atol=1e-10), "eps mismatch (rotated scheme)"
assert jnp.allclose(sigma_new, sigma_ref, atol=1e-6), "sigma mismatch (rotated scheme)"
assert jnp.allclose(delta_new, delta_ref, atol=1e-10), "delta mismatch (rotated scheme)"


# ── 3. real composite RVE (matches notebooks/lin-elastic_strain.ipynb) ──────

from generation.rve import make_square_composite_rve
from mat_models.elastic import LinearElasticIsotropic, assemble_C_field

phase_np, N, n_rve, L_rve, phi_act = make_square_composite_rve(
    phi=0.5, r_fiber=0.005, spacing=0.0002, N_min=32, nz=10,
)
dx_rve = tuple(Li / ni for Li, ni in zip(L_rve, n_rve))
phase = jnp.array(phase_np.reshape(-1))

matrix = LinearElasticIsotropic(E=3.0e3, nu=0.35, name="epoxy matrix")
fiber = LinearElasticIsotropic(E=70.0e3, nu=0.20, name="glass fiber")
C_field_rve = assemble_C_field([matrix, fiber], phase)

lam0_rve = 0.5 * (matrix.lam + fiber.lam)
mu0_rve = 0.5 * (matrix.mu + fiber.mu)
green_op_rve = GreenOperatorWillot(n_rve, L_rve, lam0_rve, mu0_rve, dx_rve)

eps, sigma, delta, converged = solve_lippmann_schwinger(
    n_rve, C_field_rve, green_op_rve, eps_bar, None, toler_lin=1e-6, maxiter=1000,
)

tau_xy = float(jnp.mean(sigma[1, 0]))
assert bool(converged), "composite RVE solve must converge"
assert abs(tau_xy - 7.625369073063829) < 1e-6, f"tau_xy mismatch: got {tau_xy}, expected ~7.625369"

print("test_elliptic_vector_lippmann_schwinger: all checks passed")
print(f"  composite RVE tau_xy (avg) = {tau_xy:.6f} MPa")
