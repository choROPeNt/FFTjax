"""
Standalone test for solve_lippmann_schwinger
(solvers/elliptic/vector/lippmann_schwinger.py).

Three checks
------------
1. Correctness invariants on a synthetic random heterogeneous problem, both
   Green's-operator schemes -- independent of any other solver
   implementation (solvers.mechanical.strain_nw_cg, this module's former
   parity reference, is retired):
   (a) mean(eps) == eps_bar exactly. Gamma0Operator is zero at the DC
       (zero-frequency) mode by construction (see green.py's n_hat comment),
       so A(v) can't see v's DC component at all -- CG starts at x0=0 and
       never moves in that direction, so delta's mean is exactly 0.
   (b) the returned delta actually solves A(delta) = b to toler_lin,
       recomputed independently of cg_solve's own internal convergence
       tracking -- this would catch a bug where the returned value doesn't
       correspond to what was actually fed into the CG solve.
2. Real composite RVE (glass fiber / epoxy, same setup as
   notebooks/lin-elastic_strain.ipynb) -- converges, and reproduces the
   known tau_xy (avg) = 7.625369 MPa result.
3. LippmannSchwingerSolver (the ElasticitySolver wrapper) reproduces
   solve_lippmann_schwinger's own output exactly -- it's a thin wrapper,
   this is a trivial-but-real check that it doesn't lose or reorder fields.

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

from operators.general_functions import ddot42
from operators.green import GreenOperatorBasic, GreenOperatorWillot
from operators.projection import Gamma0Operator
from solvers.elliptic.vector.lippmann_schwinger import solve_lippmann_schwinger, LippmannSchwingerSolver


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


# ── 1. correctness invariants, both schemes ──────────────────────────────────

for green_op in (GreenOperatorBasic(n, L, lam0, mu0), GreenOperatorWillot(n, L, lam0, mu0, dx)):
    eps, sigma, delta, converged = solve_lippmann_schwinger(
        n, C_field, green_op, eps_bar, None, toler_lin, maxiter,
    )
    tag = type(green_op).__name__

    assert bool(converged), f"{tag}: solve must converge"
    assert jnp.allclose(jnp.mean(eps, axis=-1), eps_bar, atol=1e-10), \
        f"{tag}: mean(eps) must equal eps_bar exactly (Gamma0 is zero at DC)"

    gamma0 = Gamma0Operator(n, green_op)
    Adelta = gamma0(ddot42(C_field, delta)).reshape(-1)
    eps0 = jnp.ones((3, 3, Nv)) * eps_bar[:, :, None]
    bb = -gamma0(ddot42(C_field, eps0)).reshape(-1)
    resid = float(jnp.linalg.norm(Adelta - bb) / jnp.linalg.norm(bb))
    assert resid < toler_lin * 10, f"{tag}: A(delta) != b, relative residual {resid:.3e}"


# ── 2. real composite RVE (matches notebooks/lin-elastic_strain.ipynb) ──────

from generation.rve import make_square_composite_rve
from mat_models.elastic import LinearElasticIsotropic, assemble_C_field

phase_np, N, n_rve, L_rve, phi_act = make_square_composite_rve(
    phi=0.5, r_fiber=0.005, dx=0.0002, N_min=32, nz=10,
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


# ── 3. LippmannSchwingerSolver wraps solve_lippmann_schwinger exactly ───────
#
# allclose, not array_equal: these are two independent CG solves of the same
# problem, not two reads of one cached result -- bit-identical floats aren't
# guaranteed across separate calls on a multi-threaded CPU backend (verified
# flaky with array_equal: ~1/3 standalone runs differed at the ULP level).
# Numerical agreement well inside toler_lin is the real invariant.

solver = LippmannSchwingerSolver(n_rve, green_op_rve, toler_lin=1e-6, maxiter=1000)
result = solver.solve(C_field_rve, eps_bar)

assert bool(result.converged) == bool(converged)
assert jnp.allclose(result.eps, eps, atol=1e-12), "LippmannSchwingerSolver.eps must match solve_lippmann_schwinger"
assert jnp.allclose(result.sigma, sigma, atol=1e-8), "LippmannSchwingerSolver.sigma must match solve_lippmann_schwinger"
assert jnp.allclose(result.delta, delta, atol=1e-12), "LippmannSchwingerSolver.delta must match solve_lippmann_schwinger"

print("test_elliptic_vector_lippmann_schwinger: all checks passed")
print(f"  composite RVE tau_xy (avg) = {tau_xy:.6f} MPa")
