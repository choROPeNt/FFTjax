"""
Validation script for the strain-based mixed-BC solver
(``solvers.mechanical.strain_nw_cg.dstrain_nw_cg_mixed``).

This solver (the Green's-operator DC-block override trick from FFTMAD's
``variational_nw_cg_*`` family) is only a valid one-shot CG solve for
HOMOGENEOUS materials — see the docstring for why. Cases 1-2 validate that
regime. Case 3 is a permanent regression guard for the known limitation
itself: if someone changes the implementation such that it becomes
(surprisingly) symmetric for heterogeneous materials too, this test should
be revisited — until then it documents that the asymmetry is expected, not
an oversight, so ``ddisp_nw_cg`` remains the correct choice for
heterogeneous mixed-BC problems.

Usage
-----
    python test/test_strain_nw_cg_mixed.py
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
sys.path.insert(0, "src")

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)
import jax
import jax.numpy as jnp
import numpy as np

from mat_models.elastic          import LinearElasticIsotropic, assemble_C_field
from operators.green              import build_freq_grid, build_green_operator
from solvers.mechanical.strain_nw_cg import solve_elastic, dstrain_nw_cg_mixed

jax.config.update("jax_enable_x64", True)

n  = (16, 16, 16)
L  = (1.0, 1.0, 1.0)
Nv = int(np.prod(n))

mat = LinearElasticIsotropic(E=210e3, nu=0.3, name="steel")
phase = jnp.zeros(Nv, dtype=int)
C_field = assemble_C_field([mat], phase)
xi_flat = build_freq_grid(n, L)
G_glob  = build_green_operator(xi_flat, mat.lam, mat.mu)

# ── Case 1 — pure strain BC, must match solve_elastic exactly ────────────────
print("── Case 1: pure strain BC (homogeneous) ─────────────────────────────")

eps_bar = jnp.array([
    [1.0e-3, 0.0, 0.0],
    [0.0,    0.0, 0.0],
    [0.0,    0.0, 0.0],
])
control_zero = ((0, 0, 0), (0, 0, 0), (0, 0, 0))
stress_goal_zero = jnp.zeros((3, 3))

eps_ref, sigma_ref, _, it_ref, conv_ref = solve_elastic(
    n, C_field, G_glob, eps_bar, toler_lin=1e-10, maxiter=2000
)
eps_m, sigma_m, delta_m, eps_bar_out, it_m, conv_m = dstrain_nw_cg_mixed(
    n, C_field, G_glob, eps_bar, control_zero, stress_goal_zero, toler_lin=1e-10, maxiter=2000
)

err_eps = float(jnp.max(jnp.abs(eps_m - eps_ref)))
print(f"max |eps_mixed - eps_ref| = {err_eps:.3e}")
assert err_eps < 1e-10
assert jnp.allclose(eps_bar_out, eps_bar)
print("PASSED\n")

# ── Case 2 — mixed BC, homogeneous uniaxial stress vs. closed form ───────────
print("── Case 2: mixed BC, uniaxial stress (homogeneous) ──────────────────")

sigma_11_goal = 100.0
control = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
eps_bar_guess = jnp.zeros((3, 3))
stress_goal = jnp.array([
    [sigma_11_goal, 0.0, 0.0],
    [0.0,           0.0, 0.0],
    [0.0,           0.0, 0.0],
])

eps_m, sigma_m, delta_m, eps_bar_out, it_m, conv_m = dstrain_nw_cg_mixed(
    n, C_field, G_glob, eps_bar_guess, control, stress_goal, toler_lin=1e-10, maxiter=2000
)

eps_11_analytic = sigma_11_goal / mat.E
eps_22_analytic = -mat.nu * sigma_11_goal / mat.E

print(f"CG iterations : {int(it_m)} converged={bool(conv_m)}")
print(f"eps_11: solved={float(eps_bar_out[0,0]):.6e}  analytic={eps_11_analytic:.6e}")
print(f"eps_22: solved={float(eps_bar_out[1,1]):.6e}  analytic={eps_22_analytic:.6e}")

assert abs(float(eps_bar_out[0, 0]) - eps_11_analytic) < 1e-8
assert abs(float(eps_bar_out[1, 1]) - eps_22_analytic) < 1e-8
assert abs(float(eps_bar_out[2, 2]) - eps_22_analytic) < 1e-8
print("PASSED\n")

# ── Case 3 — known limitation: heterogeneous + mixed BC is NOT symmetric ─────
print("── Case 3: heterogeneous mixed BC — documents the known limitation ──")

mat2 = LinearElasticIsotropic(E=70e3, nu=0.33, name="aluminium")
n_het  = (8, 8, 8)
Nv_het = int(np.prod(n_het))
rng = np.random.default_rng(0)
phase_het = jnp.array(rng.integers(0, 2, size=Nv_het))
C_field_het = assemble_C_field([mat, mat2], phase_het)
xi_flat_het = build_freq_grid(n_het, L)
lam0 = 0.5 * (mat.lam + mat2.lam)
mu0  = 0.5 * (mat.mu + mat2.mu)
G_glob_het = build_green_operator(xi_flat_het, lam0, mu0)

I2  = jnp.eye(3)
I4s = 0.5 * (jnp.einsum("ik,jl->ijkl", I2, I2) + jnp.einsum("il,jk->ijkl", I2, I2))
control_arr = jnp.array(control, dtype=float)
G_mixed_het = G_glob_het.at[:, :, :, :, 0].set(jnp.einsum("ijkl,kl->ijkl", I4s, control_arr))

def fft_(x):
    s = x.shape
    return jnp.fft.fftn(x.reshape(s[:-1] + n_het), axes=(-3, -2, -1)).reshape(s)

def ifft_(x):
    s = x.shape
    return jnp.fft.ifftn(x.reshape(s[:-1] + n_het), axes=(-3, -2, -1)).real.reshape(s)

def A_op(v_flat):
    v   = v_flat.reshape(3, 3, Nv_het)
    Cv  = jnp.einsum("ijklm,klm->ijm", C_field_het, v)
    GCv = jnp.einsum("ijklm,klm->ijm", G_mixed_het, fft_(Cv))
    return ifft_(GCv).reshape(-1)

key1, key2 = jax.random.PRNGKey(0), jax.random.PRNGKey(1)
N = 9 * Nv_het
v1 = jax.random.normal(key1, (N,))
v2 = jax.random.normal(key2, (N,))
lhs = jnp.dot(v1, A_op(v2))
rhs = jnp.dot(A_op(v1), v2)
relerr = float(jnp.abs(lhs - rhs) / jnp.maximum(jnp.abs(lhs), jnp.abs(rhs)))
print(f"operator symmetry relative error (heterogeneous): {relerr:.3e}")

assert relerr > 0.1   # expected to be badly asymmetric -- this is the known limitation
print("Confirmed: DC-override operator is not symmetric for heterogeneous "
      "materials, as documented. Use ddisp_nw_cg for this case.")
print("PASSED")
