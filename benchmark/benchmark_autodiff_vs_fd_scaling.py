"""
Grid-size (Nv) scaling benchmark: jax.jacfwd (forward-mode autodiff) vs.
central finite differences for computing (value, gradient) of a scalar
elastic-solve output, swept across increasingly fine discretizations of the
same random-fibre RVE. Companion to notebooks/lin-elastic_inverse-
calibration.ipynb's own single-grid-size comparison (its "Comparison:
old-school gradients" section) -- this extends that finding across problem
size instead of fixing it at one grid.

Forward-mode gets the loss *and* the gradient from one augmented forward
pass through the solve; central differences need three independent solves
(center, +h, -h) for the same pair. The wall-clock ratio between the two
should climb toward that 3x floor as Nv grows: small grids are
launch/dispatch-overhead-bound (fixed per-call cost dominates, diluting the
real difference), while large grids are compute-bound, where "3x the linear
solves" shows through directly. See notes/AUTODIFF_SOLVE.md for why
forward-mode -- not jax.grad -- is the differentiable tool used here (the
production `cg_solve`'s reverse-mode gradient is silently wrong on this
project's structurally singular operators; its forward-mode gradient is
correct).

Sweeps voxel size (``DX_SWEEP``) at fixed phi/r_fiber/size_in_r -- a
mesh-refinement sweep of the *same* physical RVE, so Nv is the only thing
that changes between runs, matching benchmark_vmap_batch_scaling.py's
"isolate one variable" convention.

Usage
-----
    python benchmark/benchmark_autodiff_vs_fd_scaling.py
"""
import sys
sys.path.insert(0, "src")

import json
import os
import time
import datetime

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from generation.rve import make_random_composite_rve
from operators.green import build_freq_grid, build_green_operator
from solvers.krylov.cg import cg_solve

PHI       = 0.35
R_FIBER   = 0.0035
SIZE_IN_R = 10
SEED      = 67
DX_SWEEP  = [0.0009, 0.0006, 0.0004, 0.0003, 0.00022, 0.00017]  # coarse -> fine

E_MATRIX, NU_MATRIX = 3.76e3, 0.39
E_FIBER,  NU_FIBER  = 70.0e3, 0.20
GAMMA     = 5.0e-3      # fixed macroscopic shear strain probed at every grid size
FD_H      = 1.0         # MPa, finite-difference step on E_matrix
TOLER_LIN = 1e-6
MAXITER   = 500
REPEATS   = 5
OUT_DIR   = "output/benchmark/autodiff_vs_fd_scaling"


def lame(E, nu):
    return E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)), E / (2.0 * (1.0 + nu))


def build_solve_fn(n, L, phase, xi_flat):
    """
    Returns tau_xy(E_matrix) -- homogenized shear stress from a Lippmann-
    Schwinger CG solve on this fixed RVE/grid, as a plain differentiable
    function of the matrix modulus. Same construction as
    notebooks/archiv/lin-elastic_gradient-check.ipynb's solve_elastic_diff
    and notebooks/lin-elastic_inverse-calibration.ipynb's solve_elastic_diff
    -- reused here rather than imported, since both of those close over
    notebook-local state (phase, n, L, ...) that this function takes as
    explicit arguments instead, one instance per grid size in the sweep.
    """
    Nv = int(np.prod(n))
    dx_vox = tuple(Li / ni for Li, ni in zip(L, n))
    I2 = jnp.eye(3)

    def stiffness(lam, mu):
        return (lam * jnp.einsum('ij,kl->ijkl', I2, I2)
                + mu * (jnp.einsum('ik,jl->ijkl', I2, I2) + jnp.einsum('il,jk->ijkl', I2, I2)))

    def fft_(x):
        s = x.shape
        return jnp.fft.fftn(x.reshape(s[:-1] + n), axes=(-3, -2, -1)).reshape(s)

    def ifft_(x):
        s = x.shape
        return jnp.fft.ifftn(x.reshape(s[:-1] + n), axes=(-3, -2, -1)).real.reshape(s)

    lam_f, mu_f = lame(E_FIBER, NU_FIBER)
    C_fiber = stiffness(lam_f, mu_f)

    def tau_xy(E_matrix, gamma=GAMMA):
        lam_m, mu_m = lame(E_matrix, NU_MATRIX)
        C_matrix = stiffness(lam_m, mu_m)
        C_field = jnp.where(phase[None, None, None, None, :] == 0, C_matrix[..., None], C_fiber[..., None])

        lam0 = 0.5 * (lam_m + lam_f)
        mu0 = 0.5 * (mu_m + mu_f)
        G_glob = build_green_operator(xi_flat, lam0, mu0, scheme="rotated", dx=dx_vox)

        eps_bar = jnp.array([[0., gamma / 2, 0.], [gamma / 2, 0., 0.], [0., 0., 0.]])

        def A_op(v_flat):
            v = v_flat.reshape(3, 3, Nv)
            Cv = jnp.einsum("ijklm,klm->ijm", C_field, v)
            GCv = jnp.einsum("ijklm,klm->ijm", G_glob, fft_(Cv))
            return ifft_(GCv).reshape(-1)

        eps0 = jnp.ones((3, 3, Nv)) * eps_bar[:, :, None]
        sigma0 = jnp.einsum("ijklm,klm->ijm", C_field, eps0)
        bb = -ifft_(jnp.einsum("ijklm,klm->ijm", G_glob, fft_(sigma0))).reshape(-1)
        x0 = jnp.zeros_like(bb)
        delta_flat, _converged = cg_solve(A_op, bb, x0, TOLER_LIN, MAXITER)
        eps = eps0 + delta_flat.reshape(3, 3, Nv)
        sigma = jnp.einsum("ijklm,klm->ijm", C_field, eps)
        return jnp.mean(sigma[0, 1])

    return tau_xy


def time_ms(fn, repeats):
    """Mean/std wall-clock time over `repeats` calls, in ms -- fn's own first
    call (compilation) must already have happened before this is called."""
    samples = []
    out = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        jax.block_until_ready(out)
        samples.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(samples)), float(np.std(samples)), out


def bench_one_grid(dx):
    phase_np, n, L, phi_act, centres = make_random_composite_rve(
        phi=PHI, r_fiber=R_FIBER, dx=dx, size_in_r=SIZE_IN_R, nz=1, K=15, seed=SEED,
    )
    Nv = int(np.prod(n))
    phase = jnp.array(phase_np.reshape(-1))
    xi_flat = build_freq_grid(n, L)
    tau_xy = build_solve_fn(n, L, phase, xi_flat)

    value_and_grad_autodiff = jax.jit(lambda E: jax.jvp(tau_xy, (E,), (1.0,)))
    value_and_grad_fd = jax.jit(
        lambda E: (tau_xy(E), (tau_xy(E + FD_H) - tau_xy(E - FD_H)) / (2.0 * FD_H))
    )

    # compile once (outside timing), then time REPEATS calls to the already-compiled function
    out = value_and_grad_autodiff(E_MATRIX)
    jax.block_until_ready(out)
    autodiff_ms, autodiff_std, (val_ad, grad_ad) = time_ms(lambda: value_and_grad_autodiff(E_MATRIX), REPEATS)

    out = value_and_grad_fd(E_MATRIX)
    jax.block_until_ready(out)
    fd_ms, fd_std, (val_fd, grad_fd) = time_ms(lambda: value_and_grad_fd(E_MATRIX), REPEATS)

    rel_grad_diff = abs(float(grad_ad) - float(grad_fd)) / abs(float(grad_fd))

    return {
        "dx": dx, "grid_n": list(n), "Nv": Nv, "fiber_volume_fraction": phi_act,
        "value": float(val_ad),
        "grad_autodiff": float(grad_ad), "grad_fd": float(grad_fd), "rel_grad_diff": rel_grad_diff,
        "autodiff_ms": autodiff_ms, "autodiff_ms_std": autodiff_std,
        "fd_ms": fd_ms, "fd_ms_std": fd_std,
        "wall_time_ratio": fd_ms / autodiff_ms,
    }


if __name__ == "__main__":
    print(f"JAX backend: {jax.default_backend()}   Devices: {jax.devices()}")
    print(f"phi={PHI}  r_fiber={R_FIBER}  size_in_r={SIZE_IN_R}  gamma={GAMMA}  E_matrix={E_MATRIX}")
    print(f"{'Nv':>8}  {'grid':>14}  {'autodiff_ms':>12}  {'fd_ms':>10}  "
          f"{'ratio':>7}  {'rel_grad_diff':>13}")

    results = []
    for dx in DX_SWEEP:
        r = bench_one_grid(dx)
        results.append(r)
        print(f"{r['Nv']:>8}  {str(tuple(r['grid_n'])):>14}  {r['autodiff_ms']:>12.2f}  "
              f"{r['fd_ms']:>10.2f}  {r['wall_time_ratio']:>6.2f}x  {r['rel_grad_diff']:>13.2e}")

    today = datetime.date.today().isoformat()
    payload = {
        "generated": today,
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "phi": PHI, "r_fiber": R_FIBER, "size_in_r": SIZE_IN_R, "seed": SEED,
        "gamma": GAMMA, "E_matrix": E_MATRIX, "fd_h": FD_H,
        "toler_lin": TOLER_LIN, "maxiter": MAXITER, "repeats": REPEATS,
        "results": results,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    json_path = os.path.join(OUT_DIR, f"results_{today}.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {len(results)} results to {json_path}")

    Nvs = [r["Nv"] for r in results]
    autodiff_ms = [r["autodiff_ms"] for r in results]
    fd_ms = [r["fd_ms"] for r in results]
    ratios = [r["wall_time_ratio"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].loglog(Nvs, autodiff_ms, "o-", label="jax.jacfwd (1 pass)")
    axes[0].loglog(Nvs, fd_ms, "s-", label="finite differences (3 passes)")
    axes[0].set_xlabel(r"grid size $N_v$")
    axes[0].set_ylabel("wall time [ms]")
    axes[0].set_title("Absolute cost per (value, gradient) call")
    axes[0].legend()
    axes[0].grid(True, which="both", linewidth=0.5, alpha=0.5)

    axes[1].semilogx(Nvs, ratios, "o-", color="C2")
    axes[1].axhline(3.0, color="gray", ls="--", lw=1.0, label="3x (equal-cost-per-solve floor)")
    axes[1].set_xlabel(r"grid size $N_v$")
    axes[1].set_ylabel("wall-time ratio (finite diff / autodiff)")
    axes[1].set_title("Autodiff's speedup grows with problem size")
    axes[1].legend()
    axes[1].grid(True, which="both", linewidth=0.5, alpha=0.5)

    fig.suptitle("jax.jacfwd vs. finite differences across RVE discretizations")
    fig.tight_layout()
    plot_path = os.path.join(OUT_DIR, f"scaling_{today}.png")
    fig.savefig(plot_path, dpi=150)
    print(f"Wrote plot to {plot_path}")
