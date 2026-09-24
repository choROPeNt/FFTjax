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

Sweeps voxels per side (``N_SWEEP``) at fixed phi/r_fiber/size_in_r -- a
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
import resource
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
N_SWEEP   = [16, 24, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256, 288, 320,
             352, 384, 416, 448, 480, 512, 544, 576, 608, 640]
# voxels per side, coarse -> fine -- ~1.1x/step, never 2x+ (unlike the old
# [..., 256, 512, 768], whose 512->768 doubling jumped straight past the OOM
# boundary), so the memory-vs-Nv curve has enough points to show a trend.

E_MATRIX, NU_MATRIX = 3.76e3, 0.39
E_FIBER,  NU_FIBER  = 70.0e3, 0.20
GAMMA     = 5.0e-3      # fixed macroscopic shear strain probed at every grid size
FD_H      = 1.0         # MPa, finite-difference step on E_matrix
TOLER_LIN = 1e-6
MAXITER   = 500
REPEATS   = 20
OUT_DIR   = "output_/benchmark/autodiff_vs_fd_scaling"


def lame(E, nu):
    return E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)), E / (2.0 * (1.0 + nu))


def build_solve_fn(n, L, phase, xi_flat):
    """
    Returns tau_xy(E_matrix) -- homogenized shear stress from a Lippmann-
    Schwinger CG solve on this fixed RVE/grid, as a plain differentiable
    function of the matrix modulus. Same construction as
    notebooks/archiv/lin-elastic_gradient-check.ipynb's solve_elastic_diff
    and notebooks/inverse_calibration/lin-elastic_inverse-calibration.ipynb's solve_elastic_diff
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


def host_peak_rss_mb() -> float:
    """Peak resident set size so far in this process, in MB. Same as
    benchmark_3/elastic_solve_vtu.py's helper -- but note this script runs
    the whole N_SWEEP in one process (no per-grid subprocess isolation), so
    this is a running peak across all grid sizes tried so far, not a
    per-grid-size peak."""
    ru_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru_maxrss / (1024 * 1024) if sys.platform == "darwin" else ru_maxrss / 1024


def compiled_device_mem_mb(jitted_fn, *args) -> dict | None:
    """Deterministic, compile-time device memory estimate (argument + output
    + temp - alias, from XLA's buffer-assignment analysis) -- unlike a
    runtime memory_stats() reading, which is a running high-water mark
    shared across the whole process (autodiff vs. fd only used to diverge
    once fd happened to exceed whatever autodiff already set), this depends
    only on the program itself, so the two are directly comparable at every
    grid size, and it's available even where *execution* would later OOM.
    None if the backend doesn't implement memory_analysis (e.g. CPU) or its
    output lacks the expected fields."""
    try:
        stats = jitted_fn.lower(*args).compile().memory_analysis()
    except Exception:
        return None
    try:
        argument, output = stats.argument_size_in_bytes, stats.output_size_in_bytes
        temp, alias = stats.temp_size_in_bytes, stats.alias_size_in_bytes
    except AttributeError:
        return None
    mb = 1024.0 * 1024.0
    return {
        "device_mb": (argument + output + temp - alias) / mb,
        "argument_mb": argument / mb, "output_mb": output / mb,
        "temp_mb": temp / mb, "alias_mb": alias / mb,
    }


def mem_snapshot(n, stage: str, jitted_fn, *args) -> dict:
    """Host peak-so-far (process-wide running peak, see host_peak_rss_mb)
    plus jitted_fn's own compile-time device estimate (see
    compiled_device_mem_mb) -- printed immediately and also returned for
    the JSON payload, same pattern as benchmark_3's mem_snapshot."""
    host_mb = host_peak_rss_mb()
    dev = compiled_device_mem_mb(jitted_fn, *args)
    dev_mb = dev["device_mb"] if dev is not None else None
    dev_s = f"{dev_mb:.1f}" if dev_mb is not None else "n/a"
    print(f"    [grid {list(n)}] after {stage}: host {host_mb:.1f} MB peak, "
          f"device {dev_s} MB (compiled estimate)")
    return {"host_peak_rss_mb": host_mb, "device_peak_mb": dev_mb, "device_mem_breakdown": dev}


def is_oom_error(exc: Exception) -> bool:
    """True if exc looks like a device/host memory-exhaustion failure (XLA's
    RESOURCE_EXHAUSTED, or a plain Python MemoryError on CPU) rather than
    some other bug -- so we only skip FD on genuine OOM and let anything
    else propagate."""
    if isinstance(exc, MemoryError):
        return True
    msg = str(exc).lower()
    return "resource_exhausted" in msg or "out of memory" in msg or "oom" in msg


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


def bench_one_grid(n_vox, run_fd=True):
    """n_vox: target voxels per side; converted to a target dx via the fixed
    physical domain size (size_in_r * r_fiber) before calling
    make_random_composite_rve, which only takes dx directly.

    ``run_fd`` lets the caller stop attempting FD once it has already OOM'd
    at a smaller grid in this sweep -- memory use only grows with Nv, so
    retrying FD at larger sizes would just OOM again -- while autodiff (one
    pass, roughly a third of FD's peak memory) keeps running as far as it
    can.
    """
    dx = SIZE_IN_R * R_FIBER / n_vox
    phase_np, n, L, phi_act, centres = make_random_composite_rve(
        phi=PHI, r_fiber=R_FIBER, dx=dx, size_in_r=SIZE_IN_R, nz=10, K=15, seed=SEED,
    )
    Nv = int(np.prod(n))
    phase = jnp.array(phase_np.reshape(-1))
    xi_flat = build_freq_grid(n, L)
    tau_xy = build_solve_fn(n, L, phase, xi_flat)

    value_and_grad_autodiff = jax.jit(lambda E: jax.jvp(tau_xy, (E,), (1.0,)))

    result = {
        "dx": dx, "grid_n": list(n), "Nv": Nv, "fiber_volume_fraction": phi_act,
        "autodiff_status": "ok", "fd_status": "skipped",
        "value": None, "grad_autodiff": None,
        "autodiff_ms": None, "autodiff_ms_std": None,
        "grad_fd": None, "rel_grad_diff": None,
        "fd_ms": None, "fd_ms_std": None,
        "wall_time_ratio": None,
        "mem_mb": {"after_autodiff": None, "after_fd": None},
        "host_peak_rss_mb": None, "device_peak_mb": None,
    }

    def _finalize(stage_mem: dict) -> dict:
        """Top-level host/device peak mirrors the most recent mem snapshot --
        same fields benchmark_3/elastic_solve_vtu.py reports per result, kept
        here too for the same "at a glance, how close to OOM did this grid
        get" read without having to dig into mem_mb."""
        result["host_peak_rss_mb"] = stage_mem["host_peak_rss_mb"]
        result["device_peak_mb"] = stage_mem["device_peak_mb"]
        return result

    # compile once (outside timing), then time REPEATS calls to the already-compiled function
    try:
        out = value_and_grad_autodiff(E_MATRIX)
        jax.block_until_ready(out)
        autodiff_ms, autodiff_std, autodiff_out = time_ms(lambda: value_and_grad_autodiff(E_MATRIX), REPEATS)
        val_ad, grad_ad = autodiff_out
    except Exception as e:
        if not is_oom_error(e):
            raise
        result["mem_mb"]["after_autodiff"] = mem_snapshot(n, "autodiff OOM", value_and_grad_autodiff, E_MATRIX)
        print(f"  !! Autodiff OOM at Nv={Nv} (grid {tuple(n)}): {e.__class__.__name__}: "
              f"{str(e).splitlines()[0][:200]} -- stopping sweep (this grid exceeds device "
              f"memory even for the cheaper autodiff pass, so FD would too)")
        result["autodiff_status"] = "oom"
        result["fd_status"] = "skipped"
        return _finalize(result["mem_mb"]["after_autodiff"])

    result["value"] = float(val_ad)
    result["grad_autodiff"] = float(grad_ad)
    result["autodiff_ms"] = autodiff_ms
    result["autodiff_ms_std"] = autodiff_std
    result["mem_mb"]["after_autodiff"] = mem_snapshot(n, "autodiff", value_and_grad_autodiff, E_MATRIX)

    if not run_fd:
        return _finalize(result["mem_mb"]["after_autodiff"])

    value_and_grad_fd = jax.jit(
        lambda E: (tau_xy(E), (tau_xy(E + FD_H) - tau_xy(E - FD_H)) / (2.0 * FD_H))
    )
    try:
        out = value_and_grad_fd(E_MATRIX)
        jax.block_until_ready(out)
        fd_ms, fd_std, fd_out = time_ms(lambda: value_and_grad_fd(E_MATRIX), REPEATS)
        val_fd, grad_fd = fd_out
    except Exception as e:
        if not is_oom_error(e):
            raise
        result["mem_mb"]["after_fd"] = mem_snapshot(n, "fd OOM", value_and_grad_fd, E_MATRIX)
        print(f"  !! FD OOM at Nv={Nv} (grid {tuple(n)}): {e.__class__.__name__}: "
              f"{str(e).splitlines()[0][:200]} -- skipping FD for larger grids, "
              f"continuing autodiff-only")
        result["fd_status"] = "oom"
        return _finalize(result["mem_mb"]["after_fd"])

    result["fd_status"] = "ok"
    result["grad_fd"] = float(grad_fd)
    result["rel_grad_diff"] = abs(float(grad_ad) - float(grad_fd)) / abs(float(grad_fd))
    result["fd_ms"] = fd_ms
    result["mem_mb"]["after_fd"] = mem_snapshot(n, "fd", value_and_grad_fd, E_MATRIX)
    result["fd_ms_std"] = fd_std
    result["wall_time_ratio"] = fd_ms / autodiff_ms
    return _finalize(result["mem_mb"]["after_fd"])


def fmt_dev_mem(mem_stage: dict | None) -> str:
    """'n/a' when the stage never ran (skipped) or the backend doesn't
    implement memory_analysis (CPU) -- see compiled_device_mem_mb's docstring."""
    if mem_stage is None or mem_stage["device_peak_mb"] is None:
        return "n/a"
    return f"{mem_stage['device_peak_mb']:.1f}"


if __name__ == "__main__":
    print(f"JAX backend: {jax.default_backend()}   Devices: {jax.devices()}")
    print(f"phi={PHI}  r_fiber={R_FIBER}  size_in_r={SIZE_IN_R}  gamma={GAMMA}  E_matrix={E_MATRIX}")
    print(f"{'Nv':>8}  {'grid':>14}  {'autodiff_ms':>16}  {'fd_ms':>16}  "
          f"{'ratio':>7}  {'rel_grad_diff':>13}  {'ad_dev_mb':>10}  {'fd_dev_mb':>10}")

    results = []
    fd_alive = True
    fd_broke_at_Nv = None
    autodiff_broke_at_n_vox = None
    for n_vox in N_SWEEP:
        r = bench_one_grid(n_vox, run_fd=fd_alive)
        results.append(r)

        if r["autodiff_status"] == "oom":
            autodiff_broke_at_n_vox = n_vox
            break
        if r["fd_status"] == "oom":
            fd_alive = False
            fd_broke_at_Nv = r["Nv"]

        autodiff_cell = f"{r['autodiff_ms']:.2f}+/-{r['autodiff_ms_std']:.2f}"
        if r["fd_status"] == "ok":
            fd_cell = f"{r['fd_ms']:.2f}+/-{r['fd_ms_std']:.2f}"
            ratio_cell = f"{r['wall_time_ratio']:.2f}x"
            reldiff_cell = f"{r['rel_grad_diff']:.2e}"
        else:
            fd_cell = r["fd_status"].upper()   # "OOM" or "SKIPPED"
            ratio_cell = "--"
            reldiff_cell = "--"
        autodiff_mem_cell = fmt_dev_mem(r["mem_mb"]["after_autodiff"])
        fd_mem_cell = fmt_dev_mem(r["mem_mb"]["after_fd"])
        print(f"{r['Nv']:>8}  {str(tuple(r['grid_n'])):>14}  {autodiff_cell:>16}  "
              f"{fd_cell:>16}  {ratio_cell:>7}  {reldiff_cell:>13}  "
              f"{autodiff_mem_cell:>10}  {fd_mem_cell:>10}")

    if fd_broke_at_Nv is not None:
        print(f"\nFD hit OOM at Nv={fd_broke_at_Nv}; skipped for all larger grids in the "
              f"sweep. Autodiff ran for the full sweep.")
    if autodiff_broke_at_n_vox is not None:
        print(f"\nAutodiff hit OOM at n_vox={autodiff_broke_at_n_vox}; sweep stopped there "
              f"({len(results)}/{len(N_SWEEP)} grid sizes attempted).")

    today = datetime.date.today().isoformat()
    payload = {
        "generated": today,
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "phi": PHI, "r_fiber": R_FIBER, "size_in_r": SIZE_IN_R, "seed": SEED,
        "gamma": GAMMA, "E_matrix": E_MATRIX, "fd_h": FD_H,
        "toler_lin": TOLER_LIN, "maxiter": MAXITER, "repeats": REPEATS,
        "fd_broke_at_Nv": fd_broke_at_Nv,
        "autodiff_broke_at_n_vox": autodiff_broke_at_n_vox,
        "results": results,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    json_path = os.path.join(OUT_DIR, f"results_{today}.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {len(results)} results to {json_path}")

    autodiff_ok = [r for r in results if r["autodiff_status"] == "ok"]
    Nvs = [r["Nv"] for r in autodiff_ok]
    autodiff_ms = [r["autodiff_ms"] for r in autodiff_ok]
    autodiff_dev_mb = [r["mem_mb"]["after_autodiff"]["device_peak_mb"] for r in autodiff_ok]
    fd_ok = [r for r in results if r["fd_status"] == "ok"]
    fd_Nvs = [r["Nv"] for r in fd_ok]
    fd_ms = [r["fd_ms"] for r in fd_ok]
    fd_dev_mb = [r["mem_mb"]["after_fd"]["device_peak_mb"] for r in fd_ok]
    ratios = [r["wall_time_ratio"] for r in fd_ok]
    have_device_mem = any(v is not None for v in autodiff_dev_mb + fd_dev_mb)

    fig, axes = plt.subplots(1, 3 if have_device_mem else 2, figsize=(16 if have_device_mem else 11, 4.5))

    axes[0].loglog(Nvs, autodiff_ms, "o-", label="jax.jacfwd (1 pass)")
    axes[0].loglog(fd_Nvs, fd_ms, "s-", label="finite differences (3 passes)")
    if fd_broke_at_Nv is not None:
        axes[0].axvline(fd_broke_at_Nv, color="gray", ls=":", lw=1.0, label="FD OOM")
    axes[0].set_xlabel(r"grid size $N_v$")
    axes[0].set_ylabel("wall time [ms]")
    axes[0].set_title("Absolute cost per (value, gradient) call")
    axes[0].legend()
    axes[0].grid(True, which="both", linewidth=0.5, alpha=0.5)

    axes[1].semilogx(fd_Nvs, ratios, "o-", color="C2")
    axes[1].axhline(3.0, color="gray", ls="--", lw=1.0, label="3x (equal-cost-per-solve floor)")
    if fd_broke_at_Nv is not None:
        axes[1].axvline(fd_broke_at_Nv, color="gray", ls=":", lw=1.0, label="FD OOM")
    axes[1].set_xlabel(r"grid size $N_v$")
    axes[1].set_ylabel("wall-time ratio (finite diff / autodiff)")
    axes[1].set_title("Autodiff's speedup grows with problem size")
    axes[1].legend()
    axes[1].grid(True, which="both", linewidth=0.5, alpha=0.5)

    if have_device_mem:
        # Compile-time estimate per grid size (see compiled_device_mem_mb),
        # so autodiff's and FD's curves are directly comparable at every
        # size, not just where FD happens to exceed a shared running peak.
        axes[2].loglog(Nvs, autodiff_dev_mb, "o-", label="jax.jacfwd (1 pass)")
        axes[2].loglog(fd_Nvs, fd_dev_mb, "s-", label="finite differences (3 passes)")
        if fd_broke_at_Nv is not None:
            axes[2].axvline(fd_broke_at_Nv, color="gray", ls=":", lw=1.0, label="FD OOM")
        if autodiff_broke_at_n_vox is not None and results and results[-1]["autodiff_status"] == "oom":
            last_mem = results[-1]["mem_mb"]["after_autodiff"]
            if last_mem["device_peak_mb"] is not None:
                axes[2].axvline(results[-1]["Nv"], color="firebrick", ls=":", lw=1.0, label="autodiff OOM")
        axes[2].set_xlabel(r"grid size $N_v$")
        axes[2].set_ylabel("device memory, compiled estimate [MB]")
        axes[2].set_title("GPU/TPU memory footprint")
        axes[2].legend()
        axes[2].grid(True, which="both", linewidth=0.5, alpha=0.5)

    fig.suptitle("jax.jacfwd vs. finite differences across RVE discretizations")
    fig.tight_layout()
    plot_path = os.path.join(OUT_DIR, f"scaling_{today}.png")
    fig.savefig(plot_path, dpi=150)
    print(f"Wrote plot to {plot_path}")
