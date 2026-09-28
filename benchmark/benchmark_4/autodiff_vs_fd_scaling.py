"""
Grid-size (Nv) scaling of the inverse calibration in
notebooks/inverse_calibration/lin-elastic_inverse-calibration.ipynb:
calibrate the epoxy matrix modulus E_m against the real digitized
transverse-tension curve (Varandas et al. 2020, phi = 0.35), once with
jax.jacfwd gradients and once with central finite differences, at
increasingly fine discretizations of the same random-fibre RVE.

Everything physical is the notebook's own setup, unchanged:
  * reference curve   benchmark/benchmark_2/assets/load_22_tension_phi_0.35_.csv,
                      N_POINTS evenly spaced points up to STRAIN_MAX_LINEAR,
                      plus the first point past it held out for validation
  * forward model     solve_displacement_based, uniaxial tension (xx
                      strain-controlled, yy/zz traction-free), transversely
                      isotropic carbon fibre, isotropic matrix with nu = 0.39
  * loss              sum_i (sigma_11^sim(eps_i; E_m) - sigma_11^ref(eps_i))^2
  * optimizer         calibration.calibrate.calibrate_adam from E_INIT with
                      the notebook's learning rate, fixed MAX_STEPS budget
Only the voxel count changes between grid sizes (fixed phi/r_fiber/
size_in_r/seed -- a mesh-refinement sweep of the same physical RVE).

Per grid size and gradient method (autodiff / fd) this records the full
calibration history -- E_m, loss and gradient at every optimizer step,
plus each step's wall time -- and the calibrated E_m, final loss, the
number of steps (and wall time) to reach LOSS_TOL_FACTOR x autodiff's final
loss (the notebook's "iterations to the same solution quality" criterion),
the held-out point's prediction error, compile time, and the device memory
XLA assigns to the compiled (loss, gradient) program. The latter is
compile().memory_analysis() -- the program's own buffer requirement, not
a runtime reading, so it isn't polluted by the allocator's history.

Every grid size runs in its own subprocess: JAX's GPU allocator never
returns freed regions, so in one long-lived process the regions left over
from smaller grids can make a later, larger allocation OOM well below the
device limit.

jax.jacfwd, not jax.grad: see notes/AUTODIFF_SOLVE.md (cg_solve's
while_loop is only correctly differentiable in forward mode).

Usage
-----
    python benchmark/benchmark_4/autodiff_vs_fd_scaling.py
    python benchmark/benchmark_4/autodiff_vs_fd_scaling.py --n-sweep 32 64 128 --max-steps 30

Writes <OUT_DIR>/results_<date>.json plus calibration_<date>.png (E_m/loss
histories, fit vs. reference) and scaling_<date>.png (calibrated E_m, cost
and memory vs. Nv).
"""
import sys
sys.path.insert(0, "src")

import argparse
import datetime
import json
import os
import resource
import subprocess
import time

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from calibration.calibrate import calibrate_adam
from generation.rve import make_random_composite_rve
from materialmodels.elastic.transverse_isotropic import TransverseIsotropic
from operators.green import build_freq_grid
from solvers.elliptic.vector.displacement_based import solve_displacement_based

# -- reference data (notebook: "Configuration") --------------------------------
REF_CSV           = "benchmark/benchmark_2/assets/load_22_tension_phi_0.35_.csv"
STRAIN_MAX_LINEAR = 0.002   # upper strain cutoff of the range treated as linear-elastic
N_POINTS          = 5       # calibration points, evenly spaced in that range

# -- microstructure (notebook: "Generate the composite RVE") -------------------
PHI       = 0.35
R_FIBER   = 0.0035
SIZE_IN_R = 10
NZ        = 1
SEED      = 67
# voxels per side, coarse -> fine. The notebook's own grid (dx = 0.0006 mm) is
# ~58 per side; steps stay well under 2x so the OOM boundary is bracketed.
N_SWEEP   = [32, 58, 96, 128, 192, 256, 384, 512, 768, 1024, 1280, 1536, 2048]

# -- materials (notebook: "Materials") -----------------------------------------
NU_MATRIX = 0.39
FIBER_KW  = dict(E_L=234.0e3, E_T=15.0e3, G_LT=15.0e3, nu_LT=0.20, G_TT=7.0e3)

# -- solver / optimizer (notebook: solve_elastic_diff, calibrate_adam call) ----
TOLER_LIN       = 1e-8
MAXITER         = 300
E_INIT          = 6.0e3     # deliberately far from a plausible epoxy modulus
INIT_LR         = 0.8e2
MAX_STEPS       = 90
FD_H            = 1.0       # MPa, central-difference step on E_m
LOSS_TOL_FACTOR = 3.0       # "same solution quality" = loss < 3x autodiff's final loss

OUT_DIR = "output/benchmark/benchmark_4"

CONTROL = ((0, 0, 0), (0, 1, 0), (0, 0, 1))   # xx strain-controlled, yy/zz traction-free
I2 = jnp.eye(3)


def load_reference():
    """Calibration points and held-out point, picked exactly as the notebook
    does."""
    ref = np.loadtxt(REF_CSV, delimiter=",", skiprows=1)
    strain_full, stress_full = ref[:, 0], ref[:, 1]
    in_range = np.where(strain_full <= STRAIN_MAX_LINEAR)[0]
    pick = np.unique(in_range[np.linspace(0, len(in_range) - 1, N_POINTS).round().astype(int)])
    holdout = in_range[-1] + 1
    return {
        "strain_full": strain_full, "stress_full": stress_full,
        "strain_obs": strain_full[pick], "stress_obs": stress_full[pick],
        "strain_holdout": float(strain_full[holdout]), "stress_holdout": float(stress_full[holdout]),
    }


def lame(E, nu):
    return E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)), E / (2.0 * (1.0 + nu))


def stiffness(lam, mu):
    return (lam * jnp.einsum('ij,kl->ijkl', I2, I2)
            + mu * (jnp.einsum('ik,jl->ijkl', I2, I2) + jnp.einsum('il,jk->ijkl', I2, I2)))


def build_forward(n, L, phase):
    """sigma_axial(E_m, eps_axial) -> (mean sigma_11, CG converged) -- the
    notebook's solve_elastic_diff, closed over this grid's RVE."""
    xi_flat = build_freq_grid(n, L)
    C_fiber = TransverseIsotropic(**FIBER_KW, name="carbon fiber").elastic_stiffness_tensor()
    stress_goal_zero = jnp.zeros((3, 3))

    def sigma_axial(E_matrix, eps_axial):
        C_matrix = stiffness(*lame(E_matrix, NU_MATRIX))
        C_field = jnp.where((phase == 0)[None, None, None, None, :],
                            C_matrix[..., None], C_fiber[..., None])
        eps_bar = jnp.zeros((3, 3)).at[0, 0].set(eps_axial)
        _eps, sigma, _delta, _eps_bar_out, converged = solve_displacement_based(
            n, C_field, xi_flat, eps_bar, CONTROL, stress_goal_zero,
            toler_lin=TOLER_LIN, maxiter=MAXITER,
        )
        return jnp.mean(sigma[0, 0]), converged

    return sigma_axial


def host_peak_rss_mb() -> float:
    """Peak resident set size of this (per-grid) process, in MB."""
    ru_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru_maxrss / (1024 * 1024) if sys.platform == "darwin" else ru_maxrss / 1024


def compiled_device_mem_mb(jitted_fn, *args) -> dict | None:
    """Device memory XLA assigns to jitted_fn's compiled program (argument +
    output + temp - alias, from its buffer assignment) -- depends only on the
    program, not on allocator history, and is available even where running
    it OOMs. None if the backend doesn't implement memory_analysis."""
    try:
        stats = jitted_fn.lower(*args).compile().memory_analysis()
        argument, output = stats.argument_size_in_bytes, stats.output_size_in_bytes
        temp, alias = stats.temp_size_in_bytes, stats.alias_size_in_bytes
    except Exception:
        return None
    mb = 1024.0 * 1024.0
    return {
        "device_mb": (argument + output + temp - alias) / mb,
        "argument_mb": argument / mb, "output_mb": output / mb,
        "temp_mb": temp / mb, "alias_mb": alias / mb,
    }


def device_limit_mb() -> float | None:
    """Allocator limit (XLA_PYTHON_CLIENT_MEM_FRACTION x device total) -- the
    real OOM threshold. None on CPU."""
    try:
        stats = jax.devices()[0].memory_stats() or {}
    except Exception:
        return None
    return stats["bytes_limit"] / (1024 * 1024) if "bytes_limit" in stats else None


def is_oom_error(exc: Exception) -> bool:
    if isinstance(exc, MemoryError):
        return True
    msg = str(exc).lower()
    return "resource_exhausted" in msg or "out of memory" in msg


def oom_message(exc: Exception) -> str:
    """XLA's own OOM line ("... trying to allocate N bytes"), which says how
    much the failing allocation needed -- more useful than the first line."""
    lines = str(exc).splitlines()
    hits = [ln for ln in lines if "allocat" in ln.lower()] or lines[:1]
    return hits[0].strip()[:400]


def run_calibration(value_and_grad_fn) -> dict:
    """calibrate_adam with a recording wrapper around value_and_grad_fn, so
    the gradient and wall time of every step are kept alongside the
    (E_m, loss) history calibrate_adam itself returns. Compilation happens in
    a separate warm-up call first, so step times are steady-state."""
    t0 = time.perf_counter()
    jax.block_until_ready(value_and_grad_fn(jnp.asarray(E_INIT)))
    compile_s = time.perf_counter() - t0

    grads, step_s = [], []

    def recorded(E):
        t = time.perf_counter()
        loss_val, grad_val = jax.block_until_ready(value_and_grad_fn(E))
        step_s.append(time.perf_counter() - t)
        grads.append(float(grad_val))
        return loss_val, grad_val

    t0 = time.perf_counter()
    E_final, history = calibrate_adam(recorded, param_init=E_INIT, max_steps=MAX_STEPS, init_lr=INIT_LR)
    total_s = time.perf_counter() - t0
    return {
        "status": "ok",
        "E_final": E_final,
        "loss_final": history[-1][1],
        "E_history": [h[0] for h in history],
        "loss_history": [h[1] for h in history],
        "grad_history": grads,
        "step_s": step_s,
        "step_s_mean": float(np.mean(step_s)),
        "total_s": total_s,
        "compile_s": compile_s,
    }


def steps_to_tol(run: dict, tol: float) -> tuple[int | None, float | None]:
    """First step whose loss is below tol, and the wall time to get there."""
    for i, loss_val in enumerate(run["loss_history"]):
        if loss_val < tol:
            return i + 1, float(sum(run["step_s"][: i + 1]))
    return None, None


def bench_one_grid(n_vox: int, run_fd: bool) -> dict:
    ref = load_reference()
    strain_obs, stress_obs = jnp.array(ref["strain_obs"]), jnp.array(ref["stress_obs"])

    dx = SIZE_IN_R * R_FIBER / n_vox
    phase_np, n, L, phi_act, _centres = make_random_composite_rve(
        phi=PHI, r_fiber=R_FIBER, dx=dx, size_in_r=SIZE_IN_R, nz=NZ, K=15, seed=SEED,
    )
    Nv = int(np.prod(n))
    phase = jnp.array(phase_np.reshape(-1))
    sigma_axial = build_forward(n, L, phase)

    def loss(E_matrix):
        pred = jax.vmap(lambda e: sigma_axial(E_matrix, e)[0])(strain_obs)
        return jnp.sum((pred - stress_obs) ** 2)

    def grad_fd(E_matrix):
        return (loss(E_matrix + FD_H) - loss(E_matrix - FD_H)) / (2.0 * FD_H)

    methods = {
        "autodiff": jax.jit(lambda E: (loss(E), jax.jacfwd(loss)(E))),
        "fd": jax.jit(lambda E: (loss(E), grad_fd(E))),
    }

    result = {
        "n_vox": n_vox, "dx": dx, "grid_n": list(n), "Nv": Nv,
        "fiber_volume_fraction": float(phi_act),
        "device_limit_mb": device_limit_mb(),
        "autodiff": {"status": "skipped"}, "fd": {"status": "skipped"},
    }

    for name, fn in methods.items():
        if name == "fd" and not run_fd:
            continue
        mem = compiled_device_mem_mb(fn, jnp.asarray(E_INIT))
        try:
            run = run_calibration(fn)
        except Exception as e:
            if not is_oom_error(e):
                raise
            result[name] = {"status": "oom", "error": oom_message(e), "device_mem": mem}
            print(f"  !! {name} OOM at Nv={Nv} (grid {tuple(n)}): {oom_message(e)}", file=sys.stderr)
            if name == "autodiff":
                break   # fd needs at least as much -- don't bother
            continue
        run["device_mem"] = mem
        result[name] = run

    ad, fd = result["autodiff"], result["fd"]
    if ad["status"] == "ok":
        tol = LOSS_TOL_FACTOR * ad["loss_final"]
        result["loss_tol"] = tol
        for run in (ad, fd):
            if run["status"] == "ok":
                run["steps_to_tol"], run["time_to_tol_s"] = steps_to_tol(run, tol)
        if fd["status"] == "ok":
            # both runs start at E_INIT, so step 0's gradients are directly comparable
            g_ad, g_fd = ad["grad_history"][0], fd["grad_history"][0]
            result["rel_grad_diff_at_init"] = abs(g_ad - g_fd) / abs(g_ad)
            result["step_time_ratio_fd_over_ad"] = fd["step_s_mean"] / ad["step_s_mean"]

        # validation (notebook: "Validation") and the fit curve for plotting
        stress_at = jax.jit(lambda E, eps: jax.vmap(lambda e: sigma_axial(E, e))(eps))
        E_cal = ad["E_final"]
        _, conv_obs = stress_at(E_cal, strain_obs)
        pred_holdout, _ = stress_at(E_cal, jnp.array([ref["strain_holdout"]]))
        strain_dense = jnp.linspace(0.0, ref["strain_holdout"] * 1.05, 20)
        stress_dense, _ = stress_at(E_cal, strain_dense)
        result["cg_converged_at_E_final"] = bool(jnp.all(conv_obs))
        result["holdout_pred_stress"] = float(pred_holdout[0])
        result["holdout_rel_error"] = (float(pred_holdout[0]) - ref["stress_holdout"]) / ref["stress_holdout"]
        result["fit_strain"] = np.asarray(strain_dense).tolist()
        result["fit_stress"] = np.asarray(stress_dense).tolist()

    result["host_peak_rss_mb"] = host_peak_rss_mb()
    return result


def run_grid_subprocess(n_vox: int, run_fd: bool, max_steps: int) -> dict:
    cmd = [sys.executable, __file__, "--single-grid", str(n_vox), "--max-steps", str(max_steps)]
    if not run_fd:
        cmd.append("--skip-fd")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode == 0:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    # hard crash (e.g. an OOM outside the guarded calls) -- keep the useful lines
    lines = proc.stderr.splitlines()
    hits = [ln for ln in lines if "RESOURCE_EXHAUSTED" in ln or "allocat" in ln.lower()]
    status = "oom" if hits else "crashed"
    msg = (hits or lines[-3:] or ["<no stderr>"])[0].strip()[:400]
    return {"n_vox": n_vox, "grid_n": None, "Nv": None,
            "autodiff": {"status": status, "error": msg}, "fd": {"status": "skipped"}}


def _mem(run: dict) -> str:
    mem = run.get("device_mem")
    return f"{mem['device_mb']:.0f}" if mem else "n/a"


def print_row(r: dict) -> None:
    ad, fd = r["autodiff"], r["fd"]
    grid = str(tuple(r["grid_n"])) if r["grid_n"] else "?"
    if ad["status"] != "ok":
        print(f"{r['n_vox']:>6} {grid:>16} {str(r['Nv']):>9}  autodiff {ad['status'].upper()}: {ad.get('error', '')}")
        return

    def cells(run):
        if run["status"] != "ok":
            return f"{run['status'].upper():>10} {'--':>6} {'--':>8} {'--':>10}"
        n_tol = run.get("steps_to_tol")
        return (f"{run['E_final']:>10.1f} {str(n_tol) if n_tol else '>max':>6} "
                f"{run['step_s_mean']:>8.3f} {_mem(run):>10}")

    rel = f"{r['rel_grad_diff_at_init']:.1e}" if "rel_grad_diff_at_init" in r else "--"
    print(f"{r['n_vox']:>6} {grid:>16} {r['Nv']:>9}  {cells(ad)}  {cells(fd)}  "
          f"{rel:>8} {r['holdout_rel_error']:>+9.2%}")


def plot(results: list[dict], ref: dict, today: str) -> list[str]:
    ok = [r for r in results if r["autodiff"]["status"] == "ok"]
    if not ok:
        return []
    paths = []
    cmap = plt.get_cmap("viridis")
    colors = {r["n_vox"]: cmap(i / max(len(ok) - 1, 1)) for i, r in enumerate(ok)}

    # -- calibration histories + fit (notebook: "Convergence and fit") --------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for r in ok:
        c, lab = colors[r["n_vox"]], f"Nv={r['Nv']:,}"
        axes[0].plot(r["autodiff"]["E_history"], "-", color=c, label=lab)
        axes[1].semilogy(r["autodiff"]["loss_history"], "-", color=c, label=lab)
        if r["fd"]["status"] == "ok":
            axes[0].plot(r["fd"]["E_history"], "--", color=c)
            axes[1].semilogy(r["fd"]["loss_history"], "--", color=c)
        axes[2].plot(r["fit_strain"], r["fit_stress"], "-", color=c, label=lab)
    axes[0].set(xlabel="optimizer step", ylabel="E_matrix [MPa]",
                title="E_matrix per step (solid: jacfwd, dashed: FD)")
    axes[1].set(xlabel="optimizer step", ylabel="loss",
                title="Loss per step (solid: jacfwd, dashed: FD)")
    axes[2].plot(ref["strain_full"], ref["stress_full"], "-", color="lightgray", lw=1.0, label="reference")
    axes[2].plot(ref["strain_obs"], ref["stress_obs"], "o", color="k", label="calibration points")
    axes[2].plot(ref["strain_holdout"], ref["stress_holdout"], "^", color="C3", ms=9, label="held-out")
    axes[2].set(xlabel=r"strain $\varepsilon_{11}$", ylabel=r"stress $\sigma_{11}$ [MPa]",
                title="Calibrated fit vs. reference",
                xlim=(0, ref["strain_holdout"] * 1.3),
                ylim=(0, ref["stress_holdout"] * 1.3))
    for ax in axes:
        ax.grid(True, linewidth=0.5, alpha=0.5)
        ax.legend(fontsize=7)
    fig.suptitle("Inverse calibration of E_matrix across RVE discretizations")
    fig.tight_layout()
    paths.append(os.path.join(OUT_DIR, f"calibration_{today}.png"))
    fig.savefig(paths[-1], dpi=150)

    # -- scaling ---------------------------------------------------------------
    def series(method, key):
        pts = [(r["Nv"], r[method][key]) for r in ok
               if r[method]["status"] == "ok" and r[method].get(key) is not None]
        return tuple(zip(*pts)) if pts else ((), ())

    def mem_series(method):
        pts = [(r["Nv"], r[method]["device_mem"]["device_mb"]) for r in ok
               if r[method]["status"] == "ok" and r[method].get("device_mem")]
        return tuple(zip(*pts)) if pts else ((), ())

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    for method, style, label in (("autodiff", "o-", "jax.jacfwd"), ("fd", "s--", "finite differences")):
        axes[0].semilogx(*series(method, "E_final"), style, label=label)
        axes[1].loglog(*series(method, "step_s_mean"), style, label=label)
        axes[2].loglog(*series(method, "time_to_tol_s"), style, label=label)
        axes[3].loglog(*mem_series(method), style, label=label)
    oom = [r for r in results if r["autodiff"]["status"] == "oom" and r.get("Nv")]
    limit = next((r["device_limit_mb"] for r in ok if r.get("device_limit_mb")), None)
    if limit:
        axes[3].axhline(limit, color="firebrick", ls=":", lw=1.0, label="allocator limit")
    if oom:
        for ax in axes:
            ax.axvline(oom[0]["Nv"], color="firebrick", ls=":", lw=1.0)
    axes[0].set(ylabel="calibrated E_matrix [MPa]", title="Mesh convergence of calibrated E_matrix")
    axes[1].set(ylabel="wall time per step [s]", title="Cost per optimizer step")
    axes[2].set(ylabel="wall time [s]", title=f"Time to loss < {LOSS_TOL_FACTOR:g}x jacfwd's final")
    axes[3].set(ylabel="device memory, compiled [MB]", title="(loss, gradient) program memory")
    for ax in axes:
        ax.set_xlabel(r"grid size $N_v$")
        ax.grid(True, which="both", linewidth=0.5, alpha=0.5)
        ax.legend(fontsize=8)
    fig.suptitle("Calibration cost across RVE discretizations")
    fig.tight_layout()
    paths.append(os.path.join(OUT_DIR, f"scaling_{today}.png"))
    fig.savefig(paths[-1], dpi=150)
    return paths


def main():
    global MAX_STEPS
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-sweep", type=int, nargs="+", default=N_SWEEP,
                        help="voxels per in-plane side to sweep (default: %(default)s)")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS,
                        help="Adam steps per calibration (default: %(default)s)")
    parser.add_argument("--single-grid", type=int, default=None, help=argparse.SUPPRESS)  # worker mode
    parser.add_argument("--skip-fd", action="store_true", help=argparse.SUPPRESS)          # worker mode
    args = parser.parse_args()
    MAX_STEPS = args.max_steps

    if args.single_grid is not None:
        print(json.dumps(bench_one_grid(args.single_grid, run_fd=not args.skip_fd)))
        return

    ref = load_reference()
    limit = device_limit_mb()
    print(f"JAX backend: {jax.default_backend()}   Devices: {jax.devices()}")
    if limit is not None:
        print(f"Device allocator limit: {limit / 1024:.2f} GiB")
    print(f"phi={PHI}  r_fiber={R_FIBER}  size_in_r={SIZE_IN_R}  nz={NZ}  "
          f"E_init={E_INIT}  steps={MAX_STEPS}  lr={INIT_LR}")
    print(f"{len(ref['strain_obs'])} calibration points up to strain {STRAIN_MAX_LINEAR}, "
          f"held-out point at strain {ref['strain_holdout']:.5f}")
    print(f"{'n_vox':>6} {'grid':>16} {'Nv':>9}  "
          f"{'E_ad [MPa]':>10} {'n_tol':>6} {'s/step':>8} {'mem [MB]':>10}  "
          f"{'E_fd [MPa]':>10} {'n_tol':>6} {'s/step':>8} {'mem [MB]':>10}  "
          f"{'dgrad':>8} {'holdout':>9}")

    results = []
    fd_alive = True
    for n_vox in args.n_sweep:
        r = run_grid_subprocess(n_vox, fd_alive, MAX_STEPS)
        results.append(r)
        print_row(r)
        if r["autodiff"]["status"] != "ok":
            print(f"\nAutodiff {r['autodiff']['status']} at n_vox={n_vox}; sweep stopped there.")
            break
        if r["fd"]["status"] == "oom":
            fd_alive = False
            print(f"  FD OOM at Nv={r['Nv']}; continuing autodiff-only.")

    today = datetime.date.today().isoformat()
    payload = {
        "generated": today,
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "device_limit_mb": limit,
        "reference_csv": REF_CSV,
        "strain_max_linear": STRAIN_MAX_LINEAR,
        "strain_obs": ref["strain_obs"].tolist(), "stress_obs": ref["stress_obs"].tolist(),
        "strain_holdout": ref["strain_holdout"], "stress_holdout": ref["stress_holdout"],
        "phi": PHI, "r_fiber": R_FIBER, "size_in_r": SIZE_IN_R, "nz": NZ, "seed": SEED,
        "nu_matrix": NU_MATRIX, "fiber": FIBER_KW,
        "toler_lin": TOLER_LIN, "maxiter": MAXITER,
        "E_init": E_INIT, "init_lr": INIT_LR, "max_steps": MAX_STEPS,
        "fd_h": FD_H, "loss_tol_factor": LOSS_TOL_FACTOR,
        "results": results,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    json_path = os.path.join(OUT_DIR, f"results_{today}.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {len(results)} results to {json_path}")
    for p in plot(results, ref, today):
        print(f"Wrote plot to {p}")


if __name__ == "__main__":
    main()
