"""
Active learning over the patches_fft latent space.

Full automated loop
-------------------
1. Load index.json → all 2250 latent codes (mu_mean, 4-D).
2. If fewer than n_init observations exist: use farthest-point sampling
   to select n_init diverse patches and run their PFF simulations.
3. Fit an exact GP (GPJax, Matérn-5/2) on (z, peak_σ) pairs.
4. Predict posterior variance at all unobserved candidates.
5. Select the candidate with maximum variance (uncertainty sampling).
6. Find the nearest actual patch file in the index.
7. Run that simulation, extract peak stress, add to observations.
8. Save updated observations back to the YAML config.
9. Repeat from step 3.

QoI = peak spatially-averaged stress component (default σ₁₁, Voigt index 0).
Simulations are run by calling pff_nw_cg_strain.py with --input <patch.npz>.

Usage
-----
    # one full iteration (initial selection if needed, then GPR step)
    python scripts/active_learning.py configs/latent_space_patches.yaml

    # run N iterations in sequence
    python scripts/active_learning.py configs/latent_space_patches.yaml --iters 5

Dependencies
------------
    pip install gpjax optax jax jaxlib pyyaml numpy h5py
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import yaml

sys.path.insert(0, "src")
import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)
from utils.config import load_config

# ── GPJax ────────────────────────────────────────────────────────────────────

try:
    import gpjax as gpx
    import jax
    import jax.numpy as jnp
    import optax
except ImportError:
    raise SystemExit("gpjax / jax not found.  pip install gpjax optax")


# ── Step 1: load patch index ──────────────────────────────────────────────────

def load_patch_index(index_path: Path) -> tuple[list[dict], np.ndarray]:
    """Returns patches list and (N, 4) latent-code matrix."""
    with open(index_path) as f:
        idx = json.load(f)
    patches = idx["patches"]
    Z = np.array([p["mu_mean"] for p in patches], dtype=float)
    return patches, Z


# ── Step 2: initial design ────────────────────────────────────────────────────

def farthest_point_sample(Z: np.ndarray, n: int) -> np.ndarray:
    """
    Greedy farthest-point sampling.
    Each new point maximises its minimum distance to the already-selected set.
    Returns (n,) index array into Z.
    """
    lo, hi = Z.min(0), Z.max(0)
    Zn = (Z - lo) / np.where(hi > lo, hi - lo, 1.0)   # normalise to [0,1]

    sel       = []
    centroid  = Zn.mean(0)
    first     = int(np.argmin(np.linalg.norm(Zn - centroid, axis=1)))
    sel.append(first)
    min_dists = np.linalg.norm(Zn - Zn[first], axis=1)

    for _ in range(n - 1):
        nxt = int(np.argmax(min_dists))
        sel.append(nxt)
        d = np.linalg.norm(Zn - Zn[nxt], axis=1)
        min_dists = np.minimum(min_dists, d)

    return np.array(sel)


# ── Step 2b: run simulation ───────────────────────────────────────────────────

def run_simulation(
    patch_file: Path,
    sim_config:  Path,
    output_dir:  Path,
) -> Path:
    """
    Call pff_nw_cg_strain.py with --input and --output overrides.
    Returns the expected output HDF5 path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Running simulation: {patch_file.name} → {output_dir}")
    subprocess.run(
        [sys.executable,
         "scripts/simulation/pff_nw_cg_strain.py",
         str(sim_config),
         "--input",  str(patch_file),
         "--output", str(output_dir)],
        check=True,
    )
    jobname = patch_file.stem
    return output_dir / f"{jobname}.h5"


# ── Step 2c: extract peak stress & stress-strain curve ───────────────────────

def extract_stress_strain_curve(
    h5_path: Path,
    component: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Read all increments and return spatially-averaged (strain, stress) arrays.

    Voigt index:  0=σ₁₁/ε₁₁  1=σ₂₂/ε₂₂  2=σ₃₃/ε₃₃
                  3=σ₁₂/ε₁₂  4=σ₁₃/ε₁₃  5=σ₂₃/ε₂₃
    """
    strains, stresses = [], []
    with h5py.File(h5_path, "r") as f:
        groups = sorted(k for k in f.keys() if k.startswith("increment_"))
        for g in groups:
            grp = f[g]
            assert isinstance(grp, h5py.Group)
            if "stress" not in grp or "strain" not in grp:
                continue
            stresses.append(float(np.mean(np.array(grp["stress"])[..., component])))
            strains.append( float(np.mean(np.array(grp["strain"])[..., component])))
    return np.array(strains), np.array(stresses)


def extract_peak_stress(h5_path: Path, component: int = 0) -> float:
    """
    Read all increments from the simulation HDF5 and return the maximum
    spatially-averaged stress component.

    Voigt index:  0=σ₁₁  1=σ₂₂  2=σ₃₃  3=σ₁₂  4=σ₁₃  5=σ₂₃
    Arrays stored in ZYX order in HDF5 with shape (*spatial, 6).
    """
    _, stresses = extract_stress_strain_curve(h5_path, component)
    return float(np.max(np.abs(stresses))) if len(stresses) else 0.0


# ── Step 3–4: GP ─────────────────────────────────────────────────────────────

def _normalize(X: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return (X - lo) / np.where(hi > lo, hi - lo, 1.0)


def fit_and_predict(
    X_obs:  np.ndarray,    # (n, 4) normalised
    y_obs:  np.ndarray,    # (n, 1)
    X_cand: np.ndarray,    # (M, 4) normalised
    n_iters: int = 500,
    lr:      float = 0.01,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit Matérn-5/2 GP and return (posterior mean, posterior variance, hyperparams)."""
    key = jax.random.PRNGKey(0)
    X   = jnp.array(X_obs)
    y   = jnp.array(y_obs)
    D   = gpx.Dataset(X=X, y=y)

    dims   = list(range(X.shape[1]))
    kernel = gpx.kernels.SumKernel(kernels=[
        gpx.kernels.Linear(active_dims=dims),
        gpx.kernels.Matern32(active_dims=dims),
    ])
    meanf     = gpx.mean_functions.Constant()
    try:
        prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)
    except AttributeError:
        prior = gpx.Prior(mean_function=meanf, kernel=kernel)      # type: ignore[attr-defined]
    likelihood = gpx.likelihoods.Gaussian(num_datapoints=D.n)
    posterior  = prior * likelihood

    opt_post, _ = gpx.fit(
        model      = posterior,
        objective  = lambda p, d: -gpx.objectives.conjugate_mll(p, d),
        train_data = D,
        optim      = optax.adam(lr),
        num_iters  = n_iters,
        key        = key,
        safe       = True,
    )
    pred = opt_post.predict(jnp.array(X_cand), train_data=D)  # type: ignore[union-attr]

    def _unwrap(p):
        return p.unwrap() if hasattr(p, "unwrap") else jnp.asarray(p)

    k_linear, k_matern = opt_post.prior.kernel.kernels
    hyperparams = {
        "linear_variance":  float(_unwrap(k_linear.variance)),
        "matern_lengthscale": np.array(_unwrap(k_matern.lengthscale)),
        "matern_variance":  float(_unwrap(k_matern.variance)),
        "noise_variance":   float(_unwrap(opt_post.likelihood.obs_stddev) ** 2),
        "mean_constant":    float(_unwrap(opt_post.prior.mean_function.constant)),
    }
    return np.array(pred.mean), np.array(pred.variance), hyperparams  # type: ignore[union-attr]


# ── Step 5: nearest patch ─────────────────────────────────────────────────────

def nearest_patch(z: np.ndarray, patches: list[dict], Z: np.ndarray) -> dict:
    idx = int(np.argmin(np.linalg.norm(Z - z, axis=1)))
    return patches[idx]


# ── Persistence ───────────────────────────────────────────────────────────────

def save_trace_snapshot(
    trace_dir: Path,
    iteration: int,
    mean_all: np.ndarray,     # (N,) posterior mean; NaN for observed
    var_all: np.ndarray,      # (N,) posterior variance; 0 for observed
    obs_mask: np.ndarray,     # (N,) bool — True where already observed
    selected_idx: int,        # index into the full N-patch array; -1 = final frame
    hyperparams: dict,        # fitted kernel/likelihood hyperparameters
    X_obs_n: np.ndarray,      # (n, D) normalised training inputs
    y_obs: np.ndarray,        # (n, 1) training targets
) -> None:
    trace_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        trace_dir / f"iter_{iteration:04d}.npz",
        mean_all=mean_all,
        var_all=var_all,
        obs_mask=obs_mask,
        selected_idx=np.array(selected_idx),
        # GP hyperparameters (Linear + Matérn-3/2 sum kernel)
        hp_linear_variance=np.array(hyperparams["linear_variance"]),
        hp_matern_lengthscale=hyperparams["matern_lengthscale"],
        hp_matern_variance=np.array(hyperparams["matern_variance"]),
        hp_noise_variance=np.array(hyperparams["noise_variance"]),
        hp_mean_constant=np.array(hyperparams["mean_constant"]),
        # training data (normalised)
        X_obs_n=X_obs_n,
        y_obs=y_obs,
    )


def save_gpr_figure(
    fig_path: Path,
    Z_all: np.ndarray,      # (N, D) full latent matrix
    mean_all: np.ndarray,   # (N,) — NaN for observed
    var_all: np.ndarray,    # (N,) — 0 for observed
    obs_mask: np.ndarray,   # (N,) bool
    selected_idx: int,      # -1 = no selection (final frame)
    names: list[str],
    iteration: int,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize

    D         = Z_all.shape[1]
    cand_mask = ~obs_mask
    Z_cand    = Z_all[cand_mask]
    var_cand  = var_all[cand_mask]
    mean_cand = mean_all[cand_mask]
    Z_obs     = Z_all[obs_mask]
    n_obs     = int(obs_mask.sum())

    var_norm  = Normalize(vmin=var_cand.min(),  vmax=var_cand.max())
    mean_norm = Normalize(vmin=mean_cand.min(), vmax=mean_cand.max())

    fig, axes = plt.subplots(D, D, figsize=(4 * D, 4 * D))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    for i in range(D):
        for j in range(D):
            ax = axes[i, j]
            ax.tick_params(labelsize=7)
            if i < D - 1:
                ax.set_xticklabels([])
            if j > 0:
                ax.set_yticklabels([])

            if i == j:
                ax.hist(Z_all[:, i],  bins=40, color="#888", alpha=0.35, density=True)
                ax.hist(Z_obs[:, i],  bins=15, color="white", alpha=0.85, density=True,
                        edgecolor="black", linewidth=0.5)
                if j == 0:
                    ax.set_ylabel(names[i], fontsize=9)
                if i == D - 1:
                    ax.set_xlabel(names[j], fontsize=9)

            elif j < i:
                # lower triangle — variance
                ax.scatter(Z_cand[:, j], Z_cand[:, i],
                           c=var_cand, cmap="plasma", norm=var_norm,
                           s=4, alpha=0.7, linewidths=0, rasterized=True)
                ax.scatter(Z_obs[:, j], Z_obs[:, i],
                           marker="*", s=60, c="white", edgecolors="black",
                           linewidths=0.5, zorder=5)
                if selected_idx >= 0:
                    ax.scatter(Z_all[selected_idx, j], Z_all[selected_idx, i],
                               marker="*", s=200, c="yellow", edgecolors="black",
                               linewidths=0.7, zorder=6)
                if j == 0:
                    ax.set_ylabel(names[i], fontsize=9)
                if i == D - 1:
                    ax.set_xlabel(names[j], fontsize=9)

            else:
                # upper triangle — mean
                ax.scatter(Z_cand[:, j], Z_cand[:, i],
                           c=mean_cand, cmap="RdBu_r", norm=mean_norm,
                           s=4, alpha=0.7, linewidths=0, rasterized=True)
                ax.scatter(Z_obs[:, j], Z_obs[:, i],
                           marker="*", s=60, c="white", edgecolors="black",
                           linewidths=0.5, zorder=5)
                if selected_idx >= 0:
                    ax.scatter(Z_all[selected_idx, j], Z_all[selected_idx, i],
                               marker="*", s=200, c="yellow", edgecolors="black",
                               linewidths=0.7, zorder=6)
                if j == 0:
                    ax.set_ylabel(names[i], fontsize=9)
                if i == D - 1:
                    ax.set_xlabel(names[j], fontsize=9)

    # shared colorbars
    sm_var = cm.ScalarMappable(cmap="plasma",  norm=var_norm)
    sm_var.set_array([])
    fig.colorbar(sm_var, ax=axes[:, -1], shrink=0.6, pad=0.03,
                 label="GP posterior variance σ²")

    sm_mean = cm.ScalarMappable(cmap="RdBu_r", norm=mean_norm)
    sm_mean.set_array([])
    fig.colorbar(sm_mean, ax=axes[0, :], shrink=0.6, pad=0.03,
                 location="top", label="GP posterior mean μ [MPa]")

    title = (
        f"Final posterior — {n_obs} observations"
        if selected_idx < 0 else
        f"Iteration {iteration} — {n_obs} obs | σ²_max={var_cand.max():.4f}"
    )
    fig.suptitle(title, fontsize=13, y=1.02)

    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure → {fig_path}")


def save_observations(config_path: Path, obs: list[dict]) -> None:
    """Overwrite the observations list in the YAML config file."""
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    raw["observations"] = obs
    with open(config_path, "w") as f:
        yaml.dump(raw, f, default_flow_style=False, sort_keys=False)
    print(f"  Saved {len(obs)} observations → {config_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Active learning — patches_fft GPR loop")
    parser.add_argument("config", type=Path, help="YAML config (latent_space_patches.yaml)")
    parser.add_argument("--iters",      type=int,   default=1,    help="GP iterations to run (default 1)")
    parser.add_argument("--n_init",     type=int,   default=None, help="Override n_init from config")
    parser.add_argument("--gp_iters",   type=int,   default=500)
    parser.add_argument("--lr",         type=float, default=0.01)
    args = parser.parse_args()

    cfg        = load_config(args.config)
    index_path = Path(cfg.get("index_file",  "data/patches_fft/index.json"))
    sim_config = Path(cfg.get("sim_config",  "configs/pff_prototype.yaml"))
    patch_dir  = Path(cfg.get("patch_dir",   "data/patches_fft"))
    component  = int(cfg.get("stress_component", 0))
    n_init     = args.n_init or int(cfg.get("n_init", 10))
    names      = cfg.get("dim_names", ["z0", "z1", "z2", "z3"])

    # ── Step 1: load index ────────────────────────────────────────────────────
    print("=" * 60)
    print(f"Step 1  Load patch index: {index_path}")
    patches, Z_all = load_patch_index(index_path)
    print(f"        {len(patches)} patches  |  latent dim = {Z_all.shape[1]}")

    # normalisation bounds fixed from the full index
    lo, hi   = Z_all.min(0), Z_all.max(0)
    base_out = Path(cfg.get("active_output", "output/active_learning"))

    for iteration in range(1, args.iters + 1):
        print(f"\n{'=' * 60}")
        print(f"Iteration {iteration}/{args.iters}")

        obs = list(cfg.get("observations", []) or [])

        # ── Step 2: initial design ────────────────────────────────────────────
        if len(obs) < n_init:
            n_needed = n_init - len(obs)
            print(f"\nStep 2  Initial design — need {n_needed} more simulations")

            # identify already-observed patches by nearest neighbour
            obs_z = np.array([[o["params"][k] for k in names] for o in obs]) if obs else np.zeros((0, 4))
            obs_files = set()
            if len(obs_z) > 0:
                for z in obs_z:
                    p = nearest_patch(z, patches, Z_all)
                    obs_files.add(p["file"])

            sel_idx = farthest_point_sample(Z_all, n_init + len(obs))
            new_patches = [patches[i] for i in sel_idx
                           if patches[i]["file"] not in obs_files][:n_needed]

            for p in new_patches:
                patch_file = patch_dir / p["file"]
                print(f"\n  ▶ {p['file']}  phi={p['phi']:.4f}")

                # run simulation
                h5 = run_simulation(patch_file, sim_config, base_out / "initial")

                # extract QoI + stress-strain curve
                eps_arr, sig_arr = extract_stress_strain_curve(h5, component=component)
                sigma_peak = float(np.max(np.abs(sig_arr))) if len(sig_arr) else 0.0
                print(f"    peak σ (component {component}) = {sigma_peak:.4f} MPa")
                curve_path = h5.with_name(h5.stem + "_curve.npz")
                np.savez(curve_path, strain=eps_arr, stress=sig_arr)

                # add observation
                obs.append({
                    "params":     {k: float(v) for k, v in zip(names, p["mu_mean"])},
                    "output":     float(sigma_peak),
                    "file":       p["file"],
                    "curve_file": str(curve_path),
                })
                cfg["observations"] = obs
                save_observations(args.config, obs)

        # ── Step 3: GP fitting ────────────────────────────────────────────────
        print(f"\nStep 3  Fit GP on {len(obs)} observations …")
        X_obs = np.array([[o["params"][k] for k in names] for o in obs])
        y_obs = np.array([[o["output"]] for o in obs])
        X_obs_n = _normalize(X_obs, lo, hi)

        # candidates = all patches not yet observed
        obs_files = {o.get("file", "") for o in obs}
        cand_mask = [p["file"] not in obs_files for p in patches]
        X_cand    = Z_all[cand_mask]
        X_cand_n  = _normalize(X_cand, lo, hi)
        cand_patches = [p for p, m in zip(patches, cand_mask) if m]
        print(f"        {sum(cand_mask)} unobserved candidates remaining")

        if len(X_cand) == 0:
            print("All patches observed — done.")
            break

        # ── Step 4: predict mean & variance ──────────────────────────────────
        print(f"Step 4  GP uncertainty (Matérn-5/2, {args.gp_iters} iters) …")
        mean, var, hyperparams = fit_and_predict(X_obs_n, y_obs, X_cand_n,
                                                 n_iters=args.gp_iters, lr=args.lr)

        # ── Step 5: select next ───────────────────────────────────────────────
        best = int(np.argmax(var))
        next_p = cand_patches[best]
        print(f"\nStep 5  Next patch  σ²={var[best]:.4e}")
        print(f"        file    : {next_p['file']}")
        print(f"        phi     : {next_p['phi']:.4f}")
        print(f"        mu_mean : {next_p['mu_mean']}")

        # top-5
        top5 = np.argsort(var)[::-1][:5]
        print(f"\n  Top-5 by variance:")
        for rank, i in enumerate(top5, 1):
            p = cand_patches[i]
            print(f"  {rank}. σ²={var[i]:.4e}  {p['file']}  phi={p['phi']:.4f}")

        # save per-iteration GP snapshot + figure
        cand_idx_full = np.where(cand_mask)[0]
        mean_all = np.full(len(patches), np.nan)
        mean_all[cand_idx_full] = mean
        var_all = np.zeros(len(patches))
        var_all[cand_idx_full] = var
        obs_mask_full = ~np.array(cand_mask)
        sel_idx_full  = int(cand_idx_full[best])

        save_trace_snapshot(
            base_out / "trace",
            iteration,
            mean_all,
            var_all,
            obs_mask=obs_mask_full,
            selected_idx=sel_idx_full,
            hyperparams=hyperparams,
            X_obs_n=X_obs_n,
            y_obs=y_obs,
        )
        save_gpr_figure(
            base_out / "figures" / f"iter_{iteration:04d}.png",
            Z_all, mean_all, var_all, obs_mask_full, sel_idx_full, names, iteration,
        )

        # ── Step 6: run next simulation ───────────────────────────────────────
        print(f"\nStep 6  Run simulation …")
        patch_file = patch_dir / next_p["file"]
        h5 = run_simulation(patch_file, sim_config, base_out / f"iter_{iteration:02d}")
        eps_arr, sig_arr = extract_stress_strain_curve(h5, component=component)
        sigma_peak = float(np.max(np.abs(sig_arr))) if len(sig_arr) else 0.0
        print(f"        peak σ = {sigma_peak:.4f} MPa")
        curve_path = h5.with_name(h5.stem + "_curve.npz")
        np.savez(curve_path, strain=eps_arr, stress=sig_arr)

        obs.append({
            "params":     {k: float(v) for k, v in zip(names, next_p["mu_mean"])},
            "output":     float(sigma_peak),
            "file":       next_p["file"],
            "curve_file": str(curve_path),
        })
        cfg["observations"] = obs
        save_observations(args.config, obs)

    # ── Final GP update after last simulation ─────────────────────────────────
    obs = list(cfg.get("observations", []) or [])
    if obs:
        print(f"\nFinal GP update on {len(obs)} observations …")
        X_obs   = np.array([[o["params"][k] for k in names] for o in obs])
        y_obs   = np.array([[o["output"]] for o in obs])
        X_obs_n = _normalize(X_obs, lo, hi)

        obs_files  = {o.get("file", "") for o in obs}
        cand_mask  = [p["file"] not in obs_files for p in patches]
        X_cand_n   = _normalize(Z_all[np.where(cand_mask)[0]], lo, hi)

        if X_cand_n.shape[0] > 0:
            mean, var, hyperparams = fit_and_predict(X_obs_n, y_obs, X_cand_n,
                                                     n_iters=args.gp_iters, lr=args.lr)
            cand_idx_full = np.where(cand_mask)[0]
            obs_mask_full = ~np.array(cand_mask)
            mean_all = np.full(len(patches), np.nan)
            mean_all[cand_idx_full] = mean
            var_all = np.zeros(len(patches))
            var_all[cand_idx_full] = var

            save_trace_snapshot(
                base_out / "trace",
                args.iters + 1,
                mean_all,
                var_all,
                obs_mask=obs_mask_full,
                selected_idx=-1,
                hyperparams=hyperparams,
                X_obs_n=X_obs_n,
                y_obs=y_obs,
            )
            save_gpr_figure(
                base_out / "figures" / "final.png",
                Z_all, mean_all, var_all, obs_mask_full, -1, names, args.iters + 1,
            )
            print(f"  σ²_max = {var.max():.4e}  (final posterior)")

    print(f"\nDone — {len(obs)} total observations.")


if __name__ == "__main__":
    main()
