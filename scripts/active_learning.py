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
os.environ["JAX_ENABLE_X64"] = "1"   # must be set before importing jax
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import yaml

sys.path.insert(0, "src")
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


# ── Step 2c: extract peak stress ──────────────────────────────────────────────

def extract_peak_stress(h5_path: Path, component: int = 0) -> float:
    """
    Read all increments from the simulation HDF5 and return the maximum
    spatially-averaged stress component.

    Voigt index:  0=σ₁₁  1=σ₂₂  2=σ₃₃  3=σ₁₂  4=σ₁₃  5=σ₂₃
    Arrays stored in ZYX order in HDF5 with shape (*spatial, 6).
    """
    peak = 0.0
    with h5py.File(h5_path, "r") as f:
        groups = sorted(k for k in f.keys() if k.startswith("increment_"))
        for g in groups:
            grp = f[g]
            assert isinstance(grp, h5py.Group)
            if "stress" not in grp:
                continue
            ds = grp["stress"]
            assert isinstance(ds, h5py.Dataset)
            arr = np.array(ds)           # shape (*spatial, 6), ZYX order
            mean_s = float(np.mean(arr[..., component]))
            peak = max(peak, abs(mean_s))
    return peak


# ── Step 3–4: GP ─────────────────────────────────────────────────────────────

def _normalize(X: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return (X - lo) / np.where(hi > lo, hi - lo, 1.0)


def fit_and_predict(
    X_obs:  np.ndarray,    # (n, 4) normalised
    y_obs:  np.ndarray,    # (n, 1)
    X_cand: np.ndarray,    # (M, 4) normalised
    n_iters: int = 500,
    lr:      float = 0.01,
) -> np.ndarray:
    """Fit Matérn-5/2 GP and return posterior variance at candidates."""
    key = jax.random.PRNGKey(0)
    X   = jnp.array(X_obs)
    y   = jnp.array(y_obs)
    D   = gpx.Dataset(X=X, y=y)

    kernel    = gpx.kernels.Matern52(active_dims=list(range(X.shape[1])))
    meanf     = gpx.mean_functions.Constant()
    # API differs between GPJax versions — try both module paths
    try:
        prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)
    except AttributeError:
        prior = gpx.Prior(mean_function=meanf, kernel=kernel)      # type: ignore[attr-defined]
    likelihood = gpx.likelihoods.Gaussian(num_datapoints=D.n)
    posterior  = prior * likelihood

    opt_post, _ = gpx.fit(
        model      = posterior,
        objective  = gpx.objectives.ConjugateMLL(negative=True),
        train_data = D,
        optim      = optax.adam(lr),
        num_iters  = n_iters,
        key        = key,
        safe       = True,
    )
    pred = opt_post.predict(jnp.array(X_cand), train_data=D)  # type: ignore[union-attr]
    return np.array(pred.variance())  # type: ignore[union-attr]


# ── Step 5: nearest patch ─────────────────────────────────────────────────────

def nearest_patch(z: np.ndarray, patches: list[dict], Z: np.ndarray) -> dict:
    idx = int(np.argmin(np.linalg.norm(Z - z, axis=1)))
    return patches[idx]


# ── Persistence ───────────────────────────────────────────────────────────────

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

                # extract QoI
                sigma_peak = extract_peak_stress(h5, component=component)
                print(f"    peak σ (component {component}) = {sigma_peak:.4f} MPa")

                # add observation
                obs.append({
                    "params":  {k: float(v) for k, v in zip(names, p["mu_mean"])},
                    "output":  float(sigma_peak),
                    "file":    p["file"],
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

        # ── Step 4: predict variance ──────────────────────────────────────────
        print(f"Step 4  GP uncertainty (Matérn-5/2, {args.gp_iters} iters) …")
        var = fit_and_predict(X_obs_n, y_obs, X_cand_n,
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

        # ── Step 6: run next simulation ───────────────────────────────────────
        print(f"\nStep 6  Run simulation …")
        patch_file = patch_dir / next_p["file"]
        h5 = run_simulation(patch_file, sim_config, base_out / f"iter_{iteration:02d}")
        sigma_peak = extract_peak_stress(h5, component=component)
        print(f"        peak σ = {sigma_peak:.4f} MPa")

        obs.append({
            "params": {k: float(v) for k, v in zip(names, next_p["mu_mean"])},
            "output": float(sigma_peak),
            "file":   next_p["file"],
        })
        cfg["observations"] = obs
        save_observations(args.config, obs)

    print(f"\nDone — {len(cfg.get('observations', []))} total observations.")


if __name__ == "__main__":
    main()
