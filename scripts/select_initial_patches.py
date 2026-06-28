"""
Initial patch selection for active learning.

Reads data/patches_fft/index.json, extracts the 4-D latent codes (mu_mean),
and selects N diverse patches using **farthest-point sampling** — each new
point is as far as possible from all already-selected points in latent space.
This gives a space-filling design without requiring any external library.

After collecting simulation results for the selected patches, use
active_learning.py to fit the GPR and select the next patch to simulate.

Usage
-----
    # select 10 initial patches
    python scripts/select_initial_patches.py

    # select 20 patches, different seed
    python scripts/select_initial_patches.py --n 20 --seed 7

    # find nearest patch to a given z vector (output of active_learning.py)
    python scripts/select_initial_patches.py --query 0.12 -0.03 0.45 0.08
"""

import argparse
import json
from pathlib import Path

import numpy as np

INDEX = Path("data/patches_fft/index.json")


def load_index() -> tuple[list[dict], np.ndarray]:
    """Load patches list and stack mu_mean vectors into (N, 4) array."""
    with open(INDEX) as f:
        idx = json.load(f)
    patches = idx["patches"]
    Z = np.array([p["mu_mean"] for p in patches], dtype=float)   # (N, 4)
    return patches, Z


def farthest_point_sample(Z: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """
    Greedy farthest-point sampling in latent space.

    Starts from the point closest to the centroid, then iteratively picks
    the point with maximum minimum-distance to the already-selected set.

    Returns
    -------
    indices : (n,) int  — row indices into Z
    """
    rng     = np.random.default_rng(seed)
    N       = len(Z)
    # normalise each dim to [0,1] for unbiased distance
    lo, hi  = Z.min(0), Z.max(0)
    Zn      = (Z - lo) / np.where(hi > lo, hi - lo, 1.0)

    selected  = []
    # start from point nearest to centroid
    centroid  = Zn.mean(0)
    first     = int(np.argmin(np.linalg.norm(Zn - centroid, axis=1)))
    selected.append(first)

    min_dists = np.linalg.norm(Zn - Zn[first], axis=1)   # (N,)

    for _ in range(n - 1):
        next_idx = int(np.argmax(min_dists))
        selected.append(next_idx)
        d = np.linalg.norm(Zn - Zn[next_idx], axis=1)
        min_dists = np.minimum(min_dists, d)

    return np.array(selected)


def nearest_patch(z: np.ndarray, Z: np.ndarray) -> int:
    """Return index of the patch whose mu_mean is closest to z."""
    return int(np.argmin(np.linalg.norm(Z - z, axis=1)))


def main():
    parser = argparse.ArgumentParser(
        description="Select initial patches for active learning"
    )
    parser.add_argument("--n",     type=int,   default=10,
                        help="Number of initial patches to select (default 10)")
    parser.add_argument("--seed",  type=int,   default=0)
    parser.add_argument("--query", type=float, nargs=4, default=None,
                        metavar=("Z0", "Z1", "Z2", "Z3"),
                        help="Find nearest patch to this 4-D z vector")
    args = parser.parse_args()

    patches, Z = load_index()
    print(f"Index   : {len(patches)} patches  |  latent dim = {Z.shape[1]}")
    print(f"  phi   : [{Z[:,0].min():.3f}, {Z[:,0].max():.3f}]  (across mu_mean[0])")

    # ── nearest-patch query mode ──────────────────────────────────────────────
    if args.query is not None:
        z = np.array(args.query)
        idx = nearest_patch(z, Z)
        p   = patches[idx]
        dist = float(np.linalg.norm(Z[idx] - z))
        print(f"\nNearest patch to z={z}:")
        print(f"  file     : {p['file']}")
        print(f"  phi      : {p['phi']:.4f}")
        print(f"  mu_mean  : {p['mu_mean']}")
        print(f"  distance : {dist:.4f}")
        return

    # ── initial selection ─────────────────────────────────────────────────────
    print(f"\nFarthest-point sampling  n={args.n}  seed={args.seed} …")
    sel = farthest_point_sample(Z, args.n, seed=args.seed)

    print(f"\n{'idx':>6}  {'phi':>7}  {'mu_mean':^38}  file")
    print("─" * 80)
    for rank, i in enumerate(sel):
        p = patches[i]
        mu = "  ".join(f"{v:+.3f}" for v in p["mu_mean"])
        print(f"{i:>6}  {p['phi']:>7.4f}  [{mu}]  {p['file']}")

    # ── YAML snippet ──────────────────────────────────────────────────────────
    print(f"\nAdd to configs/latent_space.yaml  (observations: after simulation):")
    print("observations:")
    for i in sel:
        p = patches[i]
        mu = p["mu_mean"]
        print(f"  - params:")
        print(f"      z0: {mu[0]:.6f}")
        print(f"      z1: {mu[1]:.6f}")
        print(f"      z2: {mu[2]:.6f}")
        print(f"      z3: {mu[3]:.6f}")
        print(f"    output: <run {p['file']}>")

    # ── save candidate z-vectors for active_learning.py ──────────────────────
    out = Path("data/patches_fft/candidates.npy")
    np.save(out, Z)
    print(f"\nAll {len(Z)} latent codes saved → {out}")
    print("Point active_learning.py at this file via candidates_file: in the YAML.")


if __name__ == "__main__":
    main()
