"""
Active learning over a discrete 4-D latent space using GPJax (Matérn-5/2).

Two ways to supply candidates
------------------------------
A) Grid (YAML only)
   Define ``parameters`` in the config — the Cartesian product is used.

B) File  ← preferred when you have a real VAE/encoder latent space
   Set ``candidates_file`` in the config pointing to:
     • .npy  →  (N, 4) float array  of z-vectors
     • .npz  →  keys  z (N,4)  and optionally  recon_loss (N,)
   Set ``max_recon_loss`` to discard poorly-reconstructed codes before the GP.
   The GP then selects the next z to simulate from the ones that actually
   decode well — fixing the "reconstruction not best" issue.

Workflow
--------
1. Load or build candidate set.
2. (Optional) filter by reconstruction quality.
3. Load existing (X, y) observations from the YAML config.
4. Fit exact GP (GPJax, Matérn-5/2) on observations.
5. Posterior variance at every unobserved candidate.
6. Return the max-variance candidate = next z to simulate.

Usage
-----
    python scripts/active_learning.py configs/latent_space.yaml

Dependencies
------------
    pip install gpjax optax jax jaxlib pyyaml numpy
"""

import argparse
import sys
from itertools import product
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

sys.path.insert(0, "src")
from utils.config import load_config

# ── GPJax imports ─────────────────────────────────────────────────────────────
try:
    import gpjax as gpx
except ImportError:
    raise SystemExit("gpjax not found.  Install with: pip install gpjax")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_candidates(parameters: list[dict]) -> tuple[np.ndarray, list[str]]:
    """Cartesian product of discrete parameter values (grid mode)."""
    names  = [p["name"] for p in parameters]
    values = [p["values"] for p in parameters]
    grid   = list(product(*values))
    return np.array(grid, dtype=float), names


def _load_candidates(
    cfg: dict,
) -> tuple[np.ndarray, list[str], np.ndarray | None]:
    """
    Load candidates from a file (file mode) or build from Cartesian product.

    File mode  — set ``candidates_file`` in the YAML:
        .npy   → (N, 4) array                  (no reconstruction loss)
        .npz   → keys z (N,4), recon_loss (N,) (optional quality filter)

    Returns
    -------
    X_all      : (N, d)  raw candidate z-vectors
    names      : list of dimension labels
    recon_loss : (N,) float or None
    """
    path = cfg.get("candidates_file")
    if path:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"candidates_file not found: {p}")

        recon_loss = None
        if p.suffix == ".npz":
            data = np.load(p)
            X_all = data["z"].astype(float)
            if "recon_loss" in data:
                recon_loss = data["recon_loss"].astype(float)
        else:
            X_all = np.load(p).astype(float)

        d     = X_all.shape[1]
        names = cfg.get("dim_names") or [f"z{i}" for i in range(d)]
        return X_all, names, recon_loss

    # fallback: Cartesian grid from ``parameters``
    X_all, names = _make_candidates(cfg["parameters"])
    return X_all, names, None


def _normalize(X: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Scale each column to [0, 1] using the per-parameter min/max."""
    return (X - lo) / np.where(hi > lo, hi - lo, 1.0)


def _obs_to_arrays(
    observations: list[dict],
    names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Extract (X_obs, y_obs) from the observation list in the YAML."""
    X_rows, y_vals = [], []
    for obs in observations:
        row = [float(obs["params"][n]) for n in names]
        X_rows.append(row)
        y_vals.append(float(obs["output"]))
    return np.array(X_rows), np.array(y_vals).reshape(-1, 1)


def _nearest_observed(
    candidate: np.ndarray,
    X_obs: np.ndarray,
) -> float:
    """Euclidean distance from a candidate to the nearest observation."""
    return float(np.min(np.linalg.norm(X_obs - candidate, axis=1)))


# ── GP model ──────────────────────────────────────────────────────────────────

def fit_gp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_iters: int = 500,
    lr: float = 0.01,
    key: jax.Array | None = None,
):
    """
    Fit an exact GP with Matérn-5/2 kernel and Gaussian likelihood via GPJax.

    Parameters
    ----------
    X_train : (n, d)  normalised training inputs
    y_train : (n, 1)  scalar targets
    n_iters : optimisation steps
    lr      : Adam learning rate

    Returns
    -------
    posterior : optimised GPJax posterior
    D         : gpx.Dataset
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    X = jnp.array(X_train)
    y = jnp.array(y_train)

    D = gpx.Dataset(X=X, y=y)

    kernel    = gpx.kernels.Matern52(
        active_dims=list(range(X.shape[1])),
    )
    meanf     = gpx.mean_functions.Constant()
    prior     = gpx.Prior(mean_function=meanf, kernel=kernel)
    likelihood = gpx.likelihoods.Gaussian(num_datapoints=D.n)
    posterior  = prior * likelihood

    opt_posterior, _ = gpx.fit(
        model     = posterior,
        objective = gpx.objectives.ConjugateMLL(negative=True),
        train_data= D,
        optim     = optax.adam(lr),
        num_iters = n_iters,
        key       = key,
        safe      = True,
    )

    return opt_posterior, D


def predict_variance(
    posterior,
    D,
    X_star: np.ndarray,
) -> np.ndarray:
    """
    Posterior predictive variance at candidate points X_star.

    Returns
    -------
    var : (N,) float
    """
    X_s   = jnp.array(X_star)
    pred  = posterior.predict(X_s, train_data=D)
    return np.array(pred.variance())


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Active learning: select next latent-space point via GP uncertainty"
    )
    parser.add_argument("config", type=Path, help="YAML latent-space config")
    parser.add_argument("--n_iters", type=int, default=500,
                        help="GP hyperparameter optimisation steps (default 500)")
    parser.add_argument("--lr",      type=float, default=0.01,
                        help="Adam learning rate (default 0.01)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # ── load / build candidates ───────────────────────────────────────────────
    X_all, names, recon_loss = _load_candidates(cfg)
    mode = "file" if cfg.get("candidates_file") else "grid"
    print(f"Candidates   : {len(X_all):,}  (mode={mode},  d={len(names)})")
    print(f"  dims : {names}")

    # ── reconstruction quality filter (file mode only) ────────────────────────
    if recon_loss is not None:
        thr = float(cfg.get("max_recon_loss", np.inf))
        quality_mask = recon_loss <= thr
        n_before = len(X_all)
        X_all      = X_all[quality_mask]
        recon_loss = recon_loss[quality_mask]
        print(f"  recon filter (≤ {thr:.4g}): {quality_mask.sum():,} / {n_before:,} kept")

    # ── observations ──────────────────────────────────────────────────────────
    obs = cfg.get("observations", [])
    if not obs:
        raise SystemExit("No observations found in config. "
                         "Add at least one entry under 'observations:'.")

    X_obs_raw, y_obs = _obs_to_arrays(obs, names)
    print(f"\nObservations : {len(obs)}")
    for i, o in enumerate(obs):
        print(f"  [{i}]  z={list(o['params'].values())}  output={o['output']:.4g}")

    # ── normalise ─────────────────────────────────────────────────────────────
    lo = X_all.min(axis=0)
    hi = X_all.max(axis=0)
    X_all_n = _normalize(X_all,     lo, hi)
    X_obs_n = _normalize(X_obs_raw, lo, hi)

    # ── remove already observed points ────────────────────────────────────────
    obs_set = set(map(tuple, np.round(X_obs_raw, 8).tolist()))
    mask    = np.array([tuple(np.round(row, 8)) not in obs_set
                        for row in X_all.tolist()])
    X_cand_n    = X_all_n[mask]
    X_cand_raw  = X_all[mask]
    recon_cand  = recon_loss[mask] if recon_loss is not None else None
    print(f"Unobserved candidates : {mask.sum():,}")

    if mask.sum() == 0:
        print("All candidates have been observed. No next point to suggest.")
        return

    # ── fit GP ────────────────────────────────────────────────────────────────
    print(f"\nFitting GP (Matérn-5/2, {args.n_iters} iters) …")
    posterior, D = fit_gp(X_obs_n, y_obs, n_iters=args.n_iters, lr=args.lr)
    print("GP fit complete.")

    # ── predict variance at candidates ────────────────────────────────────────
    var = predict_variance(posterior, D, X_cand_n)

    # ── select next point = maximum variance (uncertainty sampling) ───────────
    best_idx = int(np.argmax(var))
    next_raw = X_cand_raw[best_idx]
    next_var = var[best_idx]
    dist     = _nearest_observed(X_cand_n[best_idx], X_obs_n)

    next_recon = f"  recon_loss = {recon_cand[best_idx]:.4e}" if recon_cand is not None else ""
    print(f"\n{'─'*60}")
    print(f"  Next candidate  σ²={next_var:.4e}  dist_obs={dist:.3f}{next_recon}")
    print(f"{'─'*60}")
    for n, v in zip(names, next_raw):
        print(f"  {n:20s}  {v:.6g}")
    print(f"{'─'*60}")

    # ── top-5 candidates ──────────────────────────────────────────────────────
    top5     = np.argsort(var)[::-1][:5]
    has_recon = recon_cand is not None
    hdr_recon = f"  {'recon':>10}" if has_recon else ""
    print(f"\nTop-5 by variance:")
    print(f"  {'rank':>4}  {'σ²':>12}  {'dist_obs':>10}{hdr_recon}  parameters")
    for rank, idx in enumerate(top5, 1):
        row   = X_cand_raw[idx]
        d     = _nearest_observed(X_cand_n[idx], X_obs_n)
        param = "  ".join(f"{n}={v:.4g}" for n, v in zip(names, row))
        rc    = f"  {recon_cand[idx]:>10.4e}" if has_recon else ""
        print(f"  {rank:>4}  {var[idx]:>12.4e}  {d:>10.3f}{rc}  {param}")

    # ── YAML snippet for next observation ─────────────────────────────────────
    print(f"\nAdd to '{args.config}' after running the simulation:")
    print(f"  - params:")
    for n, v in zip(names, next_raw):
        print(f"      {n}: {v:.6g}")
    print(f"    output: <result>   # fill in after simulation")


if __name__ == "__main__":
    main()
