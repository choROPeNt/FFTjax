"""
Standalone test for learning.surrogates.GPSurrogate -- the shared GPJax
fitting/prediction wrapper factored out of scripts/active_learning.py's
fit_and_predict and notebooks/structure-property_phi-sweep.ipynb's fit_gp.

Four checks
-----------
1. 1-D fit (Matern52, the default kernel) on a smooth, noise-free function:
   predictions at held-out points must be close to the true function, and
   predict() before fit() must raise.
2. x_bounds override: fitting twice on the SAME data with two different
   x_bounds must normalize to two different internal coordinate spaces
   (checked directly on _X_lo/_X_hi) -- this is what lets an active-learning
   loop keep a stable normalized space across repeated fits on a growing
   observed subset, instead of it shifting every time a new point arrives.
3. Deterministic refit: two independent GPSurrogate instances fit on
   identical data with the default (fixed) key must produce identical
   predictions -- required for notebooks/structure-property_phi-sweep.ipynb's
   sequential-refit loop, where the same data must always fit to the same
   hyperparameters.
4. 4-D SumKernel(Linear+Matern32) cross-check against a standalone,
   hand-inlined copy of scripts/active_learning.py's ORIGINAL (pre-refactor)
   fit_and_predict logic, on the same synthetic data and fixed key -- must
   agree on mean/variance/hyperparameters to high precision, confirming the
   refactor of that script onto GPSurrogate preserves its exact behaviour.

Usage
-----
    python test/test_learning_surrogates.py
"""

import sys
sys.path.insert(0, "src")

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)
import jax
import jax.numpy as jnp
import numpy as np
import optax

import gpjax as gpx

from learning.surrogates import GPSurrogate

rng = np.random.default_rng(0)

# ── 1. 1-D fit on a smooth function ─────────────────────────────────────────

X_train = np.linspace(0.0, 1.0, 10)
y_train = np.sin(2.0 * np.pi * X_train) + 0.5 * X_train

fresh = GPSurrogate(num_iters=300)
try:
    fresh.predict(np.array([0.5]))
    raise AssertionError("predict() before fit() should have raised")
except RuntimeError:
    pass

surrogate = GPSurrogate(num_iters=300).fit(X_train, y_train, verbose=False)
X_test = np.linspace(0.05, 0.95, 25)
y_true = np.sin(2.0 * np.pi * X_test) + 0.5 * X_test
mean_pred, var_pred = surrogate.predict(X_test)

max_err = float(np.max(np.abs(mean_pred - y_true)))
print(f"[1] 1-D fit: max|GP mean - true| over 25 held-out points = {max_err:.3e}")
assert max_err < 0.15, "GP should interpolate this smooth function reasonably well"
assert np.all(var_pred >= 0.0), "predictive variance must be non-negative"
print("[1] PASSED")

# ── 2. x_bounds override changes the normalized coordinate space ───────────

s_default = GPSurrogate(num_iters=50).fit(X_train, y_train, verbose=False)
s_wide = GPSurrogate(num_iters=50).fit(X_train, y_train, x_bounds=(np.array([-1.0]), np.array([2.0])), verbose=False)

print(f"[2] default bounds: lo={s_default._X_lo}, hi={s_default._X_hi}  "
      f"explicit bounds: lo={s_wide._X_lo}, hi={s_wide._X_hi}")
assert np.allclose(s_default._X_lo, X_train.min()) and np.allclose(s_default._X_hi, X_train.max())
assert np.allclose(s_wide._X_lo, -1.0) and np.allclose(s_wide._X_hi, 2.0)
print("[2] PASSED")

# ── 3. deterministic refit (fixed default key) ──────────────────────────────

s_a = GPSurrogate(num_iters=200).fit(X_train, y_train, verbose=False)
s_b = GPSurrogate(num_iters=200).fit(X_train, y_train, verbose=False)
mean_a, var_a = s_a.predict(X_test)
mean_b, var_b = s_b.predict(X_test)

diff = float(np.max(np.abs(mean_a - mean_b)))
print(f"[3] two independent fits on identical data: max|mean_a - mean_b| = {diff:.3e}")
assert diff < 1e-8, "fits with the same default key on the same data must be deterministic"
assert np.allclose(var_a, var_b, atol=1e-10)
print("[3] PASSED")

# ── 4. 4-D SumKernel cross-check against the pre-refactor reference ────────


def _reference_fit_and_predict(X_obs, y_obs, X_cand, n_iters=300, lr=0.01):
    """
    Standalone copy of scripts/active_learning.py's ORIGINAL fit_and_predict
    body (pre-GPSurrogate-refactor), kept here ONLY as a correctness
    reference for this test -- not imported from the script, so this test
    still catches a regression even if the script's own copy is edited or
    removed later.
    """
    def _normalize(X, lo, hi):
        return (X - lo) / np.where(hi > lo, hi - lo, 1.0)

    lo, hi = X_obs.min(0), X_obs.max(0)
    X_obs_n = _normalize(X_obs, lo, hi)
    X_cand_n = _normalize(X_cand, lo, hi)

    key = jax.random.PRNGKey(0)
    X = jnp.array(X_obs_n)
    y = jnp.array(y_obs)
    D = gpx.Dataset(X=X, y=y)

    dims = list(range(X.shape[1]))
    kernel = gpx.kernels.SumKernel(kernels=[
        gpx.kernels.Linear(active_dims=dims),
        gpx.kernels.Matern32(active_dims=dims),
    ])
    meanf = gpx.mean_functions.Constant()
    try:
        prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)
    except AttributeError:
        prior = gpx.Prior(mean_function=meanf, kernel=kernel)
    likelihood = gpx.likelihoods.Gaussian(num_datapoints=D.n)
    posterior = prior * likelihood

    opt_post, _ = gpx.fit(
        model=posterior,
        objective=lambda p, d: -gpx.objectives.conjugate_mll(p, d),
        train_data=D,
        optim=optax.adam(lr),
        num_iters=n_iters,
        key=key,
        safe=True,
        verbose=False,
    )
    pred = opt_post.predict(jnp.array(X_cand_n), train_data=D)

    def _unwrap(p):
        return p.unwrap() if hasattr(p, "unwrap") else jnp.asarray(p)

    k_linear, k_matern = opt_post.prior.kernel.kernels
    hyperparams = {
        "linear_variance": float(_unwrap(k_linear.variance)),
        "matern_lengthscale": np.array(_unwrap(k_matern.lengthscale)),
        "matern_variance": float(_unwrap(k_matern.variance)),
        "noise_variance": float(_unwrap(opt_post.likelihood.obs_stddev) ** 2),
        "mean_constant": float(_unwrap(opt_post.prior.mean_function.constant)),
    }
    return np.array(pred.mean), np.array(pred.variance), hyperparams


X_obs_4d = rng.normal(size=(15, 4))
y_obs_4d = (X_obs_4d[:, 0] - 0.5 * X_obs_4d[:, 1] + 0.2 * X_obs_4d[:, 2] ** 2).reshape(-1, 1)
X_cand_4d = rng.normal(size=(8, 4))

lo4, hi4 = X_obs_4d.min(0), X_obs_4d.max(0)
dims4 = list(range(4))
kernel4 = gpx.kernels.SumKernel(kernels=[
    gpx.kernels.Linear(active_dims=dims4),
    gpx.kernels.Matern32(active_dims=dims4),
])
surrogate4 = GPSurrogate(kernel=kernel4, num_iters=300, lr=0.01, standardize_y=False).fit(
    X_obs_4d, y_obs_4d, x_bounds=(lo4, hi4), verbose=False,
)
mean_new, var_new = surrogate4.predict(X_cand_4d)

mean_ref, var_ref, hp_ref = _reference_fit_and_predict(X_obs_4d, y_obs_4d, X_cand_4d, n_iters=300, lr=0.01)

mean_diff = float(np.max(np.abs(mean_new - mean_ref.reshape(-1))))
var_diff = float(np.max(np.abs(var_new - var_ref.reshape(-1))))
print(f"[4] GPSurrogate vs. reference fit_and_predict: "
      f"max|mean diff|={mean_diff:.3e}  max|var diff|={var_diff:.3e}")
assert mean_diff < 1e-6, "GPSurrogate must reproduce the pre-refactor mean exactly"
assert var_diff < 1e-6, "GPSurrogate must reproduce the pre-refactor variance exactly"

k_linear4, k_matern4 = surrogate4.posterior.prior.kernel.kernels


def _unwrap(p):
    return p.unwrap() if hasattr(p, "unwrap") else jnp.asarray(p)


hp_new = {
    "linear_variance": float(_unwrap(k_linear4.variance)),
    "matern_lengthscale": np.array(_unwrap(k_matern4.lengthscale)),
    "matern_variance": float(_unwrap(k_matern4.variance)),
    "noise_variance": float(_unwrap(surrogate4.posterior.likelihood.obs_stddev) ** 2),
    "mean_constant": float(_unwrap(surrogate4.posterior.prior.mean_function.constant)),
}
for key_name in hp_ref:
    a, b = np.asarray(hp_ref[key_name]), np.asarray(hp_new[key_name])
    assert np.allclose(a, b, atol=1e-6), f"hyperparameter {key_name!r} mismatch: {a} vs {b}"
print("[4] PASSED")

print("\ntest_learning_surrogates: all checks passed")
