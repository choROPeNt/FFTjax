"""
Gaussian-process surrogate over a structure-property relationship, shared by
every sweep/active-learning workflow in this project instead of each one
hand-rolling its own gpx.Dataset/gpx.fit/predict block. Consolidates
scripts/active_learning.py's fit_and_predict and notebooks/
structure-property_phi-sweep.ipynb's fit_gp -- both were the same GPJax
pattern (normalize, gpx.Dataset, prior*likelihood, gpx.fit, predict,
un-normalize) with only the kernel and input dimensionality differing.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax

import gpjax as gpx


def _minmax(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    span = np.where(hi > lo, hi - lo, 1.0)
    return (x - lo) / span


class GPSurrogate:
    """
    Exact GP regression on (X, y) pairs, X: (n, d), y: (n,) or (n, 1).

    Normalizes X to [0, 1] per dimension (min-max) and y to zero-mean/
    unit-variance before fitting -- purely for optimizer conditioning;
    predict() undoes both transforms, so callers work in original units
    throughout.

    Parameters
    ----------
    kernel        : a gpjax kernel instance, e.g. gpx.kernels.Matern52()
                    (default) or a composite one, e.g. gpx.kernels.SumKernel(
                    kernels=[gpx.kernels.Linear(...), gpx.kernels.Matern32(...)])
                    as scripts/active_learning.py uses for its 4-D latent space.
    mean_function : gpjax mean function. Default: gpx.mean_functions.Constant().
    num_iters, lr : Adam optimizer steps / learning rate for gpx.fit.
    key           : jax PRNGKey for gpx.fit. Default: jax.random.PRNGKey(0)
                    -- deterministic refits (no randomness beyond the fixed
                    key), which matters for a sequential-refit loop (e.g.
                    notebooks/structure-property_phi-sweep.ipynb), where the
                    same data must always fit to the same hyperparameters.
    standardize_y : bool, default True -- zero-mean/unit-variance y before
                    fitting (undone in predict()). Pass False to fit on raw
                    y instead, matching scripts/active_learning.py's
                    original fit_and_predict, which never standardized y
                    (only X, externally, against a fixed candidate-pool
                    range -- see x_bounds below); True is the better default
                    for a fresh caller, since it conditions the optimizer
                    the same way regardless of y's own units/scale.
    """

    def __init__(self, kernel=None, mean_function=None, num_iters: int = 500,
                 lr: float = 0.01, key=None, standardize_y: bool = True):
        self.kernel = kernel if kernel is not None else gpx.kernels.Matern52()
        self.mean_function = mean_function if mean_function is not None else gpx.mean_functions.Constant()
        self.num_iters = num_iters
        self.lr = lr
        self.key = key if key is not None else jax.random.PRNGKey(0)
        self.standardize_y = standardize_y

        self.posterior = None    # fitted posterior, set by fit()
        self.train_data = None   # gpx.Dataset, set by fit()
        self.history = None      # objective trace returned by gpx.fit
        self._X_lo = None
        self._X_hi = None
        self._y_mean = None
        self._y_std = None

    def fit(self, X, y, x_bounds: tuple | None = None, verbose: bool = True) -> "GPSurrogate":
        """
        Parameters
        ----------
        X : (n, d) array-like -- a 1-D (n,) input is treated as n samples of
            one feature, reshaped to (n, 1)
        y : (n,) or (n, 1) array-like
        x_bounds : (lo, hi) each (d,), overriding the min-max bounds this
            call would otherwise compute from X itself -- pass the FULL
            candidate pool's bounds (not just the observed subset's) when
            fitting repeatedly on a growing subset, e.g. an active-learning
            loop, so the normalized input space stays stable across calls
            instead of shifting every time a new point arrives. None
            (default) computes bounds from X, correct for a one-shot fit.
        verbose : forwarded to gpx.fit -- False silences its per-fit
            progress bar, useful when fit() is called repeatedly (e.g. once
            per update in a sweep loop).

        Returns
        -------
        self, so ``GPSurrogate(...).fit(X, y)`` chains.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.asarray(y, dtype=float).reshape(-1, 1)

        if x_bounds is not None:
            self._X_lo = np.asarray(x_bounds[0], dtype=float)
            self._X_hi = np.asarray(x_bounds[1], dtype=float)
        else:
            self._X_lo = X.min(axis=0)
            self._X_hi = X.max(axis=0)
        if self.standardize_y:
            self._y_mean = float(y.mean())
            self._y_std = float(y.std() + 1e-12)
        else:
            self._y_mean = 0.0
            self._y_std = 1.0

        Xn = jnp.array(_minmax(X, self._X_lo, self._X_hi))
        yn = jnp.array((y - self._y_mean) / self._y_std)
        D = gpx.Dataset(X=Xn, y=yn)

        try:
            prior = gpx.gps.Prior(mean_function=self.mean_function, kernel=self.kernel)
        except AttributeError:
            prior = gpx.Prior(mean_function=self.mean_function, kernel=self.kernel)  # type: ignore[attr-defined]
        likelihood = gpx.likelihoods.Gaussian(num_datapoints=D.n)
        posterior = prior * likelihood

        self.posterior, self.history = gpx.fit(
            model=posterior,
            objective=lambda p, d: -gpx.objectives.conjugate_mll(p, d),
            train_data=D,
            optim=optax.adam(self.lr),
            num_iters=self.num_iters,
            key=self.key,
            safe=True,
            verbose=verbose,
        )
        self.train_data = D
        return self

    def predict(self, X_new):
        """
        X_new : (m, d) array-like. A 1-D input is reshaped to (m, 1) if the
            training data was 1-D (d == 1, the common sweep-over-one-
            parameter case), or to (1, d) otherwise (one d-dimensional
            query point).

        Returns
        -------
        mean, variance : (m,) arrays, in the original (unnormalized) units
            -- take sqrt(variance) for the standard deviation, e.g. for a
            +/-2*sqrt(variance) uncertainty band.
        """
        if self.posterior is None:
            raise RuntimeError("GPSurrogate.predict() called before fit()")

        X_new = np.asarray(X_new, dtype=float)
        d = self._X_lo.shape[0]
        if X_new.ndim == 1:
            X_new = X_new.reshape(-1, 1) if d == 1 else X_new.reshape(1, -1)

        Xn = jnp.array(_minmax(X_new, self._X_lo, self._X_hi))
        pred = self.posterior.predict(Xn, train_data=self.train_data)  # type: ignore[union-attr]

        mean = np.array(pred.mean).reshape(-1) * self._y_std + self._y_mean
        variance = np.array(pred.variance).reshape(-1) * (self._y_std ** 2)
        return mean, variance
