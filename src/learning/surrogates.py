"""Gaussian-process surrogate (GPJax) for structure-property sweeps and active learning."""

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
    Exact GP regression on X (n, d), y (n,). X is min-max normalized and y
    standardized for conditioning; predict() returns original units.

    Parameters
    ----------
    kernel        : gpjax kernel (default Matern52)
    mean_function : gpjax mean function (default Constant)
    num_iters, lr : Adam steps and learning rate
    key           : PRNG key (default PRNGKey(0), so refits are deterministic)
    standardize_y : standardize y (False reproduces raw-y fits)
    """

    def __init__(self, kernel=None, mean_function=None, num_iters: int = 500,
                 lr: float = 0.01, key=None, standardize_y: bool = True):
        self.kernel = kernel if kernel is not None else gpx.kernels.Matern52()
        self.mean_function = mean_function if mean_function is not None else gpx.mean_functions.Constant()
        self.num_iters = num_iters
        self.lr = lr
        self.key = key if key is not None else jax.random.PRNGKey(0)
        self.standardize_y = standardize_y

        self.posterior = None
        self.train_data = None
        self.history = None
        self._X_lo = None
        self._X_hi = None
        self._y_mean = None
        self._y_std = None

    def fit(self, X, y, x_bounds: tuple | None = None, verbose: bool = True) -> "GPSurrogate":
        """
        Fit on (X, y); a 1-D X is n samples of one feature.

        ``x_bounds = (lo, hi)`` fixes the normalization (pass the full
        candidate pool's bounds in a growing active-learning loop so it stays
        stable). ``verbose=False`` silences gpx.fit. Returns self.
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
        Posterior mean and variance at X_new (m, d), in original units; a 1-D
        X_new is m points for d = 1, else one point.
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
