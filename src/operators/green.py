"""Frequency grids and the isotropic reference-medium Green's operator."""

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)

from functools import partial

import jax.numpy as jnp
from jax.typing import ArrayLike

from materialmodels.tensors import isotropic_equivalent_lame
from operators.base import LinearOperator
from operators.fft_distributed import choose_device_count, shard_voxel_tensor
from operators.general_functions import ddot42


def build_freq_grid(
    n: tuple[int, ...],
    L: tuple[float, ...],
) -> jnp.ndarray:
    """
    Angular frequencies ``xi_i = 2 pi k_i / L_i`` for ``jnp.fft.fftn``.

    Parameters
    ----------
    n : voxel counts per dimension
    L : domain lengths per dimension

    Returns
    -------
    xi_flat : (len(n), Nv)
    """
    freqs = [jnp.fft.fftfreq(ni, d=Li / ni) * 2.0 * jnp.pi
             for ni, Li in zip(n, L)]
    grids = jnp.meshgrid(*freqs, indexing='ij')
    return jnp.stack([g.ravel() for g in grids])


def nyquist_safe_xi(xi_flat: jnp.ndarray, n: tuple[int, ...]) -> jnp.ndarray:
    """
    ``xi_flat`` with the Nyquist component zeroed along each even dimension.

    Needed by any operator with an odd power of xi (gradient, divergence): the
    Nyquist bin has no negative-frequency partner, so an imaginary factor there
    breaks Hermitian symmetry and corrupts the real-space result. The Green's
    operators use even powers only and don't need it.
    """
    idx   = [jnp.arange(ni) for ni in n]
    grids = jnp.meshgrid(*idx, indexing="ij")
    masks = [
        (g == ni // 2) if ni % 2 == 0 else jnp.zeros_like(g, dtype=bool)
        for ni, g in zip(n, grids)
    ]
    nyquist_mask = jnp.stack([m.ravel() for m in masks])
    return jnp.where(nyquist_mask, 0.0, xi_flat)


def build_willot_freq(
    xi_flat: jnp.ndarray,
    dx:      tuple[float, ...],
) -> jnp.ndarray:
    """
    Willot (2015) rotated-scheme frequencies ``xi_j^eff = (2/h_j) sin(xi_j h_j / 2)``.

    Matches xi at low wavenumbers and stays finite (2/h_j) at Nyquist; for an
    isotropic reference medium the complex phase factor cancels.

    Parameters
    ----------
    xi_flat : (ndim, Nv) frequencies from ``build_freq_grid``
    dx      : (ndim,) voxel spacing

    Returns
    -------
    xi_eff : (ndim, Nv)
    """
    dx_arr = jnp.array(dx)
    return (2.0 / dx_arr[:, None]) * jnp.sin(xi_flat * dx_arr[:, None] / 2.0)


def build_green_operator(
    xi_flat: jnp.ndarray,
    lam0:    ArrayLike,
    mu0:     ArrayLike,
    scheme:  str = 'standard',
    dx:      tuple[float, ...] | None = None,
) -> jnp.ndarray:
    """
    Green's operator ``G_ijkl`` for an isotropic reference medium (DBFFT convention):

        K_ik     = (d_ik - c n_i n_k) / mu0,   c = (lam0 + mu0) / (lam0 + 2 mu0)
        G_ijkl   = 1/4 (K_ik n_j n_l + 3 symmetric terms)

    Parameters
    ----------
    xi_flat : (3, Nv) frequencies from ``build_freq_grid``
    lam0, mu0 : reference Lame parameters
    scheme  : 'standard' (Moulinec-Suquet) or 'rotated' (Willot 2015; fewer
              Gibbs oscillations and no 45-degree bias, needs ``dx``)
    dx      : voxel spacing, required for 'rotated'

    Returns
    -------
    G : (3, 3, 3, 3, Nv), zero at xi = 0
    """
    if scheme == 'rotated':
        if dx is None:
            raise ValueError("dx must be provided for scheme='rotated'")
        xi = build_willot_freq(xi_flat, dx)
    else:
        xi = xi_flat

    # n_hat, not xi: Gamma0 is degree-0 homogeneous in xi, and using xi would
    # make the operator depend on the domain length L.
    xi_sq = jnp.sum(xi ** 2, axis=0)
    safe  = xi_sq > 0
    xi_s  = jnp.where(safe, xi_sq, 1.0)
    n_hat = xi / jnp.sqrt(xi_s)[None, :]

    c    = (lam0 + mu0) / (lam0 + 2.0 * mu0)
    d    = jnp.eye(3)
    Kinv = (d[:, :, None]
            - c * jnp.einsum('iN,jN->ijN', n_hat, n_hat)) / mu0

    nn    = jnp.einsum('iN,jN->ijN', n_hat, n_hat)
    Gamma = 0.25 * (
        jnp.einsum('ikN,jlN->ijklN', Kinv, nn) +
        jnp.einsum('ilN,jkN->ijklN', Kinv, nn) +
        jnp.einsum('jkN,ilN->ijklN', Kinv, nn) +
        jnp.einsum('jlN,ikN->ijklN', Kinv, nn)
    )
    return jnp.where(safe[None, None, None, None, :], Gamma, 0.0)


class GreenOperatorBasic(LinearOperator):
    """Green's operator, standard (Moulinec-Suquet) scheme. Self-adjoint."""

    def __init__(
        self, n: tuple[int, ...], L: tuple[float, ...], lam0: ArrayLike, mu0: ArrayLike,
        n_devices: int | None = None,
    ):
        self.n, self.L, self.lam0, self.mu0 = n, L, lam0, mu0
        self.n_devices = n_devices
        self.G = self._build_G()

    def _build_G(self) -> jnp.ndarray:
        xi_flat = build_freq_grid(self.n, self.L)
        resolved_n_devices = choose_device_count(self.n, self.n_devices)
        if resolved_n_devices > 1:
            build_local = partial(build_green_operator, lam0=self.lam0, mu0=self.mu0, scheme='standard')
            return shard_voxel_tensor(build_local, xi_flat, resolved_n_devices)
        return build_green_operator(xi_flat, self.lam0, self.mu0, scheme='standard')

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """x, result: (3, 3, Nv) Fourier-space field."""
        return ddot42(self.G, x)

    @property
    def T(self) -> LinearOperator:
        return self


class GreenOperatorWillot(GreenOperatorBasic):
    """Green's operator, Willot (2015) rotated scheme; needs voxel spacing ``dx``."""

    def __init__(
        self,
        n: tuple[int, ...],
        L: tuple[float, ...],
        lam0: ArrayLike,
        mu0: ArrayLike,
        dx: tuple[float, ...],
        n_devices: int | None = None,
    ):
        self.dx = dx
        super().__init__(n, L, lam0, mu0, n_devices=n_devices)

    def _build_G(self) -> jnp.ndarray:
        xi_flat = build_freq_grid(self.n, self.L)
        resolved_n_devices = choose_device_count(self.n, self.n_devices)
        if resolved_n_devices > 1:
            build_local = partial(build_green_operator, lam0=self.lam0, mu0=self.mu0, scheme='rotated', dx=self.dx)
            return shard_voxel_tensor(build_local, xi_flat, resolved_n_devices)
        return build_green_operator(xi_flat, self.lam0, self.mu0, scheme='rotated', dx=self.dx)


def build_reference_green_operator(
    n: tuple[int, ...],
    L: tuple[float, ...],
    materials: list,
    scheme: str = 'rotated',
    n_devices: int | None = None,
) -> GreenOperatorBasic:
    """
    Green's operator for the isotropic reference medium of ``materials``.

    The reference (lam0, mu0) is the isotropic projection of the mean
    stiffness, so anisotropic materials work too; a per-voxel stiffness field
    is averaged over voxels first.

    Parameters
    ----------
    n, L      : grid shape and domain size
    materials : each exposing ``elastic_stiffness_tensor()``
    scheme    : 'standard' (GreenOperatorBasic) or 'rotated' (GreenOperatorWillot)
    n_devices : device cap for building ``G`` sharded (None = all)

    Returns
    -------
    green_op : GreenOperatorBasic or GreenOperatorWillot
    """
    def _mean_stiffness(m) -> jnp.ndarray:
        C = m.elastic_stiffness_tensor()
        return jnp.mean(C, axis=-1) if C.ndim == 5 else C

    C_mean = jnp.mean(jnp.stack([_mean_stiffness(m) for m in materials]), axis=0)
    lam0, mu0 = isotropic_equivalent_lame(C_mean)

    if scheme == 'standard':
        return GreenOperatorBasic(n, L, lam0, mu0, n_devices=n_devices)
    elif scheme == 'rotated':
        dx = tuple(Li / ni for Li, ni in zip(L, n))
        return GreenOperatorWillot(n, L, lam0, mu0, dx, n_devices=n_devices)
    else:
        raise ValueError(f"unknown scheme {scheme!r}, expected 'standard' or 'rotated'")
