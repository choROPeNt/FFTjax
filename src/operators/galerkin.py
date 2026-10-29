"""
Reference-medium-free compatibility projector for the Fourier-Galerkin scheme
(Vondrejc et al 2014; Lucarini, Upadhyay & Segurado 2022, sec. 3.3).

Not a special case of operators.green: no (lam0, mu0) reproduces it with that
operator's ansatz (see notes/FOURIER_GALERKIN.md).
"""

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)

from functools import partial

import jax.numpy as jnp

from operators.base import LinearOperator
from operators.fft_distributed import choose_device_count, shard_voxel_tensor
from operators.general_functions import ddot42
from operators.green import build_freq_grid, build_willot_freq


def build_galerkin_projection_operator(
    xi_flat: jnp.ndarray,
    scheme:  str = 'standard',
    dx:      tuple[float, ...] | None = None,
) -> jnp.ndarray:
    """
    Symmetric projector onto compatible (curl-free) fields; depends only on n = xi/|xi|:

        G_ijkl = 1/2 (d_ik n_j n_l + d_il n_j n_k + d_jk n_i n_l + d_jl n_i n_k) - n_i n_j n_k n_l

    Parameters
    ----------
    xi_flat : (3, Nv) frequencies from build_freq_grid
    scheme  : 'standard' or 'rotated' (as build_green_operator)
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

    xi_sq = jnp.sum(xi ** 2, axis=0)
    safe  = xi_sq > 0
    xi_s  = jnp.where(safe, xi_sq, 1.0)
    n_hat = xi / jnp.sqrt(xi_s)[None, :]

    d  = jnp.eye(3)
    nn = jnp.einsum('iN,jN->ijN', n_hat, n_hat)
    G = 0.5 * (
        jnp.einsum('ik,jlN->ijklN', d, nn) +
        jnp.einsum('il,jkN->ijklN', d, nn) +
        jnp.einsum('jk,ilN->ijklN', d, nn) +
        jnp.einsum('jl,ikN->ijklN', d, nn)
    ) - jnp.einsum('iN,jN,kN,lN->ijklN', n_hat, n_hat, n_hat, n_hat)
    return jnp.where(safe[None, None, None, None, :], G, 0.0)


class GalerkinProjector(LinearOperator):
    """Compatibility projector (no reference medium). Self-adjoint."""

    def __init__(
        self, n: tuple[int, ...], L: tuple[float, ...], scheme: str = 'rotated',
        n_devices: int | None = None,
    ):
        self.n, self.L, self.scheme = n, L, scheme
        self.n_devices = n_devices
        self.G = self._build_G()

    def _build_G(self) -> jnp.ndarray:
        xi_flat = build_freq_grid(self.n, self.L)
        dx = tuple(Li / ni for Li, ni in zip(self.L, self.n)) if self.scheme == 'rotated' else None
        resolved_n_devices = choose_device_count(self.n, self.n_devices)
        if resolved_n_devices > 1:
            build_local = partial(build_galerkin_projection_operator, scheme=self.scheme, dx=dx)
            return shard_voxel_tensor(build_local, xi_flat, resolved_n_devices)
        return build_galerkin_projection_operator(xi_flat, scheme=self.scheme, dx=dx)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """x, result: (3, 3, Nv) Fourier-space field."""
        return ddot42(self.G, x)

    @property
    def T(self) -> LinearOperator:
        return self


def build_galerkin_projector(
    n: tuple[int, ...],
    L: tuple[float, ...],
    scheme: str = 'rotated',
    n_devices: int | None = None,
) -> GalerkinProjector:
    """
    GalerkinProjector for a grid; the materials-free counterpart of
    build_reference_green_operator.

    Parameters
    ----------
    n, L      : grid shape and domain size
    scheme    : 'standard' or 'rotated'
    n_devices : device cap for building ``G`` sharded (None = all)
    """
    if scheme not in ('standard', 'rotated'):
        raise ValueError(f"unknown scheme {scheme!r}, expected 'standard' or 'rotated'")
    return GalerkinProjector(n, L, scheme=scheme, n_devices=n_devices)
