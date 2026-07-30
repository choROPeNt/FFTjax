"""
Post-processing field utilities for FFTjax.

All functions accept the solver's native layout — fields with shape
``(3, 3, Nv)`` or ``(3, 3, 3, 3, Nv)`` — and return arrays reshaped to
``(*grid_n, ...)`` suitable for ParaView export via ``post.io``.
"""

import numpy as np
import jax.numpy as jnp


def field_to_grid(arr_33_nv: jnp.ndarray, grid_n: tuple) -> np.ndarray:
    """
    Reshape a 2nd-order tensor field from solver layout to grid layout.

    Parameters
    ----------
    arr_33_nv : (3, 3, Nv)   JAX or numpy array (solver layout)
    grid_n    : tuple          grid shape, e.g. (nx, ny, nz)

    Returns
    -------
    out : (*grid_n, 3, 3) numpy array
    """
    ndim = len(grid_n)
    return (np.asarray(arr_33_nv)
            .reshape(3, 3, *grid_n)
            .transpose(*range(2, 2 + ndim), 0, 1))


def von_mises(sigma_grid: np.ndarray) -> np.ndarray:
    """
    Von Mises equivalent stress.

    Parameters
    ----------
    sigma_grid : ``(*n, 3, 3)``   stress tensor field on the grid

    Returns
    -------
    sigma_vm : (*n,)   von Mises stress, same leading shape as sigma_grid
    """
    s = sigma_grid
    return np.sqrt(0.5 * (
        (s[..., 0, 0] - s[..., 1, 1]) ** 2 +
        (s[..., 1, 1] - s[..., 2, 2]) ** 2 +
        (s[..., 2, 2] - s[..., 0, 0]) ** 2 +
        6.0 * (s[..., 0, 1] ** 2 + s[..., 0, 2] ** 2 + s[..., 1, 2] ** 2)
    ))


def compute_displacement(
    eps: jnp.ndarray,
    eps_bar: jnp.ndarray,
    xi_flat: jnp.ndarray,
    grid_n: tuple,
    grid_dx: tuple,
) -> np.ndarray:
    """
    Recover the displacement field from a compatible strain field via Fourier
    integration.

    Exact inversion of ``ε̂_ij = (iξ_j û_i + iξ_i û_j) / 2`` for ξ ≠ 0:

    ::

        û_i = -2i/|ξ|² Σ_j Ê'_ij ξ_j  +  i ξ_i (Σ_{jk} ξ_j Ê'_jk ξ_k) / |ξ|⁴

    The second (longitudinal correction) term is zero for divergence-free
    (incompressible) fields; it is required for the compressible elastic case.

    Total displacement  =  macroscopic part (ε̄_ij x_j)  +  periodic fluctuation.

    Parameters
    ----------
    eps     : (3, 3, Nv)   total strain field (solver layout)
    eps_bar : (3, 3)        macroscopic strain tensor
    xi_flat : (3, Nv)       angular-frequency grid from ``operators.green.build_freq_grid``
    grid_n  : tuple          grid shape (nx, ny, nz)
    grid_dx : tuple          voxel spacing per dimension (dx, dy, dz)  [µm]

    Returns
    -------
    u : (*grid_n, 3)   displacement field in physical units (same as grid_dx)
    """
    ndim     = len(grid_n)
    fft_axes = tuple(range(-ndim, 0))
    Nv       = int(np.prod(grid_n))

    xi_sq = jnp.sum(xi_flat ** 2, axis=0)
    safe  = xi_sq > 0
    xi_s  = jnp.where(safe, xi_sq, 1.0)

    # Strain fluctuation and its DFT
    eps_fluct = eps - eps_bar[:, :, None]
    eps_hat   = jnp.fft.fftn(
        eps_fluct.reshape(3, 3, *grid_n), axes=fft_axes
    ).reshape(3, 3, Nv)

    eps_xi    = jnp.einsum('ijm,jm->im',    eps_hat, xi_flat)           # (3, Nv)
    xi_eps_xi = jnp.einsum('jm,jkm,km->m', xi_flat, eps_hat, xi_flat)  # (Nv,)

    u_hat = (-2j * eps_xi / xi_s[None, :]
             + 1j * xi_flat * xi_eps_xi[None, :] / xi_s[None, :] ** 2)
    u_hat = jnp.where(safe[None, :], u_hat, 0.0 + 0.0j)

    u_fluct = jnp.fft.ifftn(
        u_hat.reshape(3, *grid_n), axes=fft_axes
    ).real.reshape(3, Nv)

    # Macroscopic contribution: u_i^macro = ε̄_ij x_j  at voxel centres
    x_c     = [jnp.arange(grid_n[k]) * grid_dx[k] + grid_dx[k] / 2
               for k in range(ndim)]
    X       = jnp.stack([g.ravel() for g in jnp.meshgrid(*x_c, indexing='ij')])
    u_macro = jnp.einsum('ij,jm->im', eps_bar, X)

    u_total = u_macro + u_fluct
    return np.asarray(u_total).reshape(3, *grid_n).transpose(*range(1, ndim + 1), 0)
