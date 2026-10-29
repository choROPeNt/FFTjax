"""
Post-processing of solver fields: solver layout (3, 3, Nv) in, grid layout
(*n, ...) out, ready for utils.io.xdmf_writer. Voigt order 11, 22, 33, 12, 13, 23.
"""

from typing import Mapping, Sequence

import numpy as np
import jax.numpy as jnp
from jax.typing import ArrayLike

from operators.green import build_freq_grid

_VOIGT_IJ = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))


def to_voigt(tensor: np.ndarray, mandel: bool = False) -> np.ndarray:
    """
    Symmetric (..., 3, 3) -> Voigt (..., 6), order 11, 22, 33, 12, 13, 23.

    ``mandel=True`` scales shears by sqrt(2) (norm-preserving); keep False for stress.
    """
    tensor = np.asarray(tensor)
    if tensor.shape[-2:] != (3, 3):
        raise ValueError(f"Expected (..., 3, 3), got {tensor.shape}")

    voigt = np.stack([tensor[..., i, j] for i, j in _VOIGT_IJ], axis=-1)

    if mandel:
        voigt[..., 3:] *= np.sqrt(2.0)

    return voigt


def from_voigt(voigt: np.ndarray, mandel: bool = False) -> np.ndarray:
    """Voigt (..., 6), order 11, 22, 33, 12, 13, 23 -> symmetric (..., 3, 3); inverse of to_voigt."""
    voigt = np.asarray(voigt, dtype=float)
    if voigt.shape[-1] != 6:
        raise ValueError(f"Expected (..., 6), got {voigt.shape}")

    if mandel:
        voigt = voigt.copy()
        voigt[..., 3:] /= np.sqrt(2.0)

    out = np.zeros(voigt.shape[:-1] + (3, 3), dtype=voigt.dtype)
    for k, (i, j) in enumerate(_VOIGT_IJ):
        out[..., i, j] = voigt[..., k]
        out[..., j, i] = voigt[..., k]

    return out


def field_to_grid(arr_33_nv: jnp.ndarray, grid_n: tuple) -> np.ndarray:
    """(3, 3, Nv) -> (*grid_n, 3, 3)."""
    ndim = len(grid_n)
    return (np.asarray(arr_33_nv)
            .reshape(3, 3, *grid_n)
            .transpose(*range(2, 2 + ndim), 0, 1))


def vector_field_to_grid(arr_3_nv: jnp.ndarray, grid_n: tuple) -> np.ndarray:
    """(3, Nv) -> (*grid_n, 3)."""
    ndim = len(grid_n)
    return (np.asarray(arr_3_nv)
            .reshape(3, *grid_n)
            .transpose(*range(1, 1 + ndim), 0))


def von_mises(sigma_grid: np.ndarray) -> np.ndarray:
    """Von Mises stress of a (..., 3, 3) field -> (...)."""
    s = sigma_grid
    return np.sqrt(0.5 * (
        (s[..., 0, 0] - s[..., 1, 1]) ** 2 +
        (s[..., 1, 1] - s[..., 2, 2]) ** 2 +
        (s[..., 2, 2] - s[..., 0, 0]) ** 2 +
        6.0 * (s[..., 0, 1] ** 2 + s[..., 0, 2] ** 2 + s[..., 1, 2] ** 2)
    ))


def homogenize(eps: jnp.ndarray, sigma: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Voxel means of (3, 3, Nv) strain and stress (equal voxel volumes) -> (3, 3) each."""
    return jnp.mean(eps, axis=-1), jnp.mean(sigma, axis=-1)


def macroscopic_response(
    eps_bar: jnp.ndarray,
    sigma_bar: jnp.ndarray,
    *,
    scalars: Mapping[str, ArrayLike] | None = None,
    tensors: Mapping[str, ArrayLike] | None = None,
) -> dict[str, np.ndarray]:
    """
    One increment's macroscopic response as a flat dict; shared by the live
    stats rows and post.homogenize so the two cannot drift apart.

    Parameters
    ----------
    eps_bar, sigma_bar : (3, 3) averaged strain and stress
    scalars : ``{name: per-voxel array}`` -> ``{name}_mean``, ``{name}_max``
    tensors : ``{name: (3, 3) averaged tensor}`` -> ``{name}_bar`` (Voigt)
              and ``{name}_vol`` (trace)

    Returns
    -------
    dict with "eps_bar", "sigma_bar" (Voigt), "mises_stress", "pressure"
    (tr(sigma)/3, > 0 in tension) and the extras
    """
    eps_v   = to_voigt(np.asarray(eps_bar))
    sigma_v = to_voigt(np.asarray(sigma_bar))
    out: dict[str, np.ndarray] = {
        "eps_bar":      eps_v,
        "sigma_bar":    sigma_v,
        "mises_stress": float(von_mises(np.asarray(sigma_bar))),
        "pressure":     float((sigma_v[0] + sigma_v[1] + sigma_v[2]) / 3.0),
    }
    for name, field in (scalars or {}).items():
        arr = np.asarray(field)
        out[f"{name}_mean"] = float(arr.mean())
        out[f"{name}_max"]  = float(arr.max())
    for name, bar in (tensors or {}).items():
        v = to_voigt(np.asarray(bar))
        out[f"{name}_bar"] = v
        out[f"{name}_vol"] = float(v[0] + v[1] + v[2])
    return out


def macroscopic_thermal_response(
    grad_T_bar: jnp.ndarray,
    flux_bar: jnp.ndarray,
    *,
    scalars: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """
    Thermal analogue of ``macroscopic_response`` (vectors, no Voigt form).

    Returns
    -------
    dict with "grad_T_bar", "flux_bar" (3,), "flux_magnitude" |flux_bar| and
    ``{name}_mean``/``{name}_max`` per scalar
    """
    grad_v  = np.asarray(grad_T_bar)
    flux_v  = np.asarray(flux_bar)
    out: dict[str, np.ndarray] = {
        "grad_T_bar":     grad_v,
        "flux_bar":       flux_v,
        "flux_magnitude": float(np.linalg.norm(flux_v)),
    }
    for name, field in (scalars or {}).items():
        arr = np.asarray(field)
        out[f"{name}_mean"] = float(arr.mean())
        out[f"{name}_max"]  = float(arr.max())
    return out


def homogenize_response(
    results: Sequence,
    component: tuple[int, int] = (0, 0),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Averaged (eps, sigma) history of one component over load-stepped results.

    Works for stress-controlled components too, where the strain is solved.

    Parameters
    ----------
    results   : IncrementResults, or objects with ``.eps``/``.sigma``
    component : (i, j), e.g. (0, 0)

    Returns
    -------
    eps_bar, sigma_bar : (len(results),)
    """
    i, j = component
    solutions = [getattr(r, "solution", r) for r in results]
    eps_bar = np.array([float(jnp.mean(s.eps[i, j])) for s in solutions])
    sigma_bar = np.array([float(jnp.mean(s.sigma[i, j])) for s in solutions])
    return eps_bar, sigma_bar


def compute_displacement(
    eps: jnp.ndarray,
    eps_bar: jnp.ndarray,
    grid_n: tuple,
    grid_L: tuple,
) -> np.ndarray:
    """
    Displacement from a compatible strain field by Fourier integration:

        u_i = -2i/|xi|^2 E'_ij xi_j + i xi_i (xi_j E'_jk xi_k) / |xi|^4

    (the second term vanishes only for incompressible fields), plus the
    macroscopic part eps_bar . x at voxel centers.

    Parameters
    ----------
    eps     : (3, 3, Nv) total strain
    eps_bar : (3, 3) macroscopic strain
    grid_n  : grid shape
    grid_L  : domain size

    Returns
    -------
    u : (*grid_n, 3), in the units of grid_L
    """
    ndim     = len(grid_n)
    fft_axes = tuple(range(-ndim, 0))
    Nv       = int(np.prod(grid_n))

    grid_dx = tuple(Li / ni for Li, ni in zip(grid_L, grid_n))
    xi_flat = build_freq_grid(grid_n, grid_L)

    xi_sq = jnp.sum(xi_flat ** 2, axis=0)
    safe  = xi_sq > 0
    xi_s  = jnp.where(safe, xi_sq, 1.0)

    eps_fluct = eps - eps_bar[:, :, None]
    eps_hat   = jnp.fft.fftn(
        eps_fluct.reshape(3, 3, *grid_n), axes=fft_axes
    ).reshape(3, 3, Nv)

    eps_xi    = jnp.einsum('ijm,jm->im',    eps_hat, xi_flat)
    xi_eps_xi = jnp.einsum('jm,jkm,km->m', xi_flat, eps_hat, xi_flat)

    u_hat = (-2j * eps_xi / xi_s[None, :]
             + 1j * xi_flat * xi_eps_xi[None, :] / xi_s[None, :] ** 2)
    u_hat = jnp.where(safe[None, :], u_hat, 0.0 + 0.0j)

    u_fluct = jnp.fft.ifftn(
        u_hat.reshape(3, *grid_n), axes=fft_axes
    ).real.reshape(3, Nv)

    x_c     = [jnp.arange(grid_n[k]) * grid_dx[k] + grid_dx[k] / 2
               for k in range(ndim)]
    X       = jnp.stack([g.ravel() for g in jnp.meshgrid(*x_c, indexing='ij')])
    u_macro = jnp.einsum('ij,jm->im', eps_bar, X)

    u_total = u_macro + u_fluct
    return np.asarray(u_total).reshape(3, *grid_n).transpose(*range(1, ndim + 1), 0)
