import os
os.environ["JAX_ENABLE_X64"] = "1"

import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Full-spectrum frequency grid  (for fftn / ifftn solvers)
# ---------------------------------------------------------------------------

def build_freq_grid(
    n: tuple[int, ...],
    L: tuple[float, ...],
) -> jnp.ndarray:
    """
    Angular-frequency grid ξ for a periodic domain, full-spectrum (fftn).

    Returns the flat array of wave-vectors for use with ``jnp.fft.fftn``.
    Each component is  ξ_i = 2π k_i / L_i  with k_i from ``jnp.fft.fftfreq``.

    Parameters
    ----------
    n : tuple of int    voxel counts per dimension, e.g. (nx, ny, nz)
    L : tuple of float  physical domain lengths per dimension [µm]

    Returns
    -------
    xi_flat : (len(n), Nv)   Nv = prod(n)
    """
    freqs = [jnp.fft.fftfreq(ni, d=Li / ni) * 2.0 * jnp.pi
             for ni, Li in zip(n, L)]
    grids = jnp.meshgrid(*freqs, indexing='ij')
    return jnp.stack([g.ravel() for g in grids])  # (ndim, Nv)


# ---------------------------------------------------------------------------
# DBFFT Green's operator  Γ̂₀(ξ)  for isotropic reference medium
# ---------------------------------------------------------------------------

def build_green_operator(
    xi_flat: jnp.ndarray,
    lam0: float,
    mu0: float,
) -> jnp.ndarray:
    """
    4th-order Green's operator Γ̂₀_ijkl(ξ) for an isotropic reference medium.

    Uses the DBFFT / FFTMAD convention (Lucarini & Segurado 2018):

        K_eff_ik  = (δ_ik − c n̂_i n̂_k) / μ₀     (no 1/|ξ|² factor)
        Γ̂₀_ijkl  = ¼ (K_eff_ik n̂_j n̂_l + 3 sym. terms)

    where  c = (λ₀ + μ₀) / (λ₀ + 2μ₀).

    This choice makes the Lippmann–Schwinger operator A(v) ≈ v for compatible
    modes, giving a contrast-ratio condition number rather than N²-scaling.

    Parameters
    ----------
    xi_flat : (3, Nv)   angular-frequency grid from ``build_freq_grid``
    lam0    : float     reference Lamé λ₀
    mu0     : float     reference shear modulus μ₀

    Returns
    -------
    G : (3, 3, 3, 3, Nv)  Green's operator; zero at ξ = 0.
    """
    xi_sq = jnp.sum(xi_flat ** 2, axis=0)           # (Nv,)
    safe  = xi_sq > 0
    xi_s  = jnp.where(safe, xi_sq, 1.0)
    n_hat = xi_flat / jnp.sqrt(xi_s)[None, :]       # (3, Nv)

    c    = (lam0 + mu0) / (lam0 + 2.0 * mu0)
    d    = jnp.eye(3)
    Kinv = (d[:, :, None]
            - c * jnp.einsum('iN,jN->ijN', n_hat, n_hat)) / mu0  # (3,3,Nv)

    nn    = jnp.einsum('iN,jN->ijN', n_hat, n_hat)
    Gamma = 0.25 * (
        jnp.einsum('ikN,jlN->ijklN', Kinv, nn) +
        jnp.einsum('ilN,jkN->ijklN', Kinv, nn) +
        jnp.einsum('jkN,ilN->ijklN', Kinv, nn) +
        jnp.einsum('jlN,ikN->ijklN', Kinv, nn)
    )                                                 # (3,3,3,3,Nv)
    return jnp.where(safe[None, None, None, None, :], Gamma, 0.0)



def set_freq_jax(ndim: int, n: tuple[int, ...], L: tuple[float, ...]):
    # n and L are STATIC python values here
    if ndim == 2:
        nx, ny = n
        Lx, Ly = L

        kx = nx * jnp.fft.fftfreq(nx, d=Lx)
        ky = ny * jnp.fft.fftfreq(ny, d=Ly)[: ny // 2 + 1]

        mg = jnp.array(jnp.meshgrid(kx, ky, indexing="ij"))  # (2, nx, nyh)
        return (2.0j * jnp.pi) * mg.reshape(2, nx * (ny // 2 + 1))

    elif ndim == 3:
        nx, ny, nz = n
        Lx, Ly, Lz = L

        kx = jnp.fft.fftfreq(nx, d=Lx)
        ky = jnp.fft.fftfreq(ny, d=Ly)
        kz = jnp.fft.fftfreq(nz, d=Lz)[: nz // 2 + 1]

        mg = jnp.array(jnp.meshgrid(kx, ky, kz, indexing="ij"))  # (3, nx, ny, nzh)
        # match FFTMAD scaling by n[i]
        nvec = jnp.array([nx, ny, nz], dtype=mg.dtype)[:, None]
        return (2.0j * jnp.pi) * (nvec * mg.reshape(3, nx * ny * (nz // 2 + 1)))

    else:
        raise ValueError("ndim must be 2 or 3")


# mark ALL shape-relevant args as static IMPORTANT for JIT compilation and performance
set_freq_jax_jit = jax.jit(set_freq_jax, static_argnames=("ndim", "n", "L"))



## Old NumPy version for benchmarking
def set_freq_np(ndim, n, L):
    n = np.asarray(n)
    L = np.asarray(L)

    if ndim == 2:
        kk_glob = 2.0j * np.pi * np.array(
            np.meshgrid(
                n[0] * np.fft.fftfreq(n[0], L[0]),
                n[1] * np.fft.fftfreq(n[1], L[1])[0 : n[1] // 2 + 1],
                indexing="ij",
            )
        ).reshape(2, n[0] * (n[1] // 2 + 1))
    elif ndim == 3:
        kk_glob = 2.0j * np.pi * np.einsum(
            "i,ix->ix",
            n,
            np.array(
                np.meshgrid(
                    np.fft.fftfreq(n[0], L[0]),
                    np.fft.fftfreq(n[1], L[1]),
                    np.fft.fftfreq(n[2], L[2])[0 : n[2] // 2 + 1],
                    indexing="ij",
                )
            ).reshape(3, n[0] * n[1] * (n[2] // 2 + 1)),
        )
    else:
        raise ValueError("ndim must be 2 or 3")
    return kk_glob