import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)

import jax
import jax.numpy as jnp
import numpy as np

from operators.base import LinearOperator
from operators.general_functions import ddot42


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
# ---------------------------------------------------------------------------
# Willot rotated-scheme effective frequencies
# ---------------------------------------------------------------------------

def build_willot_freq(
    xi_flat: jnp.ndarray,
    dx:      tuple[float, ...],
) -> jnp.ndarray:
    """
    Willot (2015) rotated-scheme effective frequencies for an isotropic
    reference medium.

    Replaces the continuous DFT frequencies ξ_j with the discrete
    finite-difference equivalent:

        ξ_j^eff = (2 / h_j) sin(ξ_j h_j / 2)

    For an isotropic reference medium the complex phase factor of the
    full Willot frequency exp(i Σ_k ξ_k h_k / 2) cancels in the Green's
    operator, leaving these real-valued effective frequencies.

    Key properties
    --------------
    - At low wavenumbers:  ξ_j^eff ≈ ξ_j  (matches continuous limit)
    - At Nyquist:          ξ_j^eff = 2/h_j  (finite, no Gibbs overshoot)
    - Isotropic on the grid: eliminates the 45° crack-propagation bias
      present in the standard Moulinec–Suquet discretisation

    Parameters
    ----------
    xi_flat : (ndim, Nv)  continuous angular frequencies from ``build_freq_grid``
    dx      : (ndim,)     voxel spacing per dimension  h_j = L_j / n_j

    Returns
    -------
    xi_eff : (ndim, Nv)  real-valued effective frequencies
    """
    dx_arr = jnp.array(dx)                             # (ndim,)
    return (2.0 / dx_arr[:, None]) * jnp.sin(xi_flat * dx_arr[:, None] / 2.0)


# ---------------------------------------------------------------------------
# DBFFT Green's operator  Γ̂₀  for isotropic reference medium
# ---------------------------------------------------------------------------

def build_green_operator(
    xi_flat: jnp.ndarray,
    lam0:    float,
    mu0:     float,
    scheme:  str = 'standard',
    dx:      tuple[float, ...] | None = None,
) -> jnp.ndarray:
    """
    4th-order Green's operator Γ̂₀_ijkl for an isotropic reference medium.

    DBFFT / FFTMAD convention:

        K_eff_ik  = (δ_ik − c n̂_i n̂_k) / μ₀
        Γ̂₀_ijkl  = ¼ (K_eff_ik n̂_j n̂_l + 3 sym. terms)
        c = (λ₀ + μ₀) / (λ₀ + 2μ₀)

    Schemes
    -------
    'standard'
        Uses the continuous DFT frequencies ξ_j from ``build_freq_grid``.
        Equivalent to the Moulinec–Suquet discretisation.  Prone to Gibbs
        oscillations and a 45° anisotropy bias at high stiffness contrasts.

    'rotated'
        Uses the Willot (2015) effective frequencies ξ_j^eff = (2/h_j)sin(ξ_j h_j/2).
        Equivalent to linear hexahedral elements with reduced integration
        (Schneider et al. 2016).  Eliminates the diagonal crack-propagation
        bias and improves CG convergence at high stiffness contrasts.
        Requires ``dx`` to be provided.

    Parameters
    ----------
    xi_flat : (3, Nv)              angular-frequency grid from ``build_freq_grid``
    lam0    : float                reference Lamé λ₀
    mu0     : float                reference shear modulus μ₀
    scheme  : 'standard'|'rotated' discretisation scheme (default 'standard')
    dx      : (3,) tuple or None   voxel spacing — required for scheme='rotated'

    Returns
    -------
    G : (3, 3, 3, 3, Nv)  Green's operator; zero at ξ = 0.
    """
    if scheme == 'rotated':
        if dx is None:
            raise ValueError("dx must be provided for scheme='rotated'")
        xi = build_willot_freq(xi_flat, dx)
    else:
        xi = xi_flat

    xi_sq = jnp.sum(xi ** 2, axis=0)                  # (Nv,)
    safe  = xi_sq > 0
    xi_s  = jnp.where(safe, xi_sq, 1.0)
    n_hat = xi / jnp.sqrt(xi_s)[None, :]              # (3, Nv)

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
    )                                                  # (3,3,3,3,Nv)
    return jnp.where(safe[None, None, None, None, :], Gamma, 0.0)


# ---------------------------------------------------------------------------
# LinearOperator wrappers around build_green_operator
# ---------------------------------------------------------------------------

class GreenOperatorBasic(LinearOperator):
    """
    Γ̂₀ reference-medium Green's operator, 'standard' (Moulinec–Suquet)
    discretisation — continuous DFT frequencies, no Willot correction.

    Self-adjoint (major-symmetric: Γ_ijkl = Γ_klij), so ``.T`` returns self
    (the ``LinearOperator`` default).
    """

    def __init__(self, n: tuple[int, ...], L: tuple[float, ...], lam0: float, mu0: float):
        self.n, self.L, self.lam0, self.mu0 = n, L, lam0, mu0
        xi_flat = build_freq_grid(n, L)
        self.G = build_green_operator(xi_flat, lam0, mu0, scheme='standard')

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """x, result: (3, 3, Nv) — e.g. a strain or stress field in Fourier space."""
        return ddot42(self.G, x)


class GreenOperatorWillot(GreenOperatorBasic):
    """
    Γ̂₀ with Willot (2015) rotated-scheme effective frequencies — removes the
    45° anisotropy bias of the standard scheme and converges better under
    high stiffness contrast. Requires the voxel spacing ``dx``.
    """

    def __init__(
        self,
        n: tuple[int, ...],
        L: tuple[float, ...],
        lam0: float,
        mu0: float,
        dx: tuple[float, ...],
    ):
        self.n, self.L, self.lam0, self.mu0, self.dx = n, L, lam0, mu0, dx
        xi_flat = build_freq_grid(n, L)
        self.G = build_green_operator(xi_flat, lam0, mu0, scheme='rotated', dx=dx)


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