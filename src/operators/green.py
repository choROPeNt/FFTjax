import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)

import jax.numpy as jnp

from materialmodels.tensors import isotropic_equivalent_lame
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


def nyquist_safe_xi(xi_flat: jnp.ndarray, n: tuple[int, ...]) -> jnp.ndarray:
    """
    Zero the Nyquist-frequency component of ``xi_flat`` along each even grid
    dimension.

    Any operator that uses ``xi`` to an odd power (a plain gradient or
    divergence, unlike the strain-based Green's operators above, which only
    ever use even powers ``ξξ``/``ξξξξ``) needs this. For an even-length
    dimension the Nyquist bin has no distinct negative-frequency partner, so
    its FFT coefficient must be real for the transform of a real signal to
    stay Hermitian-symmetric -- multiplying it by an odd (purely imaginary)
    power of ``ξ`` breaks that symmetry and corrupts the real-space result.
    Zeroing it there is the standard fix used throughout FFT-Galerkin
    homogenization schemes. Shared by solvers.elliptic.vector.
    displacement_based (gradient/divergence of a displacement field) and
    solvers.elliptic.scalar's heterogeneous-Gc damage solve (gradient/
    divergence of a scalar damage field) -- same gotcha either way.
    """
    idx   = [jnp.arange(ni) for ni in n]
    grids = jnp.meshgrid(*idx, indexing="ij")
    masks = [
        (g == ni // 2) if ni % 2 == 0 else jnp.zeros_like(g, dtype=bool)
        for ni, g in zip(n, grids)
    ]
    nyquist_mask = jnp.stack([m.ravel() for m in masks])
    return jnp.where(nyquist_mask, 0.0, xi_flat)


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

    # n_hat (direction only), not xi (direction * magnitude): Gamma0 is exactly
    # degree-0 homogeneous in xi for an isotropic reference medium -- the |xi|^2
    # from the two gradients in Gamma0 = sym(grad) : G0 : sym(grad) exactly
    # cancels the 1/|xi|^2 in G0 itself. Using xi here instead of n_hat would
    # silently reintroduce a spurious dependence on domain length L (since
    # |xi| ~ 1/L but direction doesn't), not just a stylistic difference.
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

    Self-adjoint (major-symmetric: Γ_ijkl = Γ_klij), so ``.T`` returns self.
    """

    def __init__(self, n: tuple[int, ...], L: tuple[float, ...], lam0: float, mu0: float):
        self.n, self.L, self.lam0, self.mu0 = n, L, lam0, mu0
        self.G = self._build_G()

    def _build_G(self) -> jnp.ndarray:
        xi_flat = build_freq_grid(self.n, self.L)
        return build_green_operator(xi_flat, self.lam0, self.mu0, scheme='standard')

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """x, result: (3, 3, Nv) — e.g. a strain or stress field in Fourier space."""
        return ddot42(self.G, x)

    @property
    def T(self) -> LinearOperator:
        return self


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
        self.dx = dx
        super().__init__(n, L, lam0, mu0)

    def _build_G(self) -> jnp.ndarray:
        xi_flat = build_freq_grid(self.n, self.L)
        return build_green_operator(xi_flat, self.lam0, self.mu0, scheme='rotated', dx=self.dx)


def build_reference_green_operator(
    n: tuple[int, ...],
    L: tuple[float, ...],
    materials: list,
    scheme: str = 'rotated',
) -> LinearOperator:
    """
    Build a GreenOperatorBasic/Willot for the arithmetic-mean-Lame reference
    medium of ``materials``. Shared by every problems/ wiring layer that
    needs a Green's operator for a phase-labelled voxel grid (elasticity,
    staggered fracture, ...) so the reference-medium averaging and
    scheme-selection logic lives in exactly one place.

    Reference-medium (lam0, mu0) is a Voigt-isotropization of the mean
    stiffness tensor (materialmodels.tensors.isotropic_equivalent_lame) --
    exact for isotropic materials (reduces to averaging their own lam/mu),
    also handles anisotropic ones (e.g. TransverseIsotropicFibre, which has
    no .lam/.mu) since it works from stiffness_tensor() rather than
    assuming those attributes exist.

    Parameters
    ----------
    n, L       : grid shape and physical domain size
    materials  : list, each exposing .stiffness_tensor()
    scheme     : 'standard' (GreenOperatorBasic) or 'rotated' (GreenOperatorWillot)

    Returns
    -------
    green_op : GreenOperatorBasic or GreenOperatorWillot
    """
    C_mean = jnp.mean(jnp.stack([m.stiffness_tensor() for m in materials]), axis=0)
    lam0, mu0 = (float(v) for v in isotropic_equivalent_lame(C_mean))

    if scheme == 'standard':
        return GreenOperatorBasic(n, L, lam0, mu0)
    elif scheme == 'rotated':
        dx = tuple(Li / ni for Li, ni in zip(L, n))
        return GreenOperatorWillot(n, L, lam0, mu0, dx)
    else:
        raise ValueError(f"unknown scheme {scheme!r}, expected 'standard' or 'rotated'")