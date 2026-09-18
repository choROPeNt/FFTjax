"""
Reference-medium-free Fourier-space projector for the Fourier-Galerkin
elastic scheme (Vondrejc et al 2014; see Lucarini, Upadhyay & Segurado 2022,
doi:10.1088/1361-651X/ac34e1, sec. 3.3). A sibling to green.py's
GreenOperatorBasic/Willot, not a special case of it: build_green_operator's
Kinv = (delta - c*n_hat n_hat)/mu0 ansatz cannot reproduce this operator's
required coefficients at any (lam0, mu0) -- verified numerically before
writing this module, see notes/FOURIER_GALERKIN.md.
"""

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)

import jax.numpy as jnp

from operators.base import LinearOperator
from operators.general_functions import ddot42
from operators.green import build_freq_grid, build_willot_freq


def build_galerkin_projection_operator(
    xi_flat: jnp.ndarray,
    scheme:  str = 'standard',
    dx:      tuple[float, ...] | None = None,
) -> jnp.ndarray:
    """
    Symmetric compatibility projector Ĝₛ_ijkl -- projects an arbitrary
    symmetric 2nd-order tensor field onto its compatible (curl-free) part.
    Purely geometric: depends only on the frequency direction n̂ = ξ/|ξ|, no
    reference medium, no materials.

        Ĝₛ_ijkl = ½(δik n̂j n̂l + δil n̂j n̂k + δjk n̂i n̂l + δjl n̂i n̂k) − n̂i n̂j n̂k n̂l

    Verified (plain numpy, before this module was written): major- and
    minor-symmetric, idempotent (Ĝₛ:Ĝₛ = Ĝₛ, a true projector), fixes any
    compatible field sym(n̂⊗v) exactly, annihilates any n̂-orthogonal
    (incompatible) field, trace 3 (rank-3 projector on the 6-dim symmetric-
    tensor space) -- see test/test_operators_green_galerkin.py.

    Same 'standard'/'rotated' scheme dispatch and zero-at-DC convention as
    build_green_operator (operators.green); only uses even powers of ξ (ξξ,
    ξξξξ), same as that operator, so no nyquist_safe_xi needed either.

    Parameters
    ----------
    xi_flat : (3, Nv)              angular-frequency grid from build_freq_grid
    scheme  : 'standard'|'rotated' discretisation scheme (default 'standard')
    dx      : (3,) tuple or None   voxel spacing -- required for scheme='rotated'

    Returns
    -------
    G : (3, 3, 3, 3, Nv)  projector; zero at ξ = 0.
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
    """
    Ĝₛ symmetric compatibility projector -- no reference medium, no
    materials, unlike GreenOperatorBasic/Willot. Self-adjoint
    (major-symmetric: Ĝₛ_ijkl = Ĝₛ_klij), so ``.T`` returns self.
    """

    def __init__(self, n: tuple[int, ...], L: tuple[float, ...], scheme: str = 'rotated'):
        self.n, self.L, self.scheme = n, L, scheme
        self.G = self._build_G()

    def _build_G(self) -> jnp.ndarray:
        xi_flat = build_freq_grid(self.n, self.L)
        if self.scheme == 'rotated':
            dx = tuple(Li / ni for Li, ni in zip(self.L, self.n))
            return build_galerkin_projection_operator(xi_flat, scheme='rotated', dx=dx)
        return build_galerkin_projection_operator(xi_flat, scheme=self.scheme)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """x, result: (3, 3, Nv) -- e.g. a strain or stress field in Fourier space."""
        return ddot42(self.G, x)

    @property
    def T(self) -> LinearOperator:
        return self


def build_galerkin_projector(
    n: tuple[int, ...],
    L: tuple[float, ...],
    scheme: str = 'rotated',
) -> GalerkinProjector:
    """
    Build a GalerkinProjector for a grid -- the materials-free counterpart
    to operators.green.build_reference_green_operator, so
    problems.mechanics's formulation="fourier_galerkin" branch reads in
    parallel to its formulation="lippmann_schwinger" branch, minus the
    reference-medium averaging step (there is none here).

    Parameters
    ----------
    n, L   : grid shape and physical domain size
    scheme : 'standard' or 'rotated'

    Returns
    -------
    galerkin_op : GalerkinProjector
    """
    if scheme not in ('standard', 'rotated'):
        raise ValueError(f"unknown scheme {scheme!r}, expected 'standard' or 'rotated'")
    return GalerkinProjector(n, L, scheme=scheme)
