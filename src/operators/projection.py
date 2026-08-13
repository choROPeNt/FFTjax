"""
Gamma0 fixed-point operator for the Lippmann-Schwinger scheme.

Note on scope: Gamma0 = sym(grad) : G0 : sym(grad) is a real composition, but
G0's contribution is already fully baked into GreenOperatorBasic/Willot's `G`
tensor via n_hat = xi/|xi| (see the comment at build_green_operator's n_hat
step in green.py) -- Gamma0 is degree-0 homogeneous in xi, so there is no
separate raw-gradient LinearOperator to compose here. What remains is purely
mechanical: apply the (already-composed) Green operator in Fourier space to a
real-space per-voxel tensor field.
"""

from __future__ import annotations

import jax.numpy as jnp

from operators.base import LinearOperator


class Gamma0Operator(LinearOperator):
    """
    x -> ifftn(green_op(fftn(x))), on a per-voxel (3, 3, Nv) tensor field
    defined over a periodic grid of shape ``n``.

    Self-adjoint: ``green_op`` is self-adjoint in Fourier space, and forward/
    inverse DFT are adjoints of each other on real fields, so the composition
    is self-adjoint on real fields too.
    """

    def __init__(self, n: tuple[int, ...], green_op: LinearOperator):
        self.n = n
        self.green_op = green_op

    def _fft(self, x: jnp.ndarray) -> jnp.ndarray:
        s = x.shape
        return jnp.fft.fftn(x.reshape(s[:-1] + self.n), axes=(-3, -2, -1)).reshape(s)

    def _ifft(self, x: jnp.ndarray) -> jnp.ndarray:
        s = x.shape
        return jnp.fft.ifftn(x.reshape(s[:-1] + self.n), axes=(-3, -2, -1)).real.reshape(s)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """x, result: (3, 3, Nv) real-space tensor field, e.g. a strain field."""
        return self._ifft(self.green_op(self._fft(x)))

    @property
    def T(self) -> LinearOperator:
        return self
