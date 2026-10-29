"""
Gamma0 operator for the Lippmann-Schwinger scheme: the Green tensor applied in
Fourier space to a real-space field.

G0 is already folded into the Green operator's ``G`` (via n_hat = xi/|xi|), so
there is no separate gradient operator to compose.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from operators.base import LinearOperator
from operators.fft_distributed import choose_device_count, gather_x_slabs, pfft3d_flat, pifft3d_flat, split_x_slabs
from operators.general_functions import ddot42


class Gamma0Operator(LinearOperator):
    """
    ``x -> ifftn(green_op(fftn(x)))`` on a (3, 3, Nv) field over grid ``n``. Self-adjoint.

    Splits across ``jax.local_device_count()`` devices when more than one is
    available (``n_devices`` caps it); one device is the plain ``jnp.fft`` path.
    The distributed path slices ``green_op.G`` once up front.
    """

    def __init__(self, n: tuple[int, ...], green_op: LinearOperator, n_devices: int | None = None):
        self.n = n
        self.green_op = green_op
        self.n_devices = choose_device_count(n, n_devices)
        if self.n_devices > 1:
            self._n_local = (n[0] // self.n_devices, n[1], n[2])
            self._G_sharded = split_x_slabs(green_op.G, self.n_devices)

    def _fft(self, x: jnp.ndarray) -> jnp.ndarray:
        s = x.shape
        return jnp.fft.fftn(x.reshape(s[:-1] + self.n), axes=(-3, -2, -1)).reshape(s)

    def _ifft(self, x: jnp.ndarray) -> jnp.ndarray:
        s = x.shape
        return jnp.fft.ifftn(x.reshape(s[:-1] + self.n), axes=(-3, -2, -1)).real.reshape(s)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """x, result: (3, 3, Nv) real-space field."""
        if self.n_devices == 1:
            return self._ifft(self.green_op(self._fft(x)))

        x_sharded = split_x_slabs(x, self.n_devices)

        def local(x_local, G_local):
            xh = pfft3d_flat(x_local, self._n_local, "i")
            gh = ddot42(G_local, xh)
            return pifft3d_flat(gh, self._n_local, "i")

        out_sharded = jax.pmap(local, axis_name="i")(x_sharded, self._G_sharded)
        return gather_x_slabs(out_sharded)

    @property
    def T(self) -> LinearOperator:
        return self
