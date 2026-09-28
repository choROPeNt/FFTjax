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

import jax
import jax.numpy as jnp

from operators.base import LinearOperator
from operators.fft_distributed import choose_device_count, gather_x_slabs, pfft3d_flat, pifft3d_flat, split_x_slabs
from operators.general_functions import ddot42


class Gamma0Operator(LinearOperator):
    """
    x -> ifftn(green_op(fftn(x))), on a per-voxel (3, 3, Nv) tensor field
    defined over a periodic grid of shape ``n``.

    Automatically domain-decomposes across ``jax.local_device_count()``
    devices (a slab-decomposed distributed FFT, see operators.fft_distributed)
    whenever more than one is available and the grid divides evenly --
    transparently, no separate class or call needed. On a single-device
    machine (the common case, and every existing caller's test environment)
    this takes exactly the original single-device jnp.fft.fftn/ifftn path,
    unchanged. ``max_devices`` caps the auto-detected count -- mainly for
    forcing the single-device path on a multi-device machine (e.g. in tests);
    production callers leave it at ``None``.

    The distributed path needs ``green_op.G`` -- the cached (3,3,3,3,Nv)
    Fourier-space tensor every current green_op (GreenOperatorBasic/Willot,
    GalerkinProjector, mixed_bc._PatchedOperator) already has, all using the
    same ``ddot42(self.G, x)`` calling convention -- so it can be split into
    per-device slabs once, up front, instead of re-applying ``green_op``
    itself per call (which would need the *full*, unsharded ``G``).

    Self-adjoint: ``green_op`` is self-adjoint in Fourier space, and forward/
    inverse DFT are adjoints of each other on real fields, so the composition
    is self-adjoint on real fields too.
    """

    def __init__(self, n: tuple[int, ...], green_op: LinearOperator, max_devices: int | None = None):
        self.n = n
        self.green_op = green_op
        self.n_devices = choose_device_count(n, max_devices)
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
        """x, result: (3, 3, Nv) real-space tensor field, e.g. a strain field."""
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
