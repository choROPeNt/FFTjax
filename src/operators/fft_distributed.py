"""
Slab-decomposed distributed 3-D FFT for splitting one RVE across devices.

Real space is split along x into ``n_devices`` slabs (``n[0]`` and ``n[1]``
divisible by ``n_devices``):

1. local FFT along y, z
2. ``all_to_all``: x local, y split
3. local FFT along x
4. ``all_to_all`` back to the x-split layout

``pfft3d``/``pifft3d`` run under ``jax.pmap`` or ``jax.shard_map`` with the
same ``axis_name``; complex in, complex out like ``jnp.fft.fftn``/``ifftn``.
"""

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)

from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

AXIS = "i"


def x_slab_mesh(n_devices: int) -> Mesh:
    """1-D mesh over the first ``n_devices`` local devices, axis ``AXIS``."""
    return Mesh(np.array(jax.local_devices()[:n_devices]), (AXIS,))


def x_slab_spec(ndim: int) -> P:
    """PartitionSpec splitting the last (voxel) axis into contiguous x-slabs."""
    return P(*([None] * (ndim - 1)), AXIS)


def x_slab_shard_map(f: Callable, n_devices: int, in_specs: Any, out_specs: Any) -> Callable[..., Any]:
    """
    ``jax.shard_map`` of ``f`` over ``x_slab_mesh(n_devices)``.

    Typed here because jax.shard_map has no return annotation and checkers
    infer its one-argument decorator form.
    """
    return jax.shard_map(f, mesh=x_slab_mesh(n_devices), in_specs=in_specs, out_specs=out_specs)


def shard_x_slabs(field: jnp.ndarray, n_devices: int) -> jnp.ndarray:
    """
    Place ``(*lead, Nv)`` as x-slabs across devices (no-op if already so).

    Keeps the global shape, so plain ``jnp`` code stays correct and
    ``shard_map`` consumes it without a copy.
    """
    return jax.device_put(field, NamedSharding(x_slab_mesh(n_devices), x_slab_spec(field.ndim)))


def choose_device_count(n: tuple[int, ...], n_devices: int | None = None) -> int:
    """
    Largest ``d <= min(local devices, n_devices)`` with ``n[0] % d == n[1] % d == 0``.

    Never errors; 1 means the single-device path.
    """
    available = jax.local_device_count()
    cap = available if n_devices is None else min(available, n_devices)
    for d in range(cap, 0, -1):
        if n[0] % d == 0 and n[1] % d == 0:
            return d
    return 1


def split_x_slabs(field: jnp.ndarray, n_devices: int) -> jnp.ndarray:
    """``(*lead, Nv)`` -> ``(n_devices, *lead, Nv / n_devices)``; x is the slowest flat index."""
    lead = field.shape[:-1]
    nv_local = field.shape[-1] // n_devices
    reshaped = field.reshape(*lead, n_devices, nv_local)
    return jnp.moveaxis(reshaped, len(lead), 0)


def gather_x_slabs(field_sharded: jnp.ndarray) -> jnp.ndarray:
    """Inverse of ``split_x_slabs``."""
    lead = field_sharded.shape[1:-1]
    moved = jnp.moveaxis(field_sharded, 0, len(lead))
    return moved.reshape(*lead, -1)


def shard_voxel_tensor(build_local, xi_flat: jnp.ndarray, n_devices: int) -> jnp.ndarray:
    """
    Build a per-voxel tensor directly as x-slabs, never materialized on one device.

    Parameters
    ----------
    build_local : xi_local (3, Nv_local) -> (..., Nv_local), other arguments bound
    xi_flat     : (3, Nv) frequencies
    n_devices   : resolved device count (> 1)

    Returns
    -------
    (..., Nv) global array, sharded as x-slabs
    """
    xi = shard_x_slabs(xi_flat, n_devices)
    out_ndim = jax.eval_shape(build_local, xi_flat).ndim
    return x_slab_shard_map(
        build_local, n_devices, in_specs=x_slab_spec(xi.ndim), out_specs=x_slab_spec(out_ndim),
    )(xi)


def _transpose_x_to_y(x: jnp.ndarray, axis_name: str) -> jnp.ndarray:
    """(..., nx_local, ny, nz) -> (..., nx, ny_local, nz)."""
    split_axis, concat_axis = x.ndim - 2, x.ndim - 3
    return lax.all_to_all(x, axis_name, split_axis=split_axis, concat_axis=concat_axis, tiled=True)


def _transpose_y_to_x(x: jnp.ndarray, axis_name: str) -> jnp.ndarray:
    """(..., nx, ny_local, nz) -> (..., nx_local, ny, nz)."""
    split_axis, concat_axis = x.ndim - 3, x.ndim - 2
    return lax.all_to_all(x, axis_name, split_axis=split_axis, concat_axis=concat_axis, tiled=True)


def pfft3d(x_local: jnp.ndarray, axis_name: str) -> jnp.ndarray:
    """Forward 3-D FFT of one ``(..., nx_local, ny, nz)`` slab."""
    xh = jnp.fft.fft(x_local, axis=-2)
    xh = jnp.fft.fft(xh, axis=-1)
    xh = _transpose_x_to_y(xh, axis_name)
    xh = jnp.fft.fft(xh, axis=-3)
    xh = _transpose_y_to_x(xh, axis_name)
    return xh


def pifft3d(xhat_local: jnp.ndarray, axis_name: str) -> jnp.ndarray:
    """Inverse of ``pfft3d``; complex result."""
    x = jnp.fft.ifft(xhat_local, axis=-2)
    x = jnp.fft.ifft(x, axis=-1)
    x = _transpose_x_to_y(x, axis_name)
    x = jnp.fft.ifft(x, axis=-3)
    x = _transpose_y_to_x(x, axis_name)
    return x


def pfft3d_flat(x_local_flat: jnp.ndarray, n_local: tuple[int, ...], axis_name: str) -> jnp.ndarray:
    """``pfft3d`` on a flat ``(*lead, Nv_local)`` slab with local grid shape ``n_local``."""
    s = x_local_flat.shape
    xg = x_local_flat.reshape(s[:-1] + n_local)
    xh = pfft3d(xg, axis_name)
    return xh.reshape(s)


def pifft3d_flat(xhat_local_flat: jnp.ndarray, n_local: tuple[int, ...], axis_name: str) -> jnp.ndarray:
    """Inverse of ``pfft3d_flat``; real result."""
    s = xhat_local_flat.shape
    xg = xhat_local_flat.reshape(s[:-1] + n_local)
    x = pifft3d(xg, axis_name)
    return x.real.reshape(s)


def distributed_fft_flat(x_full: jnp.ndarray, n: tuple[int, ...], n_devices: int | None = None) -> jnp.ndarray:
    """
    ``jnp.fft.fftn`` on a flat ``(*lead, Nv)`` field, split across devices when possible.

    Full array in, full array out (gathered after each call), so callers need
    no changes, but it is not memory-scaling.
    """
    n_devices = choose_device_count(n, n_devices)
    if n_devices == 1:
        s = x_full.shape
        return jnp.fft.fftn(x_full.reshape(s[:-1] + n), axes=(-3, -2, -1)).reshape(s)

    n_local = (n[0] // n_devices, n[1], n[2])
    x_sharded = split_x_slabs(x_full, n_devices)
    out_sharded = jax.pmap(lambda s: pfft3d_flat(s, n_local, "i"), axis_name="i")(x_sharded)
    return gather_x_slabs(out_sharded)


def distributed_ifft_flat(xhat_full: jnp.ndarray, n: tuple[int, ...], n_devices: int | None = None) -> jnp.ndarray:
    """Inverse of ``distributed_fft_flat``; real result."""
    n_devices = choose_device_count(n, n_devices)
    if n_devices == 1:
        s = xhat_full.shape
        return jnp.fft.ifftn(xhat_full.reshape(s[:-1] + n), axes=(-3, -2, -1)).real.reshape(s)

    n_local = (n[0] // n_devices, n[1], n[2])
    x_sharded = split_x_slabs(xhat_full, n_devices)
    out_sharded = jax.pmap(lambda s: pifft3d_flat(s, n_local, "i"), axis_name="i")(x_sharded)
    return gather_x_slabs(out_sharded)
