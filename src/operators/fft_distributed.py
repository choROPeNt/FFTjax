"""
Slab-decomposed distributed 3-D FFT, for domain decomposition of one large
RVE across multiple devices under ``jax.pmap`` (see notes on
``dev_pmap`` branch -- this is a from-scratch prototype, not wired into any
solver yet).

Every existing solver (``solvers.elliptic.vector.displacement_based``,
``problems.mechanics``, ...) puts the whole grid on one device and calls
``jnp.fft.fftn``/``ifftn`` directly -- fine until the per-voxel fields
(``C_field`` alone is ``(3,3,3,3,Nv)``) no longer fit in one device's memory.
This module is the one genuinely non-local piece of a domain-decomposed
solver: everything else (Green's-operator apply, per-voxel material update,
CG's own vector ops) is already elementwise over voxels and decomposes for
free.

Decomposition scheme
---------------------
Real space is split along axis 0 (x) into ``n_devices`` equal slabs, one per
device -- ``n[0] % n_devices == 0`` is required. The classic slab-FFT
algorithm (as in P3DFFT/mpi4py-fft, translated to JAX's ``all_to_all``
collective under ``pmap``):

1. Local FFT along y, z (axes 1, 2) -- already fully local per slab.
2. Global transpose: ``all_to_all`` re-shards so x becomes fully local and y
   becomes split instead (``n[1] % n_devices == 0`` also required).
3. Local FFT along x (now fully local per device).
4. Global transpose back: ``all_to_all`` re-shards to the original x-split
   layout, so the output has the same per-device shape/axis convention as
   the input -- a drop-in shape-wise replacement for
   ``jnp.fft.fftn(x, axes=(0, 1, 2))`` called per-slab.

Must be called under ``jax.pmap(..., axis_name=...)`` (or an equivalent
``shard_map`` with a named mesh axis) with that same ``axis_name`` passed
through. Operates on one leading spatial slab per device, i.e. call it as
the innermost function that ``pmap`` maps over -- shape ``(nx_local, ny, nz)``
in, same shape out (complex), exactly matching ``jnp.fft.fftn``/``ifftn``'s
own complex-in/complex-out convention (callers take ``.real`` themselves
after the inverse transform, same as the existing single-device solvers do).
"""

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)

import jax
import jax.numpy as jnp
from jax import lax


def choose_device_count(n: tuple[int, ...], n_devices: int | None = None) -> int:
    """
    Real device count to use for domain-decomposing a grid of shape ``n``,
    auto-detected from ``jax.local_device_count()`` -- the actual GPU/CPU
    device count JAX sees in *this* process, with no configuration needed on
    a real multi-device node. (``XLA_FLAGS=--xla_force_host_platform_device_count=N``
    is a *testing-only* trick to simulate that on a single machine -- see
    test/test_operators_fft_distributed.py -- production code never sets it;
    ``jax.local_device_count()`` just reports whatever's really there.)

    The slab decomposition needs ``n[0] % d == 0`` and ``n[1] % d == 0`` (see
    module docstring), so this returns the largest ``d`` in
    ``[1, min(available, n_devices or available)]`` satisfying both --
    degrading toward fewer devices (never erroring) if the grid doesn't
    divide evenly by the full device count, and toward exactly 1 (no
    decomposition at all) if only one device is available or none divide
    evenly except 1. Callers can dispatch on the result unconditionally: 1
    means "use the existing single-device path unchanged."
    """
    available = jax.local_device_count()
    cap = available if n_devices is None else min(available, n_devices)
    for d in range(cap, 0, -1):
        if n[0] % d == 0 and n[1] % d == 0:
            return d
    return 1  # unreachable (d=1 always satisfies both conditions) -- explicit for the type checker


def split_x_slabs(field: jnp.ndarray, n_devices: int) -> jnp.ndarray:
    """
    ``(*lead, Nv)`` -> ``(n_devices, *lead, Nv // n_devices)``: split a flat
    per-voxel field into per-device x-slabs. Contiguous chunks of the flat
    (C-order) last axis are exactly contiguous x-slabs, since ``n``'s grid
    reshape convention used throughout this project (``x.reshape(s[:-1] + n)``)
    makes x the slowest-varying flat index.
    """
    lead = field.shape[:-1]
    nv_local = field.shape[-1] // n_devices
    reshaped = field.reshape(*lead, n_devices, nv_local)
    return jnp.moveaxis(reshaped, len(lead), 0)


def gather_x_slabs(field_sharded: jnp.ndarray) -> jnp.ndarray:
    """Inverse of ``split_x_slabs``: ``(n_devices, *lead, Nv_local)`` -> ``(*lead, Nv)``."""
    lead = field_sharded.shape[1:-1]
    moved = jnp.moveaxis(field_sharded, 0, len(lead))
    return moved.reshape(*lead, -1)


def shard_voxel_tensor(build_local, xi_flat: jnp.ndarray, n_devices: int) -> jnp.ndarray:
    """
    Build a per-voxel tensor field directly as ``n_devices`` x-slabs under
    ``jax.pmap``, instead of materializing the full ``(..., Nv)`` tensor on
    one device first -- the same purely-per-voxel sharding
    ``materialmodels.assembly._assemble_C_field_sharded`` uses, factored out
    here since ``operators.green.build_green_operator`` and
    ``operators.galerkin.build_galerkin_projection_operator`` are both
    ALSO pure (no-cross-voxel-coupling) functions of ``xi_flat`` alone,
    producing an equally large ``(3,3,3,3,Nv)`` tensor -- the second half of
    the OOM benchmark_3's own investigation found (the first being
    ``C_field`` itself): the Green's/Galerkin tensor is built fully
    unsharded regardless of ``C_field``'s own sharding, so a
    lippmann_schwinger/fourier_galerkin solve still OOMs on it alone even
    once ``C_field`` no longer does.

    Parameters
    ----------
    build_local : xi_flat_local (3, Nv_local) -> (..., Nv_local) -- every
                  non-array argument (scheme, reference moduli, voxel
                  spacing, ...) already bound in, e.g. via functools.partial.
    xi_flat     : (3, Nv) angular-frequency grid (operators.green.build_freq_grid)
    n_devices   : already resolved (> 1) -- callers dispatch via
                  operators.fft_distributed.choose_device_count themselves,
                  same convention as every other sharded entry point.

    Returns
    -------
    (..., Nv) -- gathered back to one array; not (yet) memory-scaling on its
    own the way the sharded CG solvers are (the gather step itself briefly
    needs more than one device's share of memory for a tensor this large,
    observed on real GPU hardware -- a genuinely no-gather, stays-sharded
    version is the natural follow-up, mirroring how the CG solves
    themselves never gather C_field/G mid-solve), but still genuinely
    avoids ever materializing the full, unsharded computation on one
    device, which is what OOMs first otherwise.
    """
    xi_sharded = split_x_slabs(xi_flat, n_devices)
    out_sharded = jax.pmap(build_local)(xi_sharded)
    return gather_x_slabs(out_sharded)


def _transpose_x_to_y(x: jnp.ndarray, axis_name: str) -> jnp.ndarray:
    """(..., nx_local, ny, nz) -> (..., nx, ny_local, nz): gather x, split y.
    Axis positions computed from ``x.ndim`` (not hardcoded 0/1) so this works
    with arbitrary leading batch dims, e.g. a (3,3,nx_local,ny,nz) tensor
    field -- ``lax.all_to_all`` needs non-negative axis ints, unlike the
    plain ``jnp.fft`` calls in ``pfft3d``/``pifft3d``, which can use ``-1``
    etc. directly."""
    split_axis, concat_axis = x.ndim - 2, x.ndim - 3
    return lax.all_to_all(x, axis_name, split_axis=split_axis, concat_axis=concat_axis, tiled=True)


def _transpose_y_to_x(x: jnp.ndarray, axis_name: str) -> jnp.ndarray:
    """(..., nx, ny_local, nz) -> (..., nx_local, ny, nz): gather y, split x."""
    split_axis, concat_axis = x.ndim - 3, x.ndim - 2
    return lax.all_to_all(x, axis_name, split_axis=split_axis, concat_axis=concat_axis, tiled=True)


def pfft3d(x_local: jnp.ndarray, axis_name: str) -> jnp.ndarray:
    """
    Distributed forward 3-D FFT of one x-slab. ``x_local``: ``(nx_local, ny,
    nz)`` (or with extra leading batch dims before that -- axes are counted
    from the end the same way the single-device solvers' ``fft_`` closures
    do, but this prototype assumes exactly 3 trailing spatial dims for now).
    Must run under ``pmap(..., axis_name=axis_name)``.
    """
    # x (axis -3) is split across devices at this point -- only y, z (-2, -1)
    # are locally complete, so those are the only two axes safe to transform
    # before the transpose brings x fully local.
    xh = jnp.fft.fft(x_local, axis=-2)
    xh = jnp.fft.fft(xh, axis=-1)
    xh = _transpose_x_to_y(xh, axis_name)
    xh = jnp.fft.fft(xh, axis=-3)
    xh = _transpose_y_to_x(xh, axis_name)
    return xh


def pifft3d(xhat_local: jnp.ndarray, axis_name: str) -> jnp.ndarray:
    """Inverse of ``pfft3d`` -- same slab layout in and out, complex result
    (take ``.real`` in the caller, matching ``jnp.fft.ifftn``'s convention)."""
    x = jnp.fft.ifft(xhat_local, axis=-2)
    x = jnp.fft.ifft(x, axis=-1)
    x = _transpose_x_to_y(x, axis_name)
    x = jnp.fft.ifft(x, axis=-3)
    x = _transpose_y_to_x(x, axis_name)
    return x


def pfft3d_flat(x_local_flat: jnp.ndarray, n_local: tuple[int, ...], axis_name: str) -> jnp.ndarray:
    """``(*lead, Nv_local)`` flat -> grid -> ``pfft3d`` -> flat, mirroring
    ``operators.projection.Gamma0Operator._fft``'s own reshape convention
    exactly, with ``pfft3d`` swapped in for ``jnp.fft.fftn`` and ``n_local``
    (this device's local slab shape) in place of the full grid shape."""
    s = x_local_flat.shape
    xg = x_local_flat.reshape(s[:-1] + n_local)
    xh = pfft3d(xg, axis_name)
    return xh.reshape(s)


def pifft3d_flat(xhat_local_flat: jnp.ndarray, n_local: tuple[int, ...], axis_name: str) -> jnp.ndarray:
    """Inverse of ``pfft3d_flat`` -- real result, matching ``jnp.fft.ifftn().real``."""
    s = xhat_local_flat.shape
    xg = xhat_local_flat.reshape(s[:-1] + n_local)
    x = pifft3d(xg, axis_name)
    return x.real.reshape(s)


def distributed_fft_flat(x_full: jnp.ndarray, n: tuple[int, ...], n_devices: int | None = None) -> jnp.ndarray:
    """
    Auto-decomposed drop-in replacement for the ``fft_`` closure every
    single-device solver in this project defines inline (``displacement_based.py``,
    ``mechanics.py``'s nonlinear driver, ``solvers/elliptic/scalar.py``) --
    ``jnp.fft.fftn(x.reshape(s[:-1] + n), axes=(-3, -2, -1)).reshape(s)``,
    full flat array in, full flat array out.

    Splits into x-slabs, runs ``pfft3d_flat`` under ``pmap``, and gathers
    the result back to one full array when more than one device is
    auto-detected (``choose_device_count``); at one device (the common case)
    this takes the exact original single-device ``jnp.fft.fftn`` path,
    byte-identical to before. Because the result is always a full, gathered
    array, every caller's own downstream logic -- gradient/divergence via
    ``iq``, the DC-bin macroscopic-strain trick (``.at[:, :, 0].set(...)``),
    CG's own reductions -- needs ZERO changes: only offloads the FFT
    compute itself, the same scope as ``operators.projection.Gamma0Operator``
    (not (yet) a memory-scaling win -- the full array still has to exist
    between FFT calls for that downstream logic to run on).
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
    """Inverse of ``distributed_fft_flat`` -- real result, matching
    ``jnp.fft.ifftn(...).real``, same auto-decompose/fallback behavior."""
    n_devices = choose_device_count(n, n_devices)
    if n_devices == 1:
        s = xhat_full.shape
        return jnp.fft.ifftn(xhat_full.reshape(s[:-1] + n), axes=(-3, -2, -1)).real.reshape(s)

    n_local = (n[0] // n_devices, n[1], n[2])
    x_sharded = split_x_slabs(xhat_full, n_devices)
    out_sharded = jax.pmap(lambda s: pifft3d_flat(s, n_local, "i"), axis_name="i")(x_sharded)
    return gather_x_slabs(out_sharded)
