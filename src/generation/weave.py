"""
Biaxial plain-weave geometry: ``build_weave`` resolves the layer placement
(the expensive nesting search, once), ``voxelize`` samples it at any voxel size.

Sharp interfaces: each voxel is matrix or one yarn, ``volume_fraction`` is 0
or Vf_yarn. Assemble by ``phase``/``yarn_index``, not by blending. Being hard
thresholds, the outputs have no useful jax.grad.
"""

from __future__ import annotations

from typing import NamedTuple

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)
import jax.numpy as jnp


def _search_nesting(
    width: float,
    thickness: float,
    pattern_width: float,
    resin_rich_zone: float,
    n_layer: int,
    grid_res: int = 128,
) -> tuple[list[float], list[float], list[float], float]:
    """
    Nesting search (TexGen ``MaxNestLayers``): search the relative in-plane
    offset of adjacent layers for the deepest nesting without (X, Y) overlap.
    Equal registration would only ever split layers along a flat Z plane.

    Even n_layer: odd layers share one offset (dx, dy), and the gaps alternate
    g1 (even -> odd) and g2 (odd -> even), each tightened to its own
    direction's worst case. A uniform per-layer Z shift would add nothing (it
    equals a different constant offset).

    Odd n_layer: no 2-coloring of the ring, so layer i is rotated by
    (i dx, i dy) with one uniform gap; offsets that land an intermediate layer
    on a whole pattern width are skipped (they mirror its undulation).

    ``grid_res`` is the search's own resolution, independent of voxelize().
    Remaining overlaps at voxelize() time go to the deeper ellipse (_update).

    Returns
    -------
    xy_off_x, xy_off_y, z_layer : per-layer offsets and Z positions
    Lz                          : domain height
    """
    if n_layer < 2:
        return [0.0], [0.0], [0.0], resin_rich_zone + 2.0 * thickness

    a = width / 2.0
    b = thickness / 2.0
    x_cross = y_cross = pattern_width * 0.5
    Lx = Ly = 2.0 * pattern_width

    xs = (jnp.arange(grid_res) + 0.5) * (Lx / grid_res)
    ys = (jnp.arange(grid_res) + 0.5) * (Ly / grid_res)
    GX, GY = jnp.meshgrid(xs, ys, indexing='ij')

    def _candidates(dx: float, dy: float) -> tuple[list[jnp.ndarray], list[jnp.ndarray]]:
        """Z centers and in-range masks of a layer's 4 yarns at offset (dx, dy), relative to z_layer."""
        y_weft = [pattern_width * 0.5 + dy, pattern_width * 1.5 + dy]
        x_fill = [pattern_width * 0.5 + dx, pattern_width * 1.5 + dx]
        zc: list[jnp.ndarray] = []
        active: list[jnp.ndarray] = []
        for iy, y_c0 in enumerate(y_weft):
            sign = 1 if iy == 0 else -1
            arg = jnp.pi * (GX - (x_cross + dx)) / pattern_width
            zc.append(2 * b + sign * b * jnp.cos(arg))
            dY = jnp.minimum(jnp.abs(GY - y_c0), Ly - jnp.abs(GY - y_c0))
            active.append(dY <= a)
        for ix, x_c0 in enumerate(x_fill):
            sign = -1 if ix == 0 else 1
            arg = jnp.pi * (GY - (y_cross + dy)) / pattern_width
            zc.append(2 * b + sign * b * jnp.cos(arg))
            dX = jnp.minimum(jnp.abs(GX - x_c0), Lx - jnp.abs(GX - x_c0))
            active.append(dX <= a)
        return zc, active

    zc0, active0 = _candidates(0.0, 0.0)

    def _delta_min(dx: float, dy: float) -> float:
        zc1, active1 = _candidates(dx, dy)
        best = jnp.inf
        for i in range(4):
            for j in range(4):
                pair_active = active0[i] & active1[j]
                diff = zc1[j] - zc0[i]
                best = jnp.minimum(best, jnp.min(jnp.where(pair_active, diff, jnp.inf)))
        return float(best)

    def _mirrors_some_layer(k: int) -> bool:
        if k == 0:
            return False
        return any((i * k) % n_layer == 0 for i in range(1, n_layer))

    if n_layer % 2 == 0:
        best_score, best_dx, best_dy, best_fwd, best_bwd = -float("inf"), 0.0, 0.0, 0.0, 0.0
        for kx in range(n_layer):
            for ky in range(n_layer):
                if kx == 0 and ky == 0:
                    continue
                dx, dy = kx * pattern_width / n_layer, ky * pattern_width / n_layer
                fwd = _delta_min(dx, dy)
                bwd = _delta_min(-dx, -dy)
                score = fwd + bwd
                if score > best_score:
                    best_score, best_dx, best_dy, best_fwd, best_bwd = score, dx, dy, fwd, bwd

        g1 = resin_rich_zone + thickness - best_fwd
        g2 = resin_rich_zone + thickness - best_bwd
        n_pairs = n_layer // 2
        Lz = n_pairs * (g1 + g2)

        xy_off_x: list[float] = []
        xy_off_y: list[float] = []
        z_layer:  list[float] = []
        z = 0.0
        for i in range(n_layer):
            if i % 2 == 0:
                xy_off_x.append(0.0)
                xy_off_y.append(0.0)
            else:
                xy_off_x.append(best_dx)
                xy_off_y.append(best_dy)
            z_layer.append(z)
            z += g1 if i % 2 == 0 else g2
        return xy_off_x, xy_off_y, z_layer, Lz

    best_val, best_kx, best_ky = -float("inf"), 0, 0
    for kx in range(n_layer):
        for ky in range(n_layer):
            if kx == 0 and ky == 0:
                continue
            if _mirrors_some_layer(kx) or _mirrors_some_layer(ky):
                continue
            val = _delta_min(kx * pattern_width / n_layer, ky * pattern_width / n_layer)
            if val > best_val:
                best_val, best_kx, best_ky = val, kx, ky

    dx_step = best_kx * pattern_width / n_layer
    dy_step = best_ky * pattern_width / n_layer
    nesting = thickness + best_val
    vertOffset_12 = resin_rich_zone + 2.0 * thickness - nesting
    Lz = n_layer * vertOffset_12
    xy_off_x = [i * dx_step for i in range(n_layer)]
    xy_off_y = [i * dy_step for i in range(n_layer)]
    z_layer  = [i * vertOffset_12 for i in range(n_layer)]
    return xy_off_x, xy_off_y, z_layer, Lz


class WeaveGeometry(NamedTuple):
    """Resolved weave: yarn dimensions plus per-layer offsets and Z; independent of any voxel grid."""
    width:           float
    thickness:       float
    pattern_width:   float
    Vf_yarn:         float
    n_layer:         int
    xy_off_x:        list[float]
    xy_off_y:        list[float]
    z_layer:         list[float]
    Lx:              float
    Ly:              float
    Lz:              float


def build_weave(
    width:           float = 1.0,
    thickness:       float = 0.1,
    pattern_width:   float = 2.0,
    resin_rich_zone: float = 0.05,
    n_layer:         int   = 2,
    Vf_yarn:         float = 0.7,
    max_nesting:     bool  = False,
) -> WeaveGeometry:
    """
    Resolve an n-layer plain weave; sample it with voxelize().

    Parameters
    ----------
    width, thickness : yarn width and thickness (mm)
    pattern_width    : yarn spacing (mm)
    resin_rich_zone  : interlayer gap added on top of nesting (mm)
    n_layer          : number of layers
    Vf_yarn          : fiber volume fraction within a tow
    max_nesting      : search the deepest nesting (gives each layer its own
                       footprint, needed for 3-D interleaving); False stacks
                       layers flat with gap resin_rich_zone + 2 thickness

    Returns
    -------
    WeaveGeometry
    """
    if max_nesting:
        xy_off_x_list, xy_off_y_list, z_layer_list, Lz = _search_nesting(
            width, thickness, pattern_width, resin_rich_zone, n_layer,
        )
    else:
        flat_gap = resin_rich_zone + 2.0 * thickness
        xy_off_x_list = [0.0] * n_layer
        xy_off_y_list = [0.0] * n_layer
        z_layer_list  = [i * flat_gap for i in range(n_layer)]
        Lz = n_layer * flat_gap

    Lx = Ly = 2.0 * pattern_width

    return WeaveGeometry(
        width=width, thickness=thickness, pattern_width=pattern_width,
        Vf_yarn=Vf_yarn, n_layer=n_layer,
        xy_off_x=xy_off_x_list, xy_off_y=xy_off_y_list, z_layer=z_layer_list,
        Lx=Lx, Ly=Ly, Lz=Lz,
    )


def _update(
    yarn_index: jnp.ndarray,
    yarn_tangent: jnp.ndarray,
    best_phi: jnp.ndarray,
    phi: jnp.ndarray,
    yarn_id: int,
    T: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Keep, per voxel, the yarn with the highest priority ``phi``.

    ``phi`` grows toward the yarn centerline (see voxelize's _phi), so the
    deepest yarn wins regardless of processing order; a 0/1 flag would tie and
    favor earlier layers. Fully annotated because jnp.where's 1-argument
    overload otherwise breaks type inference downstream.
    """
    better = phi > best_phi
    best_phi_new = jnp.where(better, phi, best_phi)
    T_stack = jnp.stack(T, axis=-1)
    yarn_tangent_new = jnp.where(better[..., None], T_stack, yarn_tangent)
    is_yarn = better
    # A plain int, not jnp.int32(yarn_id): that tipped jnp.where's overload
    # into the ambiguous return type.
    yarn_index_new = jnp.where(is_yarn, yarn_id, yarn_index)
    return yarn_index_new, yarn_tangent_new, best_phi_new


def voxelize(
    geometry: WeaveGeometry,
    voxelsize: float,
) -> tuple[
    tuple[int, int, int],
    tuple[float, float, float],
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]:
    """
    Sample a WeaveGeometry on an isotropic grid: cosine centerlines,
    elliptical cross-sections, sharp membership.

    Parameters
    ----------
    geometry  : from build_weave()
    voxelsize : voxel size (mm)

    Returns
    -------
    n, L            : grid shape and length
    yarn_index      : (Nv,) int, -1 matrix, one id per tow (2 weft + 2 fill per layer)
    yarn_tangent    : (3, Nv) centerline tangent
    orientation     : (3, Nv), equal to yarn_tangent
    volume_fraction : (Nv,) 0 or Vf_yarn
    phase           : (Nv,) int, 0 matrix, 1 yarn
    """
    width, thickness, pattern_width = geometry.width, geometry.thickness, geometry.pattern_width
    Vf_yarn, n_layer = geometry.Vf_yarn, geometry.n_layer
    Lx, Ly, Lz = geometry.Lx, geometry.Ly, geometry.Lz

    nx = max(1, round(float(Lx) / voxelsize))
    ny = max(1, round(float(Ly) / voxelsize))
    nz = max(1, round(float(Lz) / voxelsize))

    dx, dy, dz = Lx / nx, Ly / ny, Lz / nz

    xs = (jnp.arange(nx) + 0.5) * dx
    ys = (jnp.arange(ny) + 0.5) * dy
    zs = (jnp.arange(nz) + 0.5) * dz
    CX, CY, CZ = jnp.meshgrid(xs, ys, zs, indexing='ij')

    yarn_index   = jnp.full((nx, ny, nz), -1, dtype=jnp.int32)
    yarn_tangent = jnp.zeros((nx, ny, nz, 3))
    best_phi     = jnp.zeros((nx, ny, nz))

    a = width / 2.0
    b = thickness / 2.0

    for layer in range(n_layer):
        xy_off_x = geometry.xy_off_x[layer]
        xy_off_y = geometry.xy_off_y[layer]
        z_layer  = geometry.z_layer[layer]

        y_weft = [pattern_width * 0.5 + xy_off_y, pattern_width * 1.5 + xy_off_y]
        x_fill = [pattern_width * 0.5 + xy_off_x, pattern_width * 1.5 + xy_off_x]
        yarn_base = layer * (len(y_weft) + len(x_fill))
        x_cross = pattern_width * 0.5 + xy_off_x
        y_cross = pattern_width * 0.5 + xy_off_y

        def _phi(r_sq: jnp.ndarray) -> jnp.ndarray:
            """Priority: 0 outside the ellipse, 2 - r_sq in (1, 2] inside."""
            return jnp.where(r_sq <= 1.0, 2.0 - r_sq, 0.0)

        for iy, y_c0 in enumerate(y_weft):
            sign  = 1 if iy == 0 else -1
            arg   = jnp.pi * (CX - x_cross) / pattern_width
            z_c   = z_layer + 2 * b + sign * b * jnp.cos(arg)
            dz_dx = -sign * b * (jnp.pi / pattern_width) * jnp.sin(arg)
            dY    = jnp.minimum(jnp.abs(CY - y_c0), Ly - jnp.abs(CY - y_c0))
            # Periodic in Z too: deeply nested ellipses wrap past Lz.
            dZ    = jnp.abs(CZ - z_c)
            dZ    = jnp.minimum(dZ, Lz - dZ)
            norm  = jnp.sqrt(1.0 + dz_dx ** 2)
            phi   = _phi((dY / a) ** 2 + (dZ / b) ** 2)
            T     = (1.0 / norm, jnp.zeros_like(norm), dz_dx / norm)
            yarn_index, yarn_tangent, best_phi = _update(
                yarn_index, yarn_tangent, best_phi, phi, yarn_base + iy, T,
            )

        for ix, x_c0 in enumerate(x_fill):
            sign  = -1 if ix == 0 else 1
            arg   = jnp.pi * (CY - y_cross) / pattern_width
            z_c   = z_layer + 2 * b + sign * b * jnp.cos(arg)
            dz_dy = -sign * b * (jnp.pi / pattern_width) * jnp.sin(arg)
            dX    = jnp.minimum(jnp.abs(CX - x_c0), Lx - jnp.abs(CX - x_c0))
            dZ    = jnp.abs(CZ - z_c)
            dZ    = jnp.minimum(dZ, Lz - dZ)
            norm  = jnp.sqrt(1.0 + dz_dy ** 2)
            phi   = _phi((dX / a) ** 2 + (dZ / b) ** 2)
            T     = (jnp.zeros_like(norm), 1.0 / norm, dz_dy / norm)
            yarn_index, yarn_tangent, best_phi = _update(
                yarn_index, yarn_tangent, best_phi, phi, yarn_base + len(y_weft) + ix, T,
            )

    phase           = (yarn_index >= 0).astype(jnp.int32)
    orientation     = yarn_tangent
    # From phase, not best_phi (a priority, not a 0/1 flag).
    volume_fraction = phase.astype(jnp.float64) * Vf_yarn

    return (
        (nx, ny, nz),
        (Lx, Ly, Lz),
        yarn_index.ravel(),
        yarn_tangent.reshape(-1, 3).T,
        orientation.reshape(-1, 3).T,
        volume_fraction.ravel(),
        phase.ravel(),
    )
