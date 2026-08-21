"""
jax.numpy biaxial plain-weave geometry generator.

Public API
----------
build_weave()   — resolve an n-layer biaxial weave's geometry (layer
                  placement/nesting), independent of any voxel grid
voxelize()      — sample a WeaveGeometry onto a voxel grid at a given voxelsize

Split so that the (potentially expensive) nesting search runs once, and the
same resolved geometry can be cheaply re-voxelised at multiple voxelsizes
(e.g. for a resolution sweep) without re-solving placement each time.

Sharp interfaces, no material blending: every voxel is entirely matrix or
entirely one yarn (hard ellipse membership test, no SDF/tanh smoothing at
the boundary). ``volume_fraction`` is exactly ``0`` or ``Vf_yarn`` -- never
a partial/blended value -- matching ``phase`` everywhere. Downstream
material assembly should gather by ``phase``/``yarn_index`` (hard,
sharp-interface assignment, e.g. ``materialmodels.assembly.assemble_C_field``),
not blend by ``volume_fraction`` (that's ``materialmodels.assembly``'s smooth/
partial-volume-fraction assemblers, not applicable to this generator's output).

Because every field is a hard threshold of the underlying geometry, none of
``voxelize``'s parameters are meaningfully ``jax.grad``-able through the
output fields (a step function's gradient is zero almost everywhere) --
this module is jax.numpy purely for consistency with the rest of the
codebase, not for differentiability.
"""

from __future__ import annotations

from typing import NamedTuple

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)
import jax.numpy as jnp


# ── Nesting search ────────────────────────────────────────────────────────────

def _search_nesting(
    width: float,
    thickness: float,
    pattern_width: float,
    resin_rich_zone: float,
    n_layer: int,
    grid_res: int = 128,
) -> tuple[list[float], list[float], list[float], float]:
    """
    TexGen-style nesting search (see ``MaxNestLayers`` in TexGen's
    ``Core/TextileLayered.cpp``, github.com/louisepb/TexGen): rather than
    assuming adjacent layers share the same in-plane (X/Y) registration --
    which forces their crossover competition to always resolve along a flat
    Z-plane, never genuine 3D interleaving, no matter how deep nesting goes
    (two congruent shapes differing only by a Z-translation always split
    along their perpendicular bisector) -- search over the relative in-plane
    offset between adjacent layers for whichever offset lets them nest
    deepest without any local (X,Y) overlap.

    For even n_layer, uses an *alternating* even/odd in-plane registration
    (every even layer at offset 0, every odd layer at a single searched
    offset (dx, dy)) combined with *alternating* Z gaps g1 (even->odd) and
    g2 (odd->even), each independently tightened to its own direction's
    worst case (Delta_min(dx,dy) for g1, Delta_min(-dx,-dy) for g2 -- these
    generally differ, since the weft/fill sign convention isn't symmetric
    under negating the offset). This is strictly better than forcing both
    gaps to the *same* uniform value (the older approach): each direction
    gets exactly as much room as its own geometry needs rather than both
    being sized to whichever direction is worse. A naive *uniform* per-layer
    Z increment (layer i at i*vertOffset_12 + i*dz) would add nothing here
    -- it's mathematically identical to a different vertOffset_12
    (i*vertOffset_12 + i*dz = i*(vertOffset_12+dz)) -- so the only way a Z
    degree of freedom does anything is this alternating-gap form, where
    layer i moving dz closer to one neighbour necessarily moves it dz
    farther from the other (nonlinear in i, not absorbable into a single
    constant).

    For odd n_layer, this can't alternate cleanly (there's no way to 2-colour
    a ring with an odd number of nodes), so it falls back to the older
    approach: a single rotation increment (dx_step, dy_step) = layer i
    offset by (i*dx_step, i*dy_step), which keeps every adjacent pair
    (including the wrap) at the *same* relative offset and hence the same
    uniform gap -- see _delta_min below for how that offset is chosen.
    Candidates dx_step = kx*pattern_width/n_layer (ky similarly) that would
    mirror some intermediate layer are excluded: if (i*kx) % n_layer == 0
    for some 1 <= i < n_layer (kx != 0), layer i's cumulative offset lands
    on a whole pattern_width multiple, sending cos(arg) -> -cos(arg) in
    that layer's undulation -- sign-flipped (mirrored top-to-bottom), not
    translated. This doesn't arise in the even-n_layer alternating case:
    there every odd layer uses the *same* (non-cumulative) offset, and
    kx in [1, n_layer) can never itself be a whole multiple of n_layer.

    grid_res is a fixed, dedicated resolution for this search -- deliberately
    *not* derived from the eventual voxelize() voxelsize (see module
    docstring: geometry resolution and voxel resolution are separate
    concerns, resolved once each rather than re-coupled every voxelize()
    call). It only needs to be fine enough to find the right offset, not to
    voxelise the final geometry.

    Returns (xy_off_x, xy_off_y, z_layer, Lz): per-layer lists (length
    n_layer) of in-plane offset and Z position, plus the total domain
    height. Where two layers' ellipses still end up claiming the same voxel
    at voxelize() time (nesting pushed a gap below a yarn's natural
    extent), _update resolves it by depth-inside-ellipse, not a hard crop.
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
        # Local z-centre (baseline 2b + oscillation) and in-range mask for
        # this layer's 4 yarn candidates (weft iy0, weft iy1, fill ix0, fill
        # ix1) at in-plane offset (dx, dy) -- same formulas as voxelize's
        # main loop, but z_layer-independent (baseline only), since only the
        # *relative* Z separation between two layers matters here.
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
        # See docstring -- only relevant to the odd-n_layer rotating
        # fallback (a cumulative i*k offset); the even-n_layer alternating
        # branch below never uses a cumulative offset, so this check
        # doesn't apply there.
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
                fwd = _delta_min(dx, dy)      # even layer i -> odd layer i+1
                bwd = _delta_min(-dx, -dy)    # odd layer i -> even layer i+1
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

    # odd n_layer: uniform rotating fallback (can't 2-colour an odd ring)
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


# ── Geometry (resolution-independent) ────────────────────────────────────────

class WeaveGeometry(NamedTuple):
    """
    A fully-resolved biaxial plain-weave geometry: yarn dimensions plus each
    layer's in-plane offset and Z position. Independent of any voxel grid --
    pass to voxelize() with whatever voxelsize you want to actually sample
    it, as many times as you like, without re-running the nesting search.
    """
    width:           float
    thickness:       float
    pattern_width:   float
    Vf_yarn:         float
    n_layer:         int
    xy_off_x:        list[float]   # per layer, length n_layer
    xy_off_y:        list[float]   # per layer, length n_layer
    z_layer:         list[float]   # per layer, length n_layer
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
    Resolve an n-layer biaxial plain-weave geometry: each layer's in-plane
    offset and Z position. No voxel grid is built here -- call voxelize()
    on the result to sample it at a given voxelsize (see module docstring
    for why these are split).

    Parameters
    ----------
    width           : yarn width (mm)
    thickness       : yarn thickness (mm)
    pattern_width   : centre-to-centre yarn spacing (mm)
    resin_rich_zone : resin-rich interlayer thickness (mm), added as a
                      genuine gap on top of whatever nesting finds
    n_layer         : number of fabric layers
    Vf_yarn         : fibre volume fraction within a tow
    max_nesting     : if True, run _search_nesting() to find the per-layer
                      in-plane registration and Z position that nests
                      layers as deep as possible without any local (X,Y)
                      overlap (see _search_nesting -- for even n_layer this
                      alternates both the in-plane offset and the Z gap
                      size layer-to-layer; for odd n_layer it falls back to
                      a uniform rotating offset/gap, since an odd ring
                      can't be 2-coloured). This is also what gives each
                      layer a distinct in-plane footprint, required for
                      genuine 3D interleaving; see _update. If False
                      (default), layers stack flat: no in-plane offset, no
                      nesting (every gap = resin_rich_zone + 2*thickness)
                      -- e.g. for a quick single-layer-shape sanity check.

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


# ── Voxelization ──────────────────────────────────────────────────────────────

def _update(
    yarn_index: jnp.ndarray,
    yarn_tangent: jnp.ndarray,
    best_phi: jnp.ndarray,
    phi: jnp.ndarray,
    yarn_id: int,
    T: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Functional replacement for the old numpy in-place 'keep the best phi' update.

    ``phi`` is a continuous priority (see ``_phi`` in ``voxelize``): 0
    outside a yarn's ellipse, in (1, 2] inside it -- higher where a voxel is
    closer to that yarn's own centreline. Comparing this (not a hard 0/1
    membership flag) makes crossover resolution order-independent: whichever
    candidate is genuinely deepest inside its own ellipse wins the voxel,
    regardless of which layer/yarn happened to be processed first. A hard
    0/1 ``phi`` would tie every overlap 1-vs-1, and ``>`` (strict) then
    always favours whichever candidate got there first -- silently biasing
    every overlap toward earlier-processed layers.

    Explicitly typed (unlike the rest of this module's internal helpers) because
    jnp.where is overloaded -- where(condition) alone returns tuple[Array, ...]
    (like nonzero), where(condition, x, y) returns Array; without annotations
    here, static type checkers can't rule out the 1-arg overload and infer
    Array | tuple[Array, ...] for the outputs, breaking .reshape/.T downstream.
    """
    better = phi > best_phi
    best_phi_new = jnp.where(better, phi, best_phi)
    T_stack = jnp.stack(T, axis=-1)   # (nx, ny, nz, 3)
    yarn_tangent_new = jnp.where(better[..., None], T_stack, yarn_tangent)
    # phi > 0 iff inside the ellipse (see _phi); best_phi starts at 0, so
    # "better" already implies phi > 0 -- no separate membership threshold
    # needed.
    is_yarn = better
    # plain Python int, not jnp.int32(yarn_id): the dtype constructor's return
    # type is weakly typed enough that it was tipping jnp.where's overload
    # resolution into the ambiguous Array | tuple[Array, ...] case above --
    # a bare int broadcasts against yarn_index (already int32) the same way.
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
    Sample a WeaveGeometry (see build_weave) onto an isotropic voxel grid.

    Yarn centrelines follow cosine undulations; cross-sections are
    elliptical, with a hard (sharp-interface) membership test -- a voxel is
    entirely inside one yarn's ellipse or entirely outside it, no partial/
    blended boundary values.

    Parameters
    ----------
    geometry  : a WeaveGeometry from build_weave()
    voxelsize : isotropic voxel size (mm)

    Returns
    -------
    n              : (nx, ny, nz)
    L              : (Lx, Ly, Lz)  mm
    yarn_index     : (Nv,) int      -1=matrix, 0..4*n_layer-1=yarn -- one id per
                     individual yarn tow (2 weft + 2 fill copies per layer,
                     each physically separate, not one id per weft/fill role)
    yarn_tangent   : (3, Nv) float  unit centreline tangent per voxel
    orientation    : (3, Nv) float  = yarn_tangent (fibre direction)
    volume_fraction: (Nv,) float    exactly 0 or Vf_yarn -- matches phase everywhere,
                     never a partial/blended value
    phase          : (Nv,) int      0=matrix, 1=yarn
    """
    width, thickness, pattern_width = geometry.width, geometry.thickness, geometry.pattern_width
    Vf_yarn, n_layer = geometry.Vf_yarn, geometry.n_layer
    Lx, Ly, Lz = geometry.Lx, geometry.Ly, geometry.Lz

    # grid shape: round() needs concrete values, not JAX tracers
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
        # Each layer's in-plane offset and Z position come straight from
        # the resolved WeaveGeometry -- see build_weave/_search_nesting for
        # why these aren't simply layer*constant when max_nesting=True and
        # n_layer is even (alternating offset/gaps).
        xy_off_x = geometry.xy_off_x[layer]
        xy_off_y = geometry.xy_off_y[layer]
        z_layer  = geometry.z_layer[layer]

        y_weft = [pattern_width * 0.5 + xy_off_y, pattern_width * 1.5 + xy_off_y]
        x_fill = [pattern_width * 0.5 + xy_off_x, pattern_width * 1.5 + xy_off_x]
        # every individual yarn tow gets its own id -- the two parallel weft
        # copies (and the two fill copies) per layer used to share one id
        # each, even though they're physically separate, non-touching yarns.
        yarn_base = layer * (len(y_weft) + len(x_fill))
        x_cross = pattern_width * 0.5 + xy_off_x
        y_cross = pattern_width * 0.5 + xy_off_y

        def _phi(r_sq: jnp.ndarray) -> jnp.ndarray:
            # Hard ellipse membership (r_sq<=1), no SDF/tanh smoothing at the
            # boundary -- but a *continuous* priority within that hard
            # boundary (0 outside, in (1,2] inside, higher nearer the
            # centreline) so _update's crossover resolution is
            # order-independent (see _update docstring) rather than always
            # favouring whichever layer got processed first.
            # No separate z-window mask: the ellipse's own (dY/a)^2+(dZ/b)^2<=1
            # condition already confines matches to z_c +/- b on its own.
            return jnp.where(r_sq <= 1.0, 2.0 - r_sq, 0.0)

        for iy, y_c0 in enumerate(y_weft):
            sign  = 1 if iy == 0 else -1
            arg   = jnp.pi * (CX - x_cross) / pattern_width
            z_c   = z_layer + 2 * b + sign * b * jnp.cos(arg)
            dz_dx = -sign * b * (jnp.pi / pattern_width) * jnp.sin(arg)
            dY    = jnp.minimum(jnp.abs(CY - y_c0), Ly - jnp.abs(CY - y_c0))
            # periodic (minimum-image) Z distance, same pattern as dY/dX above:
            # at high nesting a layer's z_c can sit close enough to Lz that its
            # ellipse should wrap around to voxels near z=0 (mirrors the
            # existing periodic wrap in X/Y -- this project's solvers assume a
            # periodic unit cell throughout, not just in-plane).
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
    # best_phi is a continuous crossover-resolution priority (see _phi/
    # _update), not a 0/1 membership flag -- volume_fraction must still be
    # exactly 0 or Vf_yarn (sharp interface, matches phase everywhere), so
    # derive it from phase, not best_phi.
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
