"""
jax.numpy biaxial plain-weave geometry generator.

Public API
----------
load_db()       — load fabric parameter database (data/fabric_db.json)
list_materials()— print database summary table
max_nesting()   — compute maximum nesting depth without yarn interference
build_weave()   — generate voxelised two-layer biaxial weave geometry

Differentiability
------------------
Grid shape (nx, ny, nz) is fixed by ``round(L/voxelsize)`` from
``pattern_width``, ``thickness``, ``resin_rich_zone``, ``nesting`` and
``voxelsize`` -- inherently non-differentiable, the same constraint any
fixed-resolution discretization has (you cannot have a JAX array whose
shape depends on a traced value). Those five parameters are cast to plain
Python floats up front specifically so a caller who accidentally passes a
JAX tracer there gets an immediate, clear error instead of a confusing one
deep inside array construction.

``width``, ``Vf_yarn`` and an explicitly-passed ``epsilon`` never enter the
shape computation at all, so they come out genuinely ``jax.grad``-able
through the smooth ``volume_fraction``/``yarn_tangent`` fields (the SDF+tanh
interface already makes those fields smooth in space; this just makes them
smooth in those parameters too) -- e.g. for calibrating yarn width or tow
packing fraction against measured data at a fixed mesh. ``yarn_index`` and
``phase`` stay hard-thresholded/discrete, same as every other phase label in
this codebase -- differentiating *through* a phase assignment isn't
meaningful, only the smooth volume-fraction field is.
"""

from __future__ import annotations

import json
from pathlib import Path

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)
import jax.numpy as jnp

_DB_PATH = Path(__file__).parent.parent.parent / "data" / "fabric_db.json"


# ── Database ──────────────────────────────────────────────────────────────────

def load_db() -> dict:
    """Load fabric_db.json and return as dict."""
    with open(_DB_PATH) as f:
        return json.load(f)


def list_materials() -> None:
    """Print a summary table of all fabric IDs in the database."""
    db  = load_db()
    vox = db.get("_voxelsize", "?")
    print(f"Common voxelsize : {vox} mm")
    print(f"{'ID':>6}  {'Description':<20}  {'Width':>8}  {'Thickness':>9}"
          f"  {'Pattern':>8}  {'nesting_max':>11}")
    print("-" * 72)
    for mid, v in db.items():
        if mid.startswith("_"):
            continue
        print(f"{mid:>6}  {v['description']:<20}  {v['width']:>8.5f}"
              f"  {v['thickness']:>9.5f}  {v['pattern_width']:>8.5f}"
              f"  {v['nesting_max']:>11.4f}")


# ── Nesting ───────────────────────────────────────────────────────────────────

def max_nesting(thickness: float) -> float:
    """
    Maximum vertical nesting depth that keeps the resin interlayer gap intact.

    Derivation: bottom of upper-layer yarn (z_layer1 + b) must clear
    the top of the lower-layer yarn (3b) by exactly resin_rich_zone:

        (resin_rich_zone + 4b − δ) + b  =  3b + resin_rich_zone
        δ_max = 2b = thickness
    """
    return thickness


# ── Geometry generator ────────────────────────────────────────────────────────

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
    is_yarn = better & (phi > 0.5)
    # plain Python int, not jnp.int32(yarn_id): the dtype constructor's return
    # type is weakly typed enough that it was tipping jnp.where's overload
    # resolution into the ambiguous Array | tuple[Array, ...] case above --
    # a bare int broadcasts against yarn_index (already int32) the same way.
    yarn_index_new = jnp.where(is_yarn, yarn_id, yarn_index)
    return yarn_index_new, yarn_tangent_new, best_phi_new


def build_weave(
    width:           float = 1.55222,
    thickness:       float = 0.17084,
    pattern_width:   float = 2.01811,
    resin_rich_zone: float = 0.08,
    n_layer:         int   = 2,
    voxelsize:       float = 0.02,
    Vf_yarn:         float = 0.72,
    nesting:         float = 0.0,
    nesting_shift:   bool  = False,
    epsilon:         float | None = None,  # SDF interface half-width mm; None → 1.5 voxels
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
    Voxelise a two-layer biaxial plain-weave geometry.

    Yarn centrelines follow cosine undulations; cross-sections are
    elliptical.  Interface voxels receive a smooth volume fraction via
    a tanh profile centred on the yarn surface (SDF+tanh, ε = 1.5 voxels).

    Parameters
    ----------
    width           : yarn width (mm) -- differentiable, doesn't affect grid shape
    thickness       : yarn thickness (mm) -- affects grid shape (Lz/nz), see module docstring
    pattern_width   : centre-to-centre yarn spacing (mm) -- affects grid shape (Lx/Ly/nx/ny)
    resin_rich_zone : resin-rich interlayer thickness (mm) -- affects grid shape (Lz/nz)
    n_layer         : number of fabric layers (static)
    voxelsize       : isotropic voxel size (mm) -- affects grid shape
    Vf_yarn         : fibre volume fraction within a tow -- differentiable, doesn't affect shape
    nesting         : vertical indentation depth (mm); 0 = no nesting -- affects grid shape (Lz/nz)
    nesting_shift   : if True shift odd layers by (pattern_width/2, pattern_width/2) in XY (static)
    epsilon         : SDF interface half-width in mm; None → 1.5 × min(dx,dy,dz) (computed from
                      the shape-fixing grid spacing); pass explicitly for a value that's free to
                      differentiate

    Returns
    -------
    n              : (nx, ny, nz)
    L              : (Lx, Ly, Lz)  mm
    yarn_index     : (Nv,) int      -1=matrix, 0..2*n_layer-1=yarn
    yarn_tangent   : (3, Nv) float  unit centreline tangent per voxel
    orientation    : (3, Nv) float  = yarn_tangent (fibre direction)
    volume_fraction: (Nv,) float    smooth φ·Vf_yarn; binary if voxelsize→0
    phase          : (Nv,) int      0=matrix, 1=yarn
    """
    # ── grid shape: concrete Python floats only, see module docstring ────────
    thickness_shape       = float(thickness)
    pattern_width_shape   = float(pattern_width)
    resin_rich_zone_shape = float(resin_rich_zone)
    nesting_shape         = float(nesting)

    vertOffset_12_shape = resin_rich_zone_shape + 2.0 * thickness_shape - nesting_shape
    Lx_shape = 2.0 * pattern_width_shape
    Ly_shape = 2.0 * pattern_width_shape
    Lz_shape = vertOffset_12_shape * n_layer - resin_rich_zone_shape

    nx = max(1, round(Lx_shape / voxelsize))
    ny = max(1, round(Ly_shape / voxelsize))
    nz = max(1, round(Lz_shape / voxelsize))

    # ── field computation: original (possibly traced) parameter values ───────
    vertOffset_12 = resin_rich_zone + 2.0 * thickness - nesting
    Lx = 2.0 * pattern_width
    Ly = 2.0 * pattern_width
    Lz = vertOffset_12 * n_layer - resin_rich_zone

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
    ab_min = jnp.minimum(a, b)
    eps = epsilon if epsilon is not None else 1.5 * min(dx, dy, dz)

    for layer in range(n_layer):
        z_layer   = layer * vertOffset_12
        yarn_weft = 2 * layer
        yarn_fill = 2 * layer + 1

        xy_off = pattern_width * 0.5 if (nesting_shift and layer % 2 == 1) else 0.0

        y_weft = [pattern_width * 0.5 + xy_off, pattern_width * 1.5 + xy_off]
        x_fill = [pattern_width * 0.5 + xy_off, pattern_width * 1.5 + xy_off]
        x_cross = pattern_width * 0.5 + xy_off
        y_cross = pattern_width * 0.5 + xy_off

        z_lo   = z_layer
        z_hi   = z_layer + vertOffset_12 if layer < n_layer - 1 else Lz
        z_clip = (CZ >= z_lo) & (CZ < z_hi)

        def _phi(r_sq: jnp.ndarray) -> jnp.ndarray:
            sdf = (jnp.sqrt(jnp.maximum(r_sq, 1e-12)) - 1.0) * ab_min
            return 0.5 * (1.0 + jnp.tanh(-sdf / eps)) * z_clip

        for iy, y_c0 in enumerate(y_weft):
            sign  = 1 if iy == 0 else -1
            arg   = jnp.pi * (CX - x_cross) / pattern_width
            z_c   = z_layer + 2 * b + sign * b * jnp.cos(arg)
            dz_dx = -sign * b * (jnp.pi / pattern_width) * jnp.sin(arg)
            dY    = jnp.minimum(jnp.abs(CY - y_c0), Ly - jnp.abs(CY - y_c0))
            norm  = jnp.sqrt(1.0 + dz_dx ** 2)
            phi   = _phi((dY / a) ** 2 + ((CZ - z_c) / b) ** 2)
            T     = (1.0 / norm, jnp.zeros_like(norm), dz_dx / norm)
            yarn_index, yarn_tangent, best_phi = _update(
                yarn_index, yarn_tangent, best_phi, phi, yarn_weft, T,
            )

        for ix, x_c0 in enumerate(x_fill):
            sign  = -1 if ix == 0 else 1
            arg   = jnp.pi * (CY - y_cross) / pattern_width
            z_c   = z_layer + 2 * b + sign * b * jnp.cos(arg)
            dz_dy = -sign * b * (jnp.pi / pattern_width) * jnp.sin(arg)
            dX    = jnp.minimum(jnp.abs(CX - x_c0), Lx - jnp.abs(CX - x_c0))
            norm  = jnp.sqrt(1.0 + dz_dy ** 2)
            phi   = _phi((dX / a) ** 2 + ((CZ - z_c) / b) ** 2)
            T     = (jnp.zeros_like(norm), 1.0 / norm, dz_dy / norm)
            yarn_index, yarn_tangent, best_phi = _update(
                yarn_index, yarn_tangent, best_phi, phi, yarn_fill, T,
            )

    phase           = (yarn_index >= 0).astype(jnp.int32)
    orientation     = yarn_tangent
    volume_fraction = best_phi * Vf_yarn

    return (
        (nx, ny, nz),
        (Lx, Ly, Lz),
        yarn_index.ravel(),
        yarn_tangent.reshape(-1, 3).T,
        orientation.reshape(-1, 3).T,
        volume_fraction.ravel(),
        phase.ravel(),
    )
