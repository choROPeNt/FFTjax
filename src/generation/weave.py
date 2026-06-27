"""
Pure-Python biaxial plain-weave geometry generator.

Public API
----------
load_db()       — load fabric parameter database (data/fabric_db.json)
list_materials()— print database summary table
max_nesting()   — compute maximum nesting depth without yarn interference
build_weave()   — generate voxelised two-layer biaxial weave geometry
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

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
    print(f"{'ID':>6}  {'Description':<20}  {'Breite':>8}  {'Dicke':>7}"
          f"  {'Muster':>8}  {'nesting_max':>11}")
    print("-" * 72)
    for mid, v in db.items():
        if mid.startswith("_"):
            continue
        print(f"{mid:>6}  {v['description']:<20}  {v['Breite']:>8.5f}"
              f"  {v['Dicke']:>7.5f}  {v['Musterbreite']:>8.5f}"
              f"  {v['nesting_max']:>11.4f}")


# ── Nesting ───────────────────────────────────────────────────────────────────

def max_nesting(Dicke: float) -> float:
    """
    Maximum vertical nesting depth that keeps the resin interlayer gap intact.

    Derivation: bottom of upper-layer yarn (z_layer1 + b) must clear
    the top of the lower-layer yarn (3b) by exactly Matrixreiche_Zone:

        (Matrixreiche_Zone + 4b − δ) + b  =  3b + Matrixreiche_Zone
        δ_max = 2b = Dicke
    """
    return Dicke


# ── Geometry generator ────────────────────────────────────────────────────────

def build_weave(
    Breite:            float = 1.55222,
    Dicke:             float = 0.17084,
    Musterbreite:      float = 2.01811,
    Matrixreiche_Zone: float = 0.08,
    n_layer:           int   = 2,
    voxelsize:         float = 0.02,
    Vf_yarn:           float = 0.72,
    nesting:           float = 0.0,
    nesting_shift:     bool  = False,
    epsilon:           float | None = None,  # SDF interface half-width mm; None → 1.5 voxels
) -> tuple[
    tuple[int, int, int],
    tuple[float, float, float],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Voxelise a two-layer biaxial plain-weave geometry.

    Yarn centrelines follow cosine undulations; cross-sections are
    elliptical.  Interface voxels receive a smooth volume fraction via
    a tanh profile centred on the yarn surface (SDF+tanh, ε = 1.5 voxels).

    Parameters
    ----------
    Breite            : yarn width (mm)
    Dicke             : yarn thickness (mm)
    Musterbreite      : centre-to-centre yarn spacing (mm)
    Matrixreiche_Zone : resin-rich interlayer thickness (mm)
    n_layer           : number of fabric layers
    voxelsize         : isotropic voxel size (mm)
    Vf_yarn           : fibre volume fraction within a tow
    nesting           : vertical indentation depth (mm); 0 = no nesting
    nesting_shift     : if True shift odd layers by (M/2, M/2) in XY
    epsilon           : SDF interface half-width in mm; None → 1.5 × min(dx,dy,dz)

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
    vertOffset_12 = Matrixreiche_Zone + 2.0 * Dicke - nesting

    Lx = 2.0 * Musterbreite
    Ly = 2.0 * Musterbreite
    Lz = vertOffset_12 * n_layer - Matrixreiche_Zone

    nx = max(1, round(Lx / voxelsize))
    ny = max(1, round(Ly / voxelsize))
    nz = max(1, round(Lz / voxelsize))

    dx, dy, dz = Lx / nx, Ly / ny, Lz / nz

    xs = (np.arange(nx) + 0.5) * dx
    ys = (np.arange(ny) + 0.5) * dy
    zs = (np.arange(nz) + 0.5) * dz
    CX, CY, CZ = np.meshgrid(xs, ys, zs, indexing='ij')

    yarn_index   = np.full((nx, ny, nz), -1, dtype=np.int32)
    yarn_tangent = np.zeros((nx, ny, nz, 3), dtype=np.float64)
    best_phi     = np.zeros((nx, ny, nz), dtype=np.float64)

    a       = Breite / 2.0
    b       = Dicke  / 2.0
    epsilon = epsilon if epsilon is not None else 1.5 * min(dx, dy, dz)

    for layer in range(n_layer):
        z_layer   = layer * vertOffset_12
        yarn_weft = 2 * layer
        yarn_fill = 2 * layer + 1

        xy_off = Musterbreite * 0.5 if (nesting_shift and layer % 2 == 1) else 0.0

        y_weft = [Musterbreite * 0.5 + xy_off, Musterbreite * 1.5 + xy_off]
        x_fill = [Musterbreite * 0.5 + xy_off, Musterbreite * 1.5 + xy_off]
        x_cross = Musterbreite * 0.5 + xy_off
        y_cross = Musterbreite * 0.5 + xy_off

        z_lo   = z_layer
        z_hi   = z_layer + vertOffset_12 if layer < n_layer - 1 else Lz
        z_clip = (CZ >= z_lo) & (CZ < z_hi)

        def _phi(r_sq: np.ndarray) -> np.ndarray:
            sdf = (np.sqrt(np.maximum(r_sq, 1e-12)) - 1.0) * min(a, b)
            return 0.5 * (1.0 + np.tanh(-sdf / epsilon)) * z_clip

        def _update(phi: np.ndarray, yarn_id: int,
                    T0: np.ndarray | float,
                    T1: np.ndarray | float,
                    T2: np.ndarray | float) -> None:
            better = phi > best_phi
            best_phi[better]        = phi[better]
            for i, Ti in enumerate([T0, T1, T2]):
                if hasattr(Ti, '__len__'):
                    yarn_tangent[better, i] = Ti[better]  # type: ignore[index]
                else:
                    yarn_tangent[better, i] = float(Ti)
            yarn_index[better & (phi > 0.5)] = yarn_id

        for iy, y_c in enumerate(y_weft):
            sign  = 1 if iy == 0 else -1
            arg   = np.pi * (CX - x_cross) / Musterbreite
            z_c   = z_layer + 2*b + sign*b*np.cos(arg)
            dz_dx = -sign*b*(np.pi/Musterbreite)*np.sin(arg)
            dY    = np.minimum(np.abs(CY - y_c), Ly - np.abs(CY - y_c))
            norm  = np.sqrt(1.0 + dz_dx**2)
            _update(_phi((dY/a)**2 + ((CZ-z_c)/b)**2),
                    yarn_weft, 1.0/norm, 0.0, dz_dx/norm)

        for ix, x_c in enumerate(x_fill):
            sign  = -1 if ix == 0 else 1
            arg   = np.pi * (CY - y_cross) / Musterbreite
            z_c   = z_layer + 2*b + sign*b*np.cos(arg)
            dz_dy = -sign*b*(np.pi/Musterbreite)*np.sin(arg)
            dX    = np.minimum(np.abs(CX - x_c), Lx - np.abs(CX - x_c))
            norm  = np.sqrt(1.0 + dz_dy**2)
            _update(_phi((dX/a)**2 + ((CZ-z_c)/b)**2),
                    yarn_fill, 0.0, 1.0/norm, dz_dy/norm)

    phase           = (yarn_index >= 0).astype(np.int32)
    orientation     = yarn_tangent.copy()
    volume_fraction = best_phi * Vf_yarn

    return (
        (nx, ny, nz),
        (Lx, Ly, Lz),
        yarn_index.ravel(),
        yarn_tangent.reshape(-1, 3).T.copy(),
        orientation.reshape(-1, 3).T.copy(),
        volume_fraction.ravel(),
        phase.ravel(),
    )
