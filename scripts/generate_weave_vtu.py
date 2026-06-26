"""
Pure-Python two-layer biaxial plain-weave geometry generator.

Produces a voxel VTU compatible with the FFTjax pipeline (same CellData
fields as TexGen: YarnIndex, YarnTangent) without requiring TexGen.

Geometry
--------
Two fabric layers, each with weft (X-direction) and fill (Y-direction)
yarns.  Cross-sections are elliptical; centrelines follow cosine
undulations.  Yarn indices:
    -1  matrix
     0  layer-0 weft
     1  layer-0 fill
     2  layer-1 weft
     3  layer-1 fill

Usage
-----
    python scripts/generate_weave_vtu.py [--out data/myweave.vtu]
"""

import argparse
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from post.io import IncrementalWriter


# ── Geometry parameters ───────────────────────────────────────────────────────

def max_nesting(Dicke: float) -> float:
    """
    Maximum vertical nesting depth that keeps the Matrixreiche_Zone resin gap intact.

    Derivation: the bottom of the upper layer yarn (z_layer1 + b) must equal
    the top of the lower layer yarn (3b) plus the resin gap.  Solving for δ:

        (Matrixreiche_Zone + 4b - δ) + b  =  3b + Matrixreiche_Zone
        δ_max = 2b = Dicke
    """
    return Dicke


def build_weave(
    Breite:            float = 1.55222,   # mm  yarn width (Y for weft, X for fill)
    Dicke:             float = 0.17084,   # mm  yarn thickness
    Musterbreite:      float = 2.01811,   # mm  centre-to-centre spacing
    Matrixreiche_Zone: float = 0.08,      # mm  resin-rich interlayer
    n_layer:           int   = 2,
    voxelsize:         float = 0.02,      # mm  isotropic voxel size
    Vf_yarn:           float = 0.72,      # fibre volume fraction within a tow
    nesting:           float = 0.0,       # mm  vertical indentation (0 = no nesting)
    nesting_shift:     bool  = False,     # shift odd layers by (M/2, M/2) in XY
) -> tuple[tuple, tuple, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns
    -------
    n              : (nx, ny, nz)
    L              : (Lx, Ly, Lz)  mm
    yarn_index     : (Nv,) int      -1=matrix, 0..2*n_layer-1=yarn
    yarn_tangent   : (3, Nv) float  unit vector along centreline
    orientation    : (3, Nv) float  = yarn_tangent  (fibre direction)
    volume_fraction: (Nv,) float    Vf_yarn inside yarn, 0 in matrix
    phase          : (Nv,) int      0=matrix, 1=yarn
    """
    # ── domain ───────────────────────────────────────────────────────────────
    Faktor        = Musterbreite / Breite   # = 1/coverage
    vertOffset_12 = Matrixreiche_Zone + 2.0 * Dicke - nesting   # nesting reduces layer gap

    Lx = 2.0 * Musterbreite
    Ly = 2.0 * Musterbreite
    Lz = vertOffset_12 * n_layer - Matrixreiche_Zone

    nx = max(1, round(Lx / voxelsize))
    ny = max(1, round(Ly / voxelsize))
    nz = max(1, round(Lz / voxelsize))
    Nv = nx * ny * nz

    dx = Lx / nx
    dy = Ly / ny
    dz = Lz / nz

    # ── voxel centroids ───────────────────────────────────────────────────────
    # Shape: (nx, ny, nz)  — x changes slowest (C-order)
    xs = (np.arange(nx) + 0.5) * dx          # (nx,)
    ys = (np.arange(ny) + 0.5) * dy          # (ny,)
    zs = (np.arange(nz) + 0.5) * dz          # (nz,)

    CX, CY, CZ = np.meshgrid(xs, ys, zs, indexing='ij')   # (nx, ny, nz)

    # ── output arrays ─────────────────────────────────────────────────────────
    yarn_index   = np.full((nx, ny, nz), -1, dtype=np.int32)
    yarn_tangent = np.zeros((nx, ny, nz, 3), dtype=np.float64)

    # ── half-axes ─────────────────────────────────────────────────────────────
    a  = Breite / 2.0          # semi-axis along yarn width
    b  = Dicke  / 2.0          # semi-axis along yarn thickness

    # ── assign yarns ──────────────────────────────────────────────────────────
    #
    # Undulation: the centreline Z of each yarn follows a cosine that goes
    # from z_bottom = b + z_layer  to  z_top = 3b + z_layer.
    # Weft and fill are interlocked (opposite phase).
    #
    # For yarn undulating in direction t and crossing yarns in direction s:
    #     z_c(t) = z_layer + b + b * cos(π * (t - t_cross) / Musterbreite)
    # where t_cross is the position of the crossing yarn.
    # At t = t_cross   → z_c = z_layer + 2b   (mid-crossing)
    # Away from cross  → z_c oscillates ±b around z_layer+b
    #
    # We use two crossings per period (Musterbreite spacing):
    # z_c(t) = z_layer + b * (1 + cos(2π * t / Musterbreite))  for one phase
    # z_c(t) = z_layer + b * (1 - cos(2π * t / Musterbreite))  for opposite phase

    for layer in range(n_layer):
        z_layer   = layer * vertOffset_12
        yarn_weft = 2 * layer
        yarn_fill = 2 * layer + 1

        # XY shift for nesting: odd layers offset by (M/2, M/2) so their yarns
        # sit in the valleys of the even layers.
        xy_off = Musterbreite * 0.5 if (nesting_shift and layer % 2 == 1) else 0.0

        y_weft_centres = [Musterbreite * 0.5 + xy_off, Musterbreite * 1.5 + xy_off]
        x_fill_centres = [Musterbreite * 0.5 + xy_off, Musterbreite * 1.5 + xy_off]
        x_cross        = Musterbreite * 0.5 + xy_off
        y_cross        = Musterbreite * 0.5 + xy_off

        # Z clip: restrict voxel assignment to this layer's Z slab so the
        # elliptical cross-section never bleeds into an adjacent layer.
        z_lo = z_layer
        z_hi = z_layer + vertOffset_12 if layer < n_layer - 1 else Lz
        z_clip = (CZ >= z_lo) & (CZ < z_hi)

        # ── weft yarns (run in X) ─────────────────────────────────────────────
        for iy, y_c in enumerate(y_weft_centres):
            sign  = 1 if iy == 0 else -1

            arg    = np.pi * (CX - x_cross) / Musterbreite
            z_c    = z_layer + 2*b + sign * b * np.cos(arg)
            dz_dx  = -sign * b * (np.pi / Musterbreite) * np.sin(arg)

            dY = np.abs(CY - y_c)
            dY = np.minimum(dY, Ly - dY)
            dZ = CZ - z_c

            norm   = np.sqrt(1.0 + dz_dx**2)
            Tx, Tz = 1.0 / norm, dz_dx / norm

            inside = ((dY / a)**2 + (dZ / b)**2 <= 1.0) & z_clip
            yarn_index[inside]      = yarn_weft
            yarn_tangent[inside, 0] = Tx[inside]
            yarn_tangent[inside, 1] = 0.0
            yarn_tangent[inside, 2] = Tz[inside]

        # ── fill yarns (run in Y) ─────────────────────────────────────────────
        for ix, x_c in enumerate(x_fill_centres):
            sign  = -1 if ix == 0 else 1

            arg    = np.pi * (CY - y_cross) / Musterbreite
            z_c    = z_layer + 2*b + sign * b * np.cos(arg)
            dz_dy  = -sign * b * (np.pi / Musterbreite) * np.sin(arg)

            dX = np.abs(CX - x_c)
            dX = np.minimum(dX, Lx - dX)
            dZ = CZ - z_c

            norm   = np.sqrt(1.0 + dz_dy**2)
            Ty, Tz = 1.0 / norm, dz_dy / norm

            inside = ((dX / a)**2 + (dZ / b)**2 <= 1.0) & z_clip
            yarn_index[inside]      = yarn_fill
            yarn_tangent[inside, 0] = 0.0
            yarn_tangent[inside, 1] = Ty[inside]
            yarn_tangent[inside, 2] = Tz[inside]

    phase           = (yarn_index >= 0).astype(np.int32)
    orientation     = yarn_tangent.copy()                    # (nx,ny,nz,3) same as tangent
    volume_fraction = (phase.astype(np.float64) * Vf_yarn)  # (nx,ny,nz)

    return (
        (nx, ny, nz),
        (Lx, Ly, Lz),
        yarn_index.ravel(),
        yarn_tangent.reshape(-1, 3).T.copy(),    # (3, Nv)
        orientation.reshape(-1, 3).T.copy(),     # (3, Nv)
        volume_fraction.ravel(),                 # (Nv,)
        phase.ravel(),
    )


# ── VTU writer ────────────────────────────────────────────────────────────────

def write_vtu(
    path: Path,
    n: tuple,
    L: tuple,
    yarn_index:      np.ndarray,
    yarn_tangent:    np.ndarray,
    orientation:     np.ndarray,
    volume_fraction: np.ndarray,
) -> None:
    """
    Write an unstructured hexahedral VTU matching TexGen's CellData format.
    """
    nx, ny, nz = n
    Lx, Ly, Lz = L
    dx, dy, dz  = Lx/nx, Ly/ny, Lz/nz

    n_pts  = (nx+1)*(ny+1)*(nz+1)
    n_cells = nx*ny*nz

    # ── points ────────────────────────────────────────────────────────────────
    # ordering: i*(ny+1)*(nz+1) + j*(nz+1) + k  (X slowest — C-order)
    xi = np.arange(nx+1)*dx
    yi = np.arange(ny+1)*dy
    zi = np.arange(nz+1)*dz
    PX, PY, PZ = np.meshgrid(xi, yi, zi, indexing='ij')
    pts = np.column_stack([PX.ravel(), PY.ravel(), PZ.ravel()])   # (n_pts, 3)

    # ── connectivity ──────────────────────────────────────────────────────────
    # For voxel (i,j,k): 8 corner node indices
    def pt_idx(i, j, k):
        return i*(ny+1)*(nz+1) + j*(nz+1) + k

    I, J, K = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij')
    I, J, K = I.ravel(), J.ravel(), K.ravel()

    conn = np.column_stack([
        pt_idx(I,   J,   K  ),
        pt_idx(I+1, J,   K  ),
        pt_idx(I+1, J+1, K  ),
        pt_idx(I,   J+1, K  ),
        pt_idx(I,   J,   K+1),
        pt_idx(I+1, J,   K+1),
        pt_idx(I+1, J+1, K+1),
        pt_idx(I,   J+1, K+1),
    ])   # (n_cells, 8)

    offsets = np.arange(8, (n_cells+1)*8, 8)   # (n_cells,)
    types   = np.full(n_cells, 12, dtype=np.int32)  # VTK_HEXAHEDRON

    # ── build XML ─────────────────────────────────────────────────────────────
    root = ET.Element('VTKFile', type='UnstructuredGrid',
                      version='0.1', byte_order='LittleEndian')
    ug   = ET.SubElement(root, 'UnstructuredGrid')
    piece = ET.SubElement(ug, 'Piece',
                          NumberOfPoints=str(n_pts),
                          NumberOfCells=str(n_cells))

    # Points
    pts_node = ET.SubElement(piece, 'Points')
    da = ET.SubElement(pts_node, 'DataArray',
                       type='Float64', NumberOfComponents='3',
                       format='ascii', Name='Points')
    da.text = '\n' + ' '.join(f'{v:.8g}' for v in pts.ravel()) + '\n'

    # Cells
    cells_node = ET.SubElement(piece, 'Cells')
    for name, arr in [('connectivity', conn.ravel()),
                      ('offsets',      offsets),
                      ('types',        types)]:
        da = ET.SubElement(cells_node, 'DataArray',
                           type='Int32', Name=name, format='ascii')
        da.text = '\n' + ' '.join(str(v) for v in arr) + '\n'

    # CellData
    cd = ET.SubElement(piece, 'CellData')

    da = ET.SubElement(cd, 'DataArray', type='Int32',
                       NumberOfComponents='1', format='ascii', Name='YarnIndex')
    da.text = '\n' + ' '.join(str(v) for v in yarn_index) + '\n'

    da = ET.SubElement(cd, 'DataArray', type='Float64',
                       NumberOfComponents='3', format='ascii', Name='YarnTangent')
    da.text = '\n' + ' '.join(f'{v:.8g}' for v in yarn_tangent.T.ravel()) + '\n'

    da = ET.SubElement(cd, 'DataArray', type='Float64',
                       NumberOfComponents='3', format='ascii', Name='Orientation')
    da.text = '\n' + ' '.join(f'{v:.8g}' for v in orientation.T.ravel()) + '\n'

    da = ET.SubElement(cd, 'DataArray', type='Float64',
                       NumberOfComponents='1', format='ascii', Name='VolumeFraction')
    da.text = '\n' + ' '.join(f'{v:.8g}' for v in volume_fraction) + '\n'

    ET.indent(root, space='  ')
    tree = ET.ElementTree(root)
    tree.write(str(path), encoding='unicode', xml_declaration=True)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Generate two-layer biaxial weave VTU')
    parser.add_argument('--out',        type=Path,  default=Path('data/weave_2layer.vtu'))
    parser.add_argument('--Breite',     type=float, default=1.55222,  help='yarn width (mm)')
    parser.add_argument('--Dicke',      type=float, default=0.17084,  help='yarn thickness (mm)')
    parser.add_argument('--Muster',     type=float, default=2.01811,  help='centre-to-centre spacing (mm)')
    parser.add_argument('--resin',      type=float, default=0.08,     help='resin interlayer (mm)')
    parser.add_argument('--voxelsize',  type=float, default=0.02,     help='isotropic voxel size (mm)')
    parser.add_argument('--n_layer',       type=int,   default=2,     help='number of fabric layers')
    parser.add_argument('--Vf_yarn',       type=float, default=0.72,  help='fibre vol. fraction within tow')
    parser.add_argument('--nesting',       type=float, default=None,
                        help='vertical nesting depth mm (default: max when --nesting_shift, else 0)')
    parser.add_argument('--nesting_shift', action='store_true',
                        help='shift odd layers by (M/2,M/2) in XY; defaults nesting to max')
    args = parser.parse_args()

    # auto-compute nesting depth
    if args.nesting is None:
        args.nesting = max_nesting(args.Dicke) if args.nesting_shift else 0.0
        if args.nesting_shift:
            print(f"Auto nesting : {args.nesting:.5f} mm  (= Dicke, max without interference)")

    n, L, yarn_index, yarn_tangent, orientation, volume_fraction, phase = build_weave(
        Breite            = args.Breite,
        Dicke             = args.Dicke,
        Musterbreite      = args.Muster,
        Matrixreiche_Zone = args.resin,
        n_layer           = args.n_layer,
        voxelsize         = args.voxelsize,
        Vf_yarn           = args.Vf_yarn,
        nesting           = args.nesting,
        nesting_shift     = args.nesting_shift,
    )
    nx, ny, nz = n
    Lx, Ly, Lz = L
    Nv = nx * ny * nz

    phi_yarn = float(np.mean(phase))
    print(f"Grid      : {n}   Nv = {Nv:,}")
    print(f"Domain    : Lx={Lx:.4f}  Ly={Ly:.4f}  Lz={Lz:.4f}  mm")
    print(f"Voxelsize : {Lx/nx:.5f} × {Ly/ny:.5f} × {Lz/nz:.5f}  mm")
    print(f"Yarn Vf   : {phi_yarn:.3f}  (tow packing {args.Vf_yarn:.2f}  →  fibre Vf = {phi_yarn*args.Vf_yarn:.3f})")
    for yi in range(np.max(yarn_index)+1):
        print(f"  yarn {yi}: {int(np.sum(yarn_index == yi)):,} voxels")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # ── VTU (TexGen-compatible) ────────────────────────────────────────────────
    write_vtu(args.out, n, L, yarn_index, yarn_tangent, orientation, volume_fraction)
    print(f"Written → {args.out}")

    # ── XDMF/HDF5 (FFTjax standard) ───────────────────────────────────────────
    import h5py
    dx_grid = tuple(Li / ni for Li, ni in zip(L, n))
    xdmf_stem = str(args.out.with_suffix(""))
    with IncrementalWriter(xdmf_stem, grid_shape=n, grid_spacing=dx_grid) as w:
        w.write_increment(0, {
            "yarn_index":      yarn_index.reshape(n).astype(np.float32),
            "phase":           phase.reshape(n).astype(np.float32),
            "yarn_tangent":    yarn_tangent.T.reshape(*n, 3).astype(np.float64),
            "orientation":     orientation.T.reshape(*n, 3).astype(np.float64),
            "volume_fraction": volume_fraction.reshape(n).astype(np.float64),
        }, time=0.0)
    with h5py.File(xdmf_stem + ".h5", "a") as f:
        f.attrs["n"] = np.array(n, dtype=int)
        f.attrs["L"] = np.array(L, dtype=float)
    print(f"Written → {xdmf_stem}.h5 / .xdmf")
    print("Open the .xdmf in ParaView with the 'Xdmf3ReaderT' reader.")


if __name__ == '__main__':
    main()
