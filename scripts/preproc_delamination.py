"""
Preprocessing for mode-I delamination PFF simulation.

Reads a TexGen voxel VTU, appends matrix padding layers in Z to create
an interlaminar resin zone at the periodic boundary, and inserts a
rectangular starter crack centred on the XY mid-axes inside that zone.

Under periodic BCs the padding at z = nz … nz+n_pad−1 is adjacent to
z = 0, forming one continuous interlaminar region.  The starter crack
(d = 1) is placed in the centre of this padding, centred in X and Y.

Output: .npz file consumed by the PFF solver.

Contents of the .npz
--------------------
  n              (3,)   int    padded grid shape (nx, ny, nz_pad)
  L              (3,)   float  padded domain size (mm)
  phase          (Nv,)  int    0=matrix 1=yarn
  orientations   (3,Nv) float  YarnTangent per voxel
  yarn_index     (Nv,)  int    raw YarnIndex (-1=matrix, >=0=yarn)
  d_init         (Nv,)  float  initial damage field (1 inside crack)
  H_init         (Nv,)  float  initial history field (large inside crack)

Usage
-----
    python scripts/preproc_delamination.py data/myweave.vtu
    python scripts/preproc_delamination.py data/myweave.vtu \\
        --n_pad 6 --crack_x 0.4 --crack_y 0.8 --out output/
"""

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from post.io import IncrementalWriter


# ── VTU reader (self-contained copy so script is standalone) ──────────────────

def _read_vtu(path: str):
    tree  = ET.parse(path)
    piece = tree.getroot().find('UnstructuredGrid/Piece')
    assert piece is not None, f"No <UnstructuredGrid/Piece> in {path}"

    pts_node  = piece.find('Points/DataArray')
    conn_node = piece.find('Cells/DataArray[@Name="connectivity"]')
    assert pts_node  is not None, "No Points/DataArray in VTU"
    assert conn_node is not None, "No connectivity DataArray in VTU"

    pts  = np.fromstring(pts_node.text  or "", sep=' ').reshape(-1, 3)
    conn = np.fromstring(conn_node.text or "", sep=' ').astype(int).reshape(-1, 8)

    centroids = pts[conn].mean(axis=1)

    xu = np.unique(np.round(centroids[:, 0], 8))
    yu = np.unique(np.round(centroids[:, 1], 8))
    zu = np.unique(np.round(centroids[:, 2], 8))
    nx, ny, nz = len(xu), len(yu), len(zu)
    dx = float(xu[1] - xu[0])
    dy = float(yu[1] - yu[0])
    dz = float(zu[1] - zu[0])

    ix_arr = np.round((centroids[:, 0] - xu[0]) / dx).astype(int)
    iy_arr = np.round((centroids[:, 1] - yu[0]) / dy).astype(int)
    iz_arr = np.round((centroids[:, 2] - zu[0]) / dz).astype(int)
    fft_idx = ix_arr * (ny * nz) + iy_arr * nz + iz_arr

    cd = piece.find('CellData')
    assert cd is not None, f"No <CellData> in {path}"
    arrays = {a.attrib['Name']: a for a in cd.findall('DataArray')}

    yarn_vtk = np.fromstring(arrays['YarnIndex'].text   or "", sep=' ').astype(int)
    tang_vtk = np.fromstring(arrays['YarnTangent'].text or "", sep=' ').reshape(-1, 3)

    Nv           = nx * ny * nz
    phase_flat   = np.empty(Nv, dtype=int)
    tangent_flat = np.zeros((Nv, 3))
    phase_flat[fft_idx]   = yarn_vtk
    tangent_flat[fft_idx] = tang_vtk

    phase      = np.where(phase_flat >= 0, 1, 0)
    yarn_index = phase_flat.copy()
    orient     = tangent_flat.T.copy()   # (3, Nv)

    return (nx, ny, nz), (dx, dy, dz), phase, orient, yarn_index


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess TexGen VTU for mode-I delamination PFF"
    )
    parser.add_argument("vtu", type=Path,
                        help="TexGen voxel VTU file")
    parser.add_argument("--n_pad", type=int, default=4,
                        help="Matrix voxels to append in Z (interlaminar zone, default 4)")
    parser.add_argument("--crack_y", type=float, default=1.0,
                        help="Starter-crack width as fraction of Ly, centred (default 1.0)")

    parser.add_argument("--out", type=Path, default=Path("output/preprocessed"),
                        help="Output directory (default: output/preprocessed/)")
    args = parser.parse_args()

    # ── read VTU ─────────────────────────────────────────────────────────────
    print(f"Reading {args.vtu} …")
    (nx, ny, nz), (dx, dy, dz), phase, orient, yarn_index = _read_vtu(str(args.vtu))
    print(f"  fabric grid : ({nx}, {ny}, {nz})")

    # ── pad in Y with pure matrix (interlaminar resin zones) ─────────────────
    # Padding on both Y sides creates two resin-rich zones that form one
    # continuous interlaminar layer under periodic BC (Y=0 ≡ Y=Ly).
    # The crack is initialised entirely within this padding — never in the fabric.
    #
    #  Y:  [0 … n_pad) | [n_pad … n_pad+ny) | [n_pad+ny … n_pad+ny+n_pad)
    #      [ resin pad ] [      fabric       ] [         resin pad          ]
    #           ↑ d=1                                       ↑ d=1
    #           crack front advances → +Y toward fabric centre

    n_pad  = args.n_pad
    ny_new = ny + 2 * n_pad
    Nv_new = nx * ny_new * nz

    phase_3d  = phase.reshape(nx, ny, nz)
    orient_3d = orient.reshape(3, nx, ny, nz)
    yarn_3d   = yarn_index.reshape(nx, ny, nz)

    pad_shape  = (nx, n_pad, nz)
    phase_pad  = np.zeros(pad_shape, dtype=int)
    orient_pad = np.zeros((3, *pad_shape))
    yarn_pad   = np.full(pad_shape, -1, dtype=int)

    # concatenate along Y axis (axis=1)
    phase_new  = np.concatenate([phase_pad,  phase_3d,  phase_pad],       axis=1).ravel()
    yarn_new   = np.concatenate([yarn_pad,   yarn_3d,   yarn_pad],        axis=1).ravel()
    orient_new = np.concatenate([orient_pad, orient_3d, orient_pad],      axis=2).reshape(3, -1)

    Lx    = nx * dx
    Ly    = ny_new * dy
    Lz    = nz * dz
    n_new = (nx, ny_new, nz)
    L_new = (Lx, Ly, Lz)
    print(f"  padded grid : {n_new}   L = ({Lx:.4g}, {Ly:.4g}, {Lz:.4g}) mm")

    # ── starter crack — full X length, mid-Z plane, both Y padding zones ─────
    # Crack lives at z = nz//2 (mid-thickness of the fabric/padding stack),
    # spanning the entire X length and both Y padding strips.
    # Under PBC the two strips merge into one interlaminar crack at Y=0/Ly.
    z_crack = nz // 2

    crack_mask_3d = np.zeros((nx, ny_new, nz), dtype=bool)
    crack_mask_3d[:, :n_pad,              z_crack] = True   # low-Y pad
    crack_mask_3d[:, n_pad + ny:,         z_crack] = True   # high-Y pad
    crack_flat = crack_mask_3d.ravel()

    # crack voxels → phase 2 (void/crack), near-zero stiffness in PFF script
    phase_new[crack_flat] = 2
    yarn_new[crack_flat]  = -2   # distinct from matrix (-1) and yarn (>=0)

    # zero damage / history — PFF evolves from the crack tip, not inside it
    d_init = np.zeros(Nv_new)
    H_init = np.zeros(Nv_new)

    n_crack = int(crack_flat.sum())
    print(f"  crack plane : z={z_crack}/{nz}  x=[0,{nx})  "
          f"y=[0,{n_pad}) + [{n_pad+ny},{ny_new})  "
          f"({n_crack} voxels,  frac={n_crack/Nv_new:.3f})")

    # ── save XDMF/HDF5 ────────────────────────────────────────────────────────
    args.out.mkdir(parents=True, exist_ok=True)
    stem      = args.vtu.stem + "_delam_preproc"
    xdmf_stem = str(args.out / stem)
    dx_new    = tuple(Li / ni for Li, ni in zip(L_new, n_new))

    import h5py
    with IncrementalWriter(xdmf_stem, grid_shape=n_new, grid_spacing=dx_new) as w:
        w.write_increment(0, {
            "phase":       phase_new.reshape(n_new).astype(np.float32),
            "yarn_index":  yarn_new.reshape(n_new).astype(np.float32),
            "orientation": orient_new.T.reshape(*n_new, 3).astype(np.float64),
            "crack_void":  (phase_new == 2).reshape(n_new).astype(np.float32),
        }, time=0.0)

    # store grid metadata as HDF5 root attributes so the PFF solver can load them
    with h5py.File(xdmf_stem + ".h5", "a") as f:
        f.attrs["n"] = np.array(n_new, dtype=int)
        f.attrs["L"] = np.array(L_new, dtype=float)

    print(f"Saved → {xdmf_stem}.h5")
    print(f"        {xdmf_stem}.xdmf")
    print("Open the .xdmf in ParaView with the 'Xdmf3ReaderT' reader.")


if __name__ == "__main__":
    main()
