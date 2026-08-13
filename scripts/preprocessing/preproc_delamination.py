"""
Preprocessing for mode-I delamination PFF simulation.

Reads a voxel XDMF/HDF5 file (written by IncrementalWriter, e.g. from
texgen_export), appends matrix padding layers in Z to create an interlaminar
resin zone at the periodic boundary, and inserts a rectangular starter crack
centred on the XY mid-axes inside that zone.

Under periodic BCs the padding at z = nz … nz+n_pad−1 is adjacent to
z = 0, forming one continuous interlaminar region.  The starter crack
(d = 1) is placed in the centre of this padding, centred in X and Y.

Expected input XDMF fields
--------------------------
  phase        (nx, ny, nz)    int-valued float  0=matrix, 1=yarn
  yarn_index   (nx, ny, nz)    int-valued float  −1=matrix, >=0=yarn
  orientation  (nx, ny, nz, 3) float             YarnTangent per voxel

Output XDMF fields
------------------
  phase        (nx, ny, nz_pad)    padded phase (2=crack void)
  yarn_index   (nx, ny, nz_pad)    padded yarn index (−2=crack void)
  orientation  (nx, ny, nz_pad, 3) padded tangents
  crack_void   (nx, ny, nz_pad)    binary crack mask

Usage
-----
    python scripts/preproc_delamination.py data/myweave.xdmf
    python scripts/preproc_delamination.py data/myweave.h5 \\
        --n_pad 6 --crack_y 0.8 --out output/
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from utils.io.reader import read_xdmf
from utils.io.xdmf_writer import IncrementalWriter
def _read_input(path: str):
    """Read voxel geometry from an XDMF/HDF5 file, return 3-D arrays."""
    n, L, fields = read_xdmf(path)
    nx, ny, nz = n
    Lx, Ly, Lz = L
    dx, dy, dz = Lx / nx, Ly / ny, Lz / nz

    phase_3d  = fields["phase"].astype(int)                   # (nx, ny, nz)
    yarn_3d   = fields["yarn_index"].astype(int)               # (nx, ny, nz)
    orient_3d = fields["orientation"].transpose(3, 0, 1, 2)   # (3, nx, ny, nz)
    vf_3d     = fields["volume_fraction"].astype(np.float64)   # (nx, ny, nz)

    return (nx, ny, nz), (dx, dy, dz), phase_3d, orient_3d, yarn_3d, vf_3d


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess voxel XDMF for mode-I delamination PFF"
    )
    parser.add_argument("xdmf", type=Path,
                        help="Voxel geometry XDMF or HDF5 file (must contain phase, yarn_index, orientation fields)")
    parser.add_argument("--n_pad", type=int, default=4,
                        help="Matrix voxels added to both Y sides — resin zone (default 4)")
    parser.add_argument("--n_pad_z", type=int, default=0,
                        help="Matrix voxels added to both Z sides — prevents Z-boundary cracking (default 8)")
    parser.add_argument("--crack_ext", type=int, default=0,
                        help="Extra voxels the void crack extends into the fabric beyond the Y-pad (default 0)")
    parser.add_argument("--crack_y", type=float, default=1.0,
                        help="Starter-crack width as fraction of Ly, centred (default 1.0)")

    parser.add_argument("--out", type=Path, default=Path("output/preprocessed"),
                        help="Output directory (default: output/preprocessed/)")
    args = parser.parse_args()

    # ── read XDMF ────────────────────────────────────────────────────────────
    print(f"Reading {args.xdmf} …")
    (nx, ny, nz), (dx, dy, dz), phase_3d, orient_3d, yarn_3d, vf_3d = _read_input(str(args.xdmf))
    print(f"  fabric grid : ({nx}, {ny}, {nz})")

    # ── pad Z with pure matrix (isolates Z-periodic fabric images) ───────────
    # Prevents stress concentrations at yarn-matrix interfaces from cracking
    # under PBC (z=0 ≡ z=Lz without this padding).
    #
    #  Z:  [0 … n_pz) | [n_pz … n_pz+nz) | [n_pz+nz … nz_new)
    #      [ resin pad ] [     fabric      ] [    resin pad     ]

    n_pad   = args.n_pad
    n_pad_z = args.n_pad_z

    zpd       = (nx, ny, n_pad_z)
    z_phase   = np.zeros(zpd, dtype=int)
    z_orient  = np.zeros((3, *zpd))
    z_yarn    = np.full(zpd, -1, dtype=int)
    z_vf      = np.zeros(zpd)

    phase_3d  = np.concatenate([z_phase,  phase_3d,  z_phase],  axis=2)
    yarn_3d   = np.concatenate([z_yarn,   yarn_3d,   z_yarn],   axis=2)
    orient_3d = np.concatenate([z_orient, orient_3d, z_orient], axis=3)
    vf_3d     = np.concatenate([z_vf,     vf_3d,     z_vf],     axis=2)
    nz_new    = nz + 2 * n_pad_z

    # ── pad Y with pure matrix (interlaminar resin zones — crack lives here) ──
    # Under periodic BC the two Y-edge strips merge into one interlaminar
    # crack zone at Y=0/Ly; crack front advances toward Y-centre.
    #
    #  Y:  [0 … n_pad) | [n_pad … n_pad+ny) | [n_pad+ny … ny_new)
    #      [ resin pad ] [      fabric       ] [    resin pad      ]

    ny_new = ny + 2 * n_pad
    Nv_new = nx * ny_new * nz_new

    ypd       = (nx, n_pad, nz_new)
    y_phase   = np.zeros(ypd, dtype=int)
    y_orient  = np.zeros((3, *ypd))
    y_yarn    = np.full(ypd, -1, dtype=int)
    y_vf      = np.zeros(ypd)

    phase_new  = np.concatenate([y_phase,  phase_3d,  y_phase],  axis=1).ravel()
    yarn_new   = np.concatenate([y_yarn,   yarn_3d,   y_yarn],   axis=1).ravel()
    orient_new = np.concatenate([y_orient, orient_3d, y_orient], axis=2).reshape(3, -1)
    vf_new     = np.concatenate([y_vf,     vf_3d,     y_vf],     axis=1).ravel()

    Lx    = nx    * dx
    Ly    = ny_new * dy
    Lz    = nz_new * dz
    n_new = (nx, ny_new, nz_new)
    L_new = (Lx, Ly, Lz)
    print(f"  padded grid : {n_new}   L = ({Lx:.4g}, {Ly:.4g}, {Lz:.4g}) mm")

    # ── starter crack — full X, both Y pads + extension, mid-Z ──────────────
    # The crack occupies the Y-resin-pad AND an additional crack_ext voxels
    # into the fabric matrix, giving a longer crack arm and higher K_I at tip.
    #
    #  Y:  [0 … n_pad+crack_ext) | intact fabric | (ny_new-n_pad-crack_ext … ny_new)
    #      [  void/crack region  ]               [    void/crack region              ]
    #                             ↑ crack tip                         ↑ crack tip

    z_crack   = nz_new // 2   # mid-thickness of padded domain
    crack_ext = args.crack_ext
    arm       = n_pad + crack_ext   # total crack arm length in Y voxels

    crack_mask_3d = np.zeros((nx, ny_new, nz_new), dtype=bool)
    crack_mask_3d[:, :arm,            z_crack] = True   # low-Y arm
    crack_mask_3d[:, ny_new - arm:,   z_crack] = True   # high-Y arm
    crack_flat = crack_mask_3d.ravel()

    # crack voxels → phase 2 (void/crack), near-zero stiffness in PFF script
    phase_new[crack_flat] = 2
    yarn_new[crack_flat]  = -2   # distinct from matrix (-1) and yarn (>=0)

    # zero damage / history — PFF evolves from the crack tip naturally
    d_init = np.zeros(Nv_new)
    H_init = np.zeros(Nv_new)

    n_crack = int(crack_flat.sum())
    print(f"  crack plane : z={z_crack}/{nz_new}  x=[0,{nx})  "
          f"y=[0,{arm}) + [{ny_new-arm},{ny_new})  arm={arm} voxels  "
          f"({n_crack} total,  frac={n_crack/Nv_new:.3f})")

    # ── save XDMF/HDF5 ────────────────────────────────────────────────────────
    args.out.mkdir(parents=True, exist_ok=True)
    stem      = args.xdmf.stem + "_delam_preproc"
    xdmf_stem = str(args.out / stem)

    import h5py
    with IncrementalWriter(xdmf_stem, grid_shape=n_new, grid_length=L_new) as w:
        w.write_increment(0, {
            "phase":       phase_new.reshape(n_new).astype(np.float32),
            "yarn_index":  yarn_new.reshape(n_new).astype(np.float32),
            "orientation": orient_new.T.reshape(*n_new, 3).astype(np.float64),
            "crack_void":        (phase_new == 2).reshape(n_new).astype(np.float32),
            "volume_fraction":   vf_new.reshape(n_new).astype(np.float64),
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
