"""
CLI for the biaxial plain-weave geometry generator.

Geometry logic lives in src/generation/weave.py.
Reading a TexGen-produced VTU back in lives in src/utils/io_texgen.py.

Usage
-----
    python scripts/preprocessing/generate_weave_vtu.py --material 200
    python scripts/preprocessing/generate_weave_vtu.py --material 200 --nesting_shift
    python scripts/preprocessing/generate_weave_vtu.py --list
    python scripts/preprocessing/generate_weave_vtu.py --Breite 1.55 --Dicke 0.17
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from generation.weave import build_weave, load_db, list_materials, max_nesting
from utils.io.xdmf_writer import IncrementalWriter
def main():
    parser = argparse.ArgumentParser(description='Generate two-layer biaxial weave VTU')
    parser.add_argument('--material',      type=str,   default=None,
                        help='fabric ID from data/fabric_db.json (160 | 200 | 245)')
    parser.add_argument('--list',          action='store_true',
                        help='list available material IDs and exit')
    parser.add_argument('--out',           type=Path,  default=None,
                        help='output stem (default: data/<material>  →  .h5/.xdmf)')
    parser.add_argument('--Breite',        type=float, default=None)
    parser.add_argument('--Dicke',         type=float, default=None)
    parser.add_argument('--Muster',        type=float, default=None)
    parser.add_argument('--resin',         type=float, default=None)
    parser.add_argument('--voxelsize',     type=float, default=None)
    parser.add_argument('--n_layer',       type=int,   default=2)
    parser.add_argument('--Vf_yarn',       type=float, default=0.72)
    parser.add_argument('--nesting',       type=float, default=None)
    parser.add_argument('--nesting_shift', action='store_true',
                        help='shift odd layers by (M/2,M/2) in XY; auto-sets nesting to DB max')
    parser.add_argument('--epsilon', type=float, default=None,
                        help='SDF interface half-width mm (default: 1.5 × voxelsize)')
    args = parser.parse_args()

    if args.list:
        list_materials()
        return

    # ── resolve parameters: CLI flag > DB > built-in default ─────────────────
    db_entry: dict = {}
    if args.material is not None:
        db = load_db()
        if args.material not in db:
            raise SystemExit(f"Unknown material '{args.material}'. "
                             f"Available: {[k for k in db if not k.startswith('_')]}")
        db_entry = db[args.material]
        print(f"Material : {args.material}  —  {db_entry['description']}")

    def _get(flag_val, db_key, default):
        if flag_val is not None:
            return flag_val
        if db_key and db_key in db_entry:
            return db_entry[db_key]
        return default

    Breite    = _get(args.Breite,    'Breite',            1.55222)
    Dicke     = _get(args.Dicke,     'Dicke',             0.17084)
    Muster    = _get(args.Muster,    'Musterbreite',      2.01811)
    resin     = _get(args.resin,     'Matrixreiche_Zone', 0.02)
    voxelsize = _get(args.voxelsize, None,                load_db().get('_voxelsize', 0.02))
    nesting_db = db_entry.get('nesting_max', Dicke)
    out_stem = args.out.with_suffix("") if args.out else Path(
        f"data/{args.material}" if args.material else "data/weave"
    )

    nesting = args.nesting
    if nesting is None:
        nesting = nesting_db if args.nesting_shift else 0.0
        if args.nesting_shift:
            print(f"Auto nesting : {nesting:.5f} mm  (DB max for {args.material or 'custom'})")

    # ── build geometry ────────────────────────────────────────────────────────
    n, L, yarn_index, yarn_tangent, orientation, volume_fraction, phase = build_weave(
        Breite            = Breite,
        Dicke             = Dicke,
        Musterbreite      = Muster,
        Matrixreiche_Zone = resin,
        n_layer           = args.n_layer,
        voxelsize         = voxelsize,
        Vf_yarn           = args.Vf_yarn,
        nesting           = nesting,
        nesting_shift     = args.nesting_shift,
        epsilon           = args.epsilon,
    )
    nx, ny, nz = n
    Lx, Ly, Lz = L
    Nv = nx * ny * nz

    phi_yarn = float(np.mean(phase))
    print(f"Grid      : {n}   Nv = {Nv:,}")
    print(f"Domain    : Lx={Lx:.4f}  Ly={Ly:.4f}  Lz={Lz:.4f}  mm")
    print(f"Voxelsize : {Lx/nx:.5f} × {Ly/ny:.5f} × {Lz/nz:.5f}  mm")
    print(f"Yarn Vf   : {phi_yarn:.3f}  (tow packing {args.Vf_yarn:.2f}"
          f"  →  fibre Vf = {phi_yarn*args.Vf_yarn:.3f})")
    for yi in range(int(np.max(yarn_index)) + 1):
        print(f"  yarn {yi}: {int(np.sum(yarn_index == yi)):,} voxels")

    out_stem.parent.mkdir(parents=True, exist_ok=True)

    # ── XDMF/HDF5 (only output format) ───────────────────────────────────────
    import h5py
    xdmf_stem = str(out_stem)
    with IncrementalWriter(xdmf_stem, grid_shape=n, grid_length=L) as w:
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
