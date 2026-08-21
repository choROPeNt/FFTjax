"""
CLI for the biaxial plain-weave geometry generator -- YAML-driven.

Geometry logic lives in src/generation/weave.py.
Reading a TexGen-produced VTU back in lives in src/utils/io/reader.py.

The config is self-contained: all geometry parameters (width, thickness,
pattern_width, resin_rich_zone, voxelsize, ...) are given directly in the
YAML, not looked up from data/fabric_db.json at run time -- editing the DB
later won't silently change what an existing config builds. data/fabric_db.json
is still useful as a starting point / reference table (see --list), but once
copied into a config it's just a plain float, no live link.

String values in the YAML support {variable} interpolation:
  output:  "output/generation"
  jobname: "{weave.jobname}"

Usage
-----
    python scripts/generation/generate_weave.py configs/generation/weave_200.yaml
    python scripts/generation/generate_weave.py --list

Output
------
    <output>/<jobname>.h5
    <output>/<jobname>.xdmf
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from generation.weave import build_weave, voxelize
from utils.config import load_config
from utils.io.xdmf_writer import IncrementalWriter


def main():
    parser = argparse.ArgumentParser(
        description='Generate biaxial weave geometry from a YAML config (XDMF/HDF5)'
    )
    parser.add_argument('-c','--config', type=Path, nargs='?',
                        help='YAML configuration file (see configs/generation/weave_200.yaml)')
    args = parser.parse_args()


    if args.config is None:
        parser.error("a YAML config is required (or pass --list) -- "
                     "see configs/generation/weave_200.yaml for an example")

    cfg  = load_config(args.config)
    wcfg = cfg["weave"]

    required = ['width', 'thickness', 'pattern_width', 'resin_rich_zone', 'voxelsize']
    missing  = [k for k in required if k not in wcfg]
    if missing:
        raise SystemExit(f"weave config is missing required key(s): {missing} -- "
                          "this script no longer reads data/fabric_db.json at run time, "
                          "copy the values in directly (see configs/generation/weave_200.yaml)")

    print(f"Config      : {args.config}")
    if 'description' in wcfg:
        print(f"Description : {wcfg['description']}")

    width         = wcfg['width']
    thickness     = wcfg['thickness']
    pattern_width = wcfg['pattern_width']
    resin         = wcfg['resin_rich_zone']
    voxelsize     = wcfg['voxelsize']
    n_layers      = wcfg.get('n_layers', 2)
    Vf_yarn       = wcfg.get('Vf_yarn', 0.72)
    max_nesting   = wcfg.get('max_nesting', True)

    # ── resolve geometry (layer placement/nesting -- independent of voxelsize) ─
    geometry = build_weave(
        width           = width,
        thickness       = thickness,
        pattern_width   = pattern_width,
        resin_rich_zone = resin,
        n_layer         = n_layers,
        Vf_yarn         = Vf_yarn,
        max_nesting     = max_nesting,
    )
    print(f"Geometry  : Lx={geometry.Lx:.4f}  Ly={geometry.Ly:.4f}  Lz={geometry.Lz:.4f}  mm")

    # ── voxelize (sample the resolved geometry onto a grid) ────────────────────
    n, L, yarn_index, yarn_tangent, orientation, volume_fraction, phase = voxelize(
        geometry, voxelsize,
    )
    nx, ny, nz = n
    Lx, Ly, Lz = L
    Nv = nx * ny * nz

    phi_yarn = float(np.mean(phase))
    print(f"Grid      : {n}   Nv = {Nv:,}")
    print(f"Domain    : Lx={Lx:.4f}  Ly={Ly:.4f}  Lz={Lz:.4f}  mm")
    print(f"Voxelsize : {Lx/nx:.5f} × {Ly/ny:.5f} × {Lz/nz:.5f}  mm")
    print(f"Yarn Vf   : {phi_yarn:.3f}  (tow packing {Vf_yarn:.2f}"
          f"  →  fibre Vf = {phi_yarn*Vf_yarn:.3f})")
    for yi in range(int(np.max(yarn_index)) + 1):
        print(f"  yarn {yi}: {int(np.sum(yarn_index == yi)):,} voxels")

    # ── output ────────────────────────────────────────────────────────────────
    output  = cfg["output"]
    jobname = cfg["jobname"]
    if max_nesting:
        jobname = f"{jobname}_nesting"
    stem    = f"{output}/{jobname}"
    Path(output).mkdir(parents=True, exist_ok=True)

    # ── XDMF/HDF5 (only output format) ───────────────────────────────────────
    import h5py
    with IncrementalWriter(stem, grid_shape=n, grid_length=L) as w:
        w.write_increment(0, {
            "yarn_index":      yarn_index.reshape(n).astype(np.float32),
            "phase":           phase.reshape(n).astype(np.float32),
            "yarn_tangent":    yarn_tangent.T.reshape(*n, 3).astype(np.float64),
            "orientation":     orientation.T.reshape(*n, 3).astype(np.float64),
            "volume_fraction": volume_fraction.reshape(n).astype(np.float64),
        }, time=0.0)
    with h5py.File(stem + ".h5", "a") as f:
        f.attrs["n"] = np.array(n, dtype=int)
        f.attrs["L"] = np.array(L, dtype=float)
    print(f"Written → {stem}.h5 / .xdmf")
    print("Open the .xdmf in ParaView with the 'Xdmf3ReaderT' reader.")


if __name__ == '__main__':
    main()
