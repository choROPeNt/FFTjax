"""
Square-packed (quadratic) circular-fibre RVE generator — YAML-driven.

Geometry logic lives in src/generation/rve.py (make_square_composite_rve):
a deterministic 2-fibre square-cell unit cell, not a randomly-perturbed
packing -- no K/seed here, unlike generate_rve.py's random RVE. Reaches
any volume fraction up to the square-packing limit (~0.7854); use
generate_rve.py instead for a randomly-packed RVE or for phi beyond that.

String values in the YAML support {variable} interpolation:
  output:  "output/generation"
  jobname: "rve_quad_phi-{geometry.phi}"

Usage
-----
    python scripts/generation/generate_rve_quad.py configs/generation/rve_quad.yaml

Output
------
    <output>/<jobname>.h5
    <output>/<jobname>.xdmf
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, "src")

from generation.rve import make_square_composite_rve
from utils.io.xdmf_writer import IncrementalWriter
from utils.config   import load_config


def main():
    parser = argparse.ArgumentParser(
        description="Generate a square-packed circular-fibre RVE (XDMF/HDF5)"
    )
    parser.add_argument("config", type=Path, help="YAML configuration file")
    args = parser.parse_args()

    cfg  = load_config(args.config)
    gcfg = cfg["geometry"]
    print(f"Config  : {args.config}")

    # ── build geometry ────────────────────────────────────────────────────────
    phase_np, n, L, phi_act = make_square_composite_rve(
        phi     = float(gcfg["phi"]),
        r_fiber = float(gcfg["r_fib"]),
        dx      = float(gcfg["vox"]),
        N_min   = int(gcfg.get("N_min", 32)),
        nz      = int(gcfg.get("nz", 1)),
    )
    Nv = int(np.prod(n))

    d = np.array(gcfg.get("fibre_dir", [0.0, 0.0, 1.0]), dtype=float)
    d /= np.linalg.norm(d)

    fibre_mask   = phase_np.ravel() == 1
    orientations = np.zeros((Nv, 3))
    orientations[fibre_mask] = d

    # fibre packing within the tow (Vf_yarn)
    Vf_yarn = float(gcfg.get("Vf_yarn", 1.0))
    vf = fibre_mask.astype(float) * Vf_yarn

    print(f"phi_target = {gcfg['phi']:.3f}   phi_actual = {phi_act:.4f}")
    print(f"Grid    : {n}   Nv = {Nv:,}")
    print(f"Domain  : {tuple(f'{v:.4g}' for v in L)}")

    # ── write XDMF/HDF5 ───────────────────────────────────────────────────────
    output  = cfg["output"]
    jobname = cfg["jobname"]
    stem    = f"{output}/{jobname}"

    Path(output).mkdir(parents=True, exist_ok=True)

    with IncrementalWriter(stem, grid_shape=n, grid_length=L) as w:
        w.write_increment(0, {
            "phase":            phase_np.ravel().reshape(n).astype(np.float32),
            "orientation":      orientations.reshape(*n, 3).astype(np.float64),
            "volume_fraction":  vf.reshape(n).astype(np.float64),
        }, time=0.0)

    with h5py.File(stem + ".h5", "a") as f:
        f.attrs["n"] = np.array(n, dtype=int)
        f.attrs["L"] = np.array(L, dtype=float)

    print(f"Written → {stem}.h5 / .xdmf")
    print("Open the .xdmf in ParaView with the 'Xdmf3ReaderT' reader.")


if __name__ == "__main__":
    main()
