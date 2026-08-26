"""
Randomly-packed circular-fibre RVE generator — YAML-driven.

Geometry logic lives in src/generation/rve.py (make_random_composite_rve,
Catalanotti 2016) -- reaches any volume fraction up to the hexagonal packing
limit (~0.9069), unlike random sequential addition (RSA) which stalls well
below realistic in-tow fibre volume fractions.

String values in the YAML support {variable} interpolation:
  output:  "output/preprocessed"
  jobname: "rve_random_phi-{geometry.phi}_seed-{geometry.seed}"

Usage
-----
    python scripts/generation/generate_rve.py configs/generation/rve_random.yaml

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

from generation.rve import make_random_composite_rve
from utils.io.xdmf_writer import IncrementalWriter
from utils.config   import load_config


def main():
    parser = argparse.ArgumentParser(
        description="Generate a randomly-packed circular-fibre RVE (XDMF/HDF5)"
    )
    parser.add_argument("config", type=Path, help="YAML configuration file")
    args = parser.parse_args()

    cfg  = load_config(args.config)
    gcfg = cfg["geometry"]
    print(f"Config  : {args.config}")

    # ── build geometry ────────────────────────────────────────────────────────
    interphase_thickness = gcfg.get("interphase_thickness")
    size_in_r = gcfg.get("size_in_r")
    phase_np, n, L, phi_act, centres = make_random_composite_rve(
        phi     = float(gcfg["phi"]),
        r_fiber = float(gcfg["r_fib"]),
        dx      = float(gcfg["vox"]),
        size_in_r = float(size_in_r) if size_in_r is not None else None,
        nz      = int(gcfg.get("nz", 1)),
        K       = int(gcfg.get("K", 15)),
        seed    = gcfg.get("seed"),
        interphase_thickness = float(interphase_thickness) if interphase_thickness is not None else None,
    )
    Nv = int(np.prod(n))

    # phase index: 0=matrix, 1=fibre, 2=interphase (only present if
    # interphase_thickness was set) -- the simulation config's materials:
    # list needs a matching 3rd entry when it is.
    d = np.array(gcfg.get("fibre_dir", [0.0, 0.0, 1.0]), dtype=float)
    d /= np.linalg.norm(d)

    fibre_mask   = phase_np.ravel() == 1
    orientations = np.zeros((Nv, 3))
    orientations[fibre_mask] = d

    # fibre packing within the tow (Vf_yarn) -- interphase voxels get 0,
    # not the interphase's own phase index (only fibre voxels are packed)
    Vf_yarn = float(gcfg.get("Vf_yarn", 1.0))
    vf = fibre_mask.astype(float) * Vf_yarn

    print(f"phi_target = {gcfg['phi']:.3f}   phi_actual = {phi_act:.4f}   Np = {len(centres)}")
    if interphase_thickness is not None:
        interphase_frac = float((phase_np.ravel() == 2).mean())
        print(f"interphase_thickness = {interphase_thickness:.4g}   interphase volume fraction = {interphase_frac:.4f}")
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
