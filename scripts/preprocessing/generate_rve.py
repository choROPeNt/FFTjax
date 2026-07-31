"""
Generic geometry preprocessor — YAML-driven.

Supports two modes selected by ``geometry.type`` in the config:

  square_rve  — generate a square-packed circular-fibre RVE from parameters.
                Uniform phi; orientation fixed to ``geometry.fibre_dir``.

  textile     — load an existing VTU / XDMF (from generate_weave_vtu.py or
                TexGen) and re-export with proper HDF5 attributes.
                Spatially distributed phi (VolumeFraction field) is preserved.

String values in the YAML support {variable} interpolation:
  output:  "output/preprocessed/{geometry.type}"
  jobname: "{geometry.source.stem}"   # Path.stem of the source file

Usage
-----
    python scripts/preprocessing/generate_rve.py configs/preproc_rve.yaml
    python scripts/preprocessing/generate_rve.py configs/preproc_textile.yaml

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

from post.io        import IncrementalWriter
from utils.config   import load_config
from utils.io_read  import read_simulation_input


# ── mode handlers ─────────────────────────────────────────────────────────────

def _square_rve(gcfg: dict):
    """Generate a square-packed fibre RVE from geometry parameters."""
    from generation.rve import make_square_composite_rve

    phase_np, _, n, L, phi_act = make_square_composite_rve(
        phi     = float(gcfg["phi"]),
        r_fiber = float(gcfg["r_fib"]),
        spacing = float(gcfg["vox"]),
        nz      = int(gcfg.get("nz", 1)),
    )
    Nv = int(np.prod(n))

    d = np.array(gcfg.get("fibre_dir", [0.0, 0.0, 1.0]), dtype=float)
    d /= np.linalg.norm(d)

    fibre_mask   = phase_np.ravel() == 1
    orientations = np.zeros((Nv, 3))
    orientations[fibre_mask] = d

    # binary phi scaled by Vf_yarn (fibre packing within the tow)
    Vf_yarn = float(gcfg.get("Vf_yarn", 1.0))
    vf = phase_np.ravel().astype(float) * Vf_yarn

    print(f"phi_target = {gcfg['phi']:.3f}   phi_actual = {phi_act:.4f}")
    return n, L, phase_np.ravel(), orientations.T, vf


def _textile(gcfg: dict):
    """Load an existing textile geometry (VTU or XDMF) and pass through.

    If ``epsilon_voxels > 0`` and the loaded VolumeFraction is binary
    (TexGen VTU), applies SDF+tanh smoothing to create a diffuse interface.
    """
    src = gcfg["source"]
    print(f"Source : {src}")
    n, L, phase, orientations, _, vf, _, _ = read_simulation_input(src)
    print(f"phi_yarn  = {float(np.mean(phase == 1)):.3f}")
    print(f"phi_fibre = {float(np.mean(vf)):.3f}  (mean VolumeFraction)")

    eps_vox = float(gcfg.get("epsilon_voxels", 0.0))
    if eps_vox > 0:
        from scipy.ndimage import distance_transform_edt
        nx, ny, nz = n
        Lx, Ly, Lz = L
        spacing = [Lx/nx, Ly/ny, Lz/nz]
        eps_mm  = eps_vox * min(spacing)

        phase_3d    = phase.reshape(nx, ny, nz)
        inside_d    = distance_transform_edt(phase_3d == 1, sampling=spacing)
        outside_d   = distance_transform_edt(phase_3d == 0, sampling=spacing)
        sdf         = outside_d - inside_d          # + outside yarn, − inside
        smooth_phi  = 0.5 * (1.0 + np.tanh(-sdf / eps_mm))

        Vf_yarn = float(gcfg.get("Vf_yarn", 1.0))
        vf      = (smooth_phi * Vf_yarn).ravel()
        print(f"Applied SDF+tanh  ε={eps_mm:.4f} mm  ({eps_vox} voxels)")
        print(f"phi_fibre (smooth) = {float(np.mean(vf)):.3f}")

    return n, L, phase, orientations, vf


# ── main ──────────────────────────────────────────────────────────────────────

_MODES = {
    "square_rve": _square_rve,
    "textile":    _textile,
}


def main():
    parser = argparse.ArgumentParser(
        description="Generic geometry preprocessor → XDMF"
    )
    parser.add_argument("config", type=Path, help="YAML configuration file")
    args = parser.parse_args()

    cfg  = load_config(args.config)
    gcfg = cfg["geometry"]
    mode = gcfg["type"]

    if mode not in _MODES:
        raise SystemExit(
            f"Unknown geometry.type '{mode}'. "
            f"Choose from: {list(_MODES)}"
        )

    print(f"Config  : {args.config}")
    print(f"Mode    : {mode}")

    # ── build / load geometry ─────────────────────────────────────────────────
    n, L, phase, orientations, vf = _MODES[mode](gcfg)

    Nv = int(np.prod(n))
    dx = tuple(Li / ni for Li, ni in zip(L, n))

    print(f"Grid    : {n}   Nv = {Nv:,}")
    print(f"Domain  : {tuple(f'{v:.4g}' for v in L)}")

    # ── write XDMF/HDF5 ───────────────────────────────────────────────────────
    output  = cfg["output"]
    jobname = cfg["jobname"]
    stem    = f"{output}/{jobname}"

    Path(output).mkdir(parents=True, exist_ok=True)

    with IncrementalWriter(stem, grid_shape=n, grid_spacing=dx) as w:
        w.write_increment(0, {
            "phase":            phase.reshape(n).astype(np.float32),
            "orientation":      orientations.T.reshape(*n, 3).astype(np.float64),
            "volume_fraction":  vf.reshape(n).astype(np.float64),
        }, time=0.0)

    with h5py.File(stem + ".h5", "a") as f:
        f.attrs["n"] = np.array(n, dtype=int)
        f.attrs["L"] = np.array(L, dtype=float)

    print(f"Written → {stem}.h5 / .xdmf")
    print("Open the .xdmf in ParaView with the 'Xdmf3ReaderT' reader.")


if __name__ == "__main__":
    main()
