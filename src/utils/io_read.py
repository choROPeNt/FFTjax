"""
Readers for the three file formats used in FFTjax.

    read_xdmf  — load field arrays from an HDF5/XDMF simulation output
    read_vtu   — load a TexGen (or generate_weave_vtu) voxel mesh
    read_npz   — load a numpy .npz archive (e.g. preproc output)

All functions return plain numpy arrays in FFTjax C-order (X slowest).

Quick reference
---------------
>>> from utils.io_read import read_xdmf, read_vtu, read_npz

# simulation output
>>> n, L, fields = read_xdmf("output/simulation/myrun.h5")
>>> stress = fields["stress"]   # (nx, ny, nz, 6)

# voxel geometry
>>> n, L, phase, orient, yarn_index, vf = read_vtu("data/200.vtu")

# preprocessor output
>>> data = read_npz("output/preprocessed/myrun_delam_preproc.npz")
>>> d_init = data["d_init"]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


# ── XDMF / HDF5 ──────────────────────────────────────────────────────────────

def read_xdmf(
    path: str | Path,
    increment: int = 0,
) -> tuple[tuple[int, int, int], tuple[float, float, float], dict[str, np.ndarray]]:
    """
    Read field arrays from a simulation HDF5 file (companion to .xdmf).

    Accepts either the ``.h5`` or ``.xdmf`` path — the HDF5 file is located
    automatically by replacing the suffix with ``.h5``.

    Arrays are stored in ZYX order in HDF5 (XDMF CoRectMesh convention).
    This function transparently transposes them back to FFTjax C-order XYZ,
    so the returned arrays have shape ``(nx, ny, nz)`` for scalars and
    ``(nx, ny, nz, k)`` for k-component fields.

    Parameters
    ----------
    path      : .h5 or .xdmf file path
    increment : increment index to read (default 0)

    Returns
    -------
    n      : (nx, ny, nz)
    L      : (Lx, Ly, Lz)  mm
    fields : dict[name → numpy array in XYZ order]
    """
    import h5py

    h5_path = Path(path).with_suffix(".h5")
    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    fields: dict[str, np.ndarray] = {}
    with h5py.File(h5_path, "r") as f:
        # grid metadata (stored as root attributes by IncrementalWriter / preproc)
        if "n" in f.attrs and "L" in f.attrs:
            n = tuple(int(v) for v in np.array(f.attrs["n"]))
            L = tuple(float(v) for v in np.array(f.attrs["L"]))
        else:
            raise KeyError(
                f"HDF5 file {h5_path} has no 'n'/'L' root attributes. "
                "Was it written by IncrementalWriter with h5.attrs set?"
            )

        grp_name = f"increment_{increment:06d}"
        if grp_name not in f:
            available = [k for k in f.keys() if k.startswith("increment_")]
            raise KeyError(
                f"Increment {increment} not found in {h5_path}. "
                f"Available: {sorted(available)}"
            )
        grp = f[grp_name]
        assert isinstance(grp, h5py.Group)

        nx, ny, nz = n
        for name in grp.keys():
            ds = grp[name]
            assert isinstance(ds, h5py.Dataset)
            arr = np.array(ds)

            # HDF5 stores spatial dims in ZYX; transpose back to XYZ
            if arr.ndim == 3:                    # scalar (nz, ny, nx)
                arr = arr.transpose(2, 1, 0)
            elif arr.ndim == 4:                  # vector/tensor (nz, ny, nx, k)
                arr = arr.transpose(2, 1, 0, 3)

            fields[name] = arr

    return n, L, fields   # type: ignore[return-value]


# ── VTU ──────────────────────────────────────────────────────────────────────

def read_vtu(
    path: str | Path,
) -> tuple[
    tuple[int, int, int],
    tuple[float, float, float],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Read a TexGen (or generate_weave_vtu) voxel mesh.

    Parameters
    ----------
    path : .vtu file path

    Returns
    -------
    n              : (nx, ny, nz)
    L              : (Lx, Ly, Lz)  mm
    phase          : (Nv,) int      0=matrix, 1=yarn
    orientations   : (3, Nv) float  YarnTangent per voxel
    yarn_index     : (Nv,) int      raw YarnIndex (-1=matrix, >=0=yarn)
    volume_fraction: (Nv,) float    smooth φ·Vf_yarn or binary fallback
    """
    from generation.vtu import read_texgen_vtu
    return read_texgen_vtu(path)


# ── NPZ ──────────────────────────────────────────────────────────────────────

def read_npz(path: str | Path) -> dict[str, np.ndarray]:

    """
    Load a numpy .npz archive (e.g. preproc_delamination output).

    Returns a plain dict so arrays can be accessed by name without keeping
    the NpzFile object open.

    Common keys (preproc output)
    ----------------------------
    n, L, phase, orientations, yarn_index, d_init, H_init

    Parameters
    ----------
    path : .npz file path

    Returns
    -------
    dict[str, np.ndarray]
    """
    return dict(np.load(path))


# ── Unified simulation input loader ──────────────────────────────────────────

def read_simulation_input(
    path: str | Path,
) -> tuple[
    tuple[int, int, int],
    tuple[float, float, float],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Load geometry + optional pre-crack state from any supported format.

    Accepts ``.vtu``, ``.h5``, ``.xdmf``, or ``.npz``.

    Returns
    -------
    n              : (nx, ny, nz)
    L              : (Lx, Ly, Lz)  mm
    phase          : (Nv,) int      0=matrix, 1=yarn, 2=void/crack
    orientations   : (3, Nv) float  nearest-yarn tangent per voxel
    yarn_index     : (Nv,) int      raw YarnIndex (-1=matrix, >=0=yarn, -2=void)
    volume_fraction: (Nv,) float    smooth φ·Vf or binary fallback
    d_init         : (Nv,) float    initial damage  (zeros unless stored in npz)
    H_init         : (Nv,) float    initial history (zeros unless stored in npz)
    """
    path   = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".vtu":
        n, L, phase, orientations, yarn_index, vf = read_vtu(path)
        Nv     = int(np.prod(n))
        return n, L, phase, orientations, yarn_index, vf, np.zeros(Nv), np.zeros(Nv)

    if suffix in (".h5", ".xdmf"):
        n, L, fields = read_xdmf(path)
        Nv     = int(np.prod(n))
        phase  = fields.get("phase",     np.zeros(Nv)).ravel().astype(int)
        orient = fields.get("orientation", np.zeros((*n, 3)))
        orientations = orient.reshape(-1, 3).T          # (3, Nv)
        yarn_index   = fields.get("yarn_index", np.full(Nv, -1.0)).ravel().astype(int)
        vf = fields.get("volume_fraction",
                         (phase == 1).astype(float)).ravel()
        return n, L, phase, orientations, yarn_index, vf, np.zeros(Nv), np.zeros(Nv)

    if suffix == ".npz":
        data  = read_npz(path)
        n     = tuple(int(v) for v in data["n"])
        L     = tuple(float(v) for v in data["L"])
        Nv    = int(np.prod(n))
        phase        = data["phase"].ravel().astype(int)
        orientations = data["orientations"]              # (3, Nv)
        yarn_index   = data["yarn_index"].ravel().astype(int)
        vf           = data.get("volume_fraction",
                                 (phase == 1).astype(float)).ravel()
        d_init       = data.get("d_init",  np.zeros(Nv)).ravel()
        H_init       = data.get("H_init",  np.zeros(Nv)).ravel()
        return n, L, phase, orientations, yarn_index, vf, d_init, H_init

    raise ValueError(
        f"Unsupported file format '{suffix}'. "
        "Use .vtu, .h5, .xdmf, or .npz."
    )
