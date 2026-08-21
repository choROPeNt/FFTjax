"""
Readers for the file formats used in FFTjax.

    read_xdmf  — load field arrays from an HDF5/XDMF simulation output
    read_vtu   — load a TexGen (or generate_weave) voxel mesh
    read_npz   — load a numpy .npz archive (e.g. preproc output)

All functions return plain numpy arrays in FFTjax C-order (X slowest).

Quick reference
---------------
>>> from utils.io.reader import read_xdmf, read_vtu, read_npz

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

import xml.etree.ElementTree as ET  # used for VTU parsing
from pathlib import Path

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

def _read_texgen_vtu(
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
    Parse a TexGen voxel VTU (UnstructuredGrid of hexahedral cells).

    Also accepts VTU files produced by ``generate_weave.py``; those
    contain a ``VolumeFraction`` field with smooth SDF-based phi values.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    n              : (nx, ny, nz)
    L              : (Lx, Ly, Lz)  mm
    phase          : (Nv,) int      0=matrix, 1=yarn  (C-order)
    orientations   : (3, Nv) float  YarnTangent per voxel
    yarn_index     : (Nv,) int      raw YarnIndex (-1=matrix, >=0=yarn number)
    volume_fraction: (Nv,) float    smooth phi if present, binary fallback otherwise
    """
    tree  = ET.parse(str(path))
    piece = tree.getroot().find('UnstructuredGrid/Piece')
    assert piece is not None, f"No <UnstructuredGrid/Piece> found in {path}"

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
    Lx, Ly, Lz = nx * dx, ny * dy, nz * dz

    ix_arr  = np.round((centroids[:, 0] - xu[0]) / dx).astype(int)
    iy_arr  = np.round((centroids[:, 1] - yu[0]) / dy).astype(int)
    iz_arr  = np.round((centroids[:, 2] - zu[0]) / dz).astype(int)
    fft_idx = ix_arr * (ny * nz) + iy_arr * nz + iz_arr

    cd = piece.find('CellData')
    assert cd is not None, f"No <CellData> block found in {path}"
    arrays = {a.attrib['Name']: a for a in cd.findall('DataArray')}

    yarn_vtk = np.fromstring(arrays['YarnIndex'].text   or "", sep=' ').astype(int)
    tang_vtk = np.fromstring(arrays['YarnTangent'].text or "", sep=' ').reshape(-1, 3)

    Nv           = nx * ny * nz
    phase_flat   = np.empty(Nv, dtype=int)
    tangent_flat = np.zeros((Nv, 3))
    vf_flat      = np.zeros(Nv)

    phase_flat[fft_idx]   = yarn_vtk
    tangent_flat[fft_idx] = tang_vtk

    if 'VolumeFraction' in arrays:
        vf_raw = np.fromstring(arrays['VolumeFraction'].text or "", sep=' ')
        vf_flat[fft_idx] = vf_raw
    else:
        vf_flat = np.where(phase_flat >= 0, 1.0, 0.0)

    phase        = np.where(phase_flat >= 0, 1, 0).astype(np.int32)
    orientations = tangent_flat.T.copy()   # (3, Nv)

    return (nx, ny, nz), (Lx, Ly, Lz), phase, orientations, phase_flat, vf_flat


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
    Read a TexGen (or generate_weave) voxel mesh.

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
    return _read_texgen_vtu(path)


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

class SimulationReader:
    """
    Load geometry + optional pre-crack state from any supported format.

    Dispatches by file suffix — ``.vtu`` → read_vtu, ``.h5``/``.xdmf`` →
    read_xdmf, ``.npz`` → read_npz — then normalizes each format's fields
    into one common set of arrays.

    >>> n, L, phase, orientations, yarn_index, vf, d_init, H_init = (
    ...     SimulationReader(path).read()
    ... )

    Returns (from ``.read()``)
    ---------------------------
    n              : (nx, ny, nz)
    L              : (Lx, Ly, Lz)  mm
    phase          : (Nv,) int      0=matrix, 1=yarn, 2=void/crack
    orientations   : (3, Nv) float  nearest-yarn tangent per voxel
    yarn_index     : (Nv,) int      raw YarnIndex (-1=matrix, >=0=yarn, -2=void)
    volume_fraction: (Nv,) float    smooth φ·Vf or binary fallback
    d_init         : (Nv,) float    initial damage  (zeros unless stored in npz)
    H_init         : (Nv,) float    initial history (zeros unless stored in npz)
    """

    _DISPATCH = {
        ".vtu":  "_from_vtu",
        ".h5":   "_from_xdmf",
        ".xdmf": "_from_xdmf",
        ".npz":  "_from_npz",
    }

    def __init__(self, path: str | Path):
        self.path = Path(path)
        suffix = self.path.suffix.lower()
        if suffix not in self._DISPATCH:
            raise ValueError(
                f"Unsupported file format '{suffix}'. "
                "Use .vtu, .h5, .xdmf, or .npz."
            )
        self._method_name = self._DISPATCH[suffix]

    def read(self):
        return getattr(self, self._method_name)()

    @staticmethod
    def _defaults(n: tuple, phase: np.ndarray):
        """Fallback values for optional fields."""
        Nv = int(np.prod(n))
        return (
            np.zeros((3, Nv)),                # orientations
            np.full(Nv, -1, dtype=int),       # yarn_index  (-1 = matrix)
            (phase == 1).astype(float),       # volume_fraction (binary fallback)
            np.zeros(Nv),                     # d_init
            np.zeros(Nv),                     # H_init
        )

    def _from_vtu(self):
        n, L, phase, orientations, yarn_index, vf = read_vtu(self.path)
        Nv = int(np.prod(n))
        return n, L, phase, orientations, yarn_index, vf, np.zeros(Nv), np.zeros(Nv)

    def _from_xdmf(self):
        n, L, fields = read_xdmf(self.path)
        Nv    = int(np.prod(n))
        phase = fields.get("phase", np.zeros(Nv)).ravel().astype(int)
        ori_def, yi_def, vf_def, d_def, H_def = self._defaults(n, phase)
        _ori = fields.get("orientation")
        orientations = (_ori.reshape(-1, 3).T if _ori is not None
                        else ori_def)
        _yi = fields.get("yarn_index")
        yarn_index   = _yi.ravel().astype(int) if _yi is not None else yi_def
        _vf = fields.get("volume_fraction")
        vf           = _vf.ravel() if _vf is not None else vf_def
        return n, L, phase, orientations, yarn_index, vf, d_def, H_def

    def _from_npz(self):
        data  = read_npz(self.path)
        n_raw = tuple(int(v) for v in data["n"])
        L_raw = tuple(float(v) for v in data["L"])
        # promote 2-D grids to 3-D (nz = 1)
        n = n_raw if len(n_raw) == 3 else (*n_raw, 1)
        L = L_raw if len(L_raw) == 3 else (*L_raw, min(L_raw) / n_raw[0])
        phase = data["phase"].ravel().astype(int)
        ori_def, yi_def, vf_def, d_def, H_def = self._defaults(n, phase)
        # accept both "orientations" (3, Nv) and "orientation" (*spatial, 3)
        _ori = data.get("orientations") or data.get("orientation")
        if _ori is not None:
            arr = np.asarray(_ori)
            # (*, 3) spatial grid → (3, Nv)
            orientations = arr.reshape(-1, 3).T.copy() if arr.shape[-1] == 3 else arr
        else:
            orientations = ori_def
        _yi  = data.get("yarn_index")
        yarn_index   = _yi.ravel().astype(int) if _yi is not None else yi_def
        _vf  = data.get("volume_fraction")
        vf   = _vf.ravel() if _vf is not None else vf_def
        _di  = data.get("d_init")
        d_init  = _di.ravel()  if _di is not None else d_def
        _Hi  = data.get("H_init")
        H_init  = _Hi.ravel()  if _Hi is not None else H_def
        return n, L, phase, orientations, yarn_index, vf, d_init, H_init
