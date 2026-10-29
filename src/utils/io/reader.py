"""
Readers for FFTjax file formats; all return numpy arrays in XYZ C-order.

    read_xdmf  simulation output .h5/.xdmf (or a bare .h5, given L or dx)
    read_vtu   TexGen / generate_weave voxel mesh
    read_vti   vtkImageData field file
    read_npz   numpy archive (e.g. preprocessing output)
    read_npy   bare phase array, given L or dx

    >>> n, L, fields = read_xdmf("output/simulation/myrun.h5")
    >>> n, L, phase, orient, yarn_index, vf = read_vtu("data/200.vtu")
    >>> n, L, phase = read_npy("geom.npy", dx=(0.001, 0.001, 0.001))
"""

from __future__ import annotations

import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def _resolve_grid_length(
    n: tuple[int, ...],
    L: tuple[float, ...] | None,
    dx: tuple[float, ...] | None,
) -> tuple[float, ...]:
    """Domain length from exactly one of L or dx (L = n dx), for formats without one."""
    if L is not None and dx is not None:
        raise ValueError(
            "specify at most one of L or dx -- they're redundant (L = n * dx)"
        )
    if L is not None:
        return tuple(float(v) for v in L)
    if dx is not None:
        return tuple(float(ni) * float(di) for ni, di in zip(n, dx))
    raise ValueError(
        f"grid shape {n} has no embedded physical length -- pass L=(Lx,Ly,Lz) "
        "or dx=(dx,dy,dz)"
    )


def read_xdmf(
    path: str | Path,
    increment: int = 0,
    L: tuple[float, ...] | None = None,
    dx: tuple[float, ...] | None = None,
) -> tuple[tuple[int, int, int], tuple[float, float, float], dict[str, np.ndarray]]:
    """
    Fields of one increment from IncrementalWriter output (.h5 or .xdmf path),
    or every top-level dataset of a bare .h5.

    Stored ZYX (XDMF CoRectMesh), returned XYZ: (nx, ny, nz) or (nx, ny, nz, k).

    Parameters
    ----------
    path      : .h5 or .xdmf
    increment : increment to read
    L, dx     : domain length for a bare .h5 only (exactly one)

    Returns
    -------
    n, L   : grid shape and length
    fields : {name: array}
    """
    import h5py

    h5_path = Path(path).with_suffix(".h5")
    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    fields: dict[str, np.ndarray] = {}
    with h5py.File(h5_path, "r") as f:
        if "n" in f.attrs and "L" in f.attrs:
            n = tuple(int(v) for v in np.array(f.attrs["n"]))
            L_out = tuple(float(v) for v in np.array(f.attrs["L"]))

            grp_name = f"increment_{increment:06d}"
            if grp_name not in f:
                available = [k for k in f.keys() if k.startswith("increment_")]
                raise KeyError(
                    f"Increment {increment} not found in {h5_path}. "
                    f"Available: {sorted(available)}"
                )
            grp = f[grp_name]
            assert isinstance(grp, h5py.Group)

            for name in grp.keys():
                ds = grp[name]
                assert isinstance(ds, h5py.Dataset)
                arr = np.array(ds)
                if arr.ndim == 3:
                    arr = arr.transpose(2, 1, 0)
                elif arr.ndim == 4:
                    arr = arr.transpose(2, 1, 0, 3)
                fields[name] = arr
        else:
            # Bare .h5: datasets taken as stored (no ZYX convention to undo).
            dataset_names = [name for name in f.keys() if isinstance(f[name], h5py.Dataset)]
            if not dataset_names:
                raise KeyError(
                    f"HDF5 file {h5_path} has no 'n'/'L' root attributes and no "
                    "top-level datasets either -- nothing to read."
                )
            for name in dataset_names:
                fields[name] = np.array(f[name])
            n = tuple(int(v) for v in fields[dataset_names[0]].shape[:3])
            L_out = _resolve_grid_length(n, L, dx)

    return n, L_out, fields   # type: ignore[return-value]


_VTK_DTYPE = {
    "Int8": "i1", "UInt8": "u1",
    "Int16": "i2", "UInt16": "u2",
    "Int32": "i4", "UInt32": "u4",
    "Int64": "i8", "UInt64": "u8",
    "Float32": "f4", "Float64": "f8",
}


def _appended_data_start(raw: bytes) -> int:
    """Start of the appended binary blob (after the ``_`` marker); DataArray offsets count from here."""
    tag_end = raw.index(b"<AppendedData")
    marker = raw.index(b"_", raw.index(b">", tag_end))
    return marker + 1


def _read_appended_array(
    raw: bytes, data_start: int, offset: int, vtk_type: str,
    header_dtype: str, byte_order: str,
) -> np.ndarray:
    """One DataArray from the appended blob: a ``header_dtype`` byte count, then the payload."""
    bo = "<" if byte_order == "LittleEndian" else ">"
    header_size = np.dtype(bo + header_dtype).itemsize
    pos = data_start + offset
    n_bytes = int(np.frombuffer(raw, dtype=bo + header_dtype, count=1, offset=pos)[0])
    payload_start = pos + header_size
    return np.frombuffer(raw, dtype=bo + _VTK_DTYPE[vtk_type], count=n_bytes // np.dtype(bo + _VTK_DTYPE[vtk_type]).itemsize, offset=payload_start)


# TexGen names the fiber-direction array differently across versions.
_ORIENTATION_FIELD_NAMES = ('YarnTangent', 'Orientation')


def _da_array(
    da: ET.Element, raw: bytes | None, data_start: int | None,
    header_dtype: str, byte_order: str,
) -> np.ndarray:
    """Values of one DataArray, inline ``ascii`` or ``appended`` binary."""
    fmt = da.attrib.get('format', 'ascii')
    if fmt == 'ascii':
        return np.fromstring(da.text or "", sep=' ')
    if fmt == 'appended':
        assert raw is not None and data_start is not None, \
            "appended-format DataArray but no <AppendedData> block found"
        return _read_appended_array(
            raw, data_start, int(da.attrib['offset']), da.attrib['type'],
            _VTK_DTYPE[header_dtype], byte_order,
        )
    raise NotImplementedError(
        f"VTU DataArray {da.attrib.get('Name')!r} uses format={fmt!r} -- only "
        "'ascii' and 'appended' are supported (binary/base64 is not)."
    )


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
    Parse a TexGen or generate_weave voxel VTU (hexahedral UnstructuredGrid).

    Appended binary data is not valid XML, so only the header up to
    ``<AppendedData>`` is parsed; payloads are decoded from the raw bytes.

    Returns
    -------
    n, L            : grid shape and length
    phase           : (Nv,) int, 0 matrix, 1 yarn
    orientations    : (3, Nv) yarn tangent
    yarn_index      : (Nv,) int, -1 matrix, >= 0 yarn
    volume_fraction : (Nv,) smooth phi if present, else binary
    """
    raw = Path(path).read_bytes()
    appended_tag = raw.find(b"<AppendedData")
    if appended_tag == -1:
        root = ET.fromstring(raw)
        data_start = None
    else:
        root = ET.fromstring(raw[:appended_tag] + b"</VTKFile>")
        data_start = _appended_data_start(raw)
    header_dtype = root.attrib.get('header_type', 'UInt32')
    byte_order   = root.attrib.get('byte_order', 'LittleEndian')

    piece = root.find('UnstructuredGrid/Piece')
    assert piece is not None, f"No <UnstructuredGrid/Piece> found in {path}"

    pts_node  = piece.find('Points/DataArray')
    conn_node = piece.find('Cells/DataArray[@Name="connectivity"]')
    assert pts_node  is not None, "No Points/DataArray in VTU"
    assert conn_node is not None, "No connectivity DataArray in VTU"

    pts  = _da_array(pts_node, raw, data_start, header_dtype, byte_order).reshape(-1, 3)
    conn = _da_array(conn_node, raw, data_start, header_dtype, byte_order).astype(int).reshape(-1, 8)

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

    orient_name = next((n for n in _ORIENTATION_FIELD_NAMES if n in arrays), None)
    assert orient_name is not None, \
        f"No {'/'.join(_ORIENTATION_FIELD_NAMES)} CellData array found in {path}"

    yarn_vtk = _da_array(arrays['YarnIndex'], raw, data_start, header_dtype, byte_order).astype(int)
    tang_vtk = _da_array(arrays[orient_name], raw, data_start, header_dtype, byte_order).reshape(-1, 3)

    Nv           = nx * ny * nz
    phase_flat   = np.empty(Nv, dtype=int)
    tangent_flat = np.zeros((Nv, 3))
    vf_flat      = np.zeros(Nv)

    phase_flat[fft_idx]   = yarn_vtk
    tangent_flat[fft_idx] = tang_vtk

    if 'VolumeFraction' in arrays:
        vf_raw = _da_array(arrays['VolumeFraction'], raw, data_start, header_dtype, byte_order)
        vf_flat[fft_idx] = vf_raw
    else:
        vf_flat = np.where(phase_flat >= 0, 1.0, 0.0)

    phase        = np.where(phase_flat >= 0, 1, 0).astype(np.int32)
    orientations = tangent_flat.T.copy()

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
    """TexGen / generate_weave voxel mesh: ``n, L, phase, orientations, yarn_index, volume_fraction``."""
    return _read_texgen_vtu(path)


def _read_vti(
    path: str | Path,
) -> tuple[tuple[int, int, int], tuple[float, float, float], dict[str, np.ndarray]]:
    """
    Parse a vtkImageData file (inline ascii only). CellData is preferred;
    with PointData only, ``n`` is the point grid (one larger per axis).
    """
    tree  = ET.parse(str(path))
    image = tree.getroot().find('ImageData')
    assert image is not None, f"No <ImageData> found in {path}"
    piece = image.find('Piece')
    assert piece is not None, f"No <ImageData/Piece> found in {path}"

    x0, x1, y0, y1, z0, z1 = (int(v) for v in image.attrib['WholeExtent'].split())
    spacing = tuple(float(v) for v in image.attrib['Spacing'].split())

    cell_data  = piece.find('CellData')
    point_data = piece.find('PointData')
    if cell_data is not None and len(cell_data):
        n = (x1 - x0, y1 - y0, z1 - z0)
        data_node = cell_data
    elif point_data is not None and len(point_data):
        n = (x1 - x0 + 1, y1 - y0 + 1, z1 - z0 + 1)
        data_node = point_data
    else:
        raise KeyError(f"No <CellData> or <PointData> arrays found in {path}")

    nx, ny, nz = n
    dx_, dy_, dz_ = spacing
    L = (nx * dx_, ny * dy_, nz * dz_)

    fields: dict[str, np.ndarray] = {}
    for da in data_node.findall('DataArray'):
        name = da.attrib.get('Name', 'field')
        fmt  = da.attrib.get('format', 'ascii')
        if fmt != 'ascii':
            raise NotImplementedError(
                f"VTI DataArray '{name}' in {path} uses format={fmt!r} -- only "
                "inline ascii is supported (binary-appended/base64 VTI is not)."
            )
        ncomp = int(da.attrib.get('NumberOfComponents', 1))
        raw   = np.fromstring(da.text or "", sep=' ')

        # X fastest, like the ZYX storage in read_xdmf.
        if ncomp == 1:
            arr = raw.reshape(nz, ny, nx).transpose(2, 1, 0)
        else:
            arr = raw.reshape(nz, ny, nx, ncomp).transpose(2, 1, 0, 3)
        fields[name] = arr

    return n, L, fields


def read_vti(
    path: str | Path,
) -> tuple[tuple[int, int, int], tuple[float, float, float], dict[str, np.ndarray]]:
    """vtkImageData field file: ``n, L, fields`` (XYZ)."""
    return _read_vti(path)


def read_npz(path: str | Path) -> dict[str, np.ndarray]:
    """
    .npz archive as a plain dict (no open NpzFile); preprocessing keys include
    n, L, phase, orientations, yarn_index, d_init, H_init.
    """
    return dict(np.load(path))


def read_npy(
    path: str | Path,
    L: tuple[float, ...] | None = None,
    dx: tuple[float, ...] | None = None,
) -> tuple[tuple[int, ...], tuple[float, ...], np.ndarray]:
    """
    Bare .npy array; it has no length, so give exactly one of L or dx.

    Returns
    -------
    n, L  : shape and resolved length
    array : the array as stored (XYZ)
    """
    array = np.load(path)
    n = tuple(int(v) for v in array.shape)
    L_out = _resolve_grid_length(n, L, dx)
    return n, L_out, array


class SimulationReader:
    """
    Geometry and optional pre-crack state from any supported format, chosen by suffix.

    ``phase_key``/``orientation_key`` name the fields in .h5/.xdmf/.vti/.npz
    (.vtu derives both from YarnIndex/YarnTangent; .npz also accepts the
    legacy "orientations"). L/dx only apply to .npy and bare .h5.

        >>> n, L, phase, orientations, yarn_index, vf, d_init, H_init = SimulationReader(path).read()

    ``read()`` returns
    ------------------
    n, L            : grid shape and length
    phase           : (Nv,) int, 0 matrix, 1 yarn, 2 void/crack
    orientations    : (3, Nv)
    yarn_index      : (Nv,) int, -1 matrix, -2 void
    volume_fraction : (Nv,)
    d_init, H_init  : (Nv,) zeros unless stored in a .npz
    """

    _DISPATCH = {
        ".vtu":  "_from_vtu",
        ".vti":  "_from_vti",
        ".h5":   "_from_xdmf",
        ".xdmf": "_from_xdmf",
        ".npz":  "_from_npz",
        ".npy":  "_from_npy",
    }

    # An L/dx override would contradict the file: warn and ignore it.
    _ALWAYS_SELF_DESCRIBING = {".vtu", ".vti", ".npz"}

    def __init__(
        self,
        path: str | Path,
        L: tuple[float, ...] | None = None,
        dx: tuple[float, ...] | None = None,
        phase_key: str = "phase",
        orientation_key: str = "orientation",
    ):
        self.path = Path(path)
        suffix = self.path.suffix.lower()
        if suffix not in self._DISPATCH:
            raise ValueError(
                f"Unsupported file format '{suffix}'. "
                "Use .vtu, .vti, .h5, .xdmf, .npz, or .npy."
            )
        self._method_name = self._DISPATCH[suffix]

        if (L is not None or dx is not None) and suffix in self._ALWAYS_SELF_DESCRIBING:
            warnings.warn(
                f"L/dx given for a {suffix} file, which is always self-describing -- "
                "ignoring the override and using the file's own grid length.",
                stacklevel=2,
            )
            L = dx = None
        self.L = L
        self.dx = dx
        self.phase_key = phase_key
        self.orientation_key = orientation_key

    def read(self):
        return getattr(self, self._method_name)()

    @staticmethod
    def _defaults(n: tuple, phase: np.ndarray):
        """Defaults ``orientations, yarn_index, volume_fraction, d_init, H_init``."""
        Nv = int(np.prod(n))
        return (
            np.zeros((3, Nv)),
            np.full(Nv, -1, dtype=int),
            (phase == 1).astype(float),
            np.zeros(Nv),
            np.zeros(Nv),
        )

    def _from_vtu(self):
        n, L, phase, orientations, yarn_index, vf = read_vtu(self.path)
        Nv = int(np.prod(n))
        return n, L, phase, orientations, yarn_index, vf, np.zeros(Nv), np.zeros(Nv)

    def _from_xdmf(self):
        n, L, fields = read_xdmf(self.path, L=self.L, dx=self.dx)
        Nv    = int(np.prod(n))
        phase = fields.get(self.phase_key, np.zeros(Nv)).ravel().astype(int)
        ori_def, yi_def, vf_def, d_def, H_def = self._defaults(n, phase)
        _ori = fields.get(self.orientation_key)
        orientations = (_ori.reshape(-1, 3).T if _ori is not None
                        else ori_def)
        _yi = fields.get("yarn_index")
        yarn_index   = _yi.ravel().astype(int) if _yi is not None else yi_def
        _vf = fields.get("volume_fraction")
        vf           = _vf.ravel() if _vf is not None else vf_def
        return n, L, phase, orientations, yarn_index, vf, d_def, H_def

    def _from_vti(self):
        n, L, fields = read_vti(self.path)
        Nv    = int(np.prod(n))
        phase = fields.get(self.phase_key, np.zeros(Nv)).ravel().astype(int)
        ori_def, yi_def, vf_def, d_def, H_def = self._defaults(n, phase)
        _ori = fields.get(self.orientation_key)
        orientations = (_ori.reshape(-1, 3).T if _ori is not None
                        else ori_def)
        _yi = fields.get("yarn_index")
        yarn_index   = _yi.ravel().astype(int) if _yi is not None else yi_def
        _vf = fields.get("volume_fraction")
        vf           = _vf.ravel() if _vf is not None else vf_def
        return n, L, phase, orientations, yarn_index, vf, d_def, H_def

    def _from_npy(self):
        n_raw, L_raw, array = read_npy(self.path, L=self.L, dx=self.dx)
        n = n_raw if len(n_raw) == 3 else (*n_raw, 1)
        L = L_raw if len(L_raw) == 3 else (*L_raw, min(L_raw) / n_raw[0])
        phase = array.ravel().astype(int)
        ori_def, yi_def, vf_def, d_def, H_def = self._defaults(n, phase)
        return n, L, phase, ori_def, yi_def, vf_def, d_def, H_def

    def _from_npz(self):
        data  = read_npz(self.path)
        n_raw = tuple(int(v) for v in data["n"])
        L_raw = tuple(float(v) for v in data["L"])
        n = n_raw if len(n_raw) == 3 else (*n_raw, 1)
        L = L_raw if len(L_raw) == 3 else (*L_raw, min(L_raw) / n_raw[0])
        phase = data[self.phase_key].ravel().astype(int)
        ori_def, yi_def, vf_def, d_def, H_def = self._defaults(n, phase)
        _ori = data.get(self.orientation_key)
        if _ori is None:
            _ori = data.get("orientations")
        if _ori is not None:
            arr = np.asarray(_ori)
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
