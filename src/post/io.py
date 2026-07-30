"""
HDF5 + XDMF incremental writer for ParaView visualization.

Examples
--------
::

    writer = IncrementalWriter("results/sim", grid_shape=(32, 32, 32), grid_spacing=(1.0, 1.0, 1.0))
    for inc, data in enumerate(simulation):
        writer.write_increment(inc, {
            "stress":       to_voigt(stress),   # (nx, ny, nz, 3, 3) -> (nx, ny, nz, 6)
            "strain":       to_voigt(strain),
            "phase":        phase,              # scalar (nx, ny, nz)
            "displacement": displacement,       # (nx, ny, nz, 3)
        }, point_fields={"displacement"})
    writer.close()

Notes
-----
Voigt convention (symmetric tensors) index mapping: 0=11 1=22 2=33 3=12 4=13 5=23
(Abaqus convention). Strain factor: stress -> factor 1, strain -> factor 2 on shear
components (Mandel).

Every field is written voxel-centered (XDMF ``Center="Cell"``) by default, matching
the FFT/voxel discretization this project uses throughout (strain, stress, and
damage are naturally per-voxel state variables). Pass a field's name in
``point_fields`` to instead write it node-centered (XDMF ``Center="Node"`` -- the
spec's term; ParaView surfaces this as "Point Data"), one value per corner of the
n+1 node grid -- this is what ParaView's "Warp By Vector" filter needs to deform
the geometry, since it operates on point data. The n-cell values are interpolated
to the n+1 corner nodes by periodic box averaging (2 neighbours per axis, wrapping
across the domain boundary), consistent with the periodic boundary conditions this
project's FFT solvers assume.
"""

import os
import numpy as np
import h5py


# map numpy dtype to XDMF NumberType + Precision
_XDMF_DTYPE = {
    "float32": ("Float", "4"),
    "float64": ("Float", "8"),
    "int32":   ("Int",   "4"),
    "int64":   ("Int",   "8"),
    "uint8":   ("UInt",  "1"),
    "uint16":  ("UInt",  "2"),
    "uint32":  ("UInt",  "4"),
}


def _xdmf_dtype(arr: np.ndarray) -> tuple[str, str]:
    key = arr.dtype.name
    if key not in _XDMF_DTYPE:
        raise ValueError(f"Unsupported dtype for XDMF export: {arr.dtype}")
    return _XDMF_DTYPE[key]


class IncrementalWriter:
    """
    Writes one HDF5 file and one XDMF file that references it,
    appending one time step (increment) at a time.

    Parameters
    ----------
    base_path : str
        Path without extension, e.g. "results/simulation".
        Creates  <base_path>.h5  and  <base_path>.xdmf.
    grid_shape : tuple[int, ...]
        Number of voxels per dimension, e.g. (nx, ny) or (nx, ny, nz).
    grid_spacing : tuple[float, ...]
        Physical size of one voxel per dimension (dx, dy) or (dx, dy, dz).
    origin : tuple[float, ...] | None
        Physical origin of the grid. Defaults to all zeros.
    """

    def __init__(
        self,
        base_path: str,
        grid_shape: tuple[int, ...],
        grid_spacing: tuple[float, ...],
        origin: tuple[float, ...] | None = None,
    ):
        if len(grid_shape) not in (2, 3):
            raise ValueError("grid_shape must be 2-D or 3-D")
        if len(grid_spacing) != len(grid_shape):
            raise ValueError("grid_spacing must have the same length as grid_shape")

        self.ndim = len(grid_shape)
        self.grid_shape = grid_shape
        self.grid_spacing = grid_spacing
        self.origin = origin if origin is not None else tuple(0.0 for _ in grid_shape)

        os.makedirs(os.path.dirname(base_path) or ".", exist_ok=True)

        self.h5_path  = base_path + ".h5"
        self.xdmf_path = base_path + ".xdmf"
        self._h5_basename = os.path.basename(self.h5_path)

        self._h5 = h5py.File(self.h5_path, "w")
        # (inc_id, [(field_name, center)], time) -- center is "cell" or "point"
        self._increments: list[tuple[int, list[tuple[str, str]], float]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write_increment(
        self,
        increment: int,
        fields: dict[str, np.ndarray],
        time: float | None = None,
        point_fields: set[str] | frozenset[str] | None = None,
    ) -> None:
        """
        Write one increment to HDF5.

        Parameters
        ----------
        increment : int
            Increment index — used as the HDF5 group name.
        fields : dict[str, np.ndarray]
            Mapping of field name → numpy array, given voxel-centered (one value
            per voxel) regardless of ``point_fields`` below.
            Each array must match grid_shape for scalar fields,
            or ``(*grid_shape, components)`` for vector / tensor fields.
            JAX arrays are converted automatically via np.asarray().
        time : float | None
            Physical time value written to the XDMF ``<Time Value="..."/>``.
            Defaults to the increment index when not provided.
        point_fields : set[str] | None
            Names (a subset of ``fields``' keys) to write node-centered instead
            of voxel-centered -- e.g. ``{"displacement"}`` so ParaView's "Warp By
            Vector" filter can use it directly. Each such field is interpolated
            from its n voxel values to the n+1 corner-node values per axis via
            periodic box averaging (see module docstring) before being written.
        """
        grp = self._h5.require_group(f"increment_{increment:06d}")
        written: list[tuple[str, str]] = []

        for name, arr in fields.items():
            arr = np.asarray(arr)  # detach from JAX / device
            _check_field_shape(arr, self.grid_shape, name)

            is_point = point_fields is not None and name in point_fields
            if is_point:
                arr = _cell_to_node(arr, self.ndim)

            # XDMF CoRectMesh convention: Dimensions="nz ny nx" (last dim = X = fastest).
            # Transpose spatial axes to ZYX order so the HDF5 layout matches.
            spatial = tuple(range(self.ndim - 1, -1, -1))   # (2,1,0) for 3-D
            trailing = tuple(range(self.ndim, arr.ndim))
            arr = arr.transpose(spatial + trailing)
            grp.create_dataset(name, data=arr, compression="gzip", compression_opts=4)
            written.append((name, "point" if is_point else "cell"))

        self._h5.flush()
        time_val = float(increment) if time is None else float(time)
        self._increments.append((increment, written, time_val))
        self._write_xdmf()  # rewrite XDMF so it is always valid on disk

    def close(self) -> None:
        """Flush and close the HDF5 file. XDMF is already up to date."""
        self._h5.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _write_xdmf(self) -> None:
        lines = []
        lines.append('<?xml version="1.0" ?>')
        lines.append('<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>')
        lines.append('<Xdmf xmlns:xi="http://www.w3.org/2001/XInclude" Version="3.0">')
        lines.append("  <Domain>")
        lines.append('    <Grid Name="TimeSeries" GridType="Collection" CollectionType="Temporal">')

        for inc_id, field_names, time_val in self._increments:
            grp_path = f"increment_{inc_id:06d}"
            lines.append(f'      <Grid Name="{grp_path}" GridType="Uniform">')
            lines.append(f'        <Time Value="{time_val}" />')
            lines.append(f"        {self._topology_tag()}")
            lines.append(f"        {self._geometry_tag()}")

            for fname, center in field_names:
                ds = self._h5[grp_path][fname]
                arr_shape = ds.shape
                number_type, precision = _xdmf_dtype(ds)
                # grid_shape is (nx,ny,nz) voxels; stored data is transposed to
                # (nz,ny,nx,...), and node-centered fields have one extra value
                # per axis (n+1 corners) -- reverse (and grow) to match.
                if center == "point":
                    dims_grid = tuple(reversed(tuple(s + 1 for s in self.grid_shape)))
                else:
                    dims_grid = tuple(reversed(self.grid_shape))
                attr_type, n_components, dims_str = _attribute_meta(arr_shape, dims_grid)
                h5_ref = f"{self._h5_basename}:/{grp_path}/{fname}"

                # XDMF's Center enum is Node|Cell|Grid|Face|Edge|Other -- "Point"
                # is not a valid value (that's VTK/ParaView's user-facing term
                # for the same concept); the internal "point"/"cell" strings
                # here map to the spec's "Node"/"Cell".
                xdmf_center = "Node" if center == "point" else "Cell"
                lines.append(
                    f'        <Attribute Name="{fname}" AttributeType="{attr_type}" '
                    f'Center="{xdmf_center}">'
                )
                lines.append(
                    f'          <DataItem Dimensions="{dims_str}" '
                    f'NumberType="{number_type}" Precision="{precision}" Format="HDF">'
                )
                lines.append(f"            {h5_ref}")
                lines.append("          </DataItem>")
                lines.append("        </Attribute>")

            lines.append("      </Grid>")

        lines.append("    </Grid>")
        lines.append("  </Domain>")
        lines.append("</Xdmf>")

        with open(self.xdmf_path, "w") as f:
            f.write("\n".join(lines) + "\n")

    def _topology_tag(self) -> str:
        # XDMF CoRectMesh Dimensions = number of *nodes* (corners), listed ZYX.
        # We use n+1 nodes per direction so we get exactly n cells. Most
        # Attributes declare Center="Cell" (one value per voxel); fields in
        # point_fields declare Center="Point" instead, using these same n+1
        # corner nodes directly (see _cell_to_node).
        if self.ndim == 2:
            nx, ny = self.grid_shape
            return f'<Topology TopologyType="2DCoRectMesh" Dimensions="{ny + 1} {nx + 1}" />'
        else:
            nx, ny, nz = self.grid_shape
            return f'<Topology TopologyType="3DCoRectMesh" Dimensions="{nz + 1} {ny + 1} {nx + 1}" />'

    def _geometry_tag(self) -> str:
        # Corner nodes start at the domain origin (no half-voxel shift).
        # Geometry values listed slowest→fastest (ZYX) to match topology Dimensions.
        if self.ndim == 2:
            ox, oy = self.origin
            dx, dy = self.grid_spacing
            return (
                f'<Geometry GeometryType="ORIGIN_DXDY">'
                f'<DataItem Dimensions="2" NumberType="Float" Precision="8" Format="XML">'
                f'{oy} {ox}</DataItem>'
                f'<DataItem Dimensions="2" NumberType="Float" Precision="8" Format="XML">'
                f'{dy} {dx}</DataItem>'
                f"</Geometry>"
            )
        else:
            ox, oy, oz = self.origin
            dx, dy, dz = self.grid_spacing
            return (
                f'<Geometry GeometryType="ORIGIN_DXDYDZ">'
                f'<DataItem Dimensions="3" NumberType="Float" Precision="8" Format="XML">'
                f'{oz} {oy} {ox}</DataItem>'
                f'<DataItem Dimensions="3" NumberType="Float" Precision="8" Format="XML">'
                f'{dz} {dy} {dx}</DataItem>'
                f"</Geometry>"
            )


# ------------------------------------------------------------------
# Public Voigt helpers
# ------------------------------------------------------------------

# Voigt index pairs — Abaqus convention: [11, 22, 33, 12, 13, 23]
_VOIGT_IJ = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))


def to_voigt(tensor: np.ndarray, mandel: bool = False) -> np.ndarray:
    """
    Convert a symmetric 3×3 tensor field to Voigt notation.

    Parameters
    ----------
    tensor : array (..., 3, 3)
        Full tensor field. Leading dimensions are arbitrary (grid voxels).
    mandel : bool
        If True, apply Mandel scaling (√2 on shear components).
        Use this when you need energy-consistent norm preservation,
        e.g. for strain fields used in further computation.
        For stress fields, keep mandel=False.

    Returns
    -------
    array (..., 6)
        Voigt vector. Index order: [σ11, σ22, σ33, σ12, σ13, σ23]  (Abaqus convention)
    """
    tensor = np.asarray(tensor)
    if tensor.shape[-2:] != (3, 3):
        raise ValueError(f"Expected (..., 3, 3), got {tensor.shape}")

    voigt = np.stack([tensor[..., i, j] for i, j in _VOIGT_IJ], axis=-1)

    if mandel:
        voigt[..., 3:] *= np.sqrt(2.0)

    return voigt


def from_voigt(voigt: np.ndarray, mandel: bool = False) -> np.ndarray:
    """
    Convert a Voigt-notation field back to a full symmetric 3×3 tensor.

    Parameters
    ----------
    voigt : array (..., 6)
        Voigt field. Index order: [xx, yy, zz, yz, xz, xy]  (11,22,33,23,13,12)
    mandel : bool
        If True, undo Mandel scaling (divide shear by √2).

    Returns
    -------
    array (..., 3, 3)
    """
    voigt = np.asarray(voigt, dtype=float)
    if voigt.shape[-1] != 6:
        raise ValueError(f"Expected (..., 6), got {voigt.shape}")

    if mandel:
        voigt = voigt.copy()
        voigt[..., 3:] /= np.sqrt(2.0)

    out = np.zeros(voigt.shape[:-1] + (3, 3), dtype=voigt.dtype)
    for k, (i, j) in enumerate(_VOIGT_IJ):
        out[..., i, j] = voigt[..., k]
        out[..., j, i] = voigt[..., k]  # symmetry

    return out


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------

def _cell_to_node(arr: np.ndarray, ndim: int) -> np.ndarray:
    """
    Interpolate a voxel-centered field (n per axis) to its corner nodes
    (n+1 per axis) by periodic box averaging.

    Each node is the average of the (up to) 2**ndim voxels touching it. This is
    done as ``ndim`` sequential 1-D box averages of width 2 (averaging is
    separable), each with a periodic wrap: node i's two neighbours along an
    axis are voxels ``i-1 mod n`` and ``i mod n``, so e.g. node 0 and node n
    (the two ends of the CoRectMesh's n+1 corners) come out identical, both
    being the periodic seam -- correct for the periodic RVEs this project's
    FFT solvers assume.

    Parameters
    ----------
    arr  : (n0, n1, [n2,] ...trailing)   voxel-centered field, spatial axes first
    ndim : int   number of leading spatial axes to interpolate (2 or 3)

    Returns
    -------
    out : (n0+1, n1+1, [n2+1,] ...trailing)
    """
    out = arr
    for ax in range(ndim):
        pad_width = [(0, 0)] * out.ndim
        pad_width[ax] = (1, 1)
        ext = np.pad(out, pad_width, mode="wrap")

        n_ax = out.shape[ax]
        sl_lo = [slice(None)] * out.ndim
        sl_hi = [slice(None)] * out.ndim
        sl_lo[ax] = slice(0, n_ax + 1)
        sl_hi[ax] = slice(1, n_ax + 2)

        out = 0.5 * (ext[tuple(sl_lo)] + ext[tuple(sl_hi)])
    return out


def _check_field_shape(arr: np.ndarray, grid_shape: tuple[int, ...], name: str) -> None:
    if arr.shape[:len(grid_shape)] != grid_shape:
        raise ValueError(
            f"Field '{name}': leading dimensions {arr.shape[:len(grid_shape)]} "
            f"do not match grid_shape {grid_shape}"
        )


def _attribute_meta(arr_shape: tuple, grid_shape: tuple) -> tuple[str, int, str]:
    """Return (AttributeType, n_components, XDMF Dimensions string)."""
    extra = arr_shape[len(grid_shape):]

    if len(extra) == 0:
        # scalar
        dims_str = " ".join(str(s) for s in grid_shape)
        return "Scalar", 1, dims_str

    if len(extra) == 1:
        n = extra[0]
        dims_str = " ".join(str(s) for s in grid_shape) + f" {n}"
        attr_type = "Vector" if n in (2, 3) else "Tensor6" if n == 6 else "Tensor" if n == 9 else "Matrix"
        return attr_type, n, dims_str

    # 2-D component shape (e.g. stress tensor stored as nx,ny,nz,3,3)
    n_components = int(np.prod(extra))
    dims_str = " ".join(str(s) for s in grid_shape) + " " + " ".join(str(s) for s in extra)
    return "Tensor", n_components, dims_str
