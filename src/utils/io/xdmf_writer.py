"""
HDF5 + XDMF incremental writer for ParaView (open with Xdmf3ReaderT).

    with IncrementalWriter("results/sim", grid_shape=n, grid_length=L) as w:
        w.write_increment(0, {"stress": to_voigt(stress), "displacement": u},
                          point_fields={"displacement"})

Fields are cell-centered (one value per voxel). Names in ``point_fields`` are
written node-centered on the n+1 corner grid, by periodic averaging, which
ParaView's Warp By Vector needs.
"""

import utils.precision  # noqa: F401 -- side effect: configures JAX (X64 off on TPU, no GPU prealloc)

import os
import jax
import numpy as np
import h5py


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
    One .h5 and one .xdmf referencing it, appended one increment at a time.

    Parameters
    ----------
    base_path   : path without extension
    grid_shape  : voxels per dimension, 2-D or 3-D
    grid_length : domain size per dimension (spacing = length / shape)
    origin      : grid origin, default zeros
    """

    def __init__(
        self,
        base_path: str,
        grid_shape: tuple[int, ...],
        grid_length: tuple[float, ...],
        origin: tuple[float, ...] | None = None,
    ):
        if len(grid_shape) not in (2, 3):
            raise ValueError("grid_shape must be 2-D or 3-D")
        if len(grid_length) != len(grid_shape):
            raise ValueError("grid_length must have the same length as grid_shape")

        self.ndim = len(grid_shape)
        self.grid_shape = grid_shape
        self.grid_length = grid_length
        self.grid_spacing = tuple(Li / ni for Li, ni in zip(grid_length, grid_shape))
        self.origin = origin if origin is not None else tuple(0.0 for _ in grid_shape)

        os.makedirs(os.path.dirname(base_path) or ".", exist_ok=True)

        self.h5_path  = base_path + ".h5"
        self.xdmf_path = base_path + ".xdmf"
        self._h5_basename = os.path.basename(self.h5_path)

        self._h5 = h5py.File(self.h5_path, "w")
        self._increments: list[tuple[int, list[tuple[str, str]], float]] = []

    def write_increment(
        self,
        increment: int,
        fields: dict[str, np.ndarray | jax.Array],
        time: float | None = None,
        point_fields: set[str] | frozenset[str] | None = None,
    ) -> None:
        """
        Write one increment (and rewrite the XDMF, so it is always valid on disk).

        Parameters
        ----------
        increment    : increment index (HDF5 group name)
        fields       : {name: array}, cell-centered, shape grid_shape or
                       (*grid_shape, k); JAX arrays allowed
        time         : XDMF time value (default: the increment index)
        point_fields : names to write node-centered (see module docstring)
        """
        grp = self._h5.require_group(f"increment_{increment:06d}")
        written: list[tuple[str, str]] = []

        for name, arr in fields.items():
            arr = np.asarray(arr)
            _check_field_shape(arr, self.grid_shape, name)

            is_point = point_fields is not None and name in point_fields
            if is_point:
                arr = _cell_to_node(arr, self.ndim)

            # CoRectMesh dimensions are ZYX (X fastest): store spatial axes reversed.
            spatial = tuple(range(self.ndim - 1, -1, -1))
            trailing = tuple(range(self.ndim, arr.ndim))
            arr = arr.transpose(spatial + trailing)
            grp.create_dataset(name, data=arr, compression="gzip", compression_opts=4)
            written.append((name, "point" if is_point else "cell"))

        self._h5.flush()
        time_val = float(increment) if time is None else float(time)
        self._increments.append((increment, written, time_val))
        self._write_xdmf()

    def close(self) -> None:
        """Flush and close the HDF5 file. XDMF is already up to date."""
        self._h5.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

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
                if center == "point":
                    dims_grid = tuple(reversed(tuple(s + 1 for s in self.grid_shape)))
                else:
                    dims_grid = tuple(reversed(self.grid_shape))
                attr_type, n_components, dims_str = _attribute_meta(arr_shape, dims_grid)
                h5_ref = f"{self._h5_basename}:/{grp_path}/{fname}"

                # XDMF calls it "Node"; "Point" is ParaView's word and invalid here.
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
        # Dimensions count nodes (n+1 for n cells), listed ZYX.
        if self.ndim == 2:
            nx, ny = self.grid_shape
            return f'<Topology TopologyType="2DCoRectMesh" Dimensions="{ny + 1} {nx + 1}" />'
        else:
            nx, ny, nz = self.grid_shape
            return f'<Topology TopologyType="3DCoRectMesh" Dimensions="{nz + 1} {ny + 1} {nx + 1}" />'

    def _geometry_tag(self) -> str:
        # Nodes start at the origin (no half-voxel shift); values listed ZYX.
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


def _cell_to_node(arr: np.ndarray, ndim: int) -> np.ndarray:
    """
    Cell values (n per axis) to corner nodes (n+1) by periodic averaging of
    the touching voxels; nodes 0 and n coincide (the periodic seam).
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
        dims_str = " ".join(str(s) for s in grid_shape)
        return "Scalar", 1, dims_str

    if len(extra) == 1:
        n = extra[0]
        dims_str = " ".join(str(s) for s in grid_shape) + f" {n}"
        attr_type = "Vector" if n in (2, 3) else "Tensor6" if n == 6 else "Tensor" if n == 9 else "Matrix"
        return attr_type, n, dims_str

    n_components = int(np.prod(extra))
    dims_str = " ".join(str(s) for s in grid_shape) + " " + " ".join(str(s) for s in extra)
    return "Tensor", n_components, dims_str
