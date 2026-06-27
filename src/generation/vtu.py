"""
VTU reader for TexGen-compatible voxel meshes.

read_texgen_vtu  — parse a TexGen (or externally generated) hexahedral VTU
"""

from __future__ import annotations

import xml.etree.ElementTree as ET  # used for VTU parsing
from pathlib import Path

import numpy as np


def read_texgen_vtu(
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

    Also accepts VTU files produced by ``generate_weave_vtu.py``; those
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
