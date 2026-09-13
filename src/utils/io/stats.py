"""
Per-increment solver stats: written by every solve_*.py script as one
structured-array row per accepted increment (step index, load parameter,
convergence flags, the homogenized macroscopic response from
post.fields.macroscopic_response, ...), as either a numpy structured array
(``.npy``, compact and ``np.load``-able) or a flat, human-readable CSV --
same rows either way, format chosen purely by the destination path's
extension (the optional ``stats:`` key in each script's YAML config, default
``<jobname>_stats.npy``), so switching format never changes what's recorded,
only how it's stored.
"""

import csv
import io
from pathlib import Path
from typing import cast

import numpy as np

# Abaqus order, matches post.fields._VOIGT_IJ / scripts/postprocessing/homogenize.py
_VOIGT_NAMES = ("11", "22", "33", "12", "13", "23")


def flatten_structured(array: np.ndarray) -> np.ndarray:
    """
    Flatten a structured array's subarray fields (e.g. a Voigt-6 tensor
    field, shape (6,), such as a raw ``_stats.npy``'s ``eps_bar_voigt``)
    into plain scalar fields -- a Voigt-6 field becomes six
    ``_11/_22/_33/_12/_13/_23``-suffixed columns (Abaqus order, matching
    post.fields' own convention), any other subarray field becomes plain
    ``_0.._n``-suffixed columns, and scalar fields pass through unchanged.
    The single place this naming lives, so anything reading a raw structured
    ``.npy`` (subarrays intact) and anything reading the already-flat
    ``.csv`` (``write_stats`` uses this internally for that) agree on what
    to call each column -- e.g. scripts/postprocessing/plot_homogenized.py,
    which needs per-column access to plot an arbitrary stats table.

    Parameters
    ----------
    array : any structured numpy array, e.g. from ``np.load`` on a
            ``write_stats(..., "*.npy")`` file

    Returns
    -------
    A new structured array with only scalar fields, same length as ``array``.
    """
    dtype = array.dtype
    flat_fields: list[tuple] = []
    for name in dtype.names:
        field_dtype = dtype.fields[name][0]
        shape = field_dtype.shape
        if shape == ():
            flat_fields.append((name, field_dtype))
        elif shape == (6,):
            flat_fields += [(f"{name}_{c}", field_dtype.base) for c in _VOIGT_NAMES]
        else:
            flat_fields += [(f"{name}_{i}", field_dtype.base) for i in range(shape[0])]
    flat_dtype = np.dtype(flat_fields)

    flat = np.empty(array.shape, dtype=flat_dtype)
    for name in dtype.names:
        shape = dtype.fields[name][0].shape
        if shape == ():
            flat[name] = array[name]
        elif shape == (6,):
            for i, c in enumerate(_VOIGT_NAMES):
                flat[f"{name}_{c}"] = array[name][..., i]
        else:
            for i in range(shape[0]):
                flat[f"{name}_{i}"] = array[name][..., i]
    return flat


def write_stats(
    path: str | Path, dtype: np.dtype, rows: list[tuple], *, comment: str | None = None,
) -> Path:
    """
    Write ``rows`` (one tuple per accepted increment, matching ``dtype``) to
    ``path`` -- the format is whichever its extension names, ``.npy`` or
    ``.csv``, so a config's ``stats: myjob_stats.csv`` is all it takes to
    switch, no separate format setting to keep in sync with the filename.

    ``.npy`` saves a numpy structured array exactly as every solve_*.py
    script always has -- ``np.load(path)`` to read it back, one row per
    increment, subarray fields (e.g. ``eps_bar_voigt``, shape (6,)) intact.

    ``.csv`` flattens each row into plain columns: a Voigt-6 subarray field
    (shape (6,)) becomes six ``_11/_22/_33/_12/_13/_23``-suffixed columns
    (Abaqus order, matching post.fields' own convention), any other
    subarray field becomes plain ``_0.._n``-suffixed columns, and a bool
    field is written as 0/1 -- readable directly with
    ``pandas.read_csv``/``numpy.genfromtxt(..., names=True)``.

    Parameters
    ----------
    path    : destination, e.g. "output/simulation/myjob_stats.npy" -- suffix
              must be ".npy" or ".csv"
    dtype   : structured dtype describing one row
    rows    : list of tuples, one per accepted increment, matching dtype
    comment : optional free-text banner (e.g. job provenance/notation notes)
              written as ``#``-prefixed lines above the header -- CSV only,
              silently dropped for ``.npy`` (a structured array can't carry
              free text). A comment line that itself embeds a comma (e.g. a
              material's ``repr()``) needs manual ``#``-stripping to read
              back rather than ``genfromtxt(..., comments="#")``, which
              mis-splits it -- see scripts/postprocessing/plot_homogenized.py's
              own loader for that workaround.

    Returns
    -------
    ``Path(path)``, for a convenient one-line "Written -> ..." print
    """
    path = Path(path)
    if path.suffix == ".npy":
        np.save(path, np.array(rows, dtype=dtype))
    elif path.suffix == ".csv":
        flat = flatten_structured(np.array(rows, dtype=dtype))
        with open(path, "w", newline="") as fh:
            if comment:
                for line in comment.splitlines():
                    fh.write(f"# {line}\n" if line else "#\n")
            writer = csv.writer(fh)
            writer.writerow(flat.dtype.names)
            for row in flat:
                writer.writerow(
                    f"{v:.8e}" if isinstance(v, (float, np.floating)) else
                    (int(v) if isinstance(v, (bool, np.bool_)) else v)
                    for v in row
                )
    else:
        raise ValueError(f"stats path must end in .npy or .csv, got {path!r}")
    return path


def load_table(path: str | Path) -> np.ndarray:
    """
    Load a ``.csv`` or ``.npy`` table as a flat structured array -- the
    read-side counterpart to ``write_stats``, so either format it wrote (or
    a plain csv/npy table from elsewhere) reads back the same way: one
    structured array, one scalar field per column, no subarrays.

    ``.npy`` is loaded via ``np.load``; a structured array with subarray
    fields (e.g. a raw ``write_stats(..., "*.npy")``'s ``eps_bar_voigt``,
    shape (6,)) is flattened via ``flatten_structured``, and a plain,
    unnamed array gets synthesized ``"col0"``, ``"col1"``, ... column names
    first, so the result is always named and flat either way.

    ``.csv`` is parsed with ``genfromtxt(names=True)``, skipping any
    ``#``-prefixed comment lines by hand -- harmless when there are none,
    and necessary for a banner that embeds commas (e.g. a material's
    ``repr()``, as scripts/postprocessing/homogenize.py's CSV does), which
    genfromtxt's own comment stripping mishandles, miscounting the column
    width for every data row.

    Parameters
    ----------
    path : ".csv" or ".npy" file

    Returns
    -------
    Flat structured array (only scalar fields, see ``flatten_structured``)
    """
    path = Path(path)
    if path.suffix == ".npy":
        arr = np.load(path)
        if arr.dtype.names is None:
            arr = np.atleast_2d(arr)
            dtype = np.dtype([(f"col{i}", arr.dtype) for i in range(arr.shape[1])])
            named = np.empty(arr.shape[0], dtype=dtype)
            # cast: dtype.names is None only for a non-structured dtype --
            # dtype was just built above from a list of (name, ...) tuples,
            # so it's always structured here.
            for i, name in enumerate(cast(tuple, dtype.names)):
                named[name] = arr[:, i]
            arr = named
        return flatten_structured(arr)
    elif path.suffix == ".csv":
        with open(path) as fh:
            body = "".join(line for line in fh if not line.lstrip().startswith("#"))
        data = np.genfromtxt(io.StringIO(body), delimiter=",", names=True)
        return data.reshape(1) if data.shape == () else data  # a single row collapses to 0-D
    else:
        raise ValueError(f"table path must end in .npy or .csv, got {path!r}")
