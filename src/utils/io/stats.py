"""
Per-increment solver stats, one row per accepted increment, written as a
structured .npy or a flat CSV depending on the file extension (same rows).
"""

import csv
import io
from pathlib import Path
from typing import cast

import numpy as np

_VOIGT_NAMES = ("11", "22", "33", "12", "13", "23")


def flatten_structured(array: np.ndarray) -> np.ndarray:
    """
    Structured array with subarray fields flattened to scalar columns: Voigt-6
    fields get ``_11 .. _23`` suffixes, others ``_0 .. _n``. The one place the
    column naming lives, shared by .npy and .csv readers.
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
    Write ``rows`` (matching ``dtype``) as .npy (structured, subarrays kept)
    or .csv (flattened, bools as 0/1), chosen by the extension.

    ``comment`` lines go above the CSV header with a ``#`` prefix (dropped for
    .npy). A comment containing commas breaks ``genfromtxt(comments="#")``;
    use ``load_table``.

    Returns
    -------
    Path(path)
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
    .csv or .npy table as a flat structured array; the counterpart of write_stats.

    Unnamed .npy columns become ``col0, col1, ...``. CSV comment lines are
    stripped by hand: genfromtxt miscounts columns when a comment has commas.
    """
    path = Path(path)
    if path.suffix == ".npy":
        arr = np.load(path)
        if arr.dtype.names is None:
            arr = np.atleast_2d(arr)
            dtype = np.dtype([(f"col{i}", arr.dtype) for i in range(arr.shape[1])])
            named = np.empty(arr.shape[0], dtype=dtype)
            for i, name in enumerate(cast(tuple, dtype.names)):
                named[name] = arr[:, i]
            arr = named
        return flatten_structured(arr)
    elif path.suffix == ".csv":
        with open(path) as fh:
            body = "".join(line for line in fh if not line.lstrip().startswith("#"))
        data = np.genfromtxt(io.StringIO(body), delimiter=",", names=True)
        return data.reshape(1) if data.shape == () else data
    else:
        raise ValueError(f"table path must end in .npy or .csv, got {path!r}")
