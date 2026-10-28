"""
Homogenize per-voxel solver output (the ``increment_*`` groups of a
solve_*.py script's ``.h5``, written by utils.io.xdmf_writer.IncrementalWriter)
into one structured-array row per timestep: volume-averaged strain/stress
and their derived von Mises equivalent stress and hydrostatic pressure
(p = tr(sigma)/3, p > 0 = tension -- same sign convention as
materialmodels.inelastic.plasticity_drucker_prager), all via
post.fields.macroscopic_response -- the same function every solve_*.py
script itself calls live, per increment, to fill its own ``<stem>_stats``
file, so this post-hoc pass can't drift from the live one.

Per-voxel scalar state written by only some solvers -- accumulated plastic
strain (``strain_p``, solve_inelastic.py) and damage (``damage``,
solve_fracture.py) -- is included (mean and max over voxels) only when
actually present, so ``homogenized_table`` covers the output of all three
solve_*.py scripts unmodified, detected by field presence alone.

The plastic strain TENSOR (``eps_p``, Voigt 6) is homogenized component-wise
too, whenever the HDF5 carries it -- which it does only for a
solve_inelastic.py run configured with ``write_plastic_strain_tensor: true``.
Its trace is reported as ``plastic_vol_strain``, the macroscopic plastic
volume change: the dilatancy signal of a non-associated Drucker-Prager run,
which the scalar ``strain_p`` (alpha) cannot show, since alpha accumulates
the plastic multiplier alone.

Step metadata not carried by the HDF5 itself (the applied load parameter --
``t`` for solve_mechanics.py/solve_fracture.py, ``gamma`` for
solve_inelastic.py -- and the ``converged`` flag) is merged in from a
sibling ``<stem>_stats.npy``/``.csv`` (see ``find_stats_path``/``load_stats``)
when supplied.

Used by scripts/postprocessing/homogenize.py (the CLI); import directly for
the same table from a notebook or another script.
"""

from pathlib import Path

import h5py
import numpy as np

from post.fields import from_voigt, macroscopic_response

# Abaqus order, matches post.fields._VOIGT_IJ
_VOIGT_NAMES = ("11", "22", "33", "12", "13", "23")


def find_stats_path(stem: str | Path) -> Path:
    """
    ``<stem>_stats.npy`` (utils.io.stats.write_stats' default) if present,
    else ``<stem>_stats.csv`` (a run written with a ``stats: ..csv`` config
    key) -- the returned path may not exist either; check before use.
    """
    npy_path = Path(f"{stem}_stats.npy")
    return npy_path if npy_path.exists() else Path(f"{stem}_stats.csv")


def load_stats(stats_path: str | Path) -> dict[int, np.void] | None:
    """
    ``{step: structured-array row}``, or None if ``stats_path`` doesn't
    exist (or has no ``step`` field). Reads either format
    utils.io.stats.write_stats can produce: the ``.npy`` structured array
    as-is, or the flattened ``.csv`` via genfromtxt -- scalar column names
    (``step``, ``t``/``gamma``, ``converged``, the only ones
    ``homogenized_table`` reads back off ``stats``) are identical either
    way, only subarray fields (unused here) are flattened/renamed in the CSV.
    """
    stats_path = Path(stats_path)
    if not stats_path.exists():
        return None
    if stats_path.suffix == ".csv":
        rows = np.genfromtxt(stats_path, delimiter=",", names=True)
        rows = rows.reshape(1) if rows.shape == () else rows
    else:
        rows = np.load(stats_path)
    if "step" not in rows.dtype.names:
        return None
    return {int(row["step"]): row for row in rows}


def homogenized_table(
    h5_path: str | Path, stats: dict[int, np.void] | None = None,
) -> tuple[np.dtype, list[tuple]]:
    """
    Build the per-timestep (dtype, rows) table described in the module
    docstring -- pass straight to utils.io.stats.write_stats to save it as
    ``.npy``/``.csv``.

    Parameters
    ----------
    h5_path : the solve_*.py script's ``.h5``
    stats   : optional ``{step: row}`` map from ``load_stats``, supplying
              the load parameter (``t``/``gamma``) and ``converged``
              columns that the HDF5 itself doesn't carry

    Returns
    -------
    dtype : structured dtype -- "step", optionally the load-parameter
            column and "converged", "eps"/"sigma" (Voigt 6 each),
            "mises_stress", "pressure", plus "plastic_strain_mean"/"_max",
            "eps_p" (Voigt 6)/"plastic_vol_strain", "damage_mean"/"_max"
            wherever the run actually wrote that state.
    rows  : list of tuples, one per timestep, matching dtype
    """
    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as f:
        group_names = sorted(k for k in f.keys() if k.startswith("increment_"))
        if not group_names:
            raise ValueError(f"No increment_* groups found in {h5_path}")

        first = f[group_names[0]]
        has_plastic        = "strain_p" in first
        has_plastic_tensor = "eps_p"    in first   # opt-in, see module docstring
        has_damage         = "damage"   in first

        time_col = None
        has_converged = False
        if stats:
            sample_names = next(iter(stats.values())).dtype.names
            # "t" (solve_mechanics.py/solve_fracture.py, load fraction in
            # [0, 1]) or "gamma" (solve_inelastic.py, the driven shear/
            # normal strain).
            time_col = next((c for c in ("t", "gamma") if c in sample_names), None)
            has_converged = "converged" in sample_names

        fields: list = [("step", "i4")]
        if time_col:
            fields.append((time_col, "f8"))
        if has_converged:
            fields.append(("converged", "?"))
        fields += [("eps", "f8", (6,)), ("sigma", "f8", (6,)), ("mises_stress", "f8"), ("pressure", "f8")]
        if has_plastic:
            fields += [("plastic_strain_mean", "f8"), ("plastic_strain_max", "f8")]
        if has_plastic_tensor:
            fields += [("eps_p", "f8", (6,)), ("plastic_vol_strain", "f8")]
        if has_damage:
            fields += [("damage_mean", "f8"), ("damage_max", "f8")]
        dtype = np.dtype(fields)

        rows = []
        for gname in group_names:
            step = int(gname.split("_")[1])
            grp = f[gname]

            # mean-then-to_voigt and to_voigt-then-mean agree (linear map), so
            # reconstructing (3, 3) tensors here to hand to
            # post.fields.macroscopic_response -- see module docstring.
            eps_bar   = from_voigt(grp["strain"][...].reshape(-1, 6).mean(axis=0))
            sigma_bar = from_voigt(grp["stress"][...].reshape(-1, 6).mean(axis=0))

            scalars = {}
            if has_plastic:
                scalars["plastic_strain"] = grp["strain_p"][...]
            if has_damage:
                scalars["damage"] = grp["damage"][...]
            tensors = None
            if has_plastic_tensor:
                tensors = {"eps_p": from_voigt(grp["eps_p"][...].reshape(-1, 6).mean(axis=0))}

            resp = macroscopic_response(eps_bar, sigma_bar, scalars=scalars, tensors=tensors)

            stats_row = stats.get(step) if stats else None
            row: list = [step]
            if time_col:
                # step 0 is often written by the calling script before the
                # solve loop that fills stats (solve_mechanics.py /
                # solve_fracture.py) -- undeformed virgin state, t = 0.
                row.append(float(stats_row[time_col]) if stats_row is not None else 0.0)
            if has_converged:
                row.append(bool(stats_row["converged"]) if stats_row is not None else True)
            row.append(resp["eps_bar"])
            row.append(resp["sigma_bar"])
            row += [resp["mises_stress"], resp["pressure"]]

            if has_plastic:
                row += [resp["plastic_strain_mean"], resp["plastic_strain_max"]]
            if has_plastic_tensor:
                row.append(resp["eps_p_bar"])
                # tr(eps_p_bar) -- the plastic volume change (dV/V), not the
                # mean normal strain, so no division by 3 (unlike `pressure`).
                row.append(resp["eps_p_vol"])
            if has_damage:
                row += [resp["damage_mean"], resp["damage_max"]]

            rows.append(tuple(row))

    return dtype, rows
