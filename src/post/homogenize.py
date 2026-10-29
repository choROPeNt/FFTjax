"""
Homogenize a solve script's .h5 (IncrementalWriter output) into one row per
increment via post.fields.macroscopic_response, the same function the scripts
call live.

Optional fields are included when present: ``strain_p`` and ``damage`` (mean
and max), and the plastic strain tensor ``eps_p`` with its trace (the plastic
volume change, the dilatancy signal alpha cannot show). The load parameter
and ``converged`` come from the run's ``<stem>_stats`` file if given.
"""

from pathlib import Path

import h5py
import numpy as np

from post.fields import from_voigt, macroscopic_response

_VOIGT_NAMES = ("11", "22", "33", "12", "13", "23")


def find_stats_path(stem: str | Path) -> Path:
    """``<stem>_stats.npy`` if present, else ``<stem>_stats.csv`` (which may not exist either)."""
    npy_path = Path(f"{stem}_stats.npy")
    return npy_path if npy_path.exists() else Path(f"{stem}_stats.csv")


def load_stats(stats_path: str | Path) -> dict[int, np.void] | None:
    """``{step: row}`` from a stats .npy or .csv, or None if missing or without a ``step`` field."""
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
    Per-increment (dtype, rows) table; pass to utils.io.stats.write_stats.

    Parameters
    ----------
    h5_path : the run's .h5
    stats   : ``{step: row}`` from load_stats, for the load parameter and
              ``converged`` columns

    Returns
    -------
    dtype : "step", load parameter and "converged" if available, "eps",
            "sigma" (Voigt), "mises_stress", "pressure", and the optional
            plastic/damage columns present in the run
    rows  : one tuple per increment
    """
    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as f:
        group_names = sorted(k for k in f.keys() if k.startswith("increment_"))
        if not group_names:
            raise ValueError(f"No increment_* groups found in {h5_path}")

        first = f[group_names[0]]
        has_plastic        = "strain_p" in first
        has_plastic_tensor = "eps_p"    in first
        has_damage         = "damage"   in first

        time_col = None
        has_converged = False
        if stats:
            sample_names = next(iter(stats.values())).dtype.names
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
                # Step 0 is written before the solve loop fills stats: t = 0.
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
                # Trace, not mean: the volume change dV/V (no division by 3).
                row.append(resp["eps_p_vol"])
            if has_damage:
                row += [resp["damage_mean"], resp["damage_max"]]

            rows.append(tuple(row))

    return dtype, rows
