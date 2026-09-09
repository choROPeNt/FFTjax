"""
Homogenize per-voxel solver output into a human-readable per-timestep table.

Reads the ``increment_*`` groups of a ``<stem>.h5`` written by
utils.io.xdmf_writer.IncrementalWriter (via solve_mechanics.py /
solve_fracture.py / solve_inelastic.py) and volume-averages ("homogenizes")
the strain and stress fields over all voxels at every timestep -- exact for a
periodic, equal-volume voxel grid, same guarantee as post.fields.homogenize.
The homogenized von Mises equivalent stress and hydrostatic pressure
(p = tr(sigma)/3, p > 0 = tension -- same sign convention as
materialmodels.inelastic.plasticity_drucker_prager) are then derived from the
homogenized stress tensor. Per-voxel scalar state written by only some
solvers -- accumulated plastic strain (``strain_p``, solve_inelastic.py) and
damage (``damage``, solve_fracture.py) -- is included (mean and max over
voxels) only when actually present, so this one script covers the output of
all three solve_*.py scripts unmodified.

The plastic strain TENSOR (``eps_p``, Voigt 6) is homogenized component-wise
too, whenever the HDF5 carries it -- which it does only for a
solve_inelastic.py run configured with ``write_plastic_strain_tensor: true``,
that field being opt-in. Its trace is reported as ``plastic_vol_strain``, the
macroscopic plastic volume change: the dilatancy signal of a non-associated
Drucker-Prager run, which the scalar ``strain_p`` (alpha) cannot show, since
alpha accumulates the plastic multiplier alone. Detection is by field
presence, so no flag has to be repeated here.

Step metadata not carried by the HDF5 itself (the applied load parameter --
``t`` for solve_mechanics.py/solve_fracture.py, ``gamma`` for
solve_inelastic.py -- and the ``converged`` flag) is merged in from the
sibling ``<stem>_stats.npy`` when it is found next to the ``.h5``.

Usage
-----
    python scripts/postprocessing/homogenize.py output/simulation/<jobname>
    python scripts/postprocessing/homogenize.py output/simulation/<jobname>.h5 -o results.csv

Output
------
    <stem>_homogenized.csv   -- one row per timestep, comma-separated, plain
        text (``#``-prefixed job metadata, then a header row):
        step, [t|gamma], [converged],
        eps_11, eps_22, eps_33, eps_12, eps_13, eps_23,
        sigma_11, sigma_22, sigma_33, sigma_12, sigma_13, sigma_23,
        mises_stress, pressure,
        [plastic_strain_mean, plastic_strain_max],
        [eps_p_11, eps_p_22, eps_p_33, eps_p_12, eps_p_13, eps_p_23,
         plastic_vol_strain],
        [damage_mean, damage_max]
    Read it back with scripts/postprocessing/plot_homogenized.py, or any CSV
    reader that supports a comment prefix, e.g.
    ``pandas.read_csv(path, comment="#")`` -- plain
    ``numpy.genfromtxt(..., comments="#")`` mis-splits the metadata lines
    that embed commas (a material's ``repr()``), see plot_homogenized.py's
    own loader for a workaround.
"""

import argparse
import csv
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, "src")

from post.fields import from_voigt, von_mises  # noqa: E402

_VOIGT_NAMES = ("11", "22", "33", "12", "13", "23")  # Abaqus order, matches post.fields._VOIGT_IJ


def _stem(job: Path) -> Path:
    """Strip a known suffix (.h5 / _stats.npy / .xdmf) off a job path, if present."""
    s = str(job)
    for suffix in ("_stats.npy", ".h5", ".xdmf"):
        if s.endswith(suffix):
            return Path(s[: -len(suffix)])
    return job


def _load_stats(stats_path: Path) -> dict[int, np.void] | None:
    """``{step: structured-array row}``, or None if the file doesn't exist."""
    if not stats_path.exists():
        return None
    rows = np.load(stats_path)
    if "step" not in rows.dtype.names:
        return None
    return {int(row["step"]): row for row in rows}


def homogenize(h5_path: Path, stats: dict[int, np.void] | None) -> tuple[list[str], list[list]]:
    """Build the (header, rows) table described in the module docstring."""
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
            time_col = "gamma" if "gamma" in sample_names else ("t" if "t" in sample_names else None)
            has_converged = "converged" in sample_names

        header = ["step"]
        if time_col:
            header.append(time_col)
        if has_converged:
            header.append("converged")
        header += [f"eps_{c}" for c in _VOIGT_NAMES]
        header += [f"sigma_{c}" for c in _VOIGT_NAMES]
        header += ["mises_stress", "pressure"]
        if has_plastic:
            header += ["plastic_strain_mean", "plastic_strain_max"]
        if has_plastic_tensor:
            header += [f"eps_p_{c}" for c in _VOIGT_NAMES] + ["plastic_vol_strain"]
        if has_damage:
            header += ["damage_mean", "damage_max"]

        rows = []
        for gname in group_names:
            step = int(gname.split("_")[1])
            grp = f[gname]

            eps_bar   = grp["strain"][...].reshape(-1, 6).mean(axis=0)
            sigma_bar = grp["stress"][...].reshape(-1, 6).mean(axis=0)
            mises     = float(von_mises(from_voigt(sigma_bar)))
            pressure  = float((sigma_bar[0] + sigma_bar[1] + sigma_bar[2]) / 3.0)

            stats_row = stats.get(step) if stats else None
            row: list = [step]
            if time_col:
                # step 0 is often written by the calling script before the
                # solve loop that fills stats.npy (solve_mechanics.py /
                # solve_fracture.py) -- undeformed virgin state, t = 0.
                row.append(float(stats_row[time_col]) if stats_row is not None else 0.0)
            if has_converged:
                row.append(bool(stats_row["converged"]) if stats_row is not None else True)
            row += [float(v) for v in eps_bar]
            row += [float(v) for v in sigma_bar]
            row += [mises, pressure]

            if has_plastic:
                sp = grp["strain_p"][...]
                row += [float(sp.mean()), float(sp.max())]
            if has_plastic_tensor:
                eps_p_bar = grp["eps_p"][...].reshape(-1, 6).mean(axis=0)
                row += [float(v) for v in eps_p_bar]
                # tr(eps_p_bar) -- the plastic volume change (dV/V), not the
                # mean normal strain, so no division by 3 (unlike `pressure`).
                row.append(float(eps_p_bar[0] + eps_p_bar[1] + eps_p_bar[2]))
            if has_damage:
                d = grp["damage"][...]
                row += [float(d.mean()), float(d.max())]

            rows.append(row)

    return header, rows


def write_csv(path: Path, header: list[str], rows: list[list], h5_path: Path) -> None:
    with h5py.File(h5_path, "r") as f:
        attrs = dict(f.attrs)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        fh.write(f"# Homogenized (volume-averaged) response, generated by "
                  f"scripts/postprocessing/homogenize.py from {h5_path}\n")
        if "input" in attrs:
            fh.write(f"# input: {attrs['input']}\n")
        if "n" in attrs and "L" in attrs:
            fh.write(f"# grid n: {tuple(int(v) for v in attrs['n'])}   "
                      f"L: {tuple(float(v) for v in attrs['L'])}\n")
        if "material_repr" in attrs:
            for m in attrs["material_repr"]:
                fh.write(f"# material: {m}\n")
        fh.write("# pressure = tr(sigma_bar)/3, p > 0 = hydrostatic tension\n")
        if "eps_p_11" in header:
            fh.write("# plastic_strain_* = scalar hardening variable alpha; eps_p_* = plastic "
                      "strain tensor (Voigt); plastic_vol_strain = tr(eps_p_bar) = plastic "
                      "volume change\n")

        writer = csv.writer(fh)
        writer.writerow(header)
        for row in rows:
            writer.writerow(f"{v:.8e}" if isinstance(v, float) else
                             (int(v) if isinstance(v, bool) else v)
                             for v in row)


def main():
    parser = argparse.ArgumentParser(
        description="Homogenize per-voxel solve_mechanics.py/solve_fracture.py/"
                    "solve_inelastic.py field output into a human-readable "
                    "per-timestep CSV of volume-averaged strain, stress, von "
                    "Mises stress, hydrostatic pressure, and (where written) "
                    "plastic strain / damage."
    )
    parser.add_argument("job", type=Path,
                        help="<stem>.h5, <stem>_stats.npy, or the bare <stem> "
                             "written by one of the solve_*.py scripts")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Output CSV path (default: <stem>_homogenized.csv)")
    args = parser.parse_args()

    stem = _stem(args.job)
    h5_path = Path(str(stem) + ".h5")
    if not h5_path.exists():
        raise FileNotFoundError(f"{h5_path} not found")
    stats_path = Path(str(stem) + "_stats.npy")

    stats = _load_stats(stats_path)
    if stats is None:
        print(f"Note: {stats_path} not found (or has no 'step' field) -- "
              f"output will have no t/gamma/converged columns.")

    header, rows = homogenize(h5_path, stats)

    out_path = args.output if args.output else Path(str(stem) + "_homogenized.csv")
    write_csv(out_path, header, rows, h5_path)

    print(f"Read   ← {h5_path}   ({len(rows)} timesteps)")
    print(f"Written → {out_path}")
    print(f"columns: {header}")


if __name__ == "__main__":
    main()
