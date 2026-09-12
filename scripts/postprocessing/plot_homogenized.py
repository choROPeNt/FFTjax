"""
Visualize one or more homogenized-response CSVs written by
scripts/postprocessing/homogenize.py.

Pass several CSVs to overlay them: every figure then carries one curve per
run, which is how two boundary conditions, two hardening laws or two
microstructures get compared without leaving the tool. One CSV produces
exactly the figures it always did.

Produces, in --outdir:
    stress_strain.png     -- sigma vs eps for one tensor component (the one
                             with the largest strain range, unless --component
                             is given), e.g. the tau-gamma hysteresis loop for
                             a solve_inelastic.py load/unload/reload cycle.
    stress_path.png       -- the (p, q) stress path, with the VIRGIN yield
                             surface drawn whenever the CSV metadata names a
                             Drucker-Prager phase. The diagnostic figure for a
                             pressure-sensitive material: it shows the
                             direction the path takes relative to the cone,
                             which is what a confining boundary condition
                             changes and a stress-vs-strain plot hides.
    components.png        -- all six strain and all six stress components vs
                             the load parameter, two stacked panels. With more
                             than one CSV this becomes components_strain.png
                             and components_stress.png, a 2x3 panel per Voigt
                             component with one curve per run -- colour cannot
                             mean "component" and "run" at the same time.
    pressure.png          -- hydrostatic pressure vs the load parameter.
    hardening.png         -- von Mises stress vs mean plastic strain, only
                             if the CSV has plastic_strain_mean (solve_inelastic.py).
    plastic_activity.png  -- mean and peak accumulated plastic strain vs the
                             load parameter, two panels. Same inputs as
                             hardening.png, different question: how much
                             plastic flow a run provokes and how much of it
                             localizes (the mean can fall while the peak
                             rises).
    plastic_strain.png    -- the six homogenized plastic strain tensor
                             components (2x3 panels when comparing), only if
                             the CSV has plastic_vol_strain, i.e. the run had
                             write_plastic_strain_tensor on.
    plastic_volume.png    -- the trace of that tensor, the macroscopic plastic
                             volume change -- the dilatancy signal of a
                             non-associated Drucker-Prager run. Same condition.
    damage.png            -- max/mean damage vs the load parameter, only if
                             the CSV has damage_max (solve_fracture.py).

A figure whose column is missing from EVERY run is skipped; a run missing a
column the others have is simply left out of that one figure, so a plastic run
and an elastic one can still be compared on the figures they share.

The x-axis is each CSV's ``gamma`` or ``t`` column (the applied load
parameter) when present, else the plain step index.

Usage
-----
    python scripts/postprocessing/plot_homogenized.py output/simulation/<jobname>_homogenized.csv
    python scripts/postprocessing/plot_homogenized.py a_homogenized.csv b_homogenized.csv
    python scripts/postprocessing/plot_homogenized.py a.csv b.csv --labels confined uniaxial -o cmp/
    python scripts/postprocessing/plot_homogenized.py results.csv --component 12 --show
"""

import argparse
import io
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

_VOIGT_NAMES = ("11", "22", "33", "12", "13", "23")  # matches homogenize.py / post.fields._VOIGT_IJ

# Categorical palette, assigned to runs in this fixed order and never cycled:
# a 9th run would need a colour indistinguishable from an earlier one under
# colour-vision deficiency, so it is rejected rather than silently drawn.
_RUN_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#4a3aa7", "#e34948")


@dataclass
class Run:
    """One homogenized CSV: its data, the label it is drawn under, its path."""
    data: np.ndarray
    label: str
    path: Path

    def has(self, *columns: str) -> bool:
        return all(c in self.data.dtype.names for c in columns)


def _load(csv_path: Path) -> np.ndarray:
    """
    Load a homogenize.py CSV as a structured array.

    Filters out ``#``-prefixed metadata lines by hand rather than relying on
    ``genfromtxt(comments="#")``: some of those lines embed commas (e.g. a
    material's ``repr()``), which genfromtxt's comment stripping mishandles,
    miscounting the column width for every data row.
    """
    with open(csv_path) as fh:
        body = "".join(line for line in fh if not line.lstrip().startswith("#"))
    data = np.genfromtxt(io.StringIO(body), delimiter=",", names=True)
    return data.reshape(1) if data.shape == () else data  # a single data row collapses to 0-D


def _yield_surface(csv_path: Path) -> tuple[float, float, float] | None:
    """
    ``(a_f, a_tip, sigma_y0)`` of the first Drucker-Prager phase named in the
    CSV's ``# material:`` metadata, or None if there is none to find.

    Read off the material's ``repr()`` rather than re-read from a config,
    because the CSV is self-contained and may be the only artefact left of a
    run. Best-effort by construction: a repr that does not match returns None
    and the caller simply omits the surface, which is why every group here is
    optional and nothing raises.
    """
    with open(csv_path) as fh:
        for line in fh:
            if not line.lstrip().startswith("#"):
                break
            if "DruckerPrager" not in line:
                continue
            a_f = re.search(r"a_f=([\d.eE+-]+)", line)
            if not a_f:
                continue
            a_tip = re.search(r"a_tip=([\d.eE+-]+)", line)
            # linear law spells sigma_y0 out; a table's first row IS sigma_y0
            sig = (re.search(r"sigma_y0=([\d.eE+-]+)", line)
                   or re.search(r"pts:\s*\(\s*0[\d.]*\s*,\s*([\d.eE+-]+)\s*\)", line))
            if not sig:
                continue
            return float(a_f.group(1)), float(a_tip.group(1)) if a_tip else 0.0, float(sig.group(1))
    return None


def _xaxis(data: np.ndarray) -> tuple[np.ndarray, str]:
    names = data.dtype.names
    if "gamma" in names:
        return data["gamma"], r"applied load parameter $\bar\gamma$"
    if "t" in names:
        return data["t"], "load fraction $t$"
    return data["step"], "step"


def _pick_component(runs: list[Run], override: str | None) -> int:
    if override is not None:
        if override not in _VOIGT_NAMES:
            raise ValueError(f"--component must be one of {_VOIGT_NAMES}, got {override!r}")
        return _VOIGT_NAMES.index(override)
    # default: the component with the largest strain excursion ACROSS the runs
    # being compared, i.e. the one actually being driven -- works for shear
    # (solve_inelastic.py's default [0,1]) or normal (uniaxial) loading alike.
    ranges = [max(np.ptp(r.data[f"eps_{c}"]) for r in runs) for c in _VOIGT_NAMES]
    return int(np.argmax(ranges))


def _labels(paths: list[Path], override: list[str] | None) -> list[str]:
    """
    Short, distinct labels for the legend: whatever part of the filename
    actually differs between the runs, with the shared ``_homogenized`` suffix
    and the common prefix trimmed off. Falls back to full stems if trimming
    would make two runs collide.
    """
    if override:
        if len(override) != len(paths):
            raise ValueError(f"--labels: got {len(override)} for {len(paths)} CSVs")
        return override
    stems = [p.stem[: -len("_homogenized")] if p.stem.endswith("_homogenized") else p.stem
             for p in paths]
    if len(stems) == 1:
        return stems
    pre = os.path.commonprefix(stems)
    pre = pre[: pre.rfind("_") + 1] if "_" in pre else ""
    trimmed = [s[len(pre):] or s for s in stems]
    return trimmed if len(set(trimmed)) == len(trimmed) else stems


def _style(ax, runs_drawn: int):
    """A legend only once there is something to tell apart."""
    if runs_drawn > 1:
        ax.legend(fontsize=8)
    return ax


def _with(runs: list[Run], *columns: str) -> list[Run]:
    return [r for r in runs if r.has(*columns)]


def plot_stress_strain(runs: list[Run], comp: int, outdir: Path) -> Path:
    from utils.plotting import ResponseCurve, plot_homogenized_response

    c = _VOIGT_NAMES[comp]
    curves = [ResponseCurve(x=r.data[f"eps_{c}"], y=r.data[f"sigma_{c}"], marker="o-",
                            label=r.label if len(runs) > 1 else None) for r in runs]
    ax = plot_homogenized_response(
        curves, include_origin=False,
        xlabel=rf"$\bar\varepsilon_{{{c}}}$", ylabel=rf"$\bar\sigma_{{{c}}}$ [MPa]",
        title=f"Homogenized stress-strain response, component {c}",
    )
    path = outdir / "stress_strain.png"
    ax.figure.savefig(path, dpi=150, bbox_inches="tight")
    return path


def plot_stress_path(runs: list[Run], outdir: Path) -> Path:
    """
    The (p, q) stress path, with the virgin Drucker-Prager surface when the
    metadata supplies one.

    The surface is drawn as a REFERENCE for direction, not as a bound: yield
    is a per-voxel condition, and the homogenized point averages over elastic
    fibres and matrix voxels at different hardening states, so it legitimately
    sits outside the virgin surface. What the figure is for is the ANGLE -- a
    confined (pure-strain) BC drives q/|p| toward zero, straight at the cone's
    tensile vertex, where a stress-vs-strain plot shows nothing unusual.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for r, color in zip(runs, _RUN_COLORS):
        ax.plot(r.data["pressure"], r.data["mises_stress"], "o-", markersize=3,
                color=color, label=r.label if len(runs) > 1 else None)

    surf = _yield_surface(runs[0].path)
    if surf is not None:
        a_f, a_tip, sig0 = surf
        p_all = np.concatenate([r.data["pressure"] for r in runs])
        q_max = max(np.max(r.data["mises_stress"]) for r in runs) * 1.15
        if a_f > 0.0:
            p = np.linspace(min(p_all.min() * 1.1, 0.0), (sig0 - a_tip) / (3.0 * a_f), 400)
            q = np.sqrt(np.maximum((sig0 - 3.0 * a_f * p) ** 2 - a_tip ** 2, 0.0))
        else:                      # von Mises: a cylinder, flat in p
            p = np.linspace(p_all.min() * 1.1, p_all.max() * 1.1, 2)
            q = np.full_like(p, sig0)
        keep = q <= q_max
        if keep.any():
            ax.plot(p[keep], q[keep], "-", color="#898781", linewidth=1.1,
                    label=rf"virgin surface ($\sigma_y$={sig0:g})")
        ax.set_ylim(0.0, q_max)

    ax.set_xlabel(r"$\bar p$ [MPa]  ($p>0$ = tension)")
    ax.set_ylabel(r"$\bar q$ [MPa]")
    ax.set_title("Homogenized stress path")
    if len(runs) > 1 or surf is not None:
        ax.legend(fontsize=8)
    fig.tight_layout()
    path = outdir / "stress_path.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def plot_components(runs: list[Run], outdir: Path) -> list[Path]:
    """
    One run: the historical two-panel figure, six components per panel, keyed
    by colour. Several runs: a 2x3 grid per quantity instead, one panel per
    Voigt component with one curve per run -- because a single colour axis
    cannot carry "which component" and "which run" simultaneously.
    """
    import matplotlib.pyplot as plt

    if len(runs) == 1:
        data = runs[0].data
        x, xlabel = _xaxis(data)
        fig, (ax_eps, ax_sig) = plt.subplots(2, 1, figsize=(6.5, 7), sharex=True)
        for c in _VOIGT_NAMES:
            ax_eps.plot(x, data[f"eps_{c}"], "o-", markersize=3, label=c)
            ax_sig.plot(x, data[f"sigma_{c}"], "o-", markersize=3, label=c)
        ax_eps.set_ylabel(r"$\bar\varepsilon$")
        ax_sig.set_ylabel(r"$\bar\sigma$ [MPa]")
        ax_sig.set_xlabel(xlabel)
        ax_eps.legend(ncol=6, fontsize=8, loc="upper left")
        ax_sig.legend(ncol=6, fontsize=8, loc="upper left")
        ax_eps.set_title("Homogenized strain and stress components")
        fig.tight_layout()
        path = outdir / "components.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        return [path]

    return [_component_grid(runs, "eps", r"$\bar\varepsilon$",
                            "Homogenized strain components", outdir, "components_strain.png"),
            _component_grid(runs, "sigma", r"$\bar\sigma$ [MPa]",
                            "Homogenized stress components", outdir, "components_stress.png")]


def _component_grid(runs: list[Run], prefix: str, ylabel: str, title: str,
                    outdir: Path, filename: str) -> Path:
    """2x3 small multiples, one panel per Voigt component, runs as series."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True)
    for ax, c in zip(axes.ravel(), _VOIGT_NAMES):
        for r, color in zip(runs, _RUN_COLORS):
            x, xlabel = _xaxis(r.data)
            ax.plot(x, r.data[f"{prefix}_{c}"], "o-", markersize=2.5,
                    color=color, label=r.label)
        ax.set_title(c, fontsize=9)
        ax.axhline(0.0, color="gray", linewidth=0.6)
    for ax in axes[-1]:
        ax.set_xlabel(_xaxis(runs[0].data)[1])
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=8, ncol=len(runs))
    fig.suptitle(title, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = outdir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def plot_pressure(runs: list[Run], outdir: Path) -> Path:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4.5))
    for r, color in zip(runs, _RUN_COLORS):
        x, xlabel = _xaxis(r.data)
        ax.plot(x, r.data["pressure"], "o-", markersize=3, color=color, label=r.label)
    ax.axhline(0.0, color="gray", linewidth=0.8)
    ax.set_xlabel(_xaxis(runs[0].data)[1])
    ax.set_ylabel(r"$\bar p$ [MPa]  ($p>0$ = tension)")
    ax.set_title("Homogenized hydrostatic pressure")
    _style(ax, len(runs))
    fig.tight_layout()
    path = outdir / "pressure.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def plot_hardening(runs: list[Run], outdir: Path) -> Path:
    from utils.plotting import ResponseCurve, plot_homogenized_response

    curves = [ResponseCurve(x=r.data["plastic_strain_mean"], y=r.data["mises_stress"],
                            marker="o-", label=r.label if len(runs) > 1 else None)
              for r in runs]
    ax = plot_homogenized_response(
        curves, include_origin=False,
        xlabel=r"mean accumulated plastic strain $\bar\alpha$",
        ylabel=r"$\bar\sigma_{vM}$ [MPa]",
        title="Homogenized hardening response",
    )
    path = outdir / "hardening.png"
    ax.figure.savefig(path, dpi=150, bbox_inches="tight")
    return path


def plot_plastic_activity(runs: list[Run], outdir: Path) -> Path:
    """
    Mean and peak accumulated plastic strain against the load parameter.

    Separate from hardening.png on purpose: that one asks what stress the
    plastic strain buys, this one asks how much plastic flow there is and how
    evenly it is spread. The two panels can move in opposite directions -- a
    stiffer early hardening law lowers the mean while the peak rises, which is
    plastic strain localizing into fewer voxels rather than plasticity going
    away -- and that is invisible in either curve alone.
    """
    import matplotlib.pyplot as plt

    fig, (ax_mean, ax_max) = plt.subplots(1, 2, figsize=(9.5, 4.2))
    for r, color in zip(runs, _RUN_COLORS):
        x, _ = _xaxis(r.data)
        ax_mean.plot(x, r.data["plastic_strain_mean"], "o-", markersize=3,
                     color=color, label=r.label)
        ax_max.plot(x, r.data["plastic_strain_max"], "o-", markersize=3,
                    color=color, label=r.label)
    xlabel = _xaxis(runs[0].data)[1]
    for ax, ylab, title in ((ax_mean, r"mean $\bar\alpha$", "Mean accumulated plastic strain"),
                            (ax_max, r"max $\alpha$ over voxels", "Peak accumulated plastic strain")):
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(bottom=0.0)
    _style(ax_mean, len(runs))
    fig.tight_layout()
    path = outdir / "plastic_activity.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def plot_plastic_strain(runs: list[Run], outdir: Path) -> list[Path]:
    """Plastic strain tensor components, plus its trace in its own figure."""
    import matplotlib.pyplot as plt

    written: list[Path] = []
    if len(runs) == 1:
        data = runs[0].data
        x, xlabel = _xaxis(data)
        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        for c in _VOIGT_NAMES:
            ax.plot(x, data[f"eps_p_{c}"], "o-", markersize=3, label=c)
        ax.set_ylabel(r"$\bar\varepsilon^p$")
        ax.set_xlabel(xlabel)
        ax.legend(ncol=6, fontsize=8, loc="upper left")
        ax.set_title("Homogenized plastic strain tensor")
        fig.tight_layout()
        p = outdir / "plastic_strain.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        written.append(p)
    else:
        written.append(_component_grid(runs, "eps_p", r"$\bar\varepsilon^p$",
                                       "Homogenized plastic strain tensor",
                                       outdir, "plastic_strain.png"))

    fig, ax = plt.subplots(figsize=(5, 4.5))
    for r, color in zip(runs, _RUN_COLORS):
        x, _ = _xaxis(r.data)
        ax.plot(x, r.data["plastic_vol_strain"], "o-", markersize=3,
                color=color, label=r.label)
    ax.axhline(0.0, color="gray", linewidth=0.8)
    ax.set_xlabel(_xaxis(runs[0].data)[1])
    ax.set_ylabel(r"tr$(\bar\varepsilon^p)$")
    ax.set_title("Plastic volume change (dilatancy)")
    _style(ax, len(runs))
    fig.tight_layout()
    p = outdir / "plastic_volume.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    written.append(p)
    return written


def plot_damage(runs: list[Run], outdir: Path) -> Path:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4.5))
    single = len(runs) == 1
    for r, color in zip(runs, _RUN_COLORS):
        x, _ = _xaxis(r.data)
        ax.plot(x, r.data["damage_max"], "o-", markersize=3, color=color,
                label="max" if single else f"{r.label} (max)")
        ax.plot(x, r.data["damage_mean"], "o--", markersize=3, color=color,
                label="mean" if single else f"{r.label} (mean)")
    ax.set_xlabel(_xaxis(runs[0].data)[1])
    ax.set_ylabel("damage $d$")
    ax.set_title("Homogenized damage evolution")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = outdir / "damage.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Plot one or more homogenized-response CSVs (scripts/"
                    "postprocessing/homogenize.py output): stress-strain, stress "
                    "path, component-vs-load, pressure, and (where present) "
                    "hardening / plastic / damage curves. Several CSVs are "
                    "overlaid for comparison."
    )
    parser.add_argument("csv", type=Path, nargs="+", help="*_homogenized.csv path(s)")
    parser.add_argument("-o", "--outdir", type=Path, default=None,
                        help="Directory for the PNGs (default: <csv-parent>/"
                             "<csv-stem>_plots for one CSV, "
                             "<common-stem>_comparison_plots for several)")
    parser.add_argument("--labels", type=str, nargs="+", default=None,
                        help="Legend label per CSV, in order (default: the part "
                             "of each filename that differs from the others)")
    parser.add_argument("--component", type=str, default=None, choices=_VOIGT_NAMES,
                        help="Voigt component (e.g. 12) for the stress-strain plot; "
                             "default: the component with the largest strain excursion")
    parser.add_argument("--show", action="store_true",
                        help="Also display the figures interactively (default: save only)")
    args = parser.parse_args()

    if len(args.csv) > len(_RUN_COLORS):
        parser.error(f"at most {len(_RUN_COLORS)} CSVs can be compared at once "
                     f"(got {len(args.csv)}) -- past that the curves cannot be told "
                     "apart by colour; split the comparison instead")

    if not args.show:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = _labels(args.csv, args.labels)
    runs = [Run(_load(p), lab, p) for p, lab in zip(args.csv, labels)]

    if args.outdir:
        outdir = args.outdir
    elif len(runs) == 1:
        outdir = args.csv[0].parent / f"{args.csv[0].stem}_plots"
    else:
        common = os.path.commonprefix([p.stem for p in args.csv]).rstrip("_-") or "comparison"
        outdir = args.csv[0].parent / f"{common}_comparison_plots"
    outdir.mkdir(parents=True, exist_ok=True)

    if len(runs) > 1:
        print("Comparing: " + ", ".join(f"{r.label}" for r in runs))

    comp = _pick_component(runs, args.component)
    written = [
        plot_stress_strain(runs, comp, outdir),
        *plot_components(runs, outdir),
        plot_pressure(runs, outdir),
        plot_stress_path(runs, outdir),
    ]
    # A run missing a column drops out of that figure only; the figure itself
    # is skipped when no run has it.
    for cols, fn, note in (
        (("plastic_strain_mean",), plot_hardening, "hardening.png"),
        (("plastic_strain_mean", "plastic_strain_max"), plot_plastic_activity, "plastic_activity.png"),
        (("plastic_vol_strain",), plot_plastic_strain, "plastic_strain.png / plastic_volume.png"),
        (("damage_max", "damage_mean"), plot_damage, "damage.png"),
    ):
        subset = _with(runs, *cols)
        if not subset:
            print(f"Note: no run has {', '.join(cols)} -- skipping {note}")
            continue
        if len(subset) < len(runs):
            missing = ", ".join(r.label for r in runs if r not in subset)
            print(f"Note: {note} omits {missing} (no {', '.join(cols)} column)")
        out = fn(subset, outdir)
        written.extend(out if isinstance(out, list) else [out])

    for path in written:
        print(f"Written → {path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
