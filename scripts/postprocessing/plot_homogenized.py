"""
Visualize a homogenized-response CSV written by
scripts/postprocessing/homogenize.py.

Produces, in --outdir:
    stress_strain.png    -- sigma vs eps for one tensor component (the one
                             with the largest strain range, unless --component
                             is given), e.g. the tau-gamma hysteresis loop for
                             a solve_inelastic.py load/unload/reload cycle.
    components.png        -- all six strain and all six stress components vs
                             the load parameter, two stacked panels.
    pressure.png          -- hydrostatic pressure vs the load parameter.
    hardening.png          -- von Mises stress vs mean plastic strain, only
                             if the CSV has plastic_strain_mean (solve_inelastic.py).
    plastic_strain.png     -- the six homogenized plastic strain tensor
                             components and their trace (the plastic volume
                             change), only if the CSV has plastic_vol_strain,
                             i.e. the run had write_plastic_strain_tensor on.
    damage.png             -- max/mean damage vs the load parameter, only if
                             the CSV has damage_max (solve_fracture.py).

The x-axis is the CSV's ``gamma`` or ``t`` column (the applied load
parameter) when present, else the plain step index.

Usage
-----
    python scripts/postprocessing/plot_homogenized.py output/simulation/<jobname>_homogenized.csv
    python scripts/postprocessing/plot_homogenized.py results.csv --component 12 --show
"""

import argparse
import io
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

_VOIGT_NAMES = ("11", "22", "33", "12", "13", "23")  # matches homogenize.py / post.fields._VOIGT_IJ


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


def _xaxis(data: np.ndarray) -> tuple[np.ndarray, str]:
    names = data.dtype.names
    if "gamma" in names:
        return data["gamma"], r"applied load parameter $\bar\gamma$"
    if "t" in names:
        return data["t"], "load fraction $t$"
    return data["step"], "step"


def _pick_component(data: np.ndarray, override: str | None) -> int:
    if override is not None:
        if override not in _VOIGT_NAMES:
            raise ValueError(f"--component must be one of {_VOIGT_NAMES}, got {override!r}")
        return _VOIGT_NAMES.index(override)
    # default: the component with the largest strain excursion, i.e. the one
    # actually being driven -- works for shear (solve_inelastic.py's default
    # [0,1]) or normal (solve_mechanics.py uniaxial) loading alike.
    ranges = [np.ptp(data[f"eps_{c}"]) for c in _VOIGT_NAMES]
    return int(np.argmax(ranges))


def plot_stress_strain(data: np.ndarray, comp: int, outdir: Path) -> Path:
    from utils.plotting import ResponseCurve, plot_homogenized_response

    c = _VOIGT_NAMES[comp]
    curve = ResponseCurve(x=data[f"eps_{c}"], y=data[f"sigma_{c}"], marker="o-")
    ax = plot_homogenized_response(
        [curve], include_origin=False,
        xlabel=rf"$\bar\varepsilon_{{{c}}}$", ylabel=rf"$\bar\sigma_{{{c}}}$ [MPa]",
        title=f"Homogenized stress-strain response, component {c}",
    )
    path = outdir / "stress_strain.png"
    ax.figure.savefig(path, dpi=150, bbox_inches="tight")
    return path


def plot_components(data: np.ndarray, outdir: Path) -> Path:
    import matplotlib.pyplot as plt

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
    return path


def plot_pressure(data: np.ndarray, outdir: Path) -> Path:
    import matplotlib.pyplot as plt

    x, xlabel = _xaxis(data)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(x, data["pressure"], "o-")
    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\bar p$ [MPa]  ($p>0$ = tension)")
    ax.set_title("Homogenized hydrostatic pressure")
    fig.tight_layout()
    path = outdir / "pressure.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def plot_hardening(data: np.ndarray, outdir: Path) -> Path:
    from utils.plotting import ResponseCurve, plot_homogenized_response

    curve = ResponseCurve(x=data["plastic_strain_mean"], y=data["mises_stress"], marker="o-")
    ax = plot_homogenized_response(
        [curve], include_origin=False,
        xlabel=r"mean accumulated plastic strain $\bar\alpha$",
        ylabel=r"$\bar\sigma_{vM}$ [MPa]",
        title="Homogenized hardening response",
    )
    path = outdir / "hardening.png"
    ax.figure.savefig(path, dpi=150, bbox_inches="tight")
    return path


def plot_plastic_strain(data: np.ndarray, outdir: Path) -> Path:
    import matplotlib.pyplot as plt

    x, xlabel = _xaxis(data)
    fig, (ax_comp, ax_vol) = plt.subplots(2, 1, figsize=(6.5, 7), sharex=True)
    for c in _VOIGT_NAMES:
        ax_comp.plot(x, data[f"eps_p_{c}"], "o-", markersize=3, label=c)
    ax_comp.set_ylabel(r"$\bar\varepsilon^p$")
    ax_comp.legend(ncol=6, fontsize=8, loc="upper left")
    ax_comp.set_title("Homogenized plastic strain tensor")

    ax_vol.plot(x, data["plastic_vol_strain"], "o-", color="tab:red")
    ax_vol.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    ax_vol.set_ylabel(r"tr$(\bar\varepsilon^p)$")
    ax_vol.set_xlabel(xlabel)
    ax_vol.set_title("Plastic volume change (dilatancy)")

    fig.tight_layout()
    path = outdir / "plastic_strain.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def plot_damage(data: np.ndarray, outdir: Path) -> Path:
    import matplotlib.pyplot as plt

    x, xlabel = _xaxis(data)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(x, data["damage_max"], "o-", label="max")
    ax.plot(x, data["damage_mean"], "o-", label="mean")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("damage $d$")
    ax.set_title("Homogenized damage evolution")
    ax.legend()
    fig.tight_layout()
    path = outdir / "damage.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Plot a homogenized-response CSV (scripts/postprocessing/"
                    "homogenize.py output): stress-strain, component-vs-load, "
                    "pressure, and (where present) hardening / damage curves."
    )
    parser.add_argument("csv", type=Path, help="*_homogenized.csv path")
    parser.add_argument("-o", "--outdir", type=Path, default=None,
                        help="Directory for the PNGs (default: "
                             "<csv-parent>/<csv-stem>_plots)")
    parser.add_argument("--component", type=str, default=None, choices=_VOIGT_NAMES,
                        help="Voigt component (e.g. 12) for the stress-strain plot; "
                             "default: the component with the largest strain excursion")
    parser.add_argument("--show", action="store_true",
                        help="Also display the figures interactively (default: save only)")
    args = parser.parse_args()

    if not args.show:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = _load(args.csv)

    outdir = args.outdir if args.outdir else args.csv.parent / f"{args.csv.stem}_plots"
    outdir.mkdir(parents=True, exist_ok=True)

    comp = _pick_component(data, args.component)
    written = [
        plot_stress_strain(data, comp, outdir),
        plot_components(data, outdir),
        plot_pressure(data, outdir),
    ]
    if "plastic_strain_mean" in data.dtype.names:
        written.append(plot_hardening(data, outdir))
    else:
        print("Note: no plastic_strain_mean column -- skipping hardening.png")
    if "plastic_vol_strain" in data.dtype.names:
        written.append(plot_plastic_strain(data, outdir))
    if "damage_max" in data.dtype.names:
        written.append(plot_damage(data, outdir))
    else:
        print("Note: no damage_max column -- skipping damage.png")

    for path in written:
        print(f"Written → {path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
