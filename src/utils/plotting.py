"""Generic plotting helpers: grids of imshow field slices, and homogenized
(volume-averaged) response curves."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.colors import to_hex
from matplotlib.image import AxesImage
from matplotlib.figure import Figure
from matplotlib.layout_engine import PlaceHolderLayoutEngine

STYLE = Path(__file__).with_name("fftjax.mplstyle")
SEQUENTIAL_CMAP = "plasma"
DIVERGING_CMAP = "PuOr_r"  # signed fields; pair with a symmetric vmin/vmax


def use_style() -> None:
    """Apply the project matplotlib style (``fftjax.mplstyle``) globally."""
    plt.style.use(STYLE)


def plasma_colors(n: int, stop: float = 0.85) -> list[str]:
    """``n`` line colors sampled evenly from plasma on ``[0, stop]`` -- for
    more series than the style's 5-color cycle, or ordered sweeps (e.g. one
    line per load level). ``stop < 1`` skips the yellow end, too light on white."""
    cmap = colormaps[SEQUENTIAL_CMAP]
    return [to_hex(cmap(t)) for t in np.linspace(0.0, stop, n)]


# ── Axis convention (DIN 461) ──────────────────────────────────────────────────
# Label "quantity ->", unit printed in place of the second-to-last visible tick.
# Plots keep writing labels as "quantity [unit]"; axis_style() converts them.

_LABEL_RE = re.compile(r"^(?P<var>.*?)\s*\[(?P<unit>[^\]]*)\]\s*(?P<rest>.*)$", re.S)
DIMENSIONLESS = "–"


class UnitTickFormatter(mticker.Formatter):
    """Wraps a numeric formatter, replacing the second-to-last tick inside the
    view with ``unit`` (plus any ``x10^n`` offset, which is folded in)."""

    def __init__(self, unit: str, base: mticker.Formatter):
        self.unit, self.base = unit, base

    def set_axis(self, axis):
        super().set_axis(axis)
        self.base.set_axis(axis)

    def set_locs(self, locs):
        self.base.set_locs(locs)

    def __call__(self, x, pos=None):
        return self.base(x, pos)

    def get_offset(self):
        return ""  # folded into the unit tick

    def format_ticks(self, values):
        labels = list(self.base.format_ticks(values))
        if self.axis is None:  # typed Optional; matplotlib attaches it before formatting
            return labels
        lo, hi = sorted(self.axis.get_view_interval())
        tol = 1e-9 * max(hi - lo, 1e-300)
        inside = [k for k, v in enumerate(values) if lo - tol <= v <= hi + tol]
        if len(inside) >= 2:
            offset = self.base.get_offset()
            unit = "" if offset and self.unit == DIMENSIONLESS else self.unit
            labels[inside[-2]] = f"{offset} {unit}".strip()
        return labels


def axis_label(text: str) -> tuple[str, str]:
    """``"$\\sigma$ [MPa] (note)"`` -> (``"$\\sigma$ (note)$\\,\\rightarrow$"``, ``"MPa"``)."""
    m = _LABEL_RE.match(text)
    var, unit = (m["var"] + (" " + m["rest"] if m["rest"] else ""), m["unit"]) if m else (text, "")
    var, unit = var.strip(), unit.strip()
    if re.fullmatch(r"[A-Za-z]", var):  # quantity symbol -> italic math
        var = f"${var}$"
    if var.endswith("$"):  # arrow joins the trailing math segment
        label = var[:-1] + r"\,\rightarrow$"
    else:
        label = rf"{var}$\,\rightarrow$"
    return label, DIMENSIONLESS if unit in ("", "-", "1") else unit


def axis_style(fig_or_ax: Figure | Axes) -> int:
    """Convert every labelled numeric axis (colorbars included) of a figure to
    the DIN 461 axis convention. Idempotent; call once the labels are set, before
    savefig. Category axes (text tick labels) are left unchanged. Returns the
    number of axes converted."""
    fig = fig_or_ax if isinstance(fig_or_ax, Figure) else fig_or_ax.figure
    converted = 0
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            text = axis.get_label_text()
            base = axis.get_major_formatter()
            if not text or r"\rightarrow" in text or isinstance(base, UnitTickFormatter):
                continue
            if not isinstance(base, (mticker.ScalarFormatter, mticker.LogFormatter)):
                continue
            label, unit = axis_label(text)
            axis.set_label_text(label)
            axis.set_major_formatter(UnitTickFormatter(unit, base))
            converted += 1
    return converted


def save_figure(fig: Figure, path, **kwargs) -> None:
    """``axis_style(fig)`` then ``fig.savefig(path, **kwargs)`` -- the one way
    scripts save figures, so every exported plot follows the DIN 461 axis convention.
    Axes holding an image also lose the style's grid."""
    for ax in fig.axes:
        if ax.images:
            ax.grid(False)
    # A tight_layout() run before the conversion measured the old labels
    # (the unit tick is wider); redo it.
    if axis_style(fig) and isinstance(fig.get_layout_engine(), PlaceHolderLayoutEngine):
        fig.tight_layout()
    fig.savefig(path, **kwargs)

# Match colorbar height to a square imshow panel -- without this, fig.colorbar
# sizes to the axes' unshrunk bounding box, not the square image imshow
# actually renders inside it, so it comes out much taller.
_CBAR_KW: dict[str, Any] = dict(fraction=0.046, pad=0.04)


@dataclass
class FieldPanel:
    """One imshow panel: a 2-D field plus its display options."""
    field: np.ndarray
    title: str
    cmap: str = SEQUENTIAL_CMAP
    fmt: str | None = None
    vmin: float | None = None
    vmax: float | None = None
    cbar_label: str | None = None


def imshow_panel(
    ax: Axes,
    panel: FieldPanel,
    extent: Sequence[float],
    xlabel: str = "x [mm]",
    ylabel: str = "y [mm]",
) -> AxesImage:
    """
    Draw one ``FieldPanel`` onto an existing axes, with a colorbar sized to
    match the (square) image -- the building block ``plot_field_grid`` loops
    over; use it directly to drop a field panel into an axes shared with
    other (non-field) plots.
    """
    x0, x1, y0, y1 = extent  # any 4-sequence in; imshow's stub wants exactly a 4-tuple
    im = ax.imshow(np.asarray(panel.field).T, origin="lower", cmap=panel.cmap,
                    extent=(x0, x1, y0, y1), vmin=panel.vmin, vmax=panel.vmax)
    ax.grid(False)  # the style's grid is for line plots, not field images
    ax.set_title(panel.title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.figure.colorbar(im, ax=ax, format=panel.fmt, label=panel.cbar_label, **_CBAR_KW)
    return im


def plot_field_grid(
    panels: Sequence[Sequence[FieldPanel | None]],
    extent: Sequence[float],
    *,
    figsize: tuple[float, float] | None = None,
    xlabel: str = "x [mm]",
    ylabel: str = "y [mm]",
) -> tuple[Figure, np.ndarray]:
    """
    Lay out a grid of 2-D field slices as ``imshow`` panels (via ``imshow_panel``).

    Parameters
    ----------
    panels : rows of ``FieldPanel`` (``None`` for an empty cell)
        ``panels[row][col].field`` is transposed and shown with ``origin="lower"``,
        matching the (x, y) axis convention used throughout this project. Rows
        may be ragged -- shorter rows leave their trailing cells blank.
    extent : ``[x0, x1, y0, y1]``   physical extent passed to every panel's ``imshow``.

    Returns
    -------
    fig, axes
    """
    nrows = len(panels)
    ncols = max(len(row) for row in panels)
    if figsize is None:
        figsize = (3.2 * ncols, 3.0 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    for row, axes_row in zip(panels, axes):
        for panel, ax in zip(row, axes_row):
            if panel is None:
                ax.axis("off")
            else:
                imshow_panel(ax, panel, extent, xlabel, ylabel)
        for ax in axes_row[len(row):]:
            ax.axis("off")

    fig.tight_layout()
    return fig, axes


@dataclass
class ResponseCurve:
    """One homogenized (volume-averaged) response curve, e.g. a macroscopic
    strain-stress history for one tensor direction/component -- see
    ``post.fields.homogenize_response``."""
    x: np.ndarray
    y: np.ndarray
    label: str | None = None
    marker: str = "o-"


def plot_homogenized_response(
    curves: Sequence[ResponseCurve],
    *,
    ax: Axes | None = None,
    xlabel: str = r"$\bar\varepsilon$",
    ylabel: str = r"$\bar\sigma$ [MPa]",
    title: str | None = "Macroscopic response",
    include_origin: bool = True,
) -> Axes:
    """
    Plot one or more homogenized response curves -- one line per
    direction/component -- on a single axes.

    Parameters
    ----------
    curves         : one ``ResponseCurve`` per direction/component to overlay.
    include_origin : prepend (0, 0) to every curve, since a response curve
                      almost always starts from the unloaded state.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4.5))

    for curve in curves:
        x, y = np.asarray(curve.x), np.asarray(curve.y)
        if include_origin:
            x = np.concatenate([[0.0], x])
            y = np.concatenate([[0.0], y])
        ax.plot(x, y, curve.marker, label=curve.label)

    ax.locator_params(nbins=6)  # strain ticks (0.0025 steps) otherwise collide
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if any(curve.label for curve in curves):
        ax.legend()
    return ax
