"""Generic plotting helpers: grids of imshow field slices, and homogenized
(volume-averaged) response curves."""

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from matplotlib.figure import Figure

# Match colorbar height to a square imshow panel -- without this, fig.colorbar
# sizes to the axes' unshrunk bounding box, not the square image imshow
# actually renders inside it, so it comes out much taller.
_CBAR_KW: dict[str, Any] = dict(fraction=0.046, pad=0.04)


@dataclass
class FieldPanel:
    """One imshow panel: a 2-D field plus its display options."""
    field: np.ndarray
    title: str
    cmap: str = "plasma"
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
    im = ax.imshow(np.asarray(panel.field).T, origin="lower", cmap=panel.cmap,
                    extent=extent, vmin=panel.vmin, vmax=panel.vmax)
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

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if any(curve.label for curve in curves):
        ax.legend()
    return ax
