from __future__ import annotations

from string import ascii_lowercase
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Colormap, LinearSegmentedColormap, Normalize, rgb2hex

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from matplotlib.colorbar import Colorbar


def extract_color_map(n_points: int, cmap: str | Colormap = "viridis") -> list[str]:
    """
    Extracts a list of colors belonging to a determinate cmap

    Parameters
    ----------
    n_points: int
        Number of points that are going to be converted in RGBA colors
    cmap: str or Colormap (optional, default='viridis')
        Color map, or its name

    Returns
    -------
    colorlist: list (n_points)
        Colors in its hexadecimal string representation belonging to the selected color map
    """

    c_map = colormaps.get_cmap(cmap)

    arr = np.linspace(0, 1, n_points)
    colorlist = []
    for c in arr:
        # Select the rgba value of the cmap at point c, which is a number between 0 and 1
        rgba = c_map(c)
        clr = rgb2hex(rgba)
        colorlist.append(str(clr))

    return colorlist


def add_labels(
    axs: Sequence[Axes],
    positions: Sequence[float] | Sequence[Sequence[float]] | None = None,
    letters: Sequence[str] | None = None,
    initial_parenthesis: bool = True,
    **kwargs,
) -> None:
    """
    Add the panel labels, (a), (b), ..., to a list of axes.

    Parameters
    ----------
    axs: list
        Sorted axes to label.
    positions: list (optional, default=None)
        x and y position of each label, relative to its axis. If None, x = -0.1 and y = 1.1 is used for all axes. If
        only one position [x, y] is given, the same position is used for all axes. Otherwise, one position per axis
        must be given.
    letters: list[str] (optional, default=None)
        Letter for each label. If None, the axes are labeled in alphabetic order (a, b, ...). If letters are provided,
        then len(letters) == len(axs).
    initial_parenthesis: bool (optional, default=True)
        If True, print the labels as (a), else as a).
    kwargs: (optional)
        Parameters passed to ax.text. By default, ha='center' and va='center'.
    """

    if positions is None:
        positions = [-0.1, 1.1]

    if np.ndim(positions) == 1:  # Only one position provided
        positions_all = [positions] * len(axs)
    else:
        positions_all = positions

    if letters is None:
        letters = [ascii_lowercase[i] for i in range(len(axs))]

    initial = "(" if initial_parenthesis else ""
    kwargs.setdefault("ha", "center")
    kwargs.setdefault("va", "center")

    for ax, position, letter in zip(axs, positions_all, letters):
        x, y = np.asarray(position, dtype=float)
        ax.text(x, y, f"{initial}{letter})", transform=ax.transAxes, **kwargs)


def shiftedColorMap(
    cmap: Colormap,
    start: float = 0,
    midpoint: float = 0.5,
    stop: float = 1.0,
    name: str = "shiftedcmap",
) -> LinearSegmentedColormap:
    """
    Function to offset the "center" of a colormap. Useful for data with a negative min and positive max, and you want
    the middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from the lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. In general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0, and you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from the highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between `midpoint` and 1.0.
    """
    cdict: dict[str, list[tuple[float, float, float]]] = {
        "red": [],
        "green": [],
        "blue": [],
        "alpha": [],
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack(
        [
            np.linspace(0.0, midpoint, 128, endpoint=False),
            np.linspace(midpoint, 1.0, 129, endpoint=True),
        ]
    )

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict["red"].append((si, r, r))
        cdict["green"].append((si, g, g))
        cdict["blue"].append((si, b, b))
        cdict["alpha"].append((si, a, a))

    return LinearSegmentedColormap(name, cdict)  # ty: ignore[invalid-argument-type]


def identify_pulses(
    alphas: np.ndarray, betas: np.ndarray
) -> tuple[list[str], list[float], list[float], list[str], ScalarMappable]:
    """
    Colors, line widths, z-orders and labels to plot the pulses for different values of n_+ = (alpha + beta) / 2. The
    linear (n_+ = 0), geometric fast-QUAD (n_+ = 2), FAQUAD (n_+ = 3) and pi-pulse (n_+ = inf) are highlighted.
    """
    n_plus = (np.asarray(alphas) + np.asarray(betas)) / 2
    len_n = len(n_plus)

    vmin = np.nanmin(n_plus[n_plus != -np.inf])
    vmax = np.nanmax(n_plus[n_plus != np.inf])
    vmid = 0
    midpoint = float(np.argmin(np.abs(n_plus - vmid)) / len_n)

    mymap = shiftedColorMap(colormaps["coolwarm"], midpoint=midpoint)
    colors = extract_color_map(len_n, cmap=mymap)

    linewidths = [1.0] * len_n
    zorders = [1.0] * len_n

    n_FAQUAD = (4 + 2) / 2
    n_geo = (2 + 2) / 2
    n_pi = np.inf
    n_linear = 0

    index_FAQUAD = np.argmin(np.abs(n_plus - n_FAQUAD))
    index_geo = np.argmin(np.abs(n_plus - n_geo))
    index_pi = np.argmax(n_plus >= n_pi)
    index_linear = np.argmin(np.abs(n_plus - n_linear))
    indices = [index_FAQUAD, index_geo, index_linear, index_pi]

    colors[index_FAQUAD] = "g"
    colors[index_geo] = "indigo"
    colors[index_pi] = "r"
    colors[index_linear] = "darkorange"

    for index in indices:
        linewidths[index] = 2
        zorders[index] = 2

    labels = [f"_$n_+ = {n:.2f}$" for n in n_plus]
    labels[index_FAQUAD] = "FAQUAD"
    labels[index_geo] = "Geometrical"
    labels[index_pi] = r"$\pi$-pulse"
    labels[index_linear] = "Linear"

    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(cmap=mymap, norm=norm)

    return colors, linewidths, zorders, labels, sm


def plot_gradient_lines(
    x: np.ndarray,
    ys: np.ndarray | Sequence[np.ndarray],
    alphas: np.ndarray,
    betas: np.ndarray,
    x_label: str | None = None,
    y_label: str | None = None,
    cbar_label: str | None = None,
    ax: Axes | None = None,
    legend_bool: bool = True,
    set_limits: bool = True,
    cbar_bool: bool = True,
) -> tuple[Axes, ScalarMappable, Colorbar | None]:
    """
    Plot a set of curves, one for each pair (alpha, beta), colored according to n_+ = (alpha + beta) / 2 (see
    identify_pulses).

    Parameters
    ----------
    x: np.ndarray
        Values for the x-axis, shared by all the curves
    ys: np.ndarray or list
        Curves to plot. The dimension is [len(alphas) x len(x)]
    alphas, betas: np.ndarray
        Exponents of the hypermetric for each curve
    x_label, y_label, cbar_label: str (optional, default=None)
        Labels for the axes and the colorbar
    ax: Axes (optional, default=None)
        Axis in which the curves are plotted. If None, a new figure is created
    legend_bool: bool (optional, default=True)
        If True, show the legend with the highlighted pulses
    set_limits: bool (optional, default=True)
        If True, set the limits of the axes to the range of the data
    cbar_bool: bool (optional, default=True)
        If True, add a colorbar for n_+

    Returns
    -------
    ax: Axes
        Axis with the curves
    sm: ScalarMappable
        Mappable with the colormap for n_+, which can be used to create a colorbar
    cbar: Colorbar or None
        Colorbar, if cbar_bool is True
    """
    colors, linewidths, zorders, labels, sm = identify_pulses(alphas, betas)

    if ax is None:
        _, ax = plt.subplots()

    for i, y in enumerate(ys):
        ax.plot(
            x,
            y,
            color=colors[i],
            linewidth=linewidths[i],
            zorder=zorders[i],
            label=labels[i],
        )

    cbar = (
        plt.colorbar(sm, ax=ax, label=cbar_label, extend="both") if cbar_bool else None
    )

    if legend_bool:
        ax.legend()

    if set_limits:
        ax.set_xlim(min(x), max(x))
        ax.set_ylim(np.min(ys) * 1.01, np.max(ys) * 1.01)

    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)

    return ax, sm, cbar
