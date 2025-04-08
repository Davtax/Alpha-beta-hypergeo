from typing import List, Optional

import matplotlib.pyplot as plt
import mplcursors
import numpy as np
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap, Normalize, rgb2hex


def extract_color_map(n_points: int, cmap: Optional[str] = 'viridis') -> List[str]:
    """
    Extracts a list of colors belonging to a determinate cmap

    Parameters
    ----------
    n_points: int
        Number of points that are going to be converted in RGBA colors
    cmap: str (optional, default='viridis)
        Color map name

    Returns
    -------
    colorlist: list (n_points)
        Colors in its hexadecimal string representation belonging to the selected color map
    """

    c_map = colormaps.get_cmap(cmap)

    arr = np.linspace(0, 1, n_points)
    colorlist = list()
    for c in arr:
        rgba = c_map(c)  # select the rgba value of the cmap at point c which is a number between 0 and 1
        clr = rgb2hex(rgba)
        colorlist.append(str(clr))

    return colorlist


def shiftedColorMap(cmap: LinearSegmentedColormap, start: Optional[float] = 0, midpoint: Optional[float] = 0.5,
                    stop: Optional[float] = 1.0, name: Optional[str] = 'shiftedcmap'):
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
    cdict = {'red': [], 'green': [], 'blue': [], 'alpha': []}

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack(
        [np.linspace(0.0, midpoint, 128, endpoint=False), np.linspace(midpoint, 1.0, 129, endpoint=True)])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = LinearSegmentedColormap(name, cdict)
    colormaps.register(cmap=newcmap, force=True)

    return newcmap


def identify_pulses(alphas, betas):
    n_plus = (alphas + betas) / 2
    len_n = len(n_plus)

    vmin = np.nanmin(n_plus[n_plus != -np.inf])
    vmax = np.nanmax(n_plus[n_plus != np.inf])
    vmid = 0
    midpoint = np.argmin(np.abs(n_plus - vmid)) / len_n

    mymap = shiftedColorMap(colormaps['coolwarm'], midpoint=midpoint)
    colors = extract_color_map(len_n, cmap=mymap)

    linewidths = [1] * len_n
    zorders = [1] * len_n

    n_FAQUAD = (4 + 2) / 2
    n_geo = (2 + 2) / 2
    n_pi = np.inf
    n_linear = 0

    index_FAQUAD = np.argmin(np.abs(n_plus - n_FAQUAD))
    index_geo = np.argmin(np.abs(n_plus - n_geo))
    index_pi = np.argmax(n_plus >= n_pi)
    index_linear = np.argmin(np.abs(n_plus - n_linear))
    indices = [index_FAQUAD, index_geo, index_linear, index_pi]

    colors[index_FAQUAD] = 'g'
    colors[index_geo] = 'indigo'
    colors[index_pi] = 'r'
    colors[index_linear] = 'darkorange'

    for index in indices:
        linewidths[index] = 2
        zorders[index] = 2

    labels = [f'_$n_+ = {n:.2f}$' for n in n_plus]
    labels[index_FAQUAD] = 'FAQUAD'
    labels[index_geo] = 'Geometrical'
    labels[index_pi] = r'$\pi$-pulse'
    labels[index_linear] = 'Linear'

    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=mymap, norm=norm)

    return colors, linewidths, zorders, labels, sm


def plot_gradient_lines(x: np.ndarray, ys: List[np.ndarray], alphas: np.ndarray, betas: np.ndarray,
                        x_label: Optional[str] = None, y_label: Optional[str] = None, cbar_label: Optional[str] = None,
                        ax: Optional[plt.axis] = None, cursor: Optional[bool] = False,
                        legend_bool: Optional[bool] = True, set_limits: Optional[bool] = True,
                        cbar_bool: Optional[bool] = True):
    colors, linewidths, zorders, labels, sm = identify_pulses(alphas, betas)

    if ax is None:
        fig, ax = plt.subplots()

    lines = []
    for i in range(len(ys)):
        y = ys[i]
        lines.append(ax.plot(x, y, color=colors[i], linewidth=linewidths[i], zorder=zorders[i], label=labels[i])[0])

    if cbar_bool:
        cbar = plt.colorbar(sm, ax=ax, label=cbar_label, extend='both')

    if legend_bool:
        ax.legend()

    if set_limits:
        ax.set_xlim(min(x), max(x))
        ax.set_ylim(np.min(ys) * 1.01, np.max(ys) * 1.01)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if cursor:
        add_cursor(lines)

    if cbar_bool:
        return ax, sm, cbar
    else:
        return ax, sm


def add_cursor(lines: List[plt.Line2D], lc: Optional[str] = 'yellow'):
    annotation_kwargs = dict(bbox=dict(boxstyle="round,pad=.5", fc="w", alpha=1, ec="k"),
                             arrowprops=dict(arrowstyle="->", connectionstyle="arc3", shrinkB=0, ec="k"))

    highlight_kwargs = dict(color=lc, markeredgecolor=lc, linewidth=3, markeredgewidth=3, facecolor=lc, edgecolor=lc)

    cursor = mplcursors.cursor(lines, highlight=True, annotation_kwargs=annotation_kwargs,
                               highlight_kwargs=highlight_kwargs)

    @cursor.connect("add")
    def on_add(sel):
        label_tmp = sel.artist.get_label()
        if label_tmp[0] == '_':
            label_tmp = label_tmp[1:]
        sel.annotation.set_text(label_tmp)
