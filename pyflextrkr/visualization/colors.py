"""
Colormap helpers: concatenate and truncate colormaps (the custom Tb colormap of the MCS maps), colormap by name.

Called by: mcs_cli, cell_cli, generic_cli (the plotting info blocks). Calls: matplotlib, numpy (colorcet if installed).
"""
import numpy as np
import matplotlib as mpl
from matplotlib.colors import ListedColormap

#-----------------------------------------------------------------------
def concat_cmaps(cmaps_list, ratios, discrete=256, trim_left=0.0, trim_right=0.0):
    """
    Concatenate multiple colormaps with optional trimming.

    Parameters:
    -----------
    cmaps_list : list
        List of colormaps to concatenate
    ratios : list
        List of ratios for each colormap (should sum to 1.0)
    discrete : int
        Total number of discrete colors in the output
    trim_left : float
        Fraction to trim from the left (start) of each colormap (0.0 to 1.0)
    trim_right : float
        Fraction to trim from the right (end) of each colormap (0.0 to 1.0)
    """
    total_colors = discrete
    all_colors = []

    for i, (cmap, ratio) in enumerate(zip(cmaps_list, ratios)):
        n_colors = int(total_colors * ratio)

        # Calculate the range to sample from, accounting for trimming
        start = trim_left
        end = 1.0 - trim_right

        # Sample colors from the trimmed range
        colors = cmap(np.linspace(start, end, n_colors))
        all_colors.append(colors)

    return ListedColormap(np.vstack(all_colors))

#-----------------------------------------------------------------------
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=256):
    """
    Truncate colormap.
    """
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

#-----------------------------------------------------------------------
def named_colormap(name):
    """
    Colormap by name: colorcet names first (e.g. 'glasbey_dark', 'CET_R1'), then matplotlib names (e.g. 'RdBu_r').
    """
    try:
        import colorcet as cc
        if name in cc.cm:
            return cc.cm[name]
    except ImportError:
        pass
    return mpl.colormaps[name]

#-----------------------------------------------------------------------
def cmap_colors(name):
    """The colors of a colormap by name (named_colormap) as an (N, 4) RGBA array: all N colors, in order."""
    cmap = named_colormap(name)
    return np.asarray(cmap(np.arange(cmap.N)))
