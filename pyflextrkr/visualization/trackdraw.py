"""
Style of the track drawing: the halo (a contrasting outline) around track paths, initiation symbols and track numbers,
made with matplotlib path effects. The halo can have its own transparency.

The plot functions read the track style from plot_info (trackpath_color, trackpath_linewidth, trackpath_linestyle,
trackpath_alpha, track_marker, marker_size, track_marker_color, tracknumber_fontsize, tracknumber_color,
tracknumber_weight, tracknumber_alpha, track_halo_color, track_halo_width, track_halo_alpha), draw the artists
themselves (ax.plot, ax.scatter, ax.text as in the original scripts) and pass the path effects from here as path_effects=.

draw_perimeter draws the outlines of the tracked objects (masks.label_perimeter) in one color, or one color for each track.

Called by: mcs_map, cell_map, generic_map, cartesian_map (plot functions). Calls: matplotlib.patheffects.
"""
import numpy as np
import matplotlib.patheffects as pe
from matplotlib.colors import ListedColormap

#-----------------------------------------------------------------------
def track_number_levels(track_dict):
    """
    Color levels of the track numbers for the MCS masks (limited to 256 levels to fit in a colormap). The track number is
    the track index + 1.
    """
    tracknumbers = track_dict['lifetime'].tracks.values
    if len(tracknumbers) == 0:
        return np.array([1., 2.])
    tn_nlev = np.min([len(tracknumbers), 256])
    if tn_nlev > 1:
        tn_levels = np.linspace(np.min(tracknumbers)+1, np.max(tracknumbers)+1, tn_nlev)
    else:
        # If only 1 track, add another level for color fill
        tn_levels = np.linspace(np.min(tracknumbers)+1, np.max(tracknumbers)+2, tn_nlev+1)
    return tn_levels

#-----------------------------------------------------------------------
def halo_effects(halo_color=None, halo_width=2.0, halo_alpha=1.0, path_linewidth=1.5):
    """
    Path effects for a halo around the tracks.

    Args:
        halo_color: color of the halo; None = no halo (all path effects are None, the artists are drawn as before)
        halo_width: width of the halo outside the edge of the line / symbol / character [points]
        halo_alpha: transparency of the halo (1 = opaque)
        path_linewidth: line width of the track path [points] (the halo line is this plus twice halo_width)

    Returns:
        halo_path, halo_marker, halo_text: lists of path effects for ax.plot (track path), ax.scatter (initiation
        symbol) and ax.text (track number), or None.
    """
    if not halo_color:
        return None, None, None
    # Stroke widths are centered on the outline: twice the width outside the edge
    halo_path = [pe.Stroke(linewidth=path_linewidth + 2 * halo_width, foreground=halo_color, alpha=halo_alpha),
                 pe.Normal()]
    halo_marker = [pe.withStroke(linewidth=2 * halo_width, foreground=halo_color, alpha=halo_alpha)]
    halo_text = [pe.withStroke(linewidth=2 * halo_width, foreground=halo_color, alpha=halo_alpha)]
    return halo_path, halo_marker, halo_text

#-----------------------------------------------------------------------
def draw_perimeter(ax, xx, yy, perim, color='k', alpha=1.0, zorder=3, transform=None, cmap_colors=None, rank_tracknumbers=None):
    """
    Draw the perimeter pixels of tracked objects (masks.label_perimeter: the track number on the perimeter, 0 elsewhere) in one
    color, or, with cmap_colors, in one color for each track: the color of a track is its rank among rank_tracknumbers modulo
    the number of colors, so a track keeps its color from frame to frame and neighboring tracks have different colors.

    Args:
        ax: map axes
        xx, yy: coordinates of the pixels (as for pcolormesh)
        perim: 2D array, track number on the perimeter, 0 elsewhere (None: nothing is drawn)
        color: color of the perimeter
        alpha, zorder, transform: of the drawing (the transform is that of the data)
        cmap_colors: (N, 4) RGBA array (colors.cmap_colors), optional; the colors of the tracks instead of color
        rank_tracknumbers: sorted 1D array of the track numbers of all tracks that are drawn, optional (None: the track number
            minus 1 is the index of the color)

    Returns:
        The drawing (QuadMesh), or None.
    """
    if perim is None:
        return None
    if cmap_colors is None:
        return ax.pcolormesh(xx, yy, np.ma.masked_where(perim == 0, np.ones(perim.shape)), cmap=ListedColormap([color]),
                             vmin=0, vmax=1, transform=transform, zorder=zorder, alpha=alpha)
    ncolors = len(cmap_colors)
    if rank_tracknumbers is None:
        index = (perim - 1) % ncolors
    else:
        index = np.searchsorted(rank_tracknumbers, perim) % ncolors
    # index k is the k-th color of the ListedColormap (vmin = -0.5, vmax = N - 0.5)
    return ax.pcolormesh(xx, yy, np.ma.masked_where(perim == 0, index), cmap=ListedColormap(cmap_colors),
                         vmin=-0.5, vmax=ncolors - 0.5, transform=transform, zorder=zorder, alpha=alpha)
