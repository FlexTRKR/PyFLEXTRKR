"""
Style of the track drawing: the halo (a contrasting outline) around track paths, initiation symbols and track numbers,
made with matplotlib path effects. The halo can have its own transparency.

The plot functions read the track style from plot_info (trackpath_color, trackpath_linewidth, trackpath_linestyle,
trackpath_alpha, track_marker, marker_size, track_marker_color, tracknumber_fontsize, tracknumber_color,
tracknumber_weight, tracknumber_alpha, track_halo_color, track_halo_width, track_halo_alpha), draw the artists
themselves (ax.plot, ax.scatter, ax.text as in the original scripts) and pass the path effects from here as path_effects=.

Called by: mcs_map, cell_map, generic_map, cartesian_map (plot functions). Calls: matplotlib.patheffects.
"""
import numpy as np
import matplotlib.patheffects as pe

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
