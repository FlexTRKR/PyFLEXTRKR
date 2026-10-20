"""
Visualization of PyFLEXTRKR tracking results: maps of the tracked features on the input fields, one PNG per time.

The command-line scripts are in the top-level folder `visualization/` (README.md there); they are thin and only call
`main()` of one of the `*_cli` modules below. Importing this package does not import matplotlib or cartopy, so the
tracking never needs them; import the module you need (e.g. `from pyflextrkr.visualization import mcs_map`).

Module map (a family = one kind of plot; each family has a plot module and a command-line module):

    family            plot functions (the plotting code)   command line (arguments, main)    script in visualization/
    ----------------  -----------------------------------  ---------------------------------  --------------------------
    MCS maps          mcs_map.py                           mcs_cli.py                         plot_mcs_tracks.py
    convective cells  cell_map.py                          cell_cli.py                        plot_cell_tracks.py
    generic features  generic_map.py                       generic_cli.py                     plot_generic_tracks.py
    x/y (no map)      cartesian_map.py                     cartesian_cli.py                   plot_cartesian_tracks.py

    building blocks used by all families:
    mapcore.py     projections by name, central longitude, map domain (+ latitude-band crop), figure size, colorbars that follow the
                   map and keep their text inside the figure, longitude helpers
    masks.py       perimeters and centers of the labeled objects of a 2D mask
    colors.py      colormap helpers
    trackstats.py  track statistics of a period as dictionaries
    trackdraw.py   track path / symbol / number style, halo (path effects)
    pixelio.py     reading and subsetting a pixel-level file, figure names
    runner.py      the loop over the pixel-level files, serial or on Dask workers
    animation.py   ffmpeg: mp4 / GIF from the PNG files
    mcs_animation.py  the MCS animation of a wrapper script (visualization/examples/make_mcs_animation*.py): PNG files, mp4, GIF
    cell_animation.py the cell animation of a wrapper script (visualization/examples/make_cell_animation_nexrad.py): PNG files, mp4, GIF
    wrapper_common.py what the two animation libraries share: the --set NAME=VALUE overrides, the default output folder (next to the tracking output)
    cli_common.py  the command-line arguments that all families share
    legacy.py      the defaults of the old Analysis/plot_subset_*.py scripts (used by the stubs that keep them working)

The call chain, from a script down to the plotting code, is described in docs/visualization/structure.md.
"""
