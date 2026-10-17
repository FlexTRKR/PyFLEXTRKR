"""
The defaults of the old Analysis/plot_subset_*.py scripts, for the stubs that keep their commands working.

Each old script is a stub of a few lines that calls main(legacy_name='<old script name>') of the new command line of its
family. The new command line takes the defaults listed here for that script (dpi, product, number of panels, which times
are skipped, ...), so that the old command makes the same figures as before. Any option given on the command line replaces
the default. docs/visualization/inventory.md has the table with the new command for each old script.

Called by: mcs_cli, cell_cli, generic_cli, cartesian_cli (parse_cmd_args, main). Calls: nothing.
"""
import sys

# old script (without .py) -> new script, and the defaults of the new command line that reproduce the old script
LEGACY = {
    # --- MCS maps (mcs_cli)
    'plot_subset_tbpf_mcs_tracks_demo': dict(
        script='plot_mcs_tracks.py', options='--product tbpf --panels 2 --map_aspect auto',
        defaults=dict(product='tbpf', panels=2, map_aspect='auto'),
    ),
    'plot_subset_tbpf_mcs_tracks_1panel_demo': dict(
        script='plot_mcs_tracks.py', options='--product tbpf --panels 1 --skip_empty 1 --map_aspect auto',
        defaults=dict(product='tbpf', panels=1, skip_empty=1, map_aspect='auto'),
    ),
    'plot_subset_tbze_mcs_tracks_demo': dict(
        script='plot_mcs_tracks.py', options='--product tbze --panels 2 --dpi 300 --draw_state 1 --map_aspect auto',
        defaults=dict(product='tbze', panels=2, dpi=300, draw_state=1, map_aspect='auto'),
    ),
    'plot_subset_tbze_mcs_tracks_1panel_demo': dict(
        script='plot_mcs_tracks.py', options='--product tbze --panels 1 --dpi 300 --draw_state 1 --map_aspect auto',
        defaults=dict(product='tbze', panels=1, dpi=300, draw_state=1, map_aspect='auto'),
    ),
    'plot_subset_tb_mcs_tracks_demo': dict(
        script='plot_mcs_tracks.py', options='--product tb --dpi 300 --fontsize 13 --figsize 10 10 --skip_empty 1 --track_smooth 0 --map_aspect auto',
        defaults=dict(product='tb', panels=1, dpi=300, fontsize=13, figsize_x=12, figsize=[10., 10.], skip_empty=1, workers=None,
                      track_smooth=0, map_aspect='auto'),
    ),
    # --- convective cell maps (cell_cli)
    'plot_subset_cell_tracks_demo': dict(
        script='plot_cell_tracks.py', options='--product refl --map_aspect auto',
        defaults=dict(product='refl', map_aspect='auto'),
    ),
    'plot_subset_cell_tracks_ETH_demo': dict(
        script='plot_cell_tracks.py', options='--product eth --skip_empty 1 --map_aspect auto',
        defaults=dict(product='eth', skip_empty=1, workers=None, map_aspect='auto'),
    ),
    # --- generic feature maps (generic_cli)
    'plot_subset_generic_tracks_demo': dict(
        script='plot_generic_tracks.py', options='--map_aspect auto',
        defaults=dict(workers=None, map_aspect='auto'),
    ),
    'plot_subset_psi_tracks_demo': dict(
        script='plot_generic_tracks.py',
        options="--cmap RdBu_r --levels -3e7 3e7 0.5e6 --cbticks -3e7 3e7 1e7 --cblabel '600hPa PSI Anomaly' --fontsize 11 "
                "--map_resolution 110m --grid_linewidth 0 --track_marker_size 6 --draw_border 0 --map_aspect auto",
        defaults=dict(workers=None, map_resolution='110m', fontsize=11, track_marker_size=6, cmap='RdBu_r',
                      levels=[-3e7, 3e7, 0.5e6], cbticks=[-3e7, 3e7, 1e7], cblabel='600hPa PSI Anomaly',
                      grid_linewidth=0, draw_border=0, track_number_source='mask', show_symbols=1,
                      time_label_format='%Y-%m-%d %H:%M UTC', figname_time_format='%Y%m%d_%H%M', map_aspect='auto'),
    ),
    # --- x/y domains without a map (cartesian_cli)
    'plot_subset_tbpf_demo_pbc': dict(
        script='plot_cartesian_tracks.py', options='--product tbpf',
        defaults=dict(product='tbpf', workers=None, stats_key='mcsfinal'),
    ),
    'plot_subset_tbpf_tracks_pinacles_pbc': dict(
        script='plot_cartesian_tracks.py', options='--product tbpf --xscale 1e-3 --perim_plot pcolormesh --show_pf_markers 0',
        defaults=dict(product='tbpf', workers=None, xscale=1e-3, perim_plot='pcolormesh', show_pf_markers=0, stats_key='finalstats'),
    ),
    'plot_subset_coldpool_tracks_nomap': dict(
        script='plot_cartesian_tracks.py', options='--product field --field_varname cp_intensity',
        defaults=dict(product='field', workers=None, field_varname='cp_intensity', stats_key='finalstats'),
    ),
}

#-----------------------------------------------------------------------
def defaults(name):
    """Defaults of the new command line that reproduce the old script `name` (a copy)."""
    return dict(LEGACY[name]['defaults'])

#-----------------------------------------------------------------------
def announce(name):
    """One line on stderr: the old script has moved, and the command of the new script."""
    entry = LEGACY[name]
    print(f"NOTE: Analysis/{name}.py has moved to visualization/{entry['script']} ({entry['options']}); "
          f"this command still works with the defaults of the old script.", file=sys.stderr)
