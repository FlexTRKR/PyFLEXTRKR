"""
What the libraries of the animation wrappers (mcs_animation.py, cell_animation.py) share: the parameters of a wrapper script (the
variables at the top of the file) with `--set NAME=VALUE` on the command line, and the default folder of the PNG files and the
animation: the quicklooks_trackpaths/ folder of the tracking run, next to the tracking output, as in the demo scripts (config/demo_*.sh).

Called by: mcs_animation, cell_animation. Calls: ft_utilities.load_config (reads the config file, makes no folders).
"""
import argparse
import datetime
import inspect
import os
import sys
import yaml
from pyflextrkr.ft_utilities import load_config

#-----------------------------------------------------------------------
def to_plain(value):
    """A date in a yaml value (2025-01-02T00:00) as the string the plot command reads."""
    if isinstance(value, (datetime.datetime, datetime.date)):
        return value.isoformat()
    return value

#-----------------------------------------------------------------------
def parameters_of(namespace):
    """The parameters of a wrapper script: its variables (not the names with a leading underscore, functions and modules)."""
    return {k: v for k, v in namespace.items() if not k.startswith('_') and not callable(v) and not inspect.ismodule(v)}

#-----------------------------------------------------------------------
def apply_overrides(params, defaults, where, argv=None, description=None, deprecated=None):
    """
    Set parameters from the command line: --set NAME=VALUE (a value is read as yaml: numbers, true, null, lists). The old names
    in `deprecated` still work (with a warning).

    Args:
        params: dictionary of the parameters (changed in place)
        defaults: dictionary of all the parameters of the library (a name that is not in it is an error)
        where: where the parameters are listed (for the error message), e.g. 'pyflextrkr.visualization.mcs_animation.DEFAULTS'
        argv: list of strings (default: sys.argv[1:])
        description: text for --help
        deprecated: dictionary old name -> new name
    """
    deprecated = deprecated or {}
    ap = argparse.ArgumentParser(description=description or "Make an animation. The parameters are the variables at the top of the script.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--set', action='append', default=[], metavar='NAME=VALUE',
                    help='set a parameter, e.g. --set start_date=2025-01-02T00 --set n_workers=32 --set lat_range=[-30,30]')
    for item in ap.parse_args(argv).set:
        name, sep, value = item.partition('=')
        if name in deprecated:
            print(f"DeprecationWarning: the parameter {name} is now called {deprecated[name]}")
            name = deprecated[name]
        if not sep or name not in defaults:
            sys.exit(f"ERROR: --set {item}: expected NAME=VALUE with NAME one of the parameters of the script ({where})")
        params[name] = to_plain(yaml.safe_load(value))
    return params

#-----------------------------------------------------------------------
def default_output_folder(config_file):
    """
    The folder for the PNG files and the animation when the wrapper gives none: {root_path}/quicklooks_trackpaths/ of the tracking
    config file, the folder that the demo scripts use (quicklook_dir = dir_demo/quicklooks_trackpaths/). It is not made here.
    """
    config = load_config(config_file, make_dirs=False)
    return config['root_path'].rstrip('/') + '/quicklooks_trackpaths/'

#-----------------------------------------------------------------------
def check_writable(folder):
    """ValueError if the folder cannot be written (or made): e.g. the read-only outputs of someone else's tracking run."""
    probe = folder.rstrip('/')
    while probe and not os.path.exists(probe):
        probe = os.path.dirname(probe)
    if not os.access(probe or '.', os.W_OK):
        raise ValueError(f"The folder for the PNG files and the animation, {folder}, cannot be written (it is next to the tracking "
                         f"output of the config file): set fig_root (or figdir and animation_dir)")
