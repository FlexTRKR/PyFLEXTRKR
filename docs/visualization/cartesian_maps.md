# x / y domain maps: `plot_cartesian_tracks.py`

Track maps of domains in x / y (meters or kilometers) without a map: idealized simulations, cold pool tracking of large-eddy simulations, MCS
in PINACLES. There are no coastlines, projections or gridlines, so the map options of the other families do not exist here; the
track style options (path, symbol, number, halo) do. A periodic domain (`--pbc 1`) draws the track paths that leave the domain at one edge and come
back at the opposite edge as separate segments.

```
python visualization/plot_cartesian_tracks.py -s START -e END -c CONFIG.yml [--product tbpf|field] [options]
```

| `--product` | what | replaces |
|---|---|---|
| `tbpf` (default) | Tb (gray) + precipitation with the MCS outlined (contour), periodic tracks, red triangles at the PF positions | `plot_subset_tbpf_demo_pbc.py` |
| `tbpf --xscale 1e-3 --perim_plot pcolormesh --show_pf_markers 0` | the same for PINACLES (meters -> km) | `plot_subset_tbpf_tracks_pinacles_pbc.py` |
| `field --field_varname VAR` | one field (default: cold pool intensity, `YlGnBu`) with the tracks | `plot_subset_coldpool_tracks_nomap.py` (`--field_varname cp_intensity`) |

![x/y domain](https://github.com/FlexTRKR/PyFLEXTRKR/releases/download/docs-assets/cartesian_pbc.png)

Pixel-level files: `time`, `longitude`, `latitude` (the x / y coordinates, 2-D), the variables of the product, `cloudtracknumber`; track statistics:
`meanlon`, `meanlat` (x, y of the track), `pf_lon`, `pf_lat` (the PF positions), `base_time`, `track_duration`. The coordinates are multiplied by `--xscale` /
`--yscale` for the axes (`--xlabel`, `--ylabel`). Defaults of this family: `dpi` 200, font size 14, `--figsize_x` 12, tracks of the previous 30 days, track path
color blueviolet, `--workers` from `nprocesses` of the config.

## Options of this family

The options of all families that apply here (period, files, output, track style) are in [options.md](options.md) (not the map options).

| option | default | meaning |
|---|---|---|
| `--product` | `tbpf` | 'tbpf': Tb + precipitation with the MCS outlined, 'field': one field (--field_varname) |
| `--tb_varname` | `tb` | Tb variable name (tbpf) |
| `--var2_varname` | `precipitation` | precipitation variable name (tbpf) |
| `--field_varname` | — | variable of the field (field; default: field_varname of the config) |
| `--cmap` | — | colormap of the field (field) |
| `--levels` VMIN VMAX STEP | — | shading levels of the field (field) |
| `--cbticks` VMIN VMAX STEP | — | colorbar ticks of the field (field) |
| `--cblabel` | — | colorbar label of the field (field) |
| `--pbc` | — | periodic domain: wrap the tracks at the edges (0:no, 1:yes; default: 1 for tbpf, 0 for field) |
| `--show_pf_markers` | — | red triangles at the positions of the precipitation features (0:no, 1:yes; default 1 for tbpf with --pbc 1) |
| `--xscale` | — | scale factor of the x coordinates, e.g. 1e-3: m -> km (default: 1 for tbpf, 1e-3 for field) |
| `--yscale` | — | scale factor of the y coordinates (default: as --xscale) |
| `--xlabel` | `X (km)` | x axis label |
| `--ylabel` | `Y (km)` | y axis label |
| `--perim_plot` | — | outline of the tracked features: pixels ('pcolormesh') or a line ('contour'); default: contour for tbpf, pcolormesh for field |
| `--perim_linewidth` | `1.5` | line width of the contour outline |
| `--perim_color` | — | color of the contour outline (default: orange for tbpf) |
| `--perim_alpha` | `0.6` | transparency of the pixel outline |
| `--perim_thick` | `2` | outline width [km] (pixel_radius of the config) |

`--extent` here is `xmin xmax ymin ymax` in the units of the data (one quoted string or four numbers, `--extent "0. 600. 0. 600."`).

## Examples

```bash
cd config                                   # after bash demo_mcs_tbpf_pbc_idealized.sh
python ../visualization/plot_cartesian_tracks.py -s 2020-01-01T00 -e 2020-01-02T00 -c config_demo_pbc.yml --product tbpf \
    -p 1 --extent "0. 1500. 0. 1500." --subset 0 --output $SCRATCH/quicklooks_pbc/

# cold pool intensity, tracks in white with a black halo
python ../visualization/plot_cartesian_tracks.py -s 2000-01-30T00 -e 2000-02-01T00 -c config_coldpool_buoyancy.yml \
    --product field --field_varname cp_intensity --extent "0. 600. 0. 600." --track_color white --track_halo_color black
```

Code map: [structure.md](structure.md).
