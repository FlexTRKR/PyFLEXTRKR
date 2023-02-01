import numpy as np
from netCDF4 import chartostring
import xarray as xr
import sys
import logging
import warnings

def calc_stats_singlefile(
        tracknumbers,
        cloudidfile,
        trackstatus,
        trackmerge,
        tracksplit,
        trackreset,
        config,
):
    """
    Calculate statistics of track features from a single pixel file.

    Args:
        tracknumbers: numpy array
            Cloud track numbers.
        cloudidfile: string
            Cloudid filename.
        trackstatus: numpy array
            Status of each cloud track.
        trackmerge: numpy array
            Track numbers that the small clouds merge into.
        tracksplit: numpy array
            Track numbers that the small clouds split from.
        trackreset: numpy array
            Flag of track starts and abrupt track stops.
        config: dictionary
            Dictionary containing config parameters.

    Returns:
        out_dict: dictionary
            Dictionary containing the track statistics data.
        out_dict_attrs: dictionary
            Dictionary containing the attributes of track statistics data.
    """
    logger = logging.getLogger(__name__)

    tracking_outpath = config["tracking_outpath"]
    pixel_radius = config["pixel_radius"]
    feature_type = config.get("feature_type", None)
    terrain_file = config.get("terrain_file", None)
    rangemask_varname = config.get("rangemask_varname", 'None')
    feature_varname = config.get("feature_varname", "feature_number")

    # Only process file if that file contains a track
    if np.nanmax(tracknumbers) > 0:
        # fname = "".join(chartostring(cloudidfile))
        fname = chartostring(cloudidfile).item()
        logger.info(fname)

        # Load cloudid file
        cloudid_file = f"{tracking_outpath}{fname}"
        ds = xr.open_dataset(cloudid_file,
                             mask_and_scale=False,
                             decode_times=False)
        latitude = ds["latitude"].values
        longitude = ds["longitude"].values
        nx = ds.sizes["lon"]
        ny = ds.sizes["lat"]
        # file_cloudnumber = ds["cloudnumber"].squeeze().values
        file_corecold_cloudnumber = ds[feature_varname].squeeze().values
        file_basetime = ds["base_time"].squeeze()

        # Read feature specific variables
        if feature_type == "radar_cells":
            ref_varname = config["ref_varname"]
            # Convert x,y units to [km]
            x_coords = ds["x"] / 1000.
            y_coords = ds["y"] / 1000.
            file_dbz = ds[ref_varname].squeeze().values
            file_conv_core = ds["conv_core"].squeeze().values
            file_conv_mask = ds["conv_mask"].squeeze().values
            # Replace default cloudnumber with convective mask
            # Cell tracking uses expanded cloud area for tracking purpose only,
            # but the true cell mask is conv_mask
            file_corecold_cloudnumber = file_conv_mask
            # Convert echo-top height units to [km]
            file_echotop10 = ds["echotop10"].squeeze().values / 1000.
            file_echotop20 = ds["echotop20"].squeeze().values / 1000.
            file_echotop30 = ds["echotop30"].squeeze().values / 1000.
            file_echotop40 = ds["echotop40"].squeeze().values / 1000.
            file_echotop50 = ds["echotop50"].squeeze().values / 1000.

            # Range mask file
            if terrain_file is not None:
                dster = xr.open_dataset(terrain_file, decode_cf=False, mask_and_scale=False)
                rangemask = dster[rangemask_varname].values.astype('int8')
                dster.close()

        if feature_type == "tb_pf":
            file_tb = ds["tb"].squeeze().values
            file_cloudtype = ds["cloudtype"].squeeze().values

        # ds.close()

        # Find unique track numbers
        uniquetracknumbers = np.unique(tracknumbers)
        uniquetracknumbers = uniquetracknumbers[np.isfinite(uniquetracknumbers)]
        uniquetracknumbers = uniquetracknumbers[uniquetracknumbers > 0].astype(np.int32)

        # Create output variables
        fillval = -9999
        fillval_f = np.nan
        numtracks = len(uniquetracknumbers)
        out_basetime = np.full(numtracks, fillval, dtype=np.float64)
        out_meanlat = np.full(numtracks, fillval_f, dtype=np.float32)
        out_meanlon = np.full(numtracks, fillval_f, dtype=np.float32)
        out_area = np.full(numtracks, fillval_f, dtype=np.float32)
        out_cloudnumber = np.full(numtracks, fillval, dtype=np.int32)
        out_status = np.full(numtracks, fillval, dtype=np.int32)
        out_trackinterruptions = np.full(numtracks, fillval, dtype=np.int32)
        out_mergenumber = np.full(numtracks, fillval, dtype=np.int32)
        out_splitnumber = np.full(numtracks, fillval, dtype=np.int32)

        # Create feature specific variables
        # Radar cells
        if feature_type == "radar_cells":
            out_core_meanlat = np.full(numtracks, fillval_f, dtype=np.float32)
            out_core_meanlon = np.full(numtracks, fillval_f, dtype=np.float32)
            out_core_mean_x = np.full(numtracks, fillval_f, dtype=np.float32)
            out_core_mean_y = np.full(numtracks, fillval_f, dtype=np.float32)
            out_cell_meanlat = np.full(numtracks, fillval_f, dtype=np.float32)
            out_cell_meanlon = np.full(numtracks, fillval_f, dtype=np.float32)
            out_cell_mean_x = np.full(numtracks, fillval_f, dtype=np.float32)
            out_cell_mean_y = np.full(numtracks, fillval_f, dtype=np.float32)
            out_cell_max_dbz = np.full(numtracks, fillval_f, dtype=np.float32)
            out_cell_maxETH10dbz = np.full(numtracks, fillval_f, dtype=np.float32)
            out_cell_maxETH20dbz = np.full(numtracks, fillval_f, dtype=np.float32)
            out_cell_maxETH30dbz = np.full(numtracks, fillval_f, dtype=np.float32)
            out_cell_maxETH40dbz = np.full(numtracks, fillval_f, dtype=np.float32)
            out_cell_maxETH50dbz = np.full(numtracks, fillval_f, dtype=np.float32)
            out_core_area = np.full(numtracks, fillval_f, dtype=np.float32)
            out_cell_area = np.full(numtracks, fillval_f, dtype=np.float32)
            out_cell_rangeflag = np.full(numtracks, fillval, dtype=np.short)

        # Satellite Tb
        if feature_type == "tb_pf":
            out_corecold_mintb = np.full(numtracks, fillval_f, dtype=np.float32)
            out_corecold_meantb = np.full(numtracks, fillval_f, dtype=np.float32)
            out_core_meantb = np.full(numtracks, fillval_f, dtype=np.float32)
            out_core_area = np.full(numtracks, fillval_f, dtype=np.float32)
            out_cold_area = np.full(numtracks, fillval_f, dtype=np.float32)


        # Pre-sort cloudnumber to get location indices
        fcn_gt_0 = file_corecold_cloudnumber > 0
        corecold_cloudnumber_mask = file_corecold_cloudnumber * fcn_gt_0
        cloudnumber1d_uniq, cloudnumber1d_counts, \
        ast_corecoldarea, cumcounts_corecoldarea = pre_sort_cloudnumber(corecold_cloudnumber_mask)

        if feature_type == "radar_cells":
            # Pre-sort core number to get location indices
            core_cloudnumber_mask = file_corecold_cloudnumber * file_conv_core
            corenumber1d_uniq, corenumber1d_counts, \
            ast_corearea, cumcounts_corearea = pre_sort_cloudnumber(core_cloudnumber_mask)

            # Pre-sort dilated cell number to get location indices
            dilated_cloudnumber_mask = ds[feature_varname].squeeze().values
            dilatednumber1d_uniq, dilatednumber1d_counts, \
            ast_dilatedcellarea, cumcounts_dilatedcellarea = pre_sort_cloudnumber(dilated_cloudnumber_mask)

        if feature_type == "tb_pf":
            # Pre-sort core number to get location indices
            core_cloudnumber_mask = file_corecold_cloudnumber * (file_cloudtype == 1)
            corenumber1d_uniq, corenumber1d_counts, \
            ast_corearea, cumcounts_corearea = pre_sort_cloudnumber(core_cloudnumber_mask)

            # Pre-sort cold anvil number to get location indices
            cold_cloudnumber_mask = file_corecold_cloudnumber * (file_cloudtype == 2)
            coldnumber1d_uniq, coldnumber1d_counts, \
            ast_coldarea, cumcounts_coldarea = pre_sort_cloudnumber(cold_cloudnumber_mask)

            # import matplotlib.pyplot as plt
            # Zm = np.ma.masked_where(cold_cloudnumber_mask == 0, cold_cloudnumber_mask)
            # import pdb;
            # pdb.set_trace()
            # plt.pcolormesh(Zm)


        # Loop over unique tracknumbers
        for itrack in range(numtracks):
            # Map the tracknumbers in this frame to cloudnumbers
            cloudnumber_map = np.where(tracknumbers == uniquetracknumbers[itrack])[0] + 1
            cloudindex = cloudnumber_map - 1

            # # Get the cloudmask for the current track (this is the slow method!)
            # mask_corecold = file_corecold_cloudnumber == cloudnumber_map
            # mask_cloud = file_cloudnumber == cloudnumber_map

            # Get corecold cloud pixel location indices
            corecold_npix, corecold_indices = get_loc_indices(
                cloudnumber1d_uniq, cloudnumber1d_counts,
                ast_corecoldarea, cumcounts_corecoldarea,
                cloudnumber_map, nx, ny,
            )

            if corecold_npix > 0:
                out_area[itrack] = corecold_npix * pixel_radius ** 2
                corecold_lat = latitude[corecold_indices[0], corecold_indices[1]]
                corecold_lon = longitude[corecold_indices[0], corecold_indices[1]]
                out_meanlon[itrack] = np.nanmean(corecold_lon)
                out_meanlat[itrack] = np.nanmean(corecold_lat)

                # Calculate feature specific statistics
                # Satellite Tb
                if feature_type == "tb_pf":
                    # Get cold core pixel location indices
                    core_npix, core_indices = get_loc_indices(
                        corenumber1d_uniq, corenumber1d_counts,
                        ast_corearea, cumcounts_corearea,
                        cloudnumber_map, nx, ny,
                    )

                    # Get cold anvil pixel location indices
                    cold_npix, cold_indices = get_loc_indices(
                        coldnumber1d_uniq, coldnumber1d_counts,
                        ast_coldarea, cumcounts_coldarea,
                        cloudnumber_map, nx, ny,
                    )

                    out_core_area[itrack] = core_npix * pixel_radius ** 2
                    out_cold_area[itrack] = cold_npix * pixel_radius ** 2
                    out_corecold_mintb[itrack] = np.nanmin(file_tb[corecold_indices[0], corecold_indices[1]])
                    out_corecold_meantb[itrack] = np.nanmean(file_tb[corecold_indices[0], corecold_indices[1]])
                    if core_npix > 0:
                        out_core_meantb[itrack] = np.nanmean(file_tb[core_indices[0], core_indices[1]])

                    # iy_min, iy_max = np.min(corecold_indices[0]), np.max(corecold_indices[0])
                    # ix_min, ix_max = np.min(corecold_indices[1]), np.max(corecold_indices[1])
                    # import pdb; pdb.set_trace()

                # Calculate feature specific statistics
                # Radar cells
                if feature_type == "radar_cells":
                    # Get core pixel location indices
                    core_npix, core_indices = get_loc_indices(
                        corenumber1d_uniq, corenumber1d_counts,
                        ast_corearea, cumcounts_corearea,
                        cloudnumber_map, nx, ny,
                    )

                    # Get dilated cell pixel location indices
                    dilatedcell_npix, dilatedcell_indices = get_loc_indices(
                        dilatednumber1d_uniq, dilatednumber1d_counts,
                        ast_dilatedcellarea, cumcounts_dilatedcellarea,
                        cloudnumber_map, nx, ny,
                    )

                    # Location of core
                    core_lat = latitude[core_indices[0], core_indices[1]]
                    core_lon = longitude[core_indices[0], core_indices[1]]
                    core_y = y_coords[core_indices[0]]
                    core_x = x_coords[core_indices[1]]

                    # Location of cell (same as corecold location)
                    cell_lat = corecold_lat
                    cell_lon = corecold_lon
                    cell_y = y_coords[corecold_indices[0]]
                    cell_x = x_coords[corecold_indices[1]]

                    # Core center location
                    out_core_meanlat[itrack] = np.nanmean(core_lat)
                    out_core_meanlon[itrack] = np.nanmean(core_lon)
                    out_core_mean_y[itrack] = np.nanmean(core_y)
                    out_core_mean_x[itrack] = np.nanmean(core_x)

                    # Cell center location
                    out_cell_meanlat[itrack] = np.nanmean(cell_lat)
                    out_cell_meanlon[itrack] = np.nanmean(cell_lon)
                    out_cell_mean_y[itrack] = np.nanmean(cell_y)
                    out_cell_mean_x[itrack] = np.nanmean(cell_x)

                    out_core_area[itrack] = core_npix * pixel_radius ** 2
                    out_cell_area[itrack] = corecold_npix * pixel_radius ** 2

                    out_cell_max_dbz[itrack] = np.nanmax(
                        file_dbz[corecold_indices[0], corecold_indices[1]]
                    )

                    # Ignore "All-NaN slice encountered" error in ETH variables
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)

                        out_cell_maxETH10dbz[itrack] = np.nanmax(
                            file_echotop10[corecold_indices[0], corecold_indices[1]]
                        )
                        out_cell_maxETH20dbz[itrack] = np.nanmax(
                            file_echotop20[corecold_indices[0], corecold_indices[1]]
                        )
                        out_cell_maxETH30dbz[itrack] = np.nanmax(
                            file_echotop30[corecold_indices[0], corecold_indices[1]]
                        )
                        out_cell_maxETH40dbz[itrack] = np.nanmax(
                            file_echotop40[corecold_indices[0], corecold_indices[1]]
                        )
                        out_cell_maxETH50dbz[itrack] = np.nanmax(
                            file_echotop50[corecold_indices[0], corecold_indices[1]]
                        )

                    if terrain_file is not None:
                        # The min range mask value within the dilated cell area
                        # 1: cell completely within range mask
                        # 0: some portion of the cell outside range mask
                        out_cell_rangeflag[itrack] = np.min(
                            rangemask[dilatedcell_indices[0], dilatedcell_indices[1]])

            out_basetime[itrack] = file_basetime
            out_cloudnumber[itrack] = cloudnumber_map
            # out_cloudidfile[itrack][:] = fname
            # out_area[itrack] = np.count_nonzero(mask_corecold) * pixel_radius ** 2
            # out_meanlat[itrack] = np.nanmean(latitude[mask_corecold])
            # out_meanlon[itrack] = np.nanmean(longitude[mask_corecold])
            # out_cell_max_dbz[itrack] = np.nanmax(file_dbz[mask_corecold])

            # Save track status, merge/split information
            out_status[itrack] = trackstatus[cloudindex]
            out_mergenumber[itrack] = trackmerge[cloudindex]
            out_splitnumber[itrack] = tracksplit[cloudindex]
            out_trackinterruptions[itrack] = trackreset[cloudindex]

        # Track status explanation
        track_status_explanation = (
            "0: Track stops;  "
            + "1: Simple track continuation;  "
            + "2: This is the bigger cloud in simple merger;  "
            + "3: This is the bigger cloud from a simple split that stops at this time;  "
            + "4: This is the bigger cloud from a split and this cloud continues to the next time;  "
            + "5: This is the bigger cloud from a split that subsequently is the big cloud in a merger;  "
            + "13: This cloud splits at the next time step;  "
            + "15: This cloud is the bigger cloud in a merge that then splits at the next time step;  "
            + "16: This is the bigger cloud in a split that then splits at the next time step;  "
            + "18: Merge-split at same time (big merge, splitter, and big split);  "
            + "21: This is the smaller cloud in a simple merger;  "
            + "24: This is the bigger cloud of a split that is then the small cloud in a merger;  "
            + "31: This is the smaller cloud in a simple split that stops;  "
            + "32: This is a small split that continues onto the next time step;  "
            + "33: This is a small split that then is the bigger cloud in a merger;  "
            + "34: This is the small cloud in a merger that then splits at the next time step;  "
            + "37: Merge-split at same time (small merge, splitter, big split);  "
            + "44: This is the smaller cloud in a split that is smaller cloud in a merger at the next time step;  "
            + "46: Merge-split at same time (big merge, splitter, small split);  "
            + "52: This is the smaller cloud in a split that is smaller cloud in a merger at the next time step;  "
            + "65: Merge-split at same time (smaller merge, splitter, small split)"
        )

        # Define baseline output variables and attributes dictionary
        out_dict, \
        out_dict_attrs = define_base_vars_dict(file_basetime, fillval, fillval_f, numtracks, out_area,
                                               out_basetime, out_cloudnumber, out_meanlat, out_meanlon,
                                               out_mergenumber, out_splitnumber, out_status,
                                               out_trackinterruptions, track_status_explanation,
                                               uniquetracknumbers)

        # Define feature specific extra variables and attributes,
        # and update the baseline dictionaries
        if feature_type == "tb_pf":
            out_dict_attrs_extra, \
            out_dict_extra = define_extra_tb(fillval_f, out_cold_area, out_core_area,
                                             out_core_meantb, out_corecold_meantb,
                                             out_corecold_mintb)
            # Merge with the baseline dictionaries
            out_dict.update(out_dict_extra)
            out_dict_attrs.update(out_dict_attrs_extra)

        if feature_type == "radar_cells":
            out_dict_attrs_extra, \
            out_dict_extra = define_extra_radar_cells(fillval, fillval_f, out_cell_area,
                                                      out_cell_maxETH10dbz, out_cell_maxETH20dbz,
                                                      out_cell_maxETH30dbz, out_cell_maxETH40dbz,
                                                      out_cell_maxETH50dbz, out_cell_max_dbz,
                                                      out_cell_mean_x, out_cell_mean_y,
                                                      out_cell_meanlat, out_cell_meanlon,
                                                      out_cell_rangeflag, out_core_area,
                                                      out_core_mean_x, out_core_mean_y,
                                                      out_core_meanlat, out_core_meanlon,
                                                      rangemask_varname)
            # Merge with the baseline dictionaries
            out_dict.update(out_dict_extra)
            out_dict_attrs.update(out_dict_attrs_extra)

    else:
        logger.info("No tracks in file.")
        out_dict = None
        out_dict_attrs = None

    # import pdb; pdb.set_trace()
    return (out_dict, out_dict_attrs)





def define_base_vars_dict(file_basetime, fillval, fillval_f, numtracks, out_area, out_basetime, out_cloudnumber,
                          out_meanlat, out_meanlon, out_mergenumber, out_splitnumber, out_status,
                          out_trackinterruptions, track_status_explanation, uniquetracknumbers):
    """
    Define baseline output variables and attributes dictionary.

    Args:
        file_basetime:
        fillval:
        fillval_f:
        numtracks:
        out_area:
        out_basetime:
        out_cloudnumber:
        out_meanlat:
        out_meanlon:
        out_mergenumber:
        out_splitnumber:
        out_status:
        out_trackinterruptions:
        track_status_explanation:
        uniquetracknumbers:

    Returns:
        out_dict: dictionary
            Output variable dictionary.
        out_dict_attrs: dictionary
            Output variable attributes dictionary.

    """
    # Group outputs in dictionaries
    out_dict = {
        "uniquetracknumbers": uniquetracknumbers,
        "numtracks": numtracks,
        "base_time": out_basetime,
        "meanlat": out_meanlat,
        "meanlon": out_meanlon,
        "area": out_area,
        "cloudnumber": out_cloudnumber,
        "track_status": out_status,
        "track_interruptions": out_trackinterruptions,
        "merge_tracknumbers": out_mergenumber,
        "split_tracknumbers": out_splitnumber,
    }
    out_dict_attrs = {
        "uniquetracknumbers": {
            "long_name": "Unique track numbers in the current time frame",
            "units": "unitless",
        },
        "numtracks": {
            "long_name": "Number of tracks in the current time frame",
            "units": "unitless",
        },
        "base_time": {
            "long_name": "Epoch time of a feature",
            "units": file_basetime.attrs["units"],
            "_FillValue": fillval,
        },
        "meanlat": {
            "long_name": "Mean latitude of a feature",
            "units": "degrees_north",
            "_FillValue": fillval_f,
        },
        "meanlon": {
            "long_name": "Mean longitude of a feature",
            "units": "degrees_east",
            "_FillValue": fillval_f,
        },
        "area": {
            "long_name": "Area of a feature",
            "units": "km^2",
            "_FillValue": fillval_f,
        },
        "cloudnumber": {
            "long_name": "Corresponding cloud number in cloudid file",
            "units": "unitless",
            "_FillValue": fillval,
        },
        "track_status": {
            "long_name": "Flag indicating the status of a track",
            "units": "unitless",
            "comments": track_status_explanation,
            "_FillValue": fillval,
        },
        "track_interruptions": {
            "long_name": "0 = full track available, good data. " + \
                         "1 = track starts at first file, track cut short by data availability. " + \
                         "2 = track ends at last file, track cut short by data availability",
            "units": "unitless",
            "_FillValue": fillval,
        },
        "merge_tracknumbers": {
            "long_name": "Number of the track that this small cloud merges into",
            "units": "unitless",
            "_FillValue": fillval,
            "comments": "Each row represents a track. Each column represets a cloud in that track. " + \
                        "Numbers give the track number that this small cloud merged into."
        },
        "split_tracknumbers": {
            "long_name": "Number of the track that this small cloud splits from",
            "units": "unitless",
            "_FillValue": fillval,
            "comments": "Each row represents a track. Each column represets a cloud in that track. " + \
                        "Numbers give the track number that his msallcloud splits from."
        },
    }
    return out_dict, out_dict_attrs


def define_extra_tb(fillval_f, out_cold_area, out_core_area, out_core_meantb, out_corecold_meantb,
                    out_corecold_mintb):
    """
    Define extra output variables and attributes dictionary for satellite Tb.

    Args:
        fillval_f:
        out_cold_area:
        out_core_area:
        out_core_meantb:
        out_corecold_meantb:
        out_corecold_mintb:

    Returns:
        out_dict_attrs_extra: dictionary.
            Output variable dictionary.
        out_dict_extra: dictionary.
            Output variable attributes dictionary.
    """
    out_dict_extra = {
        "core_area": out_core_area,
        "cold_area": out_cold_area,
        "corecold_mintb": out_corecold_mintb,
        "corecold_meantb": out_corecold_meantb,
        "core_meantb": out_core_meantb,
    }
    out_dict_attrs_extra = {
        "core_area": {
            "long_name": "Area of cold core",
            "units": "km^2",
            "_FillValue": fillval_f,
        },
        "cold_area": {
            "long_name": "Area of cold anvil",
            "units": "km^2",
            "_FillValue": fillval_f,
        },
        "corecold_mintb": {
            "long_name": "Minimum Tb in cold core + cold anvil area",
            "units": "K",
            "_FillValue": fillval_f,
        },
        "corecold_meantb": {
            "long_name": "Mean Tb in cold core + cold anvil area",
            "units": "K",
            "_FillValue": fillval_f,
        },
        "core_meantb": {
            "long_name": "Mean Tb in cold core area",
            "units": "K",
            "_FillValue": fillval_f,
        },
    }
    return out_dict_attrs_extra, out_dict_extra

def define_extra_radar_cells(fillval, fillval_f, out_cell_area, out_cell_maxETH10dbz, out_cell_maxETH20dbz,
                             out_cell_maxETH30dbz, out_cell_maxETH40dbz, out_cell_maxETH50dbz, out_cell_max_dbz,
                             out_cell_mean_x, out_cell_mean_y, out_cell_meanlat, out_cell_meanlon, out_cell_rangeflag,
                             out_core_area, out_core_mean_x, out_core_mean_y, out_core_meanlat, out_core_meanlon,
                             rangemask_varname):
    """
    Define extra output variables and attributes dictionary for radar cells.

    Args:
        fillval:
        fillval_f:
        out_cell_area:
        out_cell_maxETH10dbz:
        out_cell_maxETH20dbz:
        out_cell_maxETH30dbz:
        out_cell_maxETH40dbz:
        out_cell_maxETH50dbz:
        out_cell_max_dbz:
        out_cell_mean_x:
        out_cell_mean_y:
        out_cell_meanlat:
        out_cell_meanlon:
        out_cell_rangeflag:
        out_core_area:
        out_core_mean_x:
        out_core_mean_y:
        out_core_meanlat:
        out_core_meanlon:
        rangemask_varname:

    Returns:
        out_dict_attrs_extra: dictionary.
            Output variable dictionary.
        out_dict_extra: dictionary.
            Output variable attributes dictionary.
    """
    out_dict_extra = {
        "core_meanlat": out_core_meanlat,
        "core_meanlon": out_core_meanlon,
        "core_mean_y": out_core_mean_y,
        "core_mean_x": out_core_mean_x,
        "cell_meanlat": out_cell_meanlat,
        "cell_meanlon": out_cell_meanlon,
        "cell_mean_y": out_cell_mean_y,
        "cell_mean_x": out_cell_mean_x,
        "core_area": out_core_area,
        "cell_area": out_cell_area,
        "max_dbz": out_cell_max_dbz,
        "maxETH_10dbz": out_cell_maxETH10dbz,
        "maxETH_20dbz": out_cell_maxETH20dbz,
        "maxETH_30dbz": out_cell_maxETH30dbz,
        "maxETH_40dbz": out_cell_maxETH40dbz,
        "maxETH_50dbz": out_cell_maxETH50dbz,
        "maxrange_flag": out_cell_rangeflag,
    }
    out_dict_attrs_extra = {
        "core_meanlat": {
            "long_name": "Mean latitude of a convective core",
            "units": "degrees_north",
            "_FillValue": fillval_f,
        },
        "core_meanlon": {
            "long_name": "Mean longitude of a convective core",
            "units": "degrees_east",
            "_FillValue": fillval_f,
        },
        "core_mean_y": {
            "long_name": "Mean y-distance to radar for a convective core",
            "units": "km",
            "_FillValue": fillval_f,
        },
        "core_mean_x": {
            "long_name": "Mean x-distance to radar for a convective core",
            "units": "km",
            "_FillValue": fillval_f,
        },
        "cell_meanlat": {
            "long_name": "Mean latitude of a convective cell",
            "units": "degrees_north",
            "_FillValue": fillval_f,
        },
        "cell_meanlon": {
            "long_name": "Mean longitude of a convective cell",
            "units": "degrees_east",
            "_FillValue": fillval_f,
        },
        "cell_mean_y": {
            "long_name": "Mean y-distance to radar for a convective cell",
            "units": "km",
            "_FillValue": fillval_f,
        },
        "cell_mean_x": {
            "long_name": "Mean x-distance to radar for a convective cell",
            "units": "km",
            "_FillValue": fillval_f,
        },
        "core_area": {
            "long_name": "Area of a convective core",
            "units": "km^2",
            "_FillValue": fillval_f,
        },
        "cell_area": {
            "long_name": "Area of a convective cell",
            "units": "km^2",
            "_FillValue": fillval_f,
        },
        "max_dbz": {
            "long_name": "Maximum reflectivity in a convective cell",
            "units": "dBZ",
            "_FillValue": fillval_f,
        },
        "maxETH_10dbz": {
            "long_name": "Maximum 10dBZ echo-top height in a convective cell",
            "units": "km",
            "_FillValue": fillval_f,
        },
        "maxETH_20dbz": {
            "long_name": "Maximum 20dBZ echo-top height in a convective cell",
            "units": "km",
            "_FillValue": fillval_f,
        },
        "maxETH_30dbz": {
            "long_name": "Maximum 30dBZ echo-top height in a convective cell",
            "units": "km",
            "_FillValue": fillval_f,
        },
        "maxETH_40dbz": {
            "long_name": "Maximum 40dBZ echo-top height in a convective cell",
            "units": "km",
            "_FillValue": fillval_f,
        },
        "maxETH_50dbz": {
            "long_name": "Maximum 50dBZ echo-top height in a convective cell",
            "units": "km",
            "_FillValue": fillval_f,
        },
        "maxrange_flag": {
            "long_name": "Flag indicating if tracked cell is at the maximum range of the radar",
            "units": "unitless",
            "_FillValue": fillval,
            "comments": "0 = cell partially outside range mask; 1 = cell completely within range mask",
            "rangemask_varname": rangemask_varname,
        },
    }
    return out_dict_attrs_extra, out_dict_extra


def get_loc_indices(
        cloudnumber1d_uniq,
        cloudnumber1d_counts,
        ast_cloudarea,
        cumcounts_cloudarea,
        cloudnumber_map,
        nx,
        ny):
    """
    Get the 2D pixel location indices for a given cloudnumber from a pre-sorted list.

    Args:
        cloudnumber1d_uniq: numpy array
            Unique cloudnumbers in the current pixel file.
        cloudnumber1d_counts: numpy array
            Pixel counts (area) of each unique cloud.
        ast_cloudarea: numpy array
            Cloud area flatten 1D indices sorted by cloud size.
        cumcounts_cloudarea: numpy array
            Cumulative counts for each cloud area.
        cloudnumber_map: int
            Cloud number value.
        nx: int
            Number of grids in the x-direction.
        ny: int
            Number of grids in the y-direction.

    Returns:
        corecold_npix:
            Number of pixels for the given cloudnumber.
        indices: tuple
            A tuple containing location indices [y_indices, x_indices]

    """
    # Find index of pre-sorted cloudnumber matching the current cloud
    idx = np.where(cloudnumber1d_uniq == cloudnumber_map)[0]
    if len(idx) > 0:
        corecold_npix = cloudnumber1d_counts[idx]

        # We use this to know where to index into the sorted list
        # idx > 0 excludes background non-cloud area [0]
        # unravel_index turns 1D indices back to 2D so they can be applied to 2D array
        # to access the original image data
        if idx > 0:
            indices = np.unravel_index(
                ast_cloudarea[
                cumcounts_cloudarea[idx - 1][0]:
                cumcounts_cloudarea[idx][0]
                ],
                (ny, nx),
            )
        else:
            indices = np.unravel_index(
                ast_cloudarea[0: cumcounts_cloudarea[idx][0]],
                (ny, nx),
            )
    else:
        corecold_npix = 0
        indices = None
    return (corecold_npix, indices)


def pre_sort_cloudnumber(cloudnumber_mask):
    """
    Pre-sort cloudnumber image to get pixel location indices for each cloud.

    Args:
        cloudnumber_mask: numpy array
            Cloudnumber 2D image array from pixel file.

    Returns:
        cloudnumber1d_uniq: numpy array
            Unique cloudnumbers in the current pixel file.
        cloudnumber1d_counts: numpy array
            Pixel counts (area) of each unique cloud.
        ast_cloudarea: numpy array
            Cloud area flatten 1D indices sorted by cloud size.
        cumcounts_cloudarea: numpy array
            Cumulative counts for each cloud area.

    """
    # Get unique cloudnumbers and their size (pixel counts)
    cloudnumber1d_uniq, cloudnumber1d_counts = np.unique(cloudnumber_mask, return_counts=True)
    # Sort the 2D cloudnumber and flatten to 1D indices
    # These indices are the pixel locations of each cloud
    ast_cloudarea = np.argsort(cloudnumber_mask, axis=None)
    # Apply cumulative sum on cloud size (pixel counts) to get the 1D indices start/end
    # for pixels belonging to each cloud
    cumcounts_cloudarea = np.cumsum(cloudnumber1d_counts)
    return (cloudnumber1d_uniq,
            cloudnumber1d_counts,
            ast_cloudarea,
            cumcounts_cloudarea)


def adjust_mergesplit_numbers(
        out_mergenumber,
        out_splitnumber,
        trackidx_keep,
        fillval,
):
    """
    Adjust merge, split track numbers caused by removing certain tracks.

    Args:
        out_mergenumber: numpy array
            Original merge track number.
        out_splitnumber: numpy array
            Original split track number.
        trackidx_keep: numpy array
            Track indices that are kept.
        fillval: int
            Default fill value for int arrays.

    Returns:
        adjusted_out_mergenumber:
            Adjusted merge track number.
        adjusted_out_splitnumber:
            Adjusted split track number.

    """
    numtracks = len(trackidx_keep)
    # Initialize adjusted matrices
    # adjusted_out_mergenumber = np.full(np.shape(out_mergenumber), fillval, dtype=np.int32)
    # adjusted_out_splitnumber = np.full(np.shape(out_mergenumber), fillval, dtype=np.int32)
    # logger.info(("total tracks: " + str(numtracks)))

    # Create adjuster
    indexcloudnumber = np.copy(trackidx_keep) + 1
    # adjuster = np.arange(0, np.max(trackidx_keep) + 2)
    # Modified by Zhixiao, initialize adjuster by fill values, rather than using np.arange
    adjuster = np.full(np.max(trackidx_keep) + 2, fillval, dtype=np.int32)
    for it in range(0, numtracks):
        adjuster[indexcloudnumber[it]] = it + 1
    adjuster = np.append(adjuster, np.int32(fillval))

    # Adjust mergers
    # temp_out_mergenumber = out_mergenumber.astype(np.int32).ravel()
    temp_out_mergenumber = np.ravel(out_mergenumber.astype(np.int32))
    temp_out_mergenumber[temp_out_mergenumber == fillval] = (
        np.max(trackidx_keep) + 2
    )
    # Apply
    adjusted_out_mergenumber = adjuster[temp_out_mergenumber]
    adjusted_out_mergenumber = np.reshape(
        adjusted_out_mergenumber, np.shape(out_mergenumber)
    )

    # Adjust splitters
    # temp_out_splitnumber = out_splitnumber.astype(np.int32).ravel()
    temp_out_splitnumber = np.ravel(out_splitnumber.astype(np.int32))
    temp_out_splitnumber[temp_out_splitnumber == fillval] = (
        np.max(trackidx_keep) + 2
    )
    adjusted_out_splitnumber = adjuster[temp_out_splitnumber]
    adjusted_out_splitnumber = np.reshape(
        adjusted_out_splitnumber, np.shape(out_splitnumber)
    )

    return (adjusted_out_mergenumber,
            adjusted_out_splitnumber)

def get_track_startend_status(
        out_dict,
        out_dict_attrs,
        fillval,
        max_trackduration,
        min_dt_thresh=1.0,
):
    """
    Get various track start & end point data.

    Args:
        out_dict: dictionary
            Dictionary containing the track statistics data.
        out_dict_attrs: dictionary
            Dictionary containing the attributes of track statistics data.
        fillval: int
            Default fill value for int arrays.
        max_trackduration: int
            Maximum track duration.
        min_dt_thresh: float, default=1.0
            Minimum time difference [seconds] allowed to match base time.

    Returns:
        out_dict: dictionary
            Updated dictionary containing the track statistics data.

    """
    logger = logging.getLogger(__name__)

    out_tracklength = out_dict["track_duration"]
    numtracks = len(out_tracklength)

    # Starting status
    out_startbasetime = np.full(numtracks, np.nan, dtype=np.float64)
    out_startstatus = np.full(numtracks, fillval, dtype=np.int32)
    out_startsplit_tracknumber = np.full(numtracks, fillval, dtype=np.int32)
    out_startsplit_timeindex = np.full(numtracks, fillval, dtype=np.int32)
    out_startsplit_cloudnumber = np.full(numtracks, fillval, dtype=np.int32)

    # Some tracks have 0 tracklength (no data in sparse arrays)
    # causing inconsistency in the number of tracks
    # This makes sure numtracks match
    trackid_real = np.where(out_tracklength > 0)[0]
    out_startbasetime[trackid_real] = out_dict["base_time"][:, 0].data
    out_startstatus[trackid_real] = out_dict["track_status"][:, 0].data
    out_startsplit_tracknumber[trackid_real] = out_dict["split_tracknumbers"][:, 0].data

    # Ending status
    out_endbasetime = np.full(numtracks, np.nan, dtype=np.float64)
    out_endstatus = np.full(numtracks, fillval, dtype=np.int32)
    out_endmerge_tracknumber = np.full(numtracks, fillval, dtype=np.int32)
    out_endmerge_timeindex = np.full(numtracks, fillval, dtype=np.int32)
    out_endmerge_cloudnumber = np.full(numtracks, fillval, dtype=np.int32)

    # Loop over each track
    for itrack in range(0, numtracks):

        # Make sure the track length is < max_trackduration
        # so array access would not be out of bounds
        if (out_tracklength[itrack] > 0) & \
                (out_tracklength[itrack] < max_trackduration):

            # Get the end basetime
            out_endbasetime[itrack] = out_dict["base_time"][
                itrack, out_tracklength[itrack] - 1
            ]
            # Get the status at the last time step of the track
            out_endstatus[itrack] = np.copy(
                out_dict["track_status"][itrack, out_tracklength[itrack] - 1]
            )
            out_endmerge_tracknumber[itrack] = np.copy(
                out_dict["merge_tracknumbers"][
                    itrack, out_tracklength[itrack] - 1
                ]
            )

            # If end merge tracknumber exists, this track ends by merge
            if out_endmerge_tracknumber[itrack] >= 0:
                # Get the track number if merges with, -1 convert to track index
                imerge_idx = out_endmerge_tracknumber[itrack] - 1
                # Get all the basetime for the track it merges with
                ibasetime = out_dict["base_time"][
                            imerge_idx, 0: out_tracklength[imerge_idx]
                            ].data

                # Find the closest time matching the time when merging occurs
                # If the time difference is < min_dt_thresh, consider it the same
                dt = np.abs(ibasetime - out_endbasetime[itrack])
                if np.nanmin(dt) < min_dt_thresh:
                    match_timeidx = np.nanargmin(dt)
                    #  The time to connect to the track it merges with should be 1 time step after
                    if ((match_timeidx + 1) >= 0) & (
                            (match_timeidx + 1) < max_trackduration
                    ):
                        out_endmerge_timeindex[itrack] = match_timeidx + 1
                        out_endmerge_cloudnumber[
                            itrack
                        ] = out_dict["cloudnumber"][imerge_idx, match_timeidx + 1]
                    else:
                        logger.debug(f"Merge time occur after track ends??")
                else:
                    # import pdb; pdb.set_trace()
                    logger.debug(
                        f"Error: track {itrack} has no matching time in the track it merges with!"
                    )
                    # import pdb; pdb.set_trace()
                    sys.exit(itrack)

            # If start split tracknumber exists, this track starts from a split
            if out_startsplit_tracknumber[itrack] >= 0:
                # Get the tracknumber it splits from, -1 to convert to track index
                isplit_idx = out_startsplit_tracknumber[itrack] - 1
                # Get all the basetime for the track it splits from
                ibasetime = out_dict["base_time"][
                            isplit_idx, 0: out_tracklength[isplit_idx]
                            ].data

                # Find the time index matching the time when splitting occurs
                # match_timeidx = np.where(ibasetime == out_startbasetime[itrack])[0]

                # Find the closest time matching the time when merging occurs
                # If the time difference is < min_dt_thresh, consider it the same
                dt = np.abs(ibasetime - out_startbasetime[itrack])
                if np.nanmin(dt) < min_dt_thresh:
                    match_timeidx = np.nanargmin(dt)

                    # Modified by Zhixiao: If there is no overlap time between split and reference tracks,
                    # we test whether the reference cell end time is a time step earlier than the split cell initiation.
                    # We take this as a resonable split, because the referecence cell can merge with other cells
                    # after the split moment.
                    # if len(match_timeidx) == 0:
                    #     # Zhixiao
                    #     match_timeidx = np.where(ibasetime == ibasetime[len(ibasetime) - 1])[0]
                    #     if len(match_timeidx) == 1:
                    #         match_timeidx = match_timeidx + 1
                    #         logger.debug(
                    #             f"Note: Track {itrack} splits from transient parent cloud."
                    #         )

                    # if len(match_timeidx) == 1:
                    # The time to connect to the track it splits from should be 1 time step prior
                    if (match_timeidx - 1) >= 0:
                        out_startsplit_timeindex[itrack] = match_timeidx - 1
                        out_startsplit_cloudnumber[
                            itrack
                        ] = out_dict["cloudnumber"][isplit_idx, match_timeidx - 1]
                    else:
                        logger.debug(f"Split time occur before track starts??")
                    # else:
                    #     logger.debug(
                    #         f"Error: track {itrack} has no matching time in the track it splits from!"
                    #     )
                    #     sys.exit(itrack)

    # Add new variables to the dictionary
    out_dict["start_status"] = out_startstatus
    out_dict["end_status"] = out_endstatus
    out_dict["start_basetime"] = out_startbasetime
    out_dict["end_basetime"] = out_endbasetime
    out_dict["start_split_tracknumber"] = out_startsplit_tracknumber
    out_dict["start_split_timeindex"] = out_startsplit_timeindex
    out_dict["start_split_cloudnumber"] = out_startsplit_cloudnumber
    out_dict["end_merge_tracknumber"] = out_endmerge_tracknumber
    out_dict["end_merge_timeindex"] = out_endmerge_timeindex
    out_dict["end_merge_cloudnumber"] = out_endmerge_cloudnumber

    # Add new attributes to the dictionary
    out_dict_attrs["start_status"] = {
        "long_name": "Flag indicating how the track starts",
        "units": "unitless",
        "_FillValue": fillval,
        "comments": "Refer to track_status attributes on flag value meaning",
    }
    out_dict_attrs["end_status"] = {
        "long_name": "Flag indicating how the track ends",
        "units": "unitless",
        "_FillValue": fillval,
        "comments": "Refer to track_status attributes on flag value meaning",
    }
    out_dict_attrs["start_basetime"] = {
        "long_name": "Start Epoch time of each track",
        "units": out_dict_attrs["base_time"]["units"],
        "_FillValue": fillval,
    }
    out_dict_attrs["end_basetime"] = {
        "long_name": "End Epoch time of each track",
        "units": out_dict_attrs["base_time"]["units"],
        "_FillValue": fillval,
    }
    out_dict_attrs["start_split_tracknumber"] = {
        "long_name": "Tracknumber where this track splits from",
        "units": "unitless",
        "_FillValue": fillval,
        "comments": "track_index = tracknumber - 1"
    }
    out_dict_attrs["start_split_timeindex"] = {
        "long_name": "Time index when split occurs",
        "units": "unitless",
        "_FillValue": fillval,
        "comments": "To connect with the split track, use start_split_tracknumber, start_split_timeindex together"
    }
    out_dict_attrs["start_split_cloudnumber"] = {
        "long_name": "Cloud number where this track splits from",
        "units": "unitless",
        "_FillValue": fillval,
    }
    out_dict_attrs["end_merge_tracknumber"] = {
        "long_name": "Tracknumber where this track merges with",
        "units": "unitless",
        "_FillValue": fillval,
    }
    out_dict_attrs["end_merge_timeindex"] = {
        "long_name": "Time index when merge occurs",
        "units": "unitless",
        "_FillValue": fillval,
    }
    out_dict_attrs["end_merge_cloudnumber"] = {
        "long_name": "Cloud number where this track merges with",
        "units": "unitless",
        "_FillValue": fillval,
    }
    return (out_dict, out_dict_attrs)