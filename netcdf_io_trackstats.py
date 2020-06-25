import time
import numpy as np
import xarray as xr

def write_trackstats_tb(trackstats_outfile, numtracks, maxtracklength, nbintb, numcharfilename, \
                        datasource, datadescription, startdate, enddate, \
                        track_version, tracknumbers_version, timegap, \
                        thresh_core, thresh_cold, pixel_radius, geolimits, areathresh, \
                        mintb_thresh, maxtb_thresh, \
                        basetime_units, basetime_calendar, \
                        finaltrack_tracklength, finaltrack_basetime, finaltrack_cloudidfile, finaltrack_datetimestring, \
                        finaltrack_corecold_meanlat, finaltrack_corecold_meanlon, \
                        finaltrack_corecold_minlat, finaltrack_corecold_minlon, \
                        finaltrack_corecold_maxlat, finaltrack_corecold_maxlon, \
                        finaltrack_corecold_radius, finaltrack_corecoldwarm_radius, \
                        finaltrack_ncorecoldpix, finaltrack_ncorepix, finaltrack_ncoldpix, finaltrack_nwarmpix, \
                        finaltrack_corecold_cloudnumber, finaltrack_corecold_status, \
                        finaltrack_corecold_startstatus, finaltrack_corecold_endstatus, \
                        adjusted_finaltrack_corecold_mergenumber, adjusted_finaltrack_corecold_splitnumber, \
                        finaltrack_corecold_trackinterruptions, finaltrack_corecold_boundary, \
                        finaltrack_corecold_mintb, finaltrack_corecold_meantb, finaltrack_core_meantb, finaltrack_corecold_histtb, \
                        finaltrack_corecold_majoraxis, finaltrack_corecold_orientation, finaltrack_corecold_eccentricity, \
                        finaltrack_corecold_perimeter, finaltrack_corecold_xcenter, finaltrack_corecold_ycenter, \
                        finaltrack_corecold_xweightedcenter, finaltrack_corecold_yweightedcenter, \
                        ):
        """
        Writes Tb trackstats variables to netCDF file.
        """

        # Define variable list
        varlist = {'lifetime': (['ntracks'], finaltrack_tracklength), \
                        'basetime': (['ntracks', 'nmaxlength'], finaltrack_basetime), \
                        'cloudidfiles': (['ntracks', 'nmaxlength', 'nfilenamechars'], finaltrack_cloudidfile), \
                        'datetimestrings': (['ntracks', 'nmaxlength', 'ndatetimechars'], finaltrack_datetimestring), \
                        'meanlat': (['ntracks', 'nmaxlength'], finaltrack_corecold_meanlat), \
                        'meanlon': (['ntracks', 'nmaxlength'], finaltrack_corecold_meanlon), \
                        'minlat': (['ntracks', 'nmaxlength'], finaltrack_corecold_minlat), \
                        'minlon': (['ntracks', 'nmaxlength'], finaltrack_corecold_minlon), \
                        'maxlat': (['ntracks', 'nmaxlength'], finaltrack_corecold_maxlat), \
                        'maxlon': (['ntracks', 'nmaxlength'], finaltrack_corecold_maxlon), \
                        'radius': (['ntracks', 'nmaxlength'], finaltrack_corecold_radius), \
                        'radius_warmanvil': (['ntracks', 'nmaxlength'], finaltrack_corecoldwarm_radius), \
                        'npix': (['ntracks', 'nmaxlength'], finaltrack_ncorecoldpix), \
                        'nconv': (['ntracks', 'nmaxlength'], finaltrack_ncorepix), \
                        'ncoldanvil': (['ntracks', 'nmaxlength'], finaltrack_ncoldpix), \
                        'nwarmanvil': (['ntracks', 'nmaxlength'], finaltrack_nwarmpix), \
                        'cloudnumber': (['ntracks', 'nmaxlength'], finaltrack_corecold_cloudnumber), \
                        'status': (['ntracks', 'nmaxlength'], finaltrack_corecold_status), \
                        'startstatus': (['ntracks'], finaltrack_corecold_startstatus), \
                        'endstatus': (['ntracks'], finaltrack_corecold_endstatus), \
                        'mergenumbers': (['ntracks', 'nmaxlength'], adjusted_finaltrack_corecold_mergenumber), \
                        'splitnumbers': (['ntracks', 'nmaxlength'], adjusted_finaltrack_corecold_splitnumber), \
                        'trackinterruptions': (['ntracks'], finaltrack_corecold_trackinterruptions), \
                        'boundary': (['ntracks', 'nmaxlength'], finaltrack_corecold_boundary), \
                        'mintb': (['ntracks', 'nmaxlength'], finaltrack_corecold_mintb), \
                        'meantb': (['ntracks', 'nmaxlength'], finaltrack_corecold_meantb), \
                        'meantb_conv': (['ntracks', 'nmaxlength'], finaltrack_core_meantb), \
                        'histtb': (['ntracks', 'nmaxlength', 'nbins'], finaltrack_corecold_histtb), \
                        'majoraxis': (['ntracks', 'nmaxlength'], finaltrack_corecold_majoraxis), \
                        'orientation': (['ntracks', 'nmaxlength'], finaltrack_corecold_orientation), \
                        'eccentricity': (['ntracks', 'nmaxlength'], finaltrack_corecold_eccentricity), \
                        'perimeter': (['ntracks', 'nmaxlength'], finaltrack_corecold_perimeter), \
                        'xcenter': (['ntracks', 'nmaxlength'], finaltrack_corecold_xcenter), \
                        'ycenter': (['ntracks', 'nmaxlength'], finaltrack_corecold_ycenter), \
                        'xcenter_weighted': (['ntracks', 'nmaxlength'], finaltrack_corecold_xweightedcenter), \
                        'ycenter_weighted': (['ntracks', 'nmaxlength'], finaltrack_corecold_yweightedcenter)}
        
        # Define coordinate list
        coordlist = {'ntracks': (['ntracks'], np.arange(0,numtracks)), \
                        'nmaxlength': (['nmaxlength'], np.arange(0, maxtracklength)), \
                        'nbins': (['nbins'], np.arange(0, nbintb-1)), \
                        'nfilenamechars': (['nfilenamechars'], np.arange(0, numcharfilename)), \
                        'ndatetimechars': (['ndatetimechars'], np.arange(0, 13))}

        # Define global attributes
        gattrlist = {'title':  'File containing statistics for each track', \
                        'Institution': 'Pacific Northwest National Laboratoy', \
                        'Contact': 'Katelyn Barber: katelyn.barber@pnnl.gov', \
                        'Created_on':  time.ctime(time.time()), \
                        'source': datasource, \
                        'description': datadescription, \
                        'startdate': startdate, \
                        'enddate': enddate, \
                        'track_version': track_version, \
                        'tracknumbers_version': tracknumbers_version, \
                        'timegap': str(timegap)+'-hr', \
                        'tb_core': thresh_core, \
                        'tb_coldanvil': thresh_cold, \
                        'pixel_radius_km': pixel_radius}

        # Define xarray dataset
        output_data = xr.Dataset(varlist, coords=coordlist, attrs=gattrlist)

        # Specify variable attributes
        output_data.ntracks.attrs['long_name'] = 'Total number of cloud tracks'
        output_data.ntracks.attrs['units'] = 'unitless'

        output_data.nmaxlength.attrs['long_name'] = 'Maximum length of a cloud track'
        output_data.nmaxlength.attrs['units'] = 'unitless'

        output_data.lifetime.attrs['long_name'] = 'duration of each track'
        output_data.lifetime.attrs['units'] = 'Temporal resolution of data'

        output_data.basetime.attrs['long_name'] = 'epoch time of each cloud in a track'
        output_data.basetime.attrs['standard_name'] = 'time'

        output_data.cloudidfiles.attrs['long_name'] = 'File name for each cloud in each track'

        output_data.datetimestrings.attrs['long_name'] = 'date_time for for each cloud in each track'

        output_data.meanlat.attrs['long_name'] = 'Mean latitude of the core + cold anvil for each cloud in a track'
        output_data.meanlat.attrs['standard_name'] = 'latitude'
        output_data.meanlat.attrs['units'] = 'degrees_north'
        output_data.meanlat.attrs['valid_min'] = geolimits[1]
        output_data.meanlat.attrs['valid_max'] = geolimits[3]

        output_data.meanlon.attrs['long_name'] = 'Mean longitude of the core + cold anvil for each cloud in a track'
        output_data.meanlon.attrs['standard_name'] = 'longitude'
        output_data.meanlon.attrs['units'] = 'degrees_east'
        output_data.meanlon.attrs['valid_min'] = geolimits[0]
        output_data.meanlon.attrs['valid_max'] = geolimits[2]

        output_data.minlat.attrs['long_name'] = 'Minimum latitude of the core + cold anvil for each cloud in a track'
        output_data.minlat.attrs['standard_name'] = 'latitude'
        output_data.minlat.attrs['units'] = 'degrees_north'
        output_data.minlat.attrs['valid_min'] = geolimits[1]
        output_data.minlat.attrs['valid_max'] = geolimits[3]

        output_data.minlon.attrs['long_name'] = 'Minimum longitude of the core + cold anvil for each cloud in a track'
        output_data.minlon.attrs['standard_name'] = 'longitude'
        output_data.minlon.attrs['units'] = 'degrees_east'
        output_data.minlon.attrs['valid_min'] = geolimits[0]
        output_data.minlon.attrs['valid_max'] = geolimits[2]

        output_data.maxlat.attrs['long_name'] = 'Maximum latitude of the core + cold anvil for each cloud in a track'
        output_data.maxlat.attrs['standard_name'] = 'latitude'
        output_data.maxlat.attrs['units'] = 'degrees_north'
        output_data.maxlat.attrs['valid_min'] = geolimits[1]
        output_data.maxlat.attrs['valid_max'] = geolimits[3]

        output_data.maxlon.attrs['long_name'] = 'Maximum longitude of the core + cold anvil for each cloud in a track'
        output_data.maxlon.attrs['standard_name'] = 'longitude'
        output_data.maxlon.attrs['units'] = 'degrees_east'
        output_data.maxlon.attrs['valid_min'] = geolimits[0]
        output_data.maxlon.attrs['valid_max'] = geolimits[2]

        output_data.radius.attrs['long_name'] = 'Equivalent radius of the core + cold anvil for each cloud in a track'
        output_data.radius.attrs['standard_name'] = 'Equivalent radius'
        output_data.radius.attrs['units'] = 'km'
        output_data.radius.attrs['valid_min'] = areathresh

        output_data.radius_warmanvil.attrs['long_name'] = 'Equivalent radius of the core + cold anvil  + warm anvil for each cloud in a track'
        output_data.radius_warmanvil.attrs['standard_name'] = 'Equivalent radius'
        output_data.radius_warmanvil.attrs['units'] = 'km'
        output_data.radius_warmanvil.attrs['valid_min'] = areathresh

        output_data.npix.attrs['long_name'] = 'Number of pixels in the core + cold anvil for each cloud in a track'
        output_data.npix.attrs['units'] = 'unitless'
        output_data.npix.attrs['valid_min'] =  int(areathresh/float(np.square(pixel_radius)))

        output_data.nconv.attrs['long_name'] = 'Number of pixels in the core for each cloud in a track'
        output_data.nconv.attrs['units'] = 'unitless'
        output_data.nconv.attrs['valid_min'] = int(areathresh/float(np.square(pixel_radius)))

        output_data.ncoldanvil.attrs['long_name'] = 'Number of pixels in the cold anvil for each cloud in a track'
        output_data.ncoldanvil.attrs['units'] = 'unitless'
        output_data.ncoldanvil.attrs['valid_min'] = int(areathresh/float(np.square(pixel_radius)))

        output_data.nwarmanvil.attrs['long_name'] = 'Number of pixels in the warm anvil for each cloud in a track'
        output_data.nwarmanvil.attrs['units'] = 'unitless'
        output_data.nwarmanvil.attrs['valid_min'] = int(areathresh/float(np.square(pixel_radius)))

        output_data.cloudnumber.attrs['long_name'] = 'Ccorresponding cloud identification number in cloudid file for each cloud in a track'
        output_data.cloudnumber.attrs['units'] = 'unitless'
        output_data.cloudnumber.attrs['usage'] = 'To link this tracking statistics file with corresponding pixel-level cloudid files, use the cloudidfile and cloudnumber together to identify which file and cloud this track is associated with at this time'

        output_data.status.attrs['long_name'] = 'Flag indicating evolution / behavior for each cloud in a track'
        output_data.status.attrs['units'] = 'unitless'
        output_data.status.attrs['valid_min'] = 0
        output_data.status.attrs['valid_max'] = 65

        output_data.startstatus.attrs['long_name'] = 'Flag indicating how the first cloud in a track starts'
        output_data.startstatus.attrs['units'] = 'unitless'
        output_data.startstatus.attrs['valid_min'] = 0
        output_data.startstatus.attrs['valid_max'] = 65

        output_data.endstatus.attrs['long_name'] = 'Flag indicating how the last cloud in a track ends'
        output_data.endstatus.attrs['units'] = 'unitless'
        output_data.endstatus.attrs['valid_min'] = 0
        output_data.endstatus.attrs['valid_max'] = 65

        output_data.trackinterruptions.attrs['long_name'] = 'Flag indicating if track started or ended naturally or artifically due to data availability'
        output_data.trackinterruptions.attrs['values'] = '0 = full track available, good data. 1 = track starts at first file, track cut short by data availability. 2 = track ends at last file, track cut short by data availability'
        output_data.trackinterruptions.attrs['valid_min'] = 0
        output_data.trackinterruptions.attrs['valid_max'] = 2
        output_data.trackinterruptions.attrs['units'] = 'unitless'

        output_data.mergenumbers.attrs['long_name'] = 'Number of the track that this small cloud merges into'
        output_data.mergenumbers.attrs['usuage'] = 'Each row represents a track. Each column represets a cloud in that track. Numbers give the track number that this small cloud merged into.'
        output_data.mergenumbers.attrs['units'] = 'unitless'
        output_data.mergenumbers.attrs['valid_min'] = 1
        output_data.mergenumbers.attrs['valid_max'] = numtracks

        output_data.splitnumbers.attrs['long_name'] = 'Number of the track that this small cloud splits from'
        output_data.splitnumbers.attrs['usuage'] = 'Each row represents a track. Each column represets a cloud in that track. Numbers give the track number that his msallcloud splits from.'
        output_data.splitnumbers.attrs['units'] = 'unitless'
        output_data.splitnumbers.attrs['valid_min'] = 1
        output_data.splitnumbers.attrs['valid_max'] = numtracks

        output_data.boundary.attrs['long_name'] = 'Flag indicating whether the core + cold anvil touches one of the domain edges.'
        output_data.boundary.attrs['usuage'] = ' 0 = away from edge. 1= touches edge.'
        output_data.boundary.attrs['units'] = 'unitless'
        output_data.boundary.attrs['valid_min'] = 0
        output_data.boundary.attrs['valid_max'] = 1

        output_data.mintb.attrs['long_name'] = 'Minimum brightness temperature for each core + cold anvil in a track'
        output_data.mintb.attrs['standard_name'] = 'brightness temperature'
        output_data.mintb.attrs['units'] = 'K'
        output_data.mintb.attrs['valid_min'] = mintb_thresh
        output_data.mintb.attrs['valid_max'] = maxtb_thresh

        output_data.meantb.attrs['long_name'] = 'Mean brightness temperature for each core + cold anvil in a track'
        output_data.meantb.attrs['standard_name'] = 'brightness temperature'
        output_data.meantb.attrs['units'] = 'K'
        output_data.meantb.attrs['valid_min'] = mintb_thresh
        output_data.meantb.attrs['valid_max'] = maxtb_thresh

        output_data.meantb_conv.attrs['long_name'] = 'Mean brightness temperature for each core in a track'
        output_data.meantb_conv.attrs['standard_name'] = 'brightness temperature'
        output_data.meantb_conv.attrs['units'] = 'K'
        output_data.meantb_conv.attrs['valid_min'] = mintb_thresh
        output_data.meantb_conv.attrs['valid_max'] = maxtb_thresh

        output_data.histtb.attrs['long_name'] = 'Histogram of brightess of the core + cold anvil for each cloud in a track.'
        output_data.histtb.attrs['standard_name'] = 'Brightness temperature'
        output_data.histtb.attrs['hist_value'] = mintb_thresh
        output_data.histtb.attrs['valid_max'] =  maxtb_thresh
        output_data.histtb.attrs['units'] = 'K'

        output_data.orientation.attrs['long_name'] = 'Orientation of the major axis of the core + cold anvil for each cloud in a track'
        output_data.orientation.attrs['units'] = 'Degrees clockwise from vertical'
        output_data.orientation.attrs['valid_min'] = 0
        output_data.orientation.attrs['valid_max'] = 360

        output_data.eccentricity.attrs['long_name'] = 'Eccentricity of the major axis of the core + cold anvil for each cloud in a track'
        output_data.eccentricity.attrs['units'] = 'unitless'
        output_data.eccentricity.attrs['valid_min'] = 0
        output_data.eccentricity.attrs['valid_max'] = 1

        output_data.majoraxis.attrs['long_name'] =  'Length of the major axis of the core + cold anvil for each cloud in a track'
        output_data.majoraxis.attrs['units'] = 'km'

        output_data.perimeter.attrs['long_name'] = 'Approximnate circumference of the core + cold anvil for each cloud in a track'
        output_data.perimeter.attrs['units'] = 'km'

        output_data.xcenter.attrs['long_name'] = 'X index of the geometric center of the cloud feature for each cloud in a track'
        output_data.xcenter.attrs['units'] = 'unitless'

        output_data.ycenter.attrs['long_name'] = 'Y index of the geometric center of the cloud feature for each cloud in a track'
        output_data.ycenter.attrs['units'] = 'unitless'

        output_data.xcenter_weighted.attrs['long_name'] = 'X index of the brightness temperature weighted center of the cloud feature for each cloud in a track'
        output_data.xcenter_weighted.attrs['units'] = 'unitless'

        output_data.ycenter_weighted.attrs['long_name'] = 'Y index of the brightness temperature weighted center of the cloud feature for each cloud in a track'
        output_data.ycenter_weighted.attrs['units'] = 'unitless'

        # Specify encoding list
        encodelist = {'lifetime': {'dtype': 'int', 'zlib':True, '_FillValue': -9999}, \
                        'basetime': {'zlib':True, 'units': basetime_units, 'calendar': basetime_calendar}, \
                        'ntracks': {'dtype': 'int', 'zlib':True}, \
                        'nmaxlength': {'dtype': 'int', 'zlib':True}, \
                        'cloudidfiles': {'zlib':True}, \
                        'datetimestrings': {'zlib':True}, \
                        'meanlat': {'zlib':True, '_FillValue': np.nan}, \
                        'meanlon': {'zlib':True, '_FillValue': np.nan}, \
                        'minlat': {'zlib':True, '_FillValue': np.nan}, \
                        'minlon': {'zlib':True, '_FillValue': np.nan}, \
                        'maxlat': {'zlib':True, '_FillValue': np.nan}, \
                        'maxlon': {'zlib':True, '_FillValue': np.nan}, \
                        'radius': {'zlib':True, '_FillValue': np.nan}, \
                        'radius_warmanvil': {'zlib':True, '_FillValue': np.nan}, \
                        'boundary':  {'dtype': 'int', 'zlib':True, '_FillValue': -9999}, \
                        'npix': {'dtype': 'int', 'zlib':True, '_FillValue': -9999}, \
                        'nconv': {'dtype': 'int', 'zlib':True, '_FillValue': -9999}, \
                        'ncoldanvil': {'dtype': 'int','zlib':True, '_FillValue': -9999}, \
                        'nwarmanvil': {'dtype': 'int', 'zlib':True, '_FillValue': -9999}, \
                        'cloudnumber': {'dtype': 'int', 'zlib':True, '_FillValue': -9999}, \
                        'mergenumbers': {'dtype': 'int', 'zlib':True, '_FillValue': -9999}, \
                        'splitnumbers': {'dtype': 'int', 'zlib':True, '_FillValue': -9999}, \
                        'status': {'dtype': 'int', 'zlib':True, '_FillValue': -9999}, \
                        'startstatus': {'dtype': 'int', 'zlib':True, '_FillValue': -9999}, \
                        'endstatus': {'dtype': 'int', 'zlib':True, '_FillValue': -9999}, \
                        'trackinterruptions': {'dtype': 'int', 'zlib':True, '_FillValue': -9999}, \
                        'mintb': {'zlib':True, '_FillValue': np.nan}, \
                        'meantb': {'zlib':True, '_FillValue': np.nan}, \
                        'meantb_conv': {'zlib':True, '_FillValue': np.nan}, \
                        'histtb': {'dtype': 'int', 'zlib':True}, \
                        'majoraxis': {'zlib':True, '_FillValue': np.nan}, \
                        'orientation': {'zlib':True, '_FillValue': np.nan}, \
                        'eccentricity': {'zlib':True, '_FillValue': np.nan}, \
                        'perimeter': {'zlib':True, '_FillValue': np.nan}, \
                        'xcenter': {'zlib':True, '_FillValue': -9999}, \
                        'ycenter': {'zlib':True, '_FillValue': -9999}, \
                        'xcenter_weighted': {'zlib':True, '_FillValue': -9999}, \
                        'ycenter_weighted': {'zlib':True, '_FillValue': -9999}}

        # Write netcdf file
        output_data.to_netcdf(path=trackstats_outfile, mode='w', format='NETCDF4_CLASSIC', unlimited_dims='ntracks', encoding=encodelist)



def write_trackstats_radar(trackstats_outfile, numtracks, maxtracklength, numcharfilename, \
                            trackdimname, timedimname, \
                            datasource, datadescription, startdate, enddate, \
                            track_version, tracknumbers_version, timegap, basetime_units, \
                            pixel_radius, areathresh, datatimeresolution, fillval, \
                            finaltrack_tracklength, finaltrack_basetime, \
                            finaltrack_cloudidfile, finaltrack_cloudnumber, \
                            finaltrack_core_meanlat, finaltrack_core_meanlon, \
                            finaltrack_core_mean_y, finaltrack_core_mean_x, \
                            finaltrack_cell_meanlat, finaltrack_cell_meanlon, \
                            finaltrack_cell_mean_y, finaltrack_cell_mean_x, \
                            finaltrack_cell_minlat, finaltrack_cell_maxlat, \
                            finaltrack_cell_minlon, finaltrack_cell_maxlon, \
                            finaltrack_cell_min_y, finaltrack_cell_max_y, \
                            finaltrack_cell_min_x, finaltrack_cell_max_x, \
                            finaltrack_dilatecell_meanlat, finaltrack_dilatecell_meanlon, \
                            finaltrack_dilatecell_mean_y, finaltrack_dilatecell_mean_x, \
                            finaltrack_core_area, finaltrack_cell_area, \
                            finaltrack_core_radius, finaltrack_cell_radius, \
                            finaltrack_cell_maxdbz, \
                            finaltrack_cell_maxETH10dbz, finaltrack_cell_maxETH20dbz, finaltrack_cell_maxETH30dbz, \
                            finaltrack_cell_maxETH40dbz, finaltrack_cell_maxETH50dbz, \
                            finaltrack_status, finaltrack_startstatus, finaltrack_endstatus, \
                            finaltrack_trackinterruptions, \
                            finaltrack_mergenumber, finaltrack_splitnumber, \
                            
                            # finaltrack_tracklength, finaltrack_basetime, \
                            # finaltrack_cloudidfile, finaltrack_cloudnumber, \
                            # finaltrack_core_meanlat, finaltrack_core_meanlon, \
                            # finaltrack_cell_minlat, finaltrack_cell_maxlat, \
                            # finaltrack_cell_minlon, finaltrack_cell_maxlon, \
                            # finaltrack_core_area, finaltrack_cell_area, \
                            # finaltrack_core_radius, finaltrack_cell_radius, \
                            # finaltrack_cell_maxdbz, finaltrack_status, \
                            # finaltrack_startstatus, finaltrack_endstatus, \
                            # finaltrack_trackinterruptions, \
                            # finaltrack_mergenumber, finaltrack_splitnumber, \
                            ):
    """
    Write radar trackstats variables to netCDF file.
    """

    # Define variable list
    varlist = {'lifetime': ([trackdimname], finaltrack_tracklength), \
                'basetime': ([trackdimname, timedimname], finaltrack_basetime), \
                # 'cloudidfiles': ([trackdimname, timedimname, 'nfilenamechars'], finaltrack_cloudidfile), \
                # 'datetimestrings': ([trackdimname, timedimname, 'ndatetimechars'], finaltrack_datetimestring), \
                'core_meanlat': ([trackdimname, timedimname], finaltrack_core_meanlat), \
                'core_meanlon': ([trackdimname, timedimname], finaltrack_core_meanlon), \
                'core_mean_y': ([trackdimname, timedimname], finaltrack_core_mean_y), \
                'core_mean_x': ([trackdimname, timedimname], finaltrack_core_mean_x), \
                
                'cell_meanlat': ([trackdimname, timedimname], finaltrack_cell_meanlat), \
                'cell_meanlon': ([trackdimname, timedimname], finaltrack_cell_meanlon), \
                'cell_mean_y': ([trackdimname, timedimname], finaltrack_cell_mean_y), \
                'cell_mean_x': ([trackdimname, timedimname], finaltrack_cell_mean_x), \
                'cell_minlat': ([trackdimname, timedimname], finaltrack_cell_minlat), \
                'cell_minlon': ([trackdimname, timedimname], finaltrack_cell_minlon), \
                'cell_maxlat': ([trackdimname, timedimname], finaltrack_cell_maxlat), \
                'cell_maxlon': ([trackdimname, timedimname], finaltrack_cell_maxlon), \
                'cell_min_y': ([trackdimname, timedimname], finaltrack_cell_min_y), \
                'cell_min_x': ([trackdimname, timedimname], finaltrack_cell_min_x), \
                'cell_max_y': ([trackdimname, timedimname], finaltrack_cell_max_y), \
                'cell_max_x': ([trackdimname, timedimname], finaltrack_cell_max_x), \

                'dilatecell_meanlat': ([trackdimname, timedimname], finaltrack_dilatecell_meanlat), \
                'dilatecell_meanlon': ([trackdimname, timedimname], finaltrack_dilatecell_meanlon), \
                'dilatecell_mean_y': ([trackdimname, timedimname], finaltrack_dilatecell_mean_y), \
                'dilatecell_mean_x': ([trackdimname, timedimname], finaltrack_dilatecell_mean_x), \

                'core_radius': ([trackdimname, timedimname], finaltrack_core_radius), \
                'cell_radius': ([trackdimname, timedimname], finaltrack_cell_radius), \
                'core_area': ([trackdimname, timedimname], finaltrack_core_area), \
                'cell_area': ([trackdimname, timedimname], finaltrack_cell_area), \
                
                'maxdbz': ([trackdimname, timedimname], finaltrack_cell_maxdbz), \
                'maxETH_10dbz': ([trackdimname, timedimname], finaltrack_cell_maxETH10dbz), \
                'maxETH_20dbz': ([trackdimname, timedimname], finaltrack_cell_maxETH20dbz), \
                'maxETH_30dbz': ([trackdimname, timedimname], finaltrack_cell_maxETH30dbz), \
                'maxETH_40dbz': ([trackdimname, timedimname], finaltrack_cell_maxETH40dbz), \
                'maxETH_50dbz': ([trackdimname, timedimname], finaltrack_cell_maxETH50dbz), \

                'cloudnumber': ([trackdimname, timedimname], finaltrack_cloudnumber), \
                'status': ([trackdimname, timedimname], finaltrack_status), \
                'startstatus': ([trackdimname], finaltrack_startstatus), \
                'endstatus': ([trackdimname], finaltrack_endstatus), \
                'mergenumbers': ([trackdimname, timedimname], finaltrack_mergenumber), \
                'splitnumbers': ([trackdimname, timedimname], finaltrack_splitnumber), \
                'trackinterruptions': ([trackdimname], finaltrack_trackinterruptions), \
                # 'boundary': ([trackdimname], finaltrack_boundary), \
                # 'majoraxis': ([trackdimname, timedimname], finaltrack_corecold_majoraxis), \
                # 'orientation': ([trackdimname, timedimname], finaltrack_corecold_orientation), \
                # 'eccentricity': ([trackdimname, timedimname], finaltrack_corecold_eccentricity), \
                # 'perimeter': ([trackdimname, timedimname], finaltrack_corecold_perimeter), \
                # 'xcenter': ([trackdimname, timedimname], finaltrack_corecold_xcenter), \
                # 'ycenter': ([trackdimname, timedimname], finaltrack_corecold_ycenter), \
                # 'xcenter_weighted': ([trackdimname, timedimname], finaltrack_corecold_xweightedcenter), \
                # 'ycenter_weighted': ([trackdimname, timedimname], finaltrack_corecold_yweightedcenter),\
                }

    # Define coordinate list
    coordlist = {trackdimname: ([trackdimname], np.arange(0, numtracks)), \
                    timedimname: ([timedimname], np.arange(0, maxtracklength)), \
                    # 'nfilenamechars': (['nfilenamechars'], np.arange(0, numcharfilename)), \
                    # 'ndatetimechars': (['ndatetimechars'], np.arange(0, 13)),\
                }

    # Define global attributes
    gattrlist = {'title':  'File containing statistics for each track', \
                    'Institution': 'Pacific Northwest National Laboratoy', \
                    'Contact': 'Zhe Feng, zhe.feng@pnnl.gov', \
                    'Created_on':  time.ctime(time.time()), \
                    'source': datasource, \
                    'description': datadescription, \
                    'startdate': startdate, \
                    'enddate': enddate, \
                    'track_version': track_version, \
                    'tracknumbers_version': tracknumbers_version, \
                    'timegap_hour': timegap, \
                    'time_resolution_hour': datatimeresolution, \
                    'pixel_radius_km': pixel_radius}
    
    # Define xarray dataset
    output_data = xr.Dataset(varlist, coords=coordlist, attrs=gattrlist)

    # Specify variable attributes
    output_data[trackdimname].attrs['long_name'] = 'Track number'
    output_data[trackdimname].attrs['units'] = 'unitless'

    output_data[timedimname].attrs['long_name'] = 'Time index number'
    output_data[timedimname].attrs['units'] = 'unitless'

    output_data.lifetime.attrs['long_name'] = 'Duration of each track'
    output_data.lifetime.attrs['units'] = 'count'
    output_data.lifetime.attrs['comments'] = 'Multiply by time_resolution_hour to convert to physical units'

    output_data.basetime.attrs['long_name'] = 'Epoch time of each cell in a track'
    output_data.basetime.attrs['standard_name'] = 'time'
    output_data.basetime.attrs['units'] = basetime_units

    # output_data.cloudidfiles.attrs['long_name'] = 'File name for each cell in a track'

    output_data.core_meanlat.attrs['long_name'] = 'Mean latitude of the convective core in a track'
    output_data.core_meanlat.attrs['units'] = 'degrees_north'

    output_data.core_meanlon.attrs['long_name'] = 'Mean longitude of the convective core in a track'
    output_data.core_meanlon.attrs['units'] = 'degrees_east'

    output_data.core_mean_y.attrs['long_name'] = 'Mean y-distance to radar for the convective core in a track'
    output_data.core_mean_y.attrs['units'] = 'km'

    output_data.core_mean_x.attrs['long_name'] = 'Mean x-distance to radar for the convective core in a track'
    output_data.core_mean_x.attrs['units'] = 'km'

    output_data.cell_meanlat.attrs['long_name'] = 'Mean latitude of the convective cell in a track'
    output_data.cell_meanlat.attrs['units'] = 'degrees_north'

    output_data.cell_meanlon.attrs['long_name'] = 'Mean longitude of the convective cell in a track'
    output_data.cell_meanlon.attrs['units'] = 'degrees_east'

    output_data.cell_mean_y.attrs['long_name'] = 'Mean y-distance to radar for the convective cell in a track'
    output_data.cell_mean_y.attrs['units'] = 'km'

    output_data.cell_mean_x.attrs['long_name'] = 'Mean x-distance to radar for the convective cell in a track'
    output_data.cell_mean_x.attrs['units'] = 'km'

    output_data.cell_minlat.attrs['long_name'] = 'Minimum latitude of the convective cell in a track'
    output_data.cell_minlat.attrs['units'] = 'degrees_north'

    output_data.cell_minlon.attrs['long_name'] = 'Minimum longitude of the convective cell in a track'
    output_data.cell_minlon.attrs['units'] = 'degrees_east'

    output_data.cell_maxlat.attrs['long_name'] = 'Maximum latitude of the convective cell in a track'
    output_data.cell_maxlat.attrs['units'] = 'degrees_north'

    output_data.cell_maxlon.attrs['long_name'] = 'Maximum longitude of the convective cell in a track'
    output_data.cell_maxlon.attrs['units'] = 'degrees_east'

    output_data.cell_min_y.attrs['long_name'] = 'Minimum y-distance to radar for the convective cell in a track'
    output_data.cell_min_y.attrs['units'] = 'km'

    output_data.cell_max_y.attrs['long_name'] = 'Maximum y-distance to radar for the convective cell in a track'
    output_data.cell_max_y.attrs['units'] = 'km'

    output_data.cell_min_x.attrs['long_name'] = 'Minimum x-distance to radar for the convective cell in a track'
    output_data.cell_min_x.attrs['units'] = 'km'

    output_data.cell_max_x.attrs['long_name'] = 'Maximum x-distance to radar for the convective cell in a track'
    output_data.cell_max_x.attrs['units'] = 'km'

    output_data.dilatecell_meanlat.attrs['long_name'] = 'Mean latitude of the dilated convective cell in a track'
    output_data.dilatecell_meanlat.attrs['units'] = 'degrees_north'

    output_data.dilatecell_meanlon.attrs['long_name'] = 'Mean longitude of the dilated convective cell in a track'
    output_data.dilatecell_meanlon.attrs['units'] = 'degrees_east'

    output_data.dilatecell_mean_y.attrs['long_name'] = 'Mean y-distance to radar for the dilated convective cell in a track'
    output_data.dilatecell_mean_y.attrs['units'] = 'km'

    output_data.dilatecell_mean_x.attrs['long_name'] = 'Mean x-distance to radar for the dilated convective cell in a track'
    output_data.dilatecell_mean_x.attrs['units'] = 'km'

    output_data.core_radius.attrs['long_name'] = 'Equivalent radius of the convective core in a track'
    output_data.core_radius.attrs['standard_name'] = 'Equivalent radius'
    output_data.core_radius.attrs['units'] = 'km'
    output_data.core_radius.attrs['valid_min'] = areathresh

    output_data.cell_radius.attrs['long_name'] = 'Equivalent radius of the convective cell in a track'
    output_data.cell_radius.attrs['standard_name'] = 'Equivalent radius'
    output_data.cell_radius.attrs['units'] = 'km'
    output_data.cell_radius.attrs['valid_min'] = areathresh

    output_data.core_area.attrs['long_name'] = 'Area of the convective core in a track'
    output_data.core_area.attrs['units'] = 'km^2'

    output_data.cell_area.attrs['long_name'] = 'Area of the convective cell in a track'
    output_data.cell_area.attrs['units'] = 'km^2'

    output_data.maxdbz.attrs['long_name'] = 'Maximum reflectivity in the convective cell in a track'
    output_data.maxdbz.attrs['units'] = 'dBZ'

    output_data.maxETH_10dbz.attrs['long_name'] = 'Maximum 10dBZ echo-top height in the convective cell'
    output_data.maxETH_10dbz.attrs['units'] = 'km'

    output_data.maxETH_20dbz.attrs['long_name'] = 'Maximum 20dBZ echo-top height in the convective cell'
    output_data.maxETH_20dbz.attrs['units'] = 'km'

    output_data.maxETH_30dbz.attrs['long_name'] = 'Maximum 30dBZ echo-top height in the convective cell'
    output_data.maxETH_30dbz.attrs['units'] = 'km'

    output_data.maxETH_40dbz.attrs['long_name'] = 'Maximum 40dBZ echo-top height in the convective cell'
    output_data.maxETH_40dbz.attrs['units'] = 'km'

    output_data.maxETH_50dbz.attrs['long_name'] = 'Maximum 50dBZ echo-top height in the convective cell'
    output_data.maxETH_50dbz.attrs['units'] = 'km'

    output_data.cloudnumber.attrs['long_name'] = 'Corresponding cloud number in cloudid file in a track'
    output_data.cloudnumber.attrs['units'] = 'unitless'
    output_data.cloudnumber.attrs['usage'] = 'To link this tracking statistics file with corresponding ' + \
                                            'pixel-level cloudid files, use the cloudidfile and cloudnumber ' + \
                                            'together to identify which file and cloud this track is associated with at this time'

    output_data.status.attrs['long_name'] = 'Flag indicating evolution / behavior for each cloud in a track'
    output_data.status.attrs['units'] = 'unitless'
    output_data.status.attrs['valid_min'] = 0
    output_data.status.attrs['valid_max'] = 65

    output_data.startstatus.attrs['long_name'] = 'Flag indicating how the first cloud in a track starts'
    output_data.startstatus.attrs['units'] = 'unitless'
    output_data.startstatus.attrs['valid_min'] = 0
    output_data.startstatus.attrs['valid_max'] = 65

    output_data.endstatus.attrs['long_name'] = 'Flag indicating how the last cloud in a track ends'
    output_data.endstatus.attrs['units'] = 'unitless'
    output_data.endstatus.attrs['valid_min'] = 0
    output_data.endstatus.attrs['valid_max'] = 65

    output_data.trackinterruptions.attrs['long_name'] = 'Flag indicating if track started or ended naturally or artifically due to data availability'
    output_data.trackinterruptions.attrs['values'] = '0 = full track available, good data. 1 = track starts at first file, track cut short by data availability. 2 = track ends at last file, track cut short by data availability'
    output_data.trackinterruptions.attrs['valid_min'] = 0
    output_data.trackinterruptions.attrs['valid_max'] = 2
    output_data.trackinterruptions.attrs['units'] = 'unitless'
    
    output_data.mergenumbers.attrs['long_name'] = 'Number of the track that this small cloud merges into'
    output_data.mergenumbers.attrs['usuage'] = 'Each row represents a track. Each column represets a cloud in that track. Numbers give the track number that this small cloud merged into.'
    output_data.mergenumbers.attrs['units'] = 'unitless'
    output_data.mergenumbers.attrs['valid_min'] = 1
    output_data.mergenumbers.attrs['valid_max'] = numtracks

    output_data.splitnumbers.attrs['long_name'] = 'Number of the track that this small cloud splits from'
    output_data.splitnumbers.attrs['usuage'] = 'Each row represents a track. Each column represets a cloud in that track. Numbers give the track number that his msallcloud splits from.'
    output_data.splitnumbers.attrs['units'] = 'unitless'
    output_data.splitnumbers.attrs['valid_min'] = 1
    output_data.splitnumbers.attrs['valid_max'] = numtracks

    # output_data.boundary.attrs['long_name'] = 'Flag indicating whether the core + cold anvil touches one of the domain edges.'
    # output_data.boundary.attrs['usuage'] = ' 0 = away from edge. 1= touches edge.'
    # output_data.boundary.attrs['units'] = 'unitless'
    # output_data.boundary.attrs['valid_min'] = 0
    # output_data.boundary.attrs['valid_max'] = 1

    # output_data.orientation.attrs['long_name'] = 'Orientation of the major axis of the core + cold anvil for each cloud in a track'
    # output_data.orientation.attrs['units'] = 'Degrees clockwise from vertical'
    # output_data.orientation.attrs['valid_min'] = 0
    # output_data.orientation.attrs['valid_max'] = 360

    # output_data.eccentricity.attrs['long_name'] = 'Eccentricity of the major axis of the core + cold anvil for each cloud in a track'
    # output_data.eccentricity.attrs['units'] = 'unitless'
    # output_data.eccentricity.attrs['valid_min'] = 0
    # output_data.eccentricity.attrs['valid_max'] = 1

    # output_data.majoraxis.attrs['long_name'] =  'Length of the major axis of the core + cold anvil for each cloud in a track'
    # output_data.majoraxis.attrs['units'] = 'km'

    # output_data.perimeter.attrs['long_name'] = 'Approximnate circumference of the core + cold anvil for each cloud in a track'
    # output_data.perimeter.attrs['units'] = 'km'

    # output_data.xcenter.attrs['long_name'] = 'X index of the geometric center of the cloud feature for each cloud in a track'
    # output_data.xcenter.attrs['units'] = 'unitless'

    # output_data.ycenter.attrs['long_name'] = 'Y index of the geometric center of the cloud feature for each cloud in a track'
    # output_data.ycenter.attrs['units'] = 'unitless'

    # output_data.xcenter_weighted.attrs['long_name'] = 'X index of the brightness temperature weighted center of the cloud feature for each cloud in a track'
    # output_data.xcenter_weighted.attrs['units'] = 'unitless'

    # output_data.ycenter_weighted.attrs['long_name'] = 'Y index of the brightness temperature weighted center of the cloud feature for each cloud in a track'
    # output_data.ycenter_weighted.attrs['units'] = 'unitless'

    # Specify encoding list
    var_float_encode = {'dtype':'float32', 'zlib':True, '_FillValue': np.nan}
    var_int_encode = {'dtype': 'int', 'zlib':True, '_FillValue': fillval}
    encodelist = {'lifetime': var_int_encode, \
                #     'basetime': {'zlib':True, 'units': basetime_units}, \
                    'basetime': {'zlib':True}, \
                    trackdimname: {'dtype': 'int', 'zlib':True}, \
                    timedimname: {'dtype': 'int', 'zlib':True}, \
                    # 'cloudidfiles': {'zlib':True}, \
                    'core_meanlat': var_float_encode, \
                    'core_meanlon': var_float_encode, \
                    'core_mean_y': var_float_encode, \
                    'core_mean_x': var_float_encode, \
                    'cell_meanlat': var_float_encode, \
                    'cell_meanlon': var_float_encode, \
                    'cell_mean_y': var_float_encode, \
                    'cell_mean_x': var_float_encode, \
                    'cell_minlat': var_float_encode, \
                    'cell_minlon': var_float_encode, \
                    'cell_maxlat': var_float_encode, \
                    'cell_maxlon': var_float_encode, \
                    'cell_min_y': var_float_encode, \
                    'cell_min_x': var_float_encode, \
                    'cell_max_y': var_float_encode, \
                    'cell_max_x': var_float_encode, \

                    'dilatecell_meanlat': var_float_encode, \
                    'dilatecell_meanlon': var_float_encode, \
                    'dilatecell_mean_y': var_float_encode, \
                    'dilatecell_mean_x': var_float_encode, \

                    'core_radius': var_float_encode, \
                    'cell_radius': var_float_encode, \
                    'core_area': var_float_encode, \
                    'cell_area': var_float_encode, \

                    'maxdbz': var_float_encode, \
                    'maxETH_10dbz': var_float_encode, \
                    'maxETH_20dbz': var_float_encode, \
                    'maxETH_30dbz': var_float_encode, \
                    'maxETH_40dbz': var_float_encode, \
                    'maxETH_50dbz': var_float_encode, \

                    'cloudnumber': var_int_encode, \
                    'mergenumbers': var_int_encode, \
                    'splitnumbers': var_int_encode, \
                    'status': var_int_encode, \
                    'startstatus': var_int_encode, \
                    'endstatus': var_int_encode, \
                    'trackinterruptions': var_int_encode, \

                    # 'boundary':  {'dtype': 'int', 'zlib':True, '_FillValue': fillval}, \
                    # 'majoraxis': {'zlib':True, '_FillValue': np.nan}, \
                    # 'orientation': {'zlib':True, '_FillValue': np.nan}, \
                    # 'eccentricity': {'zlib':True, '_FillValue': np.nan}, \
                    # 'perimeter': {'zlib':True, '_FillValue': np.nan}, \
                    # 'xcenter': {'zlib':True, '_FillValue': fillval}, \
                    # 'ycenter': {'zlib':True, '_FillValue': fillval}, \
                    # 'xcenter_weighted': {'zlib':True, '_FillValue': fillval}, \
                    # 'ycenter_weighted': {'zlib':True, '_FillValue': fillval},\
                }

    # Write netcdf file
    output_data.to_netcdf(path=trackstats_outfile, mode='w', format='NETCDF4_CLASSIC', unlimited_dims=trackdimname, encoding=encodelist)
