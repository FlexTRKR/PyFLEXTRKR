import time
import numpy as np
import xarray as xr
from netCDF4 import Dataset, stringtochar, num2date

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
                        'boundary': (['ntracks'], finaltrack_corecold_boundary), \
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
                        'Conventions':'CF-1.6', \
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
                            datasource, datadescription, startdate, enddate, \
                            track_version, tracknumbers_version, timegap, basetime_units, \
                            pixel_radius, areathresh, fillval, \
                            finaltrack_tracklength, finaltrack_basetime, \
                            finaltrack_cloudidfile, finaltrack_cloudnumber, \
                            finaltrack_core_meanlat, finaltrack_core_meanlon, \
                            finaltrack_cell_minlat, finaltrack_cell_maxlat, \
                            finaltrack_cell_minlon, finaltrack_cell_maxlon, \
                            finaltrack_core_npix, finaltrack_cell_npix, \
                            finaltrack_core_radius, finaltrack_cell_radius, \
                            finaltrack_cell_maxdbz, finaltrack_status, \
                            finaltrack_startstatus, finaltrack_endstatus, \
                            finaltrack_trackinterruptions, \
                            finaltrack_mergenumber, finaltrack_splitnumber, \
                            ):
    """
    Write radar trackstats variables to netCDF file.
    """

    # Define variable list
    varlist = {'lifetime': (['ntracks'], finaltrack_tracklength), \
                'basetime': (['ntracks', 'nmaxlength'], finaltrack_basetime), \
                'cloudidfiles': (['ntracks', 'nmaxlength', 'nfilenamechars'], finaltrack_cloudidfile), \
                # 'datetimestrings': (['ntracks', 'nmaxlength', 'ndatetimechars'], finaltrack_datetimestring), \
                'meanlat': (['ntracks', 'nmaxlength'], finaltrack_core_meanlat), \
                'meanlon': (['ntracks', 'nmaxlength'], finaltrack_core_meanlon), \
                'minlat': (['ntracks', 'nmaxlength'], finaltrack_cell_minlat), \
                'minlon': (['ntracks', 'nmaxlength'], finaltrack_cell_minlon), \
                'maxlat': (['ntracks', 'nmaxlength'], finaltrack_cell_maxlat), \
                'maxlon': (['ntracks', 'nmaxlength'], finaltrack_cell_maxlon), \
                'radius_core': (['ntracks', 'nmaxlength'], finaltrack_core_radius), \
                'radius_cell': (['ntracks', 'nmaxlength'], finaltrack_cell_radius), \
                'npix_core': (['ntracks', 'nmaxlength'], finaltrack_core_npix), \
                'npix_cell': (['ntracks', 'nmaxlength'], finaltrack_cell_npix), \
                'maxdbz': (['ntracks', 'nmaxlength'], finaltrack_cell_maxdbz), \
                'cloudnumber': (['ntracks', 'nmaxlength'], finaltrack_cloudnumber), \
                'status': (['ntracks', 'nmaxlength'], finaltrack_status), \
                'startstatus': (['ntracks'], finaltrack_startstatus), \
                'endstatus': (['ntracks'], finaltrack_endstatus), \
                'mergenumbers': (['ntracks', 'nmaxlength'], finaltrack_mergenumber), \
                'splitnumbers': (['ntracks', 'nmaxlength'], finaltrack_splitnumber), \
                'trackinterruptions': (['ntracks'], finaltrack_trackinterruptions), \
                # 'boundary': (['ntracks'], finaltrack_boundary), \
                # 'majoraxis': (['ntracks', 'nmaxlength'], finaltrack_corecold_majoraxis), \
                # 'orientation': (['ntracks', 'nmaxlength'], finaltrack_corecold_orientation), \
                # 'eccentricity': (['ntracks', 'nmaxlength'], finaltrack_corecold_eccentricity), \
                # 'perimeter': (['ntracks', 'nmaxlength'], finaltrack_corecold_perimeter), \
                # 'xcenter': (['ntracks', 'nmaxlength'], finaltrack_corecold_xcenter), \
                # 'ycenter': (['ntracks', 'nmaxlength'], finaltrack_corecold_ycenter), \
                # 'xcenter_weighted': (['ntracks', 'nmaxlength'], finaltrack_corecold_xweightedcenter), \
                # 'ycenter_weighted': (['ntracks', 'nmaxlength'], finaltrack_corecold_yweightedcenter),\
                }

    # Define coordinate list
    coordlist = {'ntracks': (['ntracks'], np.arange(0,numtracks)), \
                    'nmaxlength': (['nmaxlength'], np.arange(0, maxtracklength)), \
                    'nfilenamechars': (['nfilenamechars'], np.arange(0, numcharfilename)), \
                    # 'ndatetimechars': (['ndatetimechars'], np.arange(0, 13)),\
                }

    # Define global attributes
    gattrlist = {'title':  'File containing statistics for each track', \
                    'Conventions':'CF-1.6', \
                    'Institution': 'Pacific Northwest National Laboratoy', \
                    'Contact': 'Zhe Feng, zhe.feng@pnnl.gov', \
                    'Created_on':  time.ctime(time.time()), \
                    'source': datasource, \
                    'description': datadescription, \
                    'startdate': startdate, \
                    'enddate': enddate, \
                    'track_version': track_version, \
                    'tracknumbers_version': tracknumbers_version, \
                    'timegap': str(timegap)+'-hr', \
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
    output_data.basetime.attrs['units'] = basetime_units

    output_data.cloudidfiles.attrs['long_name'] = 'File name for each cloud in each track'

    output_data.meanlat.attrs['long_name'] = 'Mean latitude of the convective core in a track'
    output_data.meanlat.attrs['standard_name'] = 'latitude'
    output_data.meanlat.attrs['units'] = 'degrees_north'

    output_data.meanlon.attrs['long_name'] = 'Mean longitude of the convective core in a track'
    output_data.meanlon.attrs['standard_name'] = 'longitude'
    output_data.meanlon.attrs['units'] = 'degrees_east'

    output_data.minlat.attrs['long_name'] = 'Minimum latitude of the convective cell in a track'
    output_data.minlat.attrs['standard_name'] = 'latitude'
    output_data.minlat.attrs['units'] = 'degrees_north'

    output_data.minlon.attrs['long_name'] = 'Minimum longitude of the convective cell in a track'
    output_data.minlon.attrs['standard_name'] = 'longitude'
    output_data.minlon.attrs['units'] = 'degrees_east'

    output_data.maxlat.attrs['long_name'] = 'Maximum latitude of the convective cell in a track'
    output_data.maxlat.attrs['standard_name'] = 'latitude'
    output_data.maxlat.attrs['units'] = 'degrees_north'

    output_data.maxlon.attrs['long_name'] = 'Maximum longitude of the convective cell in a track'
    output_data.maxlon.attrs['standard_name'] = 'longitude'
    output_data.maxlon.attrs['units'] = 'degrees_east'

    output_data.radius_core.attrs['long_name'] = 'Equivalent radius of the convective core in a track'
    output_data.radius_core.attrs['standard_name'] = 'Equivalent radius'
    output_data.radius_core.attrs['units'] = 'km'
    output_data.radius_core.attrs['valid_min'] = areathresh

    output_data.radius_cell.attrs['long_name'] = 'Equivalent radius of the convective cell in a track'
    output_data.radius_cell.attrs['standard_name'] = 'Equivalent radius'
    output_data.radius_cell.attrs['units'] = 'km'
    output_data.radius_cell.attrs['valid_min'] = areathresh

    output_data.npix_core.attrs['long_name'] = 'Number of pixels in the convective core in a track'
    output_data.npix_core.attrs['units'] = 'unitless'

    output_data.npix_cell.attrs['long_name'] = 'Number of pixels in the convective cell in a track'
    output_data.npix_cell.attrs['units'] = 'unitless'

    output_data.maxdbz.attrs['long_name'] = 'Maximum reflectivity in the convective cell in a track'
    output_data.maxdbz.attrs['units'] = 'unitless'

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
    encodelist = {'lifetime': {'dtype': 'int', 'zlib':True, '_FillValue': fillval}, \
                #     'basetime': {'zlib':True, 'units': basetime_units}, \
                    'basetime': {'zlib':True}, \
                    'ntracks': {'dtype': 'int', 'zlib':True}, \
                    'nmaxlength': {'dtype': 'int', 'zlib':True}, \
                    'cloudidfiles': {'zlib':True}, \
                    # 'datetimestrings': {'zlib':True}, \
                    'meanlat': {'zlib':True, '_FillValue': np.nan}, \
                    'meanlon': {'zlib':True, '_FillValue': np.nan}, \
                    'minlat': {'zlib':True, '_FillValue': np.nan}, \
                    'minlon': {'zlib':True, '_FillValue': np.nan}, \
                    'maxlat': {'zlib':True, '_FillValue': np.nan}, \
                    'maxlon': {'zlib':True, '_FillValue': np.nan}, \
                    'radius_core': {'zlib':True, '_FillValue': np.nan}, \
                    'radius_cell': {'zlib':True, '_FillValue': np.nan}, \
                    # 'boundary':  {'dtype': 'int', 'zlib':True, '_FillValue': fillval}, \
                    'npix_core': {'dtype': 'int', 'zlib':True, '_FillValue': fillval}, \
                    'npix_cell': {'dtype': 'int', 'zlib':True, '_FillValue': fillval}, \
                    'cloudnumber': {'dtype': 'int', 'zlib':True, '_FillValue': fillval}, \
                    'mergenumbers': {'dtype': 'int', 'zlib':True, '_FillValue': fillval}, \
                    'splitnumbers': {'dtype': 'int', 'zlib':True, '_FillValue': fillval}, \
                    'status': {'dtype': 'int', 'zlib':True, '_FillValue': fillval}, \
                    'startstatus': {'dtype': 'int', 'zlib':True, '_FillValue': fillval}, \
                    'endstatus': {'dtype': 'int', 'zlib':True, '_FillValue': fillval}, \
                    'trackinterruptions': {'dtype': 'int', 'zlib':True, '_FillValue': fillval}, \
                    'maxdbz': {'zlib':True, '_FillValue': np.nan}, \
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
    output_data.to_netcdf(path=trackstats_outfile, mode='w', format='NETCDF4_CLASSIC', unlimited_dims='ntracks', encoding=encodelist)