import sys
import os
import numpy as np
import math
from scipy import ndimage
import matplotlib.pyplot as plt

def shift_fast(arr, xshift, yshift):

    fill_value=np.nan
    result = np.empty_like(arr)
    if xshift > 0:
        result[:xshift,:] = fill_value
        result[xshift:,:] = arr[:-xshift,:]
    elif xshift < 0:
        result[xshift:,:] = fill_value
        result[:xshift,:] = arr[-xshift:,:]
    else:
        result[:,:] = arr

    result2 = np.empty_like(result)
    if yshift > 0:
        result2[:,:yshift] = fill_value
        result2[:,yshift:] = result[:,:-yshift]
    elif yshift < 0:
        result2[:,yshift:] = fill_value
        result2[:,:yshift] = result[:,-yshift:]
    else:
        result2[:,:] = result

    del result
    return result2

# GridRad filter routine
def gridrad_sl3d(data, config, **kwargs):

    # Copy data structure for manipulation
    # data = data0

    radardatasource = config.get('radardatasource', None)

    # Extract dimension sizes for ease
    nx = (data['x'])['n']
    ny = (data['y'])['n']
    nz = (data['z'])['n']

    # Estimate the number of grid points based on radar data source
    if (radardatasource == 'wrf'):
        # WRF has fixed grid spacing
        # Simply divide 12 km radius by pixel_radius to get number of grid points
        pixel_radius = config.get('pixel_radius')
        n12km = int(12. / pixel_radius)

    if (radardatasource == 'gridrad'):
        # Get composite grid spacing (in degrees)
        dx = ((data['x'])['values'])[1] - ((data['x'])['values'])[0]
        # Compute latitude mid-point of grid
        ymid = 0.5 * (data['y']['values'])[ny-1] + 0.5 * (data['y']['values'])[0]
        # Get approximate number of grid points equivalent to 12 km grid spacing
        n12km = int(0.108 / (dx * math.cos(math.radians(ymid))))

    # Compute number of points in a 12-km box
    nsearch = 2 * n12km + 1

    # the 3D arrays in the codes below assumes [x, y, z] dimension
    # BUT the input data are actually in [z, y, x] dimension
    # TODO: change the order of the array when accessing certain dimensions

    # WRF lat/lon (x, y) are 2D, while GridRad lat/lon is 1D
    # TODO: build in a check here to adapt to either 2D or 1D lat/lon
    # y1d = data['y']['values'][:,0]
    # yyy = ((y1d.reshape(1, ny, 1 )).repeat(nx, axis=0)).repeat(nz,axis=2)
    # zzz = ((((data['z'])['values']).reshape(1, 1, nz )).repeat(nx, axis=0)).repeat(ny,axis=1)

    if data['y']['values'].ndim == 1:
        yyy = data['y']['values'].reshape(1, ny, 1).repeat(nz,axis=0).repeat(nx, axis=2)
    if data['y']['values'].ndim == 2:
        yyy = np.repeat(data['y']['values'][np.newaxis, :, :], nz, axis=0)
    zzz = data['z']['values'].reshape(nz, 1, 1).repeat(ny,axis=1).repeat(nx, axis=2)

    # np.repeat(data['z']['values'][:, :, np.newaxis], 3, axis=2)
    # np.tile(data['z']['values'], (ny, nx))

    # Copy latitudes to 3 dimensions for utility
    # yyy = ((((data['y'])['values']).reshape(1, ny, 1 )).repeat(nx, axis=0)).repeat(nz,axis=2)

    # zzz = ((((data['z'])['values']).reshape(1, 1, nz )).repeat(nx, axis=0)).repeat(ny,axis=1)

    #print(((data['Z_H'])['values']).size)

    # Find index of 3, 4, 5, and 9 km altitude
    if (((data['z'])['values'])[0] > 3.0) : k3km = -1
    if (((data['z'])['values'])[0] > 4.0) : k4km = -1
    if (((data['z'])['values'])[0] > 5.0) : k5km = -1
    if (((data['z'])['values'])[0] > 9.0) : k9km = -1

    if (((data['z'])['values'])[nz-1] <= 3.0) : k3km = nz-1
    if (((data['z'])['values'])[nz-1] <= 4.0) : k4km = nz-1
    if (((data['z'])['values'])[nz-1] <= 5.0) : k5km = nz-1
    if (((data['z'])['values'])[nz-1] <= 9.0) : k9km = nz-1

    for zindex in range (0,nz-1):
        if (((data['z'])['values'])[zindex] <= 3.0 and ((data['z'])['values'])[zindex+1] > 3.0): k3km = zindex
        if (((data['z'])['values'])[zindex] <= 4.0 and ((data['z'])['values'])[zindex+1] > 4.0): k4km = zindex
        if (((data['z'])['values'])[zindex] <= 5.0 and ((data['z'])['values'])[zindex+1] > 5.0): k5km = zindex
        if (((data['z'])['values'])[zindex] <= 9.0 and ((data['z'])['values'])[zindex+1] > 9.0): k9km = zindex

    #Set coefficients for 50th percentile melting level climatology
    a = [7.072, 7.896, 8.558, 7.988, 7.464, 6.728, 6.080, 6.270, 6.786, 8.670, 8.892, 7.936]
    b = [-0.124,-0.152,-0.160,-0.128,-0.100,-0.065, -0.039,-0.044,-0.067,-0.137,-0.160,-0.147]

    #Extract file month from GridRad data structure
    month = int((data['Analysis_time'])[5:7])

    if ('zmelt' not in kwargs):
        #If no melting level provided, comput expected melting level for domain based on climatology
        zml = a[month-1] + b[month-1]*yyy
    else:
        zmelt = kwargs['zmelt']
        if (zmelt.size == 1):
            if (zmelt > 0.0):
                #Copy constant melting level to three dimensions
                # zml = np.full((nx, ny, nz) ,zmelt)
                zml = np.full((nz, ny, nx), zmelt)
            else:
                #Else, revert to melting level climatology
                zml = a[month-1] + b[month-1]*yyy
        else:
            if (zmelt.ndim == 2):
                # Copy 2-D melting level to 3-D
                # zml = zmelt.reshape(nx, ny, 1 ).repeat(nz, axis=2)
                zml = zmelt.reshape(1, ny, nx).repeat(nz, axis=0)
            else:
                #Else, revert to melting level climatology
                zml = a[month-1] + b[month-1]*yyy


    # Find points with high reflectivity at 4 km and no echo at 3 km (potential gaps in coverage)
    ifix = np.where(
        (~np.isfinite(((data['Z_H'])['values'])[k3km,:,:])) & \
        (np.isfinite(((data['Z_H'])['values'])[k4km,:,:])) & \
        (((data['Z_H'])['values'])[k4km,:,:] >= 20.0))
    nfix = len(ifix[0])

    if (nfix > 0):
        # Copy reflectivity at 3 km and 4 km
        tmp = ((data['Z_H'])['values'])[k3km,:,:]
        tmp2 = ((data['Z_H'])['values'])[k4km,:,:]
        tmp[ifix] = tmp2[ifix]
        #Use high reflectivity values at 4 km to replace missing obs at 3 km
        ((data['Z_H'])['values'])[k3km,:,:]=tmp
        del tmp, tmp2

    # Create array to store SL3D classification
    sl3dclass = np.zeros((ny, nx), dtype=np.short)

    # Sum depths of echo at and above 3 km (assuming 1-km GridRad v3.1 data)
    tmp = ((data['Z_H'])['values'])[k3km:,:,:]
    dzgt00dBZ = np.sum((~np.isnan(tmp)) & (tmp >=0.0), axis=0)
    del tmp

    # Get 25.0 dBZ echo top
    tmp = (data['Z_H'])['values']
    etop25dBZ = np.nanmax(((~np.isnan(tmp)) & (tmp >=25.0)) * zzz, axis=0)

    # Get column-maximum reflectivity
    dbz_comp = np.nanmax(tmp, axis=0)

    # Get column-maximum reflectivity for above melting level altitudes
    dbz_aml = np.nanmax(tmp * (zzz > (zml + 1.0)), axis=0)
    del tmp

    # Create array to compute peakedness in lowest 9 km altitude layer
    peak = np.full((k9km+1,ny,nx), np.NaN, dtype=data['Z_H']['values'].dtype)

    # Loop over the lowest 9 km levels
    for k in range(0, k9km+1):
        tmp = ((data['Z_H'])['values'])[k,:,:]

        # TODO: The following codes need to change array dimensions from [x,y,z] to [z,y,x]
        # According to this thread: https://forum.image.sc/t/skimage-filters-median-using-mask-for-floating-point-image-with-nans/57289
        # scipy.ndimage.median_filter v1.7 (same as skimage.filters.median v0.17) above ignores NaN
        # Perhaps could give it a try to simplify the codes here
        peak[k,:,:] = tmp - ndimage.median_filter(tmp, size=nsearch)
        # import pdb; pdb.set_trace()

        # #ndimage.median_filter cannot handle with the nan problem
        # #peak[:,:,k]=tmp-ndimage.median_filter(tmp, size=nsearch)
        # #produce array for median calculation
        # formedianarr = np.empty((nx,ny,nsearch*nsearch),dtype=tmp.dtype)
        # for shiftsize_x in range(-n12km,n12km+1):
        #     for shiftsize_y in range(-n12km,n12km+1):
        #         formedianarr[:,:,(shiftsize_x+n12km)*nsearch+shiftsize_y+n12km] = shift_fast(tmp, shiftsize_x, shiftsize_y)

        # # calculate median
        # medianarr = np.nanmedian(formedianarr, axis=2)
        # medianarr[0:n12km,:] = tmp[0:n12km,:]
        # medianarr[-n12km:,:] = tmp[-n12km:,:]
        # medianarr[:,0:n12km] = tmp[:,0:n12km]
        # medianarr[:,-n12km:] = tmp[:,-n12km:]
        # del formedianarr

        # #print(tmp.dtype)
        # #print(medianarr.dtype)
        # #print(peak.dtype)
        # #Compute peakedness at each altitude level
        # peak[:,:,k]=tmp-medianarr
        # del tmp


    # Compute peakedness threshold for reflectivity value
    tmp = 10.0 - ((((data['Z_H'])['values'])[0:k9km+1,:,:])**2) / 337.5
    peak_thresh = np.full(peak.shape, np.NaN, dtype=peak.dtype)
    largeindex = (~np.isnan(tmp)) & (tmp > 4.0)
    smallindex = (~np.isnan(tmp)) & (tmp <= 4.0)
    peak_thresh[largeindex] = tmp[largeindex]
    peak_thresh[smallindex] = 4.0
    del tmp, largeindex, smallindex

    # Compute column-mean peakedness
    tmp = ((data['Z_H'])['values'])[0:k9km+1,:,:]
    mean_peak = np.sum((~np.isnan(peak_thresh)) & (peak > peak_thresh), axis=0) / np.sum(np.isfinite(tmp), axis=0)
    del tmp

    # Find convective points (those with at least 30% of column exceeding peakedness or > 45 dBZ above melting level or echo top > 10 km)
    iconv = np.where(
        (np.isfinite(mean_peak) & (mean_peak >= 0.3)) | 
        (np.isfinite(dbz_aml) & (dbz_aml >= 45.)) | 
        (np.isfinite(etop25dBZ) & (etop25dBZ >= 10.0))
    )
    nconv = len(iconv[0])
    # Flag as convective
    if (nconv > 0): sl3dclass[iconv] = 2

    # Compute fraction of neighborhood with convection
    tmp = 1.0 * (sl3dclass == 2)
    conv_test = ndimage.uniform_filter(tmp, size=3, mode='nearest')
    # print(conv_test[0,:])
    # plt.pcolormesh(conv_test)
    # Why copy the edge pixels?
    # conv_test[0,:] = tmp[0,:]
    # conv_test[-1,:] = tmp[-1,:]
    # conv_test[:,0] = tmp[:,0]
    # conv_test[:,-1] = tmp[:,-1]
    del tmp

    # TODO: this part of the original IDL code seems not included?
    # iremove = WHERE(((class EQ 2B) AND (conv_test LE 0.15)), nremove)	 ;Find single grid point convective classifications to remove
    # IF (nremove GT 0) THEN class[iremove] = 0B

    # Find single grid point convective classifications to remove
    iremove = (sl3dclass == 2) & (conv_test <= 0.15)
    sl3dclass[iremove] = 0

    # Compute fraction of neighborhood with convection (again)
    tmp = 1.0 * (sl3dclass == 2)
    conv_test = ndimage.uniform_filter(tmp, size=3, mode='nearest')
    import pdb; pdb.set_trace()
    # conv_test[0,:] = tmp[0,:]
    # conv_test[-1,:] = tmp[-1,:]
    # conv_test[:,0] = tmp[:,0]
    # conv_test[:,-1] = tmp[:,-1]

    # Find points neighboring convective classification that have similarly intense reflectivity
    # TODO: This step seems to produce weird stripes
    iconv2_mask = (conv_test > 0.0) & (dbz_comp >= 25.)
    plt.pcolormesh(iconv2_mask)
    plt.show()
    import pdb; pdb.set_trace()

    iconv2 = np.where((conv_test > 0.0) & (dbz_comp >= 25.))
    nconv2 = len(iconv2[0])
    # Set classification for similarly intense regions as convection 
    # (i.e., after convective radius of Steiner et al, but strictly for adjacent grid points)
    if (nconv2 > 0): sl3dclass[iconv2] = 2
    # plt.pcolormesh(sl3dclass)
    # plt.show()
    import pdb; pdb.set_trace()

    # Find precipitating stratiform points
    istrat = np.where(
        ((((data['Z_H'])['values'])[k3km,:,:] >= 20.0) | 
        (np.sum(((data['Z_H'])['values'])[0:k3km,:,:] >= 10.0, axis=0) > 0)) & 
        (sl3dclass == 0)
    )
    nstrat = len(istrat[0])
    # Flag as precipitating stratiform
    if (nstrat > 0): sl3dclass[istrat] = 3
    import pdb; pdb.set_trace()

    # Find non-precipitating stratiform points
    itranv = np.where(
        ((~np.isfinite(((data['Z_H'])['values'])[k3km,:,:])) | 
        (((data['Z_H'])['values'])[k3km,:,:] < 20.0)) & 
        (dzgt00dBZ > 0.0) & (sl3dclass == 0)
    )
    ntranv = len(itranv[0])
    # Flag as non-precipitating stratiform
    if (ntranv > 0): sl3dclass[itranv] = 4

    # Find anvil
    ianvil = np.where(
        (np.sum(np.isfinite(((data['Z_H'])['values'])[k3km:k5km+1,:,:]), axis=0) == 0) & 
        (dzgt00dBZ > 0.0) & 
        ((sl3dclass == 0) | (sl3dclass == 4))
    )
    nanvil = len(ianvil[0])
    # Flag anvil
    if (nanvil > 0): sl3dclass[ianvil] = 5

    # Compute reflectivity altitude gradient
    ddbzdz = np.roll((data['Z_H'])['values'], -1, axis=0) - (data['Z_H'])['values']
    # Compute fraction of neighborhood with echo
    tmp = 1.0 * (np.isfinite((data['Z_H'])['values']))
    fin_test = ndimage.uniform_filter(tmp, size=[1,3,3])
    # del tmp

    # Search for weak echo regions
    iupdraft = np.where(
        (np.sum((ddbzdz >= 8.0) * (fin_test >= 0.7) * (zzz <= 7.0), axis=0) > 0.0) & 
        (dbz_comp >= 40.0) & (sl3dclass == 2)
    )
    nupdraft = len(iupdraft[0])
    # Flag updrafts
    if (nupdraft > 0): sl3dclass[iupdraft] = 1

    # Compute fraction of neighborhood with convective updraft
    tmp = 1.0 * (sl3dclass == 1)
    updft_test = ndimage.uniform_filter(tmp, size=3)
    del tmp

    iremove = np.where((sl3dclass == 1) & (updft_test <= 0.15))
    nremove = len(iremove[0])

    if (nremove > 0): sl3dclass[iremove] = (ndimage.median_filter(sl3dclass, size=3))[iremove]

    return sl3dclass
