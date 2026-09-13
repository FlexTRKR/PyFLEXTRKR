import logging
import numpy as np
from scipy.ndimage import label, binary_dilation, generate_binary_structure
from astropy.convolution import Box2DKernel, convolve
from pyflextrkr.ftfunctions import sort_renumber, grow_cells, pad_and_extend, call_adjust_axis
from pyflextrkr.label_and_grow_features import label_and_grow_features


def label_and_grow_cold_clouds(
    ir,
    pixel_radius,
    tb_threshs,
    area_thresh,
    mincoldcorepix,
    smoothsize,
    warmanvilexpansion,
    config,
    pixel_area=None,
):
    """
    Label and grow cold clouds using infrared Tb.

    This is a backward-compatible wrapper around the generalized
    :func:`~pyflextrkr.label_and_grow_features.label_and_grow_features`.
    It calls the generalized function with ``core_operator='lt'`` and
    ``growth_method='bfs'`` to reproduce the original algorithm exactly -
    **when** ``config['pbc_direction']`` **is** ``'none'``, verified
    bit-identical against a frozen copy of the pre-refactor algorithm (see
    ``tests/test_label_grow_methods.py::TestLabelGrowBfsBackwardCompat``).

    When ``pbc_direction != 'none'``, this wrapper's output *intentionally*
    diverges from the pre-refactor algorithm: the original has two bugs in
    its PBC-crop path (returns a stale pre-crop ``final_nclouds`` that
    doesn't match its own npix arrays' length, and never renumbers the
    returned label arrays to be contiguous after cropping - so a caller
    indexing ``npix[label - 1]`` with the real, sparse label value gets a
    wrong count or an ``IndexError``, the same crash class as issue #146).
    This wrapper fixes both. Do not "fix" this wrapper to match the
    original's PBC-path output - see
    ``tests/test_label_grow_methods.py::test_pbc_bfs_deliberately_diverges_from_reference``.

    Args:
        ir: np.ndarray()
            Infrared Tb data array.
        pixel_radius: float
            Pixel size.
        tb_threshs: np.ndarray()
            Infrared Tb thresholds.
        area_thresh: float
            Minimum area to define a cloud [km^2].
        mincoldcorepix: int
            Minimum number of pixels to define a cold core.
        smoothsize: int
            Window size to smooth Tb data using Box2DKernel.
        warmanvilexpansion: int
            Flag to expand cloud to include warm anvil.
        config: dict
            Dictionary containing config parameters.
        pixel_area: float or np.ndarray, optional
            Scalar pixel_radius^2 or 2D grid_area array (ny, nx) in km^2.
            When 2D, area-based thresholds use actual grid cell areas.

    Returns:
        Dictionary:
            Containing labeled cloud array and sizes, using the legacy key
            names below (translated from label_and_grow_features's
            field-agnostic keys, shown in parentheses):
            - final_nclouds (final_nFeature)
            - final_ncorepix (final_Core_npix)
            - final_ncoldpix (final_Secondary_npix)
            - final_ncorecoldpix (final_CoreSecondary_npix)
            - final_nwarmpix (final_Tertiary_npix)
            - final_cloudnumber (final_Feature_Number)
            - final_cloudtype (final_Feature_Type)
            - final_convcold_cloudnumber (final_CoreSecondary_Number)
    """
    result = label_and_grow_features(
        field=ir,
        pixel_radius=pixel_radius,
        thresholds=tb_threshs,
        area_thresh=area_thresh,
        min_core_npix=mincoldcorepix,
        smooth_size=smoothsize,
        expand_to_tertiary=warmanvilexpansion,
        config=config,
        pixel_area=pixel_area,
        core_operator='lt',
        growth_method='bfs',
    )
    # Translate the generalized module's field-agnostic keys back to the
    # legacy names this wrapper's remaining callers still read by
    # (tests/test_area_method_clouds.py, pyflextrkr/depreciated/idclouds*.py) -
    # without this, those callers KeyError on every call.
    return {
        "final_nclouds": result["final_nFeature"],
        "final_ncorepix": result["final_Core_npix"],
        "final_ncoldpix": result["final_Secondary_npix"],
        "final_ncorecoldpix": result["final_CoreSecondary_npix"],
        "final_nwarmpix": result["final_Tertiary_npix"],
        "final_cloudnumber": result["final_Feature_Number"],
        "final_cloudtype": result["final_Feature_Type"],
        "final_convcold_cloudnumber": result["final_CoreSecondary_Number"],
    }


def find_and_label_cold_cores(smoothir, thresh_core):
    """
    Label cold cores using ndimage.label.

    Args:
        smoothir: np.array
            Array containing smoothed IR Tb data.
        thresh_core: float
            Tb threshold to define cold core.

    Returns:
        labelcore_number2d: np.array
            Array containing labeled cold core numbers.
        nlabelcores: np.array
            Array containing the number of labeled cold cores.

    """
    # Find cold cores in smoothed data
    smoothcore_flag = np.zeros(smoothir.shape, dtype=int)
    smoothcore_indices = np.where(smoothir < thresh_core)
    nsmoothcorepix = np.shape(smoothcore_indices)[1]
    if nsmoothcorepix > 0:
        smoothcore_flag[smoothcore_indices] = 1
    # Label cold cores in smoothed data
    labelcore_number2d, nlabelcores = label(smoothcore_flag)
    return labelcore_number2d, nlabelcores


def smooth_tb(ir, smoothsize):
    """
    Smooth Tb with a convolve filter.

    Args:
        ir: np.array
            Array containing IR Tb data.
        smoothsize: int
            Width of the filter kernel for smoothing Tb data.

    Returns:
        corepix: np.array
            Array containing number of pixels for cold cores.
        smoothir: np.array
            Array containing smoothed IR Tb data.

    """
    # Smooth Tb data using a convolve filter
    kernel = Box2DKernel(smoothsize)
    smoothir = convolve(
        ir, kernel, boundary="extend", nan_treatment="interpolate", preserve_nan=True
    )
    return smoothir


def generate_pixel_identification_from_threshold(
    ir, nx, ny, thresh_cloud, thresh_cold, thresh_core, thresh_warm
):
    """
    Classify pixel cloud types based on thresholds.

    Also create arrays with a flag for each type and fill in cloudid array.
    Cold core = 1
    Cold anvil = 2
    Warm anvil = 3
    Other = 4
    Clear = 5
    Areas do not overlap

    Args:
        ir: np.array
            Array containing IR Tb data.
        nx: int
            Number of pixels in the x direction.
        ny: int
            Number of pixels in the y direction.
        thresh_cloud: float
            Tb threshold to define warm clouds.
        thresh_cold: float
            Tb threshold to define cold anvil clouds.
        thresh_core: float
            Tb threshold to define cold cores.
        thresh_warm: float
            Tb threshold to define warm anvil clouds.

    Returns:
        coldanvil_flag: np.array
            Array containing cold anvil pixel flag.
        core_flag: np.array
            Array containing cold core pixel flag.
        final_cloudid: np.array
            Array containing cloud type pixel flag.
    """
    final_cloudid = np.zeros((ny, nx), dtype=int)
    core_flag = np.zeros((ny, nx), dtype=int)
    # Flag cold core
    core_indices = np.where(ir < thresh_core)
    ncorepix = np.shape(core_indices)[1]
    if ncorepix > 0:
        core_flag[core_indices] = 1
        final_cloudid[core_indices] = 1
    # Flag cold anvil
    coldanvil_flag = np.zeros((ny, nx), dtype=int)
    coldanvil_indices = np.where((ir >= thresh_core) & (ir < thresh_cold))
    ncoldanvilpix = np.shape(coldanvil_indices)[1]
    if ncoldanvilpix > 0:
        coldanvil_flag[coldanvil_indices] = 1
        final_cloudid[coldanvil_indices] = 2
    # Flag warm anvil
    warmanvil_flag = np.zeros((ny, nx), dtype=int)
    warmanvil_indices = np.where((ir >= thresh_cold) & (ir < thresh_warm))
    nwarmanvilpix = np.shape(warmanvil_indices)[1]
    if nwarmanvilpix > 0:
        warmanvil_flag[coldanvil_indices] = 1
        final_cloudid[warmanvil_indices] = 3
    # Flag warm clouds
    othercloud_flag = np.zeros((ny, nx), dtype=int)
    othercloud_indices = np.where((ir >= thresh_warm) & (ir < thresh_cloud))
    nothercloudpix = np.shape(othercloud_indices)[1]
    if nothercloudpix > 0:
        othercloud_flag[othercloud_indices] = 1
        final_cloudid[othercloud_indices] = 4
    # Flag clear area
    clear_flag = np.zeros((ny, nx), dtype=int)
    clear_indices = np.where(ir >= thresh_cloud)
    nclearpix = np.shape(clear_indices)[1]
    if nclearpix > 0:
        clear_flag[clear_indices] = 1
        final_cloudid[clear_indices] = 5
    return coldanvil_flag, core_flag, final_cloudid
