# Image-Analysis Functions Reference

A deduplicated list of every third-party image-analysis function used anywhere in PyFLEXTRKR's feature-identification code. For call-site detail (which config branch triggers each call, exact line numbers, purpose in context), see [image_analysis_usage.md](image_analysis_usage.md).

| Library | Function | Documentation |
|---|---|---|
| `astropy.convolution` | `astropy.convolution.Box2DKernel` | [astropy.convolution.Box2DKernel](https://docs.astropy.org/en/stable/api/astropy.convolution.Box2DKernel.html) |
| `astropy.convolution` | `astropy.convolution.convolve` | [astropy.convolution.convolve](https://docs.astropy.org/en/stable/api/astropy.convolution.convolve.html) |
| `scipy.ndimage` | `scipy.ndimage.binary_dilation` | [scipy.ndimage.binary_dilation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.binary_dilation.html) |
| `scipy.ndimage` | `scipy.ndimage.convolve` | [scipy.ndimage.convolve](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.convolve.html) |
| `scipy.ndimage` | `scipy.ndimage.distance_transform_edt` | [scipy.ndimage.distance_transform_edt](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.distance_transform_edt.html) |
| `scipy.ndimage` | `scipy.ndimage.generate_binary_structure` | [scipy.ndimage.generate_binary_structure](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.generate_binary_structure.html) |
| `scipy.ndimage` | `scipy.ndimage.grey_dilation` | [scipy.ndimage.grey_dilation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.grey_dilation.html) |
| `scipy.ndimage` | `scipy.ndimage.label` | [scipy.ndimage.label](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.label.html) |
| `scipy.ndimage` | `scipy.ndimage.median_filter` | [scipy.ndimage.median_filter](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.median_filter.html) |
| `scipy.ndimage` | `scipy.ndimage.uniform_filter` | [scipy.ndimage.uniform_filter](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.uniform_filter.html) |
| `scipy.signal` | `scipy.signal.convolve` | [scipy.signal.convolve](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve.html) |
| `scipy.signal` | `scipy.signal.fftconvolve` | [scipy.signal.fftconvolve](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.fftconvolve.html) |
| `scipy.signal` | `scipy.signal.medfilt2d` | [scipy.signal.medfilt2d](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.medfilt2d.html) |
| `skimage.feature` | `skimage.feature.peak_local_max` | [skimage.feature.peak_local_max](https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.peak_local_max) |
| `skimage.segmentation` | `skimage.segmentation.watershed` | [skimage.segmentation.watershed](https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.watershed) |
