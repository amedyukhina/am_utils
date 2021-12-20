# -*- coding: utf-8 -*-
"""
Auxiliary functions for image processing

:Author:
  `Anna Medyukhina`_
  email: anna.medyukhina@gmail.com

"""
from typing import Union

import numpy as np
from scipy import ndimage


def convert_to_isotropic(img: np.ndarray, voxel_size: Union[list, np.array, tuple], order: int = 1) -> np.ndarray:
    """
    Convert input 3D image to isotropic voxel size.

    Parameters
    ----------
    img : np.ndarray
        Input image to convert.
        Must be a 3D array.
    voxel_size : list
        List of voxel sizes for the axes of the current image (z, y, x).
    order : int, optional
        Interpolation order.
        Default: 1 (linear interpolation)
    Returns
    -------
    np.ndarray
        Isotropized image

    """
    voxel_size = np.array(voxel_size)
    if len(img.shape) == 3:
        return ndimage.interpolation.zoom(img * 1., zoom=voxel_size / voxel_size[-1], order=order)
    else:
        raise ValueError(rf'The number of image dimensions must be 3, {len(img.shape)} provided')


def convert_to_maxprojection(img: np.ndarray, axis: int = 0, preprocess: callable = None,
                             **preprocess_kwargs) -> np.ndarray:
    """
    Convert an image to its maximum intensity projection along given axis.

    Parameters
    ----------
    img : np.ndarray
        Input image.
    axis: int
        Axis of the maximum projection.
        Default: 0
    preprocess : callable, optional
        Function/transform to apply before generating maximum projection.
        Default: None.
    preprocess_kwargs : dict
        Keyword arguments that are passed to the `preprocess`.

    Returns
    -------
    np.ndarray
        Maximum intensity projection of the input image.
    """
    if preprocess is not None:
        img = preprocess(img, **preprocess_kwargs)

    return np.max(img, axis=axis)


def normalize_channels(img: np.ndarray, maxval: float = 255, percentiles: list = None) -> np.ndarray:
    """
    Normalize individual image channels.

    Parameters
    ----------
    img : np.ndarray
        Input image to normalize.
        The last axis must be channel.
    maxval : float
        Maximum value of the output image
    percentiles : nested list, optional
        Percentile list for each channel of the form [[low_ch1, high_ch1], [low_ch2, high_ch2], [low_ch3, high_ch3]].

    Returns
    -------
    np.ndimage
        Normalized image

    """
    img = img.astype(np.float)
    for ch in range(img.shape[-1]):
        sl = tuple([slice(0, None)] * (len(img.shape) - 1) + [ch])
        if percentiles is not None:
            if not (type(percentiles) in [list, np.array] and type(percentiles[0]) in [list, np.array]):
                raise ValueError('Percentiles must be provided as a nested list of form '
                                 '[[low_ch1, high_ch1], [low_ch2, high_ch2], [low_ch3, high_ch3]]')
            if not len(percentiles[0]) == 2:
                raise ValueError(rf"Two percentile values must be provided for each channel, "
                                 rf"{len(percentiles[0])} was provided")
            if len(percentiles) != img.shape[-1]:
                raise ValueError(rf"The number of provided percentiles must be equal to the number of "
                                 rf"image channels ({img.shape[-1]}), {len(percentiles)} were provided")
            mn, mx = [np.percentile(img[sl], p) for p in percentiles[ch]]
        else:
            mn, mx = np.min(img[sl]), np.max(img[sl])
        img[sl] = img[sl] - mn
        if mx > 0:
            img[sl] = img[sl] * maxval / mx
        img[sl] = np.clip(img[sl], 0, maxval)
    return img


def rescale_intensity_percentiles(img: np.ndarray,
                                  percentiles: Union[list, tuple] = (0.25, 99.75),
                                  minrange: float = 0) -> np.ndarray:
    """
    Rescale image intensity between 0 and 1 with provided percentiles.

    Parameters
    ----------
    img : np.ndarray
        Input image to rescale.
    percentiles : list or tuple
        Percentile of the form [low, high].
        Default: (0.25, 99.75)
    minrange : float
        Minimum difference between the low and high percentiles.
        If the difference is lower than this parameter, an array of zeros will be returned.
        Default: 0

    Returns
    -------
    np.ndimage
        Rescaled image

    """
    img = img.astype(np.float)
    mn, mx = [np.percentile(img, p) for p in percentiles]
    if mx > mn + minrange:
        return np.clip((img.astype(np.float32) - mn) / (mx - mn), 0, 1)
    else:
        return np.zeros(img.shape, dtype=np.float32)


def plot_projections(img: np.ndarray, spacing: int = 5, zoom: float = None) -> np.ndarray:
    """
    Plot three maximum intensity projections of the given image side-by-side.

    Parameters
    ----------
    img : np.ndarray
        Input 3D image.
    spacing : int
        Number of pixels to separate the projections in the output image.
        Default: 5
    zoom : float, optional
        Scale factor to interpolate the image along z axis.
        Default: None (no interpolation).

    Returns
    -------
    np.ndarray
        Output image with the three maximum projections
    """
    m0 = np.max(img, axis=0)
    m1 = np.max(img, axis=1)
    m2 = np.max(img, axis=2)

    if zoom is not None:
        zoom_arr = [1.] * len(m1.shape)
        zoom_arr[0] = zoom
        m1 = ndimage.interpolation.zoom(m1 * 1., zoom_arr, order=1)
        m2 = ndimage.interpolation.zoom(m2 * 1., zoom_arr, order=1)

    maxproj = np.zeros((m0.shape[0] + m1.shape[0] + spacing,
                        m0.shape[1] + m2.shape[0] + spacing) +
                       img.shape[3:])
    maxproj[:m0.shape[0], :m0.shape[1]] = m0
    maxproj[m0.shape[0] + spacing:, :m0.shape[1]] = m1
    maxproj[:m0.shape[0], m0.shape[1] + spacing:] = np.swapaxes(m2, 0, 1)
    return maxproj


def stack_to_mosaic(img, ncols=5):
    """
    Convert a 3D stack to mosaic.

    Parameters
    ----------
    img : np.ndarray
        3D image
    ncols : int, optional
        Number of columns in the mosaic.
        Default: 5

    Returns
    -------
    img : np.ndarray
        2D mosaic view of the input image.
    """
    img = np.array(img)
    nrows = int(np.ceil(img.shape[0] / ncols))  # calculate the number of rows
    shape = img.shape[1:]  # shape of each 3D slice
    pad = nrows * ncols - img.shape[0]
    img = np.pad(img,
                 [[0, int(pad)]] + [[0, 0]] * len(shape))  # pad the image stack to fill all columns of the last row
    img = img.reshape((nrows, ncols,) + img[0].shape)  # reshape the layers into rows and columns
    img = np.moveaxis(img, 2, 1)  # swap the columns and y axis
    img = img.reshape((nrows * shape[0], ncols * shape[1]))  # combine rows with y, and columns with x
    return img
