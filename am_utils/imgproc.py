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


def convert_to_maxprojection(img: np.ndarray, axis: int = 0, preprocess: callable = None) -> np.ndarray:
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

    Returns
    -------
    np.ndarray
        Maximum intensity projection of the input image.
    """
    if preprocess is not None:
        img = preprocess(img)

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
