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
