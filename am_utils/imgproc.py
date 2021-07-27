from typing import Union

import numpy as np
from scipy import ndimage
from skimage import io

from .parallel import run_parallel
from .utils import imsave, walk_dir


def convert_to_isotropic(img: np.ndarray, voxel_size: Union[list, np.ndarray, tuple], order: int = 1) -> np.ndarray:
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


def __convert_to_isotropic(item, **kwargs):
    """
    Helper function for parallel running
    """
    fn_in, fn_out = item
    img = io.imread(fn_in)
    imgtype = type(img[0, 0, 0])
    img = convert_to_isotropic(img, **kwargs)
    imsave(fn_out, img.astype(imgtype))


def convert_to_isotropic_batch(input_dir: str, output_dir: str, n_jobs: int = 8, verbose: bool = True, **kwargs):
    """
    Batch convert input images to isotropic voxel size.

    Parameters
    ----------
    input_dir : str
        Input directory to process.
        Can contain subdirectories.
    output_dir : str
        Path to save the output.
    n_jobs : int, optional
        Number of jobs to run in parallel.
        Default: 8.
    verbose : bool, optional
        Display computation progress.
        Default: True.
    kwargs : dict
        Keyword arguments, see below.

    Attributes
    ----------
    voxel_size : list
        List of voxel sizes for the axes of the current image (z, y, x).
    order : int, optional
        Interpolation order.
        Default: 1 (linear interpolation)

    """
    files = walk_dir(input_dir)
    items = [(fn, fn.replace(input_dir, output_dir)) for fn in files]
    run_parallel(process=__convert_to_isotropic,
                 items=items,
                 print_progress=verbose,
                 max_threads=n_jobs,
                 **kwargs)
