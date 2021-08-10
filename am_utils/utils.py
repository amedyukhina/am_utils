# -*- coding: utf-8 -*-
"""
Auxiliary functions for working with files and statistics

:Author:
  `Anna Medyukhina`_
  email: anna.medyukhina@gmail.com

"""
import os
import warnings
from typing import Union

import natsort
import numpy as np
import pandas as pd
from skimage import io
from tqdm import tqdm


def walk_dir(folder: str, extensions: Union[list, tuple, np.array] = None,
             exclude: Union[list, tuple, np.array, None] = None) -> list:
    """
    Return the list of full file paths for a given folder and give file extensions.

    Parameters
    ----------
    folder : str
        Input directory to list
    extensions : list, optional
        List of extensions to include.
        If provided, only files with these extensions will be listed.
        If None, all files will be listed.
        Default: None
    exclude : list, optional
        List of extensions to exclude.
        If None, all files, or files with extensions specified by `extensions`, will be listed.
        Default: None

    Returns
    -------
    list :
        List of files with full paths

    """
    if extensions is None:
        extensions = []
    if exclude is None:
        exclude = []
    files = []
    for fn in os.listdir(folder):
        if fn.startswith(".") or fn.split('.')[-1] in exclude:
            continue
        fn = os.path.join(folder, fn)
        if fn.split('.')[-1] in extensions or (os.path.isdir(fn) is False and len(extensions) == 0):
            files.append(fn)
        elif os.path.isdir(fn):
            files = files + walk_dir(fn, extensions=extensions, exclude=exclude)
    files = natsort.natsorted(files)

    return files


def combine_statistics(folder: str, output_name: str = None,
                       extensions: Union[list, tuple, np.array, None] = None, sep: str = ','):
    """
        Concatenates all statistic tables from the csv files located in the provided directory.

    Parameters
    ----------
    folder : str
        Name of directory, where the csv files are located that should be concatenated.
    output_name : str
        File name to save the combined statistics.
        If None, the output name will be `folder` + ".csv".
        Default: None
    extensions : list
        List of extensions to include.
        If None, "csv" and "txt" will be included.
        Default: None
    sep : str, optional
        Field separator provided to pandas
        Default: ','


    """

    if os.path.exists(folder):
        if extensions is None:
            extensions = ['csv', 'txt']
        files = walk_dir(folder, extensions=extensions)
        if output_name is None:
            output_name = folder
            if output_name.endswith('/'):
                output_name = output_name[:-1]
            output_name = output_name + '.csv'

        array = []
        for fn in tqdm(files):
            data = pd.read_csv(fn, sep=sep)
            array.append(data)
        data = pd.concat(array, ignore_index=True, sort=True)
        data.to_csv(output_name, sep=sep, index=False)


def imsave(outputfile: str, img: np.ndarray):
    """

    Creates the output directory for the image (if not exists) and saves the image with warnings catching.

    Parameters
    ----------
    outputfile : str
        Output file name
    img : np.ndarray
        Image to save

    """
    os.makedirs(os.path.dirname(outputfile), exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        io.imsave(outputfile, img)
