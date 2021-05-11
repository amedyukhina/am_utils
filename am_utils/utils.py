# -*- coding: utf-8 -*-
"""
Auxiliary functions for working with files and statistics

:Author:
  `Anna Medyukhina`_
  email: anna.medyukhina@gmail.com

"""
import os
import natsort
import pandas as pd
from skimage import io
import warnings
from tqdm import tqdm


def walk_dir(folder, extensions=None, exclude=None):
    if extensions is None:
        extensions = []
    if exclude in None:
        exclude = []
    files = []
    for fn in os.listdir(folder):
        fn = os.path.join(folder, fn)
        if fn.startswith(".") or fn.split('.')[-1] in exclude:
            continue
        elif fn.split('.')[-1] in extensions or (os.path.isdir(fn) is False and len(extensions) == 0):
            files.append(fn)
        elif os.path.isdir(fn):
            files = files + walk_dir(fn, extensions=extensions)
    files = natsort.natsorted(files)

    return files


def combine_statistics(inputfolder, extensions=['csv'], sep=','):
    """
    Concatenates all statistic tables from csv files located in a given directory.
    The resulting table will be saved into `inputfolder` + ".csv".
    
    Parameters
    ----------
    inputfolder : str
        Name of directory, where the csv files are located that should be concatenated.
               
    Examples
    --------
    >>> combine_statistics("output/statistics/")  
    # all csv files in the folder "output/statistics/" will be concatenated
    # the result will be saved in the file "output/statistics.csv".
    """

    if os.path.exists(inputfolder):
        files = walk_dir(inputfolder, extensions=extensions)
        total_length = len(files)

        array = []
        for fn in tqdm(files):
            data = pd.read_csv(fn, sep=sep)
            array.append(data)
        data = pd.concat(array, ignore_index=True, sort=True)
        data.to_csv(inputfolder[:-1] + '.csv', sep=sep, index=False)


def imsave(outputfile, img):
    """
    Creates the output directory for the image (if not exists) and saves the image with catching warnings.
    
    """
    os.makedirs(os.path.dirname(outputfile), exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        io.imsave(outputfile, img)

