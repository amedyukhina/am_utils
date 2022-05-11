# -*- coding: utf-8 -*-
"""
Auxiliary functions for parallel processing

:Author:
  `Anna Medyukhina`_
  email: anna.medyukhina@gmail.com

"""
import os
import time
from multiprocessing import Process
from typing import Union

import numpy as np
from skimage import io
from tqdm import tqdm

from .utils import imsave, walk_dir


def run_parallel(process: callable,
                 items: Union[list, np.array],
                 max_threads: int = 8,
                 process_name: str = None,
                 print_progress: bool = True,
                 **kwargs):
    """
    Apply a given function in parallel to each item from a given list.

    Parameters
    ----------
    process : callable
        The function that will be applied to each item of `items`.
        The function should accept the argument `item`, which corresponds to one item from `items`.
        An `item` is usually a name of the file that has to be processed or 
            a list of files that have to be combined / convolved /analyzed together.
        The function should not return any output, but the output should be saved in a specified directory.
    items : list
        List of items. For each item, the `process` will be called.
    max_threads : int, optional
        The maximal number of processes to run in parallel
        Default is 8
    process_name : str, optional
        Name of the process, will be printed if `print_progress` is set to True.
        If None, the name of the function given in `process` will be printed.
        Default is None.
    print_progress : bool, optional
        If True, the progress of the computation will be printed.
        Default is True.
    kwargs : key value
        Keyword arguments that are passed to the `process`.

    """
    if process_name is None:
        process_name = process.__name__

    if print_progress:
        print('Run', process_name)

    procs = []

    for i, cur_item in enumerate(tqdm(items)):

        while len(procs) >= max_threads:
            time.sleep(0.05)
            for p in procs:
                if not p.is_alive():
                    procs.remove(p)

        cur_args = kwargs.copy()
        cur_args['item'] = cur_item
        p = Process(target=process, kwargs=cur_args)
        p.start()
        procs.append(p)

    while len(procs) > 0:
        time.sleep(0.05)
        for p in procs:
            if not p.is_alive():
                procs.remove(p)

    if print_progress:
        print(process_name, 'done')


def __batch_convert_helper(item, function, imgtype=None, overwrite=True, **kwargs):
    """
    Helper function for parallel running
    """
    fn_in, fn_out = item
    if (not os.path.exists(fn_out)) or overwrite is True:
        img = io.imread(fn_in)
        if imgtype is None:
            imgtype = type(img[0, 0, 0])
        img = function(img, **kwargs)
        imsave(fn_out, img.astype(imgtype))


def batch_convert(input_dir: str, output_dir: str,
                  function: callable,
                  n_jobs: int = 8, verbose: bool = True, **kwargs):
    """
    Batch convert input images in the input directory with a given transform.

    Parameters
    ----------
    input_dir : str
        Input directory to process.
        Can contain subdirectories.
    output_dir : str
        Path to save the output.
    function : callable
        Image transform function to run in batch
    n_jobs : int, optional
        Number of jobs to run in parallel.
        Default: 8.
    verbose : bool, optional
        Display computation progress.
        Default: True.
    kwargs : key value
        Keyword arguments that are passed to the `function`

    """
    files = walk_dir(input_dir)
    items = [(fn, fn.replace(input_dir, output_dir)) for fn in files]
    kwargs['function'] = function
    run_parallel(process=__batch_convert_helper,
                 items=items,
                 process_name=function.__name__,
                 print_progress=verbose,
                 max_threads=n_jobs,
                 **kwargs)
