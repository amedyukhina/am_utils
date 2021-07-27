# -*- coding: utf-8 -*-
"""
Auxiliary functions for parallel processing

:Author:
  `Anna Medyukhina`_
  email: anna.medyukhina@gmail.com

"""
import time
from multiprocessing import Process
from typing import Union

import numpy as np
from skimage import io

from .utils import imsave, walk_dir


def __print_progress(procdone, totproc, start):
    """
    Computes and prints out the percentage of the processes that have been completed

    Parameters
    ----------
    procdone : int
        number of processes that have been completed
    totproc : int
        total number of processes
    start : float
        the time when the computation has started   
    """
    donepercent = procdone * 100 / totproc
    elapse = time.time() - start
    tottime = totproc * 1. * elapse / procdone
    left = tottime - elapse
    units = 'sec'
    if left > 60:
        left = left / 60.
        units = 'min'
        if left > 60:
            left = left / 60.
            units = 'hours'

    print('done', procdone, 'of', totproc, '(', donepercent, '% ), approx. time left: ', left, units)


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
    kwargs : dict
        Keyword arguments that are passed to the `process`.

    """
    if process_name is None:
        process_name = process.__name__

    if print_progress:
        print('Run', process_name)

    procs = []

    totproc = len(items)
    procdone = 0
    start = time.time()

    if print_progress:
        print('Started at ', time.ctime())

    for i, cur_item in enumerate(items):

        while len(procs) >= max_threads:
            time.sleep(0.05)
            for p in procs:
                if not p.is_alive():
                    procs.remove(p)
                    procdone += 1
                    if print_progress:
                        __print_progress(procdone, totproc, start)

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
                procdone += 1
                if print_progress:
                    __print_progress(procdone, totproc, start)

    if print_progress:
        print(process_name, 'done')


def __batch_convert_helper(item, function, **kwargs):
    """
    Helper function for parallel running
    """
    fn_in, fn_out = item
    img = io.imread(fn_in)
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
    kwargs : dict
        Keyword arguments that are passed to the `function`

    """
    files = walk_dir(input_dir)
    items = [(fn, fn.replace(input_dir, output_dir)) for fn in files]
    kwargs['function'] = function
    run_parallel(process=__batch_convert_helper,
                 items=items,
                 print_progress=verbose,
                 max_threads=n_jobs,
                 **kwargs)
