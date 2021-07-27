import itertools
import os
import shutil
import unittest

import numpy as np
from ddt import ddt, data
from skimage import io

from ..imgproc import convert_to_isotropic, convert_to_maxprojection, normalize_channels
from ..parallel import batch_convert
from ..utils import imsave, walk_dir


def generate_test_data(inputpath, imgtype):
    for i in range(3):
        img = np.random.randint(0, 100, [10] * 3)
        img = img.astype(imgtype)
        imsave(os.path.join(inputpath, "test/" * i, 'img.tif'), img)


@ddt
class TestIsotropicConversion(unittest.TestCase):
    @data(
        np.ones([10] * 3),
        np.random.randint(0, 100, [10] * 3)
    )
    def test_valid_shape(self, img):
        output = convert_to_isotropic(img, [0.2, 0.1, 0.1])
        self.assertEqual(len(output.shape), len(img.shape))

    @data(
        np.ones([10] * 1),
        np.random.randint(0, 100, [10] * 2),
        np.random.randint(0, 100, [10] * 4)
    )
    def test_invalid_shape(self, img):
        self.assertRaises(ValueError, convert_to_isotropic, img, [0.2, 0.1, 0.1])

    @data(
        np.uint8,
        np.uint16,
        np.float32
    )
    def test_batch(self, imgtype):
        inputpath = 'test_input'
        outputpath = 'test_output'
        generate_test_data(inputpath, imgtype)

        batch_convert(inputpath, outputpath,
                      function=convert_to_isotropic,
                      n_jobs=8, verbose=False,
                      **dict(voxel_size=[5, 1, 1]))
        self.assertEqual(len(walk_dir(inputpath)), len(walk_dir(outputpath)))
        imgout = io.imread(walk_dir(outputpath)[0])
        self.assertEqual(type(imgout[0, 0, 0]), imgtype)
        self.assertEqual(imgout.shape[0], 10 * 5)
        shutil.rmtree(inputpath)
        shutil.rmtree(outputpath)


@ddt
class TestMaxprojConversion(unittest.TestCase):
    @data(
        np.ones([10] * 3),
        np.random.randint(0, 100, [10] * 4)
    )
    def test_shape(self, img):
        output = convert_to_maxprojection(img)
        self.assertEqual(len(output.shape), len(img.shape) - 1)

    @data(
        np.uint8,
        np.uint16,
        np.float32
    )
    def test_batch(self, imgtype):
        inputpath = 'test_input'
        outputpath = 'test_output'
        generate_test_data(inputpath, imgtype)
        batch_convert(inputpath, outputpath,
                      function=convert_to_maxprojection,
                      n_jobs=8, verbose=False)
        self.assertEqual(len(walk_dir(inputpath)), len(walk_dir(outputpath)))
        imgout = io.imread(walk_dir(outputpath)[0])
        self.assertEqual(type(imgout[0, 0]), imgtype)
        self.assertEqual(len(imgout.shape), 2)
        shutil.rmtree(inputpath)
        shutil.rmtree(outputpath)


@ddt
class TestNormalizeChannels(unittest.TestCase):
    @data(
        *itertools.product([
            np.random.randint(0, 100, [10] * 3),
            np.random.randint(0, 100, [10] * 4)],
            [1, 10, 255, 65536])
    )
    def test_maxval(self, case):
        img, maxval = case
        output = normalize_channels(img * 1., maxval=maxval, percentiles=[[1, 99]] * img.shape[-1])
        self.assertEqual(np.max(output), maxval)

    @data(
        np.ones([10] * 3),
        np.zeros([5] * 4)
    )
    def test_maxval(self, img):
        output = normalize_channels(img * 1., maxval=255)
        self.assertEqual(np.max(output), 0)

    @data(
        (np.random.randint(0, 100, [10] * 3), [2, 3, 4]),
        (np.random.randint(0, 100, [10] * 3), [[1, 99]] * 9),
        (np.random.randint(0, 100, [10] * 3), 5),
    )
    def test_value_errors(self, case):
        img, percentiles = case
        self.assertRaises(ValueError, normalize_channels, img, percentiles=percentiles)


if __name__ == '__main__':
    unittest.main()
